"""Claim-level evidence extraction from fetched official-website text (M3.2).

For every fetched page, the LLM selects excerpts relevant to each of the page's
claims; deterministic code then keeps only excerpts that occur in the page text
(whitespace-equivalent, as in claim extraction), belong to a claim of that page, and
are not duplicates. Everything rejected is recorded as an `EvidenceIssue`.

This step finds candidate evidence only. `found` means relevant passages exist, not
that the claim is true; verification is M4.

Long pages are split into overlapping chunks on line boundaries, one LLM call each,
so no part of the text is silently dropped.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ValidationError, model_validator

from roamly.domain import Claim, Evidence
from roamly.evidence_prompt import (
    EVIDENCE_PROMPT_VERSION,
    EVIDENCE_SYSTEM_PROMPT,
    build_evidence_user_prompt,
)
from roamly.extraction import find_excerpt
from roamly.llm import LLMClient, LLMError
from roamly.web_fetch import FetchReason, SourcePage, SourceType

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".roamly_cache/evidence")
# Same bound per LLM call as claim extraction.
MAX_CHUNK_CHARS = 30_000
# Text repeated at the start of the next chunk, so passages at a boundary are seen whole.
CHUNK_OVERLAP_CHARS = 1_000
MAX_EXCERPTS_PER_CLAIM = 5
CACHE_FORMAT_VERSION = 1


# --- LLM output (untrusted) -----------------------------------------------------


class ExtractedEvidence(BaseModel):
    """Excerpts the LLM selected for one claim, before validation."""

    claim_id: str
    excerpts: list[str]


class ExtractedEvidenceList(BaseModel):
    """Top-level LLM response object for evidence extraction."""

    results: list[ExtractedEvidence]


# --- Validated result -----------------------------------------------------------

EvidenceStatus = Literal["found", "none_found", "page_unavailable", "failed"]
EvidenceReason = Literal[
    "no_page",  # page_unavailable: no page lists this claim
    "llm_error",  # failed: an LLM call for the page failed
    "missing_from_response",  # failed: the LLM did not return a result for the claim
]
IssueReason = Literal[
    "unknown_claim_id",  # the claim id is not one of this page's claims
    "excerpt_not_found",  # the excerpt does not occur in the page text
    "duplicate",  # the same passage was already accepted for the claim
    "too_many_excerpts",  # more than MAX_EXCERPTS_PER_CLAIM valid excerpts
]


class ClaimEvidence(BaseModel):
    """The evidence extraction outcome for one claim.

    - found: at least one validated excerpt.
    - none_found: the page was checked and has no valid excerpt for the claim.
    - page_unavailable: there is no fetched page (reason: `no_page` or the page's own
      skip/failure reason, e.g. `robots_disallowed`).
    - failed: extraction did not complete (`llm_error`, `missing_from_response`).
    """

    claim_id: str
    place_name: str
    status: EvidenceStatus
    reason: EvidenceReason | FetchReason | None = None
    error: str | None = None
    source_type: SourceType | None = None
    source_url: str | None = None  # the page's final URL
    fetched_at: AwareDatetime | None = None
    evidence: list[Evidence] = []

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if (self.status == "found") != bool(self.evidence):
            raise ValueError("Evidence is present if and only if the status is 'found'")
        if (self.reason is None) != (self.status in ("found", "none_found")):
            raise ValueError(f"Status {self.status!r} does not fit reason {self.reason!r}")
        if self.status == "failed" and not self.error:
            raise ValueError("A failed extraction needs an error message")
        return self


class EvidenceIssue(BaseModel):
    """LLM output that was rejected by validation."""

    place_name: str
    claim_id: str
    excerpt: str | None = None
    reason: IssueReason


# --- Chunking -------------------------------------------------------------------


def split_text(
    text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Split on line boundaries into chunks of at most `max_chars` characters.

    Consecutive chunks share up to `overlap` characters of whole lines. A single line
    longer than `max_chars` is split by characters.
    """
    if len(text) <= max_chars:
        return [text]
    lines: list[str] = []
    for line in text.split("\n"):
        lines += [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]

    chunks: list[str] = []
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and size + len(lines[end]) + (end > start) <= max_chars:
            size += len(lines[end]) + (end > start)
            end += 1
        chunks.append("\n".join(lines[start:end]))
        if end >= len(lines):
            break
        # Step back over whole lines for the overlap, but always make progress.
        next_start, kept = end, 0
        while next_start - 1 > start and kept + len(lines[next_start - 1]) + 1 <= overlap:
            next_start -= 1
            kept += len(lines[next_start]) + 1
        start = next_start
    return chunks


# --- Cache ----------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceCache:
    """Stores validated LLM output (`ExtractedEvidenceList`) per prompt, one file each.

    The key covers the prompt version and text, the response schema, the model
    settings, and the full user prompt (claims and page text), so a change to any of
    them is a miss. Validation runs again on every hit.
    """

    def __init__(
        self,
        directory: Path,
        model_settings: Mapping[str, Any],
        *,
        prompt_version: str = EVIDENCE_PROMPT_VERSION,
        system_prompt: str = EVIDENCE_SYSTEM_PROMPT,
    ) -> None:
        self.directory = directory
        self.model_settings = dict(model_settings)
        self.prompt_version = prompt_version
        self.system_prompt = system_prompt

    def key_fields(self, user_prompt: str) -> dict[str, Any]:
        schema = json.dumps(ExtractedEvidenceList.model_json_schema(), sort_keys=True)
        return {
            "cache_format": CACHE_FORMAT_VERSION,
            "prompt_version": self.prompt_version,
            "system_prompt_sha256": _sha256(self.system_prompt),
            "response_schema_sha256": _sha256(schema),
            "model_settings": self.model_settings,
            "user_prompt_sha256": _sha256(user_prompt),
        }

    def path(self, user_prompt: str) -> Path:
        key = _sha256(json.dumps(self.key_fields(user_prompt), sort_keys=True))
        return self.directory / f"{key}.json"

    def get(self, user_prompt: str) -> ExtractedEvidenceList | None:
        path = self.path(user_prompt)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return ExtractedEvidenceList.model_validate(entry["output"])
        except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
            logger.warning("Ignoring unreadable evidence cache entry %s: %s", path.name, exc)
            return None

    def put(self, user_prompt: str, output: ExtractedEvidenceList) -> None:
        entry = {
            "key_fields": self.key_fields(user_prompt),
            "created_at": datetime.now(UTC).isoformat(),
            "output": output.model_dump(mode="json"),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(entry, file, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.path(user_prompt))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


# --- Extraction -----------------------------------------------------------------


def extract_evidence(
    claims: list[Claim],
    pages: list[SourcePage],
    client: LLMClient,
    cache: EvidenceCache | None = None,
) -> tuple[list[ClaimEvidence], list[EvidenceIssue]]:
    """One `ClaimEvidence` per claim (in claim order) and the rejected LLM output.

    An LLM failure affects only the claims of that page; other exceptions propagate.
    """
    page_by_claim: dict[str, SourcePage] = {}
    for page in pages:
        for claim_id in page.claim_ids:
            page_by_claim.setdefault(claim_id, page)

    results: dict[str, ClaimEvidence] = {}
    issues: list[EvidenceIssue] = []
    for page in pages:
        page_claims = [c for c in claims if page_by_claim.get(c.id) is page]
        if not page_claims:
            continue
        if page.status != "fetched" or not page.text:
            for claim in page_claims:
                results[claim.id] = _unavailable(claim, page.reason or "no_page")
            continue
        page_results, page_issues = _extract_page(page, page_claims, client, cache)
        results.update(page_results)
        issues += page_issues

    return [results.get(claim.id) or _unavailable(claim, "no_page") for claim in claims], issues


def _unavailable(claim: Claim, reason: EvidenceReason | FetchReason) -> ClaimEvidence:
    return ClaimEvidence(
        claim_id=claim.id, place_name=claim.place_name, status="page_unavailable", reason=reason
    )


def _extract_page(
    page: SourcePage, claims: list[Claim], client: LLMClient, cache: EvidenceCache | None
) -> tuple[dict[str, ClaimEvidence], list[EvidenceIssue]]:
    base = {
        "source_type": page.source_type,
        "source_url": page.final_url,
        "fetched_at": page.fetched_at,
    }
    claim_ids = {claim.id for claim in claims}
    accepted: dict[str, list[str]] = {claim.id: [] for claim in claims}
    answered: set[str] = set()
    issues: list[EvidenceIssue] = []

    def issue(claim_id: str, reason: IssueReason, excerpt: str | None = None) -> None:
        issues.append(
            EvidenceIssue(
                place_name=page.place_name, claim_id=claim_id, excerpt=excerpt, reason=reason
            )
        )

    chunks = split_text(page.text)
    for index, chunk in enumerate(chunks, start=1):
        try:
            output = _call(claims, chunk, client, cache)
        except LLMError as exc:
            error = f"LLM call failed for chunk {index} of {len(chunks)}: {exc}"
            failed = {
                claim.id: ClaimEvidence(
                    claim_id=claim.id,
                    place_name=claim.place_name,
                    status="failed",
                    reason="llm_error",
                    error=error,
                    **base,
                )
                for claim in claims
            }
            return failed, issues

        for result in output.results:
            if result.claim_id not in claim_ids:
                issue(result.claim_id, "unknown_claim_id")
                continue
            answered.add(result.claim_id)
            for excerpt in result.excerpts:
                span = find_excerpt(chunk, excerpt)
                if span is None:
                    issue(result.claim_id, "excerpt_not_found", excerpt)
                elif span in accepted[result.claim_id]:
                    issue(result.claim_id, "duplicate", excerpt)
                elif len(accepted[result.claim_id]) >= MAX_EXCERPTS_PER_CLAIM:
                    issue(result.claim_id, "too_many_excerpts", excerpt)
                else:
                    accepted[result.claim_id].append(span)

    results = {}
    for claim in claims:
        if claim.id not in answered:
            results[claim.id] = ClaimEvidence(
                claim_id=claim.id,
                place_name=claim.place_name,
                status="failed",
                reason="missing_from_response",
                error="The LLM returned no result for this claim",
                **base,
            )
            continue
        evidence = [
            _to_evidence(claim.id, n, span, page)
            for n, span in enumerate(accepted[claim.id], start=1)
        ]
        results[claim.id] = ClaimEvidence(
            claim_id=claim.id,
            place_name=claim.place_name,
            status="found" if evidence else "none_found",
            evidence=evidence,
            **base,
        )
    return results, issues


def _call(
    claims: list[Claim], chunk: str, client: LLMClient, cache: EvidenceCache | None
) -> ExtractedEvidenceList:
    user_prompt = build_evidence_user_prompt(claims, chunk)
    if cache is not None and (cached := cache.get(user_prompt)) is not None:
        logger.info("Evidence cache hit")
        return cached
    response = client.complete_structured(
        system_prompt=EVIDENCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=ExtractedEvidenceList,
    )
    logger.info(
        "LLM call used %d input and %d output tokens",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )
    if cache is not None:
        try:
            cache.put(user_prompt, response.output)
        except OSError as exc:
            logger.warning("Could not write evidence cache: %s", exc)
    return response.output


def _to_evidence(claim_id: str, number: int, excerpt: str, page: SourcePage) -> Evidence:
    assert page.final_url is not None and page.fetched_at is not None  # fetched page
    return Evidence(
        id=f"{claim_id}-ev-{number}",
        claim_id=claim_id,
        source_url=page.final_url,  # validated as an HttpUrl by Evidence
        source_title=page.title or page.place_name,
        content_excerpt=excerpt,
        retrieved_at=page.fetched_at,
    )
