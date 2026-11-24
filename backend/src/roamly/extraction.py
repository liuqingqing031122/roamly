"""Schemas for claim extraction.

`ExtractedClaim` is untrusted LLM output. Only extraction post-processing turns it
into a domain `Claim`; these types alone do not guarantee that an excerpt exists in
the source or that a claim is semantically correct.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from roamly.domain import Claim, ClaimType

ExtractedClaimType = Literal[ClaimType, "unsupported"]

AssertionKind = Literal["policy", "observation"]

SkipReason = Literal[
    "unsupported_type", "observation", "excerpt_not_found", "duplicate", "missing_place"
]


class ExtractedClaim(BaseModel):
    """A candidate claim as returned by the LLM, before validation."""

    # None when the place cannot be determined from the source; never invented.
    place_name: str | None = None
    claim_type: ExtractedClaimType
    assertion_kind: AssertionKind
    original_excerpt: str
    normalized_statement: str
    scope: list[str] = Field(default_factory=list)

    @field_validator("scope", mode="before")
    @classmethod
    def coerce_scope_to_list(cls, value: Any) -> Any:
        """Accept null or a single string from the LLM without changing their meaning."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value


class ExtractedClaims(BaseModel):
    """Top-level LLM response object for claim extraction."""

    claims: list[ExtractedClaim]


class SkippedClaim(BaseModel):
    """An extracted claim that was not accepted, with the reason."""

    claim: ExtractedClaim
    reason: SkipReason


class ExtractionResult(BaseModel):
    """Accepted claims and skipped candidates from one extraction run."""

    claims: list[Claim]
    skipped: list[SkippedClaim]


type DuplicateKey = tuple[str, str, str]


def process_extracted_claims(source_text: str, extracted: ExtractedClaims) -> ExtractionResult:
    """Validate LLM candidates against the source text and convert them into claims.

    Candidates are processed in their original order. Accepted claims receive
    sequential IDs ("claim-1", "claim-2", ...), which are unique within one result.
    An accepted claim's `original_excerpt` is the matching span of the source text, so
    it differs from the LLM's excerpt only where whitespace differs (see `find_excerpt`).
    """
    claims: list[Claim] = []
    skipped: list[SkippedClaim] = []
    accepted_keys: set[DuplicateKey] = set()

    for candidate in extracted.claims:
        reason = _skip_reason(candidate, source_text, accepted_keys)
        if reason is not None:
            skipped.append(SkippedClaim(claim=candidate, reason=reason))
            continue

        # Guaranteed by _skip_reason; also narrows the type to the domain ClaimType.
        assert candidate.claim_type != "unsupported"
        excerpt = find_excerpt(source_text, candidate.original_excerpt)
        assert excerpt is not None
        accepted_keys.add(_duplicate_key(candidate))
        claims.append(
            Claim(
                id=f"claim-{len(claims) + 1}",
                place_name=(candidate.place_name or "").strip(),
                claim_type=candidate.claim_type,
                original_excerpt=excerpt,
                normalized_statement=candidate.normalized_statement,
                scope=[item.strip() for item in candidate.scope if item.strip()],
            )
        )

    return ExtractionResult(claims=claims, skipped=skipped)


def _skip_reason(
    candidate: ExtractedClaim, source_text: str, accepted_keys: set[DuplicateKey]
) -> SkipReason | None:
    """Return the first failed check for a candidate, or None if it is accepted."""
    if candidate.claim_type == "unsupported":
        return "unsupported_type"
    if candidate.assertion_kind == "observation":
        return "observation"
    if find_excerpt(source_text, candidate.original_excerpt) is None:
        return "excerpt_not_found"
    if candidate.place_name is None or not candidate.place_name.strip():
        return "missing_place"
    if _duplicate_key(candidate) in accepted_keys:
        return "duplicate"
    return None


def find_excerpt(source_text: str, excerpt: str) -> str | None:
    """Return the first span of `source_text` that equals `excerpt`, or None.

    Every run of whitespace (spaces, line breaks, tabs) matches any other run of
    whitespace, because the LLM often joins lines that the source wraps. Everything
    else must match exactly: case, punctuation, and words, with no fuzzy matching.
    Whitespace at the edges still has to be present in the source.
    """
    # An empty excerpt is found in any text, so it cannot link to the source.
    if not excerpt.strip():
        return None
    pattern = "".join(
        r"\s+" if part.isspace() else re.escape(part)
        for part in re.split(r"(\s+)", excerpt)
        if part
    )
    match = re.search(pattern, source_text)
    return match.group() if match else None


def _duplicate_key(candidate: ExtractedClaim) -> DuplicateKey:
    """Identify a claim by place, type, and statement, ignoring case and outer whitespace."""
    place_name = candidate.place_name or ""
    return (
        place_name.strip().casefold(),
        candidate.claim_type,
        candidate.normalized_statement.strip().casefold(),
    )
