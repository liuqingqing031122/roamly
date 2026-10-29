"""Schemas for claim extraction.

`ExtractedClaim` is untrusted LLM output. Only extraction post-processing turns it
into a domain `Claim`; these types alone do not guarantee that an excerpt exists in
the source or that a claim is semantically correct.
"""

from typing import Literal

from pydantic import BaseModel

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
    scope: str | None = None


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
        accepted_keys.add(_duplicate_key(candidate))
        claims.append(
            Claim(
                id=f"claim-{len(claims) + 1}",
                place_name=(candidate.place_name or "").strip(),
                claim_type=candidate.claim_type,
                original_excerpt=candidate.original_excerpt,
                normalized_statement=candidate.normalized_statement,
                scope=candidate.scope,
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
    # An empty excerpt is a substring of any text, so it cannot link to the source.
    if not candidate.original_excerpt.strip() or candidate.original_excerpt not in source_text:
        return "excerpt_not_found"
    if candidate.place_name is None or not candidate.place_name.strip():
        return "missing_place"
    if _duplicate_key(candidate) in accepted_keys:
        return "duplicate"
    return None


def _duplicate_key(candidate: ExtractedClaim) -> DuplicateKey:
    """Identify a claim by place, type, and statement, ignoring case and outer whitespace."""
    place_name = candidate.place_name or ""
    return (
        place_name.strip().casefold(),
        candidate.claim_type,
        candidate.normalized_statement.strip().casefold(),
    )
