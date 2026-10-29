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
