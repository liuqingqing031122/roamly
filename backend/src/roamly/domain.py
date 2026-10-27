"""Core domain models for the Travel Reality Checker."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, HttpUrl, model_validator

ClaimType = Literal["opening_hours", "reservation", "closure"]

Verdict = Literal["Supported", "Contradicted", "Potentially Outdated", "Unable to Verify"]


class Claim(BaseModel):
    """A checkable statement about a venue, extracted from a travel guide."""

    id: str
    place_name: str
    claim_type: ClaimType
    original_excerpt: str
    normalized_statement: str


class Evidence(BaseModel):
    """A retrieved source excerpt that may support or contradict a claim."""

    id: str
    claim_id: str
    source_url: HttpUrl
    source_title: str
    content_excerpt: str
    retrieved_at: datetime


class VerificationResult(BaseModel):
    """The verdict for a single claim and the evidence it is based on."""

    claim_id: str
    verdict: Verdict
    explanation: str
    evidence_ids: list[str]

    @model_validator(mode="after")
    def require_evidence_for_definitive_verdicts(self) -> Self:
        """Supported and Contradicted verdicts must cite at least one evidence item."""
        if self.verdict in ("Supported", "Contradicted") and not self.evidence_ids:
            raise ValueError(f"A {self.verdict} verdict requires at least one evidence ID")
        return self
