from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from roamly.domain import Claim, Evidence, VerificationResult


def make_claim(**overrides: object) -> Claim:
    data = {
        "id": "claim-1",
        "place_name": "City Museum",
        "claim_type": "opening_hours",
        "original_excerpt": "The museum is open every Monday.",
        "normalized_statement": "City Museum is open on Mondays.",
    }
    data.update(overrides)
    return Claim(**data)


def make_evidence(**overrides: object) -> Evidence:
    data = {
        "id": "evidence-1",
        "claim_id": "claim-1",
        "source_url": "https://citymuseum.example/visit",
        "source_title": "Plan your visit",
        "content_excerpt": "Closed Mondays except selected holidays.",
        "retrieved_at": datetime(2026, 10, 27, 9, 30, tzinfo=UTC),
    }
    data.update(overrides)
    return Evidence(**data)


def make_result(**overrides: object) -> VerificationResult:
    data = {
        "claim_id": "claim-1",
        "verdict": "Contradicted",
        "explanation": "The official site states the museum is closed on Mondays.",
        "evidence_ids": ["evidence-1"],
    }
    data.update(overrides)
    return VerificationResult(**data)


# Valid creation


@pytest.mark.parametrize("claim_type", ["opening_hours", "reservation", "closure"])
def test_claim_accepts_supported_types(claim_type: str) -> None:
    claim = make_claim(claim_type=claim_type)

    assert claim.claim_type == claim_type


def test_claim_scope_defaults_to_none() -> None:
    claim = make_claim()

    assert claim.scope is None


def test_claim_accepts_scope() -> None:
    claim = make_claim(scope="except public holidays")

    assert claim.scope == "except public holidays"


def test_evidence_parses_url() -> None:
    evidence = make_evidence()

    assert evidence.source_url.host == "citymuseum.example"
    assert evidence.retrieved_at.tzinfo is UTC


@pytest.mark.parametrize(
    "verdict", ["Supported", "Contradicted", "Potentially Outdated", "Unable to Verify"]
)
def test_verification_result_accepts_all_verdicts(verdict: str) -> None:
    result = make_result(verdict=verdict)

    assert result.verdict == verdict


# Verdict and evidence consistency


@pytest.mark.parametrize("verdict", ["Supported", "Contradicted"])
def test_definitive_verdict_accepts_evidence(verdict: str) -> None:
    result = make_result(verdict=verdict, evidence_ids=["evidence-1"])

    assert result.evidence_ids == ["evidence-1"]


@pytest.mark.parametrize("verdict", ["Supported", "Contradicted"])
def test_definitive_verdict_requires_evidence(verdict: str) -> None:
    with pytest.raises(ValidationError, match="requires at least one evidence ID"):
        make_result(verdict=verdict, evidence_ids=[])


@pytest.mark.parametrize("verdict", ["Potentially Outdated", "Unable to Verify"])
def test_non_definitive_verdict_allows_no_evidence(verdict: str) -> None:
    result = make_result(verdict=verdict, evidence_ids=[])

    assert result.evidence_ids == []


# Invalid values


@pytest.mark.parametrize("claim_type", ["ticket_price", "Opening_Hours", ""])
def test_claim_rejects_unsupported_type(claim_type: str) -> None:
    with pytest.raises(ValidationError):
        make_claim(claim_type=claim_type)


@pytest.mark.parametrize("verdict", ["supported", "True", "Partially Supported", ""])
def test_verification_result_rejects_unknown_verdict(verdict: str) -> None:
    with pytest.raises(ValidationError):
        make_result(verdict=verdict)


@pytest.mark.parametrize(
    "url",
    [
        "not a url",
        "ftp://citymuseum.example/visit",
        "file:///etc/passwd",
        "citymuseum.example/visit",
        "",
    ],
)
def test_evidence_rejects_invalid_url(url: str) -> None:
    with pytest.raises(ValidationError):
        make_evidence(source_url=url)


# Missing required fields


@pytest.mark.parametrize(
    "field",
    ["id", "place_name", "claim_type", "original_excerpt", "normalized_statement"],
)
def test_claim_requires_field(field: str) -> None:
    data = make_claim().model_dump()
    del data[field]

    with pytest.raises(ValidationError) as exc_info:
        Claim(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "field",
    ["id", "claim_id", "source_url", "source_title", "content_excerpt", "retrieved_at"],
)
def test_evidence_requires_field(field: str) -> None:
    data = make_evidence().model_dump()
    del data[field]

    with pytest.raises(ValidationError) as exc_info:
        Evidence(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", ["claim_id", "verdict", "explanation", "evidence_ids"])
def test_verification_result_requires_field(field: str) -> None:
    data = make_result().model_dump()
    del data[field]

    with pytest.raises(ValidationError) as exc_info:
        VerificationResult(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


# JSON serialization


@pytest.mark.parametrize("scope", [None, "in summer"])
def test_claim_json_round_trip(scope: str | None) -> None:
    claim = make_claim(scope=scope)

    assert Claim.model_validate_json(claim.model_dump_json()) == claim


def test_evidence_json_serializes_url_and_datetime() -> None:
    evidence = make_evidence(source_url="https://citymuseum.example")

    data = evidence.model_dump(mode="json")

    # HttpUrl normalises a bare host by adding a trailing slash.
    assert data["source_url"] == "https://citymuseum.example/"
    assert data["retrieved_at"] == "2026-10-27T09:30:00Z"


def test_evidence_json_round_trip() -> None:
    evidence = make_evidence()

    assert Evidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_verification_result_json_round_trip() -> None:
    result = make_result()

    assert VerificationResult.model_validate_json(result.model_dump_json()) == result
