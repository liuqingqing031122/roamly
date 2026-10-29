import pytest
from pydantic import ValidationError

from roamly.domain import Claim
from roamly.extraction import ExtractedClaim, ExtractedClaims, ExtractionResult, SkippedClaim


def make_extracted(**overrides: object) -> ExtractedClaim:
    data = {
        "place_name": "City Museum",
        "claim_type": "opening_hours",
        "assertion_kind": "policy",
        "original_excerpt": "The museum is open every Monday.",
        "normalized_statement": "City Museum is open on Mondays.",
    }
    data.update(overrides)
    return ExtractedClaim(**data)


# ExtractedClaim


@pytest.mark.parametrize("claim_type", ["opening_hours", "reservation", "closure", "unsupported"])
def test_extracted_claim_accepts_types(claim_type: str) -> None:
    claim = make_extracted(claim_type=claim_type)

    assert claim.claim_type == claim_type


@pytest.mark.parametrize("assertion_kind", ["policy", "observation"])
def test_extracted_claim_accepts_assertion_kinds(assertion_kind: str) -> None:
    claim = make_extracted(assertion_kind=assertion_kind)

    assert claim.assertion_kind == assertion_kind


def test_extracted_claim_optional_fields_default_to_none() -> None:
    data = make_extracted().model_dump(exclude={"place_name", "scope"})

    claim = ExtractedClaim(**data)

    assert claim.place_name is None
    assert claim.scope is None


def test_extracted_claim_accepts_explicit_null_place() -> None:
    claim = make_extracted(place_name=None)

    assert claim.place_name is None


@pytest.mark.parametrize("claim_type", ["ticket_price", "Unsupported", ""])
def test_extracted_claim_rejects_unknown_type(claim_type: str) -> None:
    with pytest.raises(ValidationError):
        make_extracted(claim_type=claim_type)


@pytest.mark.parametrize("assertion_kind", ["rule", "Policy", ""])
def test_extracted_claim_rejects_unknown_assertion_kind(assertion_kind: str) -> None:
    with pytest.raises(ValidationError):
        make_extracted(assertion_kind=assertion_kind)


@pytest.mark.parametrize(
    "field",
    ["claim_type", "assertion_kind", "original_excerpt", "normalized_statement"],
)
def test_extracted_claim_requires_field(field: str) -> None:
    data = make_extracted().model_dump()
    del data[field]

    with pytest.raises(ValidationError) as exc_info:
        ExtractedClaim(**data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


# ExtractedClaims


def test_extracted_claims_parses_llm_json() -> None:
    raw = """
    {
      "claims": [
        {
          "place_name": null,
          "claim_type": "reservation",
          "assertion_kind": "policy",
          "original_excerpt": "Walk-ins are welcome.",
          "normalized_statement": "Reservations are not required.",
          "scope": "on weekdays"
        }
      ]
    }
    """

    parsed = ExtractedClaims.model_validate_json(raw)

    assert len(parsed.claims) == 1
    assert parsed.claims[0].place_name is None
    assert parsed.claims[0].scope == "on weekdays"


def test_extracted_claims_accepts_empty_list() -> None:
    assert ExtractedClaims(claims=[]).claims == []


def test_extracted_claims_rejects_invalid_item() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaims.model_validate({"claims": [{"claim_type": "opening_hours"}]})


# SkippedClaim


@pytest.mark.parametrize(
    "reason",
    ["unsupported_type", "observation", "excerpt_not_found", "duplicate", "missing_place"],
)
def test_skipped_claim_accepts_reasons(reason: str) -> None:
    skipped = SkippedClaim(claim=make_extracted(), reason=reason)

    assert skipped.reason == reason


@pytest.mark.parametrize("reason", ["invalid", "Observation", ""])
def test_skipped_claim_rejects_unknown_reason(reason: str) -> None:
    with pytest.raises(ValidationError):
        SkippedClaim(claim=make_extracted(), reason=reason)


# ExtractionResult


def test_extraction_result_json_round_trip() -> None:
    result = ExtractionResult(
        claims=[
            Claim(
                id="claim-1",
                place_name="City Museum",
                claim_type="opening_hours",
                original_excerpt="The museum is open every Monday.",
                normalized_statement="City Museum is open on Mondays.",
            )
        ],
        skipped=[
            SkippedClaim(claim=make_extracted(claim_type="unsupported"), reason="unsupported_type"),
            SkippedClaim(claim=make_extracted(place_name=None), reason="missing_place"),
        ],
    )

    assert ExtractionResult.model_validate_json(result.model_dump_json()) == result


def test_extraction_result_claims_reject_unsupported_type() -> None:
    with pytest.raises(ValidationError):
        ExtractionResult(
            claims=[
                {
                    "id": "claim-1",
                    "place_name": "City Museum",
                    "claim_type": "unsupported",
                    "original_excerpt": "Tickets cost 10 euros.",
                    "normalized_statement": "Admission costs 10 euros.",
                }
            ],
            skipped=[],
        )
