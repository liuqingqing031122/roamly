from pathlib import Path

import pytest
from pydantic import ValidationError

from roamly.domain import Claim
from roamly.extraction import (
    ExtractedClaim,
    ExtractedClaims,
    ExtractionResult,
    SkippedClaim,
    process_extracted_claims,
)
from roamly.llm import FakeLLMClient

SOURCE_TEXT = (
    "The museum is open every Monday. "
    "I visited the museum on Monday. "
    "Tickets cost 10 euros. "
    "Reservations are required at Cafe Lumen. "
    "Cafe Lumen temporarily closes for renovation in January 2027."
)


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


def test_extracted_claim_optional_fields_have_defaults() -> None:
    data = make_extracted().model_dump(exclude={"place_name", "scope"})

    claim = ExtractedClaim(**data)

    assert claim.place_name is None
    assert claim.scope == []


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ([], []),
        (["on weekdays"], ["on weekdays"]),
        (["on weekends", "only during summer"], ["on weekends", "only during summer"]),
        # The LLM may still return the old null or single-string form.
        (None, []),
        ("on weekdays", ["on weekdays"]),
    ],
)
def test_extracted_claim_scope_becomes_list(scope: object, expected: list[str]) -> None:
    assert make_extracted(scope=scope).scope == expected


@pytest.mark.parametrize("scope", [42, [1, 2], {"when": "summer"}])
def test_extracted_claim_rejects_invalid_scope(scope: object) -> None:
    with pytest.raises(ValidationError):
        make_extracted(scope=scope)


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
          "scope": ["on weekdays"]
        }
      ]
    }
    """

    parsed = ExtractedClaims.model_validate_json(raw)

    assert len(parsed.claims) == 1
    assert parsed.claims[0].place_name is None
    assert parsed.claims[0].scope == ["on weekdays"]


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


# Post-processing


def process(*candidates: ExtractedClaim) -> ExtractionResult:
    return process_extracted_claims(SOURCE_TEXT, ExtractedClaims(claims=list(candidates)))


def test_valid_candidate_becomes_claim() -> None:
    result = process(make_extracted(scope=["except public holidays"]))

    assert result.skipped == []
    assert result.claims == [
        Claim(
            id="claim-1",
            place_name="City Museum",
            claim_type="opening_hours",
            original_excerpt="The museum is open every Monday.",
            normalized_statement="City Museum is open on Mondays.",
            scope=["except public holidays"],
        )
    ]


def test_scope_defaults_to_empty_list_on_claim() -> None:
    result = process(make_extracted())

    assert result.claims[0].scope == []


def test_multiple_scope_items_are_preserved_in_order() -> None:
    scope = ["on weekends", "only during summer"]

    result = process(make_extracted(scope=scope))

    assert result.claims[0].scope == scope


def test_blank_scope_items_are_dropped_and_items_trimmed() -> None:
    result = process(make_extracted(scope=[" in summer ", "", "   "]))

    assert result.claims[0].scope == ["in summer"]


def test_unsupported_type_is_skipped() -> None:
    candidate = make_extracted(
        claim_type="unsupported",
        original_excerpt="Tickets cost 10 euros.",
        normalized_statement="Admission costs 10 euros.",
    )

    result = process(candidate)

    assert result.claims == []
    assert result.skipped == [SkippedClaim(claim=candidate, reason="unsupported_type")]


def test_observation_is_skipped() -> None:
    candidate = make_extracted(
        assertion_kind="observation", original_excerpt="I visited the museum on Monday."
    )

    result = process(candidate)

    assert result.claims == []
    assert result.skipped == [SkippedClaim(claim=candidate, reason="observation")]


@pytest.mark.parametrize(
    "excerpt",
    [
        "The museum is open every day.",  # not in the source
        "the museum is open every monday.",  # case differs
        " The museum is open every Monday.",  # extra leading whitespace
        "",
        "   ",
    ],
)
def test_excerpt_not_in_source_is_skipped(excerpt: str) -> None:
    candidate = make_extracted(original_excerpt=excerpt)

    result = process(candidate)

    assert result.claims == []
    assert result.skipped == [SkippedClaim(claim=candidate, reason="excerpt_not_found")]


def test_excerpt_is_kept_verbatim() -> None:
    excerpt = "open every Monday"

    result = process(make_extracted(original_excerpt=excerpt))

    assert result.claims[0].original_excerpt == excerpt


WRAPPED_SOURCE = "Start at the Lello Bookshop. Entry requires a ticket bought\nonline in advance."


@pytest.mark.parametrize(
    "source",
    [
        WRAPPED_SOURCE,
        WRAPPED_SOURCE.replace("bought\n", "bought\r\n"),
        WRAPPED_SOURCE.replace("bought\n", "bought \n  "),
        WRAPPED_SOURCE.replace("bought\n", "bought\t"),
    ],
    ids=["line_break", "crlf", "spaces_around_break", "tab"],
)
def test_excerpt_matches_across_source_whitespace(source: str) -> None:
    candidate = make_extracted(
        place_name="Lello Bookshop",
        claim_type="reservation",
        original_excerpt="Entry requires a ticket bought online in advance",
        normalized_statement="Lello Bookshop requires a ticket bought online in advance.",
    )

    result = process_extracted_claims(source, ExtractedClaims(claims=[candidate]))

    assert result.skipped == []
    # The stored excerpt is the source span, so it is still an exact substring.
    [claim] = result.claims
    assert claim.original_excerpt in source
    assert claim.original_excerpt.startswith("Entry requires a ticket bought")
    assert claim.original_excerpt.endswith("online in advance")


def test_excerpt_whitespace_runs_in_the_llm_output_also_match() -> None:
    candidate = make_extracted(original_excerpt="The museum  is\nopen every Monday.")

    [claim] = process(candidate).claims

    assert claim.original_excerpt == "The museum is open every Monday."


@pytest.mark.parametrize(
    "excerpt",
    [
        "Entry requires a ticket bought online in  advance!",  # punctuation differs
        "Entry requires a ticket purchased online in advance",  # word differs
        "entry requires a ticket bought online in advance",  # case differs
        "Entry requires a ticket boughtonline in advance",  # whitespace removed
        "Entry requires a ticket bou ght online in advance",  # whitespace added in a word
    ],
)
def test_whitespace_tolerance_does_not_relax_content(excerpt: str) -> None:
    candidate = make_extracted(original_excerpt=excerpt)

    result = process_extracted_claims(WRAPPED_SOURCE, ExtractedClaims(claims=[candidate]))

    assert result.skipped == [SkippedClaim(claim=candidate, reason="excerpt_not_found")]


@pytest.mark.parametrize("place_name", [None, "", "   ", "\t\n"])
def test_missing_place_is_skipped(place_name: str | None) -> None:
    candidate = make_extracted(place_name=place_name)

    result = process(candidate)

    assert result.claims == []
    assert result.skipped == [SkippedClaim(claim=candidate, reason="missing_place")]


def test_place_name_is_trimmed_on_claim() -> None:
    result = process(make_extracted(place_name="  City Museum "))

    assert result.claims[0].place_name == "City Museum"


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"claim_type": "unsupported", "assertion_kind": "observation"}, "unsupported_type"),
        ({"assertion_kind": "observation", "original_excerpt": "Not in source."}, "observation"),
        ({"original_excerpt": "Not in source.", "place_name": None}, "excerpt_not_found"),
    ],
)
def test_checks_apply_in_documented_order(
    overrides: dict[str, object], expected_reason: str
) -> None:
    result = process(make_extracted(**overrides))

    assert result.skipped[0].reason == expected_reason


def test_duplicate_ignores_case_and_outer_whitespace() -> None:
    first = make_extracted()
    second = make_extracted(
        place_name="  city museum ",
        normalized_statement=" CITY MUSEUM IS OPEN ON MONDAYS. ",
    )

    result = process(first, second)

    assert [claim.id for claim in result.claims] == ["claim-1"]
    assert result.skipped == [SkippedClaim(claim=second, reason="duplicate")]


@pytest.mark.parametrize(
    "overrides",
    [
        {"place_name": "Cafe Lumen"},
        {"claim_type": "closure"},
        {"normalized_statement": "City Museum is closed on Mondays."},
    ],
)
def test_different_key_is_not_duplicate(overrides: dict[str, object]) -> None:
    result = process(make_extracted(), make_extracted(**overrides))

    assert len(result.claims) == 2
    assert result.skipped == []


def test_duplicate_compares_only_against_accepted_claims() -> None:
    rejected = make_extracted(original_excerpt="Not in source.")
    valid = make_extracted()

    result = process(rejected, valid)

    assert [claim.id for claim in result.claims] == ["claim-1"]
    assert result.skipped == [SkippedClaim(claim=rejected, reason="excerpt_not_found")]


def test_multiple_candidates_keep_order_and_get_sequential_ids() -> None:
    museum = make_extracted()
    tickets = make_extracted(
        claim_type="unsupported",
        original_excerpt="Tickets cost 10 euros.",
        normalized_statement="Admission costs 10 euros.",
    )
    reservation = make_extracted(
        place_name="Cafe Lumen",
        claim_type="reservation",
        original_excerpt="Reservations are required at Cafe Lumen.",
        normalized_statement="Cafe Lumen requires reservations.",
    )
    visit = make_extracted(
        assertion_kind="observation", original_excerpt="I visited the museum on Monday."
    )
    closure = make_extracted(
        place_name="Cafe Lumen",
        claim_type="closure",
        original_excerpt="Cafe Lumen temporarily closes for renovation in January 2027.",
        normalized_statement="Cafe Lumen is temporarily closed for renovation in January 2027.",
        scope=["in January 2027"],
    )

    result = process(museum, tickets, reservation, visit, closure)

    assert [(c.id, c.claim_type) for c in result.claims] == [
        ("claim-1", "opening_hours"),
        ("claim-2", "reservation"),
        ("claim-3", "closure"),
    ]
    assert [(s.claim, s.reason) for s in result.skipped] == [
        (tickets, "unsupported_type"),
        (visit, "observation"),
    ]


def test_processing_is_deterministic() -> None:
    candidates = [make_extracted(), make_extracted(place_name="Cafe Lumen")]

    assert process(*candidates) == process(*candidates)


def test_empty_input_gives_empty_result() -> None:
    assert process() == ExtractionResult(claims=[], skipped=[])


def test_fake_llm_response_flows_through_post_processing() -> None:
    llm_json = """
    {
      "claims": [
        {
          "place_name": "City Museum",
          "claim_type": "opening_hours",
          "assertion_kind": "policy",
          "original_excerpt": "The museum is open every Monday.",
          "normalized_statement": "City Museum is open on Mondays."
        },
        {
          "place_name": "City Museum",
          "claim_type": "opening_hours",
          "assertion_kind": "observation",
          "original_excerpt": "I visited the museum on Monday.",
          "normalized_statement": "The author visited City Museum on a Monday."
        },
        {
          "place_name": "City Museum",
          "claim_type": "unsupported",
          "assertion_kind": "policy",
          "original_excerpt": "Tickets cost 10 euros.",
          "normalized_statement": "Admission costs 10 euros."
        }
      ]
    }
    """
    client = FakeLLMClient(outputs=[ExtractedClaims.model_validate_json(llm_json)])

    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt=SOURCE_TEXT, response_model=ExtractedClaims
    )
    result = process_extracted_claims(SOURCE_TEXT, response.output)

    # AC-03: a recurring opening rule becomes a claim linked to the exact excerpt.
    assert [(c.place_name, c.claim_type, c.original_excerpt) for c in result.claims] == [
        ("City Museum", "opening_hours", "The museum is open every Monday.")
    ]
    # AC-04 and AC-06: the one-time visit and the ticket price are not claims.
    assert [s.reason for s in result.skipped] == ["observation", "unsupported_type"]


# Regression: real DeepSeek output for tests/fixtures/porto_guide.txt (M2.3 smoke test),
# where three excerpts span hard line breaks of the guide.
PORTO_GUIDE = Path(__file__).parent / "fixtures" / "porto_guide.txt"
PORTO_CANDIDATES = [
    ("Lello Bookshop", "reservation", "policy", "Entry requires a ticket bought online in advance"),
    ("Lello Bookshop", "opening_hours", "policy", "the bookshop opens at 9am every day"),
    ("Torre dos Clérigos", "opening_hours", "policy", "The tower is open daily from 9am to 7pm."),
    (
        "Majestic Café",
        "reservation",
        "policy",
        "the Majestic Café on Rua de Santa Catarina accepts reservations for groups only",
    ),
    ("Majestic Café", "closure", "policy", "It is closed on Sundays."),
    (
        "Lello Bookshop",
        "unsupported",
        "observation",
        "We queued for twenty minutes at the bookshop on a Saturday morning.",
    ),
]


def test_porto_guide_claims_pass_post_processing() -> None:
    source = PORTO_GUIDE.read_text(encoding="utf-8")
    extracted = ExtractedClaims(
        claims=[
            make_extracted(
                place_name=place,
                claim_type=claim_type,
                assertion_kind=kind,
                original_excerpt=excerpt,
                normalized_statement=f"{place}: {excerpt}",
            )
            for place, claim_type, kind, excerpt in PORTO_CANDIDATES
        ]
    )

    result = process_extracted_claims(source, extracted)

    assert [(c.place_name, c.claim_type) for c in result.claims] == [
        ("Lello Bookshop", "reservation"),
        ("Lello Bookshop", "opening_hours"),
        ("Torre dos Clérigos", "opening_hours"),
        ("Majestic Café", "reservation"),
        ("Majestic Café", "closure"),
    ]
    assert all(c.original_excerpt in source for c in result.claims)
    assert [s.reason for s in result.skipped] == ["unsupported_type"]
    assert result.skipped[0].claim.original_excerpt.startswith("We queued")
