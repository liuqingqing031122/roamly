from roamly.extraction import ExtractedClaims, process_extracted_claims
from roamly.extraction_prompt import (
    EXTRACTION_PROMPT_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_user_prompt,
)


def split_example() -> tuple[str, str]:
    """Return the example guide text and example output embedded in the system prompt."""
    example = EXTRACTION_SYSTEM_PROMPT.split("## Example", 1)[1]
    guide_text, output = example.split("Output:", 1)
    return guide_text.split("Guide text:", 1)[1].strip(), output.strip()


def test_prompt_has_version() -> None:
    assert EXTRACTION_PROMPT_VERSION == "extraction-v1"


def test_system_prompt_names_all_claim_types_and_assertion_kinds() -> None:
    for value in ["opening_hours", "reservation", "closure", "unsupported"]:
        assert f'"{value}"' in EXTRACTION_SYSTEM_PROMPT
    for value in ["policy", "observation"]:
        assert f'"{value}"' in EXTRACTION_SYSTEM_PROMPT


def test_user_prompt_wraps_source_text_in_delimiters() -> None:
    source_text = "The museum is open every Monday."

    user_prompt = build_extraction_user_prompt(source_text)

    assert user_prompt == f"<travel_guide>\n{source_text}\n</travel_guide>"
    assert source_text not in EXTRACTION_SYSTEM_PROMPT


def test_example_output_matches_schema() -> None:
    _, output = split_example()

    example = ExtractedClaims.model_validate_json(output)

    assert [claim.claim_type for claim in example.claims] == [
        "opening_hours",
        "reservation",
        "reservation",
        "unsupported",
    ]


def test_example_normalized_statements_name_the_place() -> None:
    _, output = split_example()
    example = ExtractedClaims.model_validate_json(output)

    for claim in example.claims:
        assert claim.place_name is not None
        assert claim.place_name in claim.normalized_statement
    # The venue name may come from context rather than the excerpt itself.
    assert any(claim.place_name not in claim.original_excerpt for claim in example.claims)


def test_example_output_passes_post_processing() -> None:
    guide_text, output = split_example()

    result = process_extracted_claims(guide_text, ExtractedClaims.model_validate_json(output))

    # Every example excerpt is found verbatim, so only the intended skips remain.
    assert len(result.claims) == 2
    assert [skipped.reason for skipped in result.skipped] == ["observation", "unsupported_type"]
