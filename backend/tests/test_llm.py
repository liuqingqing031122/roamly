import pytest

from roamly.extraction import ExtractedClaim, ExtractedClaims, ExtractionResult
from roamly.llm import FakeLLMClient, LLMCall, LLMClient, TokenUsage


def make_output() -> ExtractedClaims:
    return ExtractedClaims(
        claims=[
            ExtractedClaim(
                place_name="City Museum",
                claim_type="opening_hours",
                assertion_kind="policy",
                original_excerpt="The museum is open every Monday.",
                normalized_statement="City Museum is open on Mondays.",
            )
        ]
    )


def call(client: LLMClient, user_prompt: str = "guide text") -> ExtractedClaims:
    # Typed against the protocol so a type checker can confirm FakeLLMClient satisfies it.
    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt=user_prompt, response_model=ExtractedClaims
    )
    return response.output


def test_fake_client_returns_predefined_output() -> None:
    expected = make_output()
    client = FakeLLMClient(outputs=[expected])

    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt="guide text", response_model=ExtractedClaims
    )

    assert response.output == expected
    assert response.usage == TokenUsage(input_tokens=0, output_tokens=0)


def test_fake_client_returns_outputs_in_order() -> None:
    first = make_output()
    second = ExtractedClaims(claims=[])
    client = FakeLLMClient(outputs=[first, second])

    assert call(client) == first
    assert call(client) == second


def test_fake_client_records_calls() -> None:
    client = FakeLLMClient(outputs=[make_output()])

    call(client, user_prompt="The museum is open every Monday.")

    assert client.calls == [
        LLMCall(
            system_prompt="Extract claims.",
            user_prompt="The museum is open every Monday.",
            response_model=ExtractedClaims,
        )
    ]


def test_fake_client_raises_when_outputs_exhausted() -> None:
    client = FakeLLMClient(outputs=[])

    with pytest.raises(RuntimeError, match="no more predefined outputs"):
        call(client)


def test_fake_client_rejects_mismatched_output_type() -> None:
    client = FakeLLMClient(outputs=[ExtractionResult(claims=[], skipped=[])])

    with pytest.raises(TypeError, match="Expected ExtractedClaims"):
        call(client)
