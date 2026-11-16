from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from roamly.extraction import ExtractedClaim, ExtractedClaims
from roamly.extraction_cache import ExtractionCache
from roamly.extraction_prompt import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from roamly.llm import FakeLLMClient, LLMCall, LLMResponseError, StructuredResponse
from roamly.pipeline import extract_claims

SOURCE_TEXT = "The museum is open every Monday. Tickets cost 10 euros."
SETTINGS = {"provider": "fake", "model": "fake-model"}


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


@dataclass
class FailingLLMClient:
    """Raises the given error on every call."""

    error: Exception
    call_count: int = 0

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        self.call_count += 1
        raise self.error


@pytest.fixture
def cache(tmp_path: Path) -> ExtractionCache:
    return ExtractionCache(tmp_path / "cache", SETTINGS)


def test_sends_extraction_prompt_and_schema() -> None:
    client = FakeLLMClient(outputs=[ExtractedClaims(claims=[])])

    extract_claims(SOURCE_TEXT, client)

    assert client.calls == [
        LLMCall(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_extraction_user_prompt(SOURCE_TEXT),
            response_model=ExtractedClaims,
        )
    ]


def test_post_processing_filters_excerpt_not_in_source() -> None:
    invented = make_extracted(original_excerpt="The museum is open daily.")
    client = FakeLLMClient(outputs=[ExtractedClaims(claims=[make_extracted(), invented])])

    result = extract_claims(SOURCE_TEXT, client)

    assert [claim.original_excerpt for claim in result.claims] == [
        "The museum is open every Monday."
    ]
    assert [(s.claim, s.reason) for s in result.skipped] == [(invented, "excerpt_not_found")]


def test_cache_miss_calls_llm_and_stores_output(cache: ExtractionCache) -> None:
    output = ExtractedClaims(claims=[make_extracted()])
    client = FakeLLMClient(outputs=[output])

    extract_claims(SOURCE_TEXT, client, cache)

    assert len(client.calls) == 1
    assert cache.get(SOURCE_TEXT) == output


def test_cache_hit_skips_llm(cache: ExtractionCache) -> None:
    # Only one predefined output: a second LLM call would raise.
    client = FakeLLMClient(outputs=[ExtractedClaims(claims=[make_extracted()])])

    first = extract_claims(SOURCE_TEXT, client, cache)
    second = extract_claims(SOURCE_TEXT, client, cache)

    assert len(client.calls) == 1
    assert second == first


def test_cache_hit_still_runs_post_processing(cache: ExtractionCache) -> None:
    unsupported = make_extracted(
        claim_type="unsupported",
        original_excerpt="Tickets cost 10 euros.",
        normalized_statement="Admission costs 10 euros.",
    )
    invented = make_extracted(original_excerpt="The museum is open daily.")
    cache.put(SOURCE_TEXT, ExtractedClaims(claims=[make_extracted(), unsupported, invented]))
    client = FakeLLMClient(outputs=[])

    result = extract_claims(SOURCE_TEXT, client, cache)

    assert client.calls == []
    assert [claim.id for claim in result.claims] == ["claim-1"]
    assert [s.reason for s in result.skipped] == ["unsupported_type", "excerpt_not_found"]


def test_llm_failure_is_not_cached(cache: ExtractionCache) -> None:
    client = FailingLLMClient(LLMResponseError("DeepSeek returned empty content"))

    with pytest.raises(LLMResponseError):
        extract_claims(SOURCE_TEXT, client, cache)

    assert cache.get(SOURCE_TEXT) is None
    assert not cache.directory.exists()


def test_corrupt_cache_entry_falls_back_to_llm(cache: ExtractionCache) -> None:
    cache.directory.mkdir(parents=True)
    cache.path(SOURCE_TEXT).write_text("{not json", encoding="utf-8")
    output = ExtractedClaims(claims=[make_extracted()])
    client = FakeLLMClient(outputs=[output])

    result = extract_claims(SOURCE_TEXT, client, cache)

    assert len(client.calls) == 1
    assert len(result.claims) == 1
    assert cache.get(SOURCE_TEXT) == output


def test_cache_write_failure_still_returns_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A regular file where the cache directory should be makes every write fail.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    cache = ExtractionCache(blocked / "cache", SETTINGS)
    client = FakeLLMClient(outputs=[ExtractedClaims(claims=[make_extracted()])])

    result = extract_claims(SOURCE_TEXT, client, cache)

    assert len(result.claims) == 1
    assert "Could not write extraction cache" in caplog.text
