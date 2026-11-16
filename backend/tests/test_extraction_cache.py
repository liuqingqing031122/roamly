import json
from collections.abc import Callable
from pathlib import Path

import pytest

from roamly.extraction import ExtractedClaim, ExtractedClaims
from roamly.extraction_cache import ExtractionCache

SOURCE_TEXT = "The museum is open every Monday."
SETTINGS = {"provider": "fake", "model": "fake-model", "temperature": 0}


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


@pytest.fixture
def cache(tmp_path: Path) -> ExtractionCache:
    return ExtractionCache(tmp_path, SETTINGS)


def test_miss_returns_none(cache: ExtractionCache) -> None:
    assert cache.get(SOURCE_TEXT) is None


def test_put_then_get_round_trip(cache: ExtractionCache) -> None:
    cache.put(SOURCE_TEXT, make_output())

    assert cache.get(SOURCE_TEXT) == make_output()


def test_entry_records_key_fields_but_not_source_text(cache: ExtractionCache) -> None:
    cache.put("Secret guide text that should not be stored.", ExtractedClaims(claims=[]))

    [entry_path] = cache.directory.glob("*.json")
    content = entry_path.read_text(encoding="utf-8")
    entry = json.loads(content)

    assert "Secret guide text" not in content
    assert entry["key_fields"]["prompt_version"] == cache.prompt_version
    assert entry["key_fields"]["model_settings"] == SETTINGS
    assert entry["output"] == {"claims": []}


def test_no_temporary_files_are_left(cache: ExtractionCache) -> None:
    cache.put(SOURCE_TEXT, make_output())

    assert [path.suffix for path in cache.directory.iterdir()] == [".json"]


def test_same_inputs_give_same_key(tmp_path: Path) -> None:
    first = ExtractionCache(tmp_path, SETTINGS)
    second = ExtractionCache(tmp_path, dict(reversed(list(SETTINGS.items()))))

    assert first.key(SOURCE_TEXT) == second.key(SOURCE_TEXT)


@pytest.mark.parametrize(
    "make_other",
    [
        lambda d: ExtractionCache(d, SETTINGS, prompt_version="extraction-v2"),
        lambda d: ExtractionCache(d, SETTINGS, system_prompt="A different prompt."),
        lambda d: ExtractionCache(d, {**SETTINGS, "model": "other-model"}),
        lambda d: ExtractionCache(d, {**SETTINGS, "temperature": 0.7}),
    ],
    ids=["prompt_version", "system_prompt", "model", "temperature"],
)
def test_changed_configuration_does_not_hit(
    cache: ExtractionCache, tmp_path: Path, make_other: Callable[[Path], ExtractionCache]
) -> None:
    cache.put(SOURCE_TEXT, make_output())
    other = make_other(tmp_path)

    assert other.key(SOURCE_TEXT) != cache.key(SOURCE_TEXT)
    assert other.get(SOURCE_TEXT) is None


def test_different_text_does_not_hit(cache: ExtractionCache) -> None:
    cache.put(SOURCE_TEXT, make_output())

    assert cache.get(SOURCE_TEXT + " ") is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        '{"output": {"claims": [{"claim_type": "price"}]}}',
        '{"key_fields": {}}',
        "",
    ],
)
def test_corrupt_entry_is_treated_as_miss(
    cache: ExtractionCache, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.path(SOURCE_TEXT).write_text(content, encoding="utf-8")

    assert cache.get(SOURCE_TEXT) is None
    assert "Ignoring unreadable cache entry" in caplog.text


def test_corrupt_entry_is_overwritten_by_put(cache: ExtractionCache) -> None:
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.path(SOURCE_TEXT).write_text("{not json", encoding="utf-8")

    cache.put(SOURCE_TEXT, make_output())

    assert cache.get(SOURCE_TEXT) == make_output()
