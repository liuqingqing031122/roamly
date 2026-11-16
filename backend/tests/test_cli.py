import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from roamly.cli import MAX_SOURCE_CHARS, main
from roamly.extraction import ExtractionResult
from roamly.llm.deepseek import DeepSeekClient

GUIDE_TEXT = (
    "The museum is open every Monday.\nI visited the museum on Monday.\nTickets cost 10 euros.\n"
)

LLM_OUTPUT = {
    "claims": [
        {
            "place_name": "City Museum",
            "claim_type": "opening_hours",
            "assertion_kind": "policy",
            "original_excerpt": "The museum is open every Monday.",
            "normalized_statement": "City Museum is open on Mondays.",
            "scope": [],
        },
        {
            "place_name": "City Museum",
            "claim_type": "opening_hours",
            "assertion_kind": "observation",
            "original_excerpt": "I visited the museum on Monday.",
            "normalized_statement": "The author visited City Museum on a Monday.",
            "scope": [],
        },
        {
            "place_name": "City Museum",
            "claim_type": "opening_hours",
            "assertion_kind": "policy",
            "original_excerpt": "The museum is open daily.",
            "normalized_statement": "City Museum is open daily.",
            "scope": [],
        },
    ]
}


class MockDeepSeek:
    """Builds DeepSeek clients backed by a mock transport and counts API requests."""

    def __init__(self, status_code: int = 200, body: object | None = None) -> None:
        self.status_code = status_code
        self.body = body if body is not None else self.completion(LLM_OUTPUT)
        self.request_count = 0

    @staticmethod
    def completion(output: object) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(output)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        return httpx.Response(self.status_code, json=self.body)

    def factory(self) -> Callable[[], DeepSeekClient]:
        return lambda: DeepSeekClient("sk-test", transport=httpx.MockTransport(self.handler))


@pytest.fixture
def guide(tmp_path: Path) -> Path:
    path = tmp_path / "guide.txt"
    path.write_text(GUIDE_TEXT, encoding="utf-8")
    return path


def run(args: list[str], api: MockDeepSeek, cache_dir: Path) -> int:
    return main([*args, "--cache-dir", str(cache_dir)], client_factory=api.factory())


def test_prints_extraction_result_json(
    guide: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    api = MockDeepSeek()

    exit_code = run([str(guide)], api, tmp_path / "cache")

    result = ExtractionResult.model_validate_json(capsys.readouterr().out)
    assert exit_code == 0
    assert [claim.original_excerpt for claim in result.claims] == [
        "The museum is open every Monday."
    ]
    # The invented excerpt is filtered by post-processing.
    assert [s.reason for s in result.skipped] == ["observation", "excerpt_not_found"]
    assert "100 input and 50 output tokens" in caplog.text


def test_second_run_uses_cache(
    guide: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    api = MockDeepSeek()

    run([str(guide)], api, tmp_path / "cache")
    first_out = capsys.readouterr().out
    caplog.clear()
    run([str(guide)], api, tmp_path / "cache")

    assert api.request_count == 1
    assert capsys.readouterr().out == first_out
    assert "Extraction cache hit" in caplog.text


def test_no_cache_always_calls_api(guide: Path, tmp_path: Path) -> None:
    api = MockDeepSeek()

    run([str(guide), "--no-cache"], api, tmp_path / "cache")
    run([str(guide), "--no-cache"], api, tmp_path / "cache")

    assert api.request_count == 2
    assert not (tmp_path / "cache").exists()


def test_different_model_does_not_hit_cache(guide: Path, tmp_path: Path) -> None:
    api = MockDeepSeek()
    transport = httpx.MockTransport(api.handler)

    for model in ["deepseek-flash", "deepseek-v4-pro"]:
        main(
            [str(guide), "--cache-dir", str(tmp_path / "cache")],
            client_factory=lambda model=model: DeepSeekClient(
                "sk-test", model, transport=transport
            ),
        )

    assert api.request_count == 2


def test_missing_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    api = MockDeepSeek()

    exit_code = run([str(tmp_path / "missing.txt")], api, tmp_path / "cache")

    assert exit_code == 1
    assert "error: File not found" in capsys.readouterr().err
    assert api.request_count == 0


@pytest.mark.parametrize("content", ["", "  \n\t\n"])
def test_empty_file_is_reported(
    tmp_path: Path, content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")
    api = MockDeepSeek()

    exit_code = run([str(path)], api, tmp_path / "cache")

    assert exit_code == 1
    assert "error: File is empty" in capsys.readouterr().err
    assert api.request_count == 0


def test_non_txt_file_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "guide.md"
    path.write_text(GUIDE_TEXT, encoding="utf-8")

    exit_code = run([str(path)], MockDeepSeek(), tmp_path / "cache")

    assert exit_code == 1
    assert "Only .txt files are supported" in capsys.readouterr().err


def test_non_utf8_file_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "guide.txt"
    path.write_bytes(b"\xff\xfe invalid")

    exit_code = run([str(path)], MockDeepSeek(), tmp_path / "cache")

    assert exit_code == 1
    assert "not valid UTF-8" in capsys.readouterr().err


def test_missing_api_key_is_reported(
    guide: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ROAMLY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("ROAMLY_LLM_PROVIDER", raising=False)

    exit_code = main([str(guide), "--cache-dir", str(tmp_path / "cache")])

    assert exit_code == 1
    assert "error: Environment variable ROAMLY_LLM_API_KEY is not set" in capsys.readouterr().err


def test_unsupported_provider_is_reported(
    guide: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ROAMLY_LLM_PROVIDER", "openai")

    exit_code = main([str(guide), "--cache-dir", str(tmp_path / "cache")])

    assert exit_code == 1
    assert "Unsupported LLM provider 'openai'" in capsys.readouterr().err


def test_api_failure_is_reported_and_not_cached(
    guide: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = MockDeepSeek(status_code=500, body={"error": {"message": "Server error"}})

    exit_code = run([str(guide)], api, tmp_path / "cache")

    out, err = capsys.readouterr()
    assert exit_code == 1
    assert out == ""
    assert "error: LLM call failed: DeepSeek API returned HTTP 500: Server error" in err
    assert not (tmp_path / "cache").exists()


def test_invalid_llm_output_is_reported_and_not_cached(
    guide: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = MockDeepSeek(body=MockDeepSeek.completion({"claims": [{"claim_type": "price"}]}))

    exit_code = run([str(guide)], api, tmp_path / "cache")

    assert exit_code == 1
    assert "does not match ExtractedClaims" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()


def test_corrupt_cache_falls_back_to_api(
    guide: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = MockDeepSeek()
    cache_dir = tmp_path / "cache"
    run([str(guide)], api, cache_dir)
    [entry] = cache_dir.glob("*.json")
    entry.write_text("{not json", encoding="utf-8")
    capsys.readouterr()

    exit_code = run([str(guide)], api, cache_dir)

    assert exit_code == 0
    assert api.request_count == 2
    ExtractionResult.model_validate_json(capsys.readouterr().out)


def write_guide_of_length(tmp_path: Path, length: int, padding: str = "x") -> Path:
    path = tmp_path / "long.txt"
    path.write_text(GUIDE_TEXT + padding * (length - len(GUIDE_TEXT)), encoding="utf-8")
    return path


@pytest.mark.parametrize("padding", ["x", "é"])
def test_text_at_length_limit_is_accepted(
    tmp_path: Path, padding: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The limit counts characters, not bytes: "é" is two bytes in UTF-8.
    path = write_guide_of_length(tmp_path, MAX_SOURCE_CHARS, padding)
    api = MockDeepSeek()

    exit_code = run([str(path)], api, tmp_path / "cache")

    assert exit_code == 0
    assert api.request_count == 1
    ExtractionResult.model_validate_json(capsys.readouterr().out)


def test_text_over_length_limit_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_guide_of_length(tmp_path, MAX_SOURCE_CHARS + 1)
    api = MockDeepSeek()

    exit_code = run([str(path)], api, tmp_path / "cache")

    out, err = capsys.readouterr()
    assert exit_code == 1
    assert out == ""
    assert "error: File is too long: 30,001 characters (maximum 30,000)" in err
    assert api.request_count == 0
    assert not (tmp_path / "cache").exists()
