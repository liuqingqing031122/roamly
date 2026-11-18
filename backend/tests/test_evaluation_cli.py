import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from roamly.cli import MAX_SOURCE_CHARS
from roamly.evaluation.cli import EXIT_CASES_FAILED, main
from roamly.evaluation.predictions import load_run
from roamly.llm.deepseek import DeepSeekClient

API_KEY = "sk-test-secret-key"
FROZEN_DATASET = Path(__file__).parent / "fixtures" / "extraction_eval_v1.json"

GOOD_TEXT = "City Museum is open daily."
FAILING_TEXT = "Harbour Hall is closed until May 2027."


def make_dataset(path: Path, texts: list[str]) -> Path:
    cases = [
        {
            "id": f"case-{index}",
            "description": "",
            "tags": [],
            "source_text": text,
            "expected_claims": [],
            "expected_skipped": [],
            "must_not_extract": [],
            "annotation_notes": "",
        }
        for index, text in enumerate(texts, start=1)
    ]
    data = {
        "dataset": "test_dataset",
        "status": "draft",
        "guidelines": "",
        "notes": "",
        "cases": cases,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class MockDeepSeek:
    """Answers with one claim per request; returns HTTP 500 for `FAILING_TEXT`."""

    def __init__(self) -> None:
        self.request_count = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        if FAILING_TEXT in request.content.decode():
            return httpx.Response(500, json={"error": {"message": "server error"}})
        output = {
            "claims": [
                {
                    "place_name": "City Museum",
                    "claim_type": "opening_hours",
                    "assertion_kind": "policy",
                    "original_excerpt": GOOD_TEXT,
                    "normalized_statement": GOOD_TEXT,
                    "scope": [],
                }
            ]
        }
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": json.dumps(output)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return httpx.Response(200, json=body)

    def factory(self) -> Callable[[], DeepSeekClient]:
        return lambda: DeepSeekClient(API_KEY, transport=httpx.MockTransport(self.handler))


def run_cli(args: list[str], api: MockDeepSeek, tmp_path: Path) -> int:
    return main(
        ["run", *args, "--cache-dir", str(tmp_path / "cache")], client_factory=api.factory()
    )


def test_run_saves_a_new_run_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT])
    api = MockDeepSeek()

    exit_code = run_cli([str(dataset), "--output-dir", str(tmp_path / "runs")], api, tmp_path)

    path = Path(capsys.readouterr().out.strip())
    assert exit_code == 0
    assert path.parent == tmp_path / "runs"
    run = load_run(path)
    assert path.name == f"{run.run_id}.json"
    assert run.summary.succeeded == 1
    assert run.model_settings is not None
    assert run.model_settings["model"] == "deepseek-flash"
    assert run.predictions[0].input_tokens == 100


def test_run_file_contains_no_api_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT, FAILING_TEXT])

    run_cli([str(dataset), "--output-dir", str(tmp_path / "runs")], MockDeepSeek(), tmp_path)

    path = Path(capsys.readouterr().out.strip())
    assert API_KEY not in path.read_text(encoding="utf-8")


def test_failed_case_gives_distinct_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT, FAILING_TEXT])

    exit_code = run_cli(
        [str(dataset), "--output", str(tmp_path / "run.json")], MockDeepSeek(), tmp_path
    )

    run = load_run(tmp_path / "run.json")
    assert exit_code == EXIT_CASES_FAILED
    assert [p.status for p in run.predictions] == ["succeeded", "failed"]
    assert run.predictions[1].error is not None
    assert "HTTP 500" in run.predictions[1].error.message


def test_selected_case_only(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT, FAILING_TEXT])
    api = MockDeepSeek()

    exit_code = run_cli(
        [str(dataset), "--case", "case-1", "--output", str(tmp_path / "run.json")], api, tmp_path
    )

    assert exit_code == 0
    assert [p.case_id for p in load_run(tmp_path / "run.json").predictions] == ["case-1"]
    assert api.request_count == 1


def test_unknown_case_is_reported_without_api_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT])
    api = MockDeepSeek()

    exit_code = run_cli([str(dataset), "--case", "nope"], api, tmp_path)

    assert exit_code == 1
    assert "error: Unknown case ids: nope" in capsys.readouterr().err
    assert api.request_count == 0


def test_second_run_uses_cache_and_writes_a_separate_file(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT])
    api = MockDeepSeek()

    run_cli([str(dataset), "--output", str(tmp_path / "first.json")], api, tmp_path)
    run_cli([str(dataset), "--output", str(tmp_path / "second.json")], api, tmp_path)

    assert api.request_count == 1
    assert load_run(tmp_path / "second.json").predictions[0].cache_hit is True


def test_existing_output_is_not_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT])
    output = tmp_path / "run.json"
    output.write_text("earlier run", encoding="utf-8")
    api = MockDeepSeek()

    exit_code = run_cli([str(dataset), "--output", str(output)], api, tmp_path)

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "earlier run"
    assert api.request_count == 0


def test_overwrite_flag_replaces_output(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT])
    output = tmp_path / "run.json"
    output.write_text("earlier run", encoding="utf-8")

    exit_code = run_cli(
        [str(dataset), "--output", str(output), "--overwrite"], MockDeepSeek(), tmp_path
    )

    assert exit_code == 0
    assert load_run(output).summary.total == 1


def test_invalid_dataset_is_reported_without_api_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"dataset": "x"}), encoding="utf-8")
    api = MockDeepSeek()

    exit_code = run_cli([str(path)], api, tmp_path)

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "error: Invalid dataset" in err
    assert "cases: Field required" in err
    assert api.request_count == 0


def test_too_long_case_is_rejected_without_api_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "d.json", ["x" * (MAX_SOURCE_CHARS + 1)])
    api = MockDeepSeek()

    exit_code = run_cli([str(dataset)], api, tmp_path)

    assert exit_code == 1
    assert "case-1 is too long" in capsys.readouterr().err
    assert api.request_count == 0


def test_frozen_dataset_runs_end_to_end_with_mock(tmp_path: Path) -> None:
    before = FROZEN_DATASET.read_bytes()

    run_cli([str(FROZEN_DATASET), "--output", str(tmp_path / "run.json")], MockDeepSeek(), tmp_path)

    run = load_run(tmp_path / "run.json")
    assert FROZEN_DATASET.read_bytes() == before
    assert run.dataset.name == "extraction_eval_v1"
    assert run.dataset.status == "frozen"
    assert run.summary.total == 11
    assert len({p.case_id for p in run.predictions}) == 11


# Offline reading


def test_show_reads_a_saved_run_without_creating_a_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "d.json", [GOOD_TEXT, FAILING_TEXT])
    run_cli([str(dataset), "--output", str(tmp_path / "run.json")], MockDeepSeek(), tmp_path)
    capsys.readouterr()

    def no_client() -> DeepSeekClient:
        raise AssertionError("show must not create an LLM client")

    exit_code = main(["show", str(tmp_path / "run.json")], client_factory=no_client)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Dataset:  test_dataset (draft, 2 cases" in out
    assert "Cases:    2 run, 1 succeeded, 1 failed" in out
    assert "Code:     " in out
    assert "case-1  succeeded  1 accepted, 0 skipped" in out
    assert "case-2  failed     LLMAPIError: DeepSeek API returned HTTP 500" in out


def test_show_reports_invalid_run_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.json"
    path.write_text("{}", encoding="utf-8")

    exit_code = main(["show", str(path)])

    assert exit_code == 1
    assert "error: Invalid run file" in capsys.readouterr().err
