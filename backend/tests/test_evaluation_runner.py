import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from roamly.evaluation.dataset import LoadedDataset, load_dataset
from roamly.evaluation.git_info import GitInfo
from roamly.evaluation.predictions import (
    EvaluationRun,
    RunFileError,
    load_run,
    save_run,
)
from roamly.evaluation.runner import run_evaluation
from roamly.extraction import ExtractedClaim, ExtractedClaims
from roamly.extraction_cache import ExtractionCache
from roamly.extraction_prompt import EXTRACTION_PROMPT_VERSION
from roamly.llm import FakeLLMClient, LLMAPIError, StructuredResponse, TokenUsage

SETTINGS = {"provider": "fake", "model": "fake-model", "temperature": 0}

CASES = [
    {
        "id": "case-a",
        "source_text": "City Museum is open daily. Tickets cost 10 euros.",
        "excerpt": "City Museum is open daily.",
    },
    {
        "id": "case-b",
        "source_text": "Rose Cafe requires a reservation for dinner.",
        "excerpt": "Rose Cafe requires a reservation for dinner.",
    },
    {
        "id": "case-c",
        "source_text": "Harbour Hall is closed until May 2027.",
        "excerpt": "Harbour Hall is closed until May 2027.",
    },
]


def make_dataset_file(path: Path) -> Path:
    cases = [
        {
            "id": case["id"],
            "description": "",
            "tags": [],
            "source_text": case["source_text"],
            "expected_claims": [],
            "expected_skipped": [],
            "must_not_extract": [],
            "annotation_notes": "",
        }
        for case in CASES
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


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedDataset:
    return load_dataset(make_dataset_file(tmp_path / "dataset.json"))


def output_for(excerpt: str, **overrides: object) -> ExtractedClaims:
    data: dict[str, object] = {
        "place_name": "Some Venue",
        "claim_type": "opening_hours",
        "assertion_kind": "policy",
        "original_excerpt": excerpt,
        "normalized_statement": excerpt,
    }
    data.update(overrides)
    return ExtractedClaims(claims=[ExtractedClaim(**data)])


@dataclass
class ScriptedClient:
    """Answers by source text; raises the configured error for selected texts."""

    outputs: dict[str, ExtractedClaims]
    errors: dict[str, Exception] = field(default_factory=dict)
    usage: TokenUsage = field(
        default_factory=lambda: TokenUsage(input_tokens=100, output_tokens=20)
    )
    calls: int = 0

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        self.calls += 1
        for text, error in self.errors.items():
            if text in user_prompt:
                raise error
        for text, output in self.outputs.items():
            if text in user_prompt:
                assert isinstance(output, response_model)
                return StructuredResponse(output=output, usage=self.usage)
        raise AssertionError("unexpected prompt")


def all_outputs() -> dict[str, ExtractedClaims]:
    return {case["source_text"]: output_for(case["excerpt"]) for case in CASES}


def test_runs_every_case_in_dataset_order(loaded: LoadedDataset) -> None:
    client = ScriptedClient(all_outputs())

    run = run_evaluation(loaded, client, model_settings=SETTINGS)

    assert [p.case_id for p in run.predictions] == ["case-a", "case-b", "case-c"]
    assert all(p.status == "succeeded" for p in run.predictions)
    assert run.summary.model_dump() == {"total": 3, "succeeded": 3, "failed": 0}
    assert client.calls == 3


def test_reuses_pipeline_post_processing(loaded: LoadedDataset) -> None:
    outputs = all_outputs()
    source_a = CASES[0]["source_text"]
    outputs[source_a] = ExtractedClaims(
        claims=[
            *output_for("City Museum is open daily.").claims,
            *output_for("Tickets cost 10 euros.", claim_type="unsupported").claims,
            *output_for("Invented text.").claims,
        ]
    )

    run = run_evaluation(loaded, ScriptedClient(outputs), model_settings=SETTINGS)

    result = run.predictions[0].result
    assert result is not None
    assert [claim.id for claim in result.claims] == ["claim-1"]
    assert [s.reason for s in result.skipped] == ["unsupported_type", "excerpt_not_found"]


def test_records_run_metadata(loaded: LoadedDataset) -> None:
    times = iter(
        [datetime(2026, 11, 18, 10, 0, tzinfo=UTC), datetime(2026, 11, 18, 10, 5, tzinfo=UTC)]
    )

    run = run_evaluation(
        loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS, now=lambda: next(times)
    )

    assert run.run_id == f"test_dataset__{EXTRACTION_PROMPT_VERSION}__20261118T100000Z"
    assert run.finished_at == datetime(2026, 11, 18, 10, 5, tzinfo=UTC)
    assert run.dataset.name == "test_dataset"
    assert run.dataset.sha256 == loaded.sha256
    assert run.dataset.case_count == 3
    assert run.extraction.prompt_version == EXTRACTION_PROMPT_VERSION
    assert run.model_settings == SETTINGS
    assert run.cache_enabled is False


def test_missing_model_settings_are_recorded_as_unknown(loaded: LoadedDataset) -> None:
    client = FakeLLMClient([output_for(case["excerpt"]) for case in CASES])

    run = run_evaluation(loaded, client, model_settings=None)

    assert run.model_settings is None


def test_records_token_usage_per_case(loaded: LoadedDataset) -> None:
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)

    prediction = run.predictions[0]
    assert (prediction.llm_calls, prediction.input_tokens, prediction.output_tokens) == (1, 100, 20)
    assert prediction.cache_hit is False


def test_failed_case_is_recorded_and_run_continues(loaded: LoadedDataset) -> None:
    source_b = CASES[1]["source_text"]
    client = ScriptedClient(
        all_outputs(), errors={source_b: LLMAPIError("HTTP 503: busy", status_code=503)}
    )

    run = run_evaluation(loaded, client, model_settings=SETTINGS)

    assert [p.status for p in run.predictions] == ["succeeded", "failed", "succeeded"]
    failed = run.predictions[1]
    assert failed.result is None
    assert failed.error is not None
    assert failed.error.model_dump() == {"type": "LLMAPIError", "message": "HTTP 503: busy"}
    assert failed.llm_calls == 1
    assert run.summary.model_dump() == {"total": 3, "succeeded": 2, "failed": 1}


def test_unexpected_error_is_recorded_as_failed_case(loaded: LoadedDataset) -> None:
    # The fake client runs out of outputs after the first case.
    client = FakeLLMClient([output_for(CASES[0]["excerpt"])])

    run = run_evaluation(loaded, client, model_settings=None)

    assert [p.status for p in run.predictions] == ["succeeded", "failed", "failed"]
    assert run.predictions[1].error is not None
    assert run.predictions[1].error.type == "RuntimeError"


def test_selected_cases_only(loaded: LoadedDataset) -> None:
    client = ScriptedClient(all_outputs())

    run = run_evaluation(loaded, client, model_settings=SETTINGS, case_ids=["case-c", "case-a"])

    assert [p.case_id for p in run.predictions] == ["case-a", "case-c"]
    assert run.dataset.case_count == 3
    assert client.calls == 2


def test_unknown_case_id_is_rejected_before_any_call(loaded: LoadedDataset) -> None:
    client = ScriptedClient(all_outputs())

    with pytest.raises(ValueError, match="Unknown case ids: case-x"):
        run_evaluation(loaded, client, model_settings=SETTINGS, case_ids=["case-a", "case-x"])

    assert client.calls == 0


def test_second_run_uses_cache(loaded: LoadedDataset, tmp_path: Path) -> None:
    client = ScriptedClient(all_outputs())
    cache = ExtractionCache(tmp_path / "cache", SETTINGS)

    first = run_evaluation(loaded, client, model_settings=SETTINGS, cache=cache)
    second = run_evaluation(loaded, client, model_settings=SETTINGS, cache=cache)

    assert client.calls == 3
    assert [p.cache_hit for p in first.predictions] == [False, False, False]
    assert [p.cache_hit for p in second.predictions] == [True, True, True]
    assert [p.llm_calls for p in second.predictions] == [0, 0, 0]
    assert second.cache_enabled is True
    assert [p.result for p in second.predictions] == [p.result for p in first.predictions]


def test_dataset_file_is_not_modified(loaded: LoadedDataset) -> None:
    before = loaded.path.read_bytes()

    run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)

    assert loaded.path.read_bytes() == before


# Storage


def test_saved_run_round_trips(loaded: LoadedDataset, tmp_path: Path) -> None:
    source_b = CASES[1]["source_text"]
    client = ScriptedClient(all_outputs(), errors={source_b: LLMAPIError("HTTP 503: busy")})
    run = run_evaluation(loaded, client, model_settings=SETTINGS)
    path = tmp_path / "runs" / "run.json"

    save_run(run, path)

    reloaded = load_run(path)
    assert reloaded == run
    assert EvaluationRun.model_validate(json.loads(path.read_text(encoding="utf-8"))) == run
    assert [p.suffix for p in path.parent.iterdir()] == [".json"]


def test_existing_run_is_not_overwritten(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)
    path = tmp_path / "run.json"
    path.write_text("earlier run", encoding="utf-8")

    with pytest.raises(RunFileError, match="already exists"):
        save_run(run, path)

    assert path.read_text(encoding="utf-8") == "earlier run"
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_overwrite_replaces_existing_run(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)
    path = tmp_path / "run.json"
    path.write_text("earlier run", encoding="utf-8")

    save_run(run, path, overwrite=True)

    assert load_run(path) == run


def test_missing_run_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(RunFileError, match="Run file not found"):
        load_run(tmp_path / "missing.json")


def test_invalid_run_file_is_reported(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)
    data = json.loads(run.model_dump_json())
    data["summary"]["failed"] = 5
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunFileError, match="summary does not match predictions"):
        load_run(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"status": "succeeded", "result": None},
    ],
    ids=["failed_with_result", "succeeded_without_result"],
)
def test_prediction_status_must_match_content(
    loaded: LoadedDataset, changes: dict[str, object]
) -> None:
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)
    data = run.predictions[0].model_dump()
    data.update(changes)

    with pytest.raises(ValidationError):
        type(run.predictions[0]).model_validate(data)


# Git metadata


def test_records_git_metadata_read_before_any_case(loaded: LoadedDataset) -> None:
    client = ScriptedClient(all_outputs())
    calls_at_read: list[int] = []

    def fake_git_info() -> GitInfo:
        calls_at_read.append(client.calls)
        return GitInfo(commit="a" * 40, dirty=True)

    run = run_evaluation(loaded, client, model_settings=SETTINGS, git_info=fake_git_info)

    assert (run.git_commit, run.git_dirty) == ("a" * 40, True)
    assert calls_at_read == [0]


def test_unavailable_git_metadata_is_null_and_run_succeeds(loaded: LoadedDataset) -> None:
    run = run_evaluation(
        loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS, git_info=GitInfo
    )

    assert (run.git_commit, run.git_dirty) == (None, None)
    assert run.summary.failed == 0


def test_git_metadata_round_trips(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = run_evaluation(
        loaded,
        ScriptedClient(all_outputs()),
        model_settings=SETTINGS,
        git_info=lambda: GitInfo(commit="b" * 40, dirty=False),
    )
    save_run(run, tmp_path / "run.json")

    reloaded = load_run(tmp_path / "run.json")

    assert (reloaded.git_commit, reloaded.git_dirty) == ("b" * 40, False)


def test_run_file_without_git_fields_is_still_readable(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    # Run files written before Git metadata was added have no git_* keys.
    run = run_evaluation(loaded, ScriptedClient(all_outputs()), model_settings=SETTINGS)
    data = json.loads(run.model_dump_json())
    del data["git_commit"], data["git_dirty"]
    path = tmp_path / "old_run.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    reloaded = load_run(path)

    assert (reloaded.git_commit, reloaded.git_dirty) == (None, None)
    assert reloaded.predictions == run.predictions
