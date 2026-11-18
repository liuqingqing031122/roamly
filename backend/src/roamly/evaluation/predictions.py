"""Stored format of an evaluation run, and saving and loading it.

A run file holds everything a scoring step needs without calling an LLM again: the
full `ExtractionResult` (accepted and skipped claims) for every case, keyed by case
ID, plus metadata identifying the dataset, prompt, and model settings. Ground truth is
not copied; scoring reads it from the dataset file, identified by its SHA-256.

Run files never contain API keys: model settings come from the client's `settings`,
which exclude credentials.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError, model_validator

from roamly.evaluation.dataset import format_validation_error
from roamly.extraction import ExtractionResult

# Bump when the layout of run files changes.
RUN_FORMAT_VERSION = 1


class RunFileError(Exception):
    """A run file cannot be read or written."""


class _RunModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetInfo(_RunModel):
    name: str
    path: str
    sha256: str
    status: str
    case_count: int


class ExtractionInfo(_RunModel):
    prompt_version: str
    system_prompt_sha256: str


class CaseError(_RunModel):
    type: str
    message: str


class CasePrediction(_RunModel):
    """The outcome of running extraction on one dataset case."""

    case_id: str
    status: Literal["succeeded", "failed"]
    # Set exactly when status is "succeeded".
    result: ExtractionResult | None = None
    # Set exactly when status is "failed".
    error: CaseError | None = None
    # LLM calls attempted for this case; 0 when the output came from the cache.
    llm_calls: int
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    duration_seconds: float

    @model_validator(mode="after")
    def check_status_matches_content(self) -> Self:
        if self.status == "succeeded" and (self.result is None or self.error is not None):
            raise ValueError(f"case {self.case_id}: a succeeded case needs a result and no error")
        if self.status == "failed" and (self.error is None or self.result is not None):
            raise ValueError(f"case {self.case_id}: a failed case needs an error and no result")
        return self


class RunSummary(_RunModel):
    total: int
    succeeded: int
    failed: int


class EvaluationRun(_RunModel):
    format_version: Literal[1] = RUN_FORMAT_VERSION
    run_id: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    dataset: DatasetInfo
    extraction: ExtractionInfo
    # The client's non-secret request settings; None when the client does not expose them.
    model_settings: dict[str, Any] | None
    cache_enabled: bool
    # HEAD commit and uncommitted-change state when the run started; None when Git
    # metadata was unavailable. Defaults keep run files written before these fields readable.
    git_commit: str | None = None
    git_dirty: bool | None = None
    summary: RunSummary
    predictions: list[CasePrediction]

    @model_validator(mode="after")
    def check_predictions(self) -> Self:
        ids = [prediction.case_id for prediction in self.predictions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case ids in predictions")
        if self.summary != summarize(self.predictions):
            raise ValueError("summary does not match predictions")
        return self


def summarize(predictions: list[CasePrediction]) -> RunSummary:
    succeeded = sum(prediction.status == "succeeded" for prediction in predictions)
    return RunSummary(
        total=len(predictions), succeeded=succeeded, failed=len(predictions) - succeeded
    )


def make_run_id(dataset_name: str, prompt_version: str, started_at: datetime) -> str:
    """A readable, sortable identifier that is also used as the file name."""
    return f"{dataset_name}__{prompt_version}__{started_at.strftime('%Y%m%dT%H%M%SZ')}"


def save_run(run: EvaluationRun, path: Path, *, overwrite: bool = False) -> None:
    """Write a run file atomically. Refuses to replace an existing file unless `overwrite`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(run.model_dump_json(indent=2))
            file.write("\n")
        if overwrite:
            os.replace(tmp_name, path)
        else:
            # A hard link fails if the target exists, so no earlier run is ever replaced.
            os.link(tmp_name, path)
    except FileExistsError as exc:
        raise RunFileError(f"Run file already exists: {path}") from exc
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_run(path: Path) -> EvaluationRun:
    """Read and validate a saved run. Never calls an LLM."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RunFileError(f"Run file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RunFileError(f"Cannot read run file {path}: {exc}") from exc
    try:
        return EvaluationRun.model_validate_json(content)
    except ValidationError as exc:
        raise RunFileError(f"Invalid run file {path}:\n{format_validation_error(exc)}") from exc
