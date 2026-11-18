"""Runs the existing extraction pipeline over every case of an evaluation dataset."""

import hashlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from roamly.evaluation.dataset import EvalCase, LoadedDataset
from roamly.evaluation.git_info import GitInfo, read_git_info
from roamly.evaluation.predictions import (
    CaseError,
    CasePrediction,
    DatasetInfo,
    EvaluationRun,
    ExtractionInfo,
    make_run_id,
    summarize,
)
from roamly.extraction_cache import ExtractionCache
from roamly.extraction_prompt import EXTRACTION_PROMPT_VERSION, EXTRACTION_SYSTEM_PROMPT
from roamly.llm import LLMClient, StructuredResponse
from roamly.pipeline import extract_claims

logger = logging.getLogger(__name__)


@dataclass
class _UsageRecorder:
    """Wraps an `LLMClient` to count calls and tokens without changing the pipeline."""

    client: LLMClient
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        self.calls += 1
        response = self.client.complete_structured(
            system_prompt=system_prompt, user_prompt=user_prompt, response_model=response_model
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        return response


def run_evaluation(
    loaded: LoadedDataset,
    client: LLMClient,
    *,
    model_settings: Mapping[str, Any] | None,
    cache: ExtractionCache | None = None,
    case_ids: Sequence[str] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    git_info: Callable[[], GitInfo] = read_git_info,
) -> EvaluationRun:
    """Extract claims for each case (or the selected `case_ids`), in dataset order.

    A failing case is recorded with its error and the run continues. Raises
    `ValueError` if `case_ids` names a case that is not in the dataset.
    """
    cases = _select_cases(loaded.dataset.cases, case_ids)
    # Read before any work so files written by this run cannot affect the dirty state.
    git = git_info()
    started_at = now()

    predictions = []
    for index, case in enumerate(cases, start=1):
        prediction = _run_case(case, client, cache)
        predictions.append(prediction)
        if prediction.result is not None:
            logger.info(
                "Case %s (%d/%d): %d accepted, %d skipped",
                case.id,
                index,
                len(cases),
                len(prediction.result.claims),
                len(prediction.result.skipped),
            )
        else:
            assert prediction.error is not None
            logger.warning(
                "Case %s (%d/%d) failed: %s: %s",
                case.id,
                index,
                len(cases),
                prediction.error.type,
                prediction.error.message,
            )

    dataset = loaded.dataset
    return EvaluationRun(
        run_id=make_run_id(dataset.dataset, EXTRACTION_PROMPT_VERSION, started_at),
        started_at=started_at,
        finished_at=now(),
        dataset=DatasetInfo(
            name=dataset.dataset,
            path=str(loaded.path),
            sha256=loaded.sha256,
            status=dataset.status,
            case_count=len(dataset.cases),
        ),
        extraction=ExtractionInfo(
            prompt_version=EXTRACTION_PROMPT_VERSION,
            system_prompt_sha256=hashlib.sha256(EXTRACTION_SYSTEM_PROMPT.encode()).hexdigest(),
        ),
        model_settings=dict(model_settings) if model_settings is not None else None,
        cache_enabled=cache is not None,
        git_commit=git.commit,
        git_dirty=git.dirty,
        summary=summarize(predictions),
        predictions=predictions,
    )


def _select_cases(cases: list[EvalCase], case_ids: Sequence[str] | None) -> list[EvalCase]:
    if case_ids is None:
        return cases
    known = {case.id for case in cases}
    unknown = [case_id for case_id in case_ids if case_id not in known]
    if unknown:
        raise ValueError(f"Unknown case ids: {', '.join(unknown)}")
    wanted = set(case_ids)
    return [case for case in cases if case.id in wanted]


def _run_case(case: EvalCase, client: LLMClient, cache: ExtractionCache | None) -> CasePrediction:
    recorder = _UsageRecorder(client)
    start = time.perf_counter()
    result = None
    error = None
    try:
        result = extract_claims(case.source_text, recorder, cache)
    # One bad case (API error, unusable output, or an unexpected bug) must not stop the
    # run; the error is recorded so the case counts as failed rather than as empty output.
    except Exception as exc:
        error = CaseError(type=type(exc).__name__, message=str(exc))
    duration = time.perf_counter() - start

    return CasePrediction(
        case_id=case.id,
        status="succeeded" if error is None else "failed",
        result=result,
        error=error,
        llm_calls=recorder.calls,
        cache_hit=cache is not None and error is None and recorder.calls == 0,
        input_tokens=recorder.input_tokens,
        output_tokens=recorder.output_tokens,
        duration_seconds=round(duration, 3),
    )
