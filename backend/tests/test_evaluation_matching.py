import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from roamly.domain import Claim
from roamly.evaluation.cli import main
from roamly.evaluation.dataset import EvalCase, LoadedDataset, load_dataset
from roamly.evaluation.git_info import GitInfo
from roamly.evaluation.matching import (
    MATCHER_VERSION,
    CaseMatching,
    MatchingError,
    MatchingResult,
    file_sha256,
    load_matching,
    match_case,
    match_run,
    save_matching,
)
from roamly.evaluation.predictions import EvaluationRun, load_run, save_run
from roamly.evaluation.runner import run_evaluation
from roamly.extraction import ExtractedClaim, ExtractedClaims, ExtractionResult
from roamly.llm import LLMAPIError, StructuredResponse, TokenUsage

# Building blocks


def expected(
    excerpt: str,
    statement: str,
    *,
    place: str = "Sala Norte",
    claim_type: str = "reservation",
    scope: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "place_name": place,
        "claim_type": claim_type,
        "original_excerpt": excerpt,
        "normalized_statement": statement,
        "scope": scope or [],
    }


def make_case(source: str, *claims: dict[str, Any], case_id: str = "case-1") -> EvalCase:
    return EvalCase(
        id=case_id,
        description="",
        tags=[],
        source_text=source,
        expected_claims=list(claims),
        expected_skipped=[],
        must_not_extract=[],
        annotation_notes="",
    )


def predicted(
    claim_id: str,
    excerpt: str,
    statement: str,
    *,
    place: str = "Sala Norte",
    claim_type: str = "reservation",
    scope: list[str] | None = None,
) -> Claim:
    return Claim(
        id=claim_id,
        place_name=place,
        claim_type=claim_type,
        original_excerpt=excerpt,
        normalized_statement=statement,
        scope=scope or [],
    )


def result(*claims: Claim) -> ExtractionResult:
    return ExtractionResult(claims=list(claims), skipped=[])


def pairs(matching: CaseMatching) -> list[tuple[str, str]]:
    return [(m.expected_id, m.prediction_id) for m in matching.confirmed]


def review_pairs(matching: CaseMatching) -> list[tuple[str, str, str]]:
    return [(r.expected_id, r.prediction_id, r.reason) for r in matching.needs_review]


RESERVATION = "Sala Norte requires a reservation for dinner."
RESERVATION_STATEMENT = "Sala Norte requires a reservation for dinner."
TWO_SCHEDULES = (
    "Mercado Norte is open Monday to Friday from 8am to 2pm and on Saturday from 9am to 1pm."
)
UPSTAIRS = "Its upstairs restaurant, Sala Norte, requires a reservation for dinner."


# Case-level matching


def test_one_to_one_match() -> None:
    case = make_case(RESERVATION, expected(RESERVATION, RESERVATION_STATEMENT))

    matching = match_case(case, result(predicted("claim-1", RESERVATION, RESERVATION_STATEMENT)))

    assert pairs(matching) == [("gt-1", "claim-1")]
    assert matching.needs_review == []
    assert matching.unmatched_expected == []
    assert matching.unmatched_predictions == []
    evidence = matching.confirmed[0].evidence
    assert (evidence.place_match, evidence.type_match, evidence.excerpt_relation) == (
        True,
        True,
        "same",
    )
    assert evidence.statement_similarity == 1.0


def test_claims_in_a_different_order() -> None:
    source = "Mercado Norte is open Monday to Saturday from 8am to 2pm. " + RESERVATION
    hours = "Mercado Norte is open Monday to Saturday from 8am to 2pm."
    case = make_case(
        source,
        expected(hours, hours, place="Mercado Norte", claim_type="opening_hours"),
        expected(RESERVATION, RESERVATION_STATEMENT),
    )

    matching = match_case(
        case,
        result(
            predicted("claim-1", RESERVATION, RESERVATION_STATEMENT),
            predicted("claim-2", hours, hours, place="Mercado Norte", claim_type="opening_hours"),
        ),
    )

    assert pairs(matching) == [("gt-1", "claim-2"), ("gt-2", "claim-1")]


def test_paraphrased_statement_and_longer_excerpt_match() -> None:
    source = "Its upstairs restaurant, Sala Norte, requires a reservation for dinner."
    case = make_case(
        source,
        expected("Sala Norte, requires a reservation for dinner.", RESERVATION_STATEMENT),
    )

    matching = match_case(
        case,
        result(predicted("claim-1", source, "Reservations are required at Sala Norte for dinner.")),
    )

    assert pairs(matching) == [("gt-1", "claim-1")]
    assert matching.confirmed[0].evidence.excerpt_relation == "contains"


@pytest.mark.parametrize(
    ("expected_claims", "predicted_claims", "unmatched_expected", "unmatched_predicted"),
    [
        ([], [], [], []),
        ([expected(RESERVATION, RESERVATION_STATEMENT)], [], ["gt-1"], []),
        ([], [predicted("claim-1", RESERVATION, RESERVATION_STATEMENT)], [], ["claim-1"]),
    ],
    ids=["both_empty", "no_predictions", "no_expected_claims"],
)
def test_empty_sides(
    expected_claims: list[dict[str, Any]],
    predicted_claims: list[Claim],
    unmatched_expected: list[str],
    unmatched_predicted: list[str],
) -> None:
    matching = match_case(make_case(RESERVATION, *expected_claims), result(*predicted_claims))

    assert matching.confirmed == []
    assert matching.unmatched_expected == unmatched_expected
    assert matching.unmatched_predictions == unmatched_predicted


def test_scope_difference_does_not_prevent_a_match() -> None:
    source = "City Bakery opens every day from 7am to 3pm."
    case = make_case(
        source,
        expected(source, source, place="City Bakery", claim_type="opening_hours", scope=[]),
    )

    matching = match_case(
        case,
        result(
            predicted(
                "claim-1",
                source,
                source,
                place="City Bakery",
                claim_type="opening_hours",
                scope=["every day"],
            )
        ),
    )

    assert pairs(matching) == [("gt-1", "claim-1")]


def test_type_difference_for_the_same_fact_is_matched_and_flagged() -> None:
    source = "Harbour Hall is closed on Mondays and public holidays."
    case = make_case(
        source, expected(source, source, place="Harbour Hall", claim_type="opening_hours")
    )

    matching = match_case(
        case,
        result(predicted("claim-1", source, source, place="Harbour Hall", claim_type="closure")),
    )

    assert pairs(matching) == [("gt-1", "claim-1")]
    assert matching.confirmed[0].evidence.type_match is False


def test_type_difference_with_only_partial_statement_overlap_needs_review() -> None:
    source = "Harbour Hall is closed on Mondays and public holidays."
    case = make_case(
        source, expected(source, source, place="Harbour Hall", claim_type="opening_hours")
    )

    matching = match_case(
        case,
        result(
            predicted(
                "claim-1",
                source,
                "Harbour Hall is closed on Mondays.",
                place="Harbour Hall",
                claim_type="closure",
            )
        ),
    )

    assert matching.confirmed == []
    assert review_pairs(matching) == [("gt-1", "claim-1", "type_mismatch_moderate_similarity")]
    assert matching.unmatched_expected == []
    assert matching.unmatched_predictions == []


def test_wrong_place_needs_review_and_is_not_unmatched() -> None:
    source = "Mercado Norte is open daily. " + UPSTAIRS
    excerpt = UPSTAIRS
    case = make_case(source, expected(excerpt, RESERVATION_STATEMENT))

    matching = match_case(
        case,
        result(
            predicted(
                "claim-1",
                excerpt,
                "Mercado Norte requires a reservation for dinner.",
                place="Mercado Norte",
            )
        ),
    )

    assert matching.confirmed == []
    assert review_pairs(matching) == [("gt-1", "claim-1", "different_place")]
    assert matching.unmatched_expected == []
    assert matching.unmatched_predictions == []


def test_shared_excerpt_with_a_different_fact_is_not_matched() -> None:
    source = "Sala Norte requires a reservation for dinner and is closed on Mondays."
    case = make_case(
        source,
        expected(source, RESERVATION_STATEMENT),
        expected(source, "Sala Norte is closed on Mondays.", claim_type="opening_hours"),
    )

    matching = match_case(case, result(predicted("claim-1", source, RESERVATION_STATEMENT)))

    assert pairs(matching) == [("gt-1", "claim-1")]
    assert matching.unmatched_expected == ["gt-2"]


def test_shared_excerpt_with_same_type_is_separated_by_statement() -> None:
    source = "Sala Norte opens at 7pm on weekdays and at noon on Sundays."
    weekdays = "Sala Norte opens at 7pm on weekdays."
    sundays = "Sala Norte opens at noon on Sundays."
    case = make_case(
        source,
        expected(source, weekdays, claim_type="opening_hours", scope=["on weekdays"]),
        expected(source, sundays, claim_type="opening_hours", scope=["on Sundays"]),
    )

    matching = match_case(
        case,
        result(
            predicted("claim-1", source, sundays, claim_type="opening_hours"),
            predicted("claim-2", source, weekdays, claim_type="opening_hours"),
        ),
    )

    assert pairs(matching) == [("gt-1", "claim-2"), ("gt-2", "claim-1")]


def test_unrelated_excerpts_are_not_matched_even_with_same_place_and_type() -> None:
    source = RESERVATION + " Sala Norte requires a deposit for groups."
    case = make_case(source, expected(RESERVATION, RESERVATION_STATEMENT))

    matching = match_case(
        case,
        result(
            predicted(
                "claim-1",
                "Sala Norte requires a deposit for groups.",
                "Sala Norte requires a reservation for groups.",
            )
        ),
    )

    assert matching.confirmed == []
    assert matching.needs_review == []
    assert matching.unmatched_expected == ["gt-1"]
    assert matching.unmatched_predictions == ["claim-1"]


def test_one_prediction_covering_two_expected_claims_needs_review() -> None:
    source = (
        "Mercado Norte is open Monday to Friday from 8am to 2pm and on Saturday from 9am to 1pm."
    )
    case = make_case(
        source,
        expected(
            source,
            "Mercado Norte is open Monday to Friday from 8am to 2pm.",
            place="Mercado Norte",
            claim_type="opening_hours",
        ),
        expected(
            source,
            "Mercado Norte is open on Saturday from 9am to 1pm.",
            place="Mercado Norte",
            claim_type="opening_hours",
        ),
    )

    matching = match_case(
        case,
        result(
            predicted("claim-1", source, source, place="Mercado Norte", claim_type="opening_hours")
        ),
    )

    assert matching.confirmed == []
    assert review_pairs(matching) == [
        ("gt-1", "claim-1", "competing"),
        ("gt-2", "claim-1", "competing"),
    ]
    assert matching.unmatched_expected == []
    assert matching.unmatched_predictions == []


def test_two_near_identical_predictions_for_one_claim_need_review() -> None:
    case = make_case(RESERVATION, expected(RESERVATION, RESERVATION_STATEMENT))

    matching = match_case(
        case,
        result(
            predicted("claim-1", RESERVATION, RESERVATION_STATEMENT),
            predicted("claim-2", RESERVATION, "Sala Norte requires reservations for dinner."),
        ),
    )

    assert matching.confirmed == []
    assert [r.prediction_id for r in matching.needs_review] == ["claim-1", "claim-2"]
    assert matching.unmatched_predictions == []


def test_clearly_better_prediction_wins_and_the_other_is_unmatched() -> None:
    case = make_case(RESERVATION, expected(RESERVATION, RESERVATION_STATEMENT))

    matching = match_case(
        case,
        result(
            predicted("claim-1", RESERVATION, "Sala Norte requires reservations."),
            predicted("claim-2", RESERVATION, RESERVATION_STATEMENT),
        ),
    )

    assert pairs(matching) == [("gt-1", "claim-2")]
    assert matching.needs_review == []
    assert matching.unmatched_predictions == ["claim-1"]


def test_every_claim_is_in_exactly_one_outcome() -> None:
    source = TWO_SCHEDULES + " " + RESERVATION
    case = make_case(
        source,
        expected(RESERVATION, RESERVATION_STATEMENT),
        expected(
            TWO_SCHEDULES,
            "Mercado Norte is open Monday to Friday from 8am to 2pm.",
            place="Mercado Norte",
            claim_type="opening_hours",
        ),
    )
    claims = [
        predicted("claim-1", RESERVATION, RESERVATION_STATEMENT),
        predicted("claim-2", RESERVATION, "Sala Norte is closed on Sundays.", claim_type="closure"),
    ]

    matching = match_case(case, result(*claims))

    confirmed_expected = [m.expected_id for m in matching.confirmed]
    review_expected = [r.expected_id for r in matching.needs_review]
    assert sorted(confirmed_expected + review_expected + matching.unmatched_expected) == [
        "gt-1",
        "gt-2",
    ]
    confirmed_predicted = [m.prediction_id for m in matching.confirmed]
    assert len(set(confirmed_predicted)) == len(confirmed_predicted)
    assert sorted(
        confirmed_predicted
        + [r.prediction_id for r in matching.needs_review]
        + matching.unmatched_predictions
    ) == ["claim-1", "claim-2"]


# Run-level matching


SOURCES = {
    "case-a": RESERVATION,
    "case-b": "Harbour Hall is closed until May 2027.",
    "case-c": "City Bakery opens every day from 7am to 3pm.",
}


def write_dataset(path: Path) -> Path:
    cases = [
        make_case(
            source,
            expected(source, source, place="Venue", claim_type="opening_hours"),
            case_id=case_id,
        ).model_dump()
        for case_id, source in SOURCES.items()
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


class EchoClient:
    """Returns one claim that copies the source; fails for case-b."""

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        if SOURCES["case-b"] in user_prompt:
            raise LLMAPIError("HTTP 503: busy")
        source = next(text for text in SOURCES.values() if text in user_prompt)
        output = ExtractedClaims(
            claims=[
                ExtractedClaim(
                    place_name="Venue",
                    claim_type="opening_hours",
                    assertion_kind="policy",
                    original_excerpt=source,
                    normalized_statement=source,
                )
            ]
        )
        assert isinstance(output, response_model)
        return StructuredResponse(output=output, usage=TokenUsage())


@pytest.fixture
def loaded(tmp_path: Path) -> LoadedDataset:
    return load_dataset(write_dataset(tmp_path / "dataset.json"))


def make_run(loaded: LoadedDataset, case_ids: list[str] | None = None) -> EvaluationRun:
    return run_evaluation(
        loaded,
        EchoClient(),
        model_settings=None,
        case_ids=case_ids,
        git_info=lambda: GitInfo(commit="c" * 40, dirty=False),
    )


def run_match(loaded: LoadedDataset, run: EvaluationRun, tmp_path: Path) -> MatchingResult:
    run_path = tmp_path / "run.json"
    save_run(run, run_path, overwrite=True)
    return match_run(
        loaded,
        run,
        run_path=run_path,
        run_sha256=file_sha256(run_path),
        now=lambda: datetime(2026, 11, 19, tzinfo=UTC),
        git_info=GitInfo,
    )


def test_run_matching_keeps_failed_cases_and_records_sources(
    loaded: LoadedDataset, tmp_path: Path
) -> None:
    run = make_run(loaded)

    matching = run_match(loaded, run, tmp_path)

    assert [(c.case_id, c.status) for c in matching.cases] == [
        ("case-a", "matched"),
        ("case-b", "failed"),
        ("case-c", "matched"),
    ]
    failed = matching.cases[1]
    assert failed.error is not None and failed.error.type == "LLMAPIError"
    assert failed.confirmed == [] and failed.unmatched_expected == []
    assert pairs(matching.cases[0]) == [("gt-1", "claim-1")]
    assert matching.matcher_version == MATCHER_VERSION
    assert matching.dataset.sha256 == loaded.sha256
    assert matching.run.run_id == run.run_id
    assert matching.run.sha256 == file_sha256(tmp_path / "run.json")
    assert matching.run.git_commit == "c" * 40
    assert (matching.git_commit, matching.git_dirty) == (None, None)


def test_partial_run_matches_only_the_cases_that_ran(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = make_run(loaded, case_ids=["case-c"])

    matching = run_match(loaded, run, tmp_path)

    assert [c.case_id for c in matching.cases] == ["case-c"]


def test_dataset_hash_mismatch_is_rejected(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = make_run(loaded)
    other = tmp_path / "other.json"
    data = json.loads(loaded.path.read_text(encoding="utf-8"))
    data["notes"] = "edited"
    other.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(MatchingError, match="is not the dataset used for run"):
        run_match(load_dataset(other), run, tmp_path)


def test_run_case_missing_from_dataset_is_rejected(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = make_run(loaded)
    renamed = run.model_copy(
        update={
            "predictions": [
                run.predictions[0].model_copy(update={"case_id": "case-x"}),
                *run.predictions[1:],
            ]
        }
    )

    with pytest.raises(MatchingError, match="not in the dataset: case-x"):
        run_match(loaded, renamed, tmp_path)


def test_matching_round_trips_and_is_not_overwritten(loaded: LoadedDataset, tmp_path: Path) -> None:
    matching = run_match(loaded, make_run(loaded), tmp_path)
    path = tmp_path / "matching.json"

    save_matching(matching, path)

    assert load_matching(path) == matching
    with pytest.raises(MatchingError, match="already exists"):
        save_matching(matching, path)
    save_matching(matching, path, overwrite=True)


def test_invalid_matching_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "matching.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(MatchingError, match="Invalid matching file"):
        load_matching(path)


def test_inputs_are_not_modified(loaded: LoadedDataset, tmp_path: Path) -> None:
    run = make_run(loaded)
    run_path = tmp_path / "run.json"
    save_run(run, run_path)
    before = (loaded.path.read_bytes(), run_path.read_bytes())

    match_run(loaded, run, run_path=run_path, run_sha256=file_sha256(run_path), git_info=GitInfo)

    assert (loaded.path.read_bytes(), run_path.read_bytes()) == before


# CLI


def test_cli_match_saves_next_to_the_run(
    loaded: LoadedDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_path = tmp_path / "runs" / "my_run.json"
    save_run(make_run(loaded), run_path)

    exit_code = main(["match", str(run_path), "--dataset", str(loaded.path)])

    out = capsys.readouterr().out
    expected_path = tmp_path / "runs" / f"my_run__{MATCHER_VERSION}.json"
    assert exit_code == 0
    assert load_matching(expected_path).run.path == str(run_path)
    assert "case-a  confirmed 1, review 0" in out
    assert "case-b  failed run, not matched" in out


def test_cli_match_rejects_a_different_dataset(
    loaded: LoadedDataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_path = tmp_path / "run.json"
    save_run(make_run(loaded), run_path)
    frozen = Path(__file__).parent / "fixtures" / "extraction_eval_v1.json"

    exit_code = main(["match", str(run_path), "--dataset", str(frozen)])

    assert exit_code == 1
    assert "is not the dataset used for run" in capsys.readouterr().err
    assert list(tmp_path.glob("*matching*")) == []


def test_cli_run_file_is_reloadable_after_matching(loaded: LoadedDataset, tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run = make_run(loaded)
    save_run(run, run_path)

    main(["match", str(run_path), "--dataset", str(loaded.path)])

    assert load_run(run_path) == run
