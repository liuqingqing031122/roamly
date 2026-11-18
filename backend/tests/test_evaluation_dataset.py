import json
from pathlib import Path
from typing import Any

import pytest

from roamly.evaluation.dataset import DatasetError, load_dataset

FROZEN_DATASET = Path(__file__).parent / "fixtures" / "extraction_eval_v1.json"


def make_case(case_id: str = "case-1", **overrides: Any) -> dict[str, Any]:
    case = {
        "id": case_id,
        "description": "A simple opening-hours claim.",
        "tags": ["opening_hours"],
        "source_text": "City Museum is open daily. Tickets cost 10 euros.",
        "expected_claims": [
            {
                "place_name": "City Museum",
                "claim_type": "opening_hours",
                "original_excerpt": "City Museum is open daily.",
                "normalized_statement": "City Museum is open daily.",
                "scope": [],
            }
        ],
        "expected_skipped": [
            {
                "place_name": "City Museum",
                "claim_type": "unsupported",
                "original_excerpt": "Tickets cost 10 euros.",
                "reason": "unsupported_type",
            }
        ],
        "must_not_extract": [],
        "annotation_notes": "",
    }
    case.update(overrides)
    return case


def make_dataset(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "dataset": "test_dataset",
        "status": "draft",
        "guidelines": "docs/extraction_annotation_guidelines.md",
        "notes": "",
        "cases": cases if cases is not None else [make_case()],
    }


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_the_frozen_dataset_without_changing_it() -> None:
    before = FROZEN_DATASET.read_bytes()

    loaded = load_dataset(FROZEN_DATASET)

    assert FROZEN_DATASET.read_bytes() == before
    assert loaded.dataset.dataset == "extraction_eval_v1"
    assert loaded.dataset.status == "frozen"
    assert len(loaded.dataset.cases) == 11
    assert len(loaded.sha256) == 64


def test_loaded_ground_truth_is_read_only(tmp_path: Path) -> None:
    loaded = load_dataset(write_json(tmp_path / "d.json", make_dataset()))

    with pytest.raises(ValueError, match="frozen"):
        loaded.dataset.cases[0].expected_claims[0].place_name = "Other"  # type: ignore[misc]


def test_sha256_identifies_the_file_contents(tmp_path: Path) -> None:
    first = load_dataset(write_json(tmp_path / "a.json", make_dataset()))
    same = load_dataset(write_json(tmp_path / "b.json", make_dataset()))
    other = load_dataset(write_json(tmp_path / "c.json", make_dataset([make_case("case-2")])))

    assert first.sha256 == same.sha256
    assert first.sha256 != other.sha256


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Dataset not found"):
        load_dataset(tmp_path / "missing.json")


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe"])
def test_unreadable_json_is_reported(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "d.json"
    path.write_bytes(content)

    with pytest.raises(DatasetError, match="not valid UTF-8 JSON"):
        load_dataset(path)


def invalid(mutate: Any) -> dict[str, Any]:
    data = make_dataset()
    mutate(data)
    return data


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "Input should be a valid dictionary"),
        (invalid(lambda d: d.pop("cases")), "cases: Field required"),
        (invalid(lambda d: d.update(status="final")), "status: Input should be 'draft'"),
        (make_dataset([]), "dataset has no cases"),
        (make_dataset([make_case(), make_case()]), "duplicate case ids: case-1"),
        (
            invalid(lambda d: d["cases"][0].pop("source_text")),
            "cases.0.source_text: Field required",
        ),
        (
            invalid(lambda d: d["cases"][0]["expected_claims"][0].update(scope=None)),
            "cases.0.expected_claims.0.scope: Input should be a valid list",
        ),
        (
            invalid(lambda d: d["cases"][0]["expected_claims"][0].update(claim_type="price")),
            "cases.0.expected_claims.0.claim_type",
        ),
        (
            invalid(lambda d: d["cases"][0]["expected_claims"][0].update(scop=[])),
            "cases.0.expected_claims.0.scop: Extra inputs are not permitted",
        ),
        (
            invalid(lambda d: d["cases"][0].update(source_text="  ")),
            "source_text must not be empty",
        ),
        (
            invalid(
                lambda d: d["cases"][0]["expected_claims"][0].update(original_excerpt="Not there.")
            ),
            "'Not there.' is not an exact source substring",
        ),
        (
            invalid(lambda d: d["cases"][0].update(must_not_extract=["city museum"])),
            "'city museum' is not an exact source substring",
        ),
    ],
    ids=[
        "not_an_object",
        "missing_cases",
        "bad_status",
        "no_cases",
        "duplicate_ids",
        "missing_source_text",
        "null_scope",
        "bad_claim_type",
        "unknown_field",
        "blank_source_text",
        "excerpt_not_in_source",
        "must_not_extract_not_in_source",
    ],
)
def test_invalid_dataset_is_reported(tmp_path: Path, data: Any, message: str) -> None:
    path = write_json(tmp_path / "d.json", data)

    with pytest.raises(DatasetError, match="Invalid dataset") as exc:
        load_dataset(path)

    assert message in str(exc.value)
