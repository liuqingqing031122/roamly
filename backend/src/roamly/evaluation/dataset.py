"""Loading and validation of labelled extraction evaluation datasets.

The dataset file is ground truth: it is only ever read, and the loaded models are
frozen so evaluation code cannot change it by accident. The format is described in
`docs/extraction_annotation_guidelines.md` §6.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from roamly.domain import ClaimType
from roamly.extraction import ExtractedClaimType, SkipReason


class DatasetError(Exception):
    """The dataset file cannot be read or does not match the expected format."""


class _GroundTruthModel(BaseModel):
    # Unknown fields are rejected so that typos in annotations are not silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedClaim(_GroundTruthModel):
    """A claim that post-processing should accept."""

    place_name: str
    claim_type: ClaimType
    original_excerpt: str
    normalized_statement: str
    scope: list[str]


class ExpectedSkipped(_GroundTruthModel):
    """A diagnostic annotation for a statement that must not become an accepted claim."""

    place_name: str | None
    claim_type: ExtractedClaimType
    original_excerpt: str
    reason: SkipReason


class EvalCase(_GroundTruthModel):
    id: str
    description: str
    tags: list[str]
    source_text: str
    expected_claims: list[ExpectedClaim]
    expected_skipped: list[ExpectedSkipped]
    must_not_extract: list[str]
    annotation_notes: str

    @model_validator(mode="after")
    def check_annotations_quote_the_source(self) -> Self:
        if not self.id.strip():
            raise ValueError("case id must not be empty")
        if not self.source_text.strip():
            raise ValueError(f"case {self.id}: source_text must not be empty")
        quoted = [
            *(claim.original_excerpt for claim in self.expected_claims),
            *(skipped.original_excerpt for skipped in self.expected_skipped),
            *self.must_not_extract,
        ]
        for passage in quoted:
            if not passage.strip() or passage not in self.source_text:
                raise ValueError(f"case {self.id}: {passage!r} is not an exact source substring")
        return self


class EvalDataset(_GroundTruthModel):
    dataset: str
    status: Literal["draft", "frozen"]
    guidelines: str
    notes: str
    cases: list[EvalCase]

    @model_validator(mode="after")
    def check_cases(self) -> Self:
        if not self.cases:
            raise ValueError("dataset has no cases")
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True)
class LoadedDataset:
    """A validated dataset with the file it came from."""

    path: Path
    # Hash of the exact file bytes, so a run can be tied to one dataset version.
    sha256: str
    dataset: EvalDataset


def load_dataset(path: Path) -> LoadedDataset:
    """Read and validate a dataset file. Raises `DatasetError` with a readable message."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset not found: {path}") from exc
    except OSError as exc:
        raise DatasetError(f"Cannot read dataset {path}: {exc.strerror}") from exc

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Dataset is not valid UTF-8 JSON: {path}: {exc}") from exc

    try:
        dataset = EvalDataset.model_validate(data)
    except ValidationError as exc:
        raise DatasetError(f"Invalid dataset {path}:\n{format_validation_error(exc)}") from exc

    return LoadedDataset(path=path, sha256=hashlib.sha256(raw).hexdigest(), dataset=dataset)


def format_validation_error(exc: ValidationError) -> str:
    """One line per problem, with the location in the file, e.g. `cases.3.scope: ...`."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(top level)"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)
