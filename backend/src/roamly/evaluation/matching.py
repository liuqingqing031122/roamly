"""Offline one-to-one matching of expected claims to predicted claims.

This implements *detection matching* as defined in the annotation guidelines 1.1, §5:
it decides which predicted claim states the same source fact as which expected claim.
It does not judge whether the fields are correct: a matched pair may still have a wrong
type, scope, or statement, which strict extraction scoring and Level 2 evaluate later.

Rules (see `docs/extraction_annotation_guidelines.md` §5 and `docs/architecture.md` §10):

1. A pair is considered only if the two excerpts overlap in the source text, so both
   claims are supported by the same part of the source (C5).
2. Overlap alone is not enough, because one sentence can state several facts. The
   normalized statements must also be similar (token overlap, ignoring venue names).
3. Each candidate gets a level:
   - `strong`: same venue and similar statements; a type mismatch needs a higher
     statement similarity. Strong candidates can be confirmed automatically.
   - `review`: plausible but uncertain (different venue, or only moderate similarity).
     These are never confirmed automatically.
4. Strong candidates are confirmed best-first. If another open candidate for the same
   expected or predicted claim scores almost as well, none of them is confirmed and
   all are marked `needs_review`.
5. Claims that appear in a `needs_review` candidate are not reported as unmatched, so
   they are neither true nor false positives until a person decides.

The thresholds below are initial heuristics, not yet validated on large real-world data.
Any change to the rules or thresholds must bump `MATCHER_VERSION`.

Only accepted claims (`ExtractionResult.claims`) are matched; skipped candidates are
not part of claim detection.
"""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from roamly.domain import Claim
from roamly.evaluation.dataset import (
    EvalCase,
    ExpectedClaim,
    LoadedDataset,
    format_validation_error,
)
from roamly.evaluation.git_info import GitInfo, read_git_info
from roamly.evaluation.predictions import (
    CaseError,
    DatasetInfo,
    EvaluationRun,
    write_json_file,
)
from roamly.extraction import ExtractionResult

# Bump when the matching rules or thresholds change, so results stay comparable.
MATCHER_VERSION = "matching-v1"
MATCHING_FORMAT_VERSION = 1

# Statement similarity thresholds (Dice coefficient over content tokens, 0 to 1).
STRONG_SAME_TYPE = 0.5
STRONG_OTHER_TYPE = 0.8
REVIEW_SAME_TYPE = 0.25
REVIEW_OTHER_TYPE = 0.4
REVIEW_OTHER_PLACE = 0.5
# Added to the score of same-type candidates when ranking competing candidates.
TYPE_MATCH_BONUS = 0.1
# Competing candidates closer than this are ambiguous and left for review.
AMBIGUITY_MARGIN = 0.1

# Function words that carry no fact; content words such as "on" in "on Mondays" are
# kept because scope words help to tell facts apart.
_STOP_WORDS = frozenset({"a", "an", "the", "is", "are", "be", "it", "its", "at", "of", "to"})


class MatchingError(Exception):
    """The dataset and the run cannot be matched, or a matching file is unusable."""


class _MatchingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExcerptRelation = Literal["same", "contains", "overlap"]
ReviewReason = Literal[
    "different_place", "type_mismatch_moderate_similarity", "moderate_similarity", "competing"
]


class PairEvidence(_MatchingModel):
    """The signals used to compare one expected claim with one predicted claim."""

    place_match: bool
    type_match: bool
    excerpt_relation: ExcerptRelation
    statement_similarity: float


class ConfirmedMatch(_MatchingModel):
    # "gt-1", "gt-2", ...: the 1-based position in the case's expected_claims.
    expected_id: str
    # The predicted Claim.id, e.g. "claim-1".
    prediction_id: str
    evidence: PairEvidence


class ReviewCandidate(_MatchingModel):
    """A possible pair that needs a human decision before it counts either way."""

    expected_id: str
    prediction_id: str
    evidence: PairEvidence
    reason: ReviewReason


class CaseMatching(_MatchingModel):
    case_id: str
    status: Literal["matched", "failed"]
    # Copied from the run for failed cases, which are not matched.
    error: CaseError | None = None
    confirmed: list[ConfirmedMatch] = []
    needs_review: list[ReviewCandidate] = []
    unmatched_expected: list[str] = []
    unmatched_predictions: list[str] = []


class RunReference(_MatchingModel):
    run_id: str
    path: str
    # Hash of the run file bytes that were matched.
    sha256: str
    prompt_version: str
    git_commit: str | None


class MatchingResult(_MatchingModel):
    format_version: Literal[1] = MATCHING_FORMAT_VERSION
    matcher_version: str
    created_at: AwareDatetime
    dataset: DatasetInfo
    run: RunReference
    # Code that produced this matching (not the run); None when unavailable.
    git_commit: str | None
    git_dirty: bool | None
    cases: list[CaseMatching]


def expected_id(index: int) -> str:
    """Stable identifier for the expected claim at 0-based `index` within a case."""
    return f"gt-{index + 1}"


def match_run(
    loaded: LoadedDataset,
    run: EvaluationRun,
    *,
    run_path: Path,
    run_sha256: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    git_info: Callable[[], GitInfo] = read_git_info,
) -> MatchingResult:
    """Match every case in `run` against the dataset it was produced from.

    Raises `MatchingError` if the dataset file differs from the one used for the run,
    or if the run contains a case that is not in the dataset.
    """
    if loaded.sha256 != run.dataset.sha256:
        raise MatchingError(
            f"Dataset {loaded.path} (sha256 {loaded.sha256[:12]}) is not the dataset used "
            f"for run {run.run_id} (sha256 {run.dataset.sha256[:12]})"
        )
    cases = {case.id: case for case in loaded.dataset.cases}
    unknown = [p.case_id for p in run.predictions if p.case_id not in cases]
    if unknown:
        raise MatchingError(f"Run contains cases not in the dataset: {', '.join(unknown)}")

    matchings = []
    for prediction in run.predictions:
        if prediction.result is None:
            matchings.append(
                CaseMatching(case_id=prediction.case_id, status="failed", error=prediction.error)
            )
        else:
            matchings.append(match_case(cases[prediction.case_id], prediction.result))

    git = git_info()
    return MatchingResult(
        matcher_version=MATCHER_VERSION,
        created_at=now(),
        dataset=run.dataset,
        run=RunReference(
            run_id=run.run_id,
            path=str(run_path),
            sha256=run_sha256,
            prompt_version=run.extraction.prompt_version,
            git_commit=run.git_commit,
        ),
        git_commit=git.commit,
        git_dirty=git.dirty,
        cases=matchings,
    )


@dataclass(frozen=True)
class _Candidate:
    expected_id: str
    prediction_id: str
    evidence: PairEvidence
    level: Literal["strong", "review"]
    reason: ReviewReason | None
    score: float


def match_case(case: EvalCase, result: ExtractionResult) -> CaseMatching:
    """Match the accepted claims of one case to its expected claims."""
    candidates = []
    for index, expected in enumerate(case.expected_claims):
        for predicted in result.claims:
            candidate = _compare(case.source_text, expected_id(index), expected, predicted)
            if candidate is not None:
                candidates.append(candidate)

    confirmed: list[ConfirmedMatch] = []
    review: dict[tuple[str, str], ReviewCandidate] = {}
    settled: set[str] = set()  # expected and prediction ids that are confirmed or in review

    def is_open(candidate: _Candidate) -> bool:
        return candidate.expected_id not in settled and candidate.prediction_id not in settled

    def add_review(candidate: _Candidate, reason: ReviewReason) -> None:
        key = (candidate.expected_id, candidate.prediction_id)
        review.setdefault(
            key,
            ReviewCandidate(
                expected_id=candidate.expected_id,
                prediction_id=candidate.prediction_id,
                evidence=candidate.evidence,
                reason=reason,
            ),
        )

    strong = sorted(
        (c for c in candidates if c.level == "strong"),
        key=lambda c: (-c.score, c.expected_id, c.prediction_id),
    )
    for candidate in strong:
        if not is_open(candidate):
            continue
        rivals = [
            other
            for other in strong
            if other is not candidate
            and is_open(other)
            and (
                other.expected_id == candidate.expected_id
                or other.prediction_id == candidate.prediction_id
            )
            and candidate.score - other.score < AMBIGUITY_MARGIN
        ]
        if rivals:
            for item in (candidate, *rivals):
                add_review(item, "competing")
                settled.update((item.expected_id, item.prediction_id))
            continue
        confirmed.append(
            ConfirmedMatch(
                expected_id=candidate.expected_id,
                prediction_id=candidate.prediction_id,
                evidence=candidate.evidence,
            )
        )
        settled.update((candidate.expected_id, candidate.prediction_id))

    # Weak candidates matter only while neither claim has been confirmed elsewhere.
    confirmed_ids = {m.expected_id for m in confirmed} | {m.prediction_id for m in confirmed}
    for candidate in candidates:
        if candidate.level == "review" and not (
            {candidate.expected_id, candidate.prediction_id} & confirmed_ids
        ):
            assert candidate.reason is not None
            add_review(candidate, candidate.reason)

    in_review = {r.expected_id for r in review.values()} | {
        r.prediction_id for r in review.values()
    }
    expected_ids = [expected_id(index) for index in range(len(case.expected_claims))]
    return CaseMatching(
        case_id=case.id,
        status="matched",
        confirmed=sorted(confirmed, key=lambda m: _id_order(m.expected_id)),
        needs_review=sorted(
            review.values(), key=lambda r: (_id_order(r.expected_id), r.prediction_id)
        ),
        unmatched_expected=[
            i for i in expected_ids if i not in confirmed_ids and i not in in_review
        ],
        unmatched_predictions=[
            c.id for c in result.claims if c.id not in confirmed_ids and c.id not in in_review
        ],
    )


def _compare(
    source_text: str, expected_ref: str, expected: ExpectedClaim, predicted: Claim
) -> _Candidate | None:
    relation = _excerpt_relation(source_text, expected.original_excerpt, predicted.original_excerpt)
    if relation is None:
        return None

    place_match = _normalize_place(expected.place_name) == _normalize_place(predicted.place_name)
    type_match = expected.claim_type == predicted.claim_type
    place_tokens = _tokens(expected.place_name) | _tokens(predicted.place_name)
    similarity = _dice(
        _tokens(expected.normalized_statement) - place_tokens,
        _tokens(predicted.normalized_statement) - place_tokens,
    )
    evidence = PairEvidence(
        place_match=place_match,
        type_match=type_match,
        excerpt_relation=relation,
        statement_similarity=round(similarity, 3),
    )

    level: Literal["strong", "review"] | None = None
    reason: ReviewReason | None = None
    if not place_match:
        if similarity >= REVIEW_OTHER_PLACE:
            level, reason = "review", "different_place"
    elif type_match:
        if similarity >= STRONG_SAME_TYPE:
            level = "strong"
        elif similarity >= REVIEW_SAME_TYPE:
            level, reason = "review", "moderate_similarity"
    elif similarity >= STRONG_OTHER_TYPE:
        level = "strong"
    elif similarity >= REVIEW_OTHER_TYPE:
        level, reason = "review", "type_mismatch_moderate_similarity"

    if level is None:
        return None
    return _Candidate(
        expected_id=expected_ref,
        prediction_id=predicted.id,
        evidence=evidence,
        level=level,
        reason=reason,
        score=similarity + (TYPE_MATCH_BONUS if type_match else 0.0),
    )


def _excerpt_relation(source_text: str, first: str, second: str) -> ExcerptRelation | None:
    """How the source spans of two excerpts relate, considering every occurrence."""
    best: ExcerptRelation | None = None
    rank = {"overlap": 1, "contains": 2, "same": 3}
    for a_start, a_end in _spans(source_text, first):
        for b_start, b_end in _spans(source_text, second):
            if a_start >= b_end or b_start >= a_end:
                continue
            if (a_start, a_end) == (b_start, b_end):
                relation: ExcerptRelation = "same"
            elif (a_start <= b_start and b_end <= a_end) or (b_start <= a_start and a_end <= b_end):
                relation = "contains"
            else:
                relation = "overlap"
            if best is None or rank[relation] > rank[best]:
                best = relation
    return best


def _spans(text: str, part: str) -> list[tuple[int, int]]:
    if not part:
        return []
    spans = []
    start = text.find(part)
    while start != -1:
        spans.append((start, start + len(part)))
        start = text.find(part, start + 1)
    return spans


def _normalize_place(name: str) -> str:
    """Trim, case-fold, and drop a leading "the" (guidelines §5, Level 2)."""
    normalized = " ".join(name.split()).casefold()
    return normalized.removeprefix("the ")


def _tokens(text: str) -> set[str]:
    return {
        _stem(word) for word in re.findall(r"[a-z0-9]+", text.casefold()) if word not in _STOP_WORDS
    }


def _stem(word: str) -> str:
    """Very light suffix stripping so that "requires", "required", and "require" agree."""
    for _ in range(2):
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
    return word


def _dice(first: set[str], second: set[str]) -> float:
    if not first and not second:
        return 1.0
    return 2 * len(first & second) / (len(first) + len(second))


def _id_order(identifier: str) -> int:
    return int(identifier.removeprefix("gt-"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_matching(result: MatchingResult, path: Path, *, overwrite: bool = False) -> None:
    """Write a matching file atomically; refuses to replace one unless `overwrite`."""
    try:
        write_json_file(path, result.model_dump_json(indent=2), overwrite=overwrite)
    except FileExistsError as exc:
        raise MatchingError(f"Matching file already exists: {path}") from exc


def load_matching(path: Path) -> MatchingResult:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MatchingError(f"Matching file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise MatchingError(f"Cannot read matching file {path}: {exc}") from exc
    try:
        return MatchingResult.model_validate_json(content)
    except ValidationError as exc:
        raise MatchingError(
            f"Invalid matching file {path}:\n{format_validation_error(exc)}"
        ) from exc
