# Extraction Annotation Guidelines

**Version:** 1.1 (annotation rules unchanged from 1.0, which was frozen together with dataset v1; 1.1 revises only the evaluation definitions in §5 and adds decisions E1–E4)
**Applies to:** dataset `backend/tests/fixtures/extraction_eval_v1.json`; evaluated against extraction prompt `extraction-v2` and later versions
**Related:** [Requirements](requirements.md) · [Architecture](architecture.md) · [Roadmap](roadmap.md)

These guidelines define the ground truth for the M1.4 extraction evaluation. Annotators write expected outputs by following these rules, not by predicting what any model or prompt version will produce. Where the current prompt differs from these rules, the rules take precedence and the difference is recorded in §8.

**Changes in 1.1.** Version 1.0 required the same claim type for a predicted claim to match an expected claim. Version 1.1 separates *detection matching*, which pairs claims that state the same source fact even when the type or scope is wrong, from *strict extraction scoring*, which keeps the 1.0 requirement. The annotation rules in §1–§4 and §6 are unchanged, so ground truth written under 1.0, including dataset v1, remains valid.

## 1. What is annotated

Each case is a short travel-guide text. The annotator records:

- **Expected claims** — candidates that post-processing should accept as domain `Claim` objects.
- **Expected skipped claims** — diagnostic annotations for statements that must not become accepted claims: `observation` and `unsupported_type` (see §4).
- **Must-not-extract passages** — text the model should ignore entirely, such as opinions.

Skip reasons `excerpt_not_found`, `missing_place`, and `duplicate` describe model or input errors rather than desired behaviour, so they are not used as expected skips in this dataset version.

## 2. Claim identification

- A claim is a factual, practical statement about a specific venue that a traveller could check.
- One claim per checkable statement. If a sentence contains several claims, annotate each separately.
- Opinions, descriptions, and recommendations are not claims (e.g. "the view is stunning", "try the cake").

### Claim types (C3, Q2)

The distinction between `opening_hours` and `closure` is whether the statement describes the venue's **fixed, recurring schedule** or a **specific closure state**:

| Type | Covers | Examples |
| --- | --- | --- |
| `opening_hours` | The fixed, recurring operating and rest schedule: regular opening days and times, weekly closed days, public-holiday closures, and closures that recur every year | "open Tuesday to Sunday from 10am to 5pm", "closed on Mondays", "closed on public holidays", "closed every January" |
| `reservation` | Whether booking is required, recommended, or not needed, or whether walk-ins are accepted | "requires a reservation for dinner", "only accepts walk-ins" |
| `closure` | One-off temporary closures, specific closure events, and permanent closures | "closed for renovation until 30 April 2027", "closed for refurbishment from 1 to 21 August 2027", "has closed permanently" |
| `unsupported` | Any other factual, practical statement about a venue | prices, payment methods, dress codes, menu items |

## 3. Field rules

### `place_name`

- The venue name as written in the guide, without a leading article (e.g. "Harbour Tile Museum", not "the Harbour Tile Museum").
- If the claim refers to the venue indirectly ("it", "the museum"), use the name only when the guide makes clear which named venue is meant (for example, a preceding heading or sentence). Otherwise `null`.
- Never guess or invent a name.

### `claim_type`

One of the four types in §2.

### `assertion_kind`

- `policy`: a general or recurring rule.
- `observation`: a one-time personal experience, even if it mentions a day or time. Observations are never turned into general rules.

### `original_excerpt` (C5)

- Must be real, contiguous text from the source, copied character for character: an exact, case-sensitive substring of `source_text`.
- Prefer the shortest contiguous span that fully supports the claim, normally one sentence or clause. Ground-truth excerpts follow this preference.
- A longer excerpt is acceptable if it is an exact substring of the source and contains the complete information for the claim.
- Several claims from the same sentence may share an excerpt.

### `normalized_statement` (C1, C6)

- One standalone English sentence that faithfully preserves the complete factual meaning of the excerpt.
- It must keep all operating days and times, frequencies such as "daily" or "every day", and every important condition, exception, date, season, or qualifier stated in the source. Nothing may be omitted.
- It must not add information the source does not state. The only permitted addition is the venue name resolved from context.
- The reference statement in the dataset is one acceptable phrasing; evaluation judges meaning, not wording.

### `scope` (C1, C2, Q1, Q3)

`scope` is a **list of strings**. Each item is one explicitly stated temporal or applicability qualifier that limits when, or under which conditions, the claim applies.

- **What goes in:** days, dates, seasons, periods, deadlines, applicability conditions, and exceptions.
  - `opening_hours`: operating days that restrict the schedule (e.g. "on weekdays", "Tuesday to Sunday"). Clock times ("from 10am to 5pm") are the value of the claim and stay out of scope.
  - `closure`: the closure period or end date (e.g. "until 30 April 2027", "from 1 to 21 August 2027").
  - `reservation`: applicability conditions (e.g. "for dinner", "for groups of six or more", "on weekends").
- **What stays out (Q1):** phrases that state the regular frequency without restricting it, such as "daily" and "every day". They are not scope, but `normalized_statement` must still keep them.
- **Several qualifiers (Q3):** separately stated qualifiers that all apply together are separate items, e.g. "on weekends, only during summer" → `["on weekends", "only during summer"]`. Items in the list are combined with AND: the claim applies only when every item holds.
- **Coordinated qualifiers stay together (Q3):** qualifiers joined by "and" or "or" inside one phrase stay a single item, e.g. "closed on Mondays and public holidays" → `["on Mondays and public holidays"]`. Splitting them would turn "closed on Mondays or on public holidays" into the AND of both, which changes the meaning.
- `scope` is extracted even when the same information already appears in `normalized_statement`.
- Ground truth copies each item verbatim from the source, as the shortest phrase that expresses it.
- If the source states no qualifier, `scope` is an empty list `[]`, never `null` or a placeholder such as `"unspecified"`. Never infer dates, seasons, or conditions.

## 4. Expected skipped claims

- **Observations** are annotated with the `claim_type` they would have as a policy and reason `observation`.
- **Unsupported claims** are annotated with `claim_type: "unsupported"` and reason `unsupported_type`.
- For skipped claims, only `place_name`, `claim_type`, `original_excerpt`, and `reason` are annotated; `normalized_statement` and `scope` are not evaluated.

`expected_skipped` is **diagnostic information, not a required matching target** for the primary metrics:

- The model may return an observation or unsupported claim so that post-processing skips it, or it may ignore the statement entirely. Both behaviours are correct for claim detection.
- What counts is that the statement never becomes an accepted `Claim`. If it is accepted, it is a false positive.
- `expected_skipped` supports secondary diagnostics, such as how often the model labels observations and unsupported claims explicitly, and failure analysis.

## 5. Evaluation levels

Evaluation has two levels, reported separately. Level 1 asks whether claims were found; Level 2 asks whether the fields of found claims are correct. Only final accepted claims (`ExtractionResult.claims`) are evaluated; skipped candidates are neither true nor false positives (§4).

### Level 1 — Claim detection

Level 1 is reported in two forms that share one matching step.

#### Detection matching (E1)

A predicted claim **matches** an expected claim when both state the **same fact from the same part of the source**:

- Their excerpts must be supported by the same part of the source (C5). Excerpt overlap alone is not sufficient: one sentence can state several facts, so the normalized statements must also describe the same fact.
- A wrong `claim_type` or `scope` does **not** prevent a match. These are field errors, measured by strict extraction scoring and Level 2.
- A claim attributed to a different venue is not matched automatically; it is left for human review.
- Matching is one-to-one: each expected claim and each predicted claim takes part in at most one confirmed match.
- When a pair cannot be decided reliably (a different venue, only partial agreement, or several close candidates), it is marked **needs review**. Until a person decides, such claims count as neither true positives, false positives, nor false negatives. Human decisions are recorded in a separate review file; automatically generated matching files are never edited by hand (E3).
- **Merged claims (E4):** when one predicted claim covers several expected claims (for example, two schedules combined into one claim), it is marked needs review. Scoring must handle such cases explicitly and must never count one predicted claim as several true positives.

The procedure is implemented by a versioned, deterministic matcher (initially `matching-v1`; see the [architecture](architecture.md) §10). Its similarity thresholds are initial heuristics that have not yet been validated on large real-world data; any change to the matching rules or thresholds requires a new matcher version (E2).

#### Detection metrics

Precision, recall, and F1 are computed over the final accepted claims against `expected_claims`, using detection matches as true positives:

- An unmatched expected claim is a false negative; an unmatched accepted claim is a false positive.
- An accepted claim that corresponds to an expected skipped statement (an observation or unsupported claim) is a false positive.
- Any accepted claim from a must-not-extract passage counts as a false positive.

#### Strict extraction scoring

Strict scoring keeps the Level 1 matching rule of version 1.0: a predicted claim is **strictly correct** only if it is a detection match **and** it refers to the same venue (`place_name` equal under the Level 2 rule) **and** has the same `claim_type`. Strict precision, recall, and F1 use the same counting rules as the detection metrics, except that a detection match that is not strictly correct counts as one false negative (the expected claim) and one false positive (the predicted claim).

Strict scoring concerns only the fields that identify which claim was extracted. `scope`, `original_excerpt`, and `normalized_statement` are judged at Level 2 for every detection match.

### Level 2 — Claim quality

For each detection match, each field is judged separately:

| Field | Criterion |
| --- | --- |
| `place_name` | Equal after trimming, case-folding, and removing a leading "the" |
| `claim_type` | Exact match |
| `original_excerpt` | Exact substring of the source that contains the complete claim information (C5) |
| `scope` | Both empty, or the combined restriction expressed by all items has the same meaning; wording, item order, and item boundaries may differ, but no restriction may be added, lost, or changed (C4, Q3) |
| `normalized_statement` | Semantically correct: all days, times, frequencies, and conditions preserved, nothing added (C6; human or rubric judgement) |

**Detection success does not imply correctness.** Examples:

- If the source says "At Casa Lume, reservations are only required for groups of six or more" and the system returns "Casa Lume requires reservations" with `scope: []`, the claim is detected and may be strictly correct, but it fails `scope` and `normalized_statement` at Level 2, because the omitted condition reverses the meaning for most visitors.
- If the source says "It is closed on Mondays and public holidays" (expected `opening_hours`, C3) and the system returns the same statement as `closure`, the claim is detected but not strictly correct, and it fails `claim_type` at Level 2.

Examples of `scope` equivalence:

- `["in winter"]` and `["During winter"]` are equivalent (C4).
- `["on weekends during summer"]` and `["on weekends", "during summer"]` are equivalent: both express the same AND of two restrictions (Q3).
- `["on weekends"]` and `["on Saturdays"]` are not equivalent.
- `["on Mondays and public holidays"]` and `["on Mondays", "on public holidays"]` are not equivalent, because list items are combined with AND (Q3).

## 6. Dataset format

`extraction_eval_v1.json` contains a `cases` array. Each case has:

- `id`, `description`, `tags`, `source_text`;
- `expected_claims`: objects with `place_name`, `claim_type`, `original_excerpt`, `normalized_statement`, and `scope` (a list of strings), listed in source order;
- `expected_skipped`: objects with `place_name`, `claim_type`, `original_excerpt`, `reason`;
- `must_not_extract`: source passages that should produce no claim;
- `annotation_notes`: explanations for reviewers.

Claim IDs are not annotated because the system assigns them.

## 7. Decision record

| ID | Decision |
| --- | --- |
| C1 | `normalized_statement` keeps all operating days and times. Explicit days, dates, conditions, and exceptions are also extracted into `scope`, even when already present in the statement. |
| C2 | For `closure`, explicit closure periods and end dates are extracted into `scope`. |
| C3 | Fixed operating and rest schedules, including weekly closed days and public-holiday closures, are `opening_hours`. Temporary closures, specific closure events, and permanent closures are `closure`. |
| C4 | `scope` is evaluated by semantic equivalence; verbatim wording is not required, but the restriction must not change. |
| C5 | Prefer the shortest contiguous excerpt that supports the claim; longer exact substrings containing the complete claim are acceptable. Matching must consider place, type, and meaning, not excerpt overlap alone. |
| C6 | `normalized_statement` keeps every important condition and adds no facts. |
| Q1 | "daily" and "every day" state the regular frequency and are not scope; `normalized_statement` keeps them. |
| Q2 | Closures that recur every year (e.g. "closed every January") are `opening_hours`, like weekly and public-holiday rest days. One-off, specific, and permanent closures are `closure`. |
| Q3 | `scope` is `list[str]`, empty when there is no qualifier. Separately stated qualifiers are separate items combined with AND; qualifiers coordinated with "and" or "or" inside one phrase stay one item. |
| E1 | (1.1) Detection matching pairs claims that state the same fact from the same part of the source; a wrong `claim_type` or `scope` does not prevent a match. The 1.0 rule requiring the same venue and type is kept as strict extraction scoring. |
| E2 | (1.1) The matcher is deterministic and versioned; its thresholds are initial heuristics, and any rule or threshold change requires a new matcher version. |
| E3 | (1.1) Uncertain pairs are marked needs review and count as neither true nor false positives until decided; human decisions are stored in a separate review file, not in the generated matching file. |
| E4 | (1.1) Automatic matching stays one-to-one; a predicted claim covering several expected claims is marked needs review, and scoring must never count it as several true positives. |

## 8. Differences from prompt `extraction-v2`

`extraction-v2` changes only the `scope` output format to a list (Q3). The following differences remain and are expected to lower scores on some cases; they are not corrected in the ground truth.

- The prompt's `opening_hours` definition does not mention weekly closed days, public-holiday closures, or annual recurring closures, and its `closure` definition includes "seasonal" closures with the example "closed in January" (C3, Q2).
- The prompt does not state that operating days, closure periods, and reservation conditions must be extracted into `scope` (C1, C2), or that "daily" and "every day" are not scope (Q1).
- The prompt forbids adding facts but does not require `normalized_statement` to keep every condition (C6).
