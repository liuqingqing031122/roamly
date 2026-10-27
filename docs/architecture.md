# Roamly — Architecture

**Version:** 0.1 (draft)
**Scope:** V1 Travel Reality Checker MVP
**Related:** [Requirements](requirements.md) · [Roadmap](roadmap.md)

Each section is marked **Implemented** or **Planned**. Planned designs are intentions, not commitments, and are revised as milestones are delivered.

## 1. Overview and design principles

Roamly turns time-sensitive statements in travel guides into checkable claims, retrieves evidence, and reports transparent, appropriately uncertain verdicts.

- **Fixed workflow, not an autonomous agent.** The pipeline is a linear sequence with one human review step. This keeps behaviour predictable, testable, and cheap.
- **LLMs interpret language; deterministic code makes decisions.** LLM calls extract claims and assess individual evidence passages. Priority selection, citation validation, verdict aggregation, and date rules are plain Python and unit-tested.
- **Types describe structure, not truth.** Pydantic models guarantee field presence, types, and allowed values. They do not guarantee that an excerpt exists in the source or that a claim is semantically correct; those properties are enforced by post-processing steps (see §6).
- **Replaceable external dependencies.** LLM, search, place lookup, and URL fetching sit behind small interfaces so tests run on fakes and fixtures at zero API cost.
- **Honest output.** Missing or ambiguous evidence produces a qualified or "Unable to Verify" result, never invented facts or citations.

## 2. End-to-end workflow (Planned)

```
import (URL or text) → extract claims → resolve places → [user review] → retrieve evidence → verify → report
```

The user review step is a natural pause between HTTP requests; state is persisted between steps, so no workflow engine is required.

## 3. Technology stack

| Concern | Choice | Status | Rationale |
| --- | --- | --- | --- |
| Language / packaging | Python 3.12, uv | Implemented | Fast, reproducible environments with a lock file |
| Web framework | FastAPI | Implemented | Pydantic-native request/response schemas, built-in OpenAPI docs |
| Data models | Pydantic v2 | Implemented | One model definition for API schemas, LLM output validation, and domain objects |
| Testing | pytest | Implemented | Standard, simple fixtures and parametrisation |
| Lint / format | ruff | Implemented | Single fast tool for linting, import sorting, and formatting |
| LLM | DeepSeek initially, behind `LLMClient` | Planned (M1) | See §7 and decision D-006 |
| Persistence | SQLite | Planned (M2+) | Zero-ops; sufficient for traceability requirements |
| HTML fetch / extraction | httpx, trafilatura | Planned (M5) | Timeouts, streaming size limits, manual redirect handling for SSRF safety |
| Place lookup | OpenStreetMap Nominatim (fixtures first) | Planned (M2) | Free; paid alternatives remain an option |
| Web search | To be decided | Planned (M5) | Open question |
| Frontend | To be decided (Jinja2 + HTMX or React) | Planned (M5) | Open question; the API comes first |

## 4. Module layout

Current (**Implemented**):

```
backend/
  pyproject.toml, uv.lock, .python-version
  src/roamly/
    main.py        # FastAPI app; GET /health
    domain.py      # Core domain models
  tests/
    test_health.py
    test_domain.py
```

Planned additions, introduced only when a milestone needs them:

```
  src/roamly/
    extraction.py  # M1: LLM extraction schemas and deterministic post-processing
    llm/           # M1: LLMClient protocol, Fake/Cached clients, provider adapters
    places/        # M2: place resolution and ambiguity detection
    selection.py   # M2: deterministic claim preselection
    retrieval/     # M3: evidence providers, chunking, ranking
    verification/  # M4: evidence assessment, citation checks, verdict aggregation
    ingestion/     # M5: URL policy, safe fetching, content extraction
    api/           # M5: HTTP routers (thin orchestration only)
eval/              # M1+: labelled datasets and metric scripts
```

`domain.py` stays a single module until it grows enough to justify a package.

## 5. Domain model

### Implemented (`backend/src/roamly/domain.py`)

- **Claim:** `id`, `place_name`, `claim_type` (`opening_hours` | `reservation` | `closure`), `original_excerpt`, `normalized_statement`.
- **Evidence:** `id`, `claim_id`, `source_url` (`HttpUrl`), `source_title`, `content_excerpt`, `retrieved_at`.
- **VerificationResult:** `claim_id`, `verdict` (`Supported` | `Contradicted` | `Potentially Outdated` | `Unable to Verify`), `explanation`, `evidence_ids`. A validator requires at least one evidence ID for `Supported` and `Contradicted` (NFR-01).

### Planned for M1

- **Claim.scope:** `str | None = None`. Holds only temporal or applicability qualifiers explicitly stated in the source (e.g. "in summer", "except public holidays"). Dates are never inferred; parsing is deferred to verification (M4).
- **ExtractedClaim** (in `extraction.py`): the untrusted LLM output schema.
  - `place_name: str | None` — `None` when the place cannot be determined from the source or its explicit context. Place names must never be invented.
  - `claim_type`: the three supported types plus `unsupported`.
  - `assertion_kind`: `policy` (a general rule) or `observation` (a one-time experience, e.g. "I visited on Monday").
  - `original_excerpt`, `normalized_statement`, `scope`.
- **SkippedClaim:** an `ExtractedClaim` plus a `reason`: `unsupported_type`, `observation`, `missing_place`, `excerpt_not_found`, or `duplicate`.
- **ExtractionResult:** `claims: list[Claim]` and `skipped: list[SkippedClaim]`.

`ExtractedClaim` and `Claim` are deliberately separate (decision D-004). `Claim` is only constructed by extraction post-processing, so downstream code can rely on every `Claim` having a supported type, a stated place, a policy assertion, and an excerpt found in the source. IDs are assigned by the system, never taken from LLM output.

### Planned for later milestones

`Place` (with identity status and candidates), `TravelSource`, `Analysis` (a check session with optional destination and visit date), per-evidence assessments, and usage/cost records. Fields such as `published_at`, `source_type`, `limitations`, `verified_at`, and model/prompt version are added when the milestone that uses them begins.

## 6. LLM versus deterministic responsibilities (Planned)

| Step | LLM | Deterministic Python |
| --- | --- | --- |
| URL validation, fetching, content extraction | | ✓ |
| Claim and place extraction; policy vs observation | ✓ | Excerpt substring check, type filtering, missing-place check, de-duplication, ID assignment |
| Place resolution | Optional disambiguation | ✓ candidate lookup and ambiguity detection |
| Preselection of at most three claims | | ✓ closure > reservation > opening hours, then source order |
| Query construction, retrieval, ranking | | ✓ |
| Per-evidence stance assessment | ✓ | Citation check: quoted text must exist in the evidence |
| Verdict aggregation, date and holiday rules | | ✓ |
| Cost and latency recording, evaluation metrics | | ✓ |

A typical run with three claims needs about five LLM calls: one or two for extraction and up to three for verification.

## 7. LLM integration (Planned, M1)

- **Abstraction.** A small `LLMClient` protocol with one structured-output method: given a prompt and a Pydantic schema, return a validated instance plus token usage. Pipeline code depends only on this protocol and must contain no provider-specific behaviour.
- **Implementations.**
  - `FakeLLMClient` — returns predefined outputs; used by all unit and contract tests.
  - `CachedLLMClient` — wraps any client and caches responses on disk, keyed by model, prompt version, and input hash, so repeated development and evaluation runs cost nothing.
  - Provider adapter — **DeepSeek** is the initial provider for local development and the M1 extraction evaluation. It is not a permanent dependency; OpenAI or Anthropic may be used for deployment or provider comparison.
- **Configuration.** Provider, model, and API key come from environment variables (e.g. `ROAMLY_LLM_PROVIDER`, `ROAMLY_LLM_MODEL`, `ROAMLY_LLM_API_KEY`; final names are fixed in M1.3). API keys and `.env` files are never committed; `.gitignore` already excludes `.env` and `.env.*`.
- **Structured output.** DeepSeek's actual structured-output capabilities (JSON mode versus schema-constrained output) must be verified during M1.3 rather than assumed. Regardless of provider support, every response is validated against the Pydantic schema, and invalid responses are handled explicitly rather than silently accepted.
- **Prompt versioning.** Each prompt has a version constant recorded alongside results and used in cache keys.

## 8. Verification and verdict aggregation (Planned, M4)

1. The LLM receives one claim, the optional visit date, and numbered evidence passages wrapped in explicit untrusted-content delimiters. It has no tools and may only return, per passage, a stance (`supports`, `contradicts`, `suggests_change`, `irrelevant`), a verbatim quote, and a short rationale.
2. Citation check: each quote must be a substring of its evidence passage and reference an existing evidence ID; otherwise the assessment is discarded.
3. Deterministic aggregation maps valid assessments to one of the four verdicts. Unresolved places or no relevant evidence yield `Unable to Verify`. The detailed rule table is finalised in M4 with table-driven tests.
4. Date rules: if the visit date is a holiday or special date not covered by evidence, a limitation is added and no date-specific `Supported` verdict is given (AC-12). A `Supported` reservation verdict never implies availability (AC-09).

## 9. Security (Planned)

- **URL fetching (NFR-03):** HTTP(S) only; block private, loopback, and link-local addresses, including after redirects; cap response size and time. Access restrictions are never bypassed.
- **Prompt injection (NFR-04):** retrieved and user-supplied text is delimited as data, verification calls expose no tools, outputs are schema-validated, and citations are checked against stored evidence.
- **Secrets and data minimisation (NFR-08):** keys come from the environment only; full user content is not logged unnecessarily.

## 10. Testing and evaluation strategy

| Layer | Purpose | Runs in CI | Status |
| --- | --- | --- | --- |
| Unit | Deterministic logic and model validation | Yes | Implemented |
| Contract | Pipeline steps with `FakeLLMClient` and fixtures; covers acceptance criteria | Yes | Planned (M1) |
| Recorded | Replay cached real-model responses for prompt regression | Yes | Planned (M1.3) |
| Live | Real APIs and websites, marked `@pytest.mark.live` | No (manual) | Planned |
| Evaluation | Metrics on labelled datasets; reports numbers, not pass/fail | No (manual) | Planned (M1.4, M6) |

Acceptance criteria are implemented first with synthetic fixtures; live-web tests remain separate, as required by the requirements document.

## 11. Cost control

Fake clients by default in tests, an on-disk response cache, at most three verified claims per run, truncated evidence (top-k passages), recorded token usage per step, and manually triggered live tests and evaluations.

## 12. Decision log

| ID | Date | Decision | Rationale |
| --- | --- | --- | --- |
| D-001 | 2026-10-24 | Use a fixed Python workflow; do not adopt LangGraph for the MVP | The pipeline is linear and the human review pause maps onto HTTP request boundaries; revisit for V2 if loops or tool selection appear |
| D-002 | 2026-10-26 | Backend uses Python 3.12, uv, FastAPI, pytest with a `src/` layout under `backend/` | Simple, reproducible setup; leaves room for a separate frontend |
| D-003 | 2026-10-27 | Milestones M1–M6 map to requirements §11 items 1–6; M0 is the foundation stage. Domain models belong to M0 | Corrects earlier labelling of the domain-model work as "M1" |
| D-004 | 2026-10-27 | Separate untrusted `ExtractedClaim` (LLM output) from validated `Claim` | Keeps a clear trust boundary; downstream code relies on post-processing invariants; skip reasons support failure analysis |
| D-005 | 2026-10-27 | Keep `domain.py` as a single module for now | Few models; split into a package when size justifies it |
| D-006 | 2026-10-27 | DeepSeek is the initial LLM provider for development and the M1 evaluation, accessed only through `LLMClient` and configured via environment variables | Existing API credits; provider independence allows switching to OpenAI or Anthropic later; structured-output support to be verified in M1.3 |
| D-007 | 2026-10-27 | Add ruff for linting and formatting | Low-cost consistency from the start |

## 13. Open questions

- Web search provider and its usage terms.
- Frontend approach (server-rendered Jinja2 + HTMX versus React + TypeScript).
- Deployment target and the LLM provider used for deployment.
- Initial language and destination coverage.
- Whether reports persist across sessions in the first release.
- Exact verdict aggregation rules, including handling of unknown publication dates and conflicting sources.
