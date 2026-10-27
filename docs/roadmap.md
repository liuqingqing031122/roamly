# Roamly — Roadmap

**Related:** [Requirements](requirements.md) · [Architecture](architecture.md)

Milestones M1–M6 correspond to items 1–6 of the requirements delivery plan (§11). M0 is an additional foundation stage. Estimates assume roughly 6–8 hours of development per week and are revised as work progresses.

| Milestone | Goal | Status | Estimate |
| --- | --- | --- | --- |
| M0 Foundation | Project setup, core domain models, documentation, tooling | In progress | ~1 week remaining |
| M1 Extraction prototype | Pasted text → structured, source-linked claims | Not started | ~3 weeks |
| M2 Place resolution and review | Venue confirmation, priority selection, unsupported claim handling | Not started | Estimated on entry |
| M3 Evidence retrieval | Source metadata and relevant excerpts, initially from fixtures | Not started | Estimated on entry |
| M4 Verification | Four verdicts with traceable reasoning and abstention | Not started | Estimated on entry |
| M5 Integration | Safe URL import, web UI, reports, error handling | Not started | Estimated on entry |
| M6 Evaluation and release | Benchmark, failure analysis, deployment, README update | Not started | Estimated on entry |

## M0 — Foundation

**Goal:** a runnable, tested backend skeleton with core domain models and agreed documentation.

**Tasks**

- [x] Python 3.12 project managed with uv (`backend/`, `src/` layout)
- [x] Minimal FastAPI app with `GET /health`
- [x] pytest setup with a health endpoint test
- [x] Core domain models: `Claim`, `Evidence`, `VerificationResult`
- [x] Evidence requirement for `Supported` and `Contradicted` verdicts
- [x] Architecture and roadmap documents
- [x] ruff for linting and formatting
- [ ] Commit outstanding files (`backend/src/roamly/__init__.py`, `docs/`, README changes)
- [ ] Optional: GitHub Actions CI running `ruff check`, `ruff format --check`, and `pytest` (may be completed during M1)

**Acceptance criteria**

- `uv run pytest` and `uv run ruff check .` pass.
- Domain model tests cover valid creation, invalid values, missing fields, and JSON serialisation.

## M1 — Extraction prototype

**Goal:** turn pasted guide text into validated, source-linked claims, with skipped items explained.

**Tasks**

- **M1.1 Schemas and LLM abstraction (~5 h)**
  - Add `Claim.scope: str | None = None` (explicitly stated qualifiers only; no inferred dates).
  - Add `ExtractedClaim` (with optional `place_name`), `SkippedClaim`, and `ExtractionResult`.
  - Add the `LLMClient` protocol and `FakeLLMClient`.
- **M1.2 Deterministic post-processing (~6 h)**
  - Verify each `original_excerpt` is an exact substring of the source text.
  - Skip observations, unsupported types, missing places, unmatched excerpts, and duplicates, each with an explicit reason.
  - Assign claim IDs in the system.
  - Contract tests with `FakeLLMClient` for AC-03, AC-04, AC-06, and the missing-place case.
- **M1.3 Real provider and CLI (~6 h)**
  - Versioned extraction prompt.
  - DeepSeek adapter configured via environment variables; verify its structured-output capabilities before relying on them.
  - `CachedLLMClient` for on-disk response caching.
  - CLI that reads a text file and prints an `ExtractionResult` as JSON.
- **M1.4 Small extraction evaluation (~5 h)**
  - About 15 manually labelled extraction cases.
  - A script reporting precision, recall, and skip-reason breakdown, run once with the real provider and cached.

**Acceptance criteria**

- AC-03: "The museum is open every Monday" yields an `opening_hours` claim linked to the museum and the exact excerpt.
- AC-04: "I visited the museum on Monday" yields no recurring opening-hours claim.
- AC-06: an unsupported claim (e.g. a ticket price) is reported as skipped with reason `unsupported_type` and is never a `Claim`.
- A claim whose place cannot be determined is skipped with reason `missing_place`; no place name is invented.
- All of the above pass as contract tests at zero API cost.
- The CLI produces JSON for a real text sample.
- Evaluation results are recorded as measured, without a pass threshold.

**Out of scope:** URL import, place resolution, persistence, API endpoints, UI.

## M2–M6

Detailed tasks are written when each milestone begins. Goals and acceptance criteria follow the requirements document:

- **M2 Place resolution and review:** FR-07–FR-10; AC-05, AC-07, AC-13, AC-14.
- **M3 Evidence retrieval:** FR-11, FR-12; fixture-based evidence providers and ranking.
- **M4 Verification:** FR-13–FR-15; AC-08–AC-12, AC-15.
- **M5 Integration:** FR-01–FR-04, FR-16–FR-18; AC-01, AC-02; NFR-03 URL-fetch security.
- **M6 Evaluation and release:** requirements §9 benchmark (40–60 claims), comparison with a direct-LLM baseline, latency and cost measurements.

## Later versions

See [requirements §12](requirements.md#12-later-versions): V1.5 Itinerary Feasibility, V2 Disruption Copilot, V3 Proactive Monitoring.
