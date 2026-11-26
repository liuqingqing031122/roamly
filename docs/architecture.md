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
| LLM | DeepSeek initially, behind `LLMClient`; called with httpx (no SDK) | Implemented (M1.3) | See §7 and decision D-006 |
| Persistence | SQLite | Planned (M2+) | Zero-ops; sufficient for traceability requirements |
| HTML fetch / extraction | httpx, trafilatura | Planned (M5) | Timeouts, streaming size limits, manual redirect handling for SSRF safety |
| Place lookup | OpenStreetMap Nominatim, behind the `PlaceSearch` protocol | Implemented (M2.2) | Free public instance with a strict usage policy; paid alternatives remain an option |
| Web search | To be decided | Planned (M5) | Open question |
| Frontend | To be decided (Jinja2 + HTMX or React) | Planned (M5) | Open question; the API comes first |

## 4. Module layout

Current (**Implemented**):

```
backend/
  pyproject.toml, uv.lock, .python-version
  src/roamly/
    main.py               # FastAPI app; GET /health
    domain.py             # Core domain models
    extraction.py         # Extraction schemas and deterministic post-processing
    extraction_prompt.py  # Versioned extraction prompt
    extraction_cache.py   # On-disk cache of validated extraction output
    pipeline.py           # Orchestration: prompt → LLM or cache → post-processing
    cli.py                # `roamly-extract` command
    places.py             # M2.1: place resolution schemas, PlaceSearch protocol, resolver
    places_nominatim.py   # M2.2: Nominatim adapter for PlaceSearch (rate limit, disk cache)
    places_cli.py         # M2.3: `roamly-resolve` command
    places_confirm.py     # M2.4: `roamly-confirm` interactive manual confirmation
    web_fetch.py          # M3.1: SourcePage schema, safe official-website fetching, cache
    web_fetch_cli.py      # M3.1: `roamly-fetch` command
    evidence_prompt.py    # M3.2: versioned evidence extraction prompt
    evidence.py           # M3.2: evidence schemas, chunking, validation, cache
    evidence_cli.py       # M3.2: `roamly-extract-evidence` command
    evaluation/
      dataset.py          # Load and validate labelled datasets (read-only)
      runner.py           # Run the extraction pipeline over every case
      predictions.py      # Stored run format: save and reload predictions offline
      matching.py         # Offline one-to-one matching of expected and predicted claims
      git_info.py         # Best-effort Git commit and dirty state for run metadata
      cli.py              # `roamly-eval run` and `roamly-eval show` commands
    llm/
      __init__.py         # LLMClient protocol, errors, FakeLLMClient
      deepseek.py         # DeepSeek adapter
  tests/                  # One test module per source module
```

Planned additions, introduced only when a milestone needs them:

```
  src/roamly/
    selection.py   # M2: deterministic claim preselection
    retrieval/     # M3: evidence providers, chunking, ranking
    verification/  # M4: evidence assessment, citation checks, verdict aggregation
    ingestion/     # M5: URL policy, safe fetching, content extraction
    api/           # M5: HTTP routers (thin orchestration only)
```

`domain.py` stays a single module until it grows enough to justify a package.

## 5. Domain model

### Implemented (`backend/src/roamly/domain.py`)

- **Claim:** `id`, `place_name`, `claim_type` (`opening_hours` | `reservation` | `closure`), `original_excerpt`, `normalized_statement`, `scope`.
- **Evidence:** `id`, `claim_id`, `source_url` (`HttpUrl`), `source_title`, `content_excerpt`, `retrieved_at`.
- **VerificationResult:** `claim_id`, `verdict` (`Supported` | `Contradicted` | `Potentially Outdated` | `Unable to Verify`), `explanation`, `evidence_ids`. A validator requires at least one evidence ID for `Supported` and `Contradicted` (NFR-01).

### Implemented in M1 (`backend/src/roamly/extraction.py`)

- **Claim.scope:** `list[str]`, empty by default. Each item is a temporal or applicability qualifier explicitly stated in the source (e.g. `["in summer"]`, `["on weekends", "only during summer"]`); items are combined with AND. Dates are never inferred; parsing is deferred to verification (M4). `ExtractedClaim` also accepts `null` or a single string from the LLM and converts them to a list; post-processing drops blank items. See the [annotation guidelines](extraction_annotation_guidelines.md) for the full rules.
- **ExtractedClaim** (in `extraction.py`): the untrusted LLM output schema.
  - `place_name: str | None` — `None` when the place cannot be determined from the source or its explicit context. Place names must never be invented.
  - `claim_type`: the three supported types plus `unsupported`.
  - `assertion_kind`: `policy` (a general rule) or `observation` (a one-time experience, e.g. "I visited on Monday").
  - `original_excerpt`, `normalized_statement`, `scope`.
- **SkippedClaim:** an `ExtractedClaim` plus a `reason`: `unsupported_type`, `observation`, `missing_place`, `excerpt_not_found`, or `duplicate`.
- **ExtractionResult:** `claims: list[Claim]` and `skipped: list[SkippedClaim]`.

`ExtractedClaim` and `Claim` are deliberately separate (decision D-004). `Claim` is only constructed by extraction post-processing, so downstream code can rely on every `Claim` having a supported type, a stated place, a policy assertion, and an excerpt found in the source. IDs are assigned by the system, never taken from LLM output.

### Implemented in M2.1 (`backend/src/roamly/places.py`)

- `PlaceCandidate`: one provider-neutral search result (`provider`, `provider_id`, `name`, `display_name`, coordinates, administrative `areas`, optional `category` and `website`).
- `PlaceResolution`: one per distinct normalized place name, linked to its claims through `claim_ids`. `status` is `resolved`, `ambiguous`, or `unresolved`, with a `reason`; `place` is set only when resolved; all search `candidates` are kept for manual confirmation.
- `resolve_places(extraction, destination, search)` makes one search per distinct place name. A place is resolved only when exactly one candidate has the same normalized name and lies in the user-provided destination (every comma-separated destination part equals one of the candidate's areas). Partial or missing area information needs confirmation; candidates clearly elsewhere are never chosen. A `PlaceSearchError` makes only that place `unresolved` (`lookup_failed`).
- `NominatimSearch` (`places_nominatim.py`, M2.2) implements `PlaceSearch` with the Nominatim Search API (`format=jsonv2`, `addressdetails=1`, `extratags=1`, `accept-language=en`). It follows the public usage policy: an identifying User-Agent is required (`ROAMLY_NOMINATIM_USER_AGENT`), requests are spaced by at least one second per instance, successful responses (including empty ones) are cached on disk under `.roamly_cache/places/` without expiry, and the base URL is configurable (`ROAMLY_NOMINATIM_URL`). `areas` come from the address levels `suburb` to `country`; `provider_id` is `osm_type/osm_id`. Timeouts, transport errors, non-200 responses, and malformed responses raise `PlaceSearchError`; other exceptions propagate. Output that shows place data must credit © OpenStreetMap contributors (ODbL).
- `roamly-resolve` (`places_cli.py`, M2.3) reads the JSON printed by `roamly-extract`, resolves its places with `NominatimSearch`, prints one block per place (status, reason, claims, every candidate with its address and area check), and with `--output` saves a `PlaceResolutionFile` (format version, destination, the full `ExtractionResult`, resolutions, OSM attribution) as the self-contained input for evidence retrieval. Exit code 2 means at least one lookup failed; re-running retries only those, since successful lookups are cached.
- `roamly-confirm` (`places_confirm.py`, M2.4) walks through every `ambiguous` or `unresolved` place of a resolution file. The person picks a candidate by number (with a yes/no check), searches Nominatim under another name, skips, or quits without saving; automatically resolved places are not shown. A confirmed place becomes `resolved` with reason `manually_confirmed` and a `confirmation` record (the original automatic reason and the search name, if any); place name and claim IDs are kept, and the extraction is not changed. The result is written to a new file (`<input>.confirmed.json` by default) in the same `PlaceResolutionFile` format; replacing the input needs `--overwrite` and a final confirmation.
- `Claim` is unchanged; resolutions refer to claims by ID. Only resolved or user-confirmed places proceed to evidence retrieval.

### Implemented in M3.1 (`backend/src/roamly/web_fetch.py`)

- `PageFetch`: the outcome of fetching one URL — `status` (`fetched`, `skipped`, `failed`) with a `reason` (e.g. `no_website`, `unsafe_url`, `timeout`, `http_error`, `not_html`, `too_large`, `no_text`) and `error` detail, `requested_url` (normalized) and `final_url` (after redirects), `http_status`, `title`, readable `text` (capped at 100,000 characters, `text_truncated`), `fetched_at`, and `from_cache`. `fetched` means only that the page was retrieved; it says nothing about whether a claim is true (evidence extraction is M3.2, verification M4).
- `SourcePage` extends `PageFetch` with the place (`place_name`, `place_id` as `provider:provider_id`), its `claim_ids`, the stored `website` value, and `source_type` (`official_website`).
- `WebsiteFetcher` downloads one HTML page with httpx and extracts visible text with the standard-library HTML parser (scripts, styles, `noscript`, templates, SVG, and iframes dropped; one line per block element). No JavaScript rendering. Successful downloads are cached as raw HTML under `.roamly_cache/pages/` with their original download time; failures are never cached; `refresh` re-downloads. `fetch_official_websites` makes one `SourcePage` per place: places that are not resolved or have no website are skipped, and a failure affects only its own place.
- robots.txt: before each request (including every redirect hop), the origin's robots.txt is fetched once per run with the same URL and address checks, and the rules for the User-Agent's product token (or `*`) are applied with a small RFC 9309 matcher (longest match, `*` and `$` wildcards, Allow wins ties; `urllib.robotparser` lacks wildcards). A 4xx robots.txt means no rules; a 5xx, timeout, network error, or unsafe redirect means nothing on that origin is fetched (`failed` / `robots_unavailable`). A disallowed page is `skipped` / `robots_disallowed` and never requested. Cached pages are served without a new robots.txt check.
- The User-Agent is `Roamly/0.1 (...)` by default and can be set with `ROAMLY_FETCH_USER_AGENT`; its product token (the text before the first `/`) is what robots.txt groups are matched against.
- `roamly-fetch` reads a place resolution file and saves a `SourcePageFile` (the place file plus the pages) with `--output`; `--no-cache` and `--refresh` bypass the cache (no TTL).
- `domain.Evidence` (one excerpt for one claim) is unchanged; it is produced from these pages in M3.2.

### Implemented in M3.2 (`backend/src/roamly/evidence.py`)

- One LLM call per fetched page (prompt `evidence-v1`) receives all of that page's claims and the page text, each in delimiters, and returns `ExtractedEvidenceList` (`claim_id`, `excerpts`) — untrusted output. Pages longer than 30,000 characters are split on line boundaries into chunks with 1,000 characters of overlap, one call each, so no text is silently dropped.
- Deterministic validation keeps an excerpt only if it occurs in the text that was sent (whitespace-equivalent, reusing `find_excerpt`; the stored excerpt is the page span), belongs to a claim of that page, is not a duplicate, and is within 5 per claim. Rejected output is listed as `EvidenceIssue` (`unknown_claim_id`, `excerpt_not_found`, `duplicate`, `too_many_excerpts`); nothing is repaired or replaced.
- `ClaimEvidence` (one per claim) has `status` `found`, `none_found` (checked, nothing relevant or valid), `page_unavailable` (reason `no_page` or the page's own reason, e.g. `robots_disallowed`; no LLM call), or `failed` (`llm_error` or `missing_from_response`), plus `source_type`, `source_url` (final URL), `fetched_at`, and `evidence` as `domain.Evidence` records (unchanged model; `retrieved_at` is the page's download time). `found` means relevant passages exist, not that the claim is true.
- `EvidenceCache` (`.roamly_cache/evidence/`) stores the validated LLM output keyed by prompt version and text, response schema, model settings, and the hash of the full user prompt (claims and chunk text); validation re-runs on every hit; LLM failures are not cached.
- `roamly-extract-evidence` reads a pages file and saves an `EvidenceFile` (prompt version, model settings, the pages file, claim results, issues) with `--output`; the summary shows one-line excerpt previews, never full page text.

### Planned for later milestones

User confirmation of ambiguous places, `TravelSource`, `Analysis` (a check session with optional destination and visit date), per-evidence assessments, and usage/cost records. Fields such as `published_at`, `source_type`, `limitations`, `verified_at`, and model/prompt version are added when the milestone that uses them begins. A `place_id` on `Claim` is added only when persistence needs it.

## 6. LLM versus deterministic responsibilities (Planned)

| Step | LLM | Deterministic Python |
| --- | --- | --- |
| URL validation, fetching, content extraction | | ✓ |
| Claim and place extraction; policy vs observation | ✓ | Excerpt substring check (runs of whitespace are equivalent), type filtering, missing-place check, de-duplication, ID assignment |
| Place resolution | Optional disambiguation | ✓ candidate lookup and ambiguity detection |
| Preselection of at most three claims | | ✓ closure > reservation > opening hours, then source order |
| Query construction, retrieval, ranking | | ✓ |
| Per-evidence stance assessment | ✓ | Citation check: quoted text must exist in the evidence |
| Verdict aggregation, date and holiday rules | | ✓ |
| Cost and latency recording, evaluation metrics | | ✓ |

A typical run with three claims needs about five LLM calls: one or two for extraction and up to three for verification.

## 7. LLM integration (Implemented, M1)

- **Abstraction.** A small `LLMClient` protocol with one structured-output method: given a system prompt, a user prompt, and a Pydantic schema, return a validated instance plus token usage. Failures raise `LLMError` subclasses (`LLMAPIError` for network, timeout, and HTTP errors; `LLMResponseError` for empty, truncated, invalid, or schema-mismatched output) instead of returning empty results. Pipeline code depends only on this protocol and contains no provider-specific behaviour.
- **Implementations.**
  - `FakeLLMClient` — returns predefined outputs and records calls; used by unit and contract tests.
  - `DeepSeekClient` — the initial provider for local development and the M1 extraction evaluation. It is not a permanent dependency; OpenAI or Anthropic may be used for deployment or provider comparison.
- **Configuration.** `ROAMLY_LLM_PROVIDER` (only `deepseek` is supported; the default), `ROAMLY_LLM_MODEL` (default `deepseek-flash`), and `ROAMLY_LLM_API_KEY`. API keys and `.env` files are never committed and never appear in error messages or cache entries.
- **Structured output.** DeepSeek offers JSON mode (`response_format: json_object`), which guarantees syntactically valid JSON but not schema conformance, and may occasionally return empty content. The adapter therefore appends the response model's JSON Schema to the system prompt, disables thinking mode, and validates every response with Pydantic. Real-API behaviour still needs to be confirmed by a manual smoke test.
- **Prompt versioning.** Each prompt has a version constant (currently `extraction-v2`, which changed `scope` to a list) that is part of the cache key.
- **Input limit.** The CLI rejects guide text longer than 30,000 characters before any LLM call or cache access, bounding the cost of a single run.

### Extraction caching (Implemented, M1.3)

Caching lives at the extraction layer (`ExtractionCache` in `extraction_cache.py`), not in a generic `CachedLLMClient` wrapper as originally planned (decision D-008). A generic wrapper sees only prompts and a response type; the extraction layer knows the prompt version and response schema and can make them explicit parts of the key.

- **What is cached.** The Pydantic-validated `ExtractedClaims` returned by the LLM, not the final `ExtractionResult`. The LLM output is the expensive, non-deterministic part; post-processing is cheap and deterministic. Caching the raw validated output means post-processing runs again on every cache hit, so fixes to post-processing rules apply to cached data without new API calls, and accepted claims and skip reasons are never stale.
- **Cache key.** SHA-256 over a canonical JSON object containing:
  - the cache format version;
  - the prompt version and a hash of the system prompt text (so prompt edits without a version bump still miss);
  - a hash of the `ExtractedClaims` JSON Schema (the response schema version);
  - the model settings that affect output (provider, model, response format, thinking mode, temperature, max tokens);
  - a hash of the source text.
- **Writes.** Only successfully validated responses are written; API errors and invalid responses propagate and are never cached. Writes are atomic (temporary file plus rename). A failed write is logged and does not discard the result.
- **Reads.** A missing entry is a miss. An unreadable or invalid entry is logged, treated as a miss, and overwritten by the next successful call.
- **No TTL.** Entries never expire. Extraction depends only on the source text and the configuration in the key, not on facts that change over time. Verification results (M4), which depend on live external evidence, must not reuse this cache without an expiry policy.
- **Stored data.** Entries store hashes of the source text and prompt, not the full guide text. However, the cached claims contain verbatim `original_excerpt` passages and normalized statements derived from the guide, so cache files can hold fragments of user-submitted content. The cache directory (default `.roamly_cache/extractions/`) is local and excluded from git.

## 8. Verification and verdict aggregation (Planned, M4)

1. The LLM receives one claim, the optional visit date, and numbered evidence passages wrapped in explicit untrusted-content delimiters. It has no tools and may only return, per passage, a stance (`supports`, `contradicts`, `suggests_change`, `irrelevant`), a verbatim quote, and a short rationale.
2. Citation check: each quote must be a substring of its evidence passage and reference an existing evidence ID; otherwise the assessment is discarded.
3. Deterministic aggregation maps valid assessments to one of the four verdicts. Unresolved places or no relevant evidence yield `Unable to Verify`. The detailed rule table is finalised in M4 with table-driven tests.
4. Date rules: if the visit date is a holiday or special date not covered by evidence, a limitation is added and no date-specific `Supported` verdict is given (AC-12). A `Supported` reservation verdict never implies availability (AC-09).

## 9. Security (URL fetching implemented in M3.1; the rest planned)

- **URL fetching (NFR-03):** HTTP(S) only; block private, loopback, and link-local addresses, including after redirects; cap response size and time. Access restrictions are never bypassed. Implemented in `web_fetch.py`: only http/https on default ports and without credentials; `localhost` and every address that is not global (private, loopback, link-local, shared, reserved, multicast, IPv4-mapped) are refused, for the first URL and for each redirect hop, which is followed manually (at most 5); responses are limited to 2 MB and 30 seconds. robots.txt (RFC 9309) is checked for every origin before each request, including redirect hops; see M3.1 below.
- **Residual SSRF risk — DNS rebinding (not prevented):** the host is resolved for the check and again by httpx when it connects, so a name whose DNS answer changes between the two (a public address first, then a private one) can still reach an internal address. SSRF protection is therefore **not complete**. Usage limits until this is fixed: only fetch website values from place data (OpenStreetMap or a person's confirmation), never arbitrary URLs from untrusted users; do not run the fetcher on a host where internal services or a cloud metadata endpoint must stay unreachable, or add network egress rules there. Fix options: (1) a custom `httpcore` network backend whose `connect_tcp` resolves, checks, and connects to the checked IP (TLS still verifies the hostname); `httpx.HTTPTransport` does not accept a network backend, so this needs either the private `_pool` attribute or a small custom transport; (2) a third-party httpx SSRF guard, none of which was mature enough to adopt as of 2026-11; (3) egress filtering at the network level, which is the most robust and needed for any public deployment.
- **Prompt injection (NFR-04):** retrieved and user-supplied text is delimited as data, verification calls expose no tools, outputs are schema-validated, and citations are checked against stored evidence.
- **Secrets and data minimisation (NFR-08):** keys come from the environment only; full user content is not logged unnecessarily.

## 10. Testing and evaluation strategy

| Layer | Purpose | Runs in CI | Status |
| --- | --- | --- | --- |
| Unit | Deterministic logic and model validation | Yes | Implemented |
| Contract | Pipeline steps with `FakeLLMClient`, mocked HTTP, and fixtures; covers acceptance criteria | Yes | Implemented (M1) |
| Recorded | Replay cached real-model responses for prompt regression | Yes | Planned (M1.3) |
| Live | Real APIs and websites, marked `@pytest.mark.live` | No (manual) | Planned |
| Evaluation | Metrics on labelled datasets; reports numbers, not pass/fail | No (manual) | Runner and claim matching implemented (M1.4b); scoring planned |

Evaluation runs are separated from scoring. `roamly-eval run` passes each dataset case through the normal extraction pipeline (`extract_claims`, including the cache and post-processing) and saves one new JSON file per run: the full `ExtractionResult` or the error for each case ID, plus the dataset name, path, and SHA-256, the prompt version and system-prompt hash, the client's non-secret model settings, timestamps, the Git HEAD commit and whether the working tree had uncommitted changes when the run started (`null` when unavailable), and per-case LLM calls, cache hits, and token usage. A failing case is recorded and the run continues. Ground truth is not copied into run files. `roamly-eval show` and `roamly-eval match` read saved runs without calling an LLM. Existing run files are never replaced unless `--overwrite` is given.

Claim matching (`roamly-eval match RUN_FILE`, matcher `matching-v1`) implements the *detection matching* defined in the [annotation guidelines](extraction_annotation_guidelines.md) §5 (version 1.1). It pairs expected and accepted predicted claims one-to-one before any scoring. It refuses to run if the dataset's SHA-256 differs from the one recorded in the run, matches only the cases in the run, and keeps failed cases as failed. A pair is a candidate only when both excerpts overlap in the source; statements are then compared by token overlap with venue names removed. Same-venue candidates with similar statements are confirmed automatically, and a type mismatch requires a higher similarity, so a matched pair can still have a wrong type or scope for strict scoring and Level 2 to report. Candidates with a different venue, only moderate similarity, or a close competitor, including one predicted claim that covers several expected claims, are marked `needs_review` and are not counted as matched or unmatched until a person decides. The result is saved next to the run file with the dataset and run hashes.

Planned follow-ups, recorded here so that scoring is built consistently:

- **Thresholds.** The `matching-v1` similarity thresholds are initial heuristics, checked only on dataset v1 and hand-written test cases, not on large real-world data. Any change to the matching rules or thresholds must bump `MATCHER_VERSION` so results stay comparable.
- **Human review.** Decisions on `needs_review` pairs will be stored in a separate review JSON file keyed by case ID, expected claim ID, and predicted claim ID. Generated matching files are never edited, so matching can be re-run at any time.
- **Merged claims.** Automatic matching stays one-to-one. Scoring must handle a predicted claim that covers several expected claims explicitly and must never count it as several true positives.
- **Two Level 1 views.** Scoring will report detection metrics (all detection matches) and strict extraction metrics (matches that also have the correct venue and claim type), as defined in the guidelines §5.

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
| D-008 | 2026-10-29 | Cache validated `ExtractedClaims` in an extraction-level `ExtractionCache` instead of a generic `CachedLLMClient`; re-run post-processing on every hit; no TTL | The key must include the prompt version and response schema, which only the extraction layer knows; caching pre-post-processing output keeps results consistent with current rules; extraction does not depend on time-sensitive facts |
| D-009 | 2026-10-29 | Put extraction orchestration in `pipeline.py` and expose it through the `roamly-extract` CLI; limit input to 30,000 characters | Avoids a circular import between `extraction.py` and `extraction_cache.py`; the limit bounds per-run LLM cost |
| D-010 | 2026-11-18 | Evaluation runner reuses `extract_claims` unchanged and stores predictions as versioned run files separate from scoring; token usage is recorded by wrapping the `LLMClient` | One extraction path for production and evaluation; scoring can be iterated without new API calls; the pipeline needs no evaluation-specific changes |
| D-011 | 2026-11-19 | Claim matching is deterministic and separate from scoring: excerpt overlap is required, statement similarity decides, type and scope differences do not block a match, and uncertain or merged pairs go to `needs_review` instead of being guessed. Annotation guidelines 1.1 define this as detection matching and keep the 1.0 same-type rule as strict extraction scoring. Human review decisions will be stored in separate review files | Keeps detection metrics independent of field errors (a wrong type is a quality error, not a missed claim) while preserving the stricter view; uncertain cases are surfaced for human review rather than silently counted as true or false positives; thresholds are initial heuristics, so the matcher is versioned |
| D-012 | 2026-11-24 | Place resolution (M2.1) is deterministic and conservative: exact match on normalized names, a user-provided destination checked against the candidate's administrative areas, one search per distinct place name, and `resolved` only for a single same-name candidate confirmed inside the destination; everything else is `ambiguous` or `unresolved` with all candidates kept. Search providers sit behind a `PlaceSearch` protocol | A wrong place silently invalidates every later verdict, so uncertainty is surfaced for confirmation instead of guessed; the area check needs no extra API call and is easy to test; Nominatim (M2.2) or a paid provider can be plugged in without changing the resolver |
| D-013 | 2026-11-24 | Use the public Nominatim instance through `NominatimSearch`, with a per-instance one-second request interval, a disk cache of raw responses keyed by base URL and all query parameters, and no automatic retries | Free and sufficient for the few place names in one guide; caching and rate limiting are required by the usage policy; storing raw responses lets mapping fixes apply to cached entries; failed lookups are not cached, so re-running retries them |
| D-014 | 2026-11-24 | Keep extraction and place resolution as two commands joined by a file (`roamly-extract guide.txt > extraction.json`, then `roamly-resolve extraction.json`); the resolution file embeds the extraction result | No change to the M1 CLI; resolution can be re-run with another destination without new LLM calls; later steps need only one file |
| D-015 | 2026-11-24 | Excerpt validation treats any run of whitespace (spaces, line breaks, tabs) as equal to any other; everything else must match exactly. An accepted claim stores the matching source span as `original_excerpt` | Guides wrap lines, and the LLM joins them with spaces, so exact matching rejected correct excerpts (3 of 5 claims in the Porto smoke test). Storing the source span keeps every `Claim` excerpt an exact substring of the source, which evaluation matching and citation display rely on |
| D-016 | 2026-11-24 | Record manual place confirmation in the existing `PlaceResolution`: a new reason `manually_confirmed` (status `resolved`) plus an optional `confirmation` field; confirmation is an interactive terminal menu that writes a new file | Additive and backward compatible (older files load unchanged); later steps treat both kinds of `resolved` alike but can always tell a person's choice from an automatic match; no UI, database, or session state is needed for the MVP |
| D-017 | 2026-11-24 | Official websites are fetched by a dedicated `WebsiteFetcher` into page-level `SourcePage` records (separate from claim-level `Evidence`), with manual redirect handling and a public-address check per hop, standard-library text extraction, and a raw-HTML disk cache without expiry | Keeps retrieval outcomes (fetched, skipped, failed) apart from claim verification; manual redirects are the simplest way to apply the SSRF check to every hop; no new dependencies; caching raw HTML lets text-extraction fixes apply to cached pages |
| D-018 | 2026-11-26 | Respect robots.txt with a minimal RFC 9309 matcher, conservatively (unretrievable robots.txt blocks the origin); make the fetch User-Agent configurable (`ROAMLY_FETCH_USER_AGENT`); do not fix DNS rebinding yet but document it as a residual SSRF risk with usage limits | Site owners' crawl rules are honoured without a crawler framework; the standard-library parser ignores wildcards, which could fetch disallowed pages; a rebinding fix needs private httpx internals or a custom transport, so it waits for an explicit decision |
| D-019 | 2026-11-26 | Extract evidence with one LLM call per page (all its claims), validate excerpts deterministically against the page text, keep `domain.Evidence` unchanged and wrap it in a per-claim `ClaimEvidence` status, and cache raw LLM output per prompt hash | Fewer calls than one per claim; cross-language selection needs the LLM, but only verbatim page spans are accepted, so translations or invented text cannot become evidence; per-claim statuses keep "nothing found" separate from "LLM failed" and "no page"; validation changes apply to cached output |

## 13. Open questions

- Web search provider and its usage terms.
- Frontend approach (server-rendered Jinja2 + HTMX versus React + TypeScript).
- Deployment target and the LLM provider used for deployment.
- Initial language and destination coverage.
- Whether reports persist across sessions in the first release.
- Exact verdict aggregation rules, including handling of unknown publication dates and conflicting sources.
