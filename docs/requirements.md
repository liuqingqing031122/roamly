# Roamly — MVP Requirements

**Version:** 0.3 (draft)  
**Product:** Travel Reality Checker  
**Status:** Requirements baseline; subject to revision during implementation

## 1. Objective

Help independent travellers assess time-sensitive claims in travel recommendations by extracting checkable statements, retrieving relevant evidence, and showing transparent, appropriately uncertain verdicts.

### Primary user story

> As an independent traveller, I want to check claims about a venue's opening hours, closures, and reservation rules, so that I can avoid relying on potentially outdated travel advice.

### Success condition

A user can provide guide text, review extracted claims, verify selected supported claims, and inspect a report linking each conclusion to relevant evidence or explicitly stating that verification was not possible.

## 2. MVP scope

**In scope:** public URL import with text fallback; direct text import; place and claim extraction; review and selection; retrieval of accessible external evidence; source-linked verdicts; basic evaluation and safety safeguards.

**Out of scope:** bypassing login/anti-bot protections; predicting queues or crowds; bookings or purchases; generating full itineraries; background monitoring; mobile apps; guaranteed coverage of social-media URLs.

### Supported claim categories

- `opening_hours`: regular operating days and times.
- `reservation`: whether advance booking is required or walk-ins are accepted.
- `closure`: scheduled, seasonal, or temporary closures.

The system must not transform a one-time observation (e.g. “I visited on Monday”) into a universal rule (e.g. “open every Monday”).

## 3. User workflow

1. Enter a public HTTP(S) URL or paste guide text; optionally specify destination and intended visit date.
2. If content retrieval fails, keep the URL and offer a manual text fallback without generating claims from unavailable content.
3. Extract places and checkable claims with exact source excerpts.
4. Resolve venue identity where possible; request confirmation when a name maps to multiple locations.
5. Preselect **at most three** supported claims, using deterministic priority: closure > reservation > opening hours; break ties by source order.
6. Let users correct venue identification, remove erroneous claims, and select/deselect supported claims.
7. Verify only the selected supported claims.
8. Present verdict, explanation, evidence link(s), retrieval timestamp, and limitations.

## 4. Functional requirements

| ID    | Requirement                                                                               | Priority |
| ----- | ----------------------------------------------------------------------------------------- | -------- |
| FR-01 | Accept one valid HTTP(S) guide URL or pasted guide text.                                  | Must     |
| FR-02 | Fetch readable content from accessible public pages within configured safety limits.      | Must     |
| FR-03 | Preserve source URL, submitted/retrieved content, retrieval status, and timestamp.        | Must     |
| FR-04 | Offer pasted-text fallback when a link is inaccessible.                                   | Must     |
| FR-05 | Extract named venues and supported factual claims with exact source excerpts.             | Must     |
| FR-06 | Associate each claim with its venue, type, and relevant date/scope where stated.          | Must     |
| FR-07 | Detect unresolved venue ambiguity and require confirmation before verification.           | Must     |
| FR-08 | Recommend no more than three supported claims using documented priority rules.            | Must     |
| FR-09 | Let users adjust claim selection and correct venue matches or remove errors.              | Must     |
| FR-10 | Clearly mark unsupported claim categories; do not submit them for verification.           | Must     |
| FR-11 | Retrieve candidate evidence relevant to the exact venue, claim, and applicable date.      | Must     |
| FR-12 | Record evidence URL, excerpt, retrieval time, and publication/update date when available. | Must     |
| FR-13 | Assess whether evidence supports, contradicts, or fails to establish the claim.           | Must     |
| FR-14 | Return one of Supported, Contradicted, Potentially Outdated, Unable to Verify.            | Must     |
| FR-15 | Show explanation, source links, verification timestamp, and uncertainty.                  | Must     |
| FR-16 | Verify only selected supported claims; no selection means no verification request.        | Must     |
| FR-17 | Allow a user to rerun verification for an existing claim.                                 | Should   |
| FR-18 | Export a report.                                                                          | Could    |

## 5. Verdict semantics

- **Supported:** relevant, sufficiently reliable evidence directly supports the claim for its stated scope/date.
- **Contradicted:** relevant, sufficiently reliable evidence directly conflicts with the claim for its stated scope/date.
- **Potentially Outdated:** evidence suggests that the information may have changed, but is not sufficient to establish a current supported or contradicted verdict. This is a caution flag, not proof of falsehood.
- **Unable to Verify:** no sufficiently relevant/reliable evidence exists, the venue/date is unresolved, or the claim cannot safely be assessed.

A general opening-hours page cannot by itself guarantee access on a particular holiday. A reservation policy accepting walk-ins does not guarantee a table or ticket.

## 6. Acceptance criteria (Given / When / Then)

| ID    | Given                                                                                    | When                           | Then                                                                                  |
| ----- | ---------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| AC-01 | A valid, accessible guide URL                                                            | User imports it                | Readable text and original URL are shown/stored.                                      |
| AC-02 | A login-restricted or unreadable URL                                                     | Import fails                   | URL is retained, fallback is offered, and no unsupported content is invented.         |
| AC-03 | Text containing “The museum is open every Monday”                                        | Claims are extracted           | An `opening_hours` claim is linked to the museum and exact source excerpt.            |
| AC-04 | Text stating “I visited the museum on Monday”                                            | Claims are extracted           | The system does not infer a recurring Monday opening policy.                          |
| AC-05 | Six supported claims                                                                     | Review opens                   | At most three are selected using the documented priority rule; user can change them.  |
| AC-06 | An unsupported ticket-price claim                                                        | Review opens                   | It is labelled unsupported and cannot be verified in V1.                              |
| AC-07 | Two same-named café branches with different rules                                        | Venue cannot be resolved       | User confirmation is required; neither branch's policy is misattributed.              |
| AC-08 | Claim: “Open every Monday”; official evidence: “Closed Mondays except selected holidays” | Verification runs              | Verdict is Contradicted with relevant source excerpt and URL.                         |
| AC-09 | Claim: “Walk-ins welcome”; official FAQ says reservations optional                       | Verification runs              | Verdict is Supported, without promising availability.                                 |
| AC-10 | An old year-round opening claim and a newer incomplete seasonal closure notice           | Verification runs              | Verdict may be Potentially Outdated with an explicit explanation of unresolved dates. |
| AC-11 | No reliable evidence                                                                     | Verification runs              | Verdict is Unable to Verify; no fabricated citation is shown.                         |
| AC-12 | A venue's standard hours but no holiday schedule                                         | User specifies a holiday visit | The system does not assert guaranteed holiday opening.                                |
| AC-13 | Only Opening Hours is selected                                                           | User starts verification       | Reservation and Closure are not verified.                                             |
| AC-14 | Zero claims selected                                                                     | User attempts verification     | Verification is blocked with a clear prompt.                                          |
| AC-15 | Retrieved text contains instructions to ignore system rules                              | Verification runs              | Retrieved text is treated as data, not as authority to change behavior.               |

These cases should first be implemented with **controlled, synthetic evidence fixtures**. Live-web integration tests should be separate to avoid unpredictable changes in source pages.

## 7. Data entities (initial)

- **TravelSource:** `id`, `url?`, `content`, `access_status`, `retrieved_at?`.
- **Place:** `id`, `name`, `location?`, `identity_status`.
- **Claim:** `id`, `source_id`, `place_id?`, `type`, `original_excerpt`, `normalized_statement`, `applicable_date?`, `selected`.
- **Evidence:** `id`, `claim_id`, `url`, `excerpt`, `retrieved_at`, `published_at?`, `source_type`.
- **VerificationResult:** `claim_id`, `verdict`, `explanation`, `evidence_ids`, `verified_at`, `limitations?`.

Dates, exceptions, and venue identity must remain explicit rather than being silently assumed.

## 8. Non-functional requirements

| ID     | Requirement                 | Validation                                                                                                                           |
| ------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| NFR-01 | Traceability                | Every evidence-backed verdict links to actual retrieved evidence.                                                                    |
| NFR-02 | Abstention and honesty      | Missing/ambiguous evidence produces a qualified or unverified result, not invented facts.                                            |
| NFR-03 | URL-fetch security          | Block non-HTTP(S) schemes, private/loopback/link-local addresses, unsafe redirects, oversized responses, and timeouts.               |
| NFR-04 | Prompt-injection resistance | Untrusted guide/evidence text cannot override verification instructions.                                                             |
| NFR-05 | Recovery                    | Import/search errors are visible and do not silently discard user input.                                                             |
| NFR-06 | Performance                 | Measure p50/p95 end-to-end latency; provisional target: typical short guide with 3 claims within 60 seconds in the test environment. |
| NFR-07 | Cost visibility             | Record model/search usage and estimated per-run cost; cap default verification at 3 claims.                                          |
| NFR-08 | Data minimisation           | Avoid logging unnecessary full user-submitted content or secrets.                                                                    |
| NFR-09 | Reproducibility             | Record verification time, source metadata, and model/prompt version where practical.                                                 |

## 9. Evaluation plan

Create an initial **40–60 claim** manually reviewed benchmark across the three categories, including contradictions, ambiguous venues, stale claims, missing evidence, special dates, and malicious retrieved instructions. Evaluate:

- Claim extraction precision and recall.
- Place-resolution accuracy and abstention on ambiguity.
- Evidence relevance and citation-support rate.
- Verdict accuracy, with a separate breakdown for unsupported/uncertain cases.
- End-to-end latency and approximate cost.

**Provisional aspirations, not measured results:** extraction precision ≥90%, recall ≥80%, evidence relevance ≥85%, verdict accuracy ≥80%, citation support ≥90%, and zero fabricated source URLs in the controlled benchmark. Small samples require cautious interpretation. Compare against a simple direct-LLM baseline using the same cases.

## 10. Development and AI-assisted coding policy

AI coding assistants may be used for boilerplate, implementation alternatives, unit-test drafts, debugging, and refactoring. Human review is required for requirements, architecture, security-sensitive URL fetching, evaluation labels, and final merges. Verify generated code by reading it and running appropriate tests. Do not claim a tool was used until it actually was; record meaningful decisions and corrections in issues, pull requests, or development notes.

## 11. Delivery milestones

1. **Extraction prototype:** pasted text → structured, source-linked claims.
2. **Place resolution and review:** venue confirmation, priority selection, unsupported claim handling.
3. **Evidence retrieval:** source metadata and relevant excerpts, initially using fixtures.
4. **Verification:** four verdicts with traceable reasoning and abstention.
5. **Integration:** safe URL import, web UI, reports, and error handling.
6. **Evaluation and release:** benchmark, failure analysis, deployment, and README update.

## 12. Later versions

- **V1.5 — Itinerary Feasibility:** opening-hour constraints, travel time, routing, scheduling conflicts.
- **V2 — Disruption Copilot:** user-reported disruptions, constrained alternatives, preserve unchanged reservations and preferences.
- **V3 — Proactive Monitoring:** opt-in monitoring of relevant changes and carefully qualified notifications.

## 13. Open decisions

- Initial language and destination coverage (English-only pilot vs bilingual travel content).
- Search/data providers and their usage limits.
- Frontend/backend stack and deployment target.
- Whether reports persist across sessions in the first release.
