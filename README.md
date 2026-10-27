# Roamly

**Verify before you go. Adapt when plans change.**

Roamly is an in-development AI-powered travel intelligence assistant designed to help independent travellers check potentially outdated or inaccurate claims in travel guides. Its first milestone is an **evidence-grounded Travel Reality Checker**; later versions will assess itinerary feasibility and suggest minimal changes when plans are disrupted.

> **Status:** Planning / early development. Features described below are targets, not implemented capabilities.

## The problem

Travel recommendations often contain time-sensitive statements about opening hours, closures, and reservation requirements. A traveller may rely on an old post without knowing whether those details still apply. Roamly aims to turn such statements into checkable claims and provide transparent, source-linked assessments rather than unsupported answers.

## MVP: Travel Reality Checker

The intended workflow is:

1. **Import** a publicly accessible travel-guide URL or paste guide text. If a URL cannot be read, retain the link and request pasted content.
2. **Extract** named places and claims about opening hours, closures, and reservation requirements, preserving the exact source passage.
3. **Review** extracted claims. Roamly recommends up to three high-priority supported claims; users can edit place identification and change the selection before verification.
4. **Retrieve** relevant evidence, prioritising official venue information when available and recording source URLs and retrieval times.
5. **Verify** each selected claim against evidence and report **Supported**, **Contradicted**, **Potentially Outdated**, or **Unable to Verify**, with explanations and citations.

Verification is sensitive to the intended visit date. When no date is supplied, results concern general rules and must not imply a guarantee for a specific day.

## What makes this project interesting

- **Evidence-grounded verification:** distinguish the original claim, retrieved evidence, and the system's conclusion.
- **Uncertainty-aware output:** abstain when evidence is insufficient or a place cannot be identified confidently.
- **Human-in-the-loop review:** let users correct extracted claims and choose which ones to verify.
- **Measurable quality:** evaluate claim extraction, evidence relevance, verdict accuracy, citation support, latency, and cost.
- **Safe external-content handling:** treat retrieved pages as untrusted data, enforce URL-fetching safeguards, and avoid bypassing access restrictions.

## Roadmap

| Phase                            | Scope                                                                                           | Status      |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | ----------- |
| **V1 — Reality Checker**         | Import, claim extraction, place resolution, evidence retrieval, verification report, evaluation | Planned     |
| **V1.5 — Itinerary Feasibility** | Check time, distance, opening-hour, and route constraints                                       | Future      |
| **V2 — Disruption Copilot**      | Suggest minimal itinerary changes after user-reported disruptions                               | Future      |
| **V3 — Proactive Monitoring**    | With permission, monitor relevant changes and propose updates                                   | Exploratory |

## Development approach

Roamly will use **AI-assisted development** where useful for scaffolding, implementation suggestions, test generation, debugging, and code review. The developer remains responsible for requirements, architecture, validating generated code, security, and final decisions. Tools and concrete lessons learned will be documented as they are actually used.

## Evaluation plan

A small manually labelled dataset will cover supported claims, contradictions, insufficient evidence, ambiguous places, and stale information. Initial metrics will include extraction precision/recall, evidence relevance, verdict accuracy, citation support, and failure analysis. Results will be published only after tests have been run.

## Documentation

See [MVP requirements](docs/requirements.md) for user stories, acceptance criteria, constraints, and the initial delivery plan.

## Tech stack

**Implemented:** Python 3.12, uv, FastAPI, Pydantic, pytest, Ruff.

**Planned:**

- **Frontend:** React with TypeScript.
- **Persistence:** SQLite.
- **LLM integration:** provider-independent client interface; DeepSeek is the initial development provider.

No framework is adopted solely for the sake of using it. See [Architecture](docs/architecture.md) for rationale.

## Local setup

Not available yet; instructions will be added with the first runnable implementation.
