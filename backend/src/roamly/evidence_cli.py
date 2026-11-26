"""Command-line entry point: extract claim-level evidence from fetched pages.

Reads a file saved by `roamly-fetch --output`, asks the LLM for excerpts relevant to
each claim (cached), validates them, and prints one block per claim with short
excerpt previews (never full page text). With `--output`, the result is saved as
JSON. Evidence is candidate material for verification, not a verdict.

Exit codes: 0 success; 1 error; 2 extraction failed for at least one claim (the
result is still printed and saved).
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from roamly.cli import CLIError, create_client_from_env
from roamly.evaluation.dataset import format_validation_error
from roamly.evaluation.predictions import write_json_file
from roamly.evidence import (
    DEFAULT_CACHE_DIR,
    ClaimEvidence,
    EvidenceCache,
    EvidenceIssue,
    extract_evidence,
)
from roamly.evidence_prompt import EVIDENCE_PROMPT_VERSION
from roamly.llm.deepseek import DeepSeekClient
from roamly.web_fetch_cli import SourcePageFile

EVIDENCE_FORMAT_VERSION = 1
PREVIEW_CHARS = 100

EXIT_ERROR = 1
# The result was produced, but extraction failed for at least one claim.
EXIT_EXTRACTION_FAILED = 2


class EvidenceFile(BaseModel):
    """Saved output of `roamly-extract-evidence`."""

    format_version: Literal[1] = EVIDENCE_FORMAT_VERSION
    prompt_version: str
    model_settings: dict[str, object]
    # The fetched pages (with places and claims) the evidence was taken from.
    pages: SourcePageFile
    claims: list[ClaimEvidence]
    issues: list[EvidenceIssue]


def read_pages_file(path: Path) -> SourcePageFile:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CLIError(f"File not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CLIError(f"File is not valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise CLIError(f"Cannot read {path}: {exc.strerror}") from exc
    try:
        return SourcePageFile.model_validate_json(content)
    except ValidationError as exc:
        raise CLIError(
            f"Not a pages file (saved by roamly-fetch --output): {path}\n"
            f"{format_validation_error(exc)}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-extract-evidence",
        description="Find evidence excerpts for each claim in fetched official websites.",
    )
    parser.add_argument("pages", type=Path, help="pages file saved by roamly-fetch --output")
    parser.add_argument("--output", type=Path, help="also save the result as JSON here")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for cached LLM output (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    return parser


def main(
    argv: Sequence[str] | None = None,
    client_factory: Callable[[], DeepSeekClient] = create_client_from_env,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        pages = read_pages_file(args.pages)
        if args.output is not None and args.output.exists() and not args.overwrite:
            raise CLIError(f"Output file already exists: {args.output} (use --overwrite)")
        client = client_factory()
        cache = None if args.no_cache else EvidenceCache(args.cache_dir, client.settings)
        claims, issues = extract_evidence(
            pages.places.extraction.claims, pages.pages, client, cache
        )
        result = EvidenceFile(
            prompt_version=EVIDENCE_PROMPT_VERSION,
            model_settings=dict(client.settings),
            pages=pages,
            claims=claims,
            issues=issues,
        )
        if args.output is not None:
            try:
                write_json_file(
                    args.output, result.model_dump_json(indent=2), overwrite=args.overwrite
                )
            except FileExistsError as exc:
                raise CLIError(f"Output file already exists: {args.output}") from exc
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(format_evidence(result))
    if args.output is not None:
        print(f"\nSaved: {args.output}")
    return EXIT_EXTRACTION_FAILED if any(c.status == "failed" for c in claims) else 0


def format_evidence(result: EvidenceFile) -> str:
    """One block per claim with short, single-line excerpt previews."""
    claims = result.claims
    statuses = ("found", "none_found", "page_unavailable", "failed")
    counts = {status: sum(c.status == status for c in claims) for status in statuses}
    statements = {c.id: c.normalized_statement for c in result.pages.places.extraction.claims}
    lines = [
        f"Claims: {len(claims)} ({counts['found']} found, {counts['none_found']} none found, "
        f"{counts['page_unavailable']} page unavailable, {counts['failed']} failed)",
        "Evidence is candidate material for verification, not a verdict.",
    ]
    for claim in claims:
        label = claim.status if claim.reason is None else f"{claim.status}: {claim.reason}"
        lines += ["", f"[{label}] {claim.claim_id} {claim.place_name}"]
        lines.append(f"  claim:  {statements.get(claim.claim_id, '')}")
        if claim.source_url:
            lines.append(f"  source: {claim.source_url}")
        if claim.error:
            lines.append(f"  error:  {claim.error}")
        for evidence in claim.evidence:
            lines.append(f"  - {_preview(evidence.content_excerpt)}")
    if result.issues:
        lines += ["", f"Rejected LLM output: {len(result.issues)}"]
        for issue in result.issues:
            excerpt = f": {_preview(issue.excerpt)}" if issue.excerpt else ""
            lines.append(f"  - {issue.claim_id} ({issue.place_name}) {issue.reason}{excerpt}")
    return "\n".join(lines)


def _preview(text: str) -> str:
    flat = " / ".join(line.strip() for line in text.splitlines() if line.strip())
    return flat if len(flat) <= PREVIEW_CHARS else f"{flat[: PREVIEW_CHARS - 1]}…"


if __name__ == "__main__":
    sys.exit(main())
