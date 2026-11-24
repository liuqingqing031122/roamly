"""Command-line entry point: resolve the places of an extraction result.

Reads the JSON that `roamly-extract` prints (an `ExtractionResult`), looks up each
distinct place name with Nominatim, and prints a readable summary on stdout. With
`--output`, the full result is also saved as JSON for later steps. Logs and errors go
to stderr.

Exit codes: 0 success; 1 error (nothing resolved); 2 at least one lookup failed (the
result is still printed and saved; re-running retries only the failed lookups,
because successful ones are cached).
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from roamly.cli import CLIError
from roamly.evaluation.dataset import format_validation_error
from roamly.evaluation.predictions import write_json_file
from roamly.extraction import ExtractionResult
from roamly.places import (
    PlaceCandidate,
    PlaceResolution,
    PlaceSearch,
    area_check,
    destination_parts,
    resolve_places,
)
from roamly.places_nominatim import DEFAULT_CACHE_DIR, NominatimSearch

RESOLUTION_FORMAT_VERSION = 1
OSM_ATTRIBUTION = "Place data © OpenStreetMap contributors, ODbL 1.0"

EXIT_ERROR = 1
# The result was produced, but at least one place lookup failed.
EXIT_LOOKUP_FAILED = 2


class PlaceResolutionFile(BaseModel):
    """Saved output of `roamly-resolve`: the claims and the resolution of their places."""

    format_version: Literal[1] = RESOLUTION_FORMAT_VERSION
    destination: str | None
    # The extraction this resolution belongs to, copied so the file is self-contained.
    extraction: ExtractionResult
    resolutions: list[PlaceResolution]
    attribution: str = OSM_ATTRIBUTION


def create_search_from_env(cache_dir: Path | None) -> PlaceSearch:
    try:
        return NominatimSearch.from_env(cache_dir=cache_dir)
    except ValueError as exc:
        raise CLIError(str(exc)) from exc


def read_extraction(path: Path) -> ExtractionResult:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CLIError(f"File not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CLIError(f"File is not valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise CLIError(f"Cannot read {path}: {exc.strerror}") from exc
    try:
        return ExtractionResult.model_validate_json(content)
    except ValidationError as exc:
        raise CLIError(
            f"Not an extraction result (the JSON printed by roamly-extract): {path}\n"
            f"{format_validation_error(exc)}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-resolve",
        description="Resolve the places mentioned in an extraction result (uses Nominatim).",
    )
    parser.add_argument(
        "extraction", type=Path, help="JSON file written by roamly-extract (ExtractionResult)"
    )
    parser.add_argument(
        "--destination",
        help='where the guide is about, e.g. "Porto, Portugal"; without it nothing is resolved',
    )
    parser.add_argument("--output", type=Path, help="also save the full result as JSON here")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for cached place searches (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="do not read or write the place cache"
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    search_factory: Callable[[Path | None], PlaceSearch] = create_search_from_env,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        extraction = read_extraction(args.extraction)
        # Fail before any lookup rather than after.
        if args.output is not None and args.output.exists() and not args.overwrite:
            raise CLIError(f"Output file already exists: {args.output} (use --overwrite)")
        search = search_factory(None if args.no_cache else args.cache_dir)
        resolutions = resolve_places(extraction, args.destination, search)
        result = PlaceResolutionFile(
            destination=args.destination, extraction=extraction, resolutions=resolutions
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

    print(format_resolutions(result))
    if args.output is not None:
        print(f"\nSaved: {args.output}")
    failed = any(r.reason == "lookup_failed" for r in resolutions)
    return EXIT_LOOKUP_FAILED if failed else 0


def format_resolutions(result: PlaceResolutionFile) -> str:
    """A readable summary: one block per place, with every candidate and its address."""
    resolutions = result.resolutions
    counts = {
        status: sum(r.status == status for r in resolutions)
        for status in ("resolved", "ambiguous", "unresolved")
    }
    lines = [
        f"Destination: {result.destination or '(none: no place can be resolved)'}",
        f"Places:      {len(resolutions)} ({counts['resolved']} resolved, "
        f"{counts['ambiguous']} ambiguous, {counts['unresolved']} unresolved)",
    ]
    parts = destination_parts(result.destination)
    for resolution in resolutions:
        lines += ["", *_format_resolution(resolution, parts)]
    if any(r.candidates for r in resolutions):
        lines += ["", OSM_ATTRIBUTION]
    return "\n".join(lines)


def _format_resolution(resolution: PlaceResolution, parts: Sequence[str]) -> list[str]:
    lines = [
        f"[{resolution.status}] {resolution.place_name}",
        f"  claims:  {', '.join(resolution.claim_ids)}",
        f"  reason:  {resolution.reason}",
    ]
    if resolution.error is not None:
        lines.append(f"  error:   {resolution.error}")
    if not resolution.candidates:
        if resolution.reason != "lookup_failed":
            lines.append("  candidates: none")
        return lines
    lines.append(f"  candidates ({len(resolution.candidates)}):")
    for candidate in resolution.candidates:
        lines += _format_candidate(candidate, resolution.place, parts)
    return lines


def _format_candidate(
    candidate: PlaceCandidate, chosen: PlaceCandidate | None, parts: Sequence[str]
) -> list[str]:
    marker = "*" if candidate == chosen else "-"
    name = candidate.name or "(no name)"
    details = [candidate.category or "unknown category", f"area: {area_check(candidate, parts)}"]
    lines = [
        f"    {marker} {name} [{candidate.provider}:{candidate.provider_id}] "
        f"({', '.join(details)})",
        f"      {candidate.display_name or '(no address)'}",
    ]
    if candidate.website:
        lines.append(f"      {candidate.website}")
    return lines


if __name__ == "__main__":
    sys.exit(main())
