"""Command-line entry point: fetch the official websites of resolved places.

Reads a place resolution file (`roamly-resolve --output` or `roamly-confirm`), fetches
the website of every resolved place, and prints one line per place. With `--output`,
the pages and the place file are saved together as JSON for evidence extraction.
A fetched page only means the page was retrieved; it does not verify any claim.

Exit codes: 0 success; 1 error; 2 at least one website could not be fetched (the
result is still printed and saved).
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from roamly.cli import CLIError
from roamly.evaluation.predictions import write_json_file
from roamly.places_cli import PlaceResolutionFile, read_resolution_file
from roamly.web_fetch import (
    DEFAULT_CACHE_DIR,
    DEFAULT_USER_AGENT,
    USER_AGENT_ENV,
    SourcePage,
    WebsiteFetcher,
    fetch_official_websites,
)

PAGES_FORMAT_VERSION = 1

EXIT_ERROR = 1
# The result was produced, but at least one website could not be fetched.
EXIT_FETCH_FAILED = 2


class SourcePageFile(BaseModel):
    """Saved output of `roamly-fetch`: places (with claims) and their fetched pages."""

    format_version: Literal[1] = PAGES_FORMAT_VERSION
    # The place resolution these pages belong to, copied so the file is self-contained.
    places: PlaceResolutionFile
    pages: list[SourcePage]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-fetch",
        description="Fetch the official websites of resolved places (no claim checking).",
    )
    parser.add_argument(
        "places", type=Path, help="place file from roamly-resolve --output or roamly-confirm"
    )
    parser.add_argument("--output", type=Path, help="also save the pages as JSON here")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output file if it already exists"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for cached pages (default: {DEFAULT_CACHE_DIR})",
    )
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    cache.add_argument("--refresh", action="store_true", help="download again and update the cache")
    return parser


def create_fetcher(cache_dir: Path | None, refresh: bool) -> WebsiteFetcher:
    """A fetcher with the User-Agent from ROAMLY_FETCH_USER_AGENT, if set."""
    user_agent = os.environ.get(USER_AGENT_ENV, "").strip() or DEFAULT_USER_AGENT
    try:
        return WebsiteFetcher(user_agent=user_agent, cache_dir=cache_dir, refresh=refresh)
    except ValueError as exc:
        raise CLIError(f"{USER_AGENT_ENV}: {exc}") from exc


def main(
    argv: Sequence[str] | None = None,
    fetcher_factory: Callable[[Path | None, bool], WebsiteFetcher] = create_fetcher,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        places = read_resolution_file(args.places)
        if args.output is not None and args.output.exists() and not args.overwrite:
            raise CLIError(f"Output file already exists: {args.output} (use --overwrite)")
        fetcher = fetcher_factory(None if args.no_cache else args.cache_dir, args.refresh)
        result = SourcePageFile(
            places=places, pages=fetch_official_websites(places.resolutions, fetcher)
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

    print(format_pages(result.pages))
    if args.output is not None:
        print(f"\nSaved: {args.output}")
    return EXIT_FETCH_FAILED if any(p.status == "failed" for p in result.pages) else 0


def format_pages(pages: list[SourcePage]) -> str:
    """One block per place: status, URLs, title, and text size."""
    counts = {s: sum(p.status == s for p in pages) for s in ("fetched", "skipped", "failed")}
    lines = [
        f"Pages: {len(pages)} ({counts['fetched']} fetched, {counts['skipped']} skipped, "
        f"{counts['failed']} failed)"
    ]
    for page in pages:
        label = page.status if page.reason is None else f"{page.status}: {page.reason}"
        lines += ["", f"[{label}] {page.place_name}  (claims: {', '.join(page.claim_ids)})"]
        if page.requested_url:
            redirect = (
                f" -> {page.final_url}"
                if page.final_url and page.final_url != page.requested_url
                else ""
            )
            lines.append(f"  url:    {page.requested_url}{redirect}")
        if page.error:
            lines.append(f"  error:  {page.error}")
        if page.title:
            lines.append(f"  title:  {page.title}")
        if page.text:
            cached = ", from cache" if page.from_cache else ""
            truncated = ", truncated" if page.text_truncated else ""
            lines.append(f"  text:   {len(page.text):,} characters{truncated}{cached}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
