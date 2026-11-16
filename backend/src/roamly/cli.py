"""Command-line entry point: extract claims from a local .txt travel guide.

Prints an `ExtractionResult` as JSON on stdout; errors and logs go to stderr.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from roamly.extraction_cache import ExtractionCache
from roamly.llm import LLMError
from roamly.llm.deepseek import DeepSeekClient
from roamly.pipeline import extract_claims

PROVIDER_ENV = "ROAMLY_LLM_PROVIDER"
DEFAULT_CACHE_DIR = Path(".roamly_cache/extractions")
# Bounds the cost of a single LLM call.
MAX_SOURCE_CHARS = 30_000

EXIT_ERROR = 1


class CLIError(Exception):
    """An error reported to the user as a single message."""


def create_client_from_env() -> DeepSeekClient:
    provider = os.environ.get(PROVIDER_ENV) or "deepseek"
    if provider != "deepseek":
        raise CLIError(f"Unsupported LLM provider {provider!r} in {PROVIDER_ENV}")
    try:
        return DeepSeekClient.from_env()
    except ValueError as exc:
        raise CLIError(str(exc)) from exc


def read_source_text(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise CLIError(f"Only .txt files are supported: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CLIError(f"File not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CLIError(f"File is not valid UTF-8 text: {path}") from exc
    except OSError as exc:
        raise CLIError(f"Cannot read {path}: {exc.strerror}") from exc
    if not text.strip():
        raise CLIError(f"File is empty: {path}")
    if len(text) > MAX_SOURCE_CHARS:
        raise CLIError(
            f"File is too long: {len(text):,} characters (maximum {MAX_SOURCE_CHARS:,}): {path}"
        )
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-extract",
        description="Extract checkable claims from a travel-guide text file.",
    )
    parser.add_argument("file", type=Path, help="path to a UTF-8 .txt travel guide")
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
        source_text = read_source_text(args.file)
        client = client_factory()
        cache = None if args.no_cache else ExtractionCache(args.cache_dir, client.settings)
        result = extract_claims(source_text, client, cache)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except LLMError as exc:
        print(f"error: LLM call failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
