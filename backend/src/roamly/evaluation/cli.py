"""Command-line entry point for extraction evaluation runs.

- `roamly-eval run DATASET` runs extraction on every case (LLM calls, cached) and saves
  a new run file; it prints the file path on stdout.
- `roamly-eval show RUN_FILE` reads a saved run offline; it never creates an LLM client.

Logs and errors go to stderr.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from roamly.cli import DEFAULT_CACHE_DIR, MAX_SOURCE_CHARS, CLIError, create_client_from_env
from roamly.evaluation.dataset import DatasetError, LoadedDataset, load_dataset
from roamly.evaluation.predictions import EvaluationRun, RunFileError, load_run, save_run
from roamly.evaluation.runner import run_evaluation
from roamly.extraction_cache import ExtractionCache
from roamly.llm.deepseek import DeepSeekClient

DEFAULT_OUTPUT_DIR = Path("eval_runs")

EXIT_ERROR = 1
# The run file was saved, but at least one case failed.
EXIT_CASES_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-eval", description="Run and inspect extraction evaluations."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run", help="run extraction on a dataset and save the predictions (calls the LLM)"
    )
    run.add_argument("dataset", type=Path, help="path to an evaluation dataset JSON file")
    run.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        metavar="ID",
        help="run only this case; repeat to select several (default: all cases)",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for the new run file (default: {DEFAULT_OUTPUT_DIR})",
    )
    run.add_argument(
        "--output", type=Path, help="exact run file path (overrides --output-dir and the name)"
    )
    run.add_argument(
        "--overwrite", action="store_true", help="replace the run file if it already exists"
    )
    run.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"directory for cached LLM output (default: {DEFAULT_CACHE_DIR})",
    )
    run.add_argument("--no-cache", action="store_true", help="do not read or write the cache")

    show = commands.add_parser("show", help="summarise a saved run file (no LLM calls)")
    show.add_argument("run_file", type=Path, help="path to a saved run JSON file")
    return parser


def main(
    argv: Sequence[str] | None = None,
    client_factory: Callable[[], DeepSeekClient] = create_client_from_env,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        if args.command == "show":
            print(format_run(load_run(args.run_file)))
            return 0
        return _run(args, client_factory)
    except (CLIError, DatasetError, RunFileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _run(args: argparse.Namespace, client_factory: Callable[[], DeepSeekClient]) -> int:
    loaded = load_dataset(args.dataset)
    _check_source_lengths(loaded)
    # Fail early, before any LLM call; unknown case ids are rejected by run_evaluation
    # before it calls the client.
    if args.output is not None and args.output.exists() and not args.overwrite:
        raise CLIError(f"Run file already exists: {args.output} (use --overwrite to replace it)")

    client = client_factory()
    cache = None if args.no_cache else ExtractionCache(args.cache_dir, client.settings)
    try:
        run = run_evaluation(
            loaded, client, model_settings=client.settings, cache=cache, case_ids=args.case_ids
        )
    except ValueError as exc:
        raise CLIError(str(exc)) from exc

    path = args.output or args.output_dir / f"{run.run_id}.json"
    save_run(run, path, overwrite=args.overwrite)
    summary = run.summary
    logging.info(
        "Saved run with %d cases (%d succeeded, %d failed)",
        summary.total,
        summary.succeeded,
        summary.failed,
    )
    print(path)
    return EXIT_CASES_FAILED if summary.failed else 0


def _check_source_lengths(loaded: LoadedDataset) -> None:
    for case in loaded.dataset.cases:
        if len(case.source_text) > MAX_SOURCE_CHARS:
            raise CLIError(
                f"Case {case.id} is too long: {len(case.source_text):,} characters "
                f"(maximum {MAX_SOURCE_CHARS:,})"
            )


def format_run(run: EvaluationRun) -> str:
    """A short human-readable summary of a saved run."""
    settings = run.model_settings
    model = (
        f"{settings.get('provider', '?')} / {settings.get('model', '?')}"
        if settings is not None
        else "not recorded"
    )
    predictions = run.predictions
    lines = [
        f"Run:      {run.run_id}",
        f"Dataset:  {run.dataset.name} ({run.dataset.status}, "
        f"{run.dataset.case_count} cases, sha256 {run.dataset.sha256[:12]})",
        f"Prompt:   {run.extraction.prompt_version}",
        f"Code:     {_format_git(run)}",
        f"Model:    {model}",
        f"Started:  {run.started_at.isoformat()}",
        f"Cases:    {run.summary.total} run, {run.summary.succeeded} succeeded, "
        f"{run.summary.failed} failed",
        f"LLM:      {sum(p.llm_calls for p in predictions)} calls, "
        f"{sum(p.cache_hit for p in predictions)} cache hits, "
        f"{sum(p.input_tokens for p in predictions)} input / "
        f"{sum(p.output_tokens for p in predictions)} output tokens",
        "",
    ]
    width = max(len(p.case_id) for p in predictions) if predictions else 0
    for prediction in predictions:
        if prediction.result is not None:
            detail = (
                f"{len(prediction.result.claims)} accepted, "
                f"{len(prediction.result.skipped)} skipped"
            )
        else:
            assert prediction.error is not None
            detail = f"{prediction.error.type}: {prediction.error.message}"
        lines.append(f"{prediction.case_id:<{width}}  {prediction.status:<9}  {detail}")
    return "\n".join(lines)


def _format_git(run: EvaluationRun) -> str:
    commit = run.git_commit[:12] if run.git_commit else "commit not recorded"
    dirty = {True: "uncommitted changes", False: "clean", None: "state not recorded"}
    return f"{commit} ({dirty[run.git_dirty]})"


if __name__ == "__main__":
    sys.exit(main())
