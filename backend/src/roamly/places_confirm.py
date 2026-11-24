"""Command-line entry point: let a person confirm places that were not resolved.

Reads a file saved by `roamly-resolve --output`, walks through every place that is
`ambiguous` or `unresolved`, and asks the person to pick a candidate by number, search
again under another name (Nominatim), skip the place, or quit without saving.
Automatically resolved places are not shown. The result is written to a new file
(default: `<input stem>.confirmed.json` next to the input); the input file is replaced
only with `--overwrite` and a final yes/no confirmation.

Exit codes: 0 saved (or nothing to confirm); 1 error; 3 cancelled, nothing saved.
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from roamly.cli import CLIError
from roamly.evaluation.predictions import write_json_file
from roamly.places import (
    PlaceCandidate,
    PlaceResolution,
    PlaceSearch,
    PlaceSearchError,
    confirm_resolution,
    destination_parts,
)
from roamly.places_cli import (
    PlaceResolutionFile,
    create_search_from_env,
    format_candidate,
    format_resolutions,
    read_resolution_file,
)
from roamly.places_nominatim import DEFAULT_CACHE_DIR

EXIT_ERROR = 1
EXIT_CANCELLED = 3

MENU = "Choose a number, s = search another name, k = skip, q = quit without saving: "


class Cancelled(Exception):
    """The person quit; nothing is saved."""


class Session:
    """Terminal interaction for one run: prompts, output, and a lazily created search."""

    def __init__(
        self,
        destination: str | None,
        search_factory: Callable[[], PlaceSearch],
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.destination = destination
        self.parts = destination_parts(destination)
        self._search_factory = search_factory
        self._search: PlaceSearch | None = None
        self._input = input_fn

    def ask(self, prompt: str) -> str:
        try:
            return self._input(prompt).strip()
        except EOFError as exc:  # Ctrl-D or closed stdin counts as quitting
            raise Cancelled from exc

    def confirm(self, prompt: str) -> bool:
        return self.ask(f"{prompt} [y/N]: ").casefold() in ("y", "yes")

    def search(self, name: str) -> list[PlaceCandidate] | None:
        """Search for `name`; None (with a message) when the search is unavailable."""
        try:
            if self._search is None:
                self._search = self._search_factory()
            return self._search.search(name, destination=self.destination)
        except (CLIError, PlaceSearchError) as exc:
            print(f"Search failed: {exc}")
            return None

    def review(self, resolution: PlaceResolution) -> PlaceResolution:
        """Ask about one place; returns it confirmed, or unchanged if skipped."""
        candidates = resolution.candidates
        search_name: str | None = None
        show = True
        while True:
            if show:
                self._show(candidates)
                show = False
            choice = self.ask(MENU).casefold()
            if choice == "q":
                raise Cancelled
            if choice == "k":
                print("Skipped.")
                return resolution
            if choice == "s":
                name = self.ask("Search name (empty to go back): ")
                if not name:
                    continue
                found = self.search(name)
                if found is None:
                    continue
                if not found:
                    print(f"No candidates found for {name!r}.")
                    continue
                candidates, search_name, show = found, name, True
                continue
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                chosen = candidates[int(choice) - 1]
                label = f"{chosen.name or '(no name)'} [{chosen.provider_id}]"
                if self.confirm(f"Use {label} for {resolution.place_name!r}?"):
                    print("Confirmed.")
                    return confirm_resolution(
                        resolution, chosen, candidates=candidates, search_name=search_name
                    )
                continue
            print(f"Invalid choice: {choice!r}")

    def _show(self, candidates: list[PlaceCandidate]) -> None:
        if not candidates:
            print("  No candidates. Use s to search another name.")
        for number, candidate in enumerate(candidates, start=1):
            print("\n".join(format_candidate(candidate, f"{number})", self.parts)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roamly-confirm",
        description="Confirm places that roamly-resolve could not resolve (interactive).",
    )
    parser.add_argument("places", type=Path, help="JSON file saved by roamly-resolve --output")
    parser.add_argument(
        "--output",
        type=Path,
        help="where to save the result (default: <input stem>.confirmed.json next to the input)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file; replacing the input also asks first",
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
    input_fn: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        data = read_resolution_file(args.places)
        output: Path = args.output or args.places.with_name(f"{args.places.stem}.confirmed.json")
        replaces_input = output.resolve() == args.places.resolve()
        # Check before any interaction so no work is lost.
        if output.exists() and not args.overwrite:
            raise CLIError(f"Output file already exists: {output} (use --overwrite)")

        pending = [i for i, r in enumerate(data.resolutions) if r.status != "resolved"]
        if not pending:
            print("Every place is already resolved; nothing to confirm.")
            return 0

        cache_dir = None if args.no_cache else args.cache_dir
        session = Session(data.destination, lambda: search_factory(cache_dir), input_fn)
        resolutions = list(data.resolutions)
        try:
            for count, index in enumerate(pending, start=1):
                resolution = resolutions[index]
                print(
                    f"\n[{count}/{len(pending)}] {resolution.place_name} "
                    f"({resolution.status}: {resolution.reason}; "
                    f"claims: {', '.join(resolution.claim_ids)})"
                )
                resolutions[index] = session.review(resolution)
            if replaces_input and not session.confirm(f"\nReplace the input file {output}?"):
                raise Cancelled
        except (Cancelled, KeyboardInterrupt):
            print("\nCancelled; nothing saved.")
            return EXIT_CANCELLED

        result = data.model_copy(update={"resolutions": resolutions})
        # Validate the whole file again before writing it.
        result = PlaceResolutionFile.model_validate(result.model_dump())
        try:
            write_json_file(output, result.model_dump_json(indent=2), overwrite=args.overwrite)
        except FileExistsError as exc:
            raise CLIError(f"Output file already exists: {output}") from exc
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print("\n" + format_resolutions(result))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
