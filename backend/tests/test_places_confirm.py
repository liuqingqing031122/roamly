from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import pytest

from roamly.cli import CLIError
from roamly.domain import Claim
from roamly.extraction import ExtractionResult
from roamly.places import (
    FakePlaceSearch,
    ManualConfirmation,
    PlaceCandidate,
    PlaceResolution,
    PlaceSearch,
    PlaceSearchError,
)
from roamly.places_cli import PlaceResolutionFile, read_resolution_file
from roamly.places_cli import main as resolve_main
from roamly.places_confirm import EXIT_CANCELLED, EXIT_ERROR, main
from roamly.places_nominatim import NominatimSearch

PORTO = "Porto, Portugal"
FIXTURE = Path(__file__).parent / "fixtures" / "nominatim_search_casa_lume.json"


def candidate(name: str, provider_id: str, **fields: Any) -> PlaceCandidate:
    return PlaceCandidate(
        provider="nominatim",
        provider_id=provider_id,
        name=name,
        display_name=f"{name}, Porto, Portugal",
        lat=41.14,
        lon=-8.61,
        areas=["Porto", "Portugal"],
        **fields,
    )


TOWER = candidate("Clérigos Tower", "way/247168260", category="tourism:attraction")
CAFE = candidate("Café Majestic", "node/1", category="amenity:cafe", website="https://cafe.pt")
CAFE_BUILDING = candidate("Café Majestic", "way/2", category="building:yes")
LELLO = candidate("Livraria Lello", "node/3", category="shop:books")
RIBEIRA = candidate("Ribeira Square", "way/4")


def resolution(name: str, claim_ids: list[str], **fields: Any) -> PlaceResolution:
    defaults: dict[str, Any] = {"status": "unresolved", "reason": "no_match"}
    return PlaceResolution(place_name=name, claim_ids=claim_ids, **(defaults | fields))


def make_file() -> PlaceResolutionFile:
    """The real Porto smoke test: three unresolved places and one resolved."""
    names = ["Lello Bookshop", "Lello Bookshop", "Torre dos Clérigos", "Majestic Café", "Ribeira"]
    claims = [
        Claim(
            id=f"claim-{i}",
            place_name=name,
            claim_type="opening_hours",
            original_excerpt="x",
            normalized_statement="x",
        )
        for i, name in enumerate(names, start=1)
    ]
    return PlaceResolutionFile(
        destination=PORTO,
        extraction=ExtractionResult(claims=claims, skipped=[]),
        resolutions=[
            resolution("Lello Bookshop", ["claim-1", "claim-2"]),
            resolution("Torre dos Clérigos", ["claim-3"], candidates=[TOWER]),
            resolution("Majestic Café", ["claim-4"], candidates=[CAFE, CAFE_BUILDING]),
            resolution(
                "Ribeira",
                ["claim-5"],
                status="resolved",
                reason="unique_match",
                place=RIBEIRA,
                candidates=[RIBEIRA],
            ),
        ],
    )


@pytest.fixture
def places_file(tmp_path: Path) -> Path:
    path = tmp_path / "places.json"
    path.write_text(make_file().model_dump_json(indent=2), encoding="utf-8")
    return path


class Answers:
    """Scripted terminal input; running out of answers behaves like Ctrl-D."""

    def __init__(self, answers: Iterable[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class Factory:
    def __init__(self, results: dict[str, list[PlaceCandidate] | PlaceSearchError]) -> None:
        self.search = FakePlaceSearch(results=results)
        self.created = 0

    def __call__(self, cache_dir: Path | None) -> PlaceSearch:
        self.created += 1
        return self.search


def run(
    places_file: Path,
    answers: Iterable[str],
    results: dict[str, Any] | None = None,
    extra: list[str] | None = None,
) -> tuple[int, Factory, Answers]:
    factory = Factory(results or {})
    terminal = Answers(answers)
    code = main([str(places_file), *(extra or [])], search_factory=factory, input_fn=terminal)
    return code, factory, terminal


def output_of(places_file: Path) -> PlaceResolutionFile:
    return read_resolution_file(places_file.with_name("places.confirmed.json"))


# --- Choosing -----------------------------------------------------------------


def test_choose_existing_candidates_search_and_skip(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = [
        "s", "Livraria Lello", "1", "y",  # Lello: search again, pick, confirm
        "1", "y",  # Torre: pick the only candidate
        "k",  # Majestic: skip
    ]  # fmt: skip
    code, factory, _ = run(places_file, answers, {"Livraria Lello": [LELLO]})

    assert code == 0
    saved = output_of(places_file)
    lello, tower, cafe, ribeira = saved.resolutions
    assert (lello.status, lello.reason, lello.place) == ("resolved", "manually_confirmed", LELLO)
    assert lello.claim_ids == ["claim-1", "claim-2"]  # one confirmation for both claims
    assert lello.confirmation == ManualConfirmation(
        original_reason="no_match", search_name="Livraria Lello"
    )
    assert lello.candidates == [LELLO]
    assert tower.place == TOWER and tower.place_name == "Torre dos Clérigos"
    assert tower.confirmation == ManualConfirmation(original_reason="no_match")
    assert cafe == make_file().resolutions[2]  # skipped: unchanged
    assert ribeira == make_file().resolutions[3]  # automatic: never asked, unchanged
    assert saved.extraction == make_file().extraction
    assert factory.search.calls == [("Livraria Lello", PORTO)]

    out = capsys.readouterr().out
    assert "[1/3] Lello Bookshop (unresolved: no_match; claims: claim-1, claim-2)" in out
    assert "Ribeira" not in out.split("Places:")[0]  # not offered for confirmation
    assert "  1) Café Majestic [nominatim:node/1] (amenity:cafe, area: inside)" in out
    assert "manual:  was no_match, searched 'Livraria Lello'" in out
    assert f"Saved: {places_file.with_name('places.confirmed.json')}" in out


def test_choosing_among_several_candidates(places_file: Path) -> None:
    code, _, _ = run(places_file, ["k", "k", "2", "y"])

    assert code == 0
    cafe = output_of(places_file).resolutions[2]
    assert cafe.place == CAFE_BUILDING
    assert cafe.candidates == [CAFE, CAFE_BUILDING]


def test_declining_the_confirmation_returns_to_the_menu(places_file: Path) -> None:
    code, _, _ = run(places_file, ["k", "1", "n", "1", ""  , "k", "k"])  # fmt: skip

    assert code == 0
    assert output_of(places_file).resolutions[1].status == "unresolved"


@pytest.mark.parametrize("invalid", ["0", "2", "-1", "abc", "", "1.0"])
def test_invalid_choice_asks_again(
    invalid: str, places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = run(places_file, ["k", invalid, "1", "y", "k"])

    assert code == 0
    assert "Invalid choice" in capsys.readouterr().out
    assert output_of(places_file).resolutions[1].place == TOWER


def test_number_without_candidates_is_invalid(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = run(places_file, ["1", "k", "k", "k"])

    out = capsys.readouterr().out
    assert code == 0
    assert "No candidates. Use s to search another name." in out
    assert "Invalid choice: '1'" in out


# --- Searching ----------------------------------------------------------------


def test_search_failure_keeps_the_place_open(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = {"Livraria Lello": PlaceSearchError("Nominatim request timed out")}
    code, _, _ = run(places_file, ["s", "Livraria Lello", "k", "k", "k"], results)

    assert code == 0
    assert "Search failed: Nominatim request timed out" in capsys.readouterr().out
    assert output_of(places_file).resolutions[0].status == "unresolved"


def test_search_without_results_keeps_previous_candidates(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = run(places_file, ["k", "s", "Nowhere", "1", "y", "k"])

    assert code == 0
    assert "No candidates found for 'Nowhere'." in capsys.readouterr().out
    tower = output_of(places_file).resolutions[1]
    assert tower.place == TOWER
    assert tower.confirmation == ManualConfirmation(original_reason="no_match")


def test_empty_search_name_goes_back(places_file: Path) -> None:
    code, factory, _ = run(places_file, ["s", "", "k", "k", "k"])

    assert code == 0
    assert factory.created == 0  # no search client is needed unless a search runs


def test_unavailable_search_is_reported(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_user_agent(cache_dir: Path | None) -> PlaceSearch:
        raise CLIError("Environment variable ROAMLY_NOMINATIM_USER_AGENT is not set")

    code = main(
        [str(places_file)],
        search_factory=no_user_agent,
        input_fn=Answers(["s", "Livraria Lello", "k", "1", "y", "k"]),
    )

    assert code == 0
    assert "Search failed: Environment variable" in capsys.readouterr().out
    assert output_of(places_file).resolutions[1].place == TOWER


def test_search_client_is_created_once(places_file: Path) -> None:
    answers = ["s", "A", "s", "B", "k", "s", "C", "k", "k"]
    code, factory, _ = run(places_file, answers)

    assert code == 0
    assert factory.created == 1
    assert [name for name, _ in factory.search.calls] == ["A", "B", "C"]


def test_search_through_nominatim_adapter(places_file: Path, tmp_path: Path) -> None:
    def factory(cache_dir: Path | None) -> PlaceSearch:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, content=FIXTURE.read_bytes())
        )
        return NominatimSearch("roamly-tests/0.1", cache_dir=cache_dir, transport=transport)

    answers = Answers(["s", "Casa Lume", "2", "y", "k", "k"])
    argv = [str(places_file), "--cache-dir", str(tmp_path / "cache")]
    assert main(argv, search_factory=factory, input_fn=answers) == 0

    lello = output_of(places_file).resolutions[0]
    assert lello.place is not None
    assert lello.place.provider_id == "way/777777"
    assert lello.place.areas == ["Lavra", "Matosinhos", "Porto", "North", "Portugal"]
    assert lello.place.website == "https://lume-lavra.example.pt"


# --- Cancelling ---------------------------------------------------------------


@pytest.mark.parametrize("answers", [["1", "y", "q"], ["k", "1", "y"]], ids=["quit", "eof"])
def test_cancel_saves_nothing(
    answers: list[str], places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = run(places_file, ["s", "Livraria Lello", *answers], {"Livraria Lello": [LELLO]})

    assert code == EXIT_CANCELLED
    assert "Cancelled; nothing saved." in capsys.readouterr().out
    assert not places_file.with_name("places.confirmed.json").exists()


def test_keyboard_interrupt_saves_nothing(places_file: Path) -> None:
    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    code = main([str(places_file)], search_factory=Factory({}), input_fn=interrupt)

    assert code == EXIT_CANCELLED
    assert not places_file.with_name("places.confirmed.json").exists()


# --- Files --------------------------------------------------------------------


def test_custom_output_and_reload_of_confirmed_file(places_file: Path, tmp_path: Path) -> None:
    first = tmp_path / "out" / "confirmed.json"
    code, _, _ = run(places_file, ["k", "1", "y", "k"], extra=["--output", str(first)])
    assert code == 0

    # The confirmed file is a valid input again; only places still open are offered.
    code, _, terminal = run(
        first, ["k", "1", "y"], extra=["--output", str(tmp_path / "second.json")]
    )
    assert code == 0
    saved = read_resolution_file(tmp_path / "second.json")
    assert [r.status for r in saved.resolutions] == ["unresolved", "resolved"] + ["resolved"] * 2
    assert sum(p.startswith("Choose") for p in terminal.prompts) == 2  # Lello, Majestic


def test_input_is_never_modified_by_default(places_file: Path) -> None:
    before = places_file.read_text(encoding="utf-8")

    run(places_file, ["k", "1", "y", "k"])

    assert places_file.read_text(encoding="utf-8") == before


def test_existing_output_is_refused_before_any_question(places_file: Path) -> None:
    existing = places_file.with_name("places.confirmed.json")
    existing.write_text("keep", encoding="utf-8")

    code, _, terminal = run(places_file, ["k", "k", "k"])

    assert code == EXIT_ERROR
    assert terminal.prompts == []
    assert existing.read_text(encoding="utf-8") == "keep"


def test_overwrite_replaces_an_existing_output(places_file: Path) -> None:
    existing = places_file.with_name("places.confirmed.json")
    existing.write_text("old", encoding="utf-8")

    code, _, _ = run(places_file, ["k", "1", "y", "k"], extra=["--overwrite"])

    assert code == 0
    assert output_of(places_file).resolutions[1].place == TOWER


def test_replacing_the_input_needs_overwrite(places_file: Path) -> None:
    code, _, terminal = run(places_file, [], extra=["--output", str(places_file)])

    assert code == EXIT_ERROR
    assert terminal.prompts == []


@pytest.mark.parametrize(("answer", "replaced"), [("y", True), ("n", False), ("", False)])
def test_replacing_the_input_asks_for_final_confirmation(
    answer: str, replaced: bool, places_file: Path
) -> None:
    argv = ["--output", str(places_file), "--overwrite"]
    code, _, terminal = run(places_file, ["k", "1", "y", "k", answer], extra=argv)

    assert "Replace the input file" in terminal.prompts[-1]
    assert code == (0 if replaced else EXIT_CANCELLED)
    tower = read_resolution_file(places_file).resolutions[1]
    assert (tower.status == "resolved") is replaced


def test_nothing_to_confirm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = make_file()
    path = tmp_path / "places.json"
    path.write_text(data.model_copy(update={"resolutions": data.resolutions[3:]}).model_dump_json())

    code, _, terminal = run(path, [])

    assert code == 0
    assert "nothing to confirm" in capsys.readouterr().out
    assert terminal.prompts == []
    assert not path.with_name("places.confirmed.json").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "File not found"), ("{}", "Not a place resolution file")],
    ids=["missing", "wrong_json"],
)
def test_invalid_input_file(
    content: str | None, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "places.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    code, _, _ = run(path, [])

    assert code == EXIT_ERROR
    assert message in capsys.readouterr().err


def test_resolve_output_feeds_confirm(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction.json"
    claims = [
        Claim(
            id="claim-1",
            place_name="Torre dos Clérigos",
            claim_type="opening_hours",
            original_excerpt="x",
            normalized_statement="x",
        )
    ]
    extraction.write_text(ExtractionResult(claims=claims, skipped=[]).model_dump_json())
    places = tmp_path / "places.json"
    resolve_factory = Factory({"Torre dos Clérigos": [TOWER]})
    argv = [str(extraction), "--destination", PORTO, "--output", str(places)]
    assert resolve_main(argv, search_factory=resolve_factory) == 0

    code, _, _ = run(places, ["1", "y"])

    assert code == 0
    [tower] = output_of(places).resolutions
    assert (tower.reason, tower.place) == ("manually_confirmed", TOWER)
