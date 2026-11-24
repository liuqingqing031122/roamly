import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from roamly.domain import Claim
from roamly.extraction import ExtractionResult
from roamly.places import FakePlaceSearch, PlaceCandidate, PlaceSearch, PlaceSearchError
from roamly.places_cli import (
    EXIT_ERROR,
    EXIT_LOOKUP_FAILED,
    OSM_ATTRIBUTION,
    PlaceResolutionFile,
    main,
)
from roamly.places_nominatim import USER_AGENT_ENV, NominatimSearch

PORTO = "Porto, Portugal"
FIXTURE = Path(__file__).parent / "fixtures" / "nominatim_search_casa_lume.json"


def make_candidate(
    name: str, provider_id: str, areas: list[str] | None = None, **fields: Any
) -> PlaceCandidate:
    areas = ["Porto", "Portugal"] if areas is None else areas
    return PlaceCandidate(
        provider="nominatim",
        provider_id=provider_id,
        name=name,
        display_name=f"{name}, {', '.join(areas)}",
        lat=41.15,
        lon=-8.61,
        areas=areas,
        **fields,
    )


TOWER = make_candidate(
    "Torre dos Clérigos", "way/1", category="tourism:attraction", website="https://torre.pt"
)
CAFE_A = make_candidate("Majestic Café", "node/2")
CAFE_B = make_candidate("Majestic Café", "node/3", areas=["Lisbon", "Portugal"])
CAFE_C = make_candidate("Majestic Café", "node/4")
BOOKSHOP = make_candidate("Livraria Lello", "node/5", category="shop:books")

RESULTS: dict[str, list[PlaceCandidate] | PlaceSearchError] = {
    "Torre dos Clérigos": [TOWER],
    "Majestic Café": [CAFE_A, CAFE_B, CAFE_C],
    "Lello Bookshop": [BOOKSHOP],
    "Ponte Bar": PlaceSearchError("Nominatim request timed out"),
}


def make_extraction(*names: str) -> ExtractionResult:
    claims = [
        Claim(
            id=f"claim-{i}",
            place_name=name,
            claim_type="opening_hours",
            original_excerpt=f"{name} opens at 9am.",
            normalized_statement=f"{name} opens at 9am.",
        )
        for i, name in enumerate(names, start=1)
    ]
    return ExtractionResult(claims=claims, skipped=[])


@pytest.fixture
def extraction_file(tmp_path: Path) -> Path:
    path = tmp_path / "extraction.json"
    extraction = make_extraction(
        "Torre dos Clérigos", "Majestic Café", "Lello Bookshop", "the torre dos clérigos"
    )
    path.write_text(extraction.model_dump_json(indent=2), encoding="utf-8")
    return path


class Factory:
    """Search factory that hands out one FakePlaceSearch and records the cache dir."""

    def __init__(self, results: dict[str, list[PlaceCandidate] | PlaceSearchError]) -> None:
        self.search = FakePlaceSearch(results=results)
        self.cache_dirs: list[Path | None] = []

    def __call__(self, cache_dir: Path | None) -> PlaceSearch:
        self.cache_dirs.append(cache_dir)
        return self.search


def run(argv: list[str], results: Any = None) -> tuple[int, Factory]:
    factory = Factory(RESULTS if results is None else results)
    return main(argv, search_factory=factory), factory


# --- Output -------------------------------------------------------------------


def test_prints_every_status_with_candidates(
    extraction_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, factory = run([str(extraction_file), "--destination", PORTO])

    out = capsys.readouterr().out
    assert code == 0
    assert "Destination: Porto, Portugal" in out
    assert "Places:      3 (1 resolved, 1 ambiguous, 1 unresolved)" in out
    # Resolved: grouped claims, chosen candidate marked, address and website shown.
    assert "[resolved] Torre dos Clérigos\n  claims:  claim-1, claim-4\n" in out
    assert "    * Torre dos Clérigos [nominatim:way/1] (tourism:attraction, area: inside)" in out
    assert "      Torre dos Clérigos, Porto, Portugal\n      https://torre.pt" in out
    # Ambiguous: all candidates listed, none chosen, area check visible.
    assert "[ambiguous] Majestic Café\n  claims:  claim-2\n  reason:  multiple_matches" in out
    assert "    - Majestic Café [nominatim:node/3] (unknown category, area: unknown)" in out
    assert "candidates (3):" in out
    # Unresolved because the guide's name differs from the OSM name.
    assert "[unresolved] Lello Bookshop\n  claims:  claim-3\n  reason:  no_match" in out
    assert "    - Livraria Lello [nominatim:node/5]" in out
    assert OSM_ATTRIBUTION in out
    assert len(factory.search.calls) == 3


def test_lookup_failure_is_reported_and_others_continue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extraction.json"
    path.write_text(make_extraction("Ponte Bar", "Torre dos Clérigos").model_dump_json())

    code, _ = run([str(path), "--destination", PORTO])

    out = capsys.readouterr().out
    assert code == EXIT_LOOKUP_FAILED
    assert (
        "[unresolved] Ponte Bar\n  claims:  claim-1\n  reason:  lookup_failed\n"
        "  error:   Nominatim request timed out\n" in out
    )
    assert "[resolved] Torre dos Clérigos" in out


def test_without_destination_nothing_is_resolved(
    extraction_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run([str(extraction_file)])

    out = capsys.readouterr().out
    assert code == 0
    assert "Destination: (none: no place can be resolved)" in out
    assert "[resolved]" not in out


def test_extraction_without_claims(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "extraction.json"
    path.write_text(ExtractionResult(claims=[], skipped=[]).model_dump_json())

    code, factory = run([str(path), "--destination", PORTO])

    out = capsys.readouterr().out
    assert code == 0
    assert "Places:      0 (0 resolved, 0 ambiguous, 0 unresolved)" in out
    assert OSM_ATTRIBUTION not in out
    assert factory.search.calls == []


# --- Saved JSON ---------------------------------------------------------------


def test_output_file_is_complete_and_reloadable(
    extraction_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "places.json"

    code, _ = run([str(extraction_file), "--destination", PORTO, "--output", str(output)])

    assert code == 0
    assert f"Saved: {output}" in capsys.readouterr().out
    saved = PlaceResolutionFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.destination == PORTO
    assert saved.extraction == ExtractionResult.model_validate_json(extraction_file.read_text())
    assert [(r.place_name, r.status) for r in saved.resolutions] == [
        ("Torre dos Clérigos", "resolved"),
        ("Majestic Café", "ambiguous"),
        ("Lello Bookshop", "unresolved"),
    ]
    assert saved.resolutions[0].place == TOWER
    assert saved.resolutions[1].candidates == [CAFE_A, CAFE_B, CAFE_C]
    assert saved.attribution == OSM_ATTRIBUTION


def test_existing_output_is_not_replaced_before_any_lookup(
    extraction_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "places.json"
    output.write_text("keep me", encoding="utf-8")

    code, factory = run([str(extraction_file), "--destination", PORTO, "--output", str(output)])

    assert code == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "keep me"
    assert factory.cache_dirs == []  # no search client was created


def test_overwrite_replaces_output(extraction_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "places.json"
    output.write_text("old", encoding="utf-8")

    code, _ = run(
        [str(extraction_file), "--destination", PORTO, "--output", str(output), "--overwrite"]
    )

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["destination"] == PORTO


def test_lookup_failure_is_saved(tmp_path: Path) -> None:
    path = tmp_path / "extraction.json"
    path.write_text(make_extraction("Ponte Bar").model_dump_json())
    output = tmp_path / "places.json"

    code, _ = run([str(path), "--destination", PORTO, "--output", str(output)])

    assert code == EXIT_LOOKUP_FAILED
    [failed] = PlaceResolutionFile.model_validate_json(output.read_text()).resolutions
    assert (failed.reason, failed.error) == ("lookup_failed", "Nominatim request timed out")


# --- Input and configuration errors -------------------------------------------


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "File not found"),
        ("{not json", "Not an extraction result"),
        ('{"claims": [{"id": "c"}], "skipped": []}', "claims.0.place_name"),
        ('{"format_version": 1}', "Not an extraction result"),
    ],
    ids=["missing", "invalid_json", "bad_claim", "other_json"],
)
def test_invalid_extraction_file(
    content: str | None, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extraction.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    code, factory = run([str(path), "--destination", PORTO])

    assert code == EXIT_ERROR
    assert message in capsys.readouterr().err
    assert factory.cache_dirs == []


def test_cache_options_are_passed_to_the_search(extraction_file: Path, tmp_path: Path) -> None:
    _, default = run([str(extraction_file), "--destination", PORTO])
    _, custom = run([str(extraction_file), "--cache-dir", str(tmp_path / "c")])
    _, disabled = run([str(extraction_file), "--no-cache"])

    assert default.cache_dirs == [Path(".roamly_cache/places")]
    assert custom.cache_dirs == [tmp_path / "c"]
    assert disabled.cache_dirs == [None]


def test_missing_user_agent_is_a_clear_error(
    extraction_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)

    code = main([str(extraction_file), "--destination", PORTO, "--no-cache"])

    assert code == EXIT_ERROR
    assert USER_AGENT_ENV in capsys.readouterr().err


# --- With the real adapter on a mock transport --------------------------------


def test_end_to_end_with_nominatim_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "extraction.json"
    path.write_text(make_extraction("Casa Lume").model_dump_json())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=FIXTURE.read_bytes())

    def factory(cache_dir: Path | None) -> PlaceSearch:
        return NominatimSearch(
            "roamly-tests/0.1", cache_dir=cache_dir, transport=httpx.MockTransport(handler)
        )

    argv = [str(path), "--destination", "Matosinhos", "--cache-dir", str(tmp_path / "cache")]
    assert main(argv, search_factory=factory) == 0
    assert main(argv, search_factory=factory) == 0  # second run is served from the cache

    out = capsys.readouterr().out
    assert out.count("[resolved] Casa Lume") == 2
    assert "* Casa Lume [nominatim:way/777777] (amenity:restaurant, area: inside)" in out
    assert len(requests) == 1
