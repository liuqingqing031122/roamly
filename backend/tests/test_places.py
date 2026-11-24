from typing import Any

import pytest
from pydantic import ValidationError

from roamly.domain import Claim
from roamly.extraction import ExtractionResult
from roamly.places import (
    FakePlaceSearch,
    ManualConfirmation,
    PlaceCandidate,
    PlaceResolution,
    PlaceSearch,
    PlaceSearchError,
    area_check,
    confirm_resolution,
    normalize_name,
    resolve_places,
)

PORTO = "Porto, Portugal"


def make_claim(claim_id: str, place_name: str) -> Claim:
    return Claim(
        id=claim_id,
        place_name=place_name,
        claim_type="opening_hours",
        original_excerpt=f"{place_name} opens at 9am.",
        normalized_statement=f"{place_name} opens at 9am.",
    )


def make_extraction(*place_names: str) -> ExtractionResult:
    claims = [make_claim(f"claim-{i}", name) for i, name in enumerate(place_names, start=1)]
    return ExtractionResult(claims=claims, skipped=[])


def make_candidate(
    name: str = "Casa Lume",
    areas: list[str] | None = None,
    provider_id: str = "node/1",
    **overrides: Any,
) -> PlaceCandidate:
    fields: dict[str, Any] = {
        "provider": "nominatim",
        "provider_id": provider_id,
        "name": name,
        "display_name": f"{name}, {', '.join(areas or [])}",
        "lat": 41.15,
        "lon": -8.61,
        "areas": ["Porto", "Portugal"] if areas is None else areas,
    }
    return PlaceCandidate(**(fields | overrides))


def resolve_one(
    results: list[PlaceCandidate] | PlaceSearchError,
    destination: str | None = PORTO,
    place_name: str = "Casa Lume",
) -> PlaceResolution:
    search = FakePlaceSearch(results={place_name: results})
    [resolution] = resolve_places(make_extraction(place_name), destination, search)
    return resolution


# --- Name normalization ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Casa Lume", "casa lume"),
        ("  CASA   lume ", "casa lume"),
        ("The Casa Lume", "casa lume"),
        ("the  casa lume", "casa lume"),
        ("Theatro Lume", "theatro lume"),  # only a separate leading word is removed
        ("Café Lume", "café lume"),  # accents are kept
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


# --- Decision rules ---------------------------------------------------------------


def test_one_same_name_candidate_in_destination_is_resolved() -> None:
    chosen = make_candidate()
    other = make_candidate(name="Casa Lume Hostel", provider_id="node/2")

    resolution = resolve_one([other, chosen])

    assert resolution.status == "resolved"
    assert resolution.reason == "unique_match"
    assert resolution.place == chosen
    assert resolution.candidates == [other, chosen]


def test_name_comparison_uses_normalization() -> None:
    resolution = resolve_one([make_candidate(name="casa  LUME")], place_name="The Casa Lume")

    assert resolution.status == "resolved"
    assert resolution.place_name == "The Casa Lume"


def test_same_name_branches_in_destination_are_ambiguous() -> None:
    branches = [make_candidate(provider_id="node/1"), make_candidate(provider_id="node/2")]

    resolution = resolve_one(branches)

    assert resolution.status == "ambiguous"
    assert resolution.reason == "multiple_matches"
    assert resolution.place is None
    assert resolution.candidates == branches


def test_no_candidates_is_unresolved() -> None:
    resolution = resolve_one([])

    assert (resolution.status, resolution.reason) == ("unresolved", "no_match")
    assert resolution.candidates == []


def test_only_different_names_is_unresolved_but_keeps_candidates() -> None:
    near = [make_candidate(name="Casa Lumen"), make_candidate(name="Lume Bar", provider_id="n/2")]

    resolution = resolve_one(near)

    assert (resolution.status, resolution.reason) == ("unresolved", "no_match")
    assert resolution.candidates == near  # a person may still pick one


# --- Destination check ------------------------------------------------------------


def test_same_name_outside_destination_is_not_resolved() -> None:
    brazil = make_candidate(areas=["Porto Alegre", "Rio Grande do Sul", "Brazil"])

    resolution = resolve_one([brazil])

    assert (resolution.status, resolution.reason) == ("unresolved", "outside_destination")
    assert resolution.candidates == [brazil]


def test_candidate_outside_destination_does_not_compete() -> None:
    inside = make_candidate()
    outside = make_candidate(areas=["Lisbon", "Spain"], provider_id="node/2")

    resolution = resolve_one([outside, inside])

    assert resolution.status == "resolved"
    assert resolution.place == inside


def test_partly_matching_area_needs_confirmation() -> None:
    # Same city name, different country: the destination is not confirmed.
    texas = make_candidate(areas=["Porto", "Texas", "United States"])

    resolution = resolve_one([texas])

    assert (resolution.status, resolution.reason) == ("ambiguous", "area_unverified")
    assert resolution.place is None


def test_candidate_without_areas_needs_confirmation() -> None:
    resolution = resolve_one([make_candidate(areas=[])])

    assert (resolution.status, resolution.reason) == ("ambiguous", "area_unverified")


def test_unverifiable_candidate_competes_with_verified_one() -> None:
    resolution = resolve_one([make_candidate(), make_candidate(areas=[], provider_id="node/2")])

    assert (resolution.status, resolution.reason) == ("ambiguous", "multiple_matches")


@pytest.mark.parametrize("destination", [None, "", " , "])
def test_without_destination_nothing_is_resolved(destination: str | None) -> None:
    search = FakePlaceSearch(results={"Casa Lume": [make_candidate()]})

    [resolution] = resolve_places(make_extraction("Casa Lume"), destination, search)

    assert (resolution.status, resolution.reason) == ("ambiguous", "area_unverified")
    assert search.calls == [("Casa Lume", None)]


@pytest.mark.parametrize(
    ("areas", "destination", "expected"),
    [
        (["Porto", "Portugal"], "Porto, Portugal", "inside"),
        (["  porto ", "PORTUGAL"], "porto,portugal", "inside"),
        (["Porto", "Norte", "Portugal"], "Portugal", "inside"),
        (["Porto Alegre", "Brazil"], "Porto, Portugal", "outside"),
        (["Porto", "Texas"], "Porto, Portugal", "unknown"),
        ([], "Porto, Portugal", "unknown"),
        (["Porto", "Portugal"], None, "unknown"),
    ],
)
def test_area_check(areas: list[str], destination: str | None, expected: str) -> None:
    parts = [] if destination is None else [p.strip().casefold() for p in destination.split(",")]

    assert area_check(make_candidate(areas=areas), parts) == expected


# --- Grouping and failures --------------------------------------------------------


def test_claims_with_the_same_place_share_one_search() -> None:
    search = FakePlaceSearch(
        results={"Casa Lume": [make_candidate()], "Ponte Bar": [make_candidate(name="Ponte Bar")]}
    )
    extraction = make_extraction("Casa Lume", "Ponte Bar", "the casa  lume")

    resolutions = resolve_places(extraction, PORTO, search)

    assert search.calls == [("Casa Lume", PORTO), ("Ponte Bar", PORTO)]
    assert [(r.place_name, r.claim_ids) for r in resolutions] == [
        ("Casa Lume", ["claim-1", "claim-3"]),
        ("Ponte Bar", ["claim-2"]),
    ]
    assert all(r.status == "resolved" for r in resolutions)


def test_failed_search_affects_only_that_place() -> None:
    search = FakePlaceSearch(
        results={
            "Casa Lume": PlaceSearchError("Nominatim timed out"),
            "Ponte Bar": [make_candidate(name="Ponte Bar")],
        }
    )

    failed, ok = resolve_places(make_extraction("Casa Lume", "Ponte Bar"), PORTO, search)

    assert (failed.status, failed.reason) == ("unresolved", "lookup_failed")
    assert failed.error == "Nominatim timed out"
    assert failed.candidates == []
    assert ok.status == "resolved"


def test_unexpected_errors_are_not_hidden() -> None:
    class BrokenSearch:
        def search(self, name: str, *, destination: str | None) -> list[PlaceCandidate]:
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError):
        resolve_places(make_extraction("Casa Lume"), PORTO, BrokenSearch())


def test_no_claims_gives_no_resolutions() -> None:
    search = FakePlaceSearch(results={})

    assert resolve_places(ExtractionResult(claims=[], skipped=[]), PORTO, search) == []
    assert search.calls == []


def test_fake_search_satisfies_the_protocol() -> None:
    search: PlaceSearch = FakePlaceSearch(results={"Casa Lume": [make_candidate()]})

    assert search.search("Casa Lume", destination=PORTO) == [make_candidate()]
    assert search.search("Unknown", destination=None) == []


# --- Schema -----------------------------------------------------------------------


def test_resolution_round_trips_through_json() -> None:
    resolution = resolve_one([make_candidate(category="amenity:cafe", website="https://x.pt")])

    assert PlaceResolution.model_validate_json(resolution.model_dump_json()) == resolution


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "resolved", "reason": "unique_match"},  # resolved without a place
        {"status": "ambiguous", "reason": "no_match"},  # reason of another status
        {"status": "unresolved", "reason": "lookup_failed"},  # failure without error
        {"status": "unresolved", "reason": "no_match", "error": "boom"},  # error, no failure
        {"status": "ambiguous", "reason": "multiple_matches", "place": make_candidate()},
        {"status": "unresolved", "reason": "no_match", "claim_ids": []},
    ],
)
def test_inconsistent_resolution_is_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PlaceResolution.model_validate({"place_name": "Casa Lume", "claim_ids": ["c"]} | fields)


@pytest.mark.parametrize("coordinates", [{"lat": 91}, {"lon": -181}])
def test_candidate_coordinates_are_validated(coordinates: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        make_candidate(**coordinates)


# --- Manual confirmation ----------------------------------------------------------


def test_confirm_resolution_records_a_manual_choice() -> None:
    original = resolve_one([make_candidate(name="Clérigos Tower")], place_name="Torre")
    chosen = original.candidates[0]

    confirmed = confirm_resolution(original, chosen, candidates=original.candidates)

    assert (confirmed.status, confirmed.reason) == ("resolved", "manually_confirmed")
    assert confirmed.place == chosen
    assert (confirmed.place_name, confirmed.claim_ids) == (original.place_name, original.claim_ids)
    assert confirmed.confirmation == ManualConfirmation(original_reason="no_match")


def test_confirm_after_new_search_keeps_search_name_and_new_candidates() -> None:
    failed = resolve_one(PlaceSearchError("timed out"))
    found = [make_candidate(name="Livraria Lello")]

    confirmed = confirm_resolution(failed, found[0], candidates=found, search_name="Livraria Lello")

    assert confirmed.candidates == found
    assert confirmed.error is None
    assert confirmed.confirmation == ManualConfirmation(
        original_reason="lookup_failed", search_name="Livraria Lello"
    )


def test_confirm_rejects_resolved_places_and_foreign_candidates() -> None:
    resolved = resolve_one([make_candidate()])
    with pytest.raises(ValueError, match="already resolved"):
        confirm_resolution(resolved, make_candidate(), candidates=[make_candidate()])

    unresolved = resolve_one([])
    with pytest.raises(ValueError, match="one of the candidates"):
        confirm_resolution(unresolved, make_candidate(), candidates=[])


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "resolved", "reason": "manually_confirmed", "place": make_candidate()},
        {
            "status": "unresolved",
            "reason": "no_match",
            "confirmation": {"original_reason": "no_match"},
        },
    ],
    ids=["manual_without_confirmation", "confirmation_without_manual"],
)
def test_confirmation_must_match_reason(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="confirmation"):
        PlaceResolution.model_validate({"place_name": "Casa Lume", "claim_ids": ["c"]} | fields)


def test_resolutions_without_confirmation_field_still_load() -> None:
    # Files written before manual confirmation existed have no "confirmation" key.
    data = resolve_one([]).model_dump(exclude={"confirmation"})

    assert PlaceResolution.model_validate(data).confirmation is None
