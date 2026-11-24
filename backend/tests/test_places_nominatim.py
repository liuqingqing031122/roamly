import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from roamly.domain import Claim
from roamly.extraction import ExtractionResult
from roamly.places import PlaceSearch, PlaceSearchError, resolve_places
from roamly.places_nominatim import (
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    USER_AGENT_ENV,
    NominatimSearch,
    parse_results,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nominatim_search_casa_lume.json"
USER_AGENT = "roamly-tests/0.1 (test suite)"
PORTO = "Porto, Portugal"


def fixture_body() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class FakeClock:
    """Monotonic time that advances only when `sleep` is called or `advance` is used."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Server:
    """MockTransport handler that records requests and replies with a fixed response."""

    def __init__(self, response: httpx.Response | Exception | None = None) -> None:
        self.response = response
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response or httpx.Response(200, json=fixture_body())


def make_search(
    server: Server,
    cache_dir: Path | None = None,
    clock: FakeClock | None = None,
    **kwargs: Any,
) -> NominatimSearch:
    clock = clock or FakeClock()
    return NominatimSearch(
        USER_AGENT,
        cache_dir=cache_dir,
        transport=httpx.MockTransport(server),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


# --- Request ------------------------------------------------------------------


def test_request_follows_the_api_and_usage_policy() -> None:
    server = Server()

    make_search(server).search("Casa Lume", destination=PORTO)

    [request] = server.requests
    assert request.method == "GET"
    assert str(request.url).startswith(f"{DEFAULT_BASE_URL}/search?")
    assert dict(request.url.params) == {
        "q": "Casa Lume, Porto, Portugal",
        "format": "jsonv2",
        "addressdetails": "1",
        "extratags": "1",
        "limit": "10",
        "accept-language": "en",
    }
    assert request.headers["User-Agent"] == USER_AGENT


def test_query_without_destination_is_the_name_only() -> None:
    server = Server()

    make_search(server).search("  Casa Lume ", destination=None)

    assert server.requests[0].url.params["q"] == "Casa Lume"


@pytest.mark.parametrize("user_agent", ["", "   "])
def test_user_agent_is_required(user_agent: str) -> None:
    with pytest.raises(ValueError, match="User-Agent"):
        NominatimSearch(user_agent)


def test_from_env_reads_user_agent_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_AGENT_ENV, USER_AGENT)
    monkeypatch.setenv(BASE_URL_ENV, "https://nominatim.example.org/")

    search = NominatimSearch.from_env(cache_dir=None)

    assert search.base_url == "https://nominatim.example.org"


def test_from_env_requires_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(USER_AGENT_ENV, raising=False)

    with pytest.raises(ValueError, match=USER_AGENT_ENV):
        NominatimSearch.from_env()


def test_satisfies_the_protocol() -> None:
    search: PlaceSearch = make_search(Server())

    assert len(search.search("Casa Lume", destination=PORTO)) == 2


# --- Response mapping ---------------------------------------------------------


def test_city_result_is_mapped() -> None:
    first = parse_results(fixture_body())[0]

    assert first.provider == "nominatim"
    assert first.provider_id == "node/4242424242"
    assert first.name == "Casa Lume"
    assert first.display_name.startswith("Casa Lume, 12, Rua das Flores")
    assert (first.lat, first.lon) == (41.14961, -8.61099)
    assert first.category == "amenity:cafe"
    assert first.website == "https://casalume.example.pt"
    # Smallest to largest, duplicates ("Porto" city and county) removed; streets,
    # neighbourhoods, postcodes and ISO codes excluded.
    assert first.areas == ["Cedofeita", "Porto", "North", "Portugal"]


def test_village_result_uses_its_own_address_keys() -> None:
    second = parse_results(fixture_body())[1]

    assert second.provider_id == "way/777777"
    assert second.areas == ["Lavra", "Matosinhos", "Porto", "North", "Portugal"]
    assert second.website == "https://lume-lavra.example.pt"  # from contact:website


def test_results_without_an_osm_object_are_skipped() -> None:
    assert [c.provider_id for c in parse_results(fixture_body())] == [
        "node/4242424242",
        "way/777777",
    ]


def test_optional_fields_may_be_missing() -> None:
    [candidate] = parse_results([{"osm_type": "node", "osm_id": 1, "lat": "1.5", "lon": "2.5"}])

    assert candidate.name == ""
    assert candidate.display_name == ""
    assert candidate.areas == []
    assert candidate.category is None
    assert candidate.website is None


def test_empty_response_gives_no_candidates() -> None:
    assert parse_results([]) == []


@pytest.mark.parametrize(
    "body",
    [
        {"error": "Unable to geocode"},
        "not a list",
        [42],
        [{"osm_type": "node", "osm_id": 1, "lon": "2.5"}],  # missing lat
        [{"osm_type": "node", "osm_id": 1, "lat": "north", "lon": "2.5"}],
        [{"osm_type": "node", "osm_id": 1, "lat": "95", "lon": "2.5"}],  # out of range
    ],
    ids=["error_object", "string", "item_not_object", "no_lat", "bad_lat", "lat_range"],
)
def test_invalid_response_raises_place_search_error(body: Any) -> None:
    with pytest.raises(PlaceSearchError):
        parse_results(body)


# --- Errors -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.ReadTimeout("timed out"), "timed out"),
        (httpx.ConnectError("refused"), "request failed"),
        (httpx.Response(429), "HTTP 429 (rate limited or blocked)"),
        (httpx.Response(403), "HTTP 403 (rate limited or blocked)"),
        (httpx.Response(503), "HTTP 503"),
        (httpx.Response(200, text="<html>maintenance</html>"), "not JSON"),
        (httpx.Response(200, json={"error": "bad"}), "not a list"),
    ],
    ids=["timeout", "connect", "429", "403", "503", "html", "error_object"],
)
def test_expected_failures_raise_place_search_error(
    response: httpx.Response | Exception, message: str, tmp_path: Path
) -> None:
    search = make_search(Server(response), cache_dir=tmp_path)

    with pytest.raises(PlaceSearchError, match=re.escape(message)):
        search.search("Casa Lume", destination=PORTO)
    assert list(tmp_path.iterdir()) == []  # failures are never cached


def test_unexpected_errors_are_not_converted() -> None:
    search = make_search(Server(RuntimeError("bug in transport")))

    with pytest.raises(RuntimeError):
        search.search("Casa Lume", destination=PORTO)


def test_failure_is_isolated_in_resolve_places() -> None:
    class OneFailing(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.params["q"].startswith("Ponte Bar"):
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(200, json=fixture_body())

    clock = FakeClock()
    search = NominatimSearch(
        USER_AGENT, cache_dir=None, transport=OneFailing(), clock=clock, sleep=clock.sleep
    )

    first, second = resolve_places(make_extraction("Ponte Bar", "Casa Lume"), "Matosinhos", search)

    assert (first.status, first.reason) == ("unresolved", "lookup_failed")
    assert second.status == "resolved"


# --- Integration with the resolver --------------------------------------------


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


def test_same_name_places_in_the_destination_district_are_ambiguous() -> None:
    # Both results lie in Porto district (county), so "Porto, Portugal" cannot decide.
    [resolution] = resolve_places(make_extraction("Casa Lume"), PORTO, make_search(Server()))

    assert (resolution.status, resolution.reason) == ("ambiguous", "multiple_matches")
    assert len(resolution.candidates) == 2


def test_destination_matching_only_one_candidate_resolves() -> None:
    # The Porto city candidate has none of the destination's areas, so it is outside.
    [resolution] = resolve_places(make_extraction("Casa Lume"), "Matosinhos", make_search(Server()))

    assert resolution.status == "resolved"
    assert resolution.place is not None
    assert resolution.place.provider_id == "way/777777"


def test_shared_country_alone_does_not_rule_a_candidate_out() -> None:
    # "Portugal" matches both, so the Porto city candidate is only partly outside
    # "Matosinhos, Portugal" and still competes (M2.1 rule: partial match is unknown).
    [resolution] = resolve_places(
        make_extraction("Casa Lume"), "Matosinhos, Portugal", make_search(Server())
    )

    assert (resolution.status, resolution.reason) == ("ambiguous", "multiple_matches")


# --- Rate limiting ------------------------------------------------------------


def test_requests_are_at_least_one_second_apart() -> None:
    clock = FakeClock()
    search = make_search(Server(), clock=clock)

    search.search("Casa Lume", destination=PORTO)
    clock.now += 0.25
    search.search("Ponte Bar", destination=PORTO)
    search.search("Lume Bar", destination=PORTO)

    assert clock.sleeps == [pytest.approx(0.75), pytest.approx(1.0)]


def test_no_wait_when_enough_time_has_passed() -> None:
    clock = FakeClock()
    search = make_search(Server(), clock=clock)

    search.search("Casa Lume", destination=PORTO)
    clock.now += 5
    search.search("Ponte Bar", destination=PORTO)

    assert clock.sleeps == []


def test_failed_requests_count_for_the_rate_limit() -> None:
    clock = FakeClock()
    server = Server(httpx.Response(503))
    search = make_search(server, clock=clock)

    for _ in range(2):
        with pytest.raises(PlaceSearchError):
            search.search("Casa Lume", destination=PORTO)

    assert clock.sleeps == [pytest.approx(1.0)]


def test_cache_hits_do_not_wait(tmp_path: Path) -> None:
    clock = FakeClock()
    search = make_search(Server(), cache_dir=tmp_path, clock=clock)

    search.search("Casa Lume", destination=PORTO)
    search.search("Casa Lume", destination=PORTO)

    assert clock.sleeps == []


# --- Cache --------------------------------------------------------------------


def test_cache_avoids_repeated_requests_across_instances(tmp_path: Path) -> None:
    server = Server()
    first = make_search(server, cache_dir=tmp_path).search("Casa Lume", destination=PORTO)

    second = make_search(server, cache_dir=tmp_path).search("Casa Lume", destination=PORTO)

    assert len(server.requests) == 1
    assert second == first


def test_empty_results_are_cached(tmp_path: Path) -> None:
    server = Server(httpx.Response(200, json=[]))
    search = make_search(server, cache_dir=tmp_path)

    assert search.search("Nowhere", destination=PORTO) == []
    assert search.search("Nowhere", destination=PORTO) == []
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"name": "Casa Lume", "destination": PORTO}, {"name": "Ponte Bar", "destination": PORTO}),
        ({"name": "Casa Lume", "destination": PORTO}, {"name": "Casa Lume", "destination": None}),
        ({"name": "Casa Lume", "destination": PORTO, "language": "pt"}, {}),
        ({"name": "Casa Lume", "destination": PORTO, "limit": 5}, {}),
        ({"name": "Casa Lume", "destination": PORTO, "base_url": "https://other.example"}, {}),
    ],
    ids=["name", "destination", "language", "limit", "base_url"],
)
def test_cache_key_includes_query_parameters(
    first: dict[str, Any], second: dict[str, Any], tmp_path: Path
) -> None:
    server = Server()
    defaults = {"name": "Casa Lume", "destination": PORTO}

    for call in (first, second or defaults):
        options = dict(call)
        name, destination = options.pop("name"), options.pop("destination")
        make_search(server, cache_dir=tmp_path, **options).search(name, destination=destination)

    assert len(server.requests) == 2


def test_cache_entry_stores_raw_response_and_key_fields(tmp_path: Path) -> None:
    make_search(Server(), cache_dir=tmp_path).search("Casa Lume", destination=PORTO)

    [path] = tmp_path.iterdir()
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["response"] == fixture_body()
    assert entry["key_fields"]["params"]["q"] == "Casa Lume, Porto, Portugal"
    assert entry["key_fields"]["base_url"] == DEFAULT_BASE_URL
    assert USER_AGENT not in path.read_text(encoding="utf-8")


def test_unreadable_cache_entry_is_refetched(tmp_path: Path) -> None:
    server = Server()
    make_search(server, cache_dir=tmp_path).search("Casa Lume", destination=PORTO)
    [path] = tmp_path.iterdir()
    path.write_text("{not json", encoding="utf-8")

    result = make_search(server, cache_dir=tmp_path).search("Casa Lume", destination=PORTO)

    assert len(server.requests) == 2
    assert len(result) == 2


def test_cache_can_be_disabled() -> None:
    server = Server()
    search = make_search(server, cache_dir=None)

    search.search("Casa Lume", destination=PORTO)
    search.search("Casa Lume", destination=PORTO)

    assert len(server.requests) == 2
