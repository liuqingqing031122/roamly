from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from roamly.domain import Claim
from roamly.extraction import ExtractionResult
from roamly.places import PlaceCandidate, PlaceResolution
from roamly.places_cli import PlaceResolutionFile
from roamly.web_fetch import DEFAULT_USER_AGENT, USER_AGENT_ENV, WebsiteFetcher
from roamly.web_fetch_cli import (
    EXIT_ERROR,
    EXIT_FETCH_FAILED,
    SourcePageFile,
    create_fetcher,
    main,
)

HTML = (Path(__file__).parent / "fixtures" / "pages" / "official_site.html").read_bytes()


def place(provider_id: str, website: str | None) -> PlaceCandidate:
    return PlaceCandidate(
        provider="nominatim",
        provider_id=provider_id,
        name="x",
        display_name="x, Porto, Portugal",
        lat=41.1,
        lon=-8.6,
        website=website,
    )


def confirmed(name: str, claim_ids: list[str], candidate: PlaceCandidate) -> PlaceResolution:
    return PlaceResolution(
        place_name=name,
        claim_ids=claim_ids,
        status="resolved",
        reason="manually_confirmed",
        place=candidate,
        candidates=[candidate],
        confirmation={"original_reason": "no_match"},
    )


def make_places(*resolutions: PlaceResolution) -> PlaceResolutionFile:
    claims = [
        Claim(
            id=claim_id,
            place_name=r.place_name,
            claim_type="opening_hours",
            original_excerpt="x",
            normalized_statement="x",
        )
        for r in resolutions
        for claim_id in r.claim_ids
    ]
    return PlaceResolutionFile(
        destination="Porto, Portugal",
        extraction=ExtractionResult(claims=claims, skipped=[]),
        resolutions=list(resolutions),
    )


PORTO = make_places(
    confirmed("Lello Bookshop", ["claim-1", "claim-2"], place("node/1", "https://lello.example/")),
    confirmed("Torre dos Clérigos", ["claim-3"], place("way/2", "https://torre.example/")),
    confirmed("Majestic Café", ["claim-4", "claim-5"], place("node/3", "www.majestic.example")),
)


@pytest.fixture
def places_file(tmp_path: Path) -> Path:
    path = tmp_path / "places.confirmed.json"
    path.write_text(PORTO.model_dump_json(indent=2), encoding="utf-8")
    return path


class Factory:
    def __init__(self, routes: dict[str, httpx.Response | Exception] | None = None) -> None:
        self.routes = routes or {}
        self.requests: list[httpx.Request] = []
        self.calls: list[tuple[Path | None, bool]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        default = (
            httpx.Response(404)  # no robots.txt: no rules
            if request.url.path == "/robots.txt"
            else httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)
        )
        response = self.routes.get(str(request.url), default)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def pages(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path != "/robots.txt"]

    def __call__(self, cache_dir: Path | None, refresh: bool) -> WebsiteFetcher:
        self.calls.append((cache_dir, refresh))
        return WebsiteFetcher(
            cache_dir=cache_dir,
            refresh=refresh,
            transport=httpx.MockTransport(self.handler),
            resolve=lambda host, port: ["93.184.216.34"],
            now=lambda: datetime(2026, 11, 24, tzinfo=UTC),
        )


def run(argv: list[str], factory: Factory | None = None) -> tuple[int, Factory]:
    factory = factory or Factory()
    return main(argv, fetcher_factory=factory), factory


def test_fetches_every_resolved_place(
    places_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, factory = run([str(places_file), "--no-cache"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Pages: 3 (3 fetched, 0 skipped, 0 failed)" in out
    assert "[fetched] Lello Bookshop  (claims: claim-1, claim-2)" in out
    assert "  url:    https://www.majestic.example/" in out
    assert "  title:  Casa Lume & Bar | Official Site" in out
    assert len(factory.pages) == 3


def test_output_file_is_self_contained_and_reloadable(places_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "runs" / "pages.json"

    code, _ = run([str(places_file), "--no-cache", "--output", str(output)])

    assert code == 0
    saved = SourcePageFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.places == PORTO
    assert [p.place_name for p in saved.pages] == [
        "Lello Bookshop",
        "Torre dos Clérigos",
        "Majestic Café",
    ]
    assert all(p.source_type == "official_website" for p in saved.pages)
    assert "Closed on Sundays and public holidays." in saved.pages[0].text


def test_one_failed_website_does_not_stop_the_others(
    places_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = Factory({"https://torre.example/": httpx.ReadTimeout("timed out")})
    output = tmp_path / "pages.json"

    code, _ = run([str(places_file), "--no-cache", "--output", str(output)], factory)

    out = capsys.readouterr().out
    assert code == EXIT_FETCH_FAILED
    assert "[failed: timeout] Torre dos Clérigos" in out
    assert "  error:  Request timed out" in out
    saved = SourcePageFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert [p.status for p in saved.pages] == ["fetched", "failed", "fetched"]


def test_skipped_places_do_not_count_as_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "places.json"
    unresolved = PlaceResolution(
        place_name="Somewhere", claim_ids=["claim-2"], status="unresolved", reason="no_match"
    )
    no_site = confirmed("Quiet Bar", ["claim-1"], place("node/9", None))
    path.write_text(make_places(no_site, unresolved).model_dump_json())

    code, factory = run([str(path), "--no-cache"])

    out = capsys.readouterr().out
    assert code == 0
    assert "[skipped: no_website] Quiet Bar" in out
    assert "[skipped: place_not_resolved] Somewhere" in out
    assert factory.pages == []


def test_redirect_is_shown(places_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    factory = Factory({"https://lello.example/": httpx.Response(301, headers={"location": "/en/"})})

    run([str(places_file), "--no-cache"], factory)

    assert (
        "  url:    https://lello.example/ -> https://lello.example/en/" in capsys.readouterr().out
    )


def test_cache_options(
    places_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "cache"
    factory = Factory()

    run([str(places_file), "--cache-dir", str(cache)], factory)
    run([str(places_file), "--cache-dir", str(cache)], factory)
    assert "from cache" in capsys.readouterr().out
    assert len(factory.pages) == 3  # second run served from the cache

    run([str(places_file), "--cache-dir", str(cache), "--refresh"], factory)
    assert len(factory.pages) == 6
    run([str(places_file), "--no-cache"], factory)
    assert factory.calls[-2:] == [(cache, True), (None, False)]


def test_no_cache_and_refresh_are_exclusive(places_file: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(places_file), "--no-cache", "--refresh"], fetcher_factory=Factory())


def test_existing_output_is_refused_before_fetching(
    places_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "pages.json"
    output.write_text("keep", encoding="utf-8")

    code, factory = run([str(places_file), "--output", str(output)])

    assert code == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert factory.calls == []
    assert output.read_text(encoding="utf-8") == "keep"


def test_overwrite(places_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "pages.json"
    output.write_text("old", encoding="utf-8")

    code, _ = run([str(places_file), "--no-cache", "--output", str(output), "--overwrite"])

    assert code == 0
    assert SourcePageFile.model_validate_json(output.read_text(encoding="utf-8")).pages


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "File not found"), ('{"claims": [], "skipped": []}', "Not a place resolution file")],
    ids=["missing", "extraction_instead_of_places"],
)
def test_invalid_input(
    content: str | None, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "places.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    code, factory = run([str(path)])

    assert code == EXIT_ERROR
    assert message in capsys.readouterr().err
    assert factory.calls == []


def test_user_agent_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_AGENT_ENV, "roamly-dev/0.2 (contact: someone)")

    assert create_fetcher(None, False).user_agent == "roamly-dev/0.2 (contact: someone)"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_default_user_agent(value: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    if value is None:
        monkeypatch.delenv(USER_AGENT_ENV, raising=False)
    else:
        monkeypatch.setenv(USER_AGENT_ENV, value)

    assert create_fetcher(None, False).user_agent == DEFAULT_USER_AGENT


def test_invalid_user_agent_is_a_clear_error(
    places_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(USER_AGENT_ENV, "/0.1 no product")

    assert main([str(places_file), "--no-cache"]) == EXIT_ERROR
    assert USER_AGENT_ENV in capsys.readouterr().err


def test_robots_disallowed_is_skipped_not_failed(
    places_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = Factory(
        {"https://lello.example/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /")}
    )

    code, _ = run([str(places_file), "--no-cache"], factory)

    out = capsys.readouterr().out
    assert code == 0
    assert "[skipped: robots_disallowed] Lello Bookshop" in out
    assert "Pages: 3 (2 fetched, 1 skipped, 0 failed)" in out
