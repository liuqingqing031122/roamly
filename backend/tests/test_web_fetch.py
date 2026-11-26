import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from roamly.places import PlaceCandidate, PlaceResolution
from roamly.web_fetch import (
    MAX_REDIRECTS,
    MAX_TEXT_CHARS,
    PageFetch,
    SourcePage,
    WebsiteFetcher,
    _FetchFailure,
    check_public_host,
    extract_text,
    fetch_official_websites,
    normalize_url,
    parse_robots,
    product_token,
)

PAGES = Path(__file__).parent / "fixtures" / "pages"
OFFICIAL_HTML = (PAGES / "official_site.html").read_text(encoding="utf-8")
JS_ONLY_HTML = (PAGES / "javascript_only.html").read_text(encoding="utf-8")
PUBLIC_IP = "93.184.216.34"
FIXED_TIME = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)

Handler = Callable[[httpx.Request], httpx.Response]


def html_response(html: str = OFFICIAL_HTML, **kwargs: Any) -> httpx.Response:
    headers = {"content-type": "text/html; charset=utf-8", **kwargs.pop("headers", {})}
    return httpx.Response(200, headers=headers, content=html.encode("utf-8"), **kwargs)


class Site:
    """MockTransport handler: routes by URL and records every request."""

    def __init__(self, routes: dict[str, httpx.Response | Exception] | None = None) -> None:
        self.routes = routes or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        # Without a route, robots.txt is missing (no rules) and pages are the fixture.
        default = httpx.Response(404) if request.url.path == "/robots.txt" else html_response()
        response = self.routes.get(str(request.url), default)
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def pages(self) -> list[httpx.Request]:
        """Requests other than robots.txt."""
        return [r for r in self.requests if r.url.path != "/robots.txt"]


def fake_resolver(addresses: dict[str, list[str]] | None = None) -> Callable[[str, int], list[str]]:
    def resolve(host: str, port: int) -> list[str]:
        if addresses and host in addresses:
            return addresses[host]
        if host.endswith(".invalid"):
            raise OSError("Name or service not known")
        return [PUBLIC_IP]

    return resolve


def make_fetcher(
    site: Site | None = None,
    *,
    addresses: dict[str, list[str]] | None = None,
    cache_dir: Path | None = None,
    **kwargs: Any,
) -> WebsiteFetcher:
    return WebsiteFetcher(
        cache_dir=cache_dir,
        transport=httpx.MockTransport(site or Site()),
        resolve=fake_resolver(addresses),
        now=lambda: FIXED_TIME,
        **kwargs,
    )


# --- URL normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.example.com/", "https://www.example.com/"),
        ("https://www.example.com", "https://www.example.com/"),
        ("www.cafemajestic.com", "https://www.cafemajestic.com/"),
        ("  example.pt/visit?lang=en#hours ", "https://example.pt/visit?lang=en"),
        ("HTTP://Example.COM/Path", "http://example.com/Path"),
        ("https://example.com:443/", "https://example.com:443/"),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "invalid_url"),
        ("not a url", "invalid_url"),
        ("https://", "invalid_url"),
        ("https://example.com:99999/", "invalid_url"),
        ("javascript:alert(1)", "invalid_url"),
        ("ftp://example.com/file", "unsafe_url"),
        ("file:///etc/passwd", "unsafe_url"),
        ("https://user:secret@example.com/", "unsafe_url"),
        ("https://example.com:8080/", "unsafe_url"),
    ],
)
def test_normalize_url_rejects(raw: str, reason: str) -> None:
    with pytest.raises(_FetchFailure) as error:
        normalize_url(raw)
    assert error.value.reason == reason


def test_redirect_targets_must_be_absolute_http() -> None:
    with pytest.raises(_FetchFailure) as error:
        normalize_url("mailto:info@example.com", assume_https=False)
    assert error.value.reason == "unsafe_url"


# --- Host safety --------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://admin.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "http://224.0.0.1/",
    ],
)
def test_non_public_addresses_are_unsafe(url: str) -> None:
    with pytest.raises(_FetchFailure) as error:
        check_public_host(url, fake_resolver())
    assert error.value.reason == "unsafe_url"


@pytest.mark.parametrize(
    "addresses",
    [["10.1.2.3"], [PUBLIC_IP, "127.0.0.1"]],
    ids=["private_only", "one_private_among_public"],
)
def test_names_resolving_to_private_addresses_are_unsafe(addresses: list[str]) -> None:
    resolve = fake_resolver({"intranet.example.com": addresses})

    with pytest.raises(_FetchFailure) as error:
        check_public_host("https://intranet.example.com/", resolve)
    assert error.value.reason == "unsafe_url"


def test_public_host_passes() -> None:
    check_public_host("https://www.example.com/", fake_resolver())
    check_public_host(f"http://{PUBLIC_IP}/", fake_resolver())


def test_unresolvable_host_is_a_network_error() -> None:
    with pytest.raises(_FetchFailure) as error:
        check_public_host("https://nowhere.invalid/", fake_resolver())
    assert error.value.reason == "network_error"


# --- Text extraction ----------------------------------------------------------------


def test_extract_text_keeps_visible_text_in_blocks() -> None:
    title, text = extract_text(OFFICIAL_HTML)

    assert title == "Casa Lume & Bar | Official Site"
    assert text.splitlines() == [
        "Home",
        "Visit",
        "Welcome to Casa Lume",
        "We serve breakfast and lunch in the heart of Porto.",
        "Opening hours",
        "Monday – Friday",
        "9:00 – 19:00",
        "Saturday",
        "10:00 – 14:00",
        "Closed on Sundays and public holidays.",
        "Reservations are not required.",
        "Rua das Flores 12, 4050-262 Porto",
    ]


@pytest.mark.parametrize(
    "hidden",
    ["tracking", "color: red", "tail script", "enable JavaScript", "icon label", "template"],
)
def test_extract_text_drops_non_visible_content(hidden: str) -> None:
    assert hidden not in extract_text(OFFICIAL_HTML)[1]


@pytest.mark.parametrize(
    "html",
    ["", "<<<>>>", "<html><body><p>Unclosed", "\x00\x01binary\xff", "plain text, no tags"],
    ids=["empty", "garbage", "unclosed", "binary", "no_tags"],
)
def test_extract_text_tolerates_invalid_html(html: str) -> None:
    title, text = extract_text(html)

    assert title is None
    assert "<p>" not in text


# --- Fetching -----------------------------------------------------------------------


def test_fetch_success() -> None:
    site = Site()

    page = make_fetcher(site).fetch("www.casalume.example")

    assert page.status == "fetched"
    assert page.reason is None
    assert page.requested_url == page.final_url == "https://www.casalume.example/"
    assert page.http_status == 200
    assert page.title == "Casa Lume & Bar | Official Site"
    assert "Closed on Sundays and public holidays." in page.text
    assert page.fetched_at == FIXED_TIME
    assert page.from_cache is False
    [request] = site.pages
    assert request.headers["User-Agent"].startswith("Roamly/")


def test_redirects_are_followed_and_recorded() -> None:
    site = Site(
        {
            "http://casalume.example/": httpx.Response(
                301, headers={"location": "https://www.casalume.example/"}
            ),
            "https://www.casalume.example/": httpx.Response(302, headers={"location": "/en/home"}),
        }
    )

    page = make_fetcher(site).fetch("http://casalume.example")

    assert page.status == "fetched"
    assert page.requested_url == "http://casalume.example/"
    assert page.final_url == "https://www.casalume.example/en/home"
    assert len(site.pages) == 3


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://intranet.example.com/",
        "file:///etc/passwd",
        "https://casalume.example:2375/",
    ],
)
def test_unsafe_redirect_targets_are_not_requested(location: str) -> None:
    site = Site({"https://casalume.example/": httpx.Response(302, headers={"location": location})})

    page = make_fetcher(site, addresses={"intranet.example.com": ["10.0.0.1"]}).fetch(
        "https://casalume.example/"
    )

    assert (page.status, page.reason) == ("failed", "unsafe_url")
    assert len(site.pages) == 1  # the unsafe target was never contacted


def test_unsafe_url_is_never_requested() -> None:
    site = Site()

    page = make_fetcher(site, addresses={"casalume.example": ["192.168.0.10"]}).fetch(
        "https://casalume.example/"
    )

    assert (page.status, page.reason) == ("failed", "unsafe_url")
    assert site.requests == []  # not even robots.txt


def test_too_many_redirects() -> None:
    loop = {
        "https://loop.example/": httpx.Response(302, headers={"location": "https://loop.example/"})
    }
    site = Site(loop)

    page = make_fetcher(site).fetch("https://loop.example/")

    assert (page.status, page.reason) == ("failed", "too_many_redirects")
    assert len(site.pages) == MAX_REDIRECTS + 1


@pytest.mark.parametrize(
    ("response", "reason", "http_status"),
    [
        (httpx.Response(404), "http_error", 404),
        (httpx.Response(403), "http_error", 403),
        (httpx.Response(500), "http_error", 500),
        (httpx.Response(204), "http_error", 204),
        (httpx.Response(302), "http_error", 302),  # redirect without a location
        (httpx.Response(200, headers={"content-type": "application/pdf"}), "not_html", 200),
        (httpx.Response(200, headers={"content-type": "image/png"}), "not_html", 200),
        (httpx.ReadTimeout("timed out"), "timeout", None),
        (httpx.ConnectTimeout("timed out"), "timeout", None),
        (httpx.ConnectError("refused"), "network_error", None),
    ],
)
def test_expected_failures(
    response: httpx.Response | Exception, reason: str, http_status: int | None
) -> None:
    site = Site({"https://casalume.example/": response})

    page = make_fetcher(site).fetch("https://casalume.example/")

    assert (page.status, page.reason, page.http_status) == ("failed", reason, http_status)
    assert page.error
    assert page.text == ""


def test_page_without_readable_text() -> None:
    site = Site({"https://app.example/": html_response(JS_ONLY_HTML)})

    page = make_fetcher(site).fetch("https://app.example/")

    assert (page.status, page.reason) == ("failed", "no_text")
    assert page.title == "Loading…"
    assert page.final_url == "https://app.example/"


def test_missing_content_type_is_treated_as_html() -> None:
    site = Site({"https://bare.example/": httpx.Response(200, content=b"<p>Open daily.</p>")})

    page = make_fetcher(site).fetch("https://bare.example/")

    assert (page.status, page.text) == ("fetched", "Open daily.")


@pytest.mark.parametrize("declared", [True, False], ids=["content_length", "streamed"])
def test_too_large_pages_are_rejected(declared: bool) -> None:
    body = b"<p>" + b"x" * 5_000 + b"</p>"
    headers = {"content-type": "text/html"}
    if not declared:
        # Without a length header the limit is enforced while reading.
        response = httpx.Response(200, headers=headers, content=iter([body[:2500], body[2500:]]))
    else:
        response = httpx.Response(200, headers=headers, content=body)
    site = Site({"https://big.example/": response})

    page = make_fetcher(site, max_bytes=1_000).fetch("https://big.example/")

    assert (page.status, page.reason) == ("failed", "too_large")


def test_slow_download_hits_the_total_timeout() -> None:
    ticks = iter([0.0, 5.0, 50.0, 100.0])
    body = iter([b"<p>part one", b" part two</p>"])
    site = Site({"https://slow.example/": httpx.Response(200, content=body)})

    page = make_fetcher(site, clock=lambda: next(ticks), total_timeout=30).fetch(
        "https://slow.example/"
    )

    assert (page.status, page.reason) == ("failed", "timeout")


def test_charset_from_header_is_used() -> None:
    html = "<p>Café aberto às 9h</p>".encode("latin-1")
    response = httpx.Response(
        200, headers={"content-type": "text/html; charset=iso-8859-1"}, content=html
    )

    page = make_fetcher(Site({"https://latin.example/": response})).fetch("https://latin.example/")

    assert page.text == "Café aberto às 9h"


def test_unknown_charset_falls_back_to_utf8() -> None:
    response = httpx.Response(
        200, headers={"content-type": "text/html; charset=x-made-up"}, content="<p>Olá</p>".encode()
    )

    page = make_fetcher(Site({"https://odd.example/": response})).fetch("https://odd.example/")

    assert page.text == "Olá"


def test_long_text_is_truncated() -> None:
    html = "<p>" + "word " * (MAX_TEXT_CHARS // 4) + "</p>"

    page = make_fetcher(Site({"https://long.example/": html_response(html)})).fetch(
        "https://long.example/"
    )

    assert page.status == "fetched"
    assert len(page.text) == MAX_TEXT_CHARS
    assert page.text_truncated is True


def test_unexpected_errors_are_not_hidden() -> None:
    site = Site({"https://bug.example/": RuntimeError("bug")})

    with pytest.raises(RuntimeError):
        make_fetcher(site).fetch("https://bug.example/")


# --- Cache --------------------------------------------------------------------------


def test_cache_avoids_a_second_download_and_keeps_fetch_time(tmp_path: Path) -> None:
    site = Site()
    first = make_fetcher(site, cache_dir=tmp_path).fetch("www.casalume.example")

    later = WebsiteFetcher(
        cache_dir=tmp_path,
        transport=httpx.MockTransport(site),
        resolve=fake_resolver(),
        now=lambda: datetime(2027, 1, 1, tzinfo=UTC),
    )
    second = later.fetch("https://www.casalume.example/")

    assert len(site.pages) == 1
    assert second.from_cache is True
    assert second.fetched_at == FIXED_TIME == first.fetched_at
    assert second.text == first.text


def test_refresh_downloads_again_and_updates_the_cache(tmp_path: Path) -> None:
    site = Site()
    make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")
    site.routes["https://casalume.example/"] = html_response("<p>New hours: 8:00</p>")

    refreshed = make_fetcher(site, cache_dir=tmp_path, refresh=True).fetch(
        "https://casalume.example/"
    )
    cached = make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")

    assert len(site.pages) == 2
    assert refreshed.from_cache is False
    assert cached.text == "New hours: 8:00" and cached.from_cache is True


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.ReadTimeout("timed out"),
        httpx.Response(200, headers={"content-type": "application/pdf"}),
    ],
    ids=["http_error", "timeout", "not_html"],
)
def test_failures_are_not_cached(response: httpx.Response | Exception, tmp_path: Path) -> None:
    make_fetcher(Site({"https://casalume.example/": response}), cache_dir=tmp_path).fetch(
        "https://casalume.example/"
    )

    assert list(tmp_path.iterdir()) == []


def test_unsafe_urls_are_checked_before_the_cache(tmp_path: Path) -> None:
    page = make_fetcher(cache_dir=tmp_path).fetch("ftp://casalume.example/")

    assert page.reason == "unsafe_url"
    assert list(tmp_path.iterdir()) == []


def test_cache_entry_contents(tmp_path: Path) -> None:
    make_fetcher(cache_dir=tmp_path).fetch("www.casalume.example")

    [path] = tmp_path.iterdir()
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert entry["url"] == entry["final_url"] == "https://www.casalume.example/"
    assert entry["fetched_at"] == FIXED_TIME.isoformat()
    assert entry["html"] == OFFICIAL_HTML


def test_unreadable_cache_entry_is_downloaded_again(tmp_path: Path) -> None:
    site = Site()
    make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")
    [path] = tmp_path.iterdir()
    path.write_text('{"url": "x"}', encoding="utf-8")

    page = make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")

    assert page.status == "fetched" and page.from_cache is False
    assert len(site.pages) == 2


# --- Places -------------------------------------------------------------------------


def candidate(provider_id: str, website: str | None) -> PlaceCandidate:
    return PlaceCandidate(
        provider="nominatim",
        provider_id=provider_id,
        name="x",
        display_name="x, Porto, Portugal",
        lat=41.1,
        lon=-8.6,
        website=website,
    )


def resolved(
    name: str, claim_ids: list[str], website: str | None, **fields: Any
) -> PlaceResolution:
    place = candidate(f"node/{len(name)}", website)
    defaults: dict[str, Any] = {"status": "resolved", "reason": "unique_match", "place": place}
    return PlaceResolution(
        place_name=name, claim_ids=claim_ids, candidates=[place], **(defaults | fields)
    )


def test_fetch_official_websites_isolates_each_place() -> None:
    site = Site({"https://down.example/": httpx.ConnectError("refused")})
    resolutions = [
        resolved("Lello Bookshop", ["claim-1", "claim-2"], "https://lello.example/"),
        resolved("Down Café", ["claim-3"], "https://down.example/"),
        resolved("No Site Bar", ["claim-4"], None),
        PlaceResolution(
            place_name="Unknown Place",
            claim_ids=["claim-5"],
            status="unresolved",
            reason="no_match",
        ),
        resolved("Majestic Café", ["claim-6"], "www.majestic.example"),
    ]

    pages = fetch_official_websites(resolutions, make_fetcher(site))

    assert [(p.place_name, p.status, p.reason) for p in pages] == [
        ("Lello Bookshop", "fetched", None),
        ("Down Café", "failed", "network_error"),
        ("No Site Bar", "skipped", "no_website"),
        ("Unknown Place", "skipped", "place_not_resolved"),
        ("Majestic Café", "fetched", None),
    ]
    assert pages[0].claim_ids == ["claim-1", "claim-2"]
    assert pages[0].place_id == "nominatim:node/14"
    assert pages[0].source_type == "official_website"
    assert pages[3].place_id is None and pages[3].website is None
    assert pages[4].website == "www.majestic.example"
    assert pages[4].requested_url == "https://www.majestic.example/"


def test_manually_confirmed_places_are_fetched() -> None:
    place = candidate("node/1", "https://lello.example/")
    confirmed = PlaceResolution(
        place_name="Lello Bookshop",
        claim_ids=["claim-1"],
        status="resolved",
        reason="manually_confirmed",
        place=place,
        candidates=[place],
        confirmation={"original_reason": "no_match", "search_name": "Livraria Lello"},
    )

    [page] = fetch_official_websites([confirmed], make_fetcher())

    assert page.status == "fetched"


def test_source_page_round_trips_through_json() -> None:
    [page] = fetch_official_websites(
        [resolved("Lello Bookshop", ["claim-1"], "https://lello.example/")], make_fetcher()
    )

    assert SourcePage.model_validate_json(page.model_dump_json()) == page


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "fetched"},  # no text, final URL, or time
        {"status": "fetched", "text": "x", "final_url": "https://x/", "reason": "no_text",
         "fetched_at": FIXED_TIME},
        {"status": "failed"},  # failure without a reason
        {"status": "failed", "reason": "no_website"},  # reason of another status
        {"status": "skipped", "reason": "timeout"},
    ],
)  # fmt: skip
def test_inconsistent_page_fetch_is_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PageFetch.model_validate(fields)


# --- robots.txt: parsing ------------------------------------------------------------

ROBOTS = """
# Comments and unknown lines are ignored.
Sitemap: https://casalume.example/sitemap.xml
Disallow: /ignored-before-any-user-agent

User-agent: Googlebot
Disallow: /

User-agent: *
Disallow: /private/
Allow: /private/hours
Disallow: /*.pdf$
Disallow: /search?

User-agent: roamly
User-agent: OtherBot
Disallow: /admin
"""


@pytest.mark.parametrize(
    ("token", "path", "allowed"),
    [
        ("Roamly", "/", True),  # own group (case-insensitive) replaces `*`
        ("Roamly", "/admin/x", False),
        ("Roamly", "/private/", True),  # `*` rules do not apply to Roamly
        ("Unknown", "/", True),
        ("Unknown", "/private/menu", False),
        ("Unknown", "/private/hours", True),  # longer Allow wins
        ("Unknown", "/files/menu.pdf", False),  # `$` anchors the end
        ("Unknown", "/files/menu.pdf?v=2", True),
        ("Unknown", "/search?q=hours", False),  # query string is part of the path
        ("Unknown", "/ignored-before-any-user-agent", True),
        ("Googlebot", "/anything", False),
    ],
)
def test_robots_rules(token: str, path: str, allowed: bool) -> None:
    rules = parse_robots(ROBOTS, token)

    assert rules.allows(f"https://casalume.example{path}") is allowed


@pytest.mark.parametrize(
    ("robots", "allowed"),
    [
        ("User-agent: *\nDisallow: /*", False),  # wildcard that robotparser misses
        ("User-agent: *\nDisallow:", True),  # empty Disallow allows everything
        ("User-agent: *\nAllow: /\nDisallow: /", True),  # equal length: Allow wins
        ("User-agent: *\nDisallow: /\nAllow: /$", True),  # homepage explicitly allowed
        ("User-agent: roamly\nDisallow: /\n\nUser-agent: ROAMLY\nAllow: /", True),  # merged
        ("", True),
        ("<html><body>Not found</body></html>", True),  # HTML error page: no rules
    ],
)
def test_robots_rule_precedence(robots: str, allowed: bool) -> None:
    assert parse_robots(robots, "Roamly").allows("https://casalume.example/") is allowed


@pytest.mark.parametrize(
    ("user_agent", "token"),
    [
        ("Roamly/0.1 (travel guide fact checker)", "Roamly"),
        ("roamly-dev/0.2 (contact: someone)", "roamly-dev"),
        ("MyBot", "MyBot"),
    ],
)
def test_product_token(user_agent: str, token: str) -> None:
    assert product_token(user_agent) == token


@pytest.mark.parametrize("user_agent", ["", "   ", "/0.1", "(bot)"])
def test_user_agent_needs_a_product_token(user_agent: str) -> None:
    with pytest.raises(ValueError):
        WebsiteFetcher(user_agent=user_agent)


# --- robots.txt: fetching -----------------------------------------------------------


def robots_response(text: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/plain"}, text=text)


def robots_requests(site: Site) -> list[str]:
    return [str(r.url) for r in site.requests if r.url.path == "/robots.txt"]


def test_disallowed_page_is_skipped_and_never_requested(tmp_path: Path) -> None:
    site = Site(
        {"https://casalume.example/robots.txt": robots_response("User-agent: *\nDisallow: /")}
    )

    page = make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")

    assert (page.status, page.reason) == ("skipped", "robots_disallowed")
    assert "Roamly" in (page.error or "")
    assert site.pages == []
    assert list(tmp_path.iterdir()) == []


def test_rules_for_other_crawlers_do_not_apply() -> None:
    site = Site(
        {
            "https://casalume.example/robots.txt": robots_response(
                "User-agent: Googlebot\nDisallow: /"
            )
        }
    )

    assert make_fetcher(site).fetch("https://casalume.example/").status == "fetched"


def test_configured_user_agent_is_sent_and_matched() -> None:
    robots = "User-agent: roamly-dev\nDisallow: /"
    site = Site({"https://casalume.example/robots.txt": robots_response(robots)})

    default = make_fetcher(site).fetch("https://casalume.example/")
    custom = make_fetcher(site, user_agent="roamly-dev/0.2 (contact)").fetch(
        "https://casalume.example/"
    )

    assert default.status == "fetched"
    assert custom.reason == "robots_disallowed"
    assert site.requests[-1].headers["User-Agent"] == "roamly-dev/0.2 (contact)"
    assert site.requests[-1].headers["Accept"] == "text/plain"


@pytest.mark.parametrize("status", [401, 403, 404, 410, 429])
def test_robots_client_error_means_no_rules(status: int) -> None:
    site = Site({"https://casalume.example/robots.txt": httpx.Response(status)})

    assert make_fetcher(site).fetch("https://casalume.example/").status == "fetched"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(503),
        httpx.Response(302),  # redirect without a location
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
        httpx.Response(301, headers={"location": "http://127.0.0.1/robots.txt"}),
        httpx.Response(301, headers={"location": "ftp://casalume.example/robots.txt"}),
    ],
    ids=["500", "503", "302_no_location", "timeout", "connect", "to_loopback", "to_ftp"],
)
def test_unretrievable_robots_blocks_the_page(response: httpx.Response | Exception) -> None:
    site = Site({"https://casalume.example/robots.txt": response})

    page = make_fetcher(site).fetch("https://casalume.example/")

    assert (page.status, page.reason) == ("failed", "robots_unavailable")
    assert (page.error or "").startswith("robots.txt")
    assert site.pages == []
    assert all("127.0.0.1" not in str(r.url) for r in site.requests)


def test_robots_on_an_unsafe_host_is_never_requested() -> None:
    site = Site()

    page = make_fetcher(site, addresses={"casalume.example": ["10.0.0.1"]}).fetch(
        "https://casalume.example/"
    )

    assert page.reason == "unsafe_url"
    assert site.requests == []


def test_robots_redirect_is_followed() -> None:
    site = Site(
        {
            "https://casalume.example/robots.txt": httpx.Response(
                301, headers={"location": "https://www.casalume.example/robots.txt"}
            ),
            "https://www.casalume.example/robots.txt": robots_response(
                "User-agent: *\nDisallow: /"
            ),
        }
    )

    page = make_fetcher(site).fetch("https://casalume.example/")

    assert page.reason == "robots_disallowed"


def test_every_redirect_hop_is_checked_against_its_own_robots() -> None:
    site = Site(
        {
            "https://casalume.example/": httpx.Response(
                302, headers={"location": "https://booking.example/casalume"}
            ),
            "https://booking.example/robots.txt": robots_response("User-agent: *\nDisallow: /"),
        }
    )

    page = make_fetcher(site).fetch("https://casalume.example/")

    assert (page.reason, page.final_url) == (
        "robots_disallowed",
        "https://booking.example/casalume",
    )
    assert robots_requests(site) == [
        "https://casalume.example/robots.txt",
        "https://booking.example/robots.txt",
    ]
    assert [str(r.url) for r in site.pages] == ["https://casalume.example/"]


def test_robots_is_fetched_once_per_origin() -> None:
    site = Site({"https://casalume.example/": httpx.Response(302, headers={"location": "/en/"})})
    fetcher = make_fetcher(site)

    fetcher.fetch("https://casalume.example/")
    fetcher.fetch("https://casalume.example/menu")

    assert robots_requests(site) == ["https://casalume.example/robots.txt"]


def test_large_robots_is_read_up_to_the_limit() -> None:
    robots = "User-agent: *\nDisallow: /\n" + "#" * (600 * 1024)
    site = Site({"https://casalume.example/robots.txt": robots_response(robots)})

    assert make_fetcher(site).fetch("https://casalume.example/").reason == "robots_disallowed"


def test_cached_pages_need_no_robots_request(tmp_path: Path) -> None:
    site = Site()
    make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")

    page = make_fetcher(site, cache_dir=tmp_path).fetch("https://casalume.example/")

    assert page.from_cache is True
    assert len(robots_requests(site)) == 1  # only the first, uncached run
