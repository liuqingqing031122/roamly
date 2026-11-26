"""Fetch the official websites of resolved places as raw material for evidence.

A `SourcePage` records whether a page could be fetched and what readable text it
contains. Its status says nothing about whether any claim is true: extracting
evidence from the text is M3.2, and fact verification is M4.

Safety (architecture §9, NFR-03): only http/https URLs on the default ports are
fetched; URLs with credentials are rejected; every host, including every redirect
target, must resolve only to public (global) IP addresses. Redirects are followed
manually so that each hop is checked. Responses are limited in size and total time.

Known limitation (not fully SSRF-safe): the address is checked before connecting and
resolved again by the HTTP client, so a DNS answer that changes in between (DNS
rebinding) is not caught. Only fetch URLs from place data (OpenStreetMap or a person's
confirmation), not arbitrary user input, and do not run this on a host whose internal
network or cloud metadata service must stay unreachable. See architecture §9.

robots.txt (RFC 9309): before every request, including each redirect hop, the rules of
that origin's robots.txt for the User-Agent's product token are applied (longest
match, `*` and `$` wildcards, Allow wins ties). A 4xx robots.txt means no rules; a
robots.txt that cannot be retrieved (5xx, timeout, network error, unsafe redirect)
means nothing on that origin is fetched. robots.txt is requested with the same URL and
address checks as pages. Cached pages are served without a new robots.txt check.
"""

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from pydantic import AwareDatetime, BaseModel, model_validator

from roamly.places import PlaceResolution

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".roamly_cache/pages")
DEFAULT_USER_AGENT = "Roamly/0.1 (travel guide fact checker; single page requests)"
USER_AGENT_ENV = "ROAMLY_FETCH_USER_AGENT"
# RFC 9309 requires parsing at least 500 KiB; anything beyond is ignored.
MAX_ROBOTS_BYTES = 500 * 1024
MAX_REDIRECTS = 5
MAX_BYTES = 2_000_000
# Enough text for evidence extraction while keeping output files small.
MAX_TEXT_CHARS = 100_000
CACHE_FORMAT_VERSION = 1

SourceType = Literal["official_website"]
FetchStatus = Literal["fetched", "skipped", "failed"]
FetchReason = Literal[
    "place_not_resolved",  # skipped: only resolved places are fetched
    "no_website",  # skipped: the place has no website
    "invalid_url",  # failed: the website value is not a usable URL
    "unsafe_url",  # failed: scheme, port, credentials, or non-public address
    "timeout",  # failed
    "network_error",  # failed: DNS or connection failure
    "http_error",  # failed: final response status was not 200
    "too_many_redirects",  # failed
    "not_html",  # failed: the response is not an HTML page
    "too_large",  # failed: more than MAX_BYTES
    "no_text",  # failed: the page was fetched but has no readable text
    "robots_disallowed",  # skipped: robots.txt does not allow fetching this URL
    "robots_unavailable",  # failed: robots.txt could not be retrieved, so nothing is fetched
]

_STATUS_BY_REASON: dict[FetchReason, FetchStatus] = {
    "place_not_resolved": "skipped",
    "no_website": "skipped",
    "invalid_url": "failed",
    "unsafe_url": "failed",
    "timeout": "failed",
    "network_error": "failed",
    "http_error": "failed",
    "too_many_redirects": "failed",
    "not_html": "failed",
    "too_large": "failed",
    "no_text": "failed",
    "robots_disallowed": "skipped",
    "robots_unavailable": "failed",
}


class PageFetch(BaseModel):
    """The outcome of fetching one URL. `fetched` means retrieved, not verified."""

    status: FetchStatus
    reason: FetchReason | None = None  # None only when fetched
    error: str | None = None  # human-readable detail of a failure
    requested_url: str | None = None  # normalized URL that was requested
    final_url: str | None = None  # URL of the page after redirects
    http_status: int | None = None
    title: str | None = None
    text: str = ""  # readable text, one block per line
    text_truncated: bool = False
    # When the page was downloaded; for cache hits, the time of the original download.
    fetched_at: AwareDatetime | None = None
    from_cache: bool = False

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.status == "fetched":
            if self.reason is not None or not self.text or self.final_url is None:
                raise ValueError("A fetched page needs text and a final URL, and no reason")
            if self.fetched_at is None:
                raise ValueError("A fetched page needs fetched_at")
        elif self.reason is None or _STATUS_BY_REASON[self.reason] != self.status:
            raise ValueError(f"Reason {self.reason!r} does not fit status {self.status!r}")
        return self


class SourcePage(PageFetch):
    """A fetched (or not fetched) official website of one place, for its claims."""

    place_name: str
    place_id: str | None  # "<provider>:<provider_id>", e.g. "nominatim:node/1"
    claim_ids: list[str]
    source_type: SourceType = "official_website"
    website: str | None  # the website value as stored for the place


# --- URL safety -----------------------------------------------------------------


class _FetchFailure(Exception):
    def __init__(
        self,
        reason: FetchReason,
        message: str,
        *,
        final_url: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: FetchReason = reason
        self.final_url = final_url
        self.http_status = http_status


def normalize_url(raw: str, *, assume_https: bool = True) -> str:
    """Return an absolute http(s) URL without fragment, or raise `_FetchFailure`.

    "www.example.com" becomes "https://www.example.com/" when `assume_https`.
    Only http/https on default ports, without credentials, is accepted.
    """
    value = raw.strip()
    if not value or any(char.isspace() for char in value):
        raise _FetchFailure("invalid_url", f"Not a URL: {raw!r}")
    if "://" not in value:
        if not assume_https:
            raise _FetchFailure("unsafe_url", f"Not an http(s) URL: {raw!r}")
        value = f"https://{value}"
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise _FetchFailure("unsafe_url", f"Only http and https URLs are fetched: {raw!r}")
    if parts.username is not None or parts.password is not None:
        raise _FetchFailure("unsafe_url", f"URLs with credentials are not fetched: {raw!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise _FetchFailure("invalid_url", f"Invalid port in {raw!r}") from exc
    if not parts.hostname:
        raise _FetchFailure("invalid_url", f"No host in {raw!r}")
    if port not in (None, 80, 443):
        raise _FetchFailure("unsafe_url", f"Only the default ports are fetched: {raw!r}")
    return urlunsplit((scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def resolve_host(host: str, port: int) -> list[str]:
    """IP addresses of `host` from the system resolver."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]).split("%")[0] for info in infos})


def check_public_host(url: str, resolve: Callable[[str, int], list[str]]) -> None:
    """Raise `_FetchFailure` unless every address of the URL's host is public."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        raise _FetchFailure("unsafe_url", f"Local host is not fetched: {host}")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = resolve(host, parts.port or (443 if parts.scheme == "https" else 80))
        except OSError as exc:
            raise _FetchFailure("network_error", f"Cannot resolve {host}: {exc}") from exc
    if not addresses:
        raise _FetchFailure("network_error", f"No address for {host}")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise _FetchFailure("unsafe_url", f"{host} resolves to a non-public address")


# --- Readable text --------------------------------------------------------------

_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "iframe", "object", "canvas"}
)
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
        "main", "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }
)  # fmt: skip


class _TextExtractor(HTMLParser):
    """Collects visible text with line breaks at block elements, and the title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        else:
            self.parts.append(data)


def extract_text(html: str) -> tuple[str | None, str]:
    """The page title (or None) and its readable text, one non-empty block per line."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    title = " ".join("".join(parser.title_parts).split()) or None
    return title, "\n".join(line for line in lines if line)


# --- robots.txt -----------------------------------------------------------------


@dataclass(frozen=True)
class RobotsRules:
    """The Allow/Disallow rules of one robots.txt that apply to one product token."""

    rules: tuple[tuple[bool, str], ...] = ()  # (allow, path pattern)

    def allows(self, url: str) -> bool:
        """RFC 9309: the longest matching pattern decides; Allow wins a tie."""
        parts = urlsplit(url)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best_length, allowed = -1, True
        for allow, pattern in self.rules:
            if _robots_pattern(pattern).match(path) and (
                len(pattern) > best_length or (len(pattern) == best_length and allow)
            ):
                best_length, allowed = len(pattern), allow
        return allowed


def _robots_pattern(pattern: str) -> re.Pattern[str]:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + (r"\Z" if anchored else ""))


def parse_robots(text: str, product_token: str) -> RobotsRules:
    """Rules of the groups naming `product_token` (case-insensitive), else of `*`."""
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if rules:  # a user-agent line after rules starts a new group
                groups.append((agents, rules))
                agents, rules = [], []
            agents.append(value.lower())
        elif key in ("allow", "disallow") and agents and value:
            rules.append((key == "allow", value))
    if agents:
        groups.append((agents, rules))

    token = product_token.lower()
    named = [group_rules for names, group_rules in groups if token in names]
    if not named:
        named = [group_rules for names, group_rules in groups if "*" in names]
    return RobotsRules(tuple(rule for group_rules in named for rule in group_rules))


def product_token(user_agent: str) -> str:
    """The product token robots.txt groups are matched against, e.g. "Roamly"."""
    match = re.match(r"[A-Za-z_-]+", user_agent.strip())
    if match is None:
        raise ValueError(f"User-Agent must start with a product token: {user_agent!r}")
    return match.group()


# --- Fetcher --------------------------------------------------------------------


class WebsiteFetcher:
    """Fetches single HTML pages safely, with a disk cache of successful downloads."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        refresh: bool = False,
        timeout: float = 10.0,
        total_timeout: float = 30.0,
        max_bytes: int = MAX_BYTES,
        transport: httpx.BaseTransport | None = None,
        resolve: Callable[[str, int], list[str]] = resolve_host,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A User-Agent is required")
        self.user_agent = user_agent
        self.product_token = product_token(user_agent)
        self.cache_dir = cache_dir
        # Skip cache reads (but still store new downloads), to bypass old entries.
        self.refresh = refresh
        self.total_timeout = total_timeout
        self.max_bytes = max_bytes
        self._resolve = resolve
        self._now = now
        self._clock = clock
        # robots.txt per origin for the lifetime of this fetcher (one run).
        self._robots: dict[str, RobotsRules | _FetchFailure] = {}
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=False,  # every hop is checked by `fetch`
            headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"},
            transport=transport,
        )

    def fetch(self, website: str) -> PageFetch:
        """Fetch one page. Expected failures are returned as a failed `PageFetch`."""
        requested: str | None = None
        try:
            requested = normalize_url(website)
            if not self.refresh and (cached := self._cache_get(requested)) is not None:
                return self._page(requested, from_cache=True, **cached)
            final_url, html = self._download(requested)
        except _FetchFailure as failure:
            return PageFetch(
                status=_STATUS_BY_REASON[failure.reason],
                reason=failure.reason,
                error=str(failure),
                requested_url=requested,
                final_url=failure.final_url,
                http_status=failure.http_status,
            )
        fetched_at = self._now()
        self._cache_put(requested, final_url, html, fetched_at)
        return self._page(requested, final_url=final_url, html=html, fetched_at=fetched_at)

    def _download(self, url: str) -> tuple[str, str]:
        """Follow redirects with a safety check per hop; return the final URL and HTML."""
        started = self._clock()
        current = url
        try:
            for _ in range(MAX_REDIRECTS + 1):
                check_public_host(current, self._resolve)
                self._check_robots(current)
                with self._http.stream("GET", current) as response:
                    location = response.headers.get("location")
                    if response.is_redirect and location:
                        current = normalize_url(urljoin(current, location), assume_https=False)
                        continue
                    return current, self._read_html(response, current, started)
        except httpx.TimeoutException as exc:
            raise _FetchFailure("timeout", f"Request timed out: {exc}", final_url=current) from exc
        except httpx.HTTPError as exc:
            raise _FetchFailure(
                "network_error", f"Request failed: {exc}", final_url=current
            ) from exc
        raise _FetchFailure(
            "too_many_redirects", f"More than {MAX_REDIRECTS} redirects", final_url=current
        )

    def _check_robots(self, url: str) -> None:
        """Raise unless the origin's robots.txt allows fetching `url`."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            try:
                self._robots[origin] = self._fetch_robots(origin)
            except _FetchFailure as failure:
                self._robots[origin] = failure
        robots = self._robots[origin]
        if isinstance(robots, _FetchFailure):
            raise _FetchFailure("robots_unavailable", str(robots), final_url=url)
        if not robots.allows(url):
            raise _FetchFailure(
                "robots_disallowed",
                f"robots.txt of {origin} does not allow {self.product_token} to fetch this URL",
                final_url=url,
            )

    def _fetch_robots(self, origin: str) -> RobotsRules:
        """Fetch and parse robots.txt; raise `_FetchFailure` if it cannot be retrieved."""
        started = self._clock()
        current = f"{origin}/robots.txt"
        try:
            for _ in range(MAX_REDIRECTS + 1):
                check_public_host(current, self._resolve)
                with self._http.stream(
                    "GET", current, headers={"Accept": "text/plain"}
                ) as response:
                    location = response.headers.get("location")
                    if response.is_redirect and location:
                        current = normalize_url(urljoin(current, location), assume_https=False)
                        continue
                    status = response.status_code
                    if 400 <= status < 500:  # RFC 9309: unavailable, so no restrictions
                        return RobotsRules()
                    if not 200 <= status < 300:
                        raise _FetchFailure("http_error", f"returned HTTP {status}")
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) >= MAX_ROBOTS_BYTES:
                            break
                        if self._clock() - started > self.total_timeout:
                            raise _FetchFailure("timeout", "download took too long")
                    text = bytes(body[:MAX_ROBOTS_BYTES]).decode("utf-8", errors="replace")
                    return parse_robots(text, self.product_token)
        except httpx.TimeoutException as exc:
            raise _FetchFailure("timeout", f"robots.txt request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise _FetchFailure("network_error", f"robots.txt request failed: {exc}") from exc
        except _FetchFailure as failure:
            raise _FetchFailure(failure.reason, f"robots.txt: {failure}") from failure
        raise _FetchFailure("too_many_redirects", "robots.txt: too many redirects")

    def _read_html(self, response: httpx.Response, url: str, started: float) -> str:
        status = response.status_code
        if status != 200:
            raise _FetchFailure("http_error", f"HTTP {status}", final_url=url, http_status=status)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and content_type not in ("text/html", "application/xhtml+xml"):
            raise _FetchFailure(
                "not_html", f"Not an HTML page: {content_type}", final_url=url, http_status=status
            )
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_bytes:
            raise _FetchFailure("too_large", f"Page is {declared} bytes", final_url=url)
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > self.max_bytes:
                raise _FetchFailure(
                    "too_large", f"Page exceeds {self.max_bytes} bytes", final_url=url
                )
            if self._clock() - started > self.total_timeout:
                raise _FetchFailure("timeout", "Page download took too long", final_url=url)
        try:
            return bytes(body).decode(response.charset_encoding or "utf-8", errors="replace")
        except LookupError:  # unknown charset name in the header
            return bytes(body).decode("utf-8", errors="replace")

    def _page(
        self,
        requested: str,
        *,
        final_url: str,
        html: str,
        fetched_at: datetime,
        from_cache: bool = False,
    ) -> PageFetch:
        title, text = extract_text(html)
        common: dict[str, Any] = {
            "requested_url": requested,
            "final_url": final_url,
            "http_status": 200,
            "title": title,
            "fetched_at": fetched_at,
            "from_cache": from_cache,
        }
        if not text:
            return PageFetch(
                status="failed",
                reason="no_text",
                error="The page has no readable text (it may need JavaScript)",
                **common,
            )
        return PageFetch(
            status="fetched",
            text=text[:MAX_TEXT_CHARS],
            text_truncated=len(text) > MAX_TEXT_CHARS,
            **common,
        )

    # --- Cache ------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        assert self.cache_dir is not None
        key = json.dumps({"cache_format": CACHE_FORMAT_VERSION, "url": url}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _cache_get(self, url: str) -> dict[str, Any] | None:
        """Cached `final_url`, `html`, and `fetched_at`, or None on a miss or bad entry."""
        if self.cache_dir is None:
            return None
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            return {
                "final_url": str(entry["final_url"]),
                "html": str(entry["html"]),
                "fetched_at": _parse_time(entry["fetched_at"]),
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable page cache entry %s: %s", path.name, exc)
            return None

    def _cache_put(self, url: str, final_url: str, html: str, fetched_at: datetime) -> None:
        """Store a successful download (raw HTML, so text extraction fixes apply later)."""
        if self.cache_dir is None:
            return
        entry = {
            "cache_format": CACHE_FORMAT_VERSION,
            "url": url,
            "final_url": final_url,
            "fetched_at": fetched_at.isoformat(),
            "html": html,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(entry, file, ensure_ascii=False)
            os.replace(tmp_name, self._cache_path(url))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def _parse_time(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("fetched_at has no time zone")
    return parsed


# --- Places ---------------------------------------------------------------------


def fetch_official_websites(
    resolutions: list[PlaceResolution], fetcher: WebsiteFetcher
) -> list[SourcePage]:
    """One `SourcePage` per place, in order; a failure affects only its own place.

    Only resolved places (automatic or manually confirmed) are fetched.
    """
    pages = []
    for resolution in resolutions:
        place = resolution.place
        info: dict[str, Any] = {
            "place_name": resolution.place_name,
            "place_id": f"{place.provider}:{place.provider_id}" if place else None,
            "claim_ids": resolution.claim_ids,
            "website": place.website if place else None,
        }
        if resolution.status != "resolved" or place is None:
            fetch = PageFetch(status="skipped", reason="place_not_resolved")
        elif not place.website:
            fetch = PageFetch(status="skipped", reason="no_website")
        else:
            fetch = fetcher.fetch(place.website)
        pages.append(SourcePage(**info, **fetch.model_dump()))
        logger.info("%s: %s", resolution.place_name, fetch.reason or fetch.status)
    return pages
