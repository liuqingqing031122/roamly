"""OpenStreetMap Nominatim adapter for the `PlaceSearch` protocol.

Usage policy of the public instance (https://operations.osmfoundation.org/policies/nominatim/),
checked on 2026-11-24, and how this module follows it:

- At most 1 request per second for the whole application: requests made by one
  `NominatimSearch` are spaced by `min_interval` (default 1 s). Use one instance per
  process; separate instances do not share the limit.
- An identifying User-Agent is required; library defaults are not accepted. A
  non-empty `user_agent` must be passed (`ROAMLY_NOMINATIM_USER_AGENT` in `from_env`).
- Results must be cached: successful responses, including empty ones, are stored on
  disk and never requested again for the same parameters. Failures are not cached.
- The service must be switchable without a code change: `base_url` can point to
  another instance (`ROAMLY_NOMINATIM_URL`).
- No autocomplete, systematic, or bulk queries: Roamly searches only for place
  names that a user's guide mentions.
- Do not send personal data: only the place name and destination are sent.
- Data is © OpenStreetMap contributors (ODbL); user-facing output must attribute it.

API reference: https://nominatim.org/release-docs/latest/api/Search/ (format jsonv2).
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from roamly.places import PlaceCandidate, PlaceSearchError

logger = logging.getLogger(__name__)

PROVIDER = "nominatim"
DEFAULT_BASE_URL = "https://nominatim.openstreetmap.org"
DEFAULT_CACHE_DIR = Path(".roamly_cache/places")
USER_AGENT_ENV = "ROAMLY_NOMINATIM_USER_AGENT"
BASE_URL_ENV = "ROAMLY_NOMINATIM_URL"

# Bump when the layout of cache entries changes.
CACHE_FORMAT_VERSION = 1

# Address keys that name an area containing the place, from smallest to largest.
# Which keys appear depends on the country and the kind of place (e.g. a village has
# `village` and no `city`), so every level is checked. Streets, neighbourhoods,
# postcodes, and codes such as `country_code` are not areas a destination names.
AREA_KEYS = (
    "suburb",
    "borough",
    "city_district",
    "district",
    "hamlet",
    "village",
    "town",
    "city",
    "municipality",
    "county",
    "state_district",
    "state",
    "region",
    "country",
)


class NominatimSearch:
    """Searches places with the Nominatim Search API, with rate limiting and a disk cache."""

    def __init__(
        self,
        user_agent: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | None = DEFAULT_CACHE_DIR,
        timeout: float = 10.0,
        limit: int = 10,
        language: str = "en",
        min_interval: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("Nominatim requires a User-Agent that identifies the application")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.limit = limit
        self.language = language
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "NominatimSearch":
        """Create a client from ROAMLY_NOMINATIM_USER_AGENT and optional ROAMLY_NOMINATIM_URL."""
        user_agent = os.environ.get(USER_AGENT_ENV, "")
        if not user_agent.strip():
            raise ValueError(f"Environment variable {USER_AGENT_ENV} is not set")
        if base_url := os.environ.get(BASE_URL_ENV):
            kwargs.setdefault("base_url", base_url)
        return cls(user_agent, **kwargs)

    def search(self, name: str, *, destination: str | None) -> list[PlaceCandidate]:
        params = self.params(name, destination)
        cached = self._cache_get(params)
        if cached is not None:
            return parse_results(cached)
        body = self._request(params)
        candidates = parse_results(body)  # raises before an invalid response is cached
        self._cache_put(params, body)
        return candidates

    def params(self, name: str, destination: str | None) -> dict[str, str]:
        """Query parameters for one search; everything that can change the response."""
        query = f"{name.strip()}, {destination.strip()}" if destination else name.strip()
        return {
            "q": query,
            "format": "jsonv2",
            "addressdetails": "1",
            "extratags": "1",
            "limit": str(self.limit),
            "accept-language": self.language,
        }

    def _request(self, params: dict[str, str]) -> Any:
        self._wait_for_rate_limit()
        try:
            response = self._http.get("/search", params=params)
        except httpx.TimeoutException as exc:
            raise PlaceSearchError(f"Nominatim request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise PlaceSearchError(f"Nominatim request failed: {exc}") from exc
        finally:
            self._last_request = self._clock()

        if response.status_code != 200:
            hint = " (rate limited or blocked)" if response.status_code in (403, 429) else ""
            raise PlaceSearchError(f"Nominatim returned HTTP {response.status_code}{hint}")
        try:
            return response.json()
        except ValueError as exc:
            raise PlaceSearchError("Nominatim returned a response that is not JSON") from exc

    def _wait_for_rate_limit(self) -> None:
        if self._last_request is None:
            return
        remaining = self.min_interval - (self._clock() - self._last_request)
        if remaining > 0:
            self._sleep(remaining)

    # --- Cache --------------------------------------------------------------------

    def _cache_key_fields(self, params: dict[str, str]) -> dict[str, Any]:
        return {"cache_format": CACHE_FORMAT_VERSION, "base_url": self.base_url, "params": params}

    def _cache_path(self, params: dict[str, str]) -> Path:
        assert self.cache_dir is not None
        key = json.dumps(self._cache_key_fields(params), sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _cache_get(self, params: dict[str, str]) -> Any | None:
        """The cached raw response, or None on a miss or an unreadable entry."""
        if self.cache_dir is None:
            return None
        path = self._cache_path(params)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable place cache entry %s: %s", path.name, exc)
            return None

    def _cache_put(self, params: dict[str, str], body: Any) -> None:
        """Store a raw response that parsed successfully; the write is atomic.

        The raw response, not the candidates, is stored so that mapping changes also
        apply to cached entries.
        """
        if self.cache_dir is None:
            return
        entry = {
            "key_fields": self._cache_key_fields(params),
            "created_at": datetime.now(UTC).isoformat(),
            "response": body,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(entry, file, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self._cache_path(params))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


def parse_results(body: Any) -> list[PlaceCandidate]:
    """Convert a jsonv2 search response into candidates, keeping Nominatim's order.

    Results without an OSM object (e.g. postcode areas) are skipped because they are
    not places. Any other malformed result makes the whole response invalid.
    """
    if not isinstance(body, list):
        raise PlaceSearchError("Nominatim returned an unexpected response (not a list)")
    candidates = []
    for item in body:
        if not isinstance(item, dict):
            raise PlaceSearchError("Nominatim returned an unexpected result (not an object)")
        if not item.get("osm_type") or item.get("osm_id") is None:
            continue
        candidates.append(_parse_item(item))
    return candidates


def _parse_item(item: dict[str, Any]) -> PlaceCandidate:
    address = item.get("address")
    extratags = item.get("extratags")
    try:
        return PlaceCandidate(
            provider=PROVIDER,
            provider_id=f"{item['osm_type']}/{item['osm_id']}",
            name=_text(item.get("name")),
            display_name=_text(item.get("display_name")),
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            areas=_areas(address) if isinstance(address, dict) else [],
            category=_category(item),
            website=_website(extratags) if isinstance(extratags, dict) else None,
        )
    # Missing or out-of-range coordinates (ValidationError is a ValueError).
    except (KeyError, TypeError, ValueError) as exc:
        raise PlaceSearchError(f"Nominatim returned an invalid result: {exc}") from exc


def _areas(address: dict[str, Any]) -> list[str]:
    areas: list[str] = []
    for key in AREA_KEYS:
        value = address.get(key)
        if isinstance(value, str) and value.strip() and value not in areas:
            areas.append(value)
    return areas


def _category(item: dict[str, Any]) -> str | None:
    key, value = item.get("category"), item.get("type")
    if isinstance(key, str) and key and isinstance(value, str) and value:
        return f"{key}:{value}"
    return None


def _website(extratags: dict[str, Any]) -> str | None:
    for key in ("website", "contact:website", "url"):
        value = extratags.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
