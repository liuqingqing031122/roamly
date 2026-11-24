"""Place resolution: map the place names of extracted claims to real-world places.

Resolution is deterministic and conservative. A place is `resolved` only when exactly
one search candidate has the same normalized name *and* lies in the user's destination;
anything less certain is `ambiguous` (a person must choose) or `unresolved`. Places
that are not resolved or confirmed by a person must not be verified.

Destination check: the destination is split on commas ("Porto, Portugal" ->
"porto", "portugal") and each part is compared with the administrative areas the
search provider reports for a candidate (city, county, country, ...), after the same
normalization as place names. Exact comparison only, no substring or fuzzy matching:

- `inside`: every destination part is one of the candidate's areas.
- `outside`: none of the parts is one of its areas (e.g. Porto Alegre, Brazil).
- `unknown`: no destination, no areas reported, or only some parts match
  (e.g. "Paris, France" against Paris, Texas, or "Portland, OR" against "Oregon").

Only `inside` can lead to `resolved`; `unknown` keeps the candidate for a person to
confirm; `outside` candidates are kept for reference but never chosen.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, Self

from pydantic import BaseModel, Field, model_validator

from roamly.extraction import ExtractionResult

ResolutionStatus = Literal["resolved", "ambiguous", "unresolved"]

ResolutionReason = Literal[
    "unique_match",  # resolved: one same-name candidate, inside the destination
    "multiple_matches",  # ambiguous: several same-name candidates could be the place
    "area_unverified",  # ambiguous: one same-name candidate, destination not confirmed
    "outside_destination",  # unresolved: same-name candidates exist, all elsewhere
    "no_match",  # unresolved: no candidate has the same name
    "lookup_failed",  # unresolved: the search itself failed
]

_STATUS_BY_REASON: dict[ResolutionReason, ResolutionStatus] = {
    "unique_match": "resolved",
    "multiple_matches": "ambiguous",
    "area_unverified": "ambiguous",
    "outside_destination": "unresolved",
    "no_match": "unresolved",
    "lookup_failed": "unresolved",
}

AreaCheck = Literal["inside", "outside", "unknown"]


class PlaceCandidate(BaseModel):
    """One search result, provider-neutral, with enough detail for a person to choose."""

    provider: str  # e.g. "nominatim"
    provider_id: str  # stable within the provider, e.g. "node/123456"
    name: str
    display_name: str  # full address line as shown to the user
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    # Administrative areas containing the place, e.g. ["Porto", "Portugal"].
    areas: list[str] = Field(default_factory=list)
    category: str | None = None  # e.g. "amenity:cafe"
    website: str | None = None


class PlaceResolution(BaseModel):
    """The resolution of one place name, shared by every claim that mentions it."""

    place_name: str  # as first written in the guide
    claim_ids: list[str]
    status: ResolutionStatus
    reason: ResolutionReason
    # The chosen candidate; set only when resolved.
    place: PlaceCandidate | None = None
    # Every search result in provider order, kept for manual confirmation.
    candidates: list[PlaceCandidate] = Field(default_factory=list)
    # The search error message; set only when the lookup failed.
    error: str | None = None

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if _STATUS_BY_REASON[self.reason] != self.status:
            raise ValueError(f"Reason {self.reason!r} does not fit status {self.status!r}")
        if (self.place is not None) != (self.status == "resolved"):
            raise ValueError("A place is required if and only if the status is 'resolved'")
        if (self.error is not None) != (self.reason == "lookup_failed"):
            raise ValueError("An error is required if and only if the lookup failed")
        if not self.claim_ids:
            raise ValueError("A resolution must belong to at least one claim")
        return self


class PlaceSearchError(Exception):
    """The search did not succeed: network failure, timeout, HTTP error, bad response."""


class PlaceSearch(Protocol):
    """Anything that can look up places by name.

    Implementations return candidates in the provider's ranking order (an empty list
    when nothing is found) and raise `PlaceSearchError` on failure, never a partial
    result. `destination` may be used to narrow the search but does not have to be.
    """

    def search(self, name: str, *, destination: str | None) -> list[PlaceCandidate]: ...


@dataclass
class FakePlaceSearch:
    """Test double that answers from predefined results and records calls.

    Results are keyed by the exact name searched; unknown names return no candidates.
    A `PlaceSearchError` value is raised instead of returned.
    """

    results: dict[str, list[PlaceCandidate] | PlaceSearchError]
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def search(self, name: str, *, destination: str | None) -> list[PlaceCandidate]:
        self.calls.append((name, destination))
        result = self.results.get(name, [])
        if isinstance(result, PlaceSearchError):
            raise result
        return list(result)


def normalize_name(name: str) -> str:
    """Case-fold, trim and collapse whitespace, and drop a leading "the "."""
    return " ".join(name.split()).casefold().removeprefix("the ")


def resolve_places(
    extraction: ExtractionResult, destination: str | None, search: PlaceSearch
) -> list[PlaceResolution]:
    """Resolve the place of every accepted claim, one search per distinct place name.

    Claims whose names are equal after `normalize_name` share one resolution. Results
    are in order of first mention. A failed search affects only that place.
    """
    groups: dict[str, list[str]] = {}
    first_names: dict[str, str] = {}
    for claim in extraction.claims:
        key = normalize_name(claim.place_name)
        groups.setdefault(key, []).append(claim.id)
        first_names.setdefault(key, claim.place_name.strip())

    parts = destination_parts(destination)
    search_destination = destination.strip() if destination and parts else None
    return [
        _resolve_one(first_names[key], claim_ids, search_destination, parts, search)
        for key, claim_ids in groups.items()
    ]


def _resolve_one(
    place_name: str,
    claim_ids: list[str],
    destination: str | None,
    destination_parts: Sequence[str],
    search: PlaceSearch,
) -> PlaceResolution:
    try:
        candidates = search.search(place_name, destination=destination)
    except PlaceSearchError as exc:
        return PlaceResolution(
            place_name=place_name,
            claim_ids=claim_ids,
            status="unresolved",
            reason="lookup_failed",
            error=str(exc) or type(exc).__name__,
        )

    wanted = normalize_name(place_name)
    same_name = [c for c in candidates if normalize_name(c.name) == wanted]
    # Candidates that could be the place: same name and not known to be elsewhere.
    plausible = [
        (c, check) for c in same_name if (check := area_check(c, destination_parts)) != "outside"
    ]

    place = None
    if not same_name:
        reason: ResolutionReason = "no_match"
    elif not plausible:
        reason = "outside_destination"
    elif len(plausible) > 1:
        reason = "multiple_matches"
    elif plausible[0][1] == "inside":
        reason, place = "unique_match", plausible[0][0]
    else:
        reason = "area_unverified"

    return PlaceResolution(
        place_name=place_name,
        claim_ids=claim_ids,
        status=_STATUS_BY_REASON[reason],
        reason=reason,
        place=place,
        candidates=candidates,
    )


def area_check(candidate: PlaceCandidate, destination_parts: Sequence[str]) -> AreaCheck:
    """Whether a candidate lies in the destination (see the module docstring)."""
    areas = {normalize_name(area) for area in candidate.areas}
    if not destination_parts or not areas:
        return "unknown"
    found = sum(part in areas for part in destination_parts)
    if found == len(destination_parts):
        return "inside"
    if found == 0:
        return "outside"
    return "unknown"


def destination_parts(destination: str | None) -> list[str]:
    """The normalized comma-separated parts of a destination, e.g. ["porto", "portugal"]."""
    if destination is None:
        return []
    return [part for part in (normalize_name(p) for p in destination.split(",")) if part]
