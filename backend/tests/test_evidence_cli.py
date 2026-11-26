import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from roamly.cli import CLIError
from roamly.domain import Claim
from roamly.evidence_cli import EXIT_ERROR, EXIT_EXTRACTION_FAILED, EvidenceFile, main
from roamly.extraction import ExtractionResult
from roamly.llm.deepseek import DeepSeekClient
from roamly.places import PlaceCandidate, PlaceResolution
from roamly.places_cli import PlaceResolutionFile
from roamly.web_fetch import SourcePage
from roamly.web_fetch_cli import SourcePageFile

API_KEY = "sk-test-secret-key"
FETCHED_AT = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
TOWER_TEXT = "Horários\nTodos os dias\n09h00 às 19h00\n" + "Conteúdo da página. " * 40
CAFE_TEXT = "Reservas para grupos\nEncerrado ao domingo"


def claim(claim_id: str, place: str, statement: str) -> Claim:
    return Claim(
        id=claim_id,
        place_name=place,
        claim_type="opening_hours",
        original_excerpt=statement,
        normalized_statement=statement,
    )


def fetched(place: str, claim_ids: list[str], text: str, url: str) -> SourcePage:
    return SourcePage(
        place_name=place,
        place_id="nominatim:node/1",
        claim_ids=claim_ids,
        website=url,
        status="fetched",
        requested_url=url,
        final_url=url,
        http_status=200,
        title=place,
        text=text,
        fetched_at=FETCHED_AT,
    )


def make_pages_file() -> SourcePageFile:
    claims = [
        claim("claim-1", "Lello Bookshop", "Lello Bookshop opens at 9am every day."),
        claim("claim-3", "Torre dos Clérigos", "Torre dos Clérigos is open daily 9am-7pm."),
        claim("claim-4", "Majestic Café", "Majestic Café takes group bookings."),
        claim("claim-5", "Majestic Café", "Majestic Café is closed on Sundays."),
    ]
    place = PlaceCandidate(
        provider="nominatim", provider_id="node/1", name="x", display_name="x", lat=41, lon=-8
    )
    resolutions = [
        PlaceResolution(
            place_name=name,
            claim_ids=ids,
            status="resolved",
            reason="unique_match",
            place=place,
            candidates=[place],
        )
        for name, ids in [
            ("Lello Bookshop", ["claim-1"]),
            ("Torre dos Clérigos", ["claim-3"]),
            ("Majestic Café", ["claim-4", "claim-5"]),
        ]
    ]
    return SourcePageFile(
        places=PlaceResolutionFile(
            destination="Porto, Portugal",
            extraction=ExtractionResult(claims=claims, skipped=[]),
            resolutions=resolutions,
        ),
        pages=[
            SourcePage(
                place_name="Lello Bookshop",
                place_id="nominatim:node/1",
                claim_ids=["claim-1"],
                website="https://lello.example/",
                status="skipped",
                reason="robots_disallowed",
                requested_url="https://lello.example/",
            ),
            fetched("Torre dos Clérigos", ["claim-3"], TOWER_TEXT, "https://torre.example/"),
            fetched("Majestic Café", ["claim-4", "claim-5"], CAFE_TEXT, "https://cafe.example/"),
        ],
    )


def completion(output: object) -> dict[str, object]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(output)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


TOWER_OUTPUT = {"results": [{"claim_id": "claim-3", "excerpts": ["Todos os dias\n09h00 às 19h00"]}]}
CAFE_OUTPUT = {
    "results": [
        {"claim_id": "claim-4", "excerpts": ["Reservas para grupos"]},
        {"claim_id": "claim-5", "excerpts": []},
    ]
}


class MockDeepSeek:
    """Answers by page: the user prompt tells which website text was sent."""

    def __init__(self, tower: httpx.Response | None = None) -> None:
        self.tower = tower
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        user_prompt = payload["messages"][1]["content"]
        if "Todos os dias" in user_prompt:
            return self.tower or httpx.Response(200, json=completion(TOWER_OUTPUT))
        return httpx.Response(200, json=completion(CAFE_OUTPUT))

    def factory(self) -> Callable[[], DeepSeekClient]:
        return lambda: DeepSeekClient(API_KEY, transport=httpx.MockTransport(self.handler))


@pytest.fixture
def pages_file(tmp_path: Path) -> Path:
    path = tmp_path / "pages.json"
    path.write_text(make_pages_file().model_dump_json(indent=2), encoding="utf-8")
    return path


def run(pages_file: Path, api: MockDeepSeek, tmp_path: Path, *extra: str) -> int:
    argv = [str(pages_file), "--cache-dir", str(tmp_path / "cache"), *extra]
    return main(argv, client_factory=api.factory())


def test_summary_and_output_file(
    pages_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = MockDeepSeek()
    output = tmp_path / "runs" / "evidence.json"

    code = run(pages_file, api, tmp_path, "--output", str(output))

    out = capsys.readouterr().out
    assert code == 0
    assert len(api.requests) == 2  # one call per fetched page, none for the skipped one
    assert "Claims: 4 (2 found, 1 none found, 1 page unavailable, 0 failed)" in out
    assert "not a verdict" in out
    assert "[page_unavailable: robots_disallowed] claim-1 Lello Bookshop" in out
    assert "[found] claim-3 Torre dos Clérigos" in out
    assert "  - Todos os dias / 09h00 às 19h00" in out  # one-line preview
    assert "[none_found] claim-5 Majestic Café" in out
    assert "Conteúdo da página" not in out  # never the full page text
    assert API_KEY not in out

    saved = EvidenceFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.prompt_version == "evidence-v1"
    assert saved.model_settings["provider"] == "deepseek"
    assert saved.pages == make_pages_file()
    assert [c.status for c in saved.claims] == ["page_unavailable", "found", "found", "none_found"]
    assert saved.claims[1].evidence[0].content_excerpt == "Todos os dias\n09h00 às 19h00"
    assert API_KEY not in output.read_text(encoding="utf-8")


def test_second_run_uses_the_cache(
    pages_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api = MockDeepSeek()

    run(pages_file, api, tmp_path)
    run(pages_file, api, tmp_path)

    assert len(api.requests) == 2


def test_no_cache_calls_again(pages_file: Path, tmp_path: Path) -> None:
    api = MockDeepSeek()

    run(pages_file, api, tmp_path, "--no-cache")
    run(pages_file, api, tmp_path, "--no-cache")

    assert len(api.requests) == 4
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": {"message": "server error"}}),
        httpx.Response(200, json=completion({"wrong": "shape"})),
    ],
    ids=["api_error", "invalid_output"],
)
def test_llm_failure_for_one_page(
    response: httpx.Response,
    pages_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "evidence.json"

    code = run(pages_file, MockDeepSeek(tower=response), tmp_path, "--output", str(output))

    out = capsys.readouterr().out
    assert code == EXIT_EXTRACTION_FAILED
    assert "[failed: llm_error] claim-3 Torre dos Clérigos" in out
    assert "[found] claim-4 Majestic Café" in out
    saved = EvidenceFile.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.claims[1].status == "failed"


def test_rejected_output_is_summarised(
    pages_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invented = {"results": [{"claim_id": "claim-3", "excerpts": ["Every day, 9am to 7pm"]}]}
    api = MockDeepSeek(tower=httpx.Response(200, json=completion(invented)))

    code = run(pages_file, api, tmp_path)

    out = capsys.readouterr().out
    assert code == 0
    assert "[none_found] claim-3 Torre dos Clérigos" in out
    assert "Rejected LLM output: 1" in out
    assert "claim-3 (Torre dos Clérigos) excerpt_not_found: Every day, 9am to 7pm" in out


def test_existing_output_is_refused_before_any_llm_call(
    pages_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "evidence.json"
    output.write_text("keep", encoding="utf-8")
    api = MockDeepSeek()

    code = run(pages_file, api, tmp_path, "--output", str(output))

    assert code == EXIT_ERROR
    assert "already exists" in capsys.readouterr().err
    assert api.requests == []
    assert output.read_text(encoding="utf-8") == "keep"


def test_overwrite(pages_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    output.write_text("old", encoding="utf-8")

    code = run(pages_file, MockDeepSeek(), tmp_path, "--output", str(output), "--overwrite")

    assert code == 0
    assert EvidenceFile.model_validate_json(output.read_text(encoding="utf-8")).claims


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "File not found"),
        ("{not json", "Not a pages file"),
        ('{"destination": null, "extraction": {"claims": [], "skipped": []}, "resolutions": []}',
         "Not a pages file"),
    ],
    ids=["missing", "invalid_json", "places_file_instead"],
)  # fmt: skip
def test_invalid_input(
    content: str | None, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "pages.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    api = MockDeepSeek()

    code = run(path, api, tmp_path)

    assert code == EXIT_ERROR
    assert message in capsys.readouterr().err
    assert api.requests == []


def test_missing_api_key_is_a_clear_error(
    pages_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_key() -> DeepSeekClient:
        raise CLIError("Environment variable ROAMLY_LLM_API_KEY is not set")

    assert main([str(pages_file)], client_factory=no_key) == EXIT_ERROR
    assert "ROAMLY_LLM_API_KEY" in capsys.readouterr().err
