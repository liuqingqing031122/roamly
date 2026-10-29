import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from roamly.extraction import ExtractedClaims
from roamly.llm import LLMAPIError, LLMResponseError, TokenUsage
from roamly.llm.deepseek import DeepSeekClient

API_KEY = "sk-test-secret"

VALID_OUTPUT = {
    "claims": [
        {
            "place_name": "City Museum",
            "claim_type": "opening_hours",
            "assertion_kind": "policy",
            "original_excerpt": "The museum is open every Monday.",
            "normalized_statement": "City Museum is open on Mondays.",
            "scope": None,
        }
    ]
}


def completion(
    content: str | None, finish_reason: str = "stop", usage: dict[str, int] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return body


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> DeepSeekClient:
    return DeepSeekClient(API_KEY, transport=httpx.MockTransport(handler))


def respond_json(body: Any, status_code: int = 200) -> DeepSeekClient:
    return make_client(lambda request: httpx.Response(status_code, json=body))


def call(client: DeepSeekClient) -> ExtractedClaims:
    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt="guide text", response_model=ExtractedClaims
    )
    return response.output


# Successful calls


def test_valid_output_is_parsed_with_usage() -> None:
    client = respond_json(
        completion(
            json.dumps(VALID_OUTPUT),
            usage={"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
        )
    )

    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt="guide text", response_model=ExtractedClaims
    )

    assert response.output == ExtractedClaims.model_validate(VALID_OUTPUT)
    assert response.usage == TokenUsage(input_tokens=120, output_tokens=45)


def test_missing_usage_defaults_to_zero() -> None:
    client = respond_json(completion(json.dumps(VALID_OUTPUT)))

    response = client.complete_structured(
        system_prompt="Extract claims.", user_prompt="guide text", response_model=ExtractedClaims
    )

    assert response.usage == TokenUsage()


def test_request_carries_prompts_schema_and_settings() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=completion(json.dumps(VALID_OUTPUT)))

    client = DeepSeekClient(API_KEY, "deepseek-v4-pro", transport=httpx.MockTransport(handler))

    client.complete_structured(
        system_prompt="Extract claims.",
        user_prompt="<travel_guide>The museum is open.</travel_guide>",
        response_model=ExtractedClaims,
    )

    [request] = requests
    payload = json.loads(request.content)
    system, user = payload["messages"]
    assert request.url == "https://api.deepseek.com/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "disabled"}
    assert system["role"] == "system"
    assert system["content"].startswith("Extract claims.")
    assert json.dumps(ExtractedClaims.model_json_schema()) in system["content"]
    assert "json" in system["content"]
    assert user == {"role": "user", "content": "<travel_guide>The museum is open.</travel_guide>"}


# Unusable responses


def test_invalid_json_raises() -> None:
    client = respond_json(completion('{"claims": ['))

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        call(client)


def test_schema_mismatch_raises() -> None:
    client = respond_json(completion(json.dumps({"claims": [{"claim_type": "price"}]})))

    with pytest.raises(LLMResponseError, match="does not match ExtractedClaims"):
        call(client)


def test_json_that_is_not_an_object_raises() -> None:
    client = respond_json(completion("[]"))

    with pytest.raises(LLMResponseError, match="does not match ExtractedClaims"):
        call(client)


@pytest.mark.parametrize("content", [None, "", "   \n"])
def test_empty_content_raises(content: str | None) -> None:
    client = respond_json(completion(content))

    with pytest.raises(LLMResponseError, match="empty content"):
        call(client)


def test_truncated_response_raises() -> None:
    client = respond_json(completion('{"claims": [', finish_reason="length"))

    with pytest.raises(LLMResponseError, match="truncated"):
        call(client)


@pytest.mark.parametrize("finish_reason", ["content_filter", "insufficient_system_resource"])
def test_abnormal_finish_reason_raises(finish_reason: str) -> None:
    client = respond_json(completion(json.dumps(VALID_OUTPUT), finish_reason=finish_reason))

    with pytest.raises(LLMResponseError, match=finish_reason):
        call(client)


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"index": 0}]}])
def test_missing_choice_raises(body: dict[str, Any]) -> None:
    client = respond_json(body)

    with pytest.raises(LLMResponseError, match="no message choice"):
        call(client)


def test_non_json_response_body_raises() -> None:
    client = make_client(lambda request: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(LLMResponseError, match="non-JSON response body"):
        call(client)


@pytest.mark.parametrize("body", [[], ["choices"], "text", 42, None])
def test_non_object_response_body_raises(body: Any) -> None:
    client = make_client(lambda request: httpx.Response(200, content=json.dumps(body).encode()))

    with pytest.raises(LLMResponseError, match="not an object"):
        call(client)


# API failures


@pytest.mark.parametrize("status_code", [400, 401, 402, 422, 429, 500, 503])
def test_http_error_raises_with_status_code(status_code: int) -> None:
    client = respond_json(
        {"error": {"message": "Something went wrong", "type": "api_error"}}, status_code
    )

    with pytest.raises(LLMAPIError, match=f"HTTP {status_code}: Something went wrong") as exc:
        call(client)

    assert exc.value.status_code == status_code
    assert API_KEY not in str(exc.value)


def test_http_error_without_json_body_uses_text() -> None:
    client = make_client(lambda request: httpx.Response(502, text="Bad gateway"))

    with pytest.raises(LLMAPIError, match="HTTP 502: Bad gateway"):
        call(client)


def test_timeout_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMAPIError, match="timed out") as exc:
        call(make_client(handler))

    assert exc.value.status_code is None


def test_network_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMAPIError, match="ConnectError"):
        call(make_client(handler))


# Configuration


@pytest.mark.parametrize("api_key", ["", "   "])
def test_empty_api_key_is_rejected(api_key: str) -> None:
    with pytest.raises(ValueError, match="API key is required"):
        DeepSeekClient(api_key)


def test_from_env_reads_key_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROAMLY_LLM_API_KEY", API_KEY)
    monkeypatch.setenv("ROAMLY_LLM_MODEL", "deepseek-v4-pro")

    client = DeepSeekClient.from_env()

    assert client.model == "deepseek-v4-pro"


def test_from_env_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROAMLY_LLM_API_KEY", API_KEY)
    monkeypatch.delenv("ROAMLY_LLM_MODEL", raising=False)

    assert DeepSeekClient.from_env().model == "deepseek-flash"


def test_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROAMLY_LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ROAMLY_LLM_API_KEY is not set"):
        DeepSeekClient.from_env()
