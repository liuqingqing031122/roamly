"""DeepSeek adapter for the `LLMClient` protocol.

DeepSeek's JSON mode (`response_format: json_object`) guarantees syntactically
valid JSON but not conformance to a schema, so the expected JSON Schema is added
to the system prompt and every response is validated with the response model.
"""

import json
import os
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from roamly.llm import LLMAPIError, LLMResponseError, StructuredResponse, TokenUsage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
API_KEY_ENV = "ROAMLY_LLM_API_KEY"
MODEL_ENV = "ROAMLY_LLM_MODEL"


class DeepSeekClient:
    """Calls the DeepSeek chat completions API and returns validated structured output."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A DeepSeek API key is required")
        self.model = model
        self.max_tokens = max_tokens
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DeepSeekClient":
        """Create a client from ROAMLY_LLM_API_KEY and optional ROAMLY_LLM_MODEL."""
        api_key = os.environ.get(API_KEY_ENV, "")
        if not api_key.strip():
            raise ValueError(f"Environment variable {API_KEY_ENV} is not set")
        return cls(api_key, os.environ.get(MODEL_ENV) or DEFAULT_MODEL, **kwargs)

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _with_schema(system_prompt, response_model)},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            # Thinking mode is on by default and ignores temperature; extraction does not need it.
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        body = self._post(payload)
        content = _message_content(body)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"DeepSeek returned invalid JSON: {exc}") from exc
        try:
            output = response_model.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseError(
                f"DeepSeek response does not match {response_model.__name__}: {exc}"
            ) from exc

        return StructuredResponse(output=output, usage=_token_usage(body))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMAPIError("DeepSeek request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMAPIError(f"DeepSeek request failed: {type(exc).__name__}") from exc

        if response.is_error:
            raise LLMAPIError(
                f"DeepSeek API returned HTTP {response.status_code}: {_error_detail(response)}",
                status_code=response.status_code,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMResponseError("DeepSeek API returned a non-JSON response body") from exc
        if not isinstance(body, dict):
            raise LLMResponseError(
                "DeepSeek API returned a JSON response body that is not an object"
            )
        return body


def _with_schema(system_prompt: str, response_model: type[BaseModel]) -> str:
    """Append the expected JSON Schema; JSON mode also requires the word "json"."""
    schema = json.dumps(response_model.model_json_schema())
    return (
        f"{system_prompt}\n\n"
        "Respond with a single json object that conforms to this JSON Schema:\n"
        f"{schema}"
    )


def _message_content(body: dict[str, Any]) -> str:
    try:
        choice = body["choices"][0]
        content = choice["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMResponseError("DeepSeek response has no message choice") from exc

    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise LLMResponseError("DeepSeek response was truncated at the max_tokens limit")
    if finish_reason not in (None, "stop"):
        raise LLMResponseError(f"DeepSeek response ended with finish_reason={finish_reason!r}")
    # DeepSeek documents that JSON mode may occasionally return empty content.
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("DeepSeek returned empty content")
    return content


def _token_usage(body: dict[str, Any]) -> TokenUsage:
    usage = body.get("usage") or {}
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )


def _error_detail(response: httpx.Response) -> str:
    """Extract the provider's error message, truncated; never includes request headers."""
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text
    return str(message)[:300] or response.reason_phrase
