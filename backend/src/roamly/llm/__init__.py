"""Provider-independent interface for structured LLM calls."""

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """Base class for failed LLM calls."""


class LLMAPIError(LLMError):
    """The request did not succeed: network failure, timeout, or HTTP error status."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMResponseError(LLMError):
    """The response was received but is unusable: empty, truncated, invalid JSON,
    or not matching the requested schema."""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported for one LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class StructuredResponse[T: BaseModel]:
    """A validated LLM output together with its token usage."""

    output: T
    usage: TokenUsage


class LLMClient(Protocol):
    """Anything that can answer a prompt with an instance of a Pydantic model.

    Implementations must return output validated against `response_model`, and
    raise `LLMError` instead of returning partial or empty output on failure.
    Untrusted content belongs in `user_prompt`, never in `system_prompt`.
    """

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]: ...


@dataclass(frozen=True)
class LLMCall:
    """The arguments of one call made to `FakeLLMClient`."""

    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]


@dataclass
class FakeLLMClient:
    """Test double that returns predefined outputs in order and records calls."""

    outputs: list[BaseModel]
    calls: list[LLMCall] = field(default_factory=list)

    def complete_structured[T: BaseModel](
        self, *, system_prompt: str, user_prompt: str, response_model: type[T]
    ) -> StructuredResponse[T]:
        self.calls.append(LLMCall(system_prompt, user_prompt, response_model))
        if len(self.calls) > len(self.outputs):
            raise RuntimeError("FakeLLMClient has no more predefined outputs")
        output = self.outputs[len(self.calls) - 1]
        if not isinstance(output, response_model):
            raise TypeError(
                f"Expected {response_model.__name__}, "
                f"but the next predefined output is {type(output).__name__}"
            )
        return StructuredResponse(output=output, usage=TokenUsage())
