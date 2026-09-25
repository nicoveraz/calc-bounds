"""Thin LLM provider clients (Anthropic; OpenAI-compatible for local vLLM/Ollama).

Every call goes through a disk cache keyed by sha256 of canonical JSON of
(provider, model, messages, tools, parameters). Token usage and cost are recorded per call;
prices come from config, never hardcoded. Model names come from config.
"""

from typing import Any, Protocol

from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


class LLMRequest(BaseModel):
    provider: str
    model: str
    system: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = []
    params: dict[str, Any] = {}
    """temperature, max_tokens, logprobs, response schema, ..."""


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[dict[str, Any]] = []
    logprobs: list[dict[str, Any]] | None = None
    usage: Usage
    raw: dict[str, Any]


class LLMClient(Protocol):
    provider: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def cache_key(request: LLMRequest) -> str:
    raise NotImplementedError  # M3
