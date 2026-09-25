"""Thin LLM layer: request/response models, disk cache, usage ledger and budget.

Every call goes through `LLM.complete`, which looks the request up in the disk cache (keyed by
sha256 of the canonical JSON of provider, model, system, messages, tools and params) before
calling a provider client. Prices come from config, never hardcoded. Model names come from
config.

Providers:
  anthropic      Anthropic Messages API (sampling params are rejected by current models;
                 determinism comes from the cache).
  openai_compat  OpenAI-compatible local servers (vLLM, Ollama); supports temperature, seed,
                 logprobs.
  session        No API call: requests are exported as "pending" and answered offline (e.g. by
                 a Claude Code session), then imported into the cache. See `llm.session`.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


class LLMRequest(BaseModel):
    provider: str
    """Provider kind + label, e.g. 'anthropic', 'openai_compat', 'session'."""
    model: str
    system: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = []
    params: dict[str, Any] = {}
    """max_tokens, effort, temperature, seed, logprobs, response schema, ..."""


class LLMResponse(BaseModel):
    text: str
    tool_calls: list[dict[str, Any]] = []
    logprobs: list[dict[str, Any]] | None = None
    usage: Usage
    stop_reason: str | None = None
    raw: dict[str, Any] = {}


def cache_key(request: LLMRequest) -> str:
    canonical = json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


from calc_bounds.llm.core import (  # noqa: E402
    LLM,
    BudgetExceededError,
    DiskCache,
    LLMClient,
    PendingResponseError,
)

__all__ = [
    "LLM",
    "BudgetExceededError",
    "DiskCache",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "PendingResponseError",
    "Usage",
    "cache_key",
]
