"""Provider clients. Thin wrappers: build the SDK call, return an `LLMResponse`.

SDK surfaces verified against installed versions (anthropic 1.8.0, openai 3.19.2):
  anthropic.Anthropic().messages.create(model, max_tokens, system, messages, output_config)
    - current Claude models reject temperature/top_p/top_k; `output_config.effort` is the knob;
      no token logprobs are exposed.
  openai.OpenAI(base_url=...).chat.completions.create(model, messages, temperature, seed,
    max_tokens, logprobs, top_logprobs)
"""

import os
import time
from typing import Any

from calc_bounds.config import ProviderConfig
from calc_bounds.llm import LLMRequest, LLMResponse, Usage
from calc_bounds.llm.core import PendingResponseError, cost


class RefusalError(RuntimeError):
    pass


class AnthropicClient:
    def __init__(self, cfg: ProviderConfig) -> None:
        import anthropic

        self.cfg = cfg
        key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY / auth token / ant profile.
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def complete(self, request: LLMRequest) -> LLMResponse:
        p = request.params
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": p.get("max_tokens", 16000),
            "messages": request.messages,
        }
        if request.system:
            kwargs["system"] = request.system
        output_config: dict[str, Any] = {}
        if "effort" in p:
            output_config["effort"] = p["effort"]
        if "format" in p:
            output_config["format"] = p["format"]
        if output_config:
            kwargs["output_config"] = output_config
        t0 = time.perf_counter()
        msg = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - t0
        if msg.stop_reason == "refusal":
            raise RefusalError(f"refused: {msg.stop_details}")
        usage = Usage(
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_s=latency,
        )
        usage.cost_usd = cost(usage, self.cfg.price_per_mtok_in, self.cfg.price_per_mtok_out)
        return LLMResponse(
            text="".join(b.text for b in msg.content if b.type == "text"),
            usage=usage,
            stop_reason=msg.stop_reason,
            raw=msg.to_dict(),
        )


class OpenAICompatClient:
    def __init__(self, cfg: ProviderConfig) -> None:
        import openai

        self.cfg = cfg
        key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else "not-needed"
        self.client = openai.OpenAI(base_url=cfg.base_url, api_key=key or "not-needed")

    def complete(self, request: LLMRequest) -> LLMResponse:
        p = request.params
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + [
            *request.messages
        ]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": p.get("max_tokens", 4096),
        }
        for name in ("temperature", "seed", "logprobs", "top_logprobs", "response_format"):
            if name in p:
                kwargs[name] = p[name]
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_s=latency,
        )
        usage.cost_usd = cost(usage, self.cfg.price_per_mtok_in, self.cfg.price_per_mtok_out)
        logprobs = None
        if choice.logprobs is not None and choice.logprobs.content is not None:
            logprobs = [t.model_dump() for t in choice.logprobs.content]
        return LLMResponse(
            text=choice.message.content or "",
            logprobs=logprobs,
            usage=usage,
            stop_reason=choice.finish_reason,
            raw=resp.model_dump(),
        )


class SessionClient:
    """Offline provider: never calls anything. A cache miss raises PendingResponseError so the
    stage can export the request for offline answering (see llm.session)."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise PendingResponseError("", request)


def make_client(cfg: ProviderConfig) -> AnthropicClient | OpenAICompatClient | SessionClient:
    match cfg.kind:
        case "anthropic":
            return AnthropicClient(cfg)
        case "openai_compat":
            return OpenAICompatClient(cfg)
        case "session":
            return SessionClient(cfg)
