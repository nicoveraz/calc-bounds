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
from calc_bounds.llm.core import LLMClient, PendingResponseError, cost


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
        if "seed" in kwargs and p.get("attempt", 1) > 1:
            kwargs["seed"] = int(kwargs["seed"]) + int(p["attempt"]) - 1  # retries must differ
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


class ClaudeCLIClient:
    """Claude via headless Claude Code (`claude -p`), billed to the user's subscription.

    Made as close to a plain model call as the CLI allows: our own system prompt replaces
    Claude Code's, all tools are disabled, MCP servers and dynamic system-prompt sections are
    excluded, sessions are not persisted, and it runs in an empty directory (no CLAUDE.md).
    Measured harness overhead is ~1.2k input tokens per call. Structured output uses
    --json-schema. Usage is recorded; cost_usd is 0 (subscription), and the CLI's list-price
    estimate is kept in raw["total_cost_usd"]. Multi-message requests are flattened into one
    transcript prompt (used by the S2 agent loop).
    """

    MAX_RETRIES = 4

    def __init__(self, cfg: ProviderConfig) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        self.cfg = cfg
        exe = shutil.which("claude")
        if exe is None:
            raise RuntimeError("claude CLI not found on PATH")
        self.exe = exe
        self.workdir = Path(tempfile.gettempdir()) / "calc_bounds_claude_cli"
        self.workdir.mkdir(exist_ok=True)

    @staticmethod
    def _prompt(request: LLMRequest) -> str:
        msgs = request.messages
        if len(msgs) == 1 and msgs[0]["role"] == "user":
            return str(msgs[0]["content"])
        return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in msgs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        import json
        import subprocess

        p = request.params
        cmd = [
            self.exe,
            "-p",
            "--model",
            request.model,
            "--tools",
            "",
            "--output-format",
            "json",
            "--strict-mcp-config",
            "--exclude-dynamic-system-prompt-sections",
            "--no-session-persistence",
        ]
        if request.system:
            cmd += ["--system-prompt", request.system]
        if "effort" in p:
            cmd += ["--effort", str(p["effort"])]
        if "format" in p:
            cmd += ["--json-schema", json.dumps(p["format"])]
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                input=self._prompt(request),
                capture_output=True,
                text=True,
                cwd=self.workdir,
                timeout=900,
            )
            latency = time.perf_counter() - t0
            try:
                out = json.loads(proc.stdout)
            except json.JSONDecodeError:
                last_error = (proc.stdout + proc.stderr)[-500:]
                time.sleep(10 * 2**attempt)
                continue
            if out.get("is_error"):
                last_error = json.dumps(
                    {k: out.get(k) for k in ("subtype", "terminal_reason", "api_error_status")}
                    | {"result": str(out.get("result"))[:300]}
                )
                if "limit" in last_error.lower() or out.get("api_error_status") in (429, 529):
                    time.sleep(60 * 2**attempt)  # subscription rate limit: back off
                else:
                    time.sleep(5)
                continue
            mu = next(iter((out.get("modelUsage") or {}).values()), {})
            usage = Usage(
                input_tokens=int(mu.get("inputTokens", 0))
                + int(mu.get("cacheReadInputTokens", 0))
                + int(mu.get("cacheCreationInputTokens", 0)),
                output_tokens=int(mu.get("outputTokens", 0)),
                latency_s=latency,
            )
            structured = out.get("structured_output")
            text = json.dumps(structured) if structured is not None else str(out.get("result", ""))
            return LLMResponse(text=text, usage=usage, stop_reason=out.get("stop_reason"), raw=out)
        # A persistent failure is returned (not raised) so one bad input cannot stop a run.
        # LLM.complete does not cache error responses, so a rerun tries again.
        return LLMResponse(
            text="",
            usage=Usage(),
            stop_reason="error",
            raw={"error": f"claude -p failed after {self.MAX_RETRIES} attempts: {last_error}"},
        )


class OllamaClient:
    """Ollama native /api/chat (the OpenAI-compatible endpoint does not return logprobs).

    Standard library only. Supports JSON-schema `format`, `think`, temperature, seed and
    token logprobs (`logprobs`, `top_logprobs`).
    """

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg
        self.url = (cfg.base_url or "http://localhost:11434").rstrip("/") + "/api/chat"

    def complete(self, request: LLMRequest) -> LLMResponse:
        import json
        import urllib.request

        p = request.params
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + [
            *request.messages
        ]
        options: dict[str, Any] = {"num_predict": p.get("max_tokens", 4096)}
        for name in ("temperature", "seed"):
            if name in p:
                options[name] = p[name]
        if "seed" in options and p.get("attempt", 1) > 1:
            options["seed"] = int(options["seed"]) + int(p["attempt"]) - 1
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if "format" in p:
            body["format"] = p["format"]
        if "think" in p:
            body["think"] = p["think"]
        if p.get("logprobs"):
            body["logprobs"] = True
            body["top_logprobs"] = int(p.get("top_logprobs", 5))
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=1800) as resp:
            out = json.loads(resp.read())
        usage = Usage(
            input_tokens=int(out.get("prompt_eval_count", 0)),
            output_tokens=int(out.get("eval_count", 0)),
            latency_s=time.perf_counter() - t0,
        )
        return LLMResponse(
            text=out["message"].get("content", ""),
            logprobs=out.get("logprobs"),
            usage=usage,
            stop_reason=out.get("done_reason"),
            raw={k: v for k, v in out.items() if k != "logprobs"},
        )


def make_client(cfg: ProviderConfig) -> LLMClient:
    match cfg.kind:
        case "anthropic":
            return AnthropicClient(cfg)
        case "openai_compat":
            return OpenAICompatClient(cfg)
        case "session":
            return SessionClient(cfg)
        case "claude_cli":
            return ClaudeCLIClient(cfg)
        case "ollama":
            return OllamaClient(cfg)
