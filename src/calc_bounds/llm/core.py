"""Cache, budget and the `LLM` entry point used by every stage that calls a model."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from calc_bounds.llm import LLMRequest, LLMResponse, Usage, cache_key


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class BudgetExceededError(RuntimeError):
    pass


class PendingResponseError(RuntimeError):
    """The request has no cached response and its provider answers offline (session)."""

    def __init__(self, key: str, request: LLMRequest) -> None:
        super().__init__(f"pending offline response for {key[:12]}")
        self.key = key
        self.request = request


class DiskCache:
    """One JSON file per request: <dir>/<key[:2]>/<key>.json = {request, response}."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return LLMResponse.model_validate(json.loads(path.read_text())["response"])

    def put(self, key: str, request: LLMRequest, response: LLMResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        tmp.replace(path)


class LLM:
    """Cache-first completion with a per-run cost budget and a usage ledger (JSONL)."""

    def __init__(
        self,
        clients: dict[str, LLMClient] | Callable[[str], LLMClient],
        cache: DiskCache,
        *,
        max_cost_usd: float,
        ledger_path: Path | None = None,
    ) -> None:
        self._clients = clients
        self._made: dict[str, LLMClient] = {}
        self.cache = cache
        self.max_cost_usd = max_cost_usd
        self.ledger_path = ledger_path
        self.spent_usd = 0.0

    def client(self, name: str) -> LLMClient:
        """Clients are created lazily, so configs may name providers a run never calls."""
        if isinstance(self._clients, dict):
            return self._clients[name]
        if name not in self._made:
            self._made[name] = self._clients(name)
        return self._made[name]

    def complete(self, request: LLMRequest, *, provider: str, stage: str = "") -> LLMResponse:
        """`provider` is the config key used to route the call; the cache key depends only on
        the request (provider kind, model, prompt, params)."""
        key = cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            response = cached.model_copy(
                update={"usage": cached.usage.model_copy(update={"cached": True})}
            )
            self._log(key, request, response, stage)
            return response
        client = self.client(provider)
        if self.spent_usd > self.max_cost_usd:
            raise BudgetExceededError(
                f"spent ${self.spent_usd:.4f} > max_cost_usd ${self.max_cost_usd}"
            )
        try:
            response = client.complete(request)
        except PendingResponseError as e:
            e.key = key
            raise
        self.cache.put(key, request, response)
        self.spent_usd += response.usage.cost_usd
        self._log(key, request, response, stage)
        return response

    def _log(self, key: str, request: LLMRequest, response: LLMResponse, stage: str) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "stage": stage,
            "key": key,
            "provider": request.provider,
            "model": request.model,
            **response.usage.model_dump(),
        }
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(row) + "\n")


def cost(usage: Usage, price_in: float, price_out: float) -> float:
    """USD from per-million-token prices."""
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6
