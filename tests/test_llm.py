import json
from pathlib import Path

import pytest

from calc_bounds.llm import (
    LLM,
    BudgetExceededError,
    DiskCache,
    LLMRequest,
    LLMResponse,
    PendingResponseError,
    Usage,
    cache_key,
)
from calc_bounds.llm.clients import SessionClient
from calc_bounds.llm.session import PendingItem, import_responses, write_pending


def req(text: str = "hi", **params: object) -> LLMRequest:
    return LLMRequest(
        provider="fake", model="m", messages=[{"role": "user", "content": text}], params=params
    )


class FakeClient:
    def __init__(self, cost: float = 0.0) -> None:
        self.calls = 0
        self.cost = cost

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"echo {request.messages[0]['content']}",
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=self.cost),
        )


def test_cache_key_is_canonical() -> None:
    a = req(max_tokens=10, effort="low")
    b = req(effort="low", max_tokens=10)
    assert cache_key(a) == cache_key(b)
    assert cache_key(a) != cache_key(req("other", max_tokens=10, effort="low"))
    assert cache_key(a) != cache_key(a.model_copy(update={"model": "m2"}))


def test_cache_hit_skips_client_and_ledger_records(tmp_path: Path) -> None:
    fake = FakeClient()
    llm = LLM(
        {"f": fake}, DiskCache(tmp_path / "c"), max_cost_usd=0, ledger_path=tmp_path / "u.jsonl"
    )
    r1 = llm.complete(req(), provider="f", stage="t")
    r2 = llm.complete(req(), provider="f", stage="t")
    assert fake.calls == 1
    assert r1.text == r2.text and not r1.usage.cached and r2.usage.cached
    rows = [json.loads(line) for line in (tmp_path / "u.jsonl").read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, True]


def test_budget_stops_paid_calls(tmp_path: Path) -> None:
    llm = LLM({"f": FakeClient(cost=0.6)}, DiskCache(tmp_path), max_cost_usd=1.0)
    llm.complete(req("a"), provider="f")
    llm.complete(req("b"), provider="f")  # spent 0.6 <= 1.0 before this call
    with pytest.raises(BudgetExceededError):
        llm.complete(req("c"), provider="f")
    llm.complete(req("a"), provider="f")  # cached calls stay free


def test_session_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c")
    llm = LLM({"s": SessionClient.__new__(SessionClient)}, cache, max_cost_usd=0)
    r = req("render me")
    with pytest.raises(PendingResponseError) as exc:
        llm.complete(r, provider="s")
    key = exc.value.key
    assert key == cache_key(r)
    write_pending(tmp_path / "p.jsonl", [PendingItem(key=key, request=r)])
    (tmp_path / "r.jsonl").write_text(
        json.dumps({"key": key, "text": " a note \n"})
        + "\n"
        + json.dumps({"key": "0" * 64, "text": "x"})
        + "\n"
        + "not json\n"
    )
    n, problems = import_responses(tmp_path / "p.jsonl", tmp_path / "r.jsonl", cache)
    assert n == 1 and len(problems) == 2
    assert llm.complete(r, provider="s").text == "a note"


def test_provider_config_validation() -> None:
    from calc_bounds.config import ProviderConfig

    with pytest.raises(ValueError):
        ProviderConfig(kind="anthropic", model="x", temperature=0.2)
    with pytest.raises(ValueError):
        ProviderConfig(kind="openai_compat", model="x")
    p = ProviderConfig(kind="anthropic", model="x", effort="low", max_tokens=100)
    assert p.request_params() == {"max_tokens": 100, "effort": "low"}


def test_error_responses_are_not_cached(tmp_path: Path) -> None:
    class Flaky:
        calls = 0

        def complete(self, request: LLMRequest) -> LLMResponse:
            Flaky.calls += 1
            if Flaky.calls == 1:
                return LLMResponse(text="", usage=Usage(), stop_reason="error")
            return LLMResponse(text="ok", usage=Usage())

    llm = LLM({"f": Flaky()}, DiskCache(tmp_path), max_cost_usd=0)
    assert llm.complete(req(), provider="f").stop_reason == "error"
    assert llm.complete(req(), provider="f").text == "ok"  # retried, not served from cache
    assert llm.complete(req(), provider="f").usage.cached


def test_ollama_num_ctx_is_sent_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from calc_bounds.config import ProviderConfig
    from calc_bounds.llm.clients import OllamaClient

    sent: list[dict] = []

    class Resp:
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "{}"}, "done_reason": "stop"}).encode()

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> Resp:
        sent.append(json.loads(request.data))  # type: ignore[arg-type]
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    for num_ctx in (None, 16384):
        p = ProviderConfig(kind="ollama", model="m", num_ctx=num_ctx)
        r = LLMRequest(provider="ollama", model="m", messages=[], params=p.request_params())
        OllamaClient(p).complete(r)
    assert "num_ctx" not in sent[0]["options"]
    assert sent[1]["options"]["num_ctx"] == 16384
    assert "num_ctx" not in ProviderConfig(kind="ollama", model="m").request_params()
