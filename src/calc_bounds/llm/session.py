"""Offline ("session") provider: export pending requests, import responses into the cache.

Pending file (JSONL): {"key", "provider", "model", "system", "user", "params"} per request.
Response file (JSONL): {"key", "text"} per request. Import checks that every key matches a
pending request (the request is re-hashed), so a response can't land under the wrong prompt.
Imported responses record usage 0 and cost 0, and `raw.source = "session-import"`.
"""

import json
from pathlib import Path

from pydantic import BaseModel

from calc_bounds.io import read_jsonl
from calc_bounds.llm import LLMRequest, LLMResponse, Usage, cache_key
from calc_bounds.llm.core import DiskCache


class PendingItem(BaseModel):
    key: str
    request: LLMRequest


class SessionResponse(BaseModel):
    key: str
    text: str


def write_pending(path: Path, items: list[PendingItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def import_responses(
    pending_path: Path, responses_path: Path, cache: DiskCache
) -> tuple[int, list[str]]:
    """Returns (imported count, problems). Empty or unknown responses are reported, not stored."""
    pending = {it.key: it.request for it in read_jsonl(pending_path, PendingItem)}
    problems: list[str] = []
    n = 0
    for line_no, line in enumerate(responses_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = SessionResponse.model_validate(json.loads(line))
        except Exception as e:
            problems.append(f"line {line_no}: invalid JSON ({e})")
            continue
        request = pending.get(r.key)
        if request is None:
            problems.append(f"line {line_no}: key {r.key[:12]} not in pending")
            continue
        assert cache_key(request) == r.key
        if not r.text.strip():
            problems.append(f"line {line_no}: empty text for {r.key[:12]}")
            continue
        cache.put(
            r.key,
            request,
            LLMResponse(text=r.text.strip(), usage=Usage(), raw={"source": "session-import"}),
        )
        n += 1
    return n, problems
