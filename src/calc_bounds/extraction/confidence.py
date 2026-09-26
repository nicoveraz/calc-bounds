"""Token-logprob confidence for fields of a JSON output.

Confidence of a parameter = P(status tokens) x P(value tokens), where the tokens are those
overlapping the characters of the status string and the value literal of that parameter's
object in the output text. Requires the concatenated token strings to reproduce the text;
otherwise returns None (the caller falls back to self-reported confidence).
"""

import json
import math
import re
from typing import Any


def _token_spans(logprobs: list[dict[str, Any]]) -> list[tuple[int, int, float]]:
    spans, pos = [], 0
    for t in logprobs:
        tok = str(t.get("token", ""))
        spans.append((pos, pos + len(tok), float(t.get("logprob", 0.0))))
        pos += len(tok)
    return spans


def _object_bounds(text: str, pid: str) -> tuple[int, int] | None:
    m = re.search(rf'"{re.escape(pid)}"\s*:\s*\{{', text)
    if not m:
        return None
    start = m.end() - 1
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def _value_span(text: str, lo: int, hi: int, key: str) -> tuple[int, int] | None:
    m = re.compile(rf'"{key}"\s*:\s*').search(text, lo, hi)
    if not m:
        return None
    s = m.end()
    try:
        _, end = json.JSONDecoder().raw_decode(text[s:hi])
    except json.JSONDecodeError:
        return None
    return s, s + end


def field_confidence(text: str, logprobs: list[dict[str, Any]], pid: str) -> float | None:
    spans = _token_spans(logprobs)
    if not spans or "".join(str(t.get("token", "")) for t in logprobs) != text:
        return None
    bounds = _object_bounds(text, pid)
    if bounds is None:
        return None
    total = 0.0
    for key in ("status", "value"):
        vs = _value_span(text, *bounds, key)
        if vs is None:
            return None
        a, b = vs
        total += sum(lp for s, e, lp in spans if s < b and e > a)
    return math.exp(total)
