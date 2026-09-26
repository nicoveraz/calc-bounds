"""LLM extractor: prompt, per-calculator JSON schema, and parsing into tri-state extractions.

The model returns, per parameter: status (present / absent / unknown), a value in the
parameter's domain, a unit token (numeric params), an exact evidence quote, and a
self-reported confidence. Code then:
  * locates the quote in the note (exact substring) to get character offsets; a quote that
    is not an exact substring is rejected and logged, and the parameter becomes Unknown;
  * normalizes units (`units.py`), never trusting the model with conversions;
  * rejects Absent for non-negatable parameters (age, weight, ...) and values outside the
    domain, logging each rejection;
  * uses token logprobs for confidence when the provider returns them (see `confidence.py`),
    otherwise the self-reported confidence, flagged as such.
"""

import json
from typing import Any

from calc_bounds.calculators import Calculator
from calc_bounds.extraction import ExtractionResult, RejectedSpan
from calc_bounds.extraction.confidence import field_confidence
from calc_bounds.types import (
    Absent,
    BoolDomain,
    ConfidenceSource,
    EvidenceSpan,
    Extraction,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    Present,
    Unknown,
)
from calc_bounds.units import UnitError, accepted_units, to_canonical

EXTRACTION_PROMPT_VERSION = "x1"

EXTRACT_SYSTEM = """\
You extract clinical parameters from a clinical note into a fixed JSON structure. You report \
only what the note documents; you never guess. Output only JSON matching the schema."""

STATUS_RULES = """\
For every parameter give:
- "status":
  - "present": the note states the finding, or states a value/level for it.
  - "absent": the note explicitly says the finding is absent or denied, or that the \
parameter is normal (e.g. "no hemoptysis", "ECG normal", "vitals within normal limits").
  - "unknown": the note does not document it. Do not infer absence from silence: if the \
note does not mention it, the status is "unknown", never "absent".
- "value": for yes/no findings use null; for graded parameters one of the listed levels; for \
numeric parameters the number exactly as written in the note (do not convert units). Use \
null when status is not "present".
- "unit": for numeric parameters, the unit as written, mapped to one of the listed unit \
tokens; otherwise null.
- "evidence": the shortest exact quote from the note (copied character for character) that \
supports the status, or null when status is "unknown".
- "confidence": your probability (0 to 1) that status and value are correct.
If the note gives several values for the same parameter (e.g. an earlier reading or a \
previous visit), report the current one."""


def _param_line(p: ParameterSpec) -> str:
    match p.domain:
        case BoolDomain():
            kind = "yes/no finding"
        case OrdinalDomain(levels=levels):
            kind = "graded; levels: " + ", ".join(levels)
        case NumericDomain():
            kind = "numeric; unit tokens: " + ", ".join(accepted_units(p.id))
    return f"- {p.id}: {p.label} [{kind}]"


def extraction_prompt(note: str, calc: Calculator) -> str:
    params = "\n".join(_param_line(p) for p in calc.parameters)
    return f"""\
Note:
<note>
{note}
</note>

Parameters:
{params}

{STATUS_RULES}"""


def extraction_schema(calc: Calculator) -> dict[str, Any]:
    """JSON schema with one fixed key per parameter (enables constrained decoding)."""
    props: dict[str, Any] = {}
    for p in calc.parameters:
        match p.domain:
            case BoolDomain():
                value: dict[str, Any] = {"type": "null"}
                unit: dict[str, Any] = {"type": "null"}
            case OrdinalDomain(levels=levels):
                value = {"anyOf": [{"type": "string", "enum": list(levels)}, {"type": "null"}]}
                unit = {"type": "null"}
            case NumericDomain():
                value = {"anyOf": [{"type": "number"}, {"type": "null"}]}
                unit = {
                    "anyOf": [{"type": "string", "enum": accepted_units(p.id)}, {"type": "null"}]
                }
        props[p.id] = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["present", "absent", "unknown"]},
                "value": value,
                "unit": unit,
                "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "confidence": {"type": "number"},
            },
            "required": ["status", "value", "unit", "evidence", "confidence"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": props,
        "required": [p.id for p in calc.parameters],
        "additionalProperties": False,
    }


def _span(note: str, quote: str) -> EvidenceSpan | None:
    start = note.find(quote)
    if not quote or start < 0:
        return None
    return EvidenceSpan(start=start, end=start + len(quote), text=quote)


def parse_extraction(
    case_id: str,
    note: str,
    calc: Calculator,
    text: str,
    logprobs: list[dict[str, Any]] | None = None,
) -> ExtractionResult:
    """Parse the model's JSON into tri-state extractions. Anything malformed becomes Unknown
    with a logged rejection; nothing is silently coerced to Absent."""
    rejected: list[RejectedSpan] = []
    values: dict[str, Extraction] = {}
    try:
        data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        data = {}
        rejected.append(
            RejectedSpan(param="*", span=None, reason=f"unparseable output: {text[:200]!r}")
        )

    def reject(pid: str, reason: str, span: EvidenceSpan | None = None) -> None:
        rejected.append(RejectedSpan(param=pid, span=span, reason=reason))
        values[pid] = Unknown()

    for p in calc.parameters:
        item = data.get(p.id)
        if not isinstance(item, dict):
            if data:
                reject(p.id, "missing from output")
            else:
                values[p.id] = Unknown()
            continue
        status = item.get("status")
        if status == "unknown" or status not in ("present", "absent"):
            if status not in ("unknown", "present", "absent"):
                reject(p.id, f"invalid status {status!r}")
            else:
                values[p.id] = Unknown()
            continue
        quote = item.get("evidence") or ""
        span = _span(note, quote)
        if span is None:
            reject(
                p.id,
                f"evidence is not an exact substring of the note: {quote[:80]!r}",
                EvidenceSpan(start=0, end=0, text=quote) if quote else None,
            )
            continue
        conf, source = _confidence(p.id, item, text, logprobs)
        if status == "absent":
            if not p.negatable:
                reject(p.id, "absent is not meaningful for this parameter", span)
                continue
            values[p.id] = Absent(confidence=conf, confidence_source=source, evidence=span)
            continue
        # present
        raw_value = item.get("value")
        match p.domain:
            case BoolDomain():
                value: bool | int | float = True
                unit = None
            case OrdinalDomain(levels=levels):
                if raw_value not in levels:
                    reject(p.id, f"invalid level {raw_value!r}", span)
                    continue
                value, unit = levels.index(raw_value), None
            case NumericDomain(lo=lo, hi=hi):
                unit = item.get("unit") or p.domain.unit
                try:
                    canonical = to_canonical(p.id, float(raw_value), unit)
                except (TypeError, ValueError, UnitError) as e:
                    reject(p.id, f"bad numeric value/unit {raw_value!r} {unit!r}: {e}", span)
                    continue
                if not lo <= canonical <= hi:
                    reject(p.id, f"value {canonical} outside plausible range [{lo}, {hi}]", span)
                    continue
                value = float(raw_value)
        values[p.id] = Present(
            value=value, unit=unit, confidence=conf, confidence_source=source, evidence=span
        )
    return ExtractionResult(case_id=case_id, values=values, rejected=rejected)


def _confidence(
    pid: str, item: dict[str, Any], text: str, logprobs: list[dict[str, Any]] | None
) -> tuple[float, ConfidenceSource]:
    if logprobs:
        c = field_confidence(text, logprobs, pid)
        if c is not None:
            return c, "logprob"
    try:
        c = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        c = 0.5
    return min(1.0, max(0.0, c)), "self_reported"
