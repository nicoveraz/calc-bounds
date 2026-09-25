"""Validation pass. Rule checks are deterministic; the judge check is semantic (LLM).

Failures are flagged as issues on the note; nothing is silently dropped.
"""

import json
import re

from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.render import ValidationIssue, vocab
from calc_bounds.render.facts import build_facts
from calc_bounds.types import DocumentedState, OrdinalDomain

MIN_WORDS = 60


def _has_number(text: str, number: str) -> bool:
    return re.search(rf"(?<![\d.,]){re.escape(number)}(?![\d]|[.,]\d)", text) is not None


def rule_issues(case: PatientCase, calc: Calculator, text: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(param: str | None, severity: str, problem: str) -> None:
        issues.append(
            ValidationIssue(param=param, source="rule", severity=severity, problem=problem)  # type: ignore[arg-type]
        )

    if len(text.split()) < MIN_WORDS:
        add(None, "error", f"note too short ({len(text.split())} words)")
    for term in vocab.FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
            add(None, "error", f"names a score/rule: {term!r}")
    for f in build_facts(case, calc):
        for n in f.required_numbers:
            if not _has_number(text, n):
                add(f.param, "error", f"required number {n} not found")
        for n in f.leak_numbers:
            if _has_number(text, n):
                add(f.param, "warning", f"not-documented value {n} appears in the note")
    return issues


_EXPECTED = {
    DocumentedState.POSITIVE: {"stated"},
    DocumentedState.NEGATIVE: {"negated_or_normal"},
    DocumentedState.NOT_DOCUMENTED: {"not_mentioned"},
}


def parse_judge(raw: str) -> list[dict] | None:
    """Extract the JSON array from the judge's reply; None if unparseable."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def judge_issues(
    case: PatientCase, calc: Calculator, text: str, items: list[dict]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_param = {str(it.get("param")): it for it in items if isinstance(it, dict)}

    def add(param: str, severity: str, problem: str) -> None:
        issues.append(
            ValidationIssue(param=param, source="judge", severity=severity, problem=problem)  # type: ignore[arg-type]
        )

    for p in calc.parameters:
        it = by_param.get(p.id)
        if it is None:
            add(p.id, "warning", "judge returned no verdict")
            continue
        state = case.documented[p.id]
        status = it.get("status")
        expected = _EXPECTED[state]
        if (
            isinstance(p.domain, OrdinalDomain)
            and state != DocumentedState.NOT_DOCUMENTED
            and int(case.truth[p.id]) == 0
        ):
            # Level 0 of a negatable ordinal ("ECG normal") is both a stated level and a
            # normal finding.
            expected = {"stated", "negated_or_normal"}
        if status == "implied" and state != DocumentedState.NOT_DOCUMENTED:
            # The item is in the note, but indirectly (a medication, another unit, a synonym,
            # "renal function normal"). Expected for some traps; worth a look, not a failure.
            add(p.id, "warning", f"expected {state.value}, judge found it only implied")
        elif status not in expected:
            what = {
                "implied": "inferable from the note",
                "not_mentioned": "missing from the note",
                "stated": "stated in the note",
                "negated_or_normal": "negated / normal in the note",
            }.get(str(status), f"status {status!r}")
            add(p.id, "error", f"expected {state.value}, judge found it {what}")
        if (
            state != DocumentedState.NOT_DOCUMENTED
            and status == "stated"
            and isinstance(p.domain, OrdinalDomain)
        ):
            expected_level = p.domain.levels[int(case.truth[p.id])]
            if it.get("level") != expected_level:
                add(p.id, "error", f"level {it.get('level')!r}, expected {expected_level!r}")
        quote = it.get("quote")
        if quote and quote not in text:
            add(p.id, "warning", "judge quote is not an exact substring of the note")
    return issues
