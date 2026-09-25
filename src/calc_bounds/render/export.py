"""Physician review export: a seeded random sample as Markdown, structure beside the note."""

import html

import numpy as np

from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.render import JudgeResult, RenderedNote, ValidationIssue
from calc_bounds.render.facts import fmt
from calc_bounds.types import DocumentedState, OrdinalDomain


def _value(calc: Calculator, pid: str, v: object) -> str:
    p = calc.param(pid)
    if isinstance(p.domain, OrdinalDomain):
        return p.domain.levels[int(v)]  # type: ignore[call-overload]
    return fmt(v)  # type: ignore[arg-type]


_STATE = {
    DocumentedState.POSITIVE: "stated",
    DocumentedState.NEGATIVE: "negated / normal",
    DocumentedState.NOT_DOCUMENTED: "**not documented**",
}


def _structure_table(case: PatientCase, calc: Calculator) -> str:
    traps = {t.param: t for t in case.traps}
    rows = ["| Parameter | Truth | In note | Trap |", "|---|---|---|---|"]
    for p in calc.parameters:
        t = traps.get(p.id)
        trap = f"{t.kind.value} {t.detail}" if t else ""
        rows.append(
            f"| {p.id} | {_value(calc, p.id, case.truth[p.id])} | "
            f"{_STATE[case.documented[p.id]]} | {html.escape(trap)} |"
        )
    return "\n".join(rows)


def _issues(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "_No validation issues._"
    return "\n".join(
        f"- **{i.severity}** ({i.source}) `{i.param or '-'}`: {i.problem}" for i in issues
    )


def export_review_sample(
    notes: list[RenderedNote],
    cases: dict[str, PatientCase],
    calcs: dict[str, Calculator],
    judged: dict[tuple[str, str], JudgeResult],
    fraction: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    k = max(1, round(fraction * len(notes)))
    idx = sorted(rng.choice(len(notes), size=min(k, len(notes)), replace=False))
    sample = [notes[i] for i in idx]
    out = [
        f"# Note review sample ({len(sample)} of {len(notes)} notes, seed {seed})",
        "",
        "For each note: check that stated items match the truth, negations are clear, and "
        "items marked **not documented** are absent and not inferable. Record problems below "
        "each note.",
        "",
    ]
    for n in sample:
        case = cases[n.case_id]
        calc = calcs[case.calculator]
        issues = n.issues + (
            judged[(n.case_id, n.locale)].issues if (n.case_id, n.locale) in judged else []
        )
        out += [
            f"## {n.case_id} · {n.locale} · render `{n.render}` ({n.model})",
            "",
            f"True category: **{case.true_category}** (score {fmt(case.true_score)}); "
            f"determined from note: {case.determined_from_note}",
            "",
            "<table><tr><td valign='top' width='45%'>",
            "",
            _structure_table(case, calc),
            "",
            "</td><td valign='top'>",
            "",
            f"<pre style='white-space: pre-wrap'>{html.escape(n.text)}</pre>",
            "",
            "</td></tr></table>",
            "",
            "**Validation:**",
            "",
            _issues(issues),
            "",
            "**Reviewer notes:** ",
            "",
            "---",
            "",
        ]
    return "\n".join(out)
