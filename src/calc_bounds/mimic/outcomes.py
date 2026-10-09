"""MIMIC outcomes: row-level tables (outside the repo) and aggregate tables (safe to commit).

Reference standard: the decision category implied by the structured truth alone, when it is
determined (`truth.reference_category`); cases without one count only in outcomes that need
no reference. MIMIC has no documented-state labels, so extraction is scored against the
structured truth by EXTRACTED state:
  present  agrees when the value scores like the recorded one (same region between the
           calculator's cuts; continuous params within CONTINUOUS_REL_TOL), or equals it for
           yes/no and graded items; `exact` also requires the same number (rel. 1e-3);
  absent   agrees when the recorded value is negative / level 0 / in the "stated normal" range;
  unknown  no claim, nothing to verify. Claims about values missing from the record are
           unverifiable (NA).
A disagreement can be an extraction error or a real difference between note and record (e.g. a
later measurement); the two are not separable without annotation.

Aggregates are proportions with Wilson 95% intervals and means; groups with fewer than
`min_cell_count` cases are suppressed.
"""

import math
from typing import Any

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from calc_bounds.bounds import from_extractions, score_bounds
from calc_bounds.calculators import Calculator
from calc_bounds.extraction import ExtractionResult
from calc_bounds.mimic.truth import MimicCase, reference_category
from calc_bounds.policies import Trace
from calc_bounds.types import (
    Absent,
    BoolDomain,
    Extraction,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    Present,
    Unknown,
    Value,
)
from calc_bounds.units import UnitError, to_canonical

CONTINUOUS_REL_TOL = 0.05
"""Modelling choice: a note value of a continuous input (Cockcroft-Gault age, weight,
creatinine) within 5% of the recorded value agrees."""


def _region(calc: Calculator, pid: str, x: float) -> int:
    return sum(x >= c.at if c.upper_inclusive else x > c.at for c in calc.cuts[pid])


def claim_agrees(
    calc: Calculator, spec: ParameterSpec, e: Extraction, truth: Value | None
) -> tuple[bool | None, bool | None]:
    """(agrees, exact) for one extracted claim vs the structured value; None = unverifiable."""
    if truth is None or not isinstance(e, Present | Absent):
        return None, None
    if isinstance(e, Absent):
        match spec.domain:
            case BoolDomain():
                ok = truth is False
            case OrdinalDomain():
                ok = int(truth) == 0
            case NumericDomain():
                assert spec.absent_means is not None
                ok = spec.absent_means[0] <= float(truth) <= spec.absent_means[1]
        return ok, None
    match spec.domain:
        case BoolDomain() | OrdinalDomain():
            ok = e.value == truth
            return ok, ok
        case NumericDomain():
            try:
                x = to_canonical(spec.id, float(e.value), e.unit or spec.domain.unit)
            except (UnitError, ValueError):
                return False, False
            exact = math.isclose(x, float(truth), rel_tol=1e-3, abs_tol=1e-6)
            if spec.id in calc.continuous:
                close = math.isclose(x, float(truth), rel_tol=CONTINUOUS_REL_TOL)
                return close, exact
            return _region(calc, spec.id, x) == _region(calc, spec.id, float(truth)), exact
    raise TypeError(spec.domain)


RATIO_BUCKETS = [0.0, 0.5, 0.9, 1.1, 2.0, 2.5, 3.2, 10.0, float("inf")]
"""Edges for note/record ratios of numeric claims. 2.5-3.2 catches BUN mg/dL read as urea
mmol/L (factor 2.8); 0.9-1.1 is agreement."""


def numeric_ratio(spec: ParameterSpec, e: Extraction, truth: Value | None) -> float | None:
    """Canonical note value / record value for a present numeric claim (None otherwise)."""
    if truth is None or not isinstance(e, Present) or not isinstance(spec.domain, NumericDomain):
        return None
    try:
        x = to_canonical(spec.id, float(e.value), e.unit or spec.domain.unit)
    except (UnitError, ValueError):
        return None
    return x / float(truth) if float(truth) else None


def claim_rows(case: MimicCase, calc: Calculator, ext: ExtractionResult) -> list[dict[str, Any]]:
    rows = []
    for p in calc.parameters:
        e = ext.values.get(p.id, Unknown())
        agrees, exact = claim_agrees(calc, p, e, case.truth.get(p.id))
        ratio = numeric_ratio(p, e, case.truth.get(p.id))
        rows.append(
            {
                "case_id": case.case_id,
                "calculator": calc.id,
                "param": p.id,
                "extracted_state": e.kind,
                "in_record": p.id in case.truth,
                "agrees": agrees,
                "exact": exact,
                "confidence": getattr(e, "confidence", None),
                "confidence_source": getattr(e, "confidence_source", None),
                "unit_given": bool(getattr(e, "unit", None)) if isinstance(e, Present) else None,
                "ratio_bucket": None
                if ratio is None
                else str(pd.cut([ratio], RATIO_BUCKETS, right=False)[0]),
            }
        )
    return rows


def numeric_agreement(claims: pd.DataFrame, min_cell: int) -> pd.DataFrame:
    """Counts of present numeric claims by unit given / note-to-record ratio bucket."""
    cols = {"ratio_bucket", "unit_given"}
    if not cols <= set(claims.columns):
        return pd.DataFrame(columns=["calculator", "param", "unit_given", "ratio_bucket", "n"])
    c = claims.dropna(subset=["ratio_bucket"])
    t = (
        c.groupby(["calculator", "param", "unit_given", "ratio_bucket"], observed=True)
        .size()
        .reset_index(name="n")
    )
    t["n"] = t["n"].map(lambda n: str(n) if n >= min_cell else f"<{min_cell}")
    return t


def _risk_cmp(calc: Calculator, a: str | None, b: str | None) -> int | None:
    """-1 if category a is lower risk than b, 0 if equal, 1 if higher; None if either is None."""
    if a is None or b is None:
        return None
    order = calc.categories_by_risk()
    return (order.index(a) > order.index(b)) - (order.index(a) < order.index(b))


def _only(categories: frozenset[str]) -> str | None:
    return next(iter(categories)) if len(categories) == 1 else None


def case_row(
    case: MimicCase, calc: Calculator, ext: ExtractionResult, fallback_full_text: bool
) -> dict[str, Any]:
    """What the note alone settles, and what 'missing = normal' would conclude from it."""
    ref = reference_category(calc, case)
    note = score_bounds(calc, from_extractions(calc, ext.values)).categories
    binary = _only(score_bounds(calc, from_extractions(calc, ext.values, binary=True)).categories)
    cmp = _risk_cmp(calc, binary, ref)
    return {
        "case_id": case.case_id,
        "calculator": calc.id,
        "n_params": len(calc.parameters),
        "n_in_record": len(case.truth),
        "pending_annotation": bool(case.needs_annotation),
        "reference_category": ref,
        "reference_determined": ref is not None,
        "determined_by_note": len(note) == 1,
        "truth_within_note_bounds": None if ref is None else ref in note,
        "missing_as_normal_category": binary,
        "missing_as_normal_correct": None if cmp is None else cmp == 0,
        "missing_as_normal_under_triage": None if cmp is None else cmp < 0,
        "missing_as_normal_over_triage": None if cmp is None else cmp > 0,
        "rejected_claims": len(ext.rejected),
        "fallback_full_text": fallback_full_text,
    }


def policy_row(case: MimicCase, calc: Calculator, t: Trace) -> dict[str, Any]:
    ref = reference_category(calc, case)
    final = t.final_bounds.categories if t.final_bounds is not None else frozenset()
    conservative = t.final_category or (calc.highest_risk(final) if final else None)
    cmp = _risk_cmp(calc, conservative, ref)
    n_irrelevant = sum(
        s.question is not None
        and s.relevant is not None
        and s.reason != "confidence_echo"
        and s.question not in s.relevant
        for s in t.steps
    )
    return {
        "case_id": case.case_id,
        "calculator": calc.id,
        "policy": t.policy,
        "reference_category": ref,
        "final_category": t.final_category,
        "abstained": t.final_category is None,
        "correct": None if ref is None else t.final_category == ref,
        "conservative_correct": None if cmp is None else cmp == 0,
        "under_triage": None if cmp is None else cmp < 0,
        "over_triage": None if cmp is None else cmp > 0,
        "truth_within_final_bounds": None if ref is None else ref in final,
        "premature_commitment": t.committed_while_undetermined,
        "n_questions": t.n_questions,
        "n_irrelevant_questions": n_irrelevant,
        "n_unavailable": sum(
            s.answer is not None and s.answer.status == "not_available" for s in t.steps
        ),
        "n_echo_questions": sum(s.reason == "confidence_echo" for s in t.steps),
    }


# --- Aggregates -------------------------------------------------------------------------------

ALL = "all"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def _with_all(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Append the rows again with `col` = 'all' (pooled over that column)."""
    return pd.concat([df, df.assign(**{col: ALL})], ignore_index=True)


def proportions(df: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    """Long table: *by, metric, k, n, rate, ci_lo, ci_hi. NA values are outside the denominator."""
    rows = []
    for keys, g in df.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in metrics:
            vals = g[m].dropna().astype(bool)
            k, n = int(vals.sum()), len(vals)
            lo, hi = wilson(k, n)
            rows.append(
                dict(zip(by, keys, strict=True))
                | {"metric": m, "k": k, "n": n, "rate": k / n if n else math.nan}
                | {"ci_lo": lo, "ci_hi": hi}
            )
    return pd.DataFrame(rows, columns=[*by, "metric", "k", "n", "rate", "ci_lo", "ci_hi"])


def question_summary(policy_rows: pd.DataFrame) -> pd.DataFrame:
    g = _with_all(policy_rows, "calculator").groupby(["calculator", "policy"], sort=True)
    out = pd.DataFrame(
        {
            "n_cases": g.size(),
            "mean_questions": g["n_questions"].mean(),
            "sd_questions": g["n_questions"].std(),
            "median_questions": g["n_questions"].median(),
            "mean_unavailable_answers": g["n_unavailable"].mean(),
            "mean_echo_questions": g["n_echo_questions"].mean(),
            "irrelevant_questions": g["n_irrelevant_questions"].sum(),
            "total_questions": g["n_questions"].sum(),
        }
    )
    return out.reset_index()


def suppress(df: pd.DataFrame, n_col: str, min_cell: int, keys: list[str]) -> pd.DataFrame:
    """Blank every non-key value in rows whose `n_col` is below `min_cell`."""
    small = df[n_col] < min_cell
    out = df.copy()
    values = [c for c in df.columns if c not in keys]
    out[values] = out[values].astype(object)
    out.loc[small, values] = None
    out["suppressed"] = small
    return out


def suppress_counts(df: pd.DataFrame, count_cols: list[str], min_cell: int) -> pd.DataFrame:
    """Blank individual counts in [1, min_cell) (zeros and large counts are kept)."""
    out = df.copy()
    for c in count_cols:
        small = (out[c] > 0) & (out[c] < min_cell)
        out[c] = out[c].astype(object).where(~small, f"<{min_cell}")
    return out


IDENTIFIER_COLUMNS = {"case_id", "subject_id", "hadm_id", "stay_id", "note_id"}

NOTE_METRICS = [
    "reference_determined",
    "determined_by_note",
    "truth_within_note_bounds",
    "missing_as_normal_correct",
    "missing_as_normal_under_triage",
    "missing_as_normal_over_triage",
    "pending_annotation",
    "fallback_full_text",
]
POLICY_METRICS = [
    "correct",
    "abstained",
    "conservative_correct",
    "under_triage",
    "over_triage",
    "truth_within_final_bounds",
    "premature_commitment",
]


def aggregate_tables(
    case_rows: pd.DataFrame,
    policy_rows: pd.DataFrame,
    claims: pd.DataFrame,
    min_cell: int,
) -> dict[str, pd.DataFrame]:
    """Aggregate-only tables (no identifiers), with small groups suppressed."""
    note = proportions(_with_all(case_rows, "calculator"), ["calculator"], NOTE_METRICS)
    pol = proportions(
        _with_all(policy_rows, "calculator"), ["calculator", "policy"], POLICY_METRICS
    )
    ext = proportions(
        _with_all(claims, "calculator"),
        ["calculator", "param", "extracted_state"],
        ["in_record", "agrees", "exact"],
    )
    q = question_summary(policy_rows)
    tables = {
        "note_alone": suppress(note, "n", min_cell, ["calculator", "metric"]),
        "policies": suppress(pol, "n", min_cell, ["calculator", "policy", "metric"]),
        "extraction": suppress(
            ext, "n", min_cell, ["calculator", "param", "extracted_state", "metric"]
        ),
        "questions": suppress(q, "n_cases", min_cell, ["calculator", "policy"]),
        "numeric_agreement": numeric_agreement(claims, min_cell),
    }
    for name, t in tables.items():
        leaked = IDENTIFIER_COLUMNS & set(t.columns)
        assert not leaked, f"{name}: identifier columns {leaked} in an aggregate table"
    return tables
