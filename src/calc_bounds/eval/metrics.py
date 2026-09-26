"""Per-case and aggregate metrics from traces joined with the cohort's ground truth.

Primary outcome: `accuracy`, where abstention (final_category None) counts as incorrect.
Secondary: `coverage` (1 - abstain rate) and `accuracy_when_committed`.
"""

import math

import pandas as pd

from calc_bounds.calculators import REGISTRY
from calc_bounds.cohort import PatientCase
from calc_bounds.policies import Trace
from calc_bounds.types import Absent, DocumentedState, Extraction, OrdinalDomain, Present, Unknown
from calc_bounds.units import UnitError, to_canonical


def case_table(cases: list[PatientCase], traces: list[Trace]) -> pd.DataFrame:
    """One row per (policy, case)."""
    by_id = {c.case_id: c for c in cases}
    rows = []
    for t in traces:
        case = by_id[t.case_id]
        n_q = t.n_questions
        n_irrelevant = sum(
            s.question is not None
            and s.relevant is not None
            and s.reason != "confidence_echo"
            and s.question not in s.relevant
            for s in t.steps
        )
        echo = [s for s in t.steps if s.reason == "confidence_echo"]
        n_echo_caught = sum(
            s.answer is not None
            and s.answer.status == "answered"
            and s.question is not None
            and not claim_correct(case, s.question, t.extraction.values[s.question])
            for s in echo
        )
        not_documented = {
            p for p, s in case.documented.items() if s == DocumentedState.NOT_DOCUMENTED
        }
        rows.append(
            {
                "policy": t.policy,
                "calculator": t.calculator,
                "case_id": t.case_id,
                "determined_from_note": case.determined_from_note,
                "true_category": case.true_category,
                "final_category": t.final_category,
                "correct": t.final_category == case.true_category,
                "abstained": t.final_category is None,
                "premature_commitment": t.committed_while_undetermined,
                "n_questions": n_q,
                "n_irrelevant_questions": n_irrelevant,
                "n_echo_questions": len(echo),
                "n_echo_caught_errors": n_echo_caught,
                "n_unavailable": sum(
                    s.answer is not None and s.answer.status == "not_available" for s in t.steps
                ),
                "silent_missing_as_absent": len(not_documented & set(t.initial_known)),
            }
        )
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = table.groupby(by)
    out = pd.DataFrame(
        {
            "n_cases": g.size(),
            "accuracy": g["correct"].mean(),
            "abstain_rate": g["abstained"].mean(),
            "coverage": 1 - g["abstained"].mean(),
            "accuracy_when_committed": g.apply(
                lambda d: d.loc[~d["abstained"], "correct"].mean(), include_groups=False
            ),
            "premature_commitment_rate": g["premature_commitment"].mean(),
            "mean_questions": g["n_questions"].mean(),
            "median_questions": g["n_questions"].median(),
            "irrelevant_question_rate": g["n_irrelevant_questions"].sum()
            / g["n_questions"].sum().where(lambda s: s > 0),
            "silent_missing_as_absent_per_case": g["silent_missing_as_absent"].mean(),
            "echo_questions_per_case": g["n_echo_questions"].mean(),
            "echo_caught_errors": g["n_echo_caught_errors"].sum(),
        }
    )
    return out.reset_index()


def claim_correct(case: PatientCase, pid: str, e: Extraction) -> bool:
    """Is this tri-state extraction right, given the documented state and hidden truth?

    Present: documented positive and value equal to the truth (numeric in canonical units,
    relative tolerance 1e-3); Absent: documented negative; Unknown: not documented.
    """
    state = case.documented[pid]
    calc_params = {p.id: p for c in REGISTRY.values() for p in c.parameters}
    level_zero_equivalent = (
        isinstance(calc_params[pid].domain, OrdinalDomain)
        and calc_params[pid].negatable
        and case.truth[pid] == 0
        and state != DocumentedState.NOT_DOCUMENTED
    )
    match e:
        case Present(value=v) if level_zero_equivalent:
            # "ECG normal" as level 0 is the same claim as "ECG normal" as a negation.
            return v == 0
        case Absent() if level_zero_equivalent:
            return True
        case Present(value=v, unit=unit):
            if state != DocumentedState.POSITIVE:
                return False
            truth = case.truth[pid]
            if isinstance(truth, bool) or isinstance(v, bool):
                return bool(v) == bool(truth)
            if isinstance(truth, int) and not isinstance(truth, bool) and unit is None:
                return int(v) == truth
            try:
                canonical = to_canonical(pid, float(v), unit) if unit else float(v)
            except (UnitError, ValueError):
                return False
            return math.isclose(canonical, float(truth), rel_tol=1e-3, abs_tol=1e-6)
        case Absent():
            return state == DocumentedState.NEGATIVE
        case Unknown():
            return state == DocumentedState.NOT_DOCUMENTED
    return False


def extraction_table(cases: list[PatientCase], traces: list[Trace]) -> pd.DataFrame:
    """Per (policy, case, param): extraction correctness by documented state."""
    by_id = {c.case_id: c for c in cases}
    rows = []
    for t in traces:
        case = by_id[t.case_id]
        for pid, e in t.extraction.values.items():
            rows.append(
                {
                    "policy": t.policy,
                    "calculator": t.calculator,
                    "case_id": t.case_id,
                    "param": pid,
                    "documented": case.documented[pid].value,
                    "extracted": e.kind,
                    "correct": claim_correct(case, pid, e),
                    "confidence": getattr(e, "confidence", None),
                    "confidence_source": getattr(e, "confidence_source", None),
                }
            )
    return pd.DataFrame(rows)
