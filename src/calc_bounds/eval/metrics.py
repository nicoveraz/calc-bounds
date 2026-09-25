"""Per-case and aggregate metrics from traces joined with the cohort's ground truth.

Abstention (final_category None) counts as incorrect for accuracy and is reported separately.
"""

import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.policies import Trace
from calc_bounds.types import Absent, DocumentedState, Present, Unknown


def case_table(cases: list[PatientCase], traces: list[Trace]) -> pd.DataFrame:
    """One row per (policy, case)."""
    by_id = {c.case_id: c for c in cases}
    rows = []
    for t in traces:
        case = by_id[t.case_id]
        n_q = t.n_questions
        n_irrelevant = sum(
            s.question is not None and s.relevant is not None and s.question not in s.relevant
            for s in t.steps
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
            "premature_commitment_rate": g["premature_commitment"].mean(),
            "mean_questions": g["n_questions"].mean(),
            "median_questions": g["n_questions"].median(),
            "irrelevant_question_rate": g["n_irrelevant_questions"].sum()
            / g["n_questions"].sum().where(lambda s: s > 0),
            "silent_missing_as_absent_per_case": g["silent_missing_as_absent"].mean(),
        }
    )
    return out.reset_index()


def extraction_table(cases: list[PatientCase], traces: list[Trace]) -> pd.DataFrame:
    """Per (policy, case, param): extraction correctness by documented state.

    Correct = Present with the true value for documented_positive, Absent for
    documented_negative, Unknown for not_documented. Numeric values are compared as given
    (the oracle reports canonical units; LLM extractors are normalized in M4).
    """
    by_id = {c.case_id: c for c in cases}
    rows = []
    for t in traces:
        case = by_id[t.case_id]
        for pid, e in t.extraction.values.items():
            state = case.documented[pid]
            match state:
                case DocumentedState.POSITIVE:
                    ok = isinstance(e, Present) and e.value == case.truth[pid]
                case DocumentedState.NEGATIVE:
                    ok = isinstance(e, Absent)
                case DocumentedState.NOT_DOCUMENTED:
                    ok = isinstance(e, Unknown)
            rows.append(
                {
                    "policy": t.policy,
                    "calculator": t.calculator,
                    "case_id": t.case_id,
                    "param": pid,
                    "documented": state.value,
                    "extracted": e.kind,
                    "correct": ok,
                }
            )
    return pd.DataFrame(rows)
