"""Error attribution: why each wrong final category was wrong.

Categories (checked in order):
  abstained    - no category given (a decision-relevant answer was unavailable, or S2 gave up)
  extraction   - the initial extraction contained an incorrect claim (code policies), or for
                 S2 the agent committed while the category was already determined by what it
                 knew (so it misread or miscomputed; there is no separate extraction step)
  routing      - committed while the category was still undetermined (stopped asking too
                 early, or reasoned over a binary state that hid missing information)
  clinician    - the (noisy) clinician gave a wrong answer to a question the policy asked
  bounds       - knowledge was correct and determined, yet the category was wrong (would mean a
                 bounds bug; expected to be zero)
"""

import pandas as pd

from calc_bounds.cohort import PatientCase
from calc_bounds.eval.metrics import claim_correct
from calc_bounds.policies import Trace


def attribute(case: PatientCase, t: Trace) -> str | None:
    if t.final_category == case.true_category:
        return None
    if t.final_category is None:
        return "abstained"
    if any(s.answer is not None and s.answer.noise == "wrong" for s in t.steps):
        return "clinician"
    if t.policy == "s2_llm_agent":
        return "routing" if t.committed_while_undetermined else "extraction"
    if any(not claim_correct(case, pid, e) for pid, e in t.extraction.values.items()):
        return "extraction"
    if t.committed_while_undetermined:
        return "routing"
    return "bounds"


def attribution_table(cases: list[PatientCase], traces: list[Trace]) -> pd.DataFrame:
    by_id = {c.case_id: c for c in cases}
    rows = []
    for t in traces:
        cause = attribute(by_id[t.case_id], t)
        if cause is not None:
            rows.append(
                {
                    "policy": t.policy,
                    "calculator": t.calculator,
                    "case_id": t.case_id,
                    "cause": cause,
                }
            )
    df = pd.DataFrame(rows, columns=["policy", "calculator", "case_id", "cause"])
    return df
