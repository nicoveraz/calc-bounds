"""Shared ask-and-recompute loop for code-driven policies (S1, S3, S3-bin)."""

from collections.abc import Callable
from typing import Literal

from calc_bounds.bounds import (
    Constraint,
    Exact,
    decision_relevant_missing,
    from_extractions,
    score_bounds,
)
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import Extractor
from calc_bounds.policies.base import PolicyId, Step, Trace
from calc_bounds.simulator import SimulatedClinician
from calc_bounds.types import ParamId

type Chooser = Callable[[dict[ParamId, Constraint], list[ParamId], set[ParamId]], ParamId | None]
"""(known, relevant, already_asked) -> next param to ask, or None to stop."""


def run_loop(
    policy: PolicyId,
    reason: Literal["missing", "decision_relevant"],
    choose: Chooser,
    case: PatientCase,
    note: str,
    calc: Calculator,
    extractor: Extractor,
    clinician: SimulatedClinician,
    *,
    binary: bool = False,
) -> Trace:
    extraction = extractor.extract(case.case_id, note, list(calc.parameters))
    known = from_extractions(calc, extraction.values, binary=binary)
    initial_known = dict(known)
    asked: set[ParamId] = set()
    steps: list[Step] = []
    while True:
        bounds = score_bounds(calc, known)
        relevant = decision_relevant_missing(calc, known)
        param = choose(known, relevant, asked)
        if param is None:
            break
        asked.add(param)
        answer = clinician.ask(param)
        steps.append(
            Step(bounds=bounds, relevant=relevant, question=param, reason=reason, answer=answer)
        )
        if answer.status == "answered":
            assert answer.value is not None
            known[param] = Exact(value=answer.value)
    determined = len(bounds.categories) == 1
    exact = {p: c.value for p, c in known.items() if isinstance(c, Exact)}
    complete = len(exact) == len(calc.parameters)
    return Trace(
        case_id=case.case_id,
        policy=policy,
        calculator=calc.id,
        extraction=extraction,
        initial_known=initial_known,
        steps=steps,
        final_bounds=bounds,
        final_score=calc.score(exact) if complete else None,
        final_category=next(iter(bounds.categories)) if determined else None,
        committed_while_undetermined=False,
    )
