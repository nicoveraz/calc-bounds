"""S1 ask-all: extract, then ask for every parameter the extraction left unknown."""

from calc_bounds.bounds import Constraint
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import Extractor
from calc_bounds.policies.base import Trace
from calc_bounds.policies.loop import Reason, run_loop
from calc_bounds.simulator import SimulatedClinician
from calc_bounds.types import ParamId


class AskAllPolicy:
    id = "s1_ask_all"

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace:
        order = [p.id for p in calc.parameters]

        def choose(
            known: dict[ParamId, Constraint], relevant: list[ParamId], asked: set[ParamId]
        ) -> tuple[ParamId, Reason] | None:
            p = next((p for p in order if p not in known and p not in asked), None)
            return None if p is None else (p, "missing")

        return run_loop(self.id, choose, case, note, calc, extractor, clinician)
