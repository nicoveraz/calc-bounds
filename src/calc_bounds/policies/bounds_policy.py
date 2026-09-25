"""S3 bounds: ask only decision-relevant missing params, one at a time, recomputing bounds after
each answer. `binary=True` is the S3-bin ablation (missing treated as absent)."""

from calc_bounds.bounds import Constraint
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import Extractor
from calc_bounds.policies.base import Trace
from calc_bounds.policies.loop import run_loop
from calc_bounds.simulator import SimulatedClinician
from calc_bounds.types import ParamId


class BoundsPolicy:
    def __init__(self, *, binary: bool = False) -> None:
        self.binary = binary
        self.id = "s3_bin" if binary else "s3_bounds"

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace:
        def choose(
            known: dict[ParamId, Constraint], relevant: list[ParamId], asked: set[ParamId]
        ) -> ParamId | None:
            # Calculator parameter order; S4 replaces this with VOI ordering.
            return next((p for p in relevant if p not in asked), None)

        return run_loop(
            self.id,
            "decision_relevant",
            choose,
            case,
            note,
            calc,
            extractor,
            clinician,
            binary=self.binary,
        )
