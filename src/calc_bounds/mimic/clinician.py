"""The EHR as the clinician. Same interface as `SimulatedClinician` (`ask`, `log`).

A question is answered by looking up the structured value (`MimicCase.truth`, including any
physician annotations). "not available" means the value is truly absent from the record; it
is never turned into a negative or normal answer.
"""

from collections.abc import Mapping

from calc_bounds.simulator import Answer, SimulatedClinician
from calc_bounds.types import ParamId, Value


class EHRClinician(SimulatedClinician):
    def __init__(self, record: Mapping[ParamId, Value]) -> None:
        super().__init__(record)

    def ask(self, param: ParamId) -> Answer:
        if param not in self.truth:
            answer = Answer(param=param, status="not_available")
            self.log.append(answer)
            return answer
        return super().ask(param)
