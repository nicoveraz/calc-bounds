"""Question-asking policies. Each returns a full `Trace`.

S1 ask-all · S2 LLM agent · S3 bounds · S4 bounds + VOI + confidence echo · S3-bin ablation.
"""

from typing import Literal, Protocol

from pydantic import BaseModel

from calc_bounds.bounds import ScoreBounds
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import ExtractionResult, Extractor
from calc_bounds.llm import Usage
from calc_bounds.simulator import Answer, SimulatedClinician
from calc_bounds.types import ParamId

type PolicyId = Literal["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]


class Step(BaseModel):
    bounds: ScoreBounds | None
    """Bounds before this step's question (None for S2, which does not use bounds)."""
    question: ParamId | None
    reason: Literal["missing", "decision_relevant", "confidence_echo", "llm_choice"] | None
    answer: Answer | None


class Trace(BaseModel):
    case_id: str
    policy: PolicyId
    calculator: str
    extraction: ExtractionResult
    steps: list[Step]
    final_score: float | None
    final_category: str | None
    committed_while_undetermined: bool
    usage: Usage | None = None


class Policy(Protocol):
    id: PolicyId

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace: ...
