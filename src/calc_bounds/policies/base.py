"""Policy trace models and protocol.

S1 ask-all · S2 LLM agent · S3 bounds · S4 bounds + VOI + confidence echo · S3-bin ablation.
A policy that cannot determine the category (e.g. a relevant answer was not available)
abstains: `final_category` is None. It never guesses.
"""

from typing import Literal, Protocol

from pydantic import BaseModel

from calc_bounds.bounds import Constraint, ScoreBounds
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.extraction import ExtractionResult, Extractor
from calc_bounds.llm import Usage
from calc_bounds.simulator import Answer, SimulatedClinician
from calc_bounds.types import ParamId


class CaseRef(Protocol):
    """What the code policies (S1, S3, S3-bin, S4) need from a case: only its id. Synthetic
    `PatientCase`s and MIMIC cases (partial structured truth) both qualify. S2 needs the full
    `PatientCase`."""

    @property
    def case_id(self) -> str: ...


type PolicyId = Literal["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]


class Step(BaseModel):
    bounds: ScoreBounds | None
    """Bounds before this step's question (None for S2, which does not use bounds)."""
    relevant: list[ParamId] | None
    """Decision-relevant missing params before this question (for irrelevant-question rate)."""
    question: ParamId | None
    reason: Literal["missing", "decision_relevant", "confidence_echo", "llm_choice"] | None
    answer: Answer | None


class Trace(BaseModel):
    case_id: str
    policy: PolicyId
    calculator: str
    extraction: ExtractionResult
    initial_known: dict[ParamId, Constraint]
    """Constraints after converting the extraction (before any question). A constraint on a
    not-documented param is a silent missing-as-absent error (or an extraction error)."""
    steps: list[Step]
    final_bounds: ScoreBounds | None
    final_score: float | None
    """Only when every parameter is known exactly."""
    final_category: str | None
    """None = abstained (category still undetermined)."""
    committed_while_undetermined: bool
    usage: Usage | None = None

    @property
    def n_questions(self) -> int:
        return sum(s.question is not None for s in self.steps)


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
