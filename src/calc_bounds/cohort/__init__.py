"""Structured synthetic patient generator. Output: JSONL of `PatientCase` with full ground truth."""

from enum import StrEnum

from pydantic import BaseModel

from calc_bounds.config import CohortConfig
from calc_bounds.types import CalculatorId, DocumentedState, ParamId, Value


class TrapKind(StrEnum):
    MULTIPLE_ENCOUNTERS = "multiple_encounters"
    NEGATION = "negation"
    MIXED_UNITS = "mixed_units"
    CONTRADICTORY_VALUES = "contradictory_values"
    COMORBIDITY_VIA_MEDICATION = "comorbidity_via_medication"


class Trap(BaseModel):
    kind: TrapKind
    param: ParamId
    detail: dict[str, str | float | bool] = {}
    """Trap-specific instructions for the renderer (e.g. the distractor unit or old value)."""


class PatientCase(BaseModel):
    case_id: str
    seed: int
    calculator: CalculatorId
    truth: dict[ParamId, Value]
    documented: dict[ParamId, DocumentedState]
    traps: list[Trap] = []
    determined_from_note: bool
    """Whether the category is determined from documented params alone (coverage tag)."""


def generate_cohort(config: CohortConfig, seed: int) -> list[PatientCase]:
    raise NotImplementedError  # M2
