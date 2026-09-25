"""Core data model shared by every module.

Two layers per parameter:
  1. Hidden truth (`Value`): the patient's actual value. Used only by the simulator and scoring.
  2. Documented state (`DocumentedState`): what the note says about it.

The extractor's output per parameter is the tri-state `Extraction` = Present | Absent | Unknown.
Unknown must never be treated as Absent, except in the explicit S3-bin ablation.
"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type ParamId = str
type CalculatorId = str
type Value = bool | int | float
"""bool for boolean params, int for ordinal level index, float for numeric (canonical unit)."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --- Parameter domains -------------------------------------------------------------------------


class BoolDomain(Frozen):
    kind: Literal["bool"] = "bool"


class OrdinalDomain(Frozen):
    kind: Literal["ordinal"] = "ordinal"
    levels: tuple[str, ...]
    """Level labels in increasing order; values are indices into this tuple."""


class NumericDomain(Frozen):
    kind: Literal["numeric"] = "numeric"
    unit: str
    """Canonical unit (see `units.py`). All values in code are in this unit."""
    lo: float
    hi: float
    """Plausible physiological range; the interval used by bounds when the value is unknown."""


type Domain = Annotated[BoolDomain | OrdinalDomain | NumericDomain, Field(discriminator="kind")]


class ParameterSpec(Frozen):
    """A clinical parameter. Parameters are shared across calculators (e.g. age, SBP)."""

    id: ParamId
    label: str
    domain: Domain
    absent_means: tuple[float, float] | None = None
    """For numeric params: the interval implied by 'documented negative / stated normal'.
    None means an explicit negation is not meaningful for this parameter (it can only be
    Present or Unknown). For bool params Absent means False; for ordinal, level 0."""


# --- Documented state and extraction ------------------------------------------------------------


class DocumentedState(StrEnum):
    POSITIVE = "documented_positive"
    NEGATIVE = "documented_negative"
    NOT_DOCUMENTED = "not_documented"


type ConfidenceSource = Literal["oracle", "logprob", "self_reported", "calibrated"]


class EvidenceSpan(Frozen):
    """Character offsets into the note; `text` must equal note[start:end] exactly."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


class Present(Frozen):
    kind: Literal["present"] = "present"
    value: Value
    unit: str | None = None
    """Unit as written in the note (raw); code normalizes to the canonical unit."""
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_source: ConfidenceSource
    evidence: EvidenceSpan


class Absent(Frozen):
    kind: Literal["absent"] = "absent"
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_source: ConfidenceSource
    evidence: EvidenceSpan


class Unknown(Frozen):
    kind: Literal["unknown"] = "unknown"


type Extraction = Annotated[Present | Absent | Unknown, Field(discriminator="kind")]
