"""Score bounds over partially known parameters, and decision relevance.

Discrete domains (bool, ordinal, numeric-with-breakpoints) are enumerated exactly; continuous
numeric params use interval arithmetic over the physiological range (exact at interval corners
when the score is declared monotone in that param).
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from calc_bounds.calculators import Calculator
from calc_bounds.types import ParamId, Value


class Exact(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["exact"] = "exact"
    value: Value


class Interval(BaseModel):
    """Numeric param known to lie in [lo, hi] (e.g. 'stated normal')."""

    model_config = ConfigDict(frozen=True)
    kind: Literal["interval"] = "interval"
    lo: float
    hi: float


type Constraint = Annotated[Exact | Interval, Field(discriminator="kind")]
type Partial = Mapping[ParamId, Constraint]
"""A partial assignment. Params absent from the mapping are unknown (full domain)."""


class ScoreBounds(BaseModel):
    lo: float
    hi: float
    scores: frozenset[float] | None
    """Exact achievable score set when enumerable; None for continuous scores."""
    categories: frozenset[str]


def score_bounds(calc: Calculator, known: Partial) -> ScoreBounds:
    raise NotImplementedError  # M1


def is_determined(calc: Calculator, known: Partial) -> bool:
    """True iff exactly one decision category is possible."""
    raise NotImplementedError  # M1


def decision_relevant_missing(calc: Calculator, known: Partial) -> list[ParamId]:
    """Unknown params whose resolution could change the set of possible categories."""
    raise NotImplementedError  # M1


def voi_order(
    calc: Calculator, known: Partial, beliefs: Mapping[ParamId, Mapping[Value, float]]
) -> list[ParamId]:
    """Rank relevant unknowns by P(asking this one alone determines the category), greedy."""
    raise NotImplementedError  # M1
