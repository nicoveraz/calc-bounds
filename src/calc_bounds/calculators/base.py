"""Calculator declaration: parameters, score function, decision categories.

Calculators are pure functions over a complete assignment of their parameters. Every
calculator module cites its primary source in the module docstring; uncertain criteria
are marked `# TODO(physician-review)` and listed in docs/CALCULATOR_NOTES.md.
"""

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict

from calc_bounds.types import CalculatorId, ParameterSpec, ParamId, Value


class DecisionCategory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    lo: float
    """Inclusive lower score bound."""
    hi: float
    """Exclusive upper score bound (use float('inf') for the top category)."""


class Calculator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: CalculatorId
    name: str
    citation: str
    parameters: tuple[ParameterSpec, ...]
    categories: tuple[DecisionCategory, ...]
    score: Callable[[Mapping[ParamId, Value]], float]
    """Pure score function over a complete assignment (canonical units)."""
    breakpoints: Mapping[ParamId, tuple[float, ...]] = {}
    """Numeric thresholds where the score is discontinuous (e.g. RR >= 22). Lets bounds
    enumerate step-function params exactly instead of using interval arithmetic."""
    monotone: Mapping[ParamId, int] = {}
    """+1 / -1 if the score is non-decreasing / non-increasing in a numeric param. Needed for
    exact interval bounds on continuous scores (Cockcroft-Gault)."""

    def category(self, score: float) -> str:
        raise NotImplementedError  # M1
