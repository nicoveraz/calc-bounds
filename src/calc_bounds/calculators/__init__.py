"""Calculator registry. M1 adds: heart, curb65, qsofa, perc, wells_pe, cockcroft_gault."""

from calc_bounds.calculators.base import Calculator, DecisionCategory
from calc_bounds.types import CalculatorId

REGISTRY: dict[CalculatorId, Calculator] = {}

__all__ = ["REGISTRY", "Calculator", "DecisionCategory"]
