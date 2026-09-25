"""Calculator registry."""

from collections.abc import Mapping

from calc_bounds.calculators.base import Calculator, DecisionCategory
from calc_bounds.calculators.cockcroft_gault import COCKCROFT_GAULT, make_cockcroft_gault
from calc_bounds.calculators.curb65 import CURB65
from calc_bounds.calculators.heart import HEART
from calc_bounds.calculators.perc import PERC
from calc_bounds.calculators.qsofa import QSOFA
from calc_bounds.calculators.wells_pe import WELLS_PE
from calc_bounds.types import CalculatorId

REGISTRY: dict[CalculatorId, Calculator] = {
    c.id: c for c in (HEART, CURB65, QSOFA, PERC, WELLS_PE, COCKCROFT_GAULT)
}


def get_calculator(
    calc_id: CalculatorId, options: Mapping[str, object] | None = None
) -> Calculator:
    """Look up a calculator, applying config options (`calculator_options` in RunConfig)."""
    options = dict(options or {})
    if calc_id == "cockcroft_gault" and "thresholds_ml_min" in options:
        calc = make_cockcroft_gault(options.pop("thresholds_ml_min"))  # type: ignore[arg-type]
    else:
        calc = REGISTRY[calc_id]
    if options:
        raise ValueError(f"{calc_id}: unsupported options {sorted(options)}")
    return calc


__all__ = ["REGISTRY", "Calculator", "DecisionCategory", "get_calculator"]
