"""Unit normalization. Units are converted by code, never by the model.

M1 implements tested conversions, e.g.:
  urea mmol/L <-> urea mg/dL <-> BUN mg/dL
  creatinine mg/dL <-> umol/L
  weight lb <-> kg
Conversion factors will be cited and listed in docs/CALCULATOR_NOTES.md.
"""

from calc_bounds.types import ParamId


class UnitError(ValueError):
    """Raised for unknown units or conversions that are not defined for a parameter."""


def to_canonical(param: ParamId, value: float, unit: str) -> float:
    """Convert `value` in `unit` to the canonical unit of `param`. Raises UnitError."""
    raise NotImplementedError  # M1


def convert(value: float, from_unit: str, to_unit: str, *, analyte: str | None = None) -> float:
    """Convert between units; `analyte` is required for mass<->molar conversions."""
    raise NotImplementedError  # M1
