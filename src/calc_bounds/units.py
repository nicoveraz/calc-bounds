"""Unit normalization. Units are converted by code, never by the model.

Each numeric parameter has a canonical unit (the one its `NumericDomain` declares) and a small
vocabulary of accepted units, each with a multiplicative factor to the canonical unit. Unit
strings are explicit tokens; mass-concentration units for urea name the analyte
("urea_mg/dL" vs "bun_mg/dL") because "mg/dL" alone is ambiguous between urea and BUN.

Factors (see docs/CALCULATOR_NOTES.md for sources):
  urea:       1 mmol/L urea = 6.006 mg/dL urea (MW 60.06 g/mol)
              1 mmol/L urea = 2.8014 mg/dL BUN (2 N x 14.007 g/mol)
  creatinine: 1 mg/dL = 88.4 umol/L (MW 113.12 g/mol)
  weight:     1 lb = 0.45359237 kg (exact, by definition)
"""

from calc_bounds.types import ParamId

UREA_MW = 60.06
"""g/mol. Urea CO(NH2)2."""
BUN_PER_UREA = 2 * 14.007
"""mg of nitrogen per mmol of urea (two N atoms)."""
CREATININE_UMOL_PER_MG_DL = 88.4  # TODO(physician-review): 88.4 vs 88.42 (MW 113.12)
LB_TO_KG = 0.45359237

# Multiplicative factor: canonical_value = value * factor.
_FACTORS: dict[ParamId, dict[str, float]] = {
    "age": {"years": 1.0},
    "heart_rate": {"/min": 1.0},
    "resp_rate": {"/min": 1.0},
    "sbp": {"mmHg": 1.0},
    "dbp": {"mmHg": 1.0},
    "spo2": {"%": 1.0},
    "urea": {
        "mmol/L": 1.0,
        "urea_mg/dL": 10.0 / UREA_MW,  # mg/dL -> mg/L -> mmol/L
        "bun_mg/dL": 1.0 / BUN_PER_UREA * 10.0,
    },
    "creatinine": {"mg/dL": 1.0, "umol/L": 1.0 / CREATININE_UMOL_PER_MG_DL},
    "weight": {"kg": 1.0, "lb": LB_TO_KG},
}

_ALIASES: dict[str, str] = {
    "µmol/l": "umol/L",
    "μmol/l": "umol/L",
    "umol/l": "umol/L",
    "mmol/l": "mmol/L",
    "mg/dl": "mg/dL",
    "lbs": "lb",
    "kgs": "kg",
    "bpm": "/min",
    "mm hg": "mmHg",
    "mmhg": "mmHg",
}


# Decimals used when a value is displayed in a non-canonical unit (mixed-units trap).
DISPLAY_DECIMALS: dict[str, int] = {
    "urea_mg/dL": 0,
    "bun_mg/dL": 0,
    "umol/L": 0,
    "lb": 0,
}


class UnitError(ValueError):
    """Raised for unknown units or conversions that are not defined for a parameter."""


def accepted_units(param: ParamId) -> list[str]:
    """Unit tokens accepted for a numeric param (canonical unit first)."""
    try:
        return list(_FACTORS[param])
    except KeyError:
        raise UnitError(f"no units defined for parameter {param!r}") from None


def _normalize_token(unit: str) -> str:
    token = unit.strip()
    return _ALIASES.get(token.lower(), token)


def _factor(param: ParamId, unit: str) -> float:
    units = _FACTORS.get(param)
    if units is None:
        raise UnitError(f"no units defined for parameter {param!r}")
    token = _normalize_token(unit)
    if token not in units:
        raise UnitError(f"unit {unit!r} not accepted for {param!r}; expected one of {list(units)}")
    return units[token]


def to_canonical(param: ParamId, value: float, unit: str) -> float:
    """Convert `value` in `unit` to the canonical unit of `param`. Raises UnitError."""
    return value * _factor(param, unit)


def from_canonical(param: ParamId, value: float, unit: str) -> float:
    """Convert a canonical value of `param` to `unit` (used by the renderer for mixed units)."""
    return value / _factor(param, unit)
