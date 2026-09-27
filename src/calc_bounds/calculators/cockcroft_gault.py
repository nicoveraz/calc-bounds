"""Cockcroft-Gault creatinine clearance, with configurable decision thresholds.

Primary source: Cockcroft DW, Gault MH. Prediction of creatinine clearance from serum
creatinine. Nephron 1976;16(1):31-41. doi:10.1159/000180580. PMID 1244564.

  CrCl (mL/min) = (140 - age) x weight (kg) / (72 x serum creatinine (mg/dL)), x 0.85 if female.

Derived in men; the female factor ("15% less in females") was estimated, not derived.
Assumes steady-state creatinine. Default thresholds (30, 60 mL/min) follow the lower cut
points of FDA Guidance for Industry, Pharmacokinetics in Patients with Impaired Renal Function
(March 2024), Table 1 (normal >= 90, mild 60-<90, moderate 30-<60, severe < 30).

REVIEWED(physician): weight = actual body weight (as in the original); ideal/adjusted weight
in obesity is not modelled.
REVIEWED(physician): default thresholds for the decision categories (config-overridable).
"""

import itertools
import math
from collections.abc import Mapping, Sequence

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory
from calc_bounds.types import ParamId, Value

FEMALE_FACTOR = 0.85
DEFAULT_THRESHOLDS = (30.0, 60.0)


def score(v: Mapping[ParamId, Value]) -> float:
    crcl = (140 - float(v["age"])) * float(v["weight"]) / (72 * float(v["creatinine"]))
    return crcl * FEMALE_FACTOR if v["sex"] == 1 else crcl


def _fmt(x: float) -> str:
    return f"{x:g}"


def _categories(thresholds: Sequence[float]) -> tuple[DecisionCategory, ...]:
    ts = sorted(float(t) for t in thresholds)
    if not ts or len(set(ts)) != len(ts):
        raise ValueError(f"thresholds must be non-empty and distinct: {thresholds}")
    cats = [DecisionCategory(name=f"crcl_lt_{_fmt(ts[0])}", lo=-math.inf, hi=ts[0])]
    for a, b in itertools.pairwise(ts):
        cats.append(DecisionCategory(name=f"crcl_{_fmt(a)}_to_lt_{_fmt(b)}", lo=a, hi=b))
    cats.append(DecisionCategory(name=f"crcl_ge_{_fmt(ts[-1])}", lo=ts[-1], hi=math.inf))
    return tuple(cats)


def make_cockcroft_gault(thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> Calculator:
    return Calculator(
        id="cockcroft_gault",
        name="Cockcroft-Gault creatinine clearance",
        citation="Cockcroft DW, Gault MH. Nephron 1976;16(1):31-41. PMID 1244564",
        parameters=(P.AGE, P.WEIGHT, P.CREATININE, P.SEX),
        categories=_categories(thresholds),
        score=score,
        # Strictly monotone over the plausible ranges (age < 140, weight > 0, creatinine > 0).
        continuous=frozenset({"age", "weight", "creatinine"}),
        higher_score_is_higher_risk=False,
    )


COCKCROFT_GAULT = make_cockcroft_gault()
