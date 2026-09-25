"""HEART score for chest pain in the emergency department.

Primary source: Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the
HEART score. Neth Heart J 2008;16(6):191-196. doi:10.1007/BF03086144. PMID 18665203.
Validation (category cutoffs, troponin bands used here): Backus BE et al. Crit Pathw Cardiol
2010;9(3):164-169 (PMID 20802272); Backus BE et al. Int J Cardiol 2013;168(3):2153-2158
(PMID 23465250); Backus BE et al. Curr Cardiol Rev 2011;7(1):2-8 (PMC3131711, Table V).

Components, 0-2 points each:
  History:  slightly (0) / moderately (1) / highly (2) suspicious (clinician judgement).
  ECG:      normal (0) / nonspecific repolarisation disturbance (1) / significant ST deviation (2).
  Age:      < 45 (0) / 45-64 (1) / >= 65 (2).
  Risk:     none (0) / 1-2 risk factors (1) / >= 3 risk factors or history of atherosclerotic
            disease (2). Risk factors: hypertension, hypercholesterolaemia, diabetes, obesity,
            smoking, family history of CAD.
  Troponin: <= normal limit (0) / 1-3x (1) / > 3x (2).
Categories: 0-3 low, 4-6 moderate, 7-10 high.
"""

import math
from collections.abc import Mapping

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory, ge
from calc_bounds.types import ParamId, Value

RISK_FACTORS = (
    P.HYPERTENSION,
    P.HYPERCHOLESTEROLEMIA,
    P.DIABETES,
    P.OBESITY,  # TODO(physician-review): BMI > 30 (Poldervaart 2013); not defined in Six 2008
    P.SMOKING,  # TODO(physician-review): recency window: < 1 month (Six 2008) vs <= 3 months
    P.FAMILY_HISTORY_CAD,  # TODO(physician-review): definition (e.g. 1st-degree relative < 65)
)


def _age_points(age: float) -> int:
    # TODO(physician-review): age 45 -> 1 point (Six 2008 text, MDCalc); Poldervaart 2013 gives 0.
    if age >= 65:
        return 2
    if age >= 45:
        return 1
    return 0


def _risk_points(v: Mapping[ParamId, Value]) -> int:
    # TODO(physician-review): whether TIA counts as atherosclerotic disease (MDCalc: yes).
    if v["atherosclerotic_disease"]:
        return 2
    n = sum(bool(v[p.id]) for p in RISK_FACTORS)
    if n >= 3:
        return 2
    return 1 if n >= 1 else 0


def score(v: Mapping[ParamId, Value]) -> float:
    # History, ECG and troponin are ordinal levels whose index equals their points.
    # TODO(physician-review): troponin bands: Six 2008 used 1-2x / > 2x; later versions 1-3x /
    # >= 3x (Backus 2011, Poldervaart 2013) or > 3x (MDCalc). The band boundary is resolved when
    # the level is assigned, not here.
    # TODO(physician-review): ECG 2 points for ST depression only (2008 table) or ST deviation
    # including elevation (2008 text, MDCalc). Level label uses "deviation".
    return float(
        int(v["heart_history"])
        + int(v["heart_ecg"])
        + _age_points(float(v["age"]))
        + _risk_points(v)
        + int(v["heart_troponin"])
    )


HEART = Calculator(
    id="heart",
    name="HEART score",
    citation="Six AJ, Backus BE, Kelder JC. Neth Heart J 2008;16(6):191-196. PMID 18665203",
    parameters=(
        P.HEART_HISTORY,
        P.HEART_ECG,
        P.AGE,
        *RISK_FACTORS,
        P.ATHEROSCLEROTIC_DISEASE,
        P.HEART_TROPONIN,
    ),
    categories=(
        DecisionCategory(name="low", lo=-math.inf, hi=4),
        DecisionCategory(name="moderate", lo=4, hi=7),
        DecisionCategory(name="high", lo=7, hi=math.inf),
    ),
    score=score,
    cuts={"age": (ge(45), ge(65))},
)
