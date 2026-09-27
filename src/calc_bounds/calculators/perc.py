"""PERC (Pulmonary Embolism Rule-out Criteria).

Primary sources: Kline JA, Mitchell AM, Kabrhel C, Richman PB, Courtney DM. Clinical criteria
to prevent unnecessary diagnostic testing in emergency department patients with suspected
pulmonary embolism. J Thromb Haemost 2004;2(8):1247-1255. PMID 15304025.
Kline JA, Courtney DM, Kabrhel C, et al. Prospective multicenter evaluation of the pulmonary
embolism rule-out criteria. J Thromb Haemost 2008;6(5):772-780. PMID 18318689.

Rule negative only if all 8 criteria are met: age < 50; pulse < 100/min; SaO2 >= 95%
(2008; the 2004 abstract says > 94%); no unilateral leg swelling; no haemoptysis; no recent
surgery or trauma; no prior PE/DVT; no hormone use. Score here = number of criteria NOT met;
category negative (0) / positive (>= 1). Applies only to patients with low gestalt pretest
probability (< 15%, Kline 2008); that gate is outside this calculator.

Only abstracts were accessible; full-text variable definitions are unverified:
REVIEWED(physician): SaO2 threshold (>= 95% vs > 94%) and whether room air is required.
REVIEWED(physician): surgery/trauma definition (within 4 weeks requiring hospitalisation,
2008 abstract; some sources: requiring general anaesthesia).
REVIEWED(physician): hormone use definition (estrogen only vs any hormone therapy).
"""

import math
from collections.abc import Mapping

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory, ge
from calc_bounds.types import ParamId, Value


def score(v: Mapping[ParamId, Value]) -> float:
    return float(
        (v["age"] >= 50)
        + (v["heart_rate"] >= 100)
        + (v["spo2"] < 95)
        + bool(v["unilateral_leg_swelling"])
        + bool(v["hemoptysis"])
        + bool(v["recent_surgery_trauma"])
        + bool(v["prior_vte"])
        + bool(v["hormone_use"])
    )


PERC = Calculator(
    id="perc",
    name="PERC rule",
    citation="Kline JA et al. J Thromb Haemost 2004;2(8):1247-1255. PMID 15304025",
    parameters=(
        P.AGE,
        P.HEART_RATE,
        P.SPO2,
        P.UNILATERAL_LEG_SWELLING,
        P.HEMOPTYSIS,
        P.RECENT_SURGERY_TRAUMA,
        P.PRIOR_VTE,
        P.HORMONE_USE,
    ),
    categories=(
        DecisionCategory(name="negative", lo=-math.inf, hi=1),
        DecisionCategory(name="positive", lo=1, hi=math.inf),
    ),
    score=score,
    cuts={"age": (ge(50),), "heart_rate": (ge(100),), "spo2": (ge(95),)},
)
