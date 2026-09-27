"""qSOFA (quick Sequential Organ Failure Assessment).

Primary sources: Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of clinical criteria for
sepsis. JAMA 2016;315(8):762-774. doi:10.1001/jama.2016.0288. PMID 26903335.
Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for
Sepsis and Septic Shock (Sepsis-3). JAMA 2016;315(8):801-810. PMID 26903338.

One point each: respiratory rate >= 22/min; altered mentation; systolic BP <= 100 mmHg.
Positive if >= 2 (threshold set a priori, Seymour 2016).
Altered mentation = any GCS < 15 (Sepsis-3 operational definition; the derivation model used
GCS <= 13). REVIEWED(physician): GCS < 15 vs <= 13.
"""

import math
from collections.abc import Mapping

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory, ge, gt
from calc_bounds.types import ParamId, Value


def score(v: Mapping[ParamId, Value]) -> float:
    return float((v["resp_rate"] >= 22) + bool(v["altered_mentation"]) + (v["sbp"] <= 100))


QSOFA = Calculator(
    id="qsofa",
    name="qSOFA",
    citation="Seymour CW et al. JAMA 2016;315(8):762-774. PMID 26903335",
    parameters=(P.RESP_RATE, P.ALTERED_MENTATION, P.SBP),
    categories=(
        DecisionCategory(name="negative", lo=-math.inf, hi=2),
        DecisionCategory(name="positive", lo=2, hi=math.inf),
    ),
    score=score,
    cuts={"resp_rate": (ge(22),), "sbp": (gt(100),)},
)
