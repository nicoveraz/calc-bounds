"""Wells criteria for pulmonary embolism, two-tier (PE likely / unlikely).

Primary source: Wells PS, Anderson DR, Rodger M, et al. Derivation of a simple clinical model
to categorize patients probability of pulmonary embolism: increasing the models utility with
the SimpliRED D-dimer. Thromb Haemost 2000;83(3):416-420. PMID 10744147.
Two-tier cutoff: van Belle A, et al. (Christopher Study). JAMA 2006;295(2):172-179.
PMID 16403929. Item wording cross-checked with NICE NG158, Table 2.

Points: clinical signs of DVT 3; alternative diagnosis less likely than PE 3; heart rate
> 100/min 1.5; immobilisation or surgery in the previous 4 weeks 1.5; previous DVT/PE 1.5;
haemoptysis 1; malignancy 1. PE unlikely if <= 4, likely if > 4.

Only abstracts of Wells 2000/2001 were accessible:
REVIEWED(physician): immobilisation "> 3 days" (NICE) vs ">= 3 days" (MDCalc).
REVIEWED(physician): "alternative diagnosis less likely than PE" (NICE) vs MDCalc's
"PE #1 diagnosis OR equally likely".
"""

import math
from collections.abc import Mapping

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory, gt
from calc_bounds.types import ParamId, Value


def score(v: Mapping[ParamId, Value]) -> float:
    return (
        3.0 * bool(v["dvt_signs"])
        + 3.0 * bool(v["pe_most_likely"])
        + 1.5 * (v["heart_rate"] > 100)
        + 1.5 * bool(v["immobilization_or_surgery"])
        + 1.5 * bool(v["prior_vte"])
        + 1.0 * bool(v["hemoptysis"])
        + 1.0 * bool(v["malignancy"])
    )


WELLS_PE = Calculator(
    id="wells_pe",
    name="Wells criteria for PE (two-tier)",
    citation="Wells PS et al. Thromb Haemost 2000;83(3):416-420. PMID 10744147",
    parameters=(
        P.DVT_SIGNS,
        P.PE_MOST_LIKELY,
        P.HEART_RATE,
        P.IMMOBILIZATION_OR_SURGERY,
        P.PRIOR_VTE,
        P.HEMOPTYSIS,
        P.MALIGNANCY,
    ),
    categories=(
        # Scores move in 0.5 steps, so "<= 4" is [.., 4.5) and "> 4" is [4.5, ..).
        DecisionCategory(name="pe_unlikely", lo=-math.inf, hi=4.5),
        DecisionCategory(name="pe_likely", lo=4.5, hi=math.inf),
    ),
    score=score,
    cuts={"heart_rate": (gt(100),)},
)
