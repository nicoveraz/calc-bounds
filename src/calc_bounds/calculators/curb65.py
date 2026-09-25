"""CURB-65 severity score for community-acquired pneumonia.

Primary source: Lim WS, van der Eerden MM, Laing R, et al. Defining community acquired
pneumonia severity on presentation to hospital: an international derivation and validation
study. Thorax 2003;58(5):377-382. doi:10.1136/thorax.58.5.377. PMID 12728155.

One point each (Figure 2):
  Confusion: Mental Test Score <= 8, or new disorientation in person, place or time.
  Urea > 7 mmol/L.
  Respiratory rate >= 30/min.
  Blood pressure: SBP < 90 mmHg or DBP <= 60 mmHg.
  Age >= 65 years.
Groups: 0-1 low (likely suitable for home treatment); 2 moderate (consider hospital-supervised
treatment); 3-5 high (manage in hospital as severe pneumonia; assess for ICU if 4-5).

Urea is in mmol/L; BUN or urea in mg/dL is converted by `units.py`. 7 mmol/L = BUN 19.6 mg/dL.
TODO(physician-review): US implementations use "BUN > 19 mg/dL" (MDCalc), which differs from
> 7 mmol/L for BUN 19.1-19.6. This code applies the original > 7 mmol/L after conversion.
"""

import math
from collections.abc import Mapping

from calc_bounds.calculators import params as P
from calc_bounds.calculators.base import Calculator, DecisionCategory, ge, gt
from calc_bounds.types import ParamId, Value


def score(v: Mapping[ParamId, Value]) -> float:
    return float(
        bool(v["confusion"])
        + (v["urea"] > 7)
        + (v["resp_rate"] >= 30)
        + (v["sbp"] < 90 or v["dbp"] <= 60)
        + (v["age"] >= 65)
    )


CURB65 = Calculator(
    id="curb65",
    name="CURB-65",
    citation="Lim WS et al. Thorax 2003;58(5):377-382. PMID 12728155",
    parameters=(P.CONFUSION, P.UREA, P.RESP_RATE, P.SBP, P.DBP, P.AGE),
    categories=(
        DecisionCategory(name="low", lo=-math.inf, hi=2),
        DecisionCategory(name="moderate", lo=2, hi=3),
        DecisionCategory(name="high", lo=3, hi=math.inf),
    ),
    score=score,
    cuts={
        "urea": (gt(7),),
        "resp_rate": (ge(30),),
        "sbp": (ge(90),),
        "dbp": (gt(60),),
        "age": (ge(65),),
    },
)
