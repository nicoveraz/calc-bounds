"""S4: bounds + value-of-information ordering + confidence echo.

1. Confidence echo: extracted (Present/Absent) values whose calibrated confidence is below
   `echo_threshold` and that are decision-critical (treating the value as unknown would make
   it decision-relevant) are confirmed with the clinician first, lowest confidence first.
2. Then decision-relevant unknowns are asked in VOI order: the probability that the answer
   alone determines the category, under population-prior beliefs (the cohort priors,
   discretized to the calculator's scoring regions). Ties keep calculator order.
"""

from collections.abc import Mapping

import numpy as np

from calc_bounds.bounds import (
    Constraint,
    _region_representatives,
    decision_relevant_missing,
    resolve_probability,
)
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase
from calc_bounds.distributions import Bernoulli, Categorical, Distribution, TruncNormal
from calc_bounds.extraction import Extractor
from calc_bounds.extraction.calibration import Calibrator
from calc_bounds.policies.base import Trace
from calc_bounds.policies.loop import Reason, run_loop
from calc_bounds.simulator import SimulatedClinician
from calc_bounds.types import Absent, NumericDomain, ParamId, Present, Value

N_SAMPLES = 2000


def prior_beliefs(
    calc: Calculator, priors: Mapping[ParamId, Distribution], seed: int = 0
) -> dict[ParamId, dict[Value, float]]:
    """Beliefs over each parameter's scoring-relevant values (exact for bool/ordinal; region
    representatives for step params; quintile points for continuous params)."""
    rng = np.random.default_rng(seed)
    out: dict[ParamId, dict[Value, float]] = {}
    for p in calc.parameters:
        d = priors[p.id]
        match d:
            case Bernoulli(p=q):
                out[p.id] = {False: 1 - q, True: q}
            case Categorical(probs=probs):
                out[p.id] = dict(enumerate(probs))
            case TruncNormal():
                assert isinstance(p.domain, NumericDomain)
                xs = np.array([float(d.sample(rng)) for _ in range(N_SAMPLES)])
                if p.id in calc.continuous:
                    qs = np.quantile(xs, [0.1, 0.3, 0.5, 0.7, 0.9])
                    out[p.id] = {float(q): 0.2 for q in qs}
                else:
                    reps = _region_representatives(calc.cuts[p.id], p.domain.lo, p.domain.hi)
                    # A representative stands for its region; assign each sample to the region
                    # whose representative scores the same (step params are constant per region).
                    counts: dict[float, int] = dict.fromkeys(reps, 0)
                    for x in xs:
                        counts[_nearest_region_rep(calc, p.id, float(x), reps)] += 1
                    out[p.id] = {r: c / N_SAMPLES for r, c in counts.items() if c}
    return out


def _nearest_region_rep(calc: Calculator, pid: ParamId, x: float, reps: list[float]) -> float:
    """The representative of the region containing x (regions come from the param's cuts)."""
    for r in reps:
        lo_r = _region_representatives(calc.cuts[pid], min(x, r), max(x, r))
        if len(lo_r) == 1:  # x and r lie in the same region
            return r
    raise AssertionError(f"no region for {pid}={x}")


class VoiEchoPolicy:
    id = "s4_bounds_voi_echo"

    def __init__(
        self,
        priors: Mapping[str, Mapping[ParamId, Distribution]],
        calibrator: Calibrator,
        echo_threshold: float,
    ) -> None:
        self.priors = priors
        self.calibrator = calibrator
        self.echo_threshold = echo_threshold
        self._beliefs: dict[str, dict[ParamId, dict[Value, float]]] = {}

    def beliefs(self, calc: Calculator) -> dict[ParamId, dict[Value, float]]:
        if calc.id not in self._beliefs:
            self._beliefs[calc.id] = prior_beliefs(calc, self.priors[calc.id])
        return self._beliefs[calc.id]

    def run(
        self,
        case: PatientCase,
        note: str,
        calc: Calculator,
        extractor: Extractor,
        clinician: SimulatedClinician,
    ) -> Trace:
        extraction = extractor.extract(case.case_id, note, list(calc.parameters))
        conf = {
            pid: float(self.calibrator.transform([e.confidence])[0])
            for pid, e in extraction.values.items()
            if isinstance(e, Present | Absent)
        }
        beliefs = self.beliefs(calc)

        def choose(
            known: dict[ParamId, Constraint], relevant: list[ParamId], asked: set[ParamId]
        ) -> tuple[ParamId, Reason] | None:
            # 1. Confidence echo, lowest calibrated confidence first. Runs even when the
            #    category already looks determined: a wrong value is most harmful exactly then.
            for pid in sorted(conf, key=conf.__getitem__):
                if pid in asked or conf[pid] >= self.echo_threshold:
                    continue
                without = {k: v for k, v in known.items() if k != pid}
                if pid in decision_relevant_missing(calc, without):
                    return pid, "confidence_echo"
            # 2. VOI-ordered decision-relevant unknowns.
            candidates = [p for p in relevant if p not in asked]
            if not candidates:
                return None
            probs = {p: resolve_probability(calc, known, p, beliefs.get(p)) for p in candidates}
            return max(candidates, key=lambda p: (probs[p], -candidates.index(p))), (
                "decision_relevant"
            )

        # Extractors are deterministic lookups here (oracle / precomputed), so run_loop's own
        # extract call returns the same result.
        return run_loop(self.id, choose, case, note, calc, extractor, clinician)
