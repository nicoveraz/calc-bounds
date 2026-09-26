"""Deterministic simulated clinician. Answers from hidden truth; no LLM involved."""

import math
import zlib
from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel

from calc_bounds.types import (
    BoolDomain,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    ParamId,
    Value,
)


class Answer(BaseModel):
    param: ParamId
    status: Literal["answered", "range", "not_available"]
    value: Value | None = None
    lo: float | None = None
    hi: float | None = None
    """For status "range": a vague numeric answer that contains the true value."""
    noise: Literal["none", "wrong", "vague", "dont_know"] = "none"
    """What the (noisy) clinician did; for analysis only - policies never see this."""


class ClinicianNoise(BaseModel):
    """Noisy-clinician model. Rates are per question; drawn deterministically per
    (seed, case, param), so they do not depend on question order."""

    dont_know_rate: float = 0.0
    wrong_rate: float = 0.0
    """Yes/no flipped; graded off by one level; numeric off by 10-25% (misremembered)."""
    vague_rate: float = 0.0
    """Numeric only: a range of +/- `vague_width` around the true value instead of it."""
    vague_width: float = 0.1


def draw_unavailable(
    params: list[ParamId], rates: Mapping[ParamId, float], seed: int
) -> frozenset[ParamId]:
    """Which params this clinician cannot answer (e.g. troponin not yet resulted).
    Deterministic per (seed, param), independent of question order."""
    out = set()
    for p in params:
        rate = rates.get(p, 0.0)
        if rate > 0:
            rng = np.random.default_rng([seed, zlib.crc32(p.encode())])
            if rng.random() < rate:
                out.add(p)
    return frozenset(out)


class SimulatedClinician:
    """Answers questions from `truth`; params in `unavailable` return not_available.
    With `noise` (and `specs` for the parameter domains) it may also not know, answer wrongly,
    or answer a numeric value vaguely. Logs every question in `log`."""

    def __init__(
        self,
        truth: Mapping[ParamId, Value],
        unavailable: frozenset[ParamId] = frozenset(),
        *,
        specs: Mapping[ParamId, ParameterSpec] | None = None,
        noise: ClinicianNoise | None = None,
        seed: int = 0,
    ) -> None:
        self.truth = dict(truth)
        self.unavailable = unavailable
        self.specs = specs or {}
        self.noise = noise
        self.seed = seed
        self.log: list[Answer] = []

    def ask(self, param: ParamId) -> Answer:
        if param not in self.truth:
            raise KeyError(f"clinician has no truth for {param!r}")
        if param in self.unavailable:
            answer = Answer(param=param, status="not_available")
        elif self.noise is None or param not in self.specs:
            answer = Answer(param=param, status="answered", value=self.truth[param])
        else:
            answer = self._noisy(param)
        self.log.append(answer)
        return answer

    def _noisy(self, param: ParamId) -> Answer:
        assert self.noise is not None
        n, spec, v = self.noise, self.specs[param], self.truth[param]
        rng = np.random.default_rng([self.seed, zlib.crc32(param.encode()), 7])
        u = rng.random()
        if u < n.dont_know_rate:
            return Answer(param=param, status="not_available", noise="dont_know")
        u -= n.dont_know_rate
        if u < n.wrong_rate:
            return Answer(param=param, status="answered", value=_wrong(spec, v, rng), noise="wrong")
        u -= n.wrong_rate
        if u < n.vague_rate and isinstance(spec.domain, NumericDomain):
            lo = max(spec.domain.lo, math.floor(float(v) * (1 - n.vague_width)))
            hi = min(spec.domain.hi, math.ceil(float(v) * (1 + n.vague_width)))
            return Answer(param=param, status="range", lo=lo, hi=hi, noise="vague")
        return Answer(param=param, status="answered", value=v)


def _wrong(spec: ParameterSpec, v: Value, rng: np.random.Generator) -> Value:
    match spec.domain:
        case BoolDomain():
            return not v
        case OrdinalDomain(levels=levels):
            moves = [x for x in (int(v) - 1, int(v) + 1) if 0 <= x < len(levels)]
            return int(rng.choice(moves))
        case NumericDomain(lo=lo, hi=hi):
            factor = 1 + rng.choice([-1, 1]) * rng.uniform(0.10, 0.25)
            x = float(v) * factor
            x = round(x) if float(v).is_integer() else round(x, 2)
            return float(min(hi, max(lo, x)))
    raise TypeError(spec.domain)
