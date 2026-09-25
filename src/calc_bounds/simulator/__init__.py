"""Deterministic simulated clinician. Answers from hidden truth; no LLM involved."""

import zlib
from collections.abc import Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel

from calc_bounds.types import ParamId, Value


class Answer(BaseModel):
    param: ParamId
    status: Literal["answered", "not_available"]
    value: Value | None = None


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
    Logs every question in `log`."""

    def __init__(
        self, truth: Mapping[ParamId, Value], unavailable: frozenset[ParamId] = frozenset()
    ) -> None:
        self.truth = dict(truth)
        self.unavailable = unavailable
        self.log: list[Answer] = []

    def ask(self, param: ParamId) -> Answer:
        if param not in self.truth:
            raise KeyError(f"clinician has no truth for {param!r}")
        if param in self.unavailable:
            answer = Answer(param=param, status="not_available")
        else:
            answer = Answer(param=param, status="answered", value=self.truth[param])
        self.log.append(answer)
        return answer
