"""Sampling distributions for synthetic hidden truth (seeded numpy Generators only)."""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_bounds.types import Value


class _Dist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Bernoulli(_Dist):
    kind: Literal["bernoulli"] = "bernoulli"
    p: float = Field(ge=0.0, le=1.0)

    def sample(self, rng: np.random.Generator) -> Value:
        return bool(rng.random() < self.p)


class Categorical(_Dist):
    """Over ordinal level indices 0..len(probs)-1."""

    kind: Literal["categorical"] = "categorical"
    probs: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> "Categorical":
        if abs(sum(self.probs) - 1.0) > 1e-9 or min(self.probs) < 0:
            raise ValueError(f"probs must be non-negative and sum to 1: {self.probs}")
        return self

    def sample(self, rng: np.random.Generator) -> Value:
        return int(rng.choice(len(self.probs), p=self.probs))


class TruncNormal(_Dist):
    """Normal(mean, sd) truncated to [lo, hi] by rejection, rounded to `decimals`."""

    kind: Literal["truncnormal"] = "truncnormal"
    mean: float
    sd: float = Field(gt=0)
    lo: float
    hi: float
    decimals: int = 0

    def sample(self, rng: np.random.Generator) -> Value:
        for _ in range(10_000):
            x = round(float(rng.normal(self.mean, self.sd)), self.decimals)
            if self.lo <= x <= self.hi:
                return x
        raise RuntimeError(f"could not sample within [{self.lo}, {self.hi}]: {self}")


type Distribution = Annotated[Bernoulli | Categorical | TruncNormal, Field(discriminator="kind")]
