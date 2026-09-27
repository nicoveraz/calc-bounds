"""Calculator declaration: parameters, score function, decision categories.

Calculators are pure functions over a complete assignment of their parameters (canonical
units). Every calculator module cites its primary source in its docstring; uncertain criteria
are marked `# REVIEWED(physician)` and listed in docs/CALCULATOR_NOTES.md.

Every numeric parameter must be declared either as a *step* parameter (the score depends on it
only through thresholds, listed in `cuts`) or as a *continuous* parameter (the score is
continuous and strictly monotone in it). This is what lets `bounds` be exact.
"""

import itertools
import math
from collections.abc import Callable, Mapping
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from calc_bounds.types import CalculatorId, NumericDomain, ParameterSpec, ParamId, Value


class Cut(BaseModel):
    """A threshold at which a step parameter's contribution changes.

    `upper_inclusive=True` means the value `at` belongs to the region above it (a `>= at`
    criterion); False means it belongs to the region below (a `> at` criterion, or `<= at`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    at: float
    upper_inclusive: bool


def ge(at: float) -> Cut:
    """Cut for a criterion `x >= at` (equivalently, the complement `x < at`)."""
    return Cut(at=at, upper_inclusive=True)


def gt(at: float) -> Cut:
    """Cut for a criterion `x > at` (equivalently, the complement `x <= at`)."""
    return Cut(at=at, upper_inclusive=False)


class DecisionCategory(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    lo: float
    """Inclusive lower score bound."""
    hi: float
    """Exclusive upper score bound (math.inf for the top category)."""


class Calculator(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: CalculatorId
    name: str
    citation: str
    parameters: tuple[ParameterSpec, ...]
    categories: tuple[DecisionCategory, ...]
    score: Callable[[Mapping[ParamId, Value]], float]
    """Pure score function over a complete assignment (canonical units)."""
    cuts: Mapping[ParamId, tuple[Cut, ...]] = {}
    """Step numeric params and their thresholds."""
    continuous: frozenset[ParamId] = frozenset()
    """Numeric params in which the score is continuous and strictly monotone."""
    higher_score_is_higher_risk: bool = True
    """False for Cockcroft-Gault (lower clearance = more dose adjustment needed)."""

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [p.id for p in self.parameters]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.id}: duplicate parameters")
        numeric = {p.id for p in self.parameters if isinstance(p.domain, NumericDomain)}
        declared = set(self.cuts) | self.continuous
        if set(self.cuts) & self.continuous:
            raise ValueError(f"{self.id}: a param cannot be both step and continuous")
        if declared != numeric:
            raise ValueError(
                f"{self.id}: numeric params {sorted(numeric)} must each be declared in exactly "
                f"one of cuts/continuous (declared {sorted(declared)})"
            )
        cats = sorted(self.categories, key=lambda c: c.lo)
        if cats[-1].hi != math.inf:
            raise ValueError(f"{self.id}: top category must extend to inf")
        for a, b in itertools.pairwise(cats):
            if a.hi != b.lo:
                raise ValueError(f"{self.id}: categories {a.name}/{b.name} not contiguous")
        return self

    def param(self, pid: ParamId) -> ParameterSpec:
        for p in self.parameters:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def category(self, score: float) -> str:
        for c in self.categories:
            if c.lo <= score < c.hi:
                return c.name
        raise ValueError(f"{self.id}: score {score} outside all categories")

    def category_names(self) -> list[str]:
        """Category names in increasing score order."""
        return [c.name for c in sorted(self.categories, key=lambda c: c.lo)]

    def categories_by_risk(self) -> list[str]:
        """Category names from lowest to highest risk."""
        names = self.category_names()
        return names if self.higher_score_is_higher_risk else names[::-1]

    def highest_risk(self, categories: frozenset[str] | set[str]) -> str:
        return max(categories, key=self.categories_by_risk().index)

    def evaluate(self, values: Mapping[ParamId, Value]) -> tuple[float, str]:
        """Score and category for a complete assignment."""
        missing = [p.id for p in self.parameters if p.id not in values]
        if missing:
            raise ValueError(f"{self.id}: missing parameters {missing}")
        s = self.score(values)
        return s, self.category(s)
