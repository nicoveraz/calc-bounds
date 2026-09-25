"""Hypothesis strategies: partial assignments and consistent completions for any calculator."""

from hypothesis import strategies as st

from calc_bounds.bounds import Constraint, Exact, Interval
from calc_bounds.calculators import Calculator
from calc_bounds.types import BoolDomain, NumericDomain, OrdinalDomain, ParameterSpec, Value


def _special_points(calc: Calculator, p: ParameterSpec) -> list[float]:
    """Cut points and domain ends: where off-by-one errors live."""
    assert isinstance(p.domain, NumericDomain)
    pts = [p.domain.lo, p.domain.hi]
    pts += [c.at for c in calc.cuts.get(p.id, ())]
    return [x for x in pts if p.domain.lo <= x <= p.domain.hi]


def numeric_in(calc: Calculator, p: ParameterSpec, lo: float, hi: float) -> st.SearchStrategy:
    special = [x for x in _special_points(calc, p) if lo <= x <= hi] + [lo, hi]
    return st.one_of(
        st.sampled_from(special),
        st.floats(lo, hi, allow_nan=False, allow_infinity=False),
        st.integers(int(lo) + 1, int(hi)).map(float) if int(lo) + 1 <= int(hi) else st.just(lo),
    ).filter(lambda x: lo <= x <= hi)


def value_of(calc: Calculator, p: ParameterSpec) -> st.SearchStrategy[Value]:
    match p.domain:
        case BoolDomain():
            return st.booleans()
        case OrdinalDomain(levels=levels):
            return st.integers(0, len(levels) - 1)
        case NumericDomain(lo=lo, hi=hi):
            return numeric_in(calc, p, lo, hi)
    raise TypeError(p.domain)


@st.composite
def partial_and_completion(
    draw: st.DrawFn, calc: Calculator
) -> tuple[dict[str, Constraint], dict[str, Value]]:
    """A random partial assignment plus one completion consistent with it."""
    known: dict[str, Constraint] = {}
    completion: dict[str, Value] = {}
    for p in calc.parameters:
        state = draw(st.sampled_from(["unknown", "exact", "interval"]))
        if state == "interval" and isinstance(p.domain, NumericDomain):
            a = draw(numeric_in(calc, p, p.domain.lo, p.domain.hi))
            b = draw(numeric_in(calc, p, p.domain.lo, p.domain.hi))
            lo, hi = min(a, b), max(a, b)
            known[p.id] = Interval(lo=lo, hi=hi)
            completion[p.id] = draw(numeric_in(calc, p, lo, hi))
        elif state == "exact":
            v = draw(value_of(calc, p))
            known[p.id] = Exact(value=v)
            completion[p.id] = v
        else:
            completion[p.id] = draw(value_of(calc, p))
    return known, completion
