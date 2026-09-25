"""Cohort generation: hidden truth -> documented state -> coverage quota -> trap tags."""

import zlib

import numpy as np

from calc_bounds.bounds import from_extractions, is_determined
from calc_bounds.calculators import Calculator
from calc_bounds.cohort import PatientCase, Trap, TrapKind
from calc_bounds.cohort.priors import DEFAULT_PRIORS
from calc_bounds.config import CohortConfig
from calc_bounds.distributions import Distribution
from calc_bounds.types import (
    BoolDomain,
    DocumentedState,
    NumericDomain,
    OrdinalDomain,
    ParameterSpec,
    ParamId,
    Value,
)
from calc_bounds.units import DISPLAY_DECIMALS, accepted_units, from_canonical, to_canonical

MAX_ATTEMPTS_PER_CASE = 500

# Comorbidities a note may imply only through a medication. Illustrative, not exhaustive.
MEDICATION_FOR: dict[ParamId, str] = {
    "hypertension": "lisinopril",
    "diabetes": "metformin",
    "hypercholesterolemia": "atorvastatin",
}


def _priors(calc: Calculator, config: CohortConfig) -> dict[ParamId, Distribution]:
    priors = {**DEFAULT_PRIORS.get(calc.id, {}), **config.priors.get(calc.id, {})}
    missing = [p.id for p in calc.parameters if p.id not in priors]
    if missing:
        raise ValueError(f"{calc.id}: no prior for {missing}")
    return priors


def sample_truth(
    calc: Calculator, priors: dict[ParamId, Distribution], rng: np.random.Generator
) -> dict[ParamId, Value]:
    truth = {p.id: priors[p.id].sample(rng) for p in calc.parameters}
    if "sbp" in truth and "dbp" in truth:
        for _ in range(1000):  # keep pulse pressure >= 15 mmHg
            if truth["dbp"] <= truth["sbp"] - 15:
                break
            truth["dbp"] = priors["dbp"].sample(rng)
        else:
            truth["dbp"] = truth["sbp"] - 15
    for p in calc.parameters:
        if isinstance(p.domain, NumericDomain):
            assert p.domain.lo <= truth[p.id] <= p.domain.hi, (p.id, truth[p.id])
    return truth


def _is_normal(p: ParameterSpec, v: Value) -> bool:
    match p.domain:
        case BoolDomain():
            return not v
        case OrdinalDomain():
            return v == 0
        case NumericDomain():
            return p.absent_means is not None and p.absent_means[0] <= v <= p.absent_means[1]
    raise TypeError(p.domain)


def assign_documented(
    calc: Calculator,
    truth: dict[ParamId, Value],
    config: CohortConfig,
    rng: np.random.Generator,
) -> dict[ParamId, DocumentedState]:
    doc: dict[ParamId, DocumentedState] = {}
    for p in calc.parameters:
        v = truth[p.id]
        missing = rng.random() < config.missingness
        as_negation = rng.random() < config.normal_as_negation_rate
        if missing:
            doc[p.id] = DocumentedState.NOT_DOCUMENTED
        elif isinstance(p.domain, BoolDomain):
            if v:
                doc[p.id] = DocumentedState.POSITIVE
            elif p.negatable:
                doc[p.id] = DocumentedState.NEGATIVE
            else:
                doc[p.id] = DocumentedState.NOT_DOCUMENTED
        elif p.negatable and _is_normal(p, v) and as_negation:
            doc[p.id] = DocumentedState.NEGATIVE
        else:
            doc[p.id] = DocumentedState.POSITIVE
    return doc


def _different_value(dist: Distribution, v: Value, rng: np.random.Generator) -> Value:
    for _ in range(1000):
        x = dist.sample(rng)
        if x != v:
            return x
    raise RuntimeError(f"could not sample a value different from {v}")


def assign_traps(
    calc: Calculator,
    truth: dict[ParamId, Value],
    doc: dict[ParamId, DocumentedState],
    priors: dict[ParamId, Distribution],
    config: CohortConfig,
    rng: np.random.Generator,
) -> list[Trap]:
    """At most one trap per param. Traps are rendering instructions. The one exception that
    touches truth: a mixed-units trap snaps the true value to exactly the value the note will
    display in the other unit, so display rounding can never move a value across a threshold.
    Mutates `truth` in place for that case."""
    traps: list[Trap] = []
    used: set[ParamId] = set()
    positive_numeric = [
        p.id
        for p in calc.parameters
        if isinstance(p.domain, NumericDomain) and doc[p.id] == DocumentedState.POSITIVE
    ]
    for kind in TrapKind:
        rate = config.trap_rates.get(kind.value, 0.0)
        if rng.random() >= rate:
            continue
        match kind:
            case TrapKind.NEGATION:
                cands = [pid for pid, s in doc.items() if s == DocumentedState.NEGATIVE]
            case TrapKind.MIXED_UNITS:
                cands = [pid for pid in positive_numeric if len(accepted_units(pid)) > 1]
            case TrapKind.MULTIPLE_ENCOUNTERS | TrapKind.CONTRADICTORY_VALUES:
                cands = positive_numeric
            case TrapKind.COMORBIDITY_VIA_MEDICATION:
                cands = [
                    pid
                    for pid in MEDICATION_FOR
                    if pid in doc and doc[pid] == DocumentedState.POSITIVE
                ]
        cands = [c for c in cands if c not in used]
        if not cands:
            continue
        pid = str(rng.choice(cands))
        used.add(pid)
        detail: dict[str, str | float | bool] = {}
        match kind:
            case TrapKind.MIXED_UNITS:
                unit = str(rng.choice(accepted_units(pid)[1:]))
                shown = round(from_canonical(pid, float(truth[pid]), unit), DISPLAY_DECIMALS[unit])
                truth[pid] = to_canonical(pid, shown, unit)
                detail = {"unit": unit, "value_in_unit": shown}
            case TrapKind.MULTIPLE_ENCOUNTERS:
                detail = {
                    "prior_encounter_value": _different_value(priors[pid], truth[pid], rng),
                    "note": "value from an earlier encounter; the current value is the truth",
                }
            case TrapKind.CONTRADICTORY_VALUES:
                detail = {
                    "distractor_value": _different_value(priors[pid], truth[pid], rng),
                    "note": "an earlier same-visit reading superseded by the repeat (the truth)",
                }
            case TrapKind.COMORBIDITY_VIA_MEDICATION:
                detail = {"medication": MEDICATION_FOR[pid]}
        traps.append(Trap(kind=kind, param=pid, detail=detail))
    return traps


def generate_for_calculator(calc: Calculator, config: CohortConfig, seed: int) -> list[PatientCase]:
    from calc_bounds.extraction.oracle import oracle_extractions

    priors = _priors(calc, config)
    unknown_kinds = set(config.trap_rates) - {k.value for k in TrapKind}
    if unknown_kinds:
        raise ValueError(f"unknown trap kinds {sorted(unknown_kinds)}")
    n = config.n_per_calculator
    want_undetermined = round(config.undetermined_fraction * n)
    quota = {True: n - want_undetermined, False: want_undetermined}  # keyed by determined
    rng = np.random.default_rng(seed)
    cases: list[PatientCase] = []
    for _ in range(MAX_ATTEMPTS_PER_CASE * n):
        if len(cases) == n:
            break
        case_seed = int(rng.integers(2**31))
        crng = np.random.default_rng(case_seed)
        truth = sample_truth(calc, priors, crng)
        doc = assign_documented(calc, truth, config, crng)
        traps = assign_traps(calc, truth, doc, priors, config, crng)
        known = from_extractions(calc, oracle_extractions(calc.parameters, truth, doc))
        determined = is_determined(calc, known)
        if quota[determined] == 0:
            continue
        quota[determined] -= 1
        score, category = calc.evaluate(truth)
        cases.append(
            PatientCase(
                case_id=f"{calc.id}-{len(cases):04d}",
                seed=case_seed,
                calculator=calc.id,
                truth=truth,
                documented=doc,
                traps=traps,
                true_score=score,
                true_category=category,
                determined_from_note=determined,
            )
        )
    if len(cases) < n:
        raise RuntimeError(
            f"{calc.id}: filled only {len(cases)}/{n} cases (remaining quota {quota}); "
            "adjust missingness or undetermined_fraction"
        )
    return cases


def stable_seed(*parts: int | str) -> int:
    """Deterministic 32-bit seed from ints/strings (Python's hash() is salted per process)."""
    words = [p if isinstance(p, int) else zlib.crc32(p.encode()) for p in parts]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def generate_cohort(calcs: list[Calculator], config: CohortConfig, seed: int) -> list[PatientCase]:
    """Cases for each calculator. Each calculator's seed depends only on (seed, calculator id),
    so adding or removing calculators does not change the others' cases."""
    cases: list[PatientCase] = []
    for calc in calcs:
        cases += generate_for_calculator(calc, config, stable_seed(seed, calc.id))
    return cases
