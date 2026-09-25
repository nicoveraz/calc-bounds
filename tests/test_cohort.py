import pytest

from calc_bounds.bounds import from_extractions, is_determined
from calc_bounds.calculators import REGISTRY
from calc_bounds.cohort import PatientCase, TrapKind, generate_cohort
from calc_bounds.cohort.generate import MEDICATION_FOR
from calc_bounds.config import CohortConfig
from calc_bounds.extraction.oracle import oracle_extractions
from calc_bounds.types import BoolDomain, DocumentedState, NumericDomain, OrdinalDomain
from calc_bounds.units import accepted_units

CFG = CohortConfig(
    n_per_calculator=40,
    missingness=0.3,
    normal_as_negation_rate=0.5,
    undetermined_fraction=0.5,
    trap_rates={k.value: 0.5 for k in TrapKind},
)


@pytest.fixture(scope="module")
def cohort() -> list[PatientCase]:
    return generate_cohort(list(REGISTRY.values()), CFG, seed=7)


def test_deterministic(cohort: list[PatientCase]) -> None:
    again = generate_cohort(list(REGISTRY.values()), CFG, seed=7)
    assert [c.model_dump() for c in again] == [c.model_dump() for c in cohort]
    assert generate_cohort(list(REGISTRY.values()), CFG, seed=8) != cohort


def test_calculator_seeds_independent_of_calculator_list(cohort: list[PatientCase]) -> None:
    only_heart = generate_cohort([REGISTRY["heart"]], CFG, seed=7)
    assert only_heart == [c for c in cohort if c.calculator == "heart"]


def test_coverage_quota(cohort: list[PatientCase]) -> None:
    for calc_id in REGISTRY:
        cases = [c for c in cohort if c.calculator == calc_id]
        assert len(cases) == 40
        assert sum(not c.determined_from_note for c in cases) == 20


def test_ground_truth_consistent(cohort: list[PatientCase]) -> None:
    for case in cohort:
        calc = REGISTRY[case.calculator]
        assert (case.true_score, case.true_category) == calc.evaluate(case.truth)
        known = from_extractions(
            calc, oracle_extractions(calc.parameters, case.truth, case.documented)
        )
        assert is_determined(calc, known) == case.determined_from_note
        if "sbp" in case.truth and "dbp" in case.truth:
            assert case.truth["dbp"] <= case.truth["sbp"] - 15


def test_documented_states_are_truthful(cohort: list[PatientCase]) -> None:
    for case in cohort:
        calc = REGISTRY[case.calculator]
        for p in calc.parameters:
            v, s = case.truth[p.id], case.documented[p.id]
            if s == DocumentedState.NEGATIVE:
                assert p.negatable
                match p.domain:
                    case BoolDomain():
                        assert v is False
                    case OrdinalDomain():
                        assert v == 0
                    case NumericDomain():
                        assert p.absent_means is not None
                        assert p.absent_means[0] <= v <= p.absent_means[1]
            if s == DocumentedState.POSITIVE and isinstance(p.domain, BoolDomain):
                assert v is True


def test_traps_well_formed(cohort: list[PatientCase]) -> None:
    kinds = set()
    for case in cohort:
        params = [t.param for t in case.traps]
        assert len(params) == len(set(params)), "at most one trap per param"
        for t in case.traps:
            kinds.add(t.kind)
            s = case.documented[t.param]
            match t.kind:
                case TrapKind.NEGATION:
                    assert s == DocumentedState.NEGATIVE
                case TrapKind.MIXED_UNITS:
                    assert s == DocumentedState.POSITIVE
                    assert t.detail["unit"] in accepted_units(t.param)[1:]
                case TrapKind.MULTIPLE_ENCOUNTERS:
                    assert t.detail["prior_encounter_value"] != case.truth[t.param]
                case TrapKind.CONTRADICTORY_VALUES:
                    assert t.detail["distractor_value"] != case.truth[t.param]
                case TrapKind.COMORBIDITY_VIA_MEDICATION:
                    assert case.truth[t.param] is True
                    assert t.detail["medication"] == MEDICATION_FOR[t.param]
    assert kinds == set(TrapKind)


def test_impossible_quota_raises() -> None:
    cfg = CFG.model_copy(
        update={"missingness": 0.0, "undetermined_fraction": 1.0, "n_per_calculator": 2}
    )
    with pytest.raises(RuntimeError, match="undetermined_fraction"):
        generate_cohort([REGISTRY["qsofa"]], cfg, seed=1)


def test_unknown_trap_kind_rejected() -> None:
    cfg = CFG.model_copy(update={"trap_rates": {"typo": 0.1}})
    with pytest.raises(ValueError, match="unknown trap"):
        generate_cohort([REGISTRY["qsofa"]], cfg, seed=1)
