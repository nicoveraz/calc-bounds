import pytest
from hypothesis import given
from hypothesis import strategies as st

from calc_bounds.units import UnitError, accepted_units, from_canonical, to_canonical


@pytest.mark.parametrize(
    ("param", "value", "unit", "expected"),
    [
        # BUN 19.6 mg/dL = 19.6 / 2.8014 = 6.9965 mmol/L urea.
        ("urea", 19.6, "bun_mg/dL", 6.9965),
        ("urea", 20.0, "bun_mg/dL", 7.139),
        # Urea 42 mg/dL (MW 60.06) ~= 7.0 mmol/L.
        ("urea", 42.0, "urea_mg/dL", 6.993),
        ("urea", 7.0, "mmol/L", 7.0),
        ("urea", 7.0, "mmol/l", 7.0),
        ("creatinine", 88.4, "umol/L", 1.0),
        ("creatinine", 177.0, "µmol/L", 2.002),
        ("creatinine", 1.2, "mg/dl", 1.2),
        ("weight", 154.0, "lb", 69.853),
        ("weight", 70.0, "kg", 70.0),
        ("heart_rate", 110.0, "bpm", 110.0),
    ],
)
def test_known_conversions(param: str, value: float, unit: str, expected: float) -> None:
    assert to_canonical(param, value, unit) == pytest.approx(expected, abs=1e-3)


def test_bun_to_urea_mass_ratio() -> None:
    # 1 mg/dL BUN corresponds to 60.06 / 28.014 = 2.144 mg/dL urea.
    assert to_canonical("urea", 1.0, "bun_mg/dL") / to_canonical(
        "urea", 1.0, "urea_mg/dL"
    ) == pytest.approx(2.144, abs=1e-3)


@pytest.mark.parametrize(
    ("param", "unit"),
    [("urea", "mg/dL"), ("creatinine", "mmol/L"), ("weight", "stone"), ("nope", "kg")],
)
def test_rejects_ambiguous_or_unknown_units(param: str, unit: str) -> None:
    with pytest.raises(UnitError):
        to_canonical(param, 1.0, unit)


@given(st.floats(0.01, 1e4), st.sampled_from(["urea", "creatinine", "weight"]), st.data())
def test_round_trip(value: float, param: str, data: st.DataObject) -> None:
    unit = data.draw(st.sampled_from(accepted_units(param)))
    back = to_canonical(param, from_canonical(param, value, unit), unit)
    assert back == pytest.approx(value, rel=1e-12)
