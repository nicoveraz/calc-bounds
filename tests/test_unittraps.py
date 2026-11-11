"""The unit-trap grid (M7): cases and notes are built as the module docstring says."""

from pathlib import Path

import pytest

from calc_bounds import unittraps
from calc_bounds.config import load_config
from calc_bounds.types import DocumentedState


@pytest.fixture(scope="module")
def built():
    return unittraps.build_cases(load_config(Path("configs/tool_eval.yaml")))


def test_grid_is_complete(built) -> None:
    cases, notes = built
    assert len(cases) == 2 * 5 * 9 == len(notes) == len({c.case_id for c in cases})


def test_truth_is_the_value_as_written(built) -> None:
    by_id = {c.case_id: c for c in built[0]}
    # BUN 20 mg/dL / 2.8014 = 7.14 mmol/L; urea 43 mg/dL / 6.006 = 7.16; both over 7.
    assert by_id["units-one_point-bun-20"].truth["urea"] == pytest.approx(7.139, abs=1e-3)
    assert by_id["units-one_point-urea_mgdl-20"].truth["urea"] == pytest.approx(7.160, abs=1e-3)
    assert by_id["units-one_point-mmol-20"].truth["urea"] == 7.1
    # The cut-off falls between BUN 19 and 20 in every presentation and background.
    for c in built[0]:
        _, background, _, bun = c.case_id.split("-")
        over = int(bun) >= 20
        want = {"one_point": ("low", "moderate"), "two_points": ("moderate", "high")}[background]
        assert c.true_category == want[over], c.case_id


def test_a_number_without_a_unit_is_not_documented(built) -> None:
    cases, notes = built
    for c in cases:
        labelled = "no_unit" not in c.case_id
        assert (c.documented["urea"] == DocumentedState.POSITIVE) == labelled
        assert c.determined_from_note == labelled
    assert "BUN 19." in notes["units-one_point-bun_no_unit-19"]
    assert "BUN 19 mg/dL." in notes["units-one_point-bun-19"]
    assert "Urea 6.8 mmol/L." in notes["units-two_points-mmol-19"]
