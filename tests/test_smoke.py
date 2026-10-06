"""M0 smoke tests: the package imports and example configs validate."""

import importlib
from pathlib import Path

import pytest

from calc_bounds.config import load_config

MODULES = [
    "calc_bounds.types",
    "calc_bounds.units",
    "calc_bounds.config",
    "calc_bounds.calculators",
    "calc_bounds.bounds",
    "calc_bounds.cohort",
    "calc_bounds.render",
    "calc_bounds.extraction",
    "calc_bounds.extraction.calibration",
    "calc_bounds.llm",
    "calc_bounds.simulator",
    "calc_bounds.policies",
    "calc_bounds.eval",
    "calc_bounds.anchor",
    "calc_bounds.cli",
]

ALL_CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))
# MIMIC configs have their own schema (calc_bounds.mimic); validated in test_mimic_config.py.
CONFIGS = [p for p in ALL_CONFIGS if not p.name.startswith("mimic_")]


@pytest.mark.parametrize("module", MODULES)
def test_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_validate(path: Path) -> None:
    load_config(path)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_calculators_resolve(path: Path) -> None:
    from calc_bounds.calculators import get_calculator

    cfg = load_config(path)
    for calc_id in cfg.calculators:
        get_calculator(calc_id, cfg.calculator_options.get(calc_id))
