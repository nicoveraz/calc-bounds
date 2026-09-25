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

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


@pytest.mark.parametrize("module", MODULES)
def test_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_example_configs_validate(path: Path) -> None:
    load_config(path)
