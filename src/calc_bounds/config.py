"""Run configuration. Every run is driven by one YAML file and is fully seeded."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(Strict):
    kind: Literal["anthropic", "openai_compat"]
    model: str
    base_url: str | None = None
    """For openai_compat (vLLM / Ollama)."""
    api_key_env: str | None = None
    """Name of the env var holding the key; the key itself never goes in config."""
    temperature: float = 0.0
    max_tokens: int = 2048
    price_per_mtok_in: float = 0.0
    price_per_mtok_out: float = 0.0


class CohortConfig(Strict):
    n_per_calculator: int = Field(gt=0)
    missingness: float = Field(ge=0.0, le=1.0)
    negation_rate: float = Field(ge=0.0, le=1.0)
    """Among documented params, fraction documented as negative/normal where meaningful."""
    min_undetermined_fraction: float = Field(ge=0.0, le=1.0)
    trap_rates: dict[str, float] = {}


class RenderConfig(Strict):
    provider: str
    """Key into `providers`."""
    locales: list[Literal["en-US", "es-CL"]]
    review_fraction: float = 0.2


class ExtractionConfig(Strict):
    kind: Literal["oracle", "llm"]
    provider: str | None = None
    calibration: Literal["none", "temperature", "isotonic"] = "none"
    dev_fraction: float = 0.3


class SimulatorConfig(Strict):
    unavailable_rate: dict[str, float] = {}
    """Per-param probability the clinician answers 'not available'."""


class RunConfig(Strict):
    run_id: str
    seed: int
    output_dir: Path = Path("runs")
    cache_dir: Path = Path(".cache/llm")
    max_cost_usd: float = 0.0
    """Hard budget per run; 0 forbids any paid call."""
    calculators: list[str]
    calculator_options: dict[str, dict[str, list[float] | float | str]] = {}
    cohort: CohortConfig
    render: RenderConfig | None = None
    extraction: ExtractionConfig
    simulator: SimulatorConfig = SimulatorConfig()
    policies: list[
        Literal["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]
    ]
    providers: dict[str, ProviderConfig] = {}


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))
