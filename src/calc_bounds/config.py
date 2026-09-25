"""Run configuration. Every run is driven by one YAML file and is fully seeded."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_bounds.distributions import Distribution


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(Strict):
    kind: Literal["anthropic", "openai_compat", "session"]
    model: str
    """Exact model id. For `session`, the model that answers offline (recorded, not called)."""
    base_url: str | None = None
    """For openai_compat (vLLM / Ollama)."""
    api_key_env: str | None = None
    """Name of the env var holding the key; the key itself never goes in config."""
    max_tokens: int = 4096
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """anthropic only (output_config.effort)."""
    temperature: float | None = None
    """openai_compat only; current Claude models reject sampling parameters."""
    seed: int | None = None
    """openai_compat only."""
    price_per_mtok_in: float = 0.0
    price_per_mtok_out: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind == "anthropic" and (self.temperature is not None or self.seed is not None):
            raise ValueError("anthropic: temperature/seed are not supported by current models")
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat needs base_url")
        return self

    def request_params(self) -> dict[str, object]:
        params: dict[str, object] = {"max_tokens": self.max_tokens}
        for name in ("effort", "temperature", "seed"):
            if (v := getattr(self, name)) is not None:
                params[name] = v
        return params


class CohortConfig(Strict):
    n_per_calculator: int = Field(gt=0)
    missingness: float = Field(ge=0.0, le=1.0)
    """Per-parameter probability that the note does not document it."""
    normal_as_negation_rate: float = Field(ge=0.0, le=1.0)
    """For documented numeric/ordinal values that are normal: probability the note says
    'normal' (documented_negative) rather than stating the value. Booleans that are false are
    always documented as negated."""
    undetermined_fraction: float = Field(ge=0.0, le=1.0)
    """Exact fraction of cases per calculator whose category is undetermined from the note."""
    trap_rates: dict[str, float] = {}
    """Per trap kind (see cohort.TrapKind): probability a case gets one trap of that kind."""
    priors: dict[str, dict[str, Distribution]] = {}
    """Overrides of cohort.priors.DEFAULT_PRIORS, by calculator then parameter."""


class RenderConfig(Strict):
    provider: str
    """Key into `providers`."""
    locales: list[Literal["en-US", "es-CL"]]
    subset_per_calculator: int | None = None
    """Render only a seeded, coverage-stratified subset of this many cases per calculator."""


class ValidationConfig(Strict):
    judge_provider: str | None = None
    """Key into `providers` for the semantic check (negations expressed, nothing inferable).
    Should be a different model from the renderer. None = deterministic checks only."""
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
    renders: dict[str, RenderConfig] = {}
    """Named render sets, e.g. {"sonnet": ..., "local": ...}; the same cases can be rendered
    by several providers."""
    validation: ValidationConfig = ValidationConfig()
    extraction: ExtractionConfig
    simulator: SimulatorConfig = SimulatorConfig()
    policies: list[
        Literal["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]
    ]
    providers: dict[str, ProviderConfig] = {}

    @model_validator(mode="after")
    def _check_refs(self) -> Self:
        for name, r in self.renders.items():
            if r.provider not in self.providers:
                raise ValueError(f"renders.{name}: unknown provider {r.provider!r}")
        judge = self.validation.judge_provider
        if judge is not None and judge not in self.providers:
            raise ValueError(f"validation.judge_provider: unknown provider {judge!r}")
        return self


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))
