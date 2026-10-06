"""Run configuration. Every run is driven by one YAML file and is fully seeded."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_bounds.distributions import Distribution
from calc_bounds.simulator import ClinicianNoise


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(Strict):
    kind: Literal["anthropic", "openai_compat", "session", "claude_cli", "ollama"]
    model: str
    """Exact model id. For `session`, the model that answers offline (recorded, not called)."""
    base_url: str | None = None
    """For openai_compat (vLLM / Ollama)."""
    api_key_env: str | None = None
    """Name of the env var holding the key; the key itself never goes in config."""
    max_tokens: int = 4096
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """anthropic / claude_cli (output_config.effort / --effort)."""
    temperature: float | None = None
    """openai_compat / ollama only; current Claude models reject sampling parameters."""
    seed: int | None = None
    """openai_compat / ollama only."""
    think: bool | None = None
    """ollama only: enable/disable the model's thinking mode."""
    num_ctx: int | None = Field(default=None, gt=0)
    """ollama only: context window in tokens (Ollama's default can silently truncate long
    notes). Unset = server default; unset values do not enter the request or its cache key."""
    price_per_mtok_in: float = 0.0
    price_per_mtok_out: float = 0.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        claude = self.kind in ("anthropic", "claude_cli")
        if claude and (self.temperature is not None or self.seed is not None):
            raise ValueError(f"{self.kind}: temperature/seed are not supported by current models")
        if self.kind == "openai_compat" and not self.base_url:
            raise ValueError("openai_compat needs base_url")
        return self

    def request_params(self) -> dict[str, object]:
        params: dict[str, object] = {"max_tokens": self.max_tokens}
        for name in ("effort", "temperature", "seed", "think", "num_ctx"):
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
    style: str = "standard"
    """Style guide variant (render/styles/<locale>.<style>.md), e.g. "messy"."""
    judge_provider: str | None = None
    """Override `validation.judge_provider` for this render set."""
    max_attempts: int = Field(default=3, ge=1)
    max_workers: int = Field(default=1, ge=1)
    """Concurrent render/judge requests (scripted providers such as claude_cli)."""
    """Re-render a note that fails validation (rule or judge errors) up to this many attempts
    in total. The kept attempt is recorded on the note."""


class ValidationConfig(Strict):
    judge_provider: str | None = None
    """Key into `providers` for the semantic check (negations expressed, nothing inferable).
    Should be a different model from the renderer. None = deterministic checks only."""
    review_fraction: float = 0.2


class ExtractorConfig(Strict):
    provider: str
    """Key into `providers`."""
    max_workers: int = Field(default=1, ge=1)
    """Concurrent requests (1 for a local model; a few for claude_cli)."""


class ExtractionConfig(Strict):
    """Which extraction the policies use."""

    kind: Literal["oracle", "llm"]
    extractor: str | None = None
    """Key into `extractors` (kind llm)."""
    render: str | None = None
    """Render set whose notes were extracted (kind llm)."""
    calibration: Literal["none", "temperature", "isotonic"] = "none"
    dev_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)
    """Share of cases (per calculator) used only for fitting calibration."""


class AgentConfig(Strict):
    """S2 end-to-end LLM agent."""

    provider: str
    render: str
    """Render set whose notes the agent reads."""
    max_turns: int = 12
    max_workers: int = Field(default=3, ge=1)


class SimulatorConfig(Strict):
    name: str = "ideal"
    """Condition label; traces for non-ideal clinicians get their own files."""
    unavailable_rate: dict[str, float] = {}
    """Per-param probability the clinician answers 'not available'."""
    noise: ClinicianNoise | None = None


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
    extractors: dict[str, ExtractorConfig] = {}
    extraction: ExtractionConfig
    simulator: SimulatorConfig = SimulatorConfig()
    clinicians: dict[str, SimulatorConfig] = {}
    """Named alternative clinician conditions, selected with `--clinician <name>`."""
    agent: AgentConfig | None = None
    echo_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    """S4: confirm extracted decision-critical values whose calibrated confidence is below."""
    policies: list[
        Literal["s1_ask_all", "s2_llm_agent", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]
    ]
    providers: dict[str, ProviderConfig] = {}

    @model_validator(mode="after")
    def _check_refs(self) -> Self:
        for name, r in self.renders.items():
            if r.provider not in self.providers:
                raise ValueError(f"renders.{name}: unknown provider {r.provider!r}")
            if r.judge_provider is not None and r.judge_provider not in self.providers:
                raise ValueError(f"renders.{name}: unknown judge provider {r.judge_provider!r}")
        for name, x in self.extractors.items():
            if x.provider not in self.providers:
                raise ValueError(f"extractors.{name}: unknown provider {x.provider!r}")
        if self.extraction.kind == "llm":
            if self.extraction.extractor not in self.extractors:
                raise ValueError("extraction.extractor must name an entry in `extractors`")
            if self.extraction.render not in self.renders:
                raise ValueError("extraction.render must name an entry in `renders`")
        if self.agent is not None:
            if self.agent.provider not in self.providers:
                raise ValueError(f"agent: unknown provider {self.agent.provider!r}")
            if self.agent.render not in self.renders:
                raise ValueError(f"agent: unknown render {self.agent.render!r}")
        judge = self.validation.judge_provider
        if judge is not None and judge not in self.providers:
            raise ValueError(f"validation.judge_provider: unknown provider {judge!r}")
        return self


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))
