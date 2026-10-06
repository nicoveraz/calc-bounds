"""MIMIC run configuration and the guards that keep credentialed data out of this repo and away
from third-party APIs.

Guards (hard errors, checked when the config is loaded and again before any model call):
  * every data, output and cache path must resolve OUTSIDE this repository (LLM cache entries
    contain note text; row-level outputs contain MIMIC identifiers);
  * only local model providers: `ollama` or `openai_compat`, with a base URL on localhost,
    127.0.0.1 or ::1. `anthropic`, `claude_cli` and `session` (answered offline by a Claude
    session) are refused. PhysioNet's data use agreement forbids sending the data to third
    parties, including through APIs (PhysioNet guidance on LLM use, 2025-09-24);
  * at run time, an HTTP(S) proxy in the environment must not apply to localhost, otherwise a
    "local" request could leave the machine.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_bounds.config import ProviderConfig

LOCAL_PROVIDER_KINDS = ("ollama", "openai_compat")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
ORACLE = "oracle"
"""Reserved extractor name: structured truth as if the note documented every recorded value
(no model; for dry runs and tests)."""


class DataUseError(RuntimeError):
    """A configuration or environment that could leak MIMIC data. Not a ValueError, so pydantic
    does not wrap it: it surfaces as itself, with its message, and nothing catches it."""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def repo_root() -> Path | None:
    """This repository's root (the nearest ancestor of this file with a pyproject.toml)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def assert_outside_repo(name: str, path: Path, root: Path | None = None) -> Path:
    """Resolve `path` (~ expanded, symlinks followed) and refuse it if it lies in the repo."""
    root = root if root is not None else repo_root()
    resolved = path.expanduser().resolve()
    if root is not None and resolved.is_relative_to(root.resolve()):
        raise DataUseError(
            f"{name} = {resolved} is inside the repository ({root}). MIMIC data, row-level "
            "outputs and the LLM cache must live outside the repo (PhysioNet DUA; no real "
            "patient data in this repository)."
        )
    return resolved


def assert_local_provider(name: str, p: ProviderConfig) -> None:
    """Refuse any provider that could send MIMIC text off this machine."""
    if p.kind not in LOCAL_PROVIDER_KINDS:
        raise DataUseError(
            f"providers.{name}: kind {p.kind!r} is not allowed for MIMIC runs. Only locally "
            f"running models ({', '.join(LOCAL_PROVIDER_KINDS)}) may process MIMIC text: the "
            "PhysioNet DUA forbids sending the data to third parties, including through APIs."
        )
    url = p.base_url or ("http://localhost:11434" if p.kind == "ollama" else "")
    host = urlsplit(url).hostname
    if host not in LOCAL_HOSTS:
        raise DataUseError(
            f"providers.{name}: base_url {url!r} is not local (host {host!r}). MIMIC runs may "
            f"only call a model server on {', '.join(LOCAL_HOSTS)}."
        )


_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy")


def assert_no_proxy_for_local(env: Mapping[str, str] | None = None) -> None:
    """Refuse to run if an HTTP(S) proxy is configured and NO_PROXY does not exempt localhost
    and 127.0.0.1 (clients honour these variables, so requests could leave the machine)."""
    env = os.environ if env is None else env
    lower = {k.lower(): v for k, v in env.items()}
    proxies = [k for k in _PROXY_VARS if lower.get(k)]
    if not proxies:
        return
    exempt = {h.strip() for h in lower.get("no_proxy", "").split(",")}
    if "*" in exempt or {"localhost", "127.0.0.1"} <= exempt:
        return
    raise DataUseError(
        f"proxy variables {proxies} are set and NO_PROXY does not exempt localhost and "
        "127.0.0.1; requests to the local model could leave the machine. Unset the proxy or "
        "add localhost,127.0.0.1 to NO_PROXY."
    )


class MimicPaths(Strict):
    """Where the credentialed data lives and where outputs go. All outside the repo.

    `hosp_dir`, `icu_dir` and `ed_dir` are relative to `data_dir` unless absolute."""

    data_dir: Path
    """MIMIC-IV root (contains hosp/ and icu/ by default)."""
    hosp_dir: Path = Path("hosp")
    icu_dir: Path = Path("icu")
    ed_dir: Path = Path("ed")
    """MIMIC-IV-ED `ed/` directory (a separate PhysioNet project)."""
    notes_path: Path
    """MIMIC-IV-Note discharge summaries (discharge.csv.gz)."""
    output_dir: Path
    """Row-level outputs (cases, notes, extractions, traces). Contains identifiers and text."""
    cache_dir: Path
    """LLM disk cache for MIMIC runs (entries contain note text)."""

    def resolved(self, root: Path | None = None) -> dict[str, Path]:
        data = assert_outside_repo("data_dir", self.data_dir, root)
        out = {"data_dir": data}
        for name in ("hosp_dir", "icu_dir", "ed_dir"):
            p = getattr(self, name).expanduser()
            out[name] = assert_outside_repo(name, p if p.is_absolute() else data / p, root)
        for name in ("notes_path", "output_dir", "cache_dir"):
            out[name] = assert_outside_repo(name, getattr(self, name), root)
        return out


class MimicExtractor(Strict):
    provider: str
    """Key into `providers` (a local model)."""
    max_workers: int = Field(default=1, ge=1)


type MimicPolicy = Literal["s1_ask_all", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"]
"""S2 (frontier LLM agent) cannot run on MIMIC: it would send note text to a remote API."""

type MimicCalculator = Literal["curb65", "qsofa", "heart", "cockcroft_gault", "perc", "wells_pe"]


class MimicRunConfig(Strict):
    run_id: str
    seed: int
    paths: MimicPaths
    criteria: Path = Path("configs/mimic_criteria.yaml")
    """Cohort definitions, ICD codes, itemids, windows (code/config, may live in the repo)."""
    calculators: list[MimicCalculator]
    calculator_options: dict[str, dict[str, list[float] | float | str]] = {}
    max_cases_per_calculator: int | None = Field(default=None, gt=0)
    """Seeded sample per calculator (None = every eligible case)."""
    extractors: dict[str, MimicExtractor] = {}
    policies: list[MimicPolicy]
    echo_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    """S4: confirm extracted decision-critical values whose (uncalibrated) confidence is below."""
    min_cell_count: int = Field(default=10, ge=1)
    """Aggregate tables suppress rates for groups smaller than this (privacy margin)."""
    providers: dict[str, ProviderConfig] = {}

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.paths.resolved()
        for name, p in self.providers.items():
            assert_local_provider(name, p)
        if ORACLE in self.extractors:
            raise ValueError(f"extractor name {ORACLE!r} is reserved")
        for name, x in self.extractors.items():
            if x.provider not in self.providers:
                raise ValueError(f"extractors.{name}: unknown provider {x.provider!r}")
        return self

    def run_dir(self) -> Path:
        return self.paths.resolved()["output_dir"] / self.run_id


def load_mimic_config(path: Path) -> MimicRunConfig:
    return MimicRunConfig.model_validate(yaml.safe_load(path.read_text()))
