"""MIMIC config guards: paths outside the repo, local providers only, no proxy leaks."""

from pathlib import Path

import pytest
import yaml

from calc_bounds.config import ProviderConfig
from calc_bounds.mimic.config import (
    DataUseError,
    MimicRunConfig,
    assert_local_provider,
    assert_no_proxy_for_local,
    assert_outside_repo,
    load_mimic_config,
    repo_root,
)
from calc_bounds.mimic.criteria import load_criteria

EXAMPLE = Path("configs/mimic_example.yaml")


def _raw(tmp_path: Path) -> dict:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["paths"] = {
        "data_dir": str(tmp_path / "mimic"),
        "ed_dir": str(tmp_path / "ed"),
        "notes_path": str(tmp_path / "note" / "discharge.csv.gz"),
        "output_dir": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
    }
    return raw


def test_example_config_and_criteria_load() -> None:
    cfg = load_mimic_config(EXAMPLE)
    assert set(cfg.policies) == {"s1_ask_all", "s3_bounds", "s4_bounds_voi_echo", "s3_bin"}
    crit = load_criteria(cfg.criteria)
    assert "heart_history" in crit.annotation_params


def test_repo_root_is_this_repo() -> None:
    root = repo_root()
    assert root is not None and (root / "configs" / "mimic_example.yaml").exists()


@pytest.mark.parametrize(
    "field", ["data_dir", "ed_dir", "notes_path", "output_dir", "cache_dir", "hosp_dir"]
)
def test_paths_inside_repo_are_refused(tmp_path: Path, field: str) -> None:
    raw = _raw(tmp_path)
    raw["paths"][field] = str(repo_root() / "data" / "private" / "x")  # type: ignore[operator]
    with pytest.raises(DataUseError, match="inside the repository"):
        MimicRunConfig.model_validate(raw)


def test_relative_paths_resolve_against_cwd_and_are_refused(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    raw["paths"]["cache_dir"] = ".cache/mimic"  # tests run from the repo root
    with pytest.raises(DataUseError, match="cache_dir"):
        MimicRunConfig.model_validate(raw)


def test_subdirs_relative_to_data_dir(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    raw["paths"]["hosp_dir"] = "../../" + str(repo_root()).lstrip("/")  # escapes into the repo
    raw["paths"]["data_dir"] = "/a/b"
    with pytest.raises(DataUseError):
        MimicRunConfig.model_validate(raw)
    ok = MimicRunConfig.model_validate(_raw(tmp_path))
    assert ok.paths.resolved()["hosp_dir"] == (tmp_path / "mimic" / "hosp").resolve()


def test_symlink_into_repo_is_refused(tmp_path: Path) -> None:
    link = tmp_path / "looks_outside"
    link.symlink_to(repo_root())  # type: ignore[arg-type]
    with pytest.raises(DataUseError):
        assert_outside_repo("output_dir", link / "runs")


@pytest.mark.parametrize("kind", ["anthropic", "claude_cli", "session"])
def test_remote_provider_kinds_are_refused(tmp_path: Path, kind: str) -> None:
    raw = _raw(tmp_path)
    raw["providers"]["qwen9b"] = {"kind": kind, "model": "any-model"}
    with pytest.raises(DataUseError, match="not allowed for MIMIC"):
        MimicRunConfig.model_validate(raw)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://192.168.1.20:11434",
        "http://my-gpu-box:8000/v1",
        "http://localhost.evil.com:11434",
        "http://127.0.0.2:8000/v1",
    ],
)
def test_remote_base_urls_are_refused(url: str) -> None:
    for kind in ("ollama", "openai_compat"):
        p = ProviderConfig(kind=kind, model="m", base_url=url)  # type: ignore[arg-type]
        with pytest.raises(DataUseError, match="not local"):
            assert_local_provider("x", p)


@pytest.mark.parametrize(
    "url", ["http://localhost:11434", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"]
)
def test_local_base_urls_are_accepted(url: str) -> None:
    assert_local_provider("x", ProviderConfig(kind="openai_compat", model="m", base_url=url))
    assert_local_provider("x", ProviderConfig(kind="ollama", model="m"))  # default localhost


def test_s2_agent_is_not_a_mimic_policy(tmp_path: Path) -> None:
    raw = _raw(tmp_path)
    raw["policies"] = ["s2_llm_agent"]
    with pytest.raises(ValueError):
        MimicRunConfig.model_validate(raw)


def test_proxy_environment_guard() -> None:
    assert_no_proxy_for_local({})
    assert_no_proxy_for_local(
        {"HTTPS_PROXY": "http://proxy:3128", "NO_PROXY": "localhost,127.0.0.1"}
    )
    assert_no_proxy_for_local({"https_proxy": "http://proxy:3128", "no_proxy": "*"})
    with pytest.raises(DataUseError, match="NO_PROXY"):
        assert_no_proxy_for_local({"HTTP_PROXY": "http://proxy:3128"})
    with pytest.raises(DataUseError):
        assert_no_proxy_for_local({"ALL_PROXY": "socks5://p:1080", "NO_PROXY": "localhost"})
