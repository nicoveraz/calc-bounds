"""End-to-end MIMIC pipeline on SYNTHETIC fixtures, copied outside the repo (the config
refuses data paths inside it). No real data, no network, no real model."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

from calc_bounds.llm import LLMRequest, LLMResponse, Usage
from calc_bounds.mimic import runner
from calc_bounds.mimic.config import DataUseError, MimicRunConfig, load_mimic_config

FIX = Path(__file__).parent / "fixtures" / "mimic_synthetic"
IDS = [str(i) for i in range(99000001, 99000007)] + [str(i) for i in range(97000001, 97000007)]


@pytest.fixture
def cfg(tmp_path: Path) -> MimicRunConfig:
    phys = tmp_path / "physionet"
    shutil.copytree(FIX / "hosp", phys / "mimic-iv" / "hosp")
    shutil.copytree(FIX / "icu", phys / "mimic-iv" / "icu")
    shutil.copytree(FIX / "ed", phys / "mimic-iv-ed" / "ed")
    shutil.copytree(FIX / "note", phys / "mimic-iv-note" / "note")
    raw = yaml.safe_load(Path("configs/mimic_example.yaml").read_text())
    raw["paths"] = {
        "data_dir": str(phys / "mimic-iv"),
        "ed_dir": str(phys / "mimic-iv-ed" / "ed"),
        "notes_path": str(phys / "mimic-iv-note" / "note" / "discharge.csv"),
        "output_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
    }
    raw["min_cell_count"] = 1
    raw["providers"]["qwen9b"]["model"] = "fake-local-model"
    path = tmp_path / "mimic.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_mimic_config(path)


def test_cohort_flow_and_cases(cfg: MimicRunConfig) -> None:
    out = runner.build_cohort(cfg)
    assert out.is_relative_to(cfg.paths.resolved()["output_dir"])
    flow = pd.read_csv(out / "cohort_flow.csv").set_index("calculator")
    assert flow.loc["curb65", ["eligible", "admitted", "with_note", "selected"]].tolist() == [
        2,
        2,
        1,  # 98000004 has no discharge note
        1,
    ]
    assert flow.loc["heart", ["eligible", "admitted", "with_note"]].tolist() == [3, 2, 2]
    assert flow.loc["heart", "pending_annotation"] == 2 and flow.loc["heart", "fallback"] == 1
    assert flow.loc["qsofa", "selected"] == 1 and flow.loc["cockcroft_gault", "selected"] == 1
    cases = {c.case_id: c for c in runner.load_cases(cfg)}
    assert set(cases) == {
        "curb65-97000001",
        "qsofa-97000001",
        "heart-97000002",
        "heart-97000006",
        "cockcroft_gault-97000001",
    }
    assert cases["cockcroft_gault-97000001"].truth["sex"] == 0
    assert cases["heart-97000006"].implausible == []  # heart rate is not a HEART input


def test_oracle_run_annotate_and_aggregate(cfg: MimicRunConfig, tmp_path: Path) -> None:
    runner.build_cohort(cfg)
    rows = runner.run(cfg, "oracle")
    pol = pd.read_csv(rows / "policies.csv").set_index(["case_id", "policy"])
    assert set(pol.index.get_level_values("policy")) == {
        "s1_ask_all",
        "s3_bounds",
        "s4_bounds_voi_echo",
        "s3_bin",
    }
    curb = pol.loc[("curb65-97000001", "s3_bounds")]
    assert curb["final_category"] == "high" and curb["n_questions"] == 0
    # HEART judgement items are not in the record: the EHR says "not available".
    heart = pol.loc[("heart-97000002", "s1_ask_all")]
    assert heart["n_unavailable"] >= 2 and pd.isna(heart["reference_category"])
    tri_state = pol.drop(index="s3_bin", level="policy")
    assert not tri_state["premature_commitment"].any()
    assert pol.xs("s3_bin", level="policy")["premature_commitment"].any()  # the ablation does
    claims = pd.read_csv(rows / "claims.csv")
    assert claims["agrees"].dropna().astype(bool).all()  # oracle: never contradicts the record

    # Physician annotation of the judgement items, then a re-run.
    template = pd.read_csv(runner.export_annotations(cfg, None), dtype=str, keep_default_na=False)
    assert len(template) == 4 and set(template["param"]) == {"heart_history", "heart_ecg"}
    filled = template.copy()
    filled.loc[filled["param"] == "heart_history", "value"] = "highly_suspicious"
    filled.loc[filled["param"] == "heart_ecg", "value"] = "normal"
    filled_path = tmp_path / "filled.csv"
    filled.to_csv(filled_path, index=False)
    runner.import_annotations(cfg, filled_path)
    rows = runner.run(cfg, "oracle")
    cases = pd.read_csv(rows / "cases.csv").set_index("case_id")
    # history 2 + ECG 0 + age 50 (1) + risk 1-2 + troponin 2x (1) = 5-6: moderate either way.
    assert cases.loc["heart-97000002", "reference_category"] == "moderate"
    assert cases.loc["heart-97000002", "truth_within_note_bounds"]

    dest = runner.aggregate(cfg, "oracle", tmp_path / "results")
    names = {p.name for p in dest.iterdir()}
    assert names == {
        "cohort_flow.csv",
        "note_alone.csv",
        "policies.csv",
        "extraction.csv",
        "questions.csv",
    }
    for p in dest.iterdir():
        text = p.read_text()
        assert not any(i in text for i in IDS), f"identifier in aggregate {p.name}"
        assert "SYNTHETIC" not in text


def _fake_model_reply(request: LLMRequest) -> str:
    """A stand-in for a local model: states RR when the note says 'RR 32', else unknown."""
    schema = request.params["format"]
    note = request.messages[0]["content"]
    out = {}
    for pid in schema["required"]:
        item = {"status": "unknown", "value": None, "unit": None, "evidence": None}
        if pid == "resp_rate" and "RR 32" in note:
            item = {"status": "present", "value": 32, "unit": "/min", "evidence": "RR 32"}
        out[pid] = item | {"confidence": 0.9}
    return json.dumps(out)


class FakeLocalClient:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=_fake_model_reply(request), usage=Usage(input_tokens=1))


def test_local_model_run_uses_external_cache(
    cfg: MimicRunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLocalClient()
    monkeypatch.setattr("calc_bounds.mimic.extract.make_client", lambda p: fake)
    runner.build_cohort(cfg)
    rows = runner.run(cfg, "qwen9b")
    assert len(fake.requests) == 5 and all(r.provider == "ollama" for r in fake.requests)
    sent = " ".join(r.messages[0]["content"] for r in fake.requests)
    assert "Fictional course text" not in sent  # Brief Hospital Course is not sent
    assert "HR 80 BP 120/70" not in sent  # nor the discharge exam
    cache = cfg.paths.resolved()["cache_dir"]
    assert len(list(cache.rglob("*.json"))) == 5
    claims = pd.read_csv(rows / "claims.csv").set_index(["case_id", "param"])
    rr = claims.loc[("curb65-97000001", "resp_rate")]
    assert rr["extracted_state"] == "present" and rr["agrees"]
    runner.run(cfg, "qwen9b")  # second run: served from the cache
    assert len(fake.requests) == 5


def test_run_refuses_proxy_and_unknown_extractor(
    cfg: MimicRunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.build_cohort(cfg)
    with pytest.raises(ValueError, match="unknown extractor"):
        runner.run(cfg, "gpt")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with pytest.raises(DataUseError):
        runner.run(cfg, "qwen9b")


def test_fixtures_inside_repo_are_refused_as_data_dir(cfg: MimicRunConfig) -> None:
    raw = cfg.model_dump(mode="json")
    raw["paths"]["data_dir"] = str(FIX)
    with pytest.raises(DataUseError):
        MimicRunConfig.model_validate(raw)


def test_diagnose_is_aggregate_only(cfg: MimicRunConfig) -> None:
    from calc_bounds.mimic.diagnose import gcs_report, troponin_report

    runner.build_cohort(cfg)
    trop = troponin_report(cfg)
    assert list(trop["itemid"]) == [51003]
    assert trop["heart_stays"].iloc[0] == 2
    gcs = gcs_report(cfg)
    assert {"curb65", "qsofa"} <= set(gcs["calculator"])
    text = trop.to_csv() + gcs.to_csv()
    assert not any(i in text for i in IDS)


def test_value_shape_hides_digits() -> None:
    from calc_bounds.mimic.diagnose import _shape

    assert _shape(" <0.01 ") == "<9.99"
    assert _shape("less than 0.01") == "LESS THAN 9.99"
