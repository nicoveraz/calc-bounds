from pathlib import Path

import pandas as pd
import yaml

from calc_bounds import pipeline
from calc_bounds.config import load_config


def test_end_to_end_oracle(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/m2_oracle.yaml").read_text())
    raw["output_dir"] = str(tmp_path)
    raw["cohort"]["n_per_calculator"] = 12
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)

    pipeline.make_cohort(cfg)
    pipeline.run_policies(cfg)
    out = pipeline.evaluate(cfg)

    summary = pd.read_csv(out / "summary.csv").set_index("policy")
    assert set(summary.index) == {"s1_ask_all", "s3_bounds"}
    assert summary.loc["s3_bounds", "mean_questions"] <= summary.loc["s1_ask_all", "mean_questions"]
    assert summary.loc["s3_bounds", "irrelevant_question_rate"] in (0, 0.0) or pd.isna(
        summary.loc["s3_bounds", "irrelevant_question_rate"]
    )
    assert (out / "accuracy_vs_questions.png").stat().st_size > 0
    ext = pd.read_csv(out / "extraction_by_documented_state.csv")
    assert (ext["correct"] == 1.0).all()  # oracle


def test_subsets_are_nested(tmp_path: Path) -> None:
    from calc_bounds.cohort import generate_cohort

    cfg = load_config(Path("configs/main.yaml"))
    cfg = cfg.model_copy(update={"cohort": cfg.cohort.model_copy(update={"n_per_calculator": 30})})
    cases = generate_cohort(list(pipeline.calculators(cfg).values()), cfg.cohort, cfg.seed)
    small = {c.case_id for c in pipeline.select_cases(cfg, cases, 4)}
    big = {c.case_id for c in pipeline.select_cases(cfg, cases, 10)}
    assert len(small) == 24 and len(big) == 60 and small < big


def test_session_render_retries_failed_notes(tmp_path: Path) -> None:
    """render -> pending -> import a failing note -> render retries (attempt 2) -> import a
    passing note -> the kept note is attempt 2."""
    import json

    from calc_bounds.io import read_jsonl
    from calc_bounds.llm.session import PendingItem
    from calc_bounds.render import RenderedNote
    from calc_bounds.render.facts import build_facts

    raw = yaml.safe_load(Path("configs/main.yaml").read_text())
    raw.update(output_dir=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"))
    raw["calculators"] = ["qsofa"]
    raw["cohort"]["n_per_calculator"] = 4
    raw["renders"] = {"t": {"provider": "sonnet_session", "locales": ["en-US"], "max_attempts": 2}}
    raw["validation"]["judge_provider"] = None
    raw["simulator"] = {"unavailable_rate": {}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    pipeline.make_cohort(cfg)
    cases = {
        c.case_id: c
        for c in pipeline.read_jsonl(pipeline.run_dir(cfg) / "cohort.jsonl", pipeline.PatientCase)
    }
    calc = pipeline.calculators(cfg)["qsofa"]
    filler = " ".join(["word"] * 80)

    def answer(good: bool) -> None:
        pend = pipeline.pending_path(cfg, "render", "t")
        items = read_jsonl(pend, PendingItem)
        lines = []
        for it in items:
            case = next(
                c
                for c in cases.values()
                if pipeline.render_prompt(c, calc, "en-US") == it.request.messages[0]["content"]
            )
            nums = [n for f in build_facts(case, calc) for n in f.required_numbers]
            text = filler + (" " + " ".join(nums) if good else "")
            lines.append(json.dumps({"key": it.key, "text": text}))
        resp = tmp_path / "resp.jsonl"
        resp.write_text("\n".join(lines) + "\n")
        pipeline.import_session_responses(cfg, pend, resp)

    s1 = pipeline.render_notes(cfg, "t")
    assert s1["notes"] == 0 and s1["pending"] > 0
    answer(good=False)
    s2 = pipeline.render_notes(cfg, "t")
    needs_numbers = s2["pending"]  # notes with required numbers failed and were retried
    answer(good=True)
    s3 = pipeline.render_notes(cfg, "t")
    notes = read_jsonl(pipeline.notes_path(cfg, "t"), RenderedNote)
    assert s3["pending"] == 0 and len(notes) == 4
    assert sum(n.attempt == 2 for n in notes) == needs_numbers > 0
    assert not any(n.validation_failed for n in notes)
    assert all(n.prompt_version == "v3" for n in notes)
