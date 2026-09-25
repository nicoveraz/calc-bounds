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
