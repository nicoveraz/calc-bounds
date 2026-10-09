"""Terminal annotation helper, with scripted answers (synthetic ids)."""

from pathlib import Path

import pandas as pd

from calc_bounds.mimic.annotate import load_progress, parse_answer, run_session
from calc_bounds.mimic.annotation import COLUMNS


def _template(path: Path) -> Path:
    rows = [
        ("heart-97000001", "heart", "heart_history", "primary", "low | moderate | high | unknown"),
        ("heart-97000001", "heart", "heart_ecg", "primary", "normal | nonspecific | st | unknown"),
        ("curb65-97000002", "curb65", "confusion", "secondary", "yes | no | unknown"),
    ]
    df = pd.DataFrame(
        [
            {
                "case_id": c, "calculator": k, "subject_id": "99000001", "hadm_id": "98000001",
                "note_id": "98000001-DS-1", "param": p, "analysis": a, "allowed_values": v,
                "value": "", "annotator": "", "comment": "",
            }
            for c, k, p, a, v in rows
        ],
        columns=COLUMNS,
    )  # fmt: skip
    df.to_csv(path, index=False)
    return path


def test_parse_answer() -> None:
    allowed = ["yes", "no", "unknown"]
    assert parse_answer("1", allowed) == "yes"
    assert parse_answer("No", allowed) == "no"
    assert parse_answer("u", allowed) == "unknown"
    assert parse_answer("7", allowed) is None
    assert parse_answer("72", ["number in kg", "unknown"]) == "72"


def test_session_saves_and_resumes(tmp_path: Path) -> None:
    template, filled = _template(tmp_path / "t.csv"), tmp_path / "f.csv"
    shown: list[str] = []
    answers = iter(["9", "2", "q"])  # invalid, then 'moderate', then quit
    p = run_session(
        load_progress(template, filled),
        lambda _id: "SYNTHETIC NOTE",
        lambda df: df.to_csv(filled, index=False),
        "NV",
        ask=lambda _prompt: next(answers),
        show=shown.append,
    )
    assert p.set_index("param").at["heart_history", "value"] == "moderate"
    assert "  not one of the options" in shown
    resumed = load_progress(template, filled).set_index("param")
    assert resumed.at["heart_history", "value"] == "moderate"
    assert resumed.at["heart_history", "annotator"] == "NV"
    assert resumed.at["heart_ecg", "value"] == ""  # stopped before it
