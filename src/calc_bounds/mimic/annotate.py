"""Terminal helper for the physician annotation, run locally next to the MIMIC data.

Shows, case by case, the same note sections the model reads (admission information only, so
the hospital course does not leak the outcome), then asks each pending item. Answers are saved
after every item to `annotation_filled.csv` in the run directory (outside the repo), so a
session can stop and resume. Nothing leaves the machine.
"""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from calc_bounds.mimic.annotation import COLUMNS, UNKNOWN

QUIT, SKIP = "q", ""


def load_progress(template: Path, filled: Path) -> pd.DataFrame:
    """The template with any answers already saved in `filled` (matched by case and item)."""
    t = pd.read_csv(template, dtype=str, keep_default_na=False)
    if filled.exists():
        f = pd.read_csv(filled, dtype=str, keep_default_na=False).set_index(["case_id", "param"])
        for i, row in t.iterrows():
            key = (row["case_id"], row["param"])
            if key in f.index:
                for c in ("value", "annotator", "comment"):
                    t.at[i, c] = f.at[key, c]
    return t[COLUMNS]


def choices(allowed: str) -> list[str]:
    return [a.strip() for a in allowed.split("|")]


def parse_answer(raw: str, allowed: list[str]) -> str | None:
    """A number (1-based) or the text of an allowed value; 'u' = unknown. None = invalid."""
    text = raw.strip().lower()
    if text == "u":
        return UNKNOWN
    if text.isdigit() and 1 <= int(text) <= len(allowed):
        return allowed[int(text) - 1]
    numeric = [a for a in allowed if a.startswith("number in")]
    if numeric:
        try:
            float(text)
            return text
        except ValueError:
            pass
    return text if text in allowed else None


def run_session(
    progress: pd.DataFrame,
    note_text: Callable[[str], str],
    save: Callable[[pd.DataFrame], None],
    annotator: str,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Interactive loop over unanswered rows, primary items first. Enter skips, q quits."""
    order = progress.assign(_secondary=progress["analysis"] == "secondary").sort_values(
        ["_secondary", "calculator", "case_id"], kind="stable"
    )
    pending = order[order["value"] == ""]
    total, done = len(progress), int((progress["value"] != "").sum())
    current = None
    for i, row in pending.iterrows():
        if row["case_id"] != current:
            current = row["case_id"]
            show("\n" + "=" * 78)
            show(
                f"{row['case_id']}  ({row['calculator']}, note {row['note_id']})  [{done}/{total}]"
            )
            show("=" * 78)
            show(note_text(row["note_id"]))
        allowed = choices(row["allowed_values"])
        menu = "  ".join(f"{k}) {a}" for k, a in enumerate(allowed, start=1))
        while True:
            raw = ask(
                f"\n{row['param']} [{row['analysis']}]  {menu}  (u=unknown, Enter=skip, q=quit): "
            )
            if raw.strip().lower() == QUIT:
                return progress
            if raw.strip() == SKIP:
                break
            value = parse_answer(raw, allowed)
            if value is None:
                show("  not one of the options")
                continue
            progress.at[i, "value"] = value
            progress.at[i, "annotator"] = annotator
            save(progress)
            done += 1
            break
    return progress
