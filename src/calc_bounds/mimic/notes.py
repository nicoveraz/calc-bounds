"""Discharge-summary sections closest to what was known in the ED.

MIMIC-IV-Note has no ED provider notes, so the extractor reads selected sections of the
discharge summary of the same admission (HPI, history, admission exam, admission results).
Sections are found by heading patterns at the start of a line followed by ':'
(`MimicCriteria.note_sections`). A kept section runs until the next heading from either the
`keep` or the `stop` list, so the discharge exam and discharge labs are cut off. If no kept
heading is found, the full text is used and the fallback is recorded.

The extractor sees `SectionedNote.text`; evidence spans index into that text.
"""

import re

import pandas as pd
from pydantic import BaseModel

from calc_bounds.mimic.criteria import NoteSections

SEPARATOR = "\n\n"


class SectionedNote(BaseModel):
    note_id: str
    hadm_id: int
    text: str
    sections_found: list[str]
    """Kept section names found, in note order (unique)."""
    fallback_full_text: bool


def _heading(patterns: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(rf"^[ \t]*(?:{alternatives})[ \t]*:", re.IGNORECASE | re.MULTILINE)


def split_sections(text: str, cfg: NoteSections) -> tuple[str, list[str], bool]:
    """(selected text, kept section names found, used full-text fallback)."""
    keep = {name: _heading(patterns) for name, patterns in cfg.keep.items()}
    stop = _heading(cfg.stop)
    # Every heading position: (start, kept section name or None for a stop heading).
    marks: list[tuple[int, str | None]] = [
        (m.start(), name) for name, rx in keep.items() for m in rx.finditer(text)
    ]
    kept_starts = {pos for pos, _ in marks}
    marks += [(m.start(), None) for m in stop.finditer(text) if m.start() not in kept_starts]
    marks.sort()
    pieces: list[str] = []
    found: list[str] = []
    for i, (start, name) in enumerate(marks):
        if name is None:
            continue
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        pieces.append(text[start:end].strip())
        if name not in found:
            found.append(name)
    if not pieces:
        return text, [], True
    return SEPARATOR.join(pieces), found, False


def first_note_per_admission(notes: pd.DataFrame) -> pd.DataFrame:
    """One discharge summary per hadm_id: the lowest note_seq (then earliest charttime)."""
    ds = notes[notes["note_type"].fillna("DS").str.upper() == "DS"]
    ds = ds.sort_values(["hadm_id", "note_seq", "charttime"], kind="stable")
    return ds.drop_duplicates("hadm_id")


def sectioned_notes(notes: pd.DataFrame, cfg: NoteSections) -> dict[int, SectionedNote]:
    """hadm_id -> the selected sections of its discharge summary."""
    out: dict[int, SectionedNote] = {}
    for row in first_note_per_admission(notes).itertuples(index=False):
        text, found, fallback = split_sections(str(row.text), cfg)
        out[int(row.hadm_id)] = SectionedNote(
            note_id=str(row.note_id),
            hadm_id=int(row.hadm_id),
            text=text,
            sections_found=found,
            fallback_full_text=fallback,
        )
    return out
