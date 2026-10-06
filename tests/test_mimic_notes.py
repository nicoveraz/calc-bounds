"""Discharge-note section extraction on SYNTHETIC notes."""

from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from calc_bounds.mimic import tables as T
from calc_bounds.mimic.criteria import load_criteria
from calc_bounds.mimic.notes import SEPARATOR, sectioned_notes, split_sections

FIX = Path(__file__).parent / "fixtures" / "mimic_synthetic"
DIRS = T.TableDirs(
    hosp=FIX / "hosp", icu=FIX / "icu", ed=FIX / "ed", notes_path=FIX / "note" / "discharge.csv"
)
SECTIONS = load_criteria(Path("configs/mimic_criteria.yaml")).note_sections


def _notes() -> dict:
    return sectioned_notes(T.load_discharge_notes(DIRS, {98000001, 98000002, 98000006}), SECTIONS)


def test_sections_found_and_discharge_parts_cut() -> None:
    n = _notes()[98000001]
    assert n.fallback_full_text is False
    assert n.sections_found == [
        "chief_complaint",
        "hpi",
        "past_medical_history",
        "social_history",
        "family_history",
        "physical_exam",
        "pertinent_results",
    ]
    assert "RR 32" in n.text and "BUN-30" in n.text  # admission exam and labs kept
    assert "RR 16" not in n.text and "BUN-12" not in n.text  # discharge exam and labs cut
    assert "Allergies" not in n.text and "Fictional course" not in n.text
    assert "Unit No" not in n.text


def test_fallback_to_full_text_without_headings() -> None:
    n = _notes()[98000006]
    assert n.fallback_full_text is True and n.sections_found == []
    assert n.text.startswith("SYNTHETIC TEST NOTE WITHOUT STANDARD HEADINGS")


def test_heading_must_start_a_line() -> None:
    text = "Comment: see physical exam: below\nHistory of Present Illness:\nfictional\n"
    selected, found, fallback = split_sections(text, SECTIONS)
    assert found == ["hpi"] and not fallback
    assert selected == "History of Present Illness:\nfictional"


lines = st.lists(
    st.sampled_from(
        [
            "History of Present Illness:",
            "Physical Exam:",
            "DISCHARGE PHYSICAL EXAM:",
            "Pertinent Results:",
            "Brief Hospital Course:",
            "fictional text line",
            "HR 80",
        ]
    ),
    max_size=15,
)


@given(lines)
def test_selected_pieces_are_substrings_of_the_note(note_lines: list[str]) -> None:
    """Every selected piece is verbatim note text (so evidence quotes stay checkable)."""
    text = "\n".join(note_lines)
    selected, found, fallback = split_sections(text, SECTIONS)
    if fallback:
        assert selected == text and found == []
    else:
        for piece in selected.split(SEPARATOR):
            assert piece in text
        assert "DISCHARGE PHYSICAL EXAM" not in selected
        assert "Brief Hospital Course" not in selected
