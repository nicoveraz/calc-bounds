"""MIMIC table loaders on SYNTHETIC fixtures (fictional rows, public column names)."""

from pathlib import Path

import pandas as pd
import pytest

from calc_bounds.mimic import tables as T

FIX = Path(__file__).parent / "fixtures" / "mimic_synthetic"
DIRS = T.TableDirs(
    hosp=FIX / "hosp", icu=FIX / "icu", ed=FIX / "ed", notes_path=FIX / "note" / "discharge.csv"
)


def test_table_path_prefers_gz_then_csv(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a\n1\n")
    assert T.table_path(tmp_path, "x").name == "x.csv"
    (tmp_path / "x.csv.gz").write_bytes(b"not empty")
    assert T.table_path(tmp_path, "x").name == "x.csv.gz"
    with pytest.raises(FileNotFoundError):
        T.table_path(tmp_path, "missing")


def test_gz_is_read_like_csv(tmp_path: Path) -> None:
    df = pd.read_csv(FIX / "ed" / "edstays.csv")
    df.to_csv(tmp_path / "edstays.csv.gz", index=False)
    a = T.load_edstays(DIRS)
    b = T.load_edstays(DIRS.model_copy(update={"ed": tmp_path}))
    pd.testing.assert_frame_equal(a, b)


def test_edstays_types() -> None:
    df = T.load_edstays(DIRS)
    assert list(df.columns) == T.EDSTAYS
    assert str(df["stay_id"].dtype) == "Int64"
    assert df["hadm_id"].isna().sum() == 1  # visit discharged from the ED
    assert pd.api.types.is_datetime64_any_dtype(df["intime"])


def test_triage_and_vitals_numeric() -> None:
    tri = T.load_triage(DIRS)
    assert pd.api.types.is_numeric_dtype(tri["heartrate"])
    vs = T.load_vitalsign(DIRS, {97000006})
    assert set(vs["stay_id"]) == {97000006} and len(vs) == 2


def test_labevents_filtered_by_item_and_subject() -> None:
    labs = T.load_labevents(DIRS, {51006}, {99000001})
    assert set(labs["itemid"]) == {51006} and set(labs["subject_id"]) == {99000001}
    assert len(labs) == 2
    trop = T.load_labevents(DIRS, {51003}, {99000006})
    assert trop["valuenum"].isna().all() and trop["value"].iloc[0] == "<0.01"


def test_filtered_read_with_no_match_keeps_columns() -> None:
    labs = T.load_labevents(DIRS, {1}, {1})
    assert labs.empty and list(labs.columns) == T.LABEVENTS


def test_other_loaders() -> None:
    assert len(T.load_patients(DIRS)) == 6
    assert len(T.load_admissions(DIRS)) == 5
    assert len(T.load_ed_diagnosis(DIRS)) == 2
    assert len(T.load_diagnoses_icd(DIRS, {98000002})) == 3
    assert len(T.load_diagnoses_icd(DIRS)) == 7
    gcs = T.load_chartevents(DIRS, {220739, 223900, 223901}, {99000001})
    assert sorted(gcs["valuenum"]) == [4, 5, 6]
    omr = T.load_omr(DIRS, {"Weight (Lbs)"}, {99000001, 99000006})
    assert set(omr["subject_id"]) == {99000001}
    assert len(T.load_prescriptions(DIRS, {98000002})) == 2
    assert len(T.load_microbiologyevents(DIRS, {99000001})) == 1
    notes = T.load_discharge_notes(DIRS, {98000001, 98000006})
    assert set(notes["note_id"]) == {"98000001-DS-1", "98000006-DS-1"}
    assert notes["text"].str.contains("SYNTHETIC").all()


def test_empty_or_truncated_table_names_the_file(tmp_path) -> None:
    import gzip

    import pytest

    from calc_bounds.mimic.tables import TableFileError, read_table, table_path

    (tmp_path / "omr.csv.gz").write_bytes(b"")
    with pytest.raises(TableFileError, match="0 bytes"):
        table_path(tmp_path, "omr")

    data = gzip.compress(b"subject_id,hadm_id\n1,2\n" * 1000)
    (tmp_path / "admissions.csv.gz").write_bytes(data[: len(data) // 2])
    with pytest.raises(TableFileError, match="incomplete download"):
        read_table(tmp_path / "admissions.csv.gz", ["subject_id", "hadm_id"])
