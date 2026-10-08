"""Loaders for the MIMIC-IV, MIMIC-IV-ED and MIMIC-IV-Note tables the pipeline needs.

Plain pandas over CSV or CSV.GZ (`<dir>/<table>.csv.gz`, else `<dir>/<table>.csv`), using the
public column names. Large tables (labevents, chartevents, prescriptions, microbiologyevents,
discharge) are read in chunks and filtered to the subjects / items needed, so they never sit
in memory whole. Times are parsed to pandas datetimes; ids are nullable integers.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

CHUNKSIZE = 1_000_000

ID_COLUMNS = ("subject_id", "hadm_id", "stay_id", "itemid", "micro_specimen_id", "seq_num")

# Public column names (MIMIC-IV 2.2, MIMIC-IV-ED 2.2, MIMIC-IV-Note 2.2); only what we use.
EDSTAYS = ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
TRIAGE = ["subject_id", "stay_id", "heartrate", "resprate", "o2sat", "sbp", "dbp", "chiefcomplaint"]
VITALSIGN = ["subject_id", "stay_id", "charttime", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
ED_DIAGNOSIS = ["subject_id", "stay_id", "seq_num", "icd_code", "icd_version"]
PATIENTS = ["subject_id", "gender", "anchor_age", "anchor_year"]
ADMISSIONS = ["subject_id", "hadm_id", "admittime", "dischtime"]
DIAGNOSES_ICD = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
LABEVENTS = [
    "subject_id",
    "hadm_id",
    "itemid",
    "charttime",
    "value",
    "valuenum",
    "valueuom",
    "ref_range_upper",
]
CHARTEVENTS = ["subject_id", "stay_id", "charttime", "itemid", "valuenum"]
OMR = ["subject_id", "chartdate", "result_name", "result_value"]
PRESCRIPTIONS = ["subject_id", "hadm_id", "starttime", "drug", "route"]
MICROBIOLOGYEVENTS = ["subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime"]
DISCHARGE = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "text"]

TIME_COLUMNS = ("intime", "outtime", "charttime", "admittime", "dischtime", "starttime")
DATE_COLUMNS = ("chartdate",)


class TableDirs(BaseModel):
    model_config = ConfigDict(frozen=True)

    hosp: Path
    icu: Path
    ed: Path
    notes_path: Path


class TableFileError(RuntimeError):
    """A MIMIC table file is missing, empty or corrupt (e.g. an interrupted download)."""


def table_path(directory: Path, name: str) -> Path:
    for suffix in (".csv.gz", ".csv"):
        path = directory / f"{name}{suffix}"
        if path.exists():
            if path.stat().st_size == 0:
                raise TableFileError(f"{path} is empty (0 bytes); download it again")
            return path
    raise FileNotFoundError(f"no {name}.csv.gz or {name}.csv in {directory}")


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ID_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif c in TIME_COLUMNS or c in DATE_COLUMNS:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def read_table(
    path: Path,
    columns: list[str],
    keep: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> pd.DataFrame:
    """Read `columns` of a CSV(.gz); with `keep`, read in chunks and keep matching rows only.

    A truncated .gz raises EOFError, which the CLI framework would turn into a bare
    "Aborted!"; re-raise it naming the file.
    """
    try:
        return _read_table(path, columns, keep)
    except (EOFError, OSError, UnicodeDecodeError) as exc:
        raise TableFileError(
            f"{path} could not be read ({type(exc).__name__}: {exc}); "
            "it is probably an incomplete download: verify it against SHA256SUMS.txt"
        ) from exc


def _read_table(
    path: Path,
    columns: list[str],
    keep: Callable[[pd.DataFrame], pd.Series] | None,
) -> pd.DataFrame:
    if keep is None:
        return _typed(pd.read_csv(path, usecols=columns, dtype=str))
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, chunksize=CHUNKSIZE):
        chunk = _typed(chunk)
        parts.append(chunk[keep(chunk).fillna(False).astype(bool)])
    return pd.concat(parts, ignore_index=True) if parts else _typed(pd.DataFrame(columns=columns))


def _in(column: str, values: Iterable[int]) -> Callable[[pd.DataFrame], pd.Series]:
    wanted = set(values)
    return lambda d: d[column].isin(wanted)


# --- MIMIC-IV-ED ------------------------------------------------------------------------------


def load_edstays(d: TableDirs) -> pd.DataFrame:
    return read_table(table_path(d.ed, "edstays"), EDSTAYS)


def load_triage(d: TableDirs) -> pd.DataFrame:
    df = read_table(table_path(d.ed, "triage"), TRIAGE)
    for c in ("heartrate", "resprate", "o2sat", "sbp", "dbp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_vitalsign(d: TableDirs, stay_ids: Iterable[int]) -> pd.DataFrame:
    df = read_table(table_path(d.ed, "vitalsign"), VITALSIGN, _in("stay_id", stay_ids))
    for c in ("heartrate", "resprate", "o2sat", "sbp", "dbp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_ed_diagnosis(d: TableDirs) -> pd.DataFrame:
    return read_table(table_path(d.ed, "diagnosis"), ED_DIAGNOSIS)


# --- MIMIC-IV hosp / icu ----------------------------------------------------------------------


def load_patients(d: TableDirs) -> pd.DataFrame:
    df = read_table(table_path(d.hosp, "patients"), PATIENTS)
    df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
    return df


def load_admissions(d: TableDirs) -> pd.DataFrame:
    return read_table(table_path(d.hosp, "admissions"), ADMISSIONS)


def load_diagnoses_icd(d: TableDirs, hadm_ids: Iterable[int] | None = None) -> pd.DataFrame:
    keep = None if hadm_ids is None else _in("hadm_id", hadm_ids)
    return read_table(table_path(d.hosp, "diagnoses_icd"), DIAGNOSES_ICD, keep)


def load_labevents(
    d: TableDirs, itemids: Iterable[int], subject_ids: Iterable[int]
) -> pd.DataFrame:
    items, subjects = set(itemids), set(subject_ids)
    df = read_table(
        table_path(d.hosp, "labevents"),
        LABEVENTS,
        lambda c: c["itemid"].isin(items) & c["subject_id"].isin(subjects),
    )
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df["ref_range_upper"] = pd.to_numeric(df["ref_range_upper"], errors="coerce")
    return df


def load_chartevents(
    d: TableDirs, itemids: Iterable[int], subject_ids: Iterable[int]
) -> pd.DataFrame:
    items, subjects = set(itemids), set(subject_ids)
    df = read_table(
        table_path(d.icu, "chartevents"),
        CHARTEVENTS,
        lambda c: c["itemid"].isin(items) & c["subject_id"].isin(subjects),
    )
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    return df


def load_omr(d: TableDirs, result_names: Iterable[str], subject_ids: Iterable[int]) -> pd.DataFrame:
    names, subjects = set(result_names), set(subject_ids)
    return read_table(
        table_path(d.hosp, "omr"),
        OMR,
        lambda c: c["result_name"].isin(names) & c["subject_id"].isin(subjects),
    )


def load_prescriptions(d: TableDirs, hadm_ids: Iterable[int]) -> pd.DataFrame:
    return read_table(table_path(d.hosp, "prescriptions"), PRESCRIPTIONS, _in("hadm_id", hadm_ids))


def load_microbiologyevents(d: TableDirs, subject_ids: Iterable[int]) -> pd.DataFrame:
    return read_table(
        table_path(d.hosp, "microbiologyevents"), MICROBIOLOGYEVENTS, _in("subject_id", subject_ids)
    )


# --- MIMIC-IV-Note ------------------------------------------------------------------------------


def load_discharge_notes(d: TableDirs, hadm_ids: Iterable[int]) -> pd.DataFrame:
    df = read_table(d.notes_path, DISCHARGE, _in("hadm_id", hadm_ids))
    df["note_seq"] = pd.to_numeric(df["note_seq"], errors="coerce")
    return df
