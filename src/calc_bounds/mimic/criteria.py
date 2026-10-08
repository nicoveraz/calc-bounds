"""Cohort and structured-truth definitions for MIMIC, loaded from a YAML file
(`configs/mimic_criteria.yaml`).

Every ICD list, itemid, pattern and time window is an operational definition chosen by us, not
taken from a primary source. Each is marked TODO(physician-review) in the YAML file and listed
in docs/MIMIC_VALIDATION.md. Keeping them in data (not code) lets them be reviewed and changed
without touching the pipeline.

ICD codes are prefixes without dots, as stored in MIMIC-IV (e.g. "J18" matches "J189").
"""

import re
from pathlib import Path
from typing import Self

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_bounds.types import ParamId


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IcdSet(Strict):
    """ICD code prefixes (no dots), by ICD version."""

    icd9: tuple[str, ...] = ()
    icd10: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_dots(self) -> Self:
        for code in (*self.icd9, *self.icd10):
            if "." in code or code != code.strip().upper():
                raise ValueError(f"ICD prefix {code!r}: use MIMIC format (upper case, no dots)")
        return self

    def matches(self, codes: pd.DataFrame) -> pd.Series:
        """Boolean mask over rows with columns `icd_code`, `icd_version`."""
        code = codes["icd_code"].astype(str).str.strip().str.upper()
        version = pd.to_numeric(codes["icd_version"], errors="coerce")
        hit9 = code.str.startswith(self.icd9) if self.icd9 else False
        hit10 = code.str.startswith(self.icd10) if self.icd10 else False
        return ((version == 9) & hit9) | ((version == 10) & hit10)


class Windows(Strict):
    """Time windows relative to ED arrival (edstays.intime)."""

    vitals_hours: float = Field(gt=0)
    """Fallback to the first `vitalsign` row within this window when triage lacks a value."""
    labs_hours: float = Field(gt=0)
    """First lab result in [intime, intime + labs_hours]."""
    gcs_hours: float = Field(gt=0)
    """First complete GCS in [intime, intime + gcs_hours] (chartevents; ICU only)."""
    weight_days: float = Field(gt=0)
    """omr weight / BMI closest to ED arrival within +/- this many days."""
    infection_hours: tuple[float, float]
    """Suspicion-of-infection time must fall in [intime + a, intime + b] hours."""


class ItemIds(Strict):
    bun: int
    creatinine: int
    troponin_t: int
    d_dimer: int
    gcs_eye: int
    gcs_verbal: int
    gcs_motor: int


class OmrNames(Strict):
    weight_lbs: str
    bmi: str


class SuspectedInfection(Strict):
    """Antibiotic + culture pairing (Seymour 2016 / mimic-code suspicion_of_infection)."""

    antibiotic_pattern: str
    """Case-insensitive regex on prescriptions.drug."""
    exclude_drug_pattern: str
    exclude_routes: tuple[str, ...]
    culture_then_antibiotic_hours: float = Field(gt=0)
    antibiotic_then_culture_hours: float = Field(gt=0)


class Cohorts(Strict):
    pneumonia: IcdSet
    pneumonia_max_seq_num: int | None = Field(ge=1)
    """Only diagnoses with seq_num <= this (1 = principal diagnosis); None = any position."""
    suspected_infection: SuspectedInfection
    chest_pain_complaint: str
    """Case-insensitive regex on triage.chiefcomplaint."""
    chest_pain_ed_icd: IcdSet
    """ED diagnosis (MIMIC-IV-ED `diagnosis`) codes that also qualify a chest-pain visit."""


class NoteSections(Strict):
    keep: dict[str, tuple[str, ...]]
    """Section name -> heading regexes (case-insensitive, at line start, followed by ':')."""
    stop: tuple[str, ...]
    """Other headings that end a kept section (e.g. 'Brief Hospital Course')."""

    @model_validator(mode="after")
    def _compile(self) -> Self:
        for pattern in (*[p for ps in self.keep.values() for p in ps], *self.stop):
            re.compile(pattern)
        return self


class MimicCriteria(Strict):
    windows: Windows
    itemids: ItemIds
    omr: OmrNames
    cohorts: Cohorts
    obesity_bmi_above: float
    """BMI strictly above this counts as obesity (HEART parameter label: BMI > 30)."""
    gcs_altered_below: int
    """GCS total below this = altered mentation / confusion proxy."""
    comorbidity_icd: dict[ParamId, IcdSet]
    """Boolean params set True when a code is present. A missing code leaves them Unknown,
    never False (Unknown is never Absent)."""
    annotation_params: tuple[ParamId, ...]
    """Judgement items that structured data cannot provide; physician annotation only."""
    secondary_annotation_params: tuple[ParamId, ...] = ()
    """Items the record usually lacks (e.g. confusion without ICU GCS). Annotated from the
    note for a SECONDARY analysis only; the primary analysis leaves them Unknown."""
    note_sections: NoteSections

    @model_validator(mode="after")
    def _check(self) -> Self:
        re.compile(self.cohorts.chest_pain_complaint)
        re.compile(self.cohorts.suspected_infection.antibiotic_pattern)
        re.compile(self.cohorts.suspected_infection.exclude_drug_pattern)
        overlap = set(self.comorbidity_icd) & set(self.annotation_params)
        overlap |= set(self.annotation_params) & set(self.secondary_annotation_params)
        if overlap:
            raise ValueError(f"params both ICD-mapped and annotated: {sorted(overlap)}")
        return self


def load_criteria(path: Path) -> MimicCriteria:
    return MimicCriteria.model_validate(yaml.safe_load(path.read_text()))
