"""Per-calculator cohorts over MIMIC tables. Each function returns a set of ED `stay_id`s.

The definitions (codes, patterns, windows) come from `MimicCriteria`; see
configs/mimic_criteria.yaml, where every item is marked TODO(physician-review).

  CURB-65          pneumonia diagnosis of the admission that followed the ED visit
  qSOFA            suspected infection (antibiotic + culture pairing) around ED arrival
  HEART            chest-pain chief complaint, or a chest-pain ED diagnosis
  Cockcroft-Gault  creatinine and weight both available (decided after the truth mapping)
  PERC / Wells     D-dimer measured around ED arrival (optional)
"""

import pandas as pd

from calc_bounds.mimic.criteria import Cohorts, MimicCriteria, SuspectedInfection


def _hours(h: float) -> pd.Timedelta:
    return pd.Timedelta(hours=h)


def pneumonia_stays(edstays: pd.DataFrame, diagnoses_icd: pd.DataFrame, c: Cohorts) -> set[int]:
    dx = diagnoses_icd[c.pneumonia.matches(diagnoses_icd)]
    if c.pneumonia_max_seq_num is not None:
        dx = dx[dx["seq_num"] <= c.pneumonia_max_seq_num]
    hadm = set(dx["hadm_id"].dropna())
    return set(edstays.loc[edstays["hadm_id"].isin(hadm), "stay_id"])


def chest_pain_stays(
    edstays: pd.DataFrame, triage: pd.DataFrame, ed_dx: pd.DataFrame, c: Cohorts
) -> set[int]:
    complaint = (
        triage["chiefcomplaint"]
        .fillna("")
        .str.contains(c.chest_pain_complaint, case=False, regex=True)
    )
    by_complaint = set(triage.loc[complaint, "stay_id"])
    by_dx = set(ed_dx.loc[c.chest_pain_ed_icd.matches(ed_dx), "stay_id"])
    return (by_complaint | by_dx) & set(edstays["stay_id"])


def antibiotics(prescriptions: pd.DataFrame, s: SuspectedInfection) -> pd.DataFrame:
    """Systemic antibiotic orders: columns subject_id, time."""
    drug = prescriptions["drug"].fillna("")
    route = prescriptions["route"].fillna("").str.strip().str.upper()
    keep = (
        drug.str.contains(s.antibiotic_pattern, case=False, regex=True)
        & ~drug.str.contains(s.exclude_drug_pattern, case=False, regex=True)
        & ~route.isin(s.exclude_routes)
    )
    out = prescriptions.loc[keep, ["subject_id", "starttime"]].rename(columns={"starttime": "time"})
    return out.dropna(subset=["time"])


def cultures(micro: pd.DataFrame) -> pd.DataFrame:
    """One row per specimen: columns subject_id, time (charttime, else chartdate)."""
    df = micro.assign(time=micro["charttime"].fillna(micro["chartdate"]))
    df = df.dropna(subset=["time"]).drop_duplicates(["subject_id", "micro_specimen_id"])
    return df[["subject_id", "time"]]


def suspicion_times(
    edstays: pd.DataFrame,
    abx: pd.DataFrame,
    cult: pd.DataFrame,
    s: SuspectedInfection,
    window_hours: tuple[float, float],
) -> pd.DataFrame:
    """Suspected-infection times per ED stay (Seymour 2016 pairing): a culture followed by an
    antibiotic within `culture_then_antibiotic_hours` (time = culture), or an antibiotic
    followed by a culture within `antibiotic_then_culture_hours` (time = antibiotic). Only
    times within `window_hours` of ED arrival are returned."""
    lo, hi = (_hours(h) for h in window_hours)
    horizon = _hours(max(s.culture_then_antibiotic_hours, s.antibiotic_then_culture_hours))
    stays = edstays[["stay_id", "subject_id", "intime"]]

    def near_stay(events: pd.DataFrame, name: str) -> pd.DataFrame:
        # Keep each subject's events only around their ED stays, before pairing (keeps the
        # pairing join small on the full tables).
        m = stays.merge(events.rename(columns={"time": name}), on="subject_id")
        ok = m[name].between(m["intime"] + lo - horizon, m["intime"] + hi + horizon)
        return m[ok]

    pairs = near_stay(abx, "abx_time").merge(
        near_stay(cult, "culture_time"), on=["stay_id", "subject_id", "intime"]
    )
    gap = pairs["abx_time"] - pairs["culture_time"]
    culture_first = (gap >= pd.Timedelta(0)) & (gap <= _hours(s.culture_then_antibiotic_hours))
    abx_first = (gap < pd.Timedelta(0)) & (-gap <= _hours(s.antibiotic_then_culture_hours))
    pairs["suspicion_time"] = pairs["culture_time"].where(culture_first, pairs["abx_time"])
    pairs = pairs[culture_first | abx_first]
    near = pairs["suspicion_time"].between(pairs["intime"] + lo, pairs["intime"] + hi)
    return pairs.loc[near, ["stay_id", "suspicion_time"]]


def suspected_infection_stays(
    edstays: pd.DataFrame, prescriptions: pd.DataFrame, micro: pd.DataFrame, crit: MimicCriteria
) -> set[int]:
    s = crit.cohorts.suspected_infection
    t = suspicion_times(
        edstays, antibiotics(prescriptions, s), cultures(micro), s, crit.windows.infection_hours
    )
    return set(t["stay_id"])


def lab_in_window_stays(
    edstays: pd.DataFrame, labs: pd.DataFrame, itemid: int, hours: float
) -> set[int]:
    """Stays with any result of `itemid` in [ED arrival, arrival + hours] (e.g. D-dimer)."""
    m = edstays[["stay_id", "subject_id", "intime"]].merge(
        labs[labs["itemid"] == itemid], on="subject_id"
    )
    near = m["charttime"].between(m["intime"], m["intime"] + _hours(hours))
    return set(m.loc[near, "stay_id"])
