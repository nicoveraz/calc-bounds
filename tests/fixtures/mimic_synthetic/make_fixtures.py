"""Generate the SYNTHETIC MIMIC-shaped fixture tables in this directory.

Every row is invented for tests: fictional ids (99xxxxxx subjects, 98xxxxxx admissions,
97xxxxxx ED stays), invented values and invented note text. Nothing here comes from MIMIC.
Only the table and column names follow the public MIMIC-IV / -ED / -Note schemas.

Regenerate from the repo root:
  uv run python tests/fixtures/mimic_synthetic/make_fixtures.py

Patients (all fictional; ED arrival on 2150-01-01 08:00 unless noted):
  99000001  70 M  pneumonia (principal dx), suspected infection, creatinine + weight, GCS 15
  99000002  50 F  chest pain (complaint), troponin 2x ULN, HTN + HLD codes, eye-drop antibiotic
  99000003  60 M  chest pain (ED dx only), discharged from the ED -> no admission, no note
  99000004  80 F  pneumonia (principal dx), no discharge note; RR only in vitalsign
  99000005  66 M  pneumonia as a secondary diagnosis only -> not in the CURB-65 cohort
  99000006  40 M  chest pain, troponin "<0.01", BMI 32.1, implausible triage HR, note has no
                  recognised headings (full-text fallback)
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
T0 = "2150-01-01 08:00:00"


def at(hours: float) -> str:
    return str(pd.Timestamp(T0) + pd.Timedelta(hours=hours))


def write(sub: str, name: str, rows: list[dict]) -> None:
    out = HERE / sub / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.endswith("_id") and c != "note_id":
            df[c] = df[c].astype("Int64")  # ids as integers, empty when missing
    df.to_csv(out, index=False)


NOTE_1 = """\
Name:  ___                     Unit No:   ___
Sex:   M
Service: MEDICINE
Allergies:
No Known Allergies (synthetic)
Chief Complaint:
SYNTHETIC TEST: cough and fever
History of Present Illness:
SYNTHETIC TEST NOTE about a fictional patient. Three days of productive cough and fever.
Denies chest pain. No hemoptysis.
Past Medical History:
None (fictional).
Social History:
___
Family History:
Noncontributory (fictional).
Physical Exam:
ADMISSION PHYSICAL EXAM:
Vitals: T 38.9 HR 110 BP 85/50 RR 32 O2 91% RA
General: alert and oriented x3, not confused
DISCHARGE PHYSICAL EXAM:
Vitals: HR 80 BP 120/70 RR 16
Pertinent Results:
ADMISSION LABS:
BUN-30 Creat-1.5
DISCHARGE LABS:
BUN-12 Creat-0.9
Brief Hospital Course:
Fictional course text.
"""

NOTE_2 = """\
Chief Complaint:
SYNTHETIC TEST: chest pain
History of Present Illness:
SYNTHETIC TEST NOTE. Fictional 50 year old woman with chest pressure at rest.
Past Medical History:
Hypertension. Hyperlipidemia.
Social History:
Never smoker (fictional).
Physical Exam:
HR 88 BP 150/90 RR 16 SpO2 98% RA
Pertinent Results:
Troponin T 0.02
Brief Hospital Course:
Fictional course text.
"""

NOTE_6 = """\
SYNTHETIC TEST NOTE WITHOUT STANDARD HEADINGS. Fictional 40 year old man, chest pain after
exercise, troponin negative. BMI 32. Course uneventful (fictional).
"""


def main() -> None:
    write(
        "ed",
        "edstays",
        [
            {"subject_id": 99000001, "hadm_id": 98000001, "stay_id": 97000001},
            {"subject_id": 99000002, "hadm_id": 98000002, "stay_id": 97000002},
            {"subject_id": 99000003, "hadm_id": None, "stay_id": 97000003},
            {"subject_id": 99000004, "hadm_id": 98000004, "stay_id": 97000004},
            {"subject_id": 99000005, "hadm_id": 98000005, "stay_id": 97000005},
            {"subject_id": 99000006, "hadm_id": 98000006, "stay_id": 97000006},
        ],
    )
    edstays = pd.read_csv(HERE / "ed" / "edstays.csv")
    edstays["intime"] = T0
    edstays["outtime"] = at(6)
    edstays["gender"] = ["M", "F", "M", "F", "M", "M"]
    edstays["disposition"] = ["ADMITTED", "ADMITTED", "HOME", "ADMITTED", "ADMITTED", "ADMITTED"]
    edstays["hadm_id"] = edstays["hadm_id"].astype("Int64")
    edstays.to_csv(HERE / "ed" / "edstays.csv", index=False)

    write(
        "ed",
        "triage",
        [
            {"subject_id": 99000001, "stay_id": 97000001, "temperature": 102.0, "heartrate": 110,
             "resprate": 32, "o2sat": 91, "sbp": 85, "dbp": 50, "pain": 3, "acuity": 2,
             "chiefcomplaint": "Cough, Fever"},
            {"subject_id": 99000002, "stay_id": 97000002, "temperature": 98.1, "heartrate": 88,
             "resprate": 16, "o2sat": 98, "sbp": 150, "dbp": 90, "pain": 6, "acuity": 2,
             "chiefcomplaint": "Chest pain"},
            {"subject_id": 99000003, "stay_id": 97000003, "temperature": 98.0, "heartrate": 72,
             "resprate": 14, "o2sat": 99, "sbp": 130, "dbp": 80, "pain": 2, "acuity": 3,
             "chiefcomplaint": "Dyspnea"},
            {"subject_id": 99000004, "stay_id": 97000004, "temperature": 99.0, "heartrate": 95,
             "resprate": None, "o2sat": 93, "sbp": 110, "dbp": 65, "pain": 0, "acuity": 2,
             "chiefcomplaint": "Fever"},
            {"subject_id": 99000005, "stay_id": 97000005, "temperature": 98.6, "heartrate": 80,
             "resprate": 18, "o2sat": 97, "sbp": 140, "dbp": 85, "pain": 0, "acuity": 3,
             "chiefcomplaint": "Fall"},
            {"subject_id": 99000006, "stay_id": 97000006, "temperature": 98.4, "heartrate": 999,
             "resprate": 16, "o2sat": 99, "sbp": 125, "dbp": 75, "pain": 5, "acuity": 3,
             "chiefcomplaint": "CHEST PRESSURE"},
        ],
    )  # fmt: skip
    write(
        "ed",
        "vitalsign",
        [
            {"subject_id": 99000004, "stay_id": 97000004, "charttime": at(0.5), "temperature": 99,
             "heartrate": 96, "resprate": 26, "o2sat": 93, "sbp": 108, "dbp": 64, "rhythm": "",
             "pain": 0},
            {"subject_id": 99000006, "stay_id": 97000006, "charttime": at(0.25), "temperature": 98,
             "heartrate": 76, "resprate": 16, "o2sat": 99, "sbp": 124, "dbp": 74, "rhythm": "",
             "pain": 4},
            {"subject_id": 99000006, "stay_id": 97000006, "charttime": at(10), "temperature": 98,
             "heartrate": 150, "resprate": 16, "o2sat": 99, "sbp": 124, "dbp": 74, "rhythm": "",
             "pain": 4},
        ],
    )  # fmt: skip
    write(
        "ed",
        "diagnosis",
        [
            {"subject_id": 99000003, "stay_id": 97000003, "seq_num": 1, "icd_code": "R079",
             "icd_version": 10, "icd_title": "SYNTHETIC CHEST PAIN"},
            {"subject_id": 99000001, "stay_id": 97000001, "seq_num": 1, "icd_code": "J189",
             "icd_version": 10, "icd_title": "SYNTHETIC PNEUMONIA"},
        ],
    )  # fmt: skip

    write(
        "hosp",
        "patients",
        [
            {"subject_id": 99000001, "gender": "M", "anchor_age": 70, "anchor_year": 2150,
             "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 99000002, "gender": "F", "anchor_age": 48, "anchor_year": 2148,
             "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 99000003, "gender": "M", "anchor_age": 60, "anchor_year": 2150,
             "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 99000004, "gender": "F", "anchor_age": 80, "anchor_year": 2150,
             "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 99000005, "gender": "M", "anchor_age": 66, "anchor_year": 2150,
             "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 99000006, "gender": "M", "anchor_age": 40, "anchor_year": 2150,
             "anchor_year_group": "2017 - 2019", "dod": ""},
        ],
    )  # fmt: skip
    write(
        "hosp",
        "admissions",
        [
            {"subject_id": s, "hadm_id": h, "admittime": at(5), "dischtime": at(80)}
            for s, h in [
                (99000001, 98000001),
                (99000002, 98000002),
                (99000004, 98000004),
                (99000005, 98000005),
                (99000006, 98000006),
            ]
        ],
    )
    write(
        "hosp",
        "diagnoses_icd",
        [
            {"subject_id": 99000001, "hadm_id": 98000001, "seq_num": 1, "icd_code": "J189",
             "icd_version": 10},
            {"subject_id": 99000002, "hadm_id": 98000002, "seq_num": 1, "icd_code": "R079",
             "icd_version": 10},
            {"subject_id": 99000002, "hadm_id": 98000002, "seq_num": 2, "icd_code": "I10",
             "icd_version": 10},
            {"subject_id": 99000002, "hadm_id": 98000002, "seq_num": 3, "icd_code": "2724",
             "icd_version": 9},
            {"subject_id": 99000004, "hadm_id": 98000004, "seq_num": 1, "icd_code": "486",
             "icd_version": 9},
            {"subject_id": 99000005, "hadm_id": 98000005, "seq_num": 1, "icd_code": "S0990XA",
             "icd_version": 10},
            {"subject_id": 99000005, "hadm_id": 98000005, "seq_num": 3, "icd_code": "J189",
             "icd_version": 10},
        ],
    )  # fmt: skip
    write(
        "hosp",
        "labevents",
        [
            # BUN (51006), creatinine (50912), troponin T (51003). Fictional values.
            {"labevent_id": 1, "subject_id": 99000001, "hadm_id": None, "itemid": 51006,
             "charttime": at(1), "value": "30", "valuenum": 30, "valueuom": "mg/dL",
             "ref_range_lower": 6, "ref_range_upper": 20, "flag": "abnormal"},
            {"labevent_id": 2, "subject_id": 99000001, "hadm_id": 98000001, "itemid": 51006,
             "charttime": at(30), "value": "12", "valuenum": 12, "valueuom": "mg/dL",
             "ref_range_lower": 6, "ref_range_upper": 20, "flag": ""},
            {"labevent_id": 3, "subject_id": 99000001, "hadm_id": None, "itemid": 50912,
             "charttime": at(1), "value": "1.5", "valuenum": 1.5, "valueuom": "mg/dL",
             "ref_range_lower": 0.5, "ref_range_upper": 1.2, "flag": "abnormal"},
            {"labevent_id": 4, "subject_id": 99000002, "hadm_id": None, "itemid": 51003,
             "charttime": at(1), "value": "0.02", "valuenum": 0.02, "valueuom": "ng/mL",
             "ref_range_lower": 0, "ref_range_upper": 0.01, "flag": "abnormal"},
            {"labevent_id": 5, "subject_id": 99000006, "hadm_id": None, "itemid": 51003,
             "charttime": at(1), "value": "<0.01", "valuenum": None, "valueuom": "ng/mL",
             "ref_range_lower": 0, "ref_range_upper": 0.01, "flag": ""},
            {"labevent_id": 6, "subject_id": 99000004, "hadm_id": None, "itemid": 51006,
             "charttime": at(-48), "value": "40", "valuenum": 40, "valueuom": "mg/dL",
             "ref_range_lower": 6, "ref_range_upper": 20, "flag": "abnormal"},
        ],
    )  # fmt: skip
    write(
        "hosp",
        "omr",
        [
            {"subject_id": 99000001, "chartdate": "2149-11-01", "seq_num": 1,
             "result_name": "Weight (Lbs)", "result_value": "154"},
            {"subject_id": 99000001, "chartdate": "2148-01-01", "seq_num": 1,
             "result_name": "Weight (Lbs)", "result_value": "200"},
            {"subject_id": 99000006, "chartdate": "2150-02-01", "seq_num": 1,
             "result_name": "BMI (kg/m2)", "result_value": "32.1"},
            {"subject_id": 99000006, "chartdate": "2150-02-01", "seq_num": 1,
             "result_name": "Blood Pressure", "result_value": "120/80"},
        ],
    )  # fmt: skip
    write(
        "hosp",
        "prescriptions",
        [
            {"subject_id": 99000001, "hadm_id": 98000001, "starttime": at(2),
             "drug": "CefTRIAXone", "route": "IV"},
            {"subject_id": 99000002, "hadm_id": 98000002, "starttime": at(2),
             "drug": "Ciprofloxacin 0.3% Ophth Soln", "route": "OU"},
            {"subject_id": 99000002, "hadm_id": 98000002, "starttime": at(3),
             "drug": "Aspirin", "route": "PO"},
        ],
    )  # fmt: skip
    write(
        "hosp",
        "microbiologyevents",
        [
            {"microevent_id": 1, "subject_id": 99000001, "hadm_id": None,
             "micro_specimen_id": 1, "chartdate": "2150-01-01", "charttime": at(1),
             "spec_type_desc": "BLOOD CULTURE"},
            {"microevent_id": 2, "subject_id": 99000002, "hadm_id": 98000002,
             "micro_specimen_id": 2, "chartdate": "2150-01-01", "charttime": at(1),
             "spec_type_desc": "URINE"},
        ],
    )  # fmt: skip
    write(
        "icu",
        "chartevents",
        [
            {
                "subject_id": 99000001,
                "hadm_id": 98000001,
                "stay_id": 96000001,
                "charttime": at(5),
                "itemid": item,
                "value": str(v),
                "valuenum": v,
                "valueuom": "",
            }
            for item, v in [(220739, 4), (223900, 5), (223901, 6)]
        ],
    )
    write(
        "note",
        "discharge",
        [
            {"note_id": "98000001-DS-1", "subject_id": 99000001, "hadm_id": 98000001,
             "note_type": "DS", "note_seq": 1, "charttime": at(80), "storetime": at(81),
             "text": NOTE_1},
            {"note_id": "98000002-DS-1", "subject_id": 99000002, "hadm_id": 98000002,
             "note_type": "DS", "note_seq": 1, "charttime": at(80), "storetime": at(81),
             "text": NOTE_2},
            {"note_id": "98000005-DS-1", "subject_id": 99000005, "hadm_id": 98000005,
             "note_type": "DS", "note_seq": 1, "charttime": at(80), "storetime": at(81),
             "text": "SYNTHETIC TEST NOTE. Fictional fall."},
            {"note_id": "98000006-DS-1", "subject_id": 99000006, "hadm_id": 98000006,
             "note_type": "DS", "note_seq": 1, "charttime": at(80), "storetime": at(81),
             "text": NOTE_6},
        ],
    )  # fmt: skip


if __name__ == "__main__":
    main()
