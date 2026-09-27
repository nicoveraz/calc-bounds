### Renderer (system)

```text
You write realistic clinical notes for a research dataset. Every patient is fictional; the notes are synthetic. You follow the fact sheet exactly: each listed item must be documented exactly as instructed, and items marked DO NOT MENTION must be completely absent and not inferable. Output only the note text.
```

### Renderer (user), example CURB-65 case

```text
Write one clinical note in US English (en-US).

Setting: Emergency department. Adult with suspected community-acquired pneumonia.

Style guide:
- US emergency department or inpatient documentation conventions.
- Typical sections: Chief complaint, HPI, PMH, Medications, Allergies, Social history,
  Exam (vitals first), Results, Assessment and plan. Not every section is required.
- Common US abbreviations are fine (HPI, PMH, SOB, CP, ROS, HR, RR, BP, SpO2, NKDA).
- Units: US conventional (mg/dL, lb or kg, °F or °C) except where the fact sheet gives a unit.
- Terse clinical style; sentence fragments are normal.

Fact sheet (each item is mandatory):
- [confusion] STATE: the patient is alert and fully oriented, not confused.
- [urea] STATE (number exactly): serum urea 8 mmol/L. Also mention a Serum urea (mmol/L) of 6.6 from a previous visit (clearly dated, e.g. '3 months ago'); today's value above is the current one.
- [resp_rate] DO NOT MENTION Respiratory rate (/min). Include nothing from which it could be inferred (no value, no 'normal'/'abnormal', no related medication, no blanket statements such as 'vitals stable' or 'labs unremarkable' that would cover it). No descriptors that imply a respiratory rate ('tachypneic', 'unlabored', 'breathing comfortably', 'no respiratory distress').
- [sbp] DO NOT MENTION Systolic blood pressure (mmHg). Include nothing from which it could be inferred (no value, no 'normal'/'abnormal', no related medication, no blanket statements such as 'vitals stable' or 'labs unremarkable' that would cover it). Do not give a blood pressure or describe it.
- [dbp] DO NOT MENTION Diastolic blood pressure (mmHg). Include nothing from which it could be inferred (no value, no 'normal'/'abnormal', no related medication, no blanket statements such as 'vitals stable' or 'labs unremarkable' that would cover it). Do not give a blood pressure or describe it.
- [age] DO NOT MENTION Age (years). Include nothing from which it could be inferred (no value, no 'normal'/'abnormal', no related medication, no blanket statements such as 'vitals stable' or 'labs unremarkable' that would cover it). Give no age, date of birth, or age descriptor (e.g. 'elderly', 'young').

Rules:
- Write numbers exactly as given in the fact sheet (same digits and decimals), with the unit given; you may use natural phrasing and abbreviations around them.
- Express each item in natural clinical language, as a clinician would chart it; do not copy the fact-sheet wording. Numbers, units and ordinal meanings must stay exactly as specified.
- Every STATE item must appear explicitly in the note, including negative ones (e.g. "no diabetes", "never smoked").
- Follow each item exactly. Do not add information about any fact-sheet item beyond what it says, and do not include anything that implies an item marked DO NOT MENTION.
- No blanket or complete statements that would cover a DO NOT MENTION item: no "Medications: none", "PMH: none/unremarkable", "ROS otherwise negative", "vitals stable" or "labs normal". If a list (medications, history) would normally reveal such an item, leave that list out.
- You may add other realistic details (presenting complaint, unrelated history, unrelated examination findings, investigations and plan) as long as they do not imply any fact-sheet item.
- Do not name any clinical score or rule (HEART score, CURB-65, CURB65, qSOFA, PERC, Wells, Cockcroft, creatinine clearance, CrCl), do not compute scores, and do not state a risk category or disposition that reveals one.
- Do not refer to these instructions, the fact sheet, or the note being synthetic.
- Output only the note, 150-350 words, plain text (simple section headings allowed).
```

### Judge (system)

```text
You audit synthetic clinical notes against a list of parameters. For each parameter, decide only from the note text how it is documented. Be strict: a parameter that is not stated but can reasonably be inferred (e.g. from a medication, a related finding, or a blanket statement such as 'vitals normal') is 'implied'. Output only JSON.
```

### Judge (user)

```text
Note:
<note>
<the rendered note>
</note>

Parameters:
- confusion: Confusion (AMT <= 8 or new disorientation to person/place/time) [yes/no finding]
- urea: Serum urea (mmol/L) [numeric (mmol/L)]
- resp_rate: Respiratory rate (/min) [numeric (/min)]
- sbp: Systolic blood pressure (mmHg) [numeric (mmHg)]
- dbp: Diastolic blood pressure (mmHg) [numeric (mmHg)]
- age: Age (years) [numeric (years)]

For every parameter, return one object:
{"param": "<id>", "status": "stated" | "negated_or_normal" | "not_mentioned" | "implied", "level": "<for 'one of' parameters: the level stated, else null>", "quote": "<shortest exact quote from the note supporting the status, or null>"}

- stated: the value or finding is explicitly documented (for yes/no findings: present).
- negated_or_normal: explicitly absent, denied, or documented as normal without a value.
- not_mentioned: nothing in the note documents or implies it.
- implied: not explicitly documented, but inferable from the note.

Return a JSON array with exactly one object per parameter, in the order listed, and nothing else.
```

### Extractor (system)

```text
You extract clinical parameters from a clinical note into a fixed JSON structure. You report only what the note documents; you never guess. Output only JSON matching the schema.
```

### Extractor (user)

```text
Note:
<note>
<the rendered note>
</note>

Parameters:
- confusion: Confusion (AMT <= 8 or new disorientation to person/place/time) [yes/no finding]
- urea: Serum urea (mmol/L) [numeric; unit tokens: mmol/L, urea_mg/dL, bun_mg/dL]
- resp_rate: Respiratory rate (/min) [numeric; unit tokens: /min]
- sbp: Systolic blood pressure (mmHg) [numeric; unit tokens: mmHg]
- dbp: Diastolic blood pressure (mmHg) [numeric; unit tokens: mmHg]
- age: Age (years) [numeric; unit tokens: years]

For every parameter give:
- "status":
  - "present": the note states the finding, or states a value/level for it.
  - "absent": the note explicitly says the finding is absent or denied, or that the parameter is normal (e.g. "no hemoptysis", "ECG normal", "vitals within normal limits").
  - "unknown": the note does not document it. Do not infer absence from silence: if the note does not mention it, the status is "unknown", never "absent".
- "value": for yes/no findings use null; for graded parameters one of the listed levels; for numeric parameters the number exactly as written in the note (do not convert units). Use null when status is not "present".
- "unit": for numeric parameters, the unit as written, mapped to one of the listed unit tokens; otherwise null.
- "evidence": the shortest exact quote from the note (copied character for character) that supports the status, or null when status is "unknown".
- "confidence": your probability (0 to 1) that status and value are correct.
If the note gives several values for the same parameter (e.g. an earlier reading or a previous visit), report the current one.
```

### S2 agent (system)

```text
You are a clinical decision-support agent working with an emergency clinician. You determine a patient's decision category for one clinical calculator. You can ask the treating clinician for any parameter, and you can run the calculator. Ask only what you need; answer when you are confident. Each turn, output exactly one JSON action and nothing else.
```

### S2 agent (first user turn)

```text
Calculator: CURB-65
Decision categories: low, moderate, high

Clinical note:
<note>
<the rendered note>
</note>

Calculator parameters (ids you can ask about or pass to calculate):
- confusion: Confusion (AMT <= 8 or new disorientation to person/place/time) [yes/no]
- urea: Serum urea (mmol/L) [number in mmol/L]
- resp_rate: Respiratory rate (/min) [number in /min]
- sbp: Systolic blood pressure (mmHg) [number in mmHg]
- dbp: Diastolic blood pressure (mmHg) [number in mmHg]
- age: Age (years) [number in years]

Actions (reply with one JSON object per turn):
1. {"action": "ask", "parameter": "<id>", "question": "<your question to the clinician>"}
   The clinician answers from the patient (or says the value is not available).
2. {"action": "calculate", "values": {"<id>": <value>, ...}}
   Runs the calculator. Give every parameter: yes/no as true/false, graded ones as the level name, numbers in the unit shown. Returns the score and category, or the missing parameters.
3. {"action": "answer", "category": "<one of: low, moderate, high, cannot_determine>"}
   Your final decision category. Use "cannot_determine" only if it truly cannot be determined.
```
