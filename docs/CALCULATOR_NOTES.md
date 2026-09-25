# Calculator notes: sources and items for physician review

Every calculator cites its primary source in its module docstring. This file lists how each
source was verified and every `TODO(physician-review)` item. The code implements the reading
marked **Implemented**; change it after review.

Verification key: **[P]** primary full text read · **[A]** abstract only (publisher full text
not accessible) · **[S]** secondary source (NICE, MDCalc, NKF, later papers by the same group).

## Summary of review items

| # | Calculator | Item | Implemented | Alternatives |
|---|---|---|---|---|
| 1 | HEART | Troponin bands | levels ≤ normal / 1–3× / > 3× (level assigned upstream) | Six 2008: 1–2× / > 2×; Backus 2011, Poldervaart 2013: ≥ 3× → 2 |
| 2 | HEART | Age exactly 45 | 1 point (`age >= 45`) | Poldervaart 2013: ≤ 45 → 0 |
| 3 | HEART | ECG 2 points | "significant ST deviation" (depression or elevation) | 2008 table: ST depression only |
| 4 | HEART | Smoking recency | boolean "current or recent smoker" | < 1 month (Six 2008) vs ≤ 3 months (MDCalc) |
| 5 | HEART | Family history definition | boolean | 1st-degree relative with CVD < 65 (MDCalc) [U in primary] |
| 6 | HEART | Obesity | boolean, labelled BMI > 30 | not defined in Six 2008; BMI > 30 in Poldervaart 2013 |
| 7 | HEART | TIA as atherosclerotic disease | included in label | Six 2008 lists revascularisation, MI, stroke, PAD |
| 8 | CURB-65 | BUN equivalent | code converts BUN → urea mmol/L and applies > 7 (= BUN > 19.6) | MDCalc: BUN > 19 mg/dL |
| 9 | CURB-65 / qSOFA | Confusion vs altered mentation | two separate parameters | share one parameter |
| 10 | qSOFA | Altered mentation | GCS < 15 (Sepsis-3) | GCS ≤ 13 (Seymour derivation model) |
| 11 | PERC | SaO2 | positive if < 95% (2008) | 2004 abstract: rule needs > 94%; room-air requirement [U] |
| 12 | PERC | Surgery/trauma | within 4 weeks requiring hospitalisation (2008 abstract) | requiring general anaesthesia (some sources) |
| 13 | PERC | Hormone use | exogenous oestrogen | any hormone therapy (2004 abstract: "hormone use") |
| 14 | Wells | Immobilisation | "> 3 days" (NICE) | "≥ 3 days" (MDCalc) |
| 15 | Wells | PE-likely item | "alternative diagnosis less likely than PE" (NICE) | "PE #1 or equally likely" (MDCalc) |
| 16 | Cockcroft-Gault | Weight | actual body weight (original) | ideal / adjusted weight in obesity |
| 17 | Cockcroft-Gault | Decision thresholds | 30, 60 mL/min (config-overridable) | FDA 2024 Table 1 also has 90 |
| 18 | Units | Creatinine factor | 88.4 µmol/L per mg/dL | 88.42 (10,000 / 113.12) |
| 19 | Params | "Stated normal" intervals | see table below | — |
| 20 | Params | Creatinine can't be "normal" | Present or Unknown only | allow a normal interval |

## HEART

- Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the HEART score.
  *Neth Heart J* 2008;16(6):191–196. doi:10.1007/BF03086144. PMID 18665203. **[P]**
- Backus BE et al. *Crit Pathw Cardiol* 2010;9(3):164–169. PMID 20802272. **[A]**
- Backus BE et al. *Int J Cardiol* 2013;168(3):2153–2158. PMID 23465250. **[A]**
- Backus BE et al. *Curr Cardiol Rev* 2011;7(1):2–8 (PMC3131711); Poldervaart JM et al. *BMC
  Cardiovasc Disord* 2013;13:77 (PMC3849098). **[P]**

Implemented:
- History (clinician judgement), ECG and troponin are ordinal levels 0/1/2 = points.
- Age: < 45 → 0, 45–64 → 1, ≥ 65 → 2.
- Risk: 0 factors → 0, 1–2 → 1, ≥ 3 or atherosclerotic disease → 2. The factors are
  hypertension, hypercholesterolaemia, diabetes, obesity, smoking and family history of CAD.
- Categories: 0–3 low, 4–6 moderate, 7–10 high.

Notes:
- The Six 2008 table says "≤ 65 year" for 2 points; the text says "65 years or older". This is
  treated as a typo.
- Troponin is modelled as a level relative to the assay's normal limit, not a raw
  concentration. Mapping a raw value to a level needs the assay's 99th percentile. Rendering
  and extraction will state the level ("troponin 2× upper limit"), so review item 1 decides
  where "exactly 3×" falls when levels are assigned.
- Worked example (Six 2008): a 30-year-old man with nonspecific pain and normal ECG and
  troponin scored 0. This is a unit test. The 42-year-old woman who scored 1 has no
  per-component breakdown in the paper, so she isn't a test.

## CURB-65

- Lim WS et al. Defining community acquired pneumonia severity on presentation to hospital.
  *Thorax* 2003;58(5):377–382. doi:10.1136/thorax.58.5.377. PMID 12728155. **[P]**

Implemented:
- Confusion: Mental Test Score ≤ 8 or new disorientation.
- Urea > 7 mmol/L.
- RR ≥ 30.
- SBP < 90 or DBP ≤ 60 (counts once).
- Age ≥ 65.
- Groups: 0–1 low, 2 moderate, 3–5 high (Figure 2 management wording is in the docstring).

Units: urea is stored in mmol/L. BUN mg/dL is divided by 2.8014, and urea mg/dL by 6.006.
The extractor must name the analyte (`bun_mg/dL` vs `urea_mg/dL`); a bare "mg/dL" is rejected.

## qSOFA

- Seymour CW et al. *JAMA* 2016;315(8):762–774. PMID 26903335. **[P]**
- Singer M et al. (Sepsis-3). *JAMA* 2016;315(8):801–810. PMID 26903338. **[P]**

Implemented: RR ≥ 22, SBP ≤ 100 and altered mentation (GCS < 15) score 1 point each.
Positive if ≥ 2.

Notes:
- Sepsis-3 scores mentation as "abnormal", not "changed from baseline", so a patient with
  baseline dementia always has 1 point.
- The Surviving Sepsis Campaign 2021 recommends against qSOFA as a single screening tool.
  That doesn't affect this experiment.

## PERC

- Kline JA et al. *J Thromb Haemost* 2004;2(8):1247–1255. PMID 15304025. **[A]**
- Kline JA et al. *J Thromb Haemost* 2008;6(5):772–780. PMID 18318689. **[A]**

Implemented:
- Score = number of criteria not met: age ≥ 50, HR ≥ 100, SaO2 < 95%, unilateral leg
  swelling, haemoptysis, recent surgery/trauma, prior VTE, hormone use.
- 0 → negative, ≥ 1 → positive.
- PERC applies only when gestalt pretest probability is low (< 15%). The calculator doesn't
  model that gate; the cohort generator should only create PERC cases in that setting.

## Wells criteria for PE (two-tier)

- Wells PS et al. *Thromb Haemost* 2000;83(3):416–420. PMID 10744147. **[A]**
- van Belle A et al. (Christopher Study). *JAMA* 2006;295(2):172–179. PMID 16403929. **[A]**
- NICE NG158, Table 2 (adapted from Wells 2000). **[S]**

Implemented:
- Points: DVT signs 3, PE-most-likely 3, HR > 100 1.5, immobilisation/surgery 1.5, prior
  DVT/PE 1.5, haemoptysis 1, malignancy 1.
- ≤ 4 → PE unlikely, > 4 → PE likely.
- Scores move in 0.5 steps, so the category boundary is at 4.5.

## Cockcroft-Gault

- Cockcroft DW, Gault MH. Prediction of creatinine clearance from serum creatinine. *Nephron*
  1976;16(1):31–41. doi:10.1159/000180580. PMID 1244564. **[A]**
- Formula confirmed in Cockcroft's 1992 Citation Classic commentary. **[S]**
- FDA Guidance for Industry, *Pharmacokinetics in Patients with Impaired Renal Function*,
  March 2024, Table 1. **[P]**

Implemented:
- CrCl = (140 − age) × weight / (72 × SCr mg/dL), × 0.85 if female.
- Weight is actual body weight.
- Categories come from configurable thresholds (default 30, 60), named like
  `crcl_30_to_lt_60`.
- Lower bounds are inclusive, e.g. exactly 30 → `crcl_30_to_lt_60`.

Notes:
- The formula was derived in 249 men; the 0.85 female factor was estimated, not derived.
- It assumes steady-state creatinine.
- The 2024 FDA table has no 15–29 / < 15 split; that came from earlier drafts.

## Unit conversions (`src/calc_bounds/units.py`)

| Conversion | Factor | Basis |
|---|---|---|
| Creatinine mg/dL → µmol/L | × 88.4 | MW 113.12 g/mol (10,000 / 113.12 = 88.40) |
| Urea mmol/L → urea mg/dL | × 6.006 | MW 60.06 g/mol |
| Urea mmol/L → BUN mg/dL | × 2.8014 | 2 N per urea, 14.007 g/mol each |
| lb → kg | × 0.45359237 | exact by definition (NIST SP 811) |

## Parameter modelling choices (`src/calc_bounds/calculators/params.py`)

These are modelling choices, not clinical criteria, but they affect bounds.

"Stated normal" (`absent_means`) is the interval a numeric parameter is assumed to lie in when
the note says it is normal. Each interval is chosen not to straddle any threshold on that
parameter; a test enforces this.

| Param | Plausible range | "Stated normal" |
|---|---|---|
| heart_rate | 20–250 /min | 60–99 |
| resp_rate | 4–60 /min | 12–20 |
| sbp | 50–250 mmHg | 101–139 |
| dbp | 20–150 mmHg | 61–89 |
| spo2 | 50–100 % | 95–100 |
| urea | 1–60 mmol/L | 2.5–7.0 |
| age | 18–110 years | never "normal" |
| weight | 30–250 kg | never "normal" |
| creatinine | 0.2–15 mg/dL | never "normal" (item 20) |

These parameters can only be Present or Unknown (they are judgements or demographics): age,
sex, weight, creatinine and HEART history.
