# MedCalc-Bench anchor

A sanity check of our extraction + code pipeline against a published benchmark.

## Source and provenance
- **Dataset:** MedCalc-Bench Verified, test split (`test_data.csv`, 1,100 rows), from
  https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified. It's the successor that
  the original `ncbi-nlp/MedCalc-Bench` repository points to.
- **Downloaded:** 2026-09-26 to `data/raw/medcalc/`, which is gitignored and never committed.
- **SHA-256:** `9d296b09668d945d7c4ad8136032e984a3a3b8b0a7b046eb0f9f787331d9d97d`.
- **Training split** (`train_data.csv`), downloaded 2026-09-26, SHA-256
  `bd0292576be31e2fa8140c2e9eb456168335a85d1986155f010082d64f497845`.
  - It has 947 notes for our five calculators, all extracted from real case reports.
  - 3 notes that also appear in the test split are excluded.
  - We use a seeded sample of at most 125 per calculator (585 notes), via
    `calc-bounds anchor --split train --per-calc 125`.
- **License:** CC-BY-SA 4.0. The notes are largely drawn from PMC-Patients (CC-BY-SA 4.0).
  We don't redistribute them.
- **Label caveat:** no fully physician-adjudicated corrected labels exist.
  - Verified gives no correction method or changelog.
  - Ye et al. (arXiv:2512.19691) relabeled v1.0 by LLM consensus, with physicians reviewing 50
    contentious cases.
- **Exception to the synthetic-only rule.** Approved by the user on 2026-09-26. The notes are
  de-identified published case reports, not clinical records, and are used only for this
  anchor. No other real patient text enters the repository.

## Overlap with our calculators
100 test notes: 20 each for HEART, CURB-65, PERC, Wells PE and Cockcroft-Gault. qSOFA is not
in MedCalc-Bench. 95 of the notes are extracted case reports and 5 are synthetic.

## What `calc-bounds anchor` reports, per note
1. **`impl_agrees`.** Does our calculator, applied to MedCalc's own annotated entities (using
   their convention that a missing entity is normal), land inside their answer range? This
   separates implementation differences from extraction errors.
2. **`medcalc_convention_*`.** Our extraction scored under their convention (Unknown treated
   as normal), compared with their answer range and category. This is the published metric.
3. **`tristate_*`.** Our actual pipeline:
   - Is the category determined from the note?
   - If so, is it correct?
   - Is the true category still among the possible ones?
4. **`entity_agreement`.** Our extracted values vs their annotated entities.

## Implementation agreement (our code on their entities)
| Calculator | Agreement | Why |
|---|---|---|
| Wells PE | 20/20 | |
| CURB-65 | 20/20 | |
| PERC | 20/20 | |
| HEART | 19/20 | One patient has a TIA history. MedCalc counts TIA as an ordinary risk factor; we count it as atherosclerotic disease (2 points), as MDCalc does (review item 7). Krohn-Grimberghe (arXiv:2603.02222) also flags MedCalc's HEART atherosclerosis logic. |
| Cockcroft-Gault | 8/20 | By design. MedCalc picks the weight by BMI: actual if < 18.5, min(ideal, actual) at 18.5–24.9, adjusted above 24.9. We use actual weight (review item 16). |
