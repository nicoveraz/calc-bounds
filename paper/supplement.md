---
title: "Supplementary material — Unknown is not normal"
---

## S1. Calculator readings and clinical review

Each calculator cites its primary source in code. Where the source was ambiguous or differed from common implementations, the implemented reading and the alternatives are listed below. The author, an emergency physician, reviewed and accepted the implemented readings.

{{section:../docs/CALCULATOR_NOTES.md|## Summary of review items}}

## S2. Synthetic cohort and population priors

Cohort (n = 1,200). Undetermined: category not determined by the documented facts (50% by design).

{{table:table1_cohort}}


Plausible emergency-department populations for each calculator's intended use; parameters sampled independently except diastolic < systolic blood pressure − 15 mmHg. The PERC cohort represents low gestalt pre-test probability. Reviewed by the author.

{{table:s2_priors}}

## S3. Missingness sensitivity (oracle extraction, natural share of undetermined cases)

Fresh cohorts (200 cases per calculator) at each missingness level; stratum results reweighted to the natural share of undetermined cases.

{{table:s3_missingness}}

## S4. Confirmation threshold sweep (S4 policy, clean notes, ideal clinician)

S4 asks the clinician to confirm a decision-critical extracted value when its calibrated confidence is below the threshold.

{{table:s4_echo_sweep}}

## S5. Calibration of extraction confidence

Present/absent claims; isotonic and temperature calibration fitted on a seeded 30% development split, evaluated on the rest. Haiku 4.5 (self-reported): ECE 0.014 raw, 0.003 temperature, 0.002 isotonic. Qwen3.5-9B (token log-probabilities): ECE 0.048 raw, 0.033 temperature, 0.002 isotonic.

![Reliability of raw extraction confidence (test split).](figures/figS5_reliability.pdf)

## S6. Full results grids

Clean notes, ideal clinician, all 1,200 cases, every extractor:

{{table:table2_compact}}

559 paired cases with validated notes in both styles; both extractors, both note styles, both clinicians:

{{table:table3_conditions}}

## S7. Paired comparisons (exact McNemar for accuracy, Wilcoxon signed-rank for questions; Holm-adjusted within condition)

Full cohort (n = 1,200):

{{table:table4_comparisons_full}}

Paired subset (n = 559):

{{table:table5_comparisons_paired559}}

## S8. Real case reports, all splits and extractors

{{table:table6_real_notes}}

## S9. Exact prompts

Prompt versions: renderer v3, extractor x1, agent a1. The example fact sheet is for one CURB-65 case with a trap; `<the rendered note>` stands for the note text.

{{file:tables/s7_prompts.md}}

## S10. Sensitivity to implausible synthetic cases

Characteristics were sampled independently, so some cases are clinically implausible. We flagged established atherosclerotic disease with no risk factors, and age under 40 with three or more risk factors or atherosclerotic disease. We then recomputed the main results without those cases (clean notes, ideal clinician).

{{file:tables/s10_flagged_cases.md}}

{{table:s10_implausible_sensitivity}}
