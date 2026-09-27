> **Superseded by `RESULTS.md`** (kept for provenance of the M4 milestone).

# M4 results: LLM extractors, calibration, anchor

All runs cost $0 (see the memory note on the zero-budget plan):
- **Haiku 4.5** via headless Claude Code (`claude -p`, subscription). Confidence is
  self-reported.
- **Qwen 3.5 9B** (Q4_K_M) on a laptop through native Ollama. Confidence comes from token
  logprobs.
- The cohort is the 1,200 Sonnet-rendered en-US notes (v3). Policies use the deterministic
  simulated clinician.

## Policies with LLM extraction (full cohort, n = 1,200)
| Extraction | Policy | Accuracy | Abstain | Questions/case | Irrelevant-question rate | Silent missing-as-absent/case |
|---|---|---|---|---|---|---|
| Oracle | S1 | 99.8% | 0.2% | 1.74 | 47% | 0 |
| Oracle | S3 | 99.8% | 0.2% | 0.92 | 0% | 0 |
| Haiku | S1 | 99.4% | 0.1% | 1.78 | 48% | 0.022 |
| Haiku | S3 | 99.4% | 0.1% | 0.92 | 0% | 0.022 |
| Qwen 9B | S1 | 99.8% | 0.2% | 2.21 | 44% | 0.006 |
| Qwen 9B | S3 | 99.8% | 0.2% | 1.23 | 0% | 0.006 |

- **Different failure modes.** Qwen misses documented items (17% of documented negatives
  read as Unknown), and S3 recovers them by asking. Haiku reads more completely, but it
  occasionally marks an undocumented item as absent (16 claims). That is the silent error H3
  targets, and it causes most of Haiku's 0.4-point accuracy loss.
- **H1 holds with real extractors.** S3 asks about 45% fewer questions than S1 at identical
  accuracy, and none of its questions are irrelevant.

## Calibration (test split, Present/Absent claims)
| Extractor | Confidence | Claim accuracy | ECE raw | ECE temperature | ECE isotonic |
|---|---|---|---|---|---|
| Haiku | self-reported | 99.6% | 0.014 | 0.003 | 0.002 |
| Qwen 9B | logprob | 99.5% | 0.048 | 0.033 | 0.002 |

## MedCalc-Bench anchor (100 real case reports; see ANCHOR.md)
| | Haiku | Qwen 9B |
|---|---|---|
| Exact answer (MedCalc convention: missing counts as normal) | 62% | 56% |
| Category (MedCalc convention) | 84% | 79% |
| Category determined by the three-state bounds | 59% | 51% |
| Determined and correct | 56% | 49% |
| True category still possible within the bounds | 97% | 98% |
| Entity agreement with MedCalc's annotations | 94% | 91% |

- **Our code agrees with MedCalc's on 99 of the 100 non-Cockcroft-Gault notes** when fed
  their own entities. The one exception is TIA-as-atherosclerosis in HEART (review item 7).
  Cockcroft-Gault differs by design because of MedCalc's BMI-dependent weight rule.
- **The bounds are sound on real notes.** The true category is excluded only 2–3% of the
  time.
- **Half the real notes are undetermined.** That is where S3 would ask. MedCalc's convention
  instead fills those gaps with "normal".

## Renderer-family bias (exploratory)
- **Cases:** 46 cases whose notes passed validation under both renderers (Sonnet 5 and
  Qwen 9B).
- **Selection:** Qwen passed validation for only 46 of 150 notes, mostly failing through
  leaks, so the subset is unbalanced. It holds 15 CURB-65, 11 qSOFA, 8 HEART, 5 PERC,
  5 Cockcroft-Gault and 2 Wells cases.

Claim accuracy by extractor and note author:

| Extractor ↓ / notes by → | Sonnet | Qwen |
|---|---|---|
| Haiku | 100.0% | 98.9% |
| Qwen 9B | 86.7% | 88.8% |

- **Same-family interaction:** +3.2 points (95% case-bootstrap CI +0.4 to +6.4).
- **Interpretation:** a small same-family advantage. It is exploratory: small n, and a
  subset selected by validation. Report it as a limitation and sensitivity analysis, not a
  main result.

## Process notes
- **Local render.** Qwen 9B rendered the 150-note bias subset with up to 3 attempts. 104
  notes still failed validation and are excluded. Notes were judged by Opus 5.5 via
  `claude -p`.
- **Structured-output failures.** Some `claude -p` calls failed structured output on long
  real notes. The client now retries, and on persistent failure returns an uncached error:
  the extraction becomes all-Unknown with a logged rejection. 64 extra anchor calls were
  logged because of retries.
