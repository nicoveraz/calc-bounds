# Results (English, M6)

The question: does separating calibrated tri-state extraction (a model) from
decision-relevance computation (code, using score bounds) cut the number of questions asked
without losing decision-category accuracy, compared with an end-to-end LLM agent?

Everything here comes from `calc-bounds report configs/main.yaml` (`runs/main/report/`, not
committed). The Spanish (es-CL) arm is paused.

## Setup
- **Cohort:** 1,200 synthetic cases, 200 each for HEART, CURB-65, qSOFA, PERC, Wells PE and
  Cockcroft-Gault.
  - 30% parameter missingness.
  - 50% of cases are undetermined from the note; natural-share reweighting is shown below.
  - Traps: negation, mixed units, prior encounters, contradictory values, and comorbidity
    implied only by a medication.
- **Notes:** written by Sonnet 5 and validated by an Opus 5.5 judge plus rule checks. Every
  note passes validation.
- **Extraction sources:**
  - oracle (reads the ground truth);
  - Haiku 4.5 via headless Claude Code (self-reported confidence);
  - Qwen 3.5 9B running locally (logprob confidence).
- **Clinician:** a deterministic simulator. Troponin is "not available" 10% of the time.
- **S2:** Opus 5.5 as a JSON-action agent (ask / calculate / answer), reading the same notes.
- **Primary outcome:** decision-category accuracy, with abstentions counted as wrong, plus
  questions asked per case. Paired tests compare systems on the same cases: exact McNemar
  for accuracy, Wilcoxon signed-rank for question counts.
- **Cost:** $0. Claude models run on the user's subscription; Qwen runs on the laptop.

## Main results (n = 1,200)
| Extraction | System | Accuracy | Questions/case | Irrelevant questions | Premature commitment | Silent missing→absent / case |
|---|---|---|---|---|---|---|
| Oracle | S1 ask-all | 99.8% | 1.74 | 47% | 0% | 0 |
| Oracle | S3 bounds | 99.8% | 0.92 | 0% | 0% | 0 |
| Oracle | S4 bounds+VOI+echo | 99.8% | 0.87 | 0% | 0% | 0 |
| Oracle | S3-bin | 91.7% | 0.21 | 0% | 39% | 1.42 |
| Haiku | S1 | 99.4% | 1.78 | 48% | 0% | 0.022 |
| Haiku | S3 | 99.4% | 0.92 | 0% | 0% | 0.022 |
| Haiku | S4 | 99.7% | 0.90 | 0% | 0% | 0.022 |
| Haiku | S3-bin | 91.2% | 0.21 | 0% | 40% | 1.43 |
| Qwen 9B | S1 | 99.8% | 2.21 | 44% | 0% | 0.006 |
| Qwen 9B | S3 | 99.8% | 1.23 | 0% | 0% | 0.006 |
| Qwen 9B | S4 | 99.8% | 1.20 | 0% | 0% | 0.006 |
| Qwen 9B | S3-bin | 91.2% | 0.26 | 0% | 49% | 1.42 |
| (reads the note) | **S2 Opus agent** | 99.6% | 0.99 | 9.7% | 0.2% | — |

## Hypotheses
**H1: supported.** The bounds policy (S3) asks 44–48% fewer questions than ask-all with the
same accuracy, for every extraction source.

| Extraction | S3 vs S1 questions/case | Accuracy disagreements | McNemar p | Wilcoxon p |
|---|---|---|---|---|
| Oracle | 0.92 vs 1.74 | 0 | 1.0 | < 0.001 |
| Haiku | 0.92 vs 1.78 | 0 | 1.0 | < 0.001 |
| Qwen 9B | 1.23 vs 2.21 | 0 | 1.0 | < 0.001 |

- About half of S1's questions cannot change the decision. S3 asks none of those.
- **Missingness sensitivity** (oracle, fresh cohorts, reweighted to the natural share of
  undetermined cases):

  | Missingness | S3 questions/case | S1 questions/case | Reduction |
  |---|---|---|---|
  | 10% | 0.33 | 0.64 | 49% |
  | 30% | 1.14 | 1.93 | 41% |
  | 50% | 2.16 | 3.29 | 34% |

  Accuracy is unchanged at every level.

**H2: partly supported.** The Opus agent is strong: 99.6% accuracy with 0.99 questions per
case.
- **It does over-ask.** 9.7% of its questions cannot change the decision, versus 0% for S3.
  It asks significantly more questions than S3 (Wilcoxon p < 0.001).
- **It rarely under-asks.** It committed while the category was still undetermined in 2 of
  1,200 cases. One of those guesses was wrong.
- **Its accuracy doesn't differ significantly from S3's** (for example 3 vs 5 discordant
  cases with Haiku extraction; McNemar p = 0.73).
- **Bottom line:** on these clean synthetic notes, a frontier agent is almost as efficient
  as the bounds policy. The advantages of S3 are:
  - no irrelevant questions;
  - somewhat fewer questions;
  - determinism and auditability;
  - it works with a small local extractor, whereas S2 needs a frontier model at every turn,
    about 2.5 turns per case.

  "The end-to-end agent fails badly" is **not** supported here.

**H3: strongly supported.**
- **Binary schema vs tri-state:**

  | | S3-bin (missing = absent) | S3 tri-state |
  |---|---|---|
  | Accuracy | 91.2–91.7% | 99.4–99.8% |
  | Silent missing→absent errors per case | about 1.4 | 0 to 0.02 |
  | Commits while undetermined | 39–49% of cases | 0% |

  McNemar p < 0.001 for every extraction source (99–105 vs 2 discordant cases).
- **Attribution.** Most S3-bin errors are routing errors: it looks determined only because
  missing items were assumed normal.
- **The same failure shows up in the extractors themselves.** Haiku occasionally turns an
  undocumented item into "absent" (0.022 per case), and that accounts for most of its small
  accuracy loss under S3.

**H4: supported, with small numbers.** The confidence echo in S4 catches extraction errors at
a small question cost. Echo threshold sweep (`s4_echo_threshold_sweep.csv`):

| Extraction | Echo threshold | Accuracy | Echo questions/case | Errors caught |
|---|---|---|---|---|
| Haiku | none (S3) | 99.42% | 0 | 0 |
| Haiku | 0.9 | 99.67% | 0.015 | 9 |
| Haiku | 0.99 | 99.83% (oracle level) | 0.82 | 19 |
| Qwen 9B | 0.9 | 99.83% | 0.02 | 2 |

- **Accuracy gain is not significant.** S4 vs S3 differs on only 3 cases with Haiku
  (McNemar p = 0.25), because Haiku makes few errors to begin with.
- **Fewer questions overall.** Even with echo questions included, S4 asks fewer questions
  than S3 thanks to VOI ordering (Wilcoxon p < 0.01).

## Harder conditions: noisy clinician, messy notes (added 2026-09-27)
**Noisy clinician.** Per question, the clinician:
- doesn't know 10% of the time;
- answers wrongly 5% of the time (a flipped yes/no, a graded item one level off, or a number
  off by 10–25%);
- answers a number as a ±10% range 25% of the time.

**Messy notes.** 600 cases (100 per calculator) re-rendered from the same fact sheets in an
end-of-shift style: fragments, abbreviations, typos, copied-forward text, no headings. Of
these, 41 still failed validation after 3 attempts and are excluded, leaving 559.

**Conservative fallback.** An undecided case gets the highest-risk category still possible
(for S2, which doesn't track bounds: the calculator's highest-risk category).
- **Under-triage:** the decision is lower-risk than the truth. This is the dangerous error.
- **Over-triage:** the decision is higher-risk than the truth.

Key rows below. The full grid is in `runs/main/report/summary_all.csv`. Note sets differ in
size (1,200 clean vs 559 messy), so compare within a row group.

| Condition | System | Accuracy | Correct when answering | Under-triage | Over-triage | Premature | Questions/case |
|---|---|---|---|---|---|---|---|
| Clean, ideal (Haiku) | S3 | 99.4% | 99.5% | 0.1% | 0.5% | 0% | 0.92 |
| | S2 Opus | 99.6% | 99.7% | 0.1% | 0.3% | 0.2% | 0.99 |
| | S3-bin | 91.2% | 91.2% | **8.5%** | 0.3% | 40% | 0.21 |
| Clean, noisy (Haiku) | S3 | 87.0% | 97.6% | 0.3% | 9.8% | 0% | 0.96 |
| | S4 | 87.2% | 97.8% | 0.3% | 9.6% | 0% | 0.94 |
| | S2 Opus | 83.5% | 97.2% | 0.5% | 13.8% | 2.7% | 1.03 |
| | S3-bin | 88.0% | 90.9% | **8.4%** | 2.8% | 39% | 0.21 |
| Messy, ideal (Haiku) | S3 | 98.6% | 98.7% | 0.5% | 0.9% | 0% | 1.04 |
| | S4 | 99.1% | 99.3% | 0.5% | 0.4% | 0% | 1.04 |
| | S2 Opus | 99.8% | 100% | 0.0% | 0.2% | 0.7% | 0.92 |
| | S3-bin | 90.7% | 90.7% | **8.8%** | 0.5% | 39% | 0.30 |
| Messy, ideal (Qwen 9B) | S3 | 99.5% | 99.6% | 0.2% | 0.4% | 0% | 1.17 |
| Messy, noisy (Haiku) | S3 | 85.5% | 96.8% | 0.5% | 11.1% | 0% | 1.08 |
| | S2 Opus | 85.0% | 97.3% | 1.1% | 12.0% | 3.4% | 0.97 |
| | S3-bin | 86.8% | 90.3% | **8.6%** | 3.8% | 38% | 0.31 |

**Findings**
1. **Assuming missing means normal under-triages about 1 in 12 patients in every
   condition** (8.2–10.2%), including a perfect clinician and clean notes. Headline
   accuracy hides this: under noise S3-bin's accuracy (86–88%) matches S3's because it
   rarely asks, but when it answers it is wrong about 10% of the time and commits
   prematurely in about 40% of cases.
2. **The bounds policies essentially never under-triage** (0–0.5%, and only after the
   clinician gives a wrong answer). When information is truly unavailable they fall back to
   the higher-risk category (about 10–13% over-triage under noise), which is the safe
   direction.
3. **The noisy clinician costs every system about 12–15 accuracy points, mostly through
   abstention.** A "don't know" to the one decisive question leaves the category open.
   When they do commit, S3, S4 and S2 are 97–98% correct.
4. **Messy notes hurt Haiku extraction a little and Qwen barely.**
   - Haiku under S3 drops from 99.4% to 98.6%, and its silent missing→absent rate doubles to
     0.045 per case. S4's confidence echo recovers part of this (99.1%).
   - Qwen stays at 99.5%.
5. **The Opus agent reading the raw notes is a strong baseline.**
   - On messy notes with the ideal clinician it beats S3 with Haiku extraction (99.8% vs
     98.6%, McNemar p = 0.016) and ties S3 with Qwen extraction.
   - Under noise it ties S3 on messy notes and falls behind on clean notes (83.5% vs 87.0%,
     p < 0.001).
   - It always has more premature commitment (0.2–3.4% vs 0%) and more irrelevant questions
     (about 10% vs 0%).
   - **Revised H2:** a frontier agent is competitive on accuracy. The bounds approach wins on
     safety properties: it never commits prematurely, never asks an irrelevant question,
     and has the lowest under-triage. It also wins on cost and determinism, and it works
     with a 9B local model.

## Real clinical notes (MedCalc-Bench training split, 584 case reports; see ANCHOR.md)
| Calculator | Our code reproduces their label | Note alone settles the category | Category correct if missing = normal |
|---|---|---|---|
| PERC | 100% | 88–90% | 91–94% |
| Cockcroft-Gault | 51% (weight-rule difference) | 85–88% | 71–72% |
| CURB-65 | 100% | 41–49% | 81–82% |
| Wells | 100% | 10–19% | 78–81% |
| HEART | 100% | 12–13% | 9–22% |
| All | 90% | 47–52% | 65–69% |

- **Real notes usually don't contain enough to settle the score.** For HEART and Wells that
  holds for about 85% of notes. Assuming missing = normal gets the category right for only
  65–69% of notes overall, and 9–22% for HEART.
- **The bounds keep the true category possible for 93–95% of notes.** The misses come from
  extraction errors, mainly HEART entities (entity agreement 73–74%).

## Extraction and calibration
- **Claim accuracy:** 99.6% for Haiku and 99.5% for Qwen.
- **Qwen's misses are safe.** It reads only 83% of documented negatives, but a miss becomes
  Unknown and gets asked, never silently absent.
- **Calibration:**

  | Extractor | ECE raw | ECE after isotonic calibration (dev-split fit) |
  |---|---|---|
  | Haiku | 0.014 | 0.002 |
  | Qwen 9B | 0.048 | 0.002 |

  See `reliability.png`.

## Error attribution (wrong final categories)
- **S3 and S4:** errors are either abstentions (troponin unavailable while it could still
  change the category) or extraction errors (Haiku: 6 under S3, 3 under S4). There are no
  routing errors and no bounds errors.
- **S2:** 2 abstentions, 2 wrong answers given while the case was already determined
  (misreads), and 1 premature commitment.
- **S3-bin:** 74–99 routing errors, plus extraction errors.
- **Bounds errors:** none anywhere, consistent with the property tests.

## External anchor (MedCalc-Bench Verified, 100 real case reports; see ANCHOR.md)
- **Our code matches MedCalc's labels** on 99 of 100 non-Cockcroft-Gault notes when fed
  their annotated entities. The one exception is the TIA definition in HEART.
- **The bounds are sound on real notes.** The true category stays possible 97–98% of the
  time.
- **Real notes are often incomplete.** Only about half are determined from the note alone
  (59% with Haiku, 51% with Qwen). MedCalc's convention fills those gaps with "normal".
- **Under MedCalc's own convention:** 62% exact answers and 84% category accuracy (Haiku).

## Renderer-family bias (exploratory)
- **Data:** a 2×2 on the 46 cases whose notes passed validation under both renderers.
- **Result:** a small same-family advantage of +3.2 points (95% CI +0.4 to +6.4).
- **Caveats:** small n, and a subset selected by validation (Qwen 9B wrote faithful notes
  for only 31% of cases). Treat this as a limitation, not a finding.

## Limitations
- **Noise model.** The clinician noise is a simple parametric model, not calibrated to
  real clinicians.
- **Messy notes are a subset.** They cover 559 cases rather than the full cohort, and the
  41 notes that failed validation are excluded.
- **Clean notes.** The notes are synthetic and validated to be faithful, so extraction is
  easier than on real notes. The anchor shows real notes are far less complete.
- **Near-ceiling accuracy.** Accuracy differences are therefore small and many comparisons
  are underpowered. Question counts carry most of the signal.
- **Idealized clinician.** The simulated clinician answers exactly, never wrong and never
  vague.
- **Claude calls are not bare API calls.** They run through Claude Code headless (`claude -p`,
  with tools off and our own system prompt). Confidence for Claude models is self-reported.
- **Unreviewed choices.** Several calculator definitions, population priors and "stated
  normal" intervals await physician review (`docs/CALCULATOR_NOTES.md`).
- **Spanish (es-CL) is not yet evaluated.** 113 notes are rendered and cached; the style
  guide awaits physician review.
