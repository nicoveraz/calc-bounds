# Results

**The question:** when an AI computes a clinical risk score from a note, can it ask the
clinician fewer questions without getting more patients wrong? We compare having the AI only
read the note, with ordinary code deciding what is still unknown and whether it matters,
against the usual alternatives.

All numbers come from `calc-bounds report configs/main.yaml` (`runs/main/report/`, not
committed) and the paper tables (`paper/tables/`). The Spanish (es-CL) arm is paused.

## Summary in plain language
- **Assuming "not mentioned = normal" puts about 1 in 12 patients in a lower-risk group than
  they belong to.** This happened in every condition we tested (8–10% of patients). Our
  approach did it in 0–0.5%.
- **Our approach asks about half as many questions as asking for everything missing, with
  the same accuracy.** Every question it asks could change the decision. About half the
  questions of the ask-everything approach could not.
- **It never gives an answer before the answer is settled.**
- **A leading AI agent working alone (Claude Opus 5.5) is a strong competitor.** It is as
  accurate when the clinician always answers correctly. But about 1 in 10 of its questions
  are unnecessary, it sometimes answers too early (up to 3.4% of patients), and it is less
  accurate when the clinician's answers are imperfect.
- **A small AI model that runs on a laptop reads notes well enough.** Patient text never
  has to leave the computer.
- **Real notes are often incomplete.** In 584 published case reports, only about half had
  enough information to settle the risk group; for HEART, 13%.

## Words used in this document
| Term | Meaning |
|---|---|
| **Risk group** (decision category) | The group a score puts the patient in, e.g. HEART low / moderate / high risk. Accuracy means getting this group right. |
| **Ask-all** (S1) | Ask the clinician about every item the note does not mention. |
| **Our approach** (S3, "bounds") | The AI marks each item *present*, *absent* or *not mentioned*. Code works out the lowest and highest score still possible. If both give the same risk group, it answers; otherwise it asks only about items that could change the group. |
| **Our approach + checks** (S4) | As S3, but asks the most useful question first, and asks the clinician to confirm items the AI was unsure about. |
| **Missing = normal** (S3-bin) | Same as S3, except anything not mentioned is assumed normal. This is the common convention in tools and benchmarks. |
| **AI agent** (S2) | Claude Opus 5.5 reads the note, decides itself what to ask, and calls the calculator. |
| **Reader** (extraction source) | Who reads the note: *perfect reader* (reads the true answers, a best-case baseline), *Haiku* (Claude Haiku 4.5, a small commercial model) or *Qwen 9B* (a small model running on a laptop). |
| **Under-triage** | The patient is put in a *lower*-risk group than the truth. The dangerous error. |
| **Over-triage** | The patient is put in a *higher*-risk group than the truth. Safer, but costly. |
| **Answers too early** (premature commitment) | Gives a risk group while missing items could still change it. It may be right by luck. |
| **Unnecessary question** (irrelevant question) | A question whose answer could not change the risk group. |
| **Abstains** | Cannot give a risk group, e.g. a decisive test result is not available. Counted as wrong. |

## How the test was set up
- **Patients:** 1,200 synthetic emergency patients, 200 each for HEART, CURB-65, qSOFA,
  PERC, Wells (pulmonary embolism) and Cockcroft-Gault (kidney function).
  - Each item had a 30% chance of not being mentioned in the note.
  - For half the patients, the note alone is not enough to settle the risk group.
  - Notes contain deliberate traps: negations ("denies chest pain"), mixed units, findings
    from a previous visit, contradictory values, and diseases implied only by a medication
    (e.g. metformin but no "diabetes").
- **Notes:** written by Claude Sonnet 5 from each patient's true data. A second model
  (Claude Opus 5.5) and rule checks confirmed that every note matches the data.
- **Clinician:** simulated by code. It answers each question from the true data. A troponin
  result is "not available" 10% of the time.
- **Main measure:** the percentage of patients put in the right risk group (giving no answer
  counts as wrong), and the number of questions per patient.
- **Statistics:** approaches are compared on the same patients (McNemar test for accuracy,
  Wilcoxon test for question counts).
- **Cost:** $0. Claude models ran on a subscription; Qwen ran on the laptop.

## Main results (1,200 patients, perfect clinician, clean notes)
| Reader | Approach | Right risk group | Questions per patient | Unnecessary questions | Answered too early | Items silently assumed normal, per patient |
|---|---|---|---|---|---|---|
| Perfect | Ask-all | 99.8% | 1.74 | 47% | 0% | 0 |
| Perfect | Our approach | 99.8% | 0.92 | 0% | 0% | 0 |
| Perfect | Our approach + checks | 99.8% | 0.87 | 0% | 0% | 0 |
| Perfect | Missing = normal | 91.8% | 0.21 | 0% | 39% | 1.42 |
| Haiku | Ask-all | 99.4% | 1.78 | 48% | 0% | 0.022 |
| Haiku | Our approach | 99.4% | 0.92 | 0% | 0% | 0.022 |
| Haiku | Our approach + checks | 99.7% | 0.90 | 0% | 0% | 0.022 |
| Haiku | Missing = normal | 91.2% | 0.21 | 0% | 40% | 1.43 |
| Qwen 9B | Ask-all | 99.8% | 2.21 | 44% | 0% | 0.006 |
| Qwen 9B | Our approach | 99.8% | 1.23 | 0% | 0% | 0.006 |
| Qwen 9B | Our approach + checks | 99.8% | 1.20 | 0% | 0% | 0.006 |
| Qwen 9B | Missing = normal | 91.2% | 0.26 | 0% | 49% | 1.42 |
| (reads the note itself) | **AI agent (Opus)** | 99.6% | 0.99 | 9.5% | 0.2% | — |

## What we expected, and what we found

### 1. Fewer questions, same accuracy: confirmed
Our approach asked 44–48% fewer questions than ask-all, with the same accuracy, whichever
model read the note.

| Reader | Questions per patient (ours vs ask-all) | Patients where accuracy differed | Accuracy difference p | Question difference p |
|---|---|---|---|---|
| Perfect | 0.92 vs 1.74 | 0 | 1.0 | < 0.001 |
| Haiku | 0.92 vs 1.78 | 0 | 1.0 | < 0.001 |
| Qwen 9B | 1.23 vs 2.21 | 0 | 1.0 | < 0.001 |

- About half of ask-all's questions could not change the decision. Our approach asked none
  of those.
- **The saving holds when notes are more or less complete.** We generated new patients with
  10%, 30% and 50% of items not mentioned, adjusted to a realistic share of unsettled cases:

  | Items not mentioned | Our approach, questions | Ask-all, questions | Fewer questions |
  |---|---|---|---|
  | 10% | 0.33 | 0.64 | 49% |
  | 30% | 1.14 | 1.93 | 41% |
  | 50% | 2.16 | 3.29 | 34% |

  Accuracy was the same at every level.

### 2. Better than an AI agent working alone: partly confirmed
The Opus agent is strong: 99.6% accuracy with 0.99 questions per patient.
- **It asks some unnecessary questions.** 9.5% of its questions could not change the
  decision (our approach: 0%). It asked significantly more questions overall (p < 0.001).
- **It rarely answers too early.** It gave a risk group before it was settled in 2 of 1,200
  patients; one of those guesses was wrong.
- **Its accuracy was not significantly different** (e.g. 3 vs 5 patients where only one
  approach was right, with Haiku reading; p = 0.73).
- **So, with clean notes and a perfect clinician, a leading AI agent is almost as
  efficient.** Our approach still:
  - asks no unnecessary questions, and somewhat fewer questions overall;
  - gives the same answer every time and shows why (the code is auditable);
  - works with a small laptop model, whereas the agent needs a large model at every step
    (about 2.5 steps per patient).

  "The AI agent fails badly" is **not** supported. Harder conditions (below) show where it
  falls behind.

### 3. "Missing = normal" is unsafe: strongly confirmed
| | Missing = normal | Our approach |
|---|---|---|
| Right risk group | 91.2–91.8% | 99.4–99.8% |
| Items silently assumed normal, per patient | about 1.4 | 0 to 0.02 |
| Answers before the risk group is settled | 39–49% of patients | 0% |

- The difference is highly significant for every reader (p < 0.001; 99–105 patients where
  only our approach was right, vs 2 the other way).
- **Why it goes wrong:** most errors are patients who look settled only because missing
  items were assumed normal.
- **The readers themselves sometimes do this.** Haiku occasionally marked an item that was
  not mentioned as "absent" (0.022 per patient). This explains most of its small accuracy
  loss with our approach.

### 4. Confirming uncertain readings helps a little: confirmed, small numbers
In S4, the clinician is asked to confirm items the reader was unsure about. The higher the
threshold, the more is confirmed:

| Reader | Confirm when confidence is below | Right risk group | Confirmation questions per patient | Reading errors caught |
|---|---|---|---|---|
| Haiku | never (our approach) | 99.42% | 0 | 0 |
| Haiku | 0.9 | 99.67% | 0.015 | 9 |
| Haiku | 0.99 | 99.83% (perfect-reader level) | 0.82 | 19 |
| Qwen 9B | 0.9 | 99.83% | 0.02 | 2 |

- **The accuracy gain is not significant** (3 patients differ with Haiku; p = 0.25), because
  Haiku makes few errors to begin with.
- **It still asks fewer questions overall,** because it asks the most useful question
  first (p < 0.01).

## Harder conditions: imperfect clinician, messy notes
**Imperfect ("noisy") clinician.** For each question, the clinician:
- does not know, 10% of the time;
- answers wrongly, 5% of the time (yes/no flipped, a graded item one level off, or a number
  off by 10–25%);
- gives a range instead of a number (±10%), 25% of the time.

**Messy notes.** 600 patients (100 per calculator) had their notes rewritten in an
end-of-shift style: fragments, abbreviations, typos, copied-forward text, no headings. 41
notes still did not match the data after 3 attempts and were dropped, leaving 559.

**When the risk group cannot be settled,** each approach falls back to the highest-risk
group still possible. (The AI agent, which does not track this, falls back to the
calculator's highest-risk group.) This lets us count under- and over-triage.

The table shows key rows; the full grid is in `runs/main/report/summary_all.csv`. Clean
notes cover 1,200 patients and messy notes 559, so compare within each group of rows.

| Condition | Approach | Right risk group | Right when it answered | Under-triage | Over-triage | Answered too early | Questions per patient |
|---|---|---|---|---|---|---|---|
| Clean notes, perfect clinician (Haiku) | Ours | 99.4% | 99.5% | 0.1% | 0.5% | 0% | 0.92 |
| | AI agent | 99.6% | 99.7% | 0.1% | 0.3% | 0.2% | 0.99 |
| | Missing = normal | 91.2% | 91.2% | **8.5%** | 0.3% | 40% | 0.21 |
| Clean notes, noisy clinician (Haiku) | Ours | 87.0% | 97.6% | 0.3% | 9.8% | 0% | 0.96 |
| | Ours + checks | 87.2% | 97.8% | 0.3% | 9.6% | 0% | 0.94 |
| | AI agent | 83.5% | 97.2% | 0.5% | 13.8% | 2.7% | 1.03 |
| | Missing = normal | 88.0% | 90.9% | **8.4%** | 2.8% | 39% | 0.21 |
| Messy notes, perfect clinician (Haiku) | Ours | 98.6% | 98.7% | 0.5% | 0.9% | 0% | 1.04 |
| | Ours + checks | 99.1% | 99.3% | 0.5% | 0.4% | 0% | 1.04 |
| | AI agent | 99.8% | 100% | 0.0% | 0.2% | 0.7% | 0.92 |
| | Missing = normal | 90.7% | 90.7% | **8.8%** | 0.5% | 39% | 0.30 |
| Messy notes, perfect clinician (Qwen 9B) | Ours | 99.5% | 99.6% | 0.2% | 0.4% | 0% | 1.17 |
| Messy notes, noisy clinician (Haiku) | Ours | 85.5% | 96.8% | 0.5% | 11.1% | 0% | 1.08 |
| | AI agent | 85.0% | 97.3% | 1.1% | 12.0% | 3.4% | 0.97 |
| | Missing = normal | 86.8% | 90.3% | **8.6%** | 3.8% | 38% | 0.31 |

**What this shows**
1. **"Missing = normal" under-rates about 1 in 12 patients in every condition** (8.2–10.2%),
   even with clean notes and a perfect clinician. Its overall accuracy can look fine: with a
   noisy clinician it scores 86–88%, like our approach, because it rarely asks and so is
   rarely misled. But when it answers, it is wrong about 1 time in 10, and it answers too
   early in about 40% of patients.
2. **Our approach almost never under-rates risk** (0–0.5%, and only after the clinician gave
   a wrong answer). When information is truly unavailable, it falls back to the higher-risk
   group (10–13% over-triage with a noisy clinician). That is the safe direction.
3. **An imperfect clinician costs every approach 12–15 accuracy points, mostly because no
   answer can be given.** If the clinician does not know the one decisive item, the risk
   group stays open. When they do answer, our approach and the AI agent are right 97–98% of
   the time.
4. **Messy notes hurt Haiku a little and Qwen hardly at all.**
   - Haiku with our approach drops from 99.4% to 98.6%, and how often it marks an item not
     mentioned as "absent" doubles (0.045 per patient). Confirming uncertain readings
     recovers part of this (99.1%).
   - Qwen stays at 99.5%.
5. **The AI agent is a strong competitor.**
   - With messy notes and a perfect clinician it beat our approach with Haiku reading (99.8%
     vs 98.6%, p = 0.016) and tied it with Qwen reading.
   - With a noisy clinician it tied on messy notes and fell behind on clean notes (83.5% vs
     87.0%, p < 0.001).
   - It always answered too early more often (0.2–3.4% vs 0%) and asked more unnecessary
     questions (about 10% vs 0%).
   - **Overall:** a leading AI agent is competitive on accuracy. Our approach is safer: it
     never answers too early, never asks an unnecessary question, and under-rates risk
     least. It is also cheaper, gives the same answer every time, and works with a small
     laptop model.

## Real clinical notes (584 published case reports, MedCalc-Bench; see ANCHOR.md)
The range shows the two readers (Haiku, Qwen 9B).

| Calculator | Our code gives the benchmark's answer | Note alone settles the risk group | Right risk group if missing = normal |
|---|---|---|---|
| PERC | 100% | 88–90% | 91–94% |
| Cockcroft-Gault | 51% (the benchmark uses a different weight rule) | 85–88% | 71–72% |
| CURB-65 | 100% | 41–49% | 81–82% |
| Wells | 100% | 10–19% | 78–81% |
| HEART | 100% | 12–13% | 9–22% |
| All | 90% | 47–52% | 65–69% |

- **Real notes usually do not contain enough to settle the score.** For HEART and Wells,
  about 85% of notes do not. Assuming missing = normal gives the right risk group for only
  65–69% of notes overall, and 9–22% for HEART.
- **Our approach keeps the true risk group among the possibilities in 93–95% of notes.** The
  misses come from reading errors, mainly HEART items (the readers agreed with the
  benchmark's item labels 73–74% of the time).

## How well the models read notes, and how well they know when they are unsure
- **Items read correctly:** 99.6% for Haiku and 99.5% for Qwen.
- **Qwen's mistakes are safe.** It picks up only 83% of documented negatives ("no chest
  pain"), but a missed item becomes "not mentioned" and is asked about, never silently
  assumed normal.
- **Confidence matches reality after adjustment.** Calibration error (0 is perfect):

  | Reader | Raw | After adjustment on a separate 30% of patients |
  |---|---|---|
  | Haiku | 0.014 | 0.002 |
  | Qwen 9B | 0.048 | 0.002 |

  See `reliability.png`.

## Why patients were misclassified
- **Our approach (with or without checks):** either no answer could be given (troponin
  unavailable while it could still change the risk group), or the reader misread the note
  (Haiku: 6 patients without checks, 3 with checks). The code's logic caused no errors.
- **AI agent:** 2 with no answer, 2 wrong answers after the risk group was already settled
  (misreading), and 1 answer given too early.
- **Missing = normal:** 76–99 patients who looked settled only because missing items were
  assumed normal, plus reading errors.
- **The score-range logic itself never caused an error,** consistent with the automated
  property tests.

## Check against real, annotated case reports (MedCalc-Bench Verified, 100 notes; see ANCHOR.md)
- **Our calculators agree with the benchmark** on 99 of 100 notes (excluding
  Cockcroft-Gault) when given the benchmark's own item labels. The one exception is how
  HEART counts a TIA.
- **The score-range logic holds on real notes.** The true risk group stays among the
  possibilities 97–98% of the time.
- **Real notes are often incomplete.** Only about half settle the risk group from the note
  alone (59% with Haiku, 51% with Qwen). The benchmark fills those gaps with "normal".
- **Under the benchmark's own rules**, Haiku gave the exact score in 62% and the right risk
  group in 84%.

## Does a model read its own family's notes better? (exploratory)
- **Data:** 46 patients whose notes passed checks when written by both Claude Sonnet and
  Qwen, each read by both families.
- **Result:** a small advantage when the same family wrote and read the note, +3.2 points
  (95% CI +0.4 to +6.4).
- **Caveats:** few patients, and a selected subset (Qwen wrote faithful notes for only 31%
  of patients). Treat this as a limitation, not a finding.

## Limitations
- **Synthetic patients and notes.** The notes were checked to match the data, so they are
  easier to read than real notes; the real case reports show real notes are far less
  complete.
- **Simulated clinician.** The main results use a clinician who always answers exactly. The
  noisy clinician is a simple model, not measured from real clinicians.
- **Messy notes are a subset:** 559 patients rather than all 1,200, with 41 notes that
  failed checks excluded.
- **Accuracy is close to 100% in ideal conditions,** so accuracy differences are small and
  many comparisons lack statistical power. Question counts carry most of the signal.
- **Claude models were used through Claude Code** (`claude -p`, tools off, our own system
  prompt), not the bare API. Their confidence is self-reported.
- **One physician reviewed the calculator definitions,** population assumptions and
  "stated normal" ranges: the author (`docs/CALCULATOR_NOTES.md`).
- **Spanish (es-CL) is not yet evaluated.** 113 notes are written and cached; the style
  guide awaits physician review.
