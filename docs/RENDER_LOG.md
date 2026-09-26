# Render log: main cohort, en-US (M3)

Record of how the `sonnet` render set of `configs/main.yaml` was produced, for the methods.

## Setup
- Cohort: 1,200 cases (200 per calculator), 50% undetermined from the note, seed 20260925.
- Renderer: Claude Sonnet 5, answering exported prompts as Claude Code subagents (the
  `session` provider). Each subagent saw only its batch of prompts, with no repository access.
- Judge: Claude Opus 5.5 via the same route. It is a different model from the renderer.
- Prompt version: v3 for all 1,200 notes. v1 and v2 were used in the 60-note pilots and the
  300-note stage 1 only.
- 1,200 cases map to 1,130 unique prompts. Cases with identical fact sheets (the same
  documented facts, differing only in undocumented hidden values) share one note by design.

## Results
| Calculator | Notes | Retried (attempt 2) | Failing after retries |
|---|---|---|---|
| Cockcroft-Gault | 200 | 0 | 0 |
| CURB-65 | 200 | 4 | 0 |
| HEART | 200 | 5 | 0 |
| PERC | 200 | 6 | 0 |
| qSOFA | 200 | 5 | 0 |
| Wells PE | 200 | 2 | 0 |
| **Total** | **1,200** | **22 (1.8%)** | **0** |

- Rule checks: 0 errors.
- Warnings: 24 documented items expressed only indirectly (medication trap, other unit,
  synonym) and 4 leaked-number matches that were coincidences.
- Note length: median 159 words (range 111–277).

## Process notes
- **Batch timing.** Some batches were imported before their subagent had finished, and the
  subagent later edited notes in them (15 + 10). Those notes were re-imported and re-judged.
  From then on, batches were imported only after the subagent reported completion.
- **Duplicate prompts.** The first pending export listed a few duplicate prompts. Pending
  exports are now de-duplicated by key.
- **Retry instructions.** Retry batches were given one extra sentence in the subagent
  instructions ("be especially careful that every STATE item appears explicitly and nothing
  implies a DO NOT MENTION item"). The prompt itself was unchanged.
- **Interrupted run.** A laptop sleep interrupted the full-render run. Completed notes were
  already cached, and the remaining 166 prompts were rendered in a fresh run.
- **Backups.** Raw subagent outputs are kept in `runs/main/session_backup/` (gitignored). The
  LLM cache (`.cache/llm/`) holds every request/response pair.
