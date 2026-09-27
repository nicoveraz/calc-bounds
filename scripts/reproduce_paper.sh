#!/bin/zsh
# Regenerate every Paper 1 number, table and figure.
#
# From the LLM cache (.cache/llm) this makes no model calls. Without the cache, the same
# commands re-run the models through the providers in configs/main.yaml (Claude via
# `claude -p` on a Claude subscription; Qwen3.5-9B via local Ollama) - see README.
set -euo pipefail
cd "$(dirname "$0")/.."
cb() { uv run calc-bounds "$@"; }
C=configs/main.yaml

cb cohort $C                                   # 1,200 synthetic cases (seeded)
cb render $C --name sonnet                     # clean notes (cache)
cb render $C --name messy                      # messy notes (cache)
for x in haiku qwen_local; do
  for r in sonnet messy; do
    cb extract $C --extractor $x --render $r
    cb calibrate $C --extractor $x --render $r
    cb run $C --extractor $x --render $r
    cb run $C --extractor $x --render $r --clinician noisy
  done
  cb anchor $C --extractor $x                                # MedCalc-Bench test (needs data/raw)
  cb anchor $C --extractor $x --split train --per-calc 125   # MedCalc-Bench train sample
done
cb run $C                                      # oracle extraction, incl. S2 agent
cb run $C --clinician noisy
cb report $C                                   # runs/main/report/
cb paper $C                                    # paper/tables, paper/figures
