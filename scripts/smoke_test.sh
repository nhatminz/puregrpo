#!/usr/bin/env bash
# Explicit small REAL-model smoke; never substituted with synthetic results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export MAX_TRAIN_SAMPLES="${SMOKE_SAMPLES:-16}"
export GEN_MAX_LENGTH="${SMOKE_MAX_LENGTH:-512}"
export MAX_PROMPT_LENGTH="${SMOKE_MAX_PROMPT_LENGTH:-128}"
export NUM_EPOCHS=1
bash "$ROOT/train_${MODEL_KEY}.sh" --max_grpo_steps 1 "$@"
