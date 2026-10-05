#!/usr/bin/env bash
# Explicit bounded training comparison; NOT automatically run during installation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_SPECNAACL_ROOT="${SOURCE_SPECNAACL_ROOT:-$ROOT/../SpecNaacl}"
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SOURCE_SPECNAACL_ROOT/outputs/benchmarks/pure_vs_spec_${MODEL_KEY}_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ ! -e "$BENCHMARK_ROOT" ]] || { echo 'ERROR: benchmark directory already exists' >&2; exit 2; }
export MAX_TRAIN_SAMPLES="${BENCHMARK_SAMPLES:-128}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}" RESUME='' EVAL_INTERVAL=0
export REFLEX_DIAGNOSTICS=0 REFLEX_PROFILE=0 DRAFT_TRAIN_PROFILE=0 STATISTICAL_TIME=False
"$PYTHON_BIN" "$ROOT/scripts/check_fair_config.py" --model-key "$MODEL_KEY" --source "$SOURCE_SPECNAACL_ROOT"
if [[ "${DRY_RUN:-false}" != true ]]; then
  : "${TARGET_ADAPTER:?Provide the SAME target LoRA initialization for both methods; see huongdanchay.md}"
fi
for method in specnaacl puregrpo; do
  run="$BENCHMARK_ROOT/$method"
  project="$ROOT"
  [[ "$method" != specnaacl ]] || project="$SOURCE_SPECNAACL_ROOT"
  printf 'Comparison: %s → %s\n' "$method" "$run"
  env RUN_DIR="$run" RUN_NAME="paired_$method" PYTHON_BIN="$PYTHON_BIN" \
    bash "$project/train_${MODEL_KEY}.sh" "$@"
done
if [[ "${DRY_RUN:-false}" != true ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/plot_training_time.py" "$BENCHMARK_ROOT/specnaacl" "$BENCHMARK_ROOT/puregrpo" \
    --output "$BENCHMARK_ROOT/training_time.png"
fi
