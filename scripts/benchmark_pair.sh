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
export OPD_DIAGNOSTICS=0 OPD_PROFILE=0 DRAFT_TRAIN_PROFILE=0 STATISTICAL_TIME=False
BENCHMARK_STEPS="${BENCHMARK_STEPS:-2}"
checker_args=(--training-arg="--max_grpo_steps=$BENCHMARK_STEPS")
for arg in "$@"; do checker_args+=(--training-arg="$arg"); done
"$PYTHON_BIN" "$ROOT/scripts/check_fair_config.py" --model-key "$MODEL_KEY" --source "$SOURCE_SPECNAACL_ROOT" \
  --use-environment "${checker_args[@]}"
if [[ "${DRY_RUN:-false}" != true ]]; then
  : "${TARGET_ADAPTER:?Provide the SAME target LoRA initialization for all three methods; see huongdanchay.md}"
  "$PYTHON_BIN" -m pip check
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
    'import sys; from helper.shared_adapter import adapter_artifact; print("Shared adapter:", adapter_artifact(sys.argv[1]))' "$TARGET_ADAPTER"
fi
for method in fastgrpo opd_reflex puregrpo; do
  run="$BENCHMARK_ROOT/$method"
  project="$ROOT"
  [[ "$method" == puregrpo ]] || project="$SOURCE_SPECNAACL_ROOT"
  launcher="train_${MODEL_KEY}.sh"
  [[ "$method" != fastgrpo ]] || launcher="train_${MODEL_KEY}_fastgrpo.sh"
  printf 'Comparison: %s → %s\n' "$method" "$run"
  env RUN_DIR="$run" RUN_NAME="paired_$method" PYTHON_BIN="$PYTHON_BIN" \
    bash "$project/$launcher" --max_grpo_steps "$BENCHMARK_STEPS" "$@"
done
if [[ "${DRY_RUN:-false}" != true ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/plot_training_time.py" "$BENCHMARK_ROOT/fastgrpo" "$BENCHMARK_ROOT/opd_reflex" "$BENCHMARK_ROOT/puregrpo" \
    --output "$BENCHMARK_ROOT/training_time.png"
fi
