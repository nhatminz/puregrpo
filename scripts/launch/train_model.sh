#!/usr/bin/env bash
# Target-only launcher: resource paths and common defaults inherited from SpecNaacl.
set -euo pipefail
: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Preserve SpecNaacl's configured output/checkpoint parent as requested. Runs
# and resume/latest links have method-specific names and never overwrite its links.
SOURCE_SPECNAACL_ROOT="${SOURCE_SPECNAACL_ROOT:-$PROJECT_DIR/../SpecNaacl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SOURCE_SPECNAACL_ROOT/outputs}"
case "${DATASET,,}" in
  gsm8k) TRAIN_OPTION=gsm8k; DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}" ;;
  simplelr|simplelr_abel|simplelr_abel_level3to5)
    TRAIN_OPTION=simplelr_abel_level3to5; DATASET=simplelr
    DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
  dapo|dapo-math|dapo_math)
    TRAIN_OPTION=DAPO-math; DATASET=dapo
    DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
  *) : "${DATASET_PATH:?Set DATASET_PATH for a custom dataset}"; TRAIN_OPTION="$DATASET" ;;
esac
TRAIN_MODEL_ROOT="${TRAIN_MODEL_ROOT:-$OUTPUT_ROOT/train/$MODEL_KEY}"
REQUESTED_RUN_DIR="${RUN_DIR:-}"
timestamp="$(date -u +%Y%m%dT%H%M%S)"
short_uuid="$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${DATASET}__method-puregrpo__seed${TRAIN_SUBSET_SEED}__${timestamp}__${short_uuid}}"
RUN_DIR="${RUN_DIR:-$TRAIN_MODEL_ROOT/$RUN_NAME}"
if [[ "$RESUME" == auto && -z "$REQUESTED_RUN_DIR" && -e "$TRAIN_MODEL_ROOT/active_run_puregrpo" ]]; then
  active_run="$(readlink -f "$TRAIN_MODEL_ROOT/active_run_puregrpo")"
  if [[ -f "$active_run/checkpoints/resume/latest.pt" && ! -f "$active_run/summary.json" ]]; then
    RUN_DIR="$active_run"; RUN_NAME="$(basename "$RUN_DIR")"
  fi
fi
RESUME_CHECKPOINT=""
if [[ "$RESUME" == auto ]]; then
  [[ ! -f "$RUN_DIR/checkpoints/resume/latest.pt" ]] || RESUME_CHECKPOINT="$RUN_DIR/checkpoints/resume/latest.pt"
elif [[ -n "$RESUME" ]]; then RESUME_CHECKPOINT="$RESUME"; fi
cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE"
  "$PROJECT_DIR/grpo.py" --method puregrpo --model_dir "$MODEL" --model_type "$MODEL_TYPE"
  --dtype "$MODEL_DTYPE" --attn_implementation "$ATTENTION_IMPLEMENTATION" --load_lora_path "$TARGET_ADAPTER"
  --train_option "$TRAIN_OPTION" --dataset_path "$DATASET_PATH" --train_split "${TRAIN_SPLIT:-train}"
  --train_data_fraction "$TRAIN_DATA_FRACTION" --train_subset_seed "$TRAIN_SUBSET_SEED" --max_train_samples "$MAX_TRAIN_SAMPLES"
  --version_name "$RUN_NAME" --batch_size "$BATCH_SIZE" --num_epochs "$NUM_EPOCHS" --sample_num "$SAMPLE_NUM"
  --accumulation_steps "$ACCUMULATION_STEPS" --target_lr "$TARGET_LR"
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K"
  --max_length "$GEN_MAX_LENGTH" --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_training_padding_gap "$MAX_TRAINING_PADDING_GAP" --max_training_token "$MAX_TRAINING_TOKEN"
  --logps_chunk_size "$LOGPS_CHUNK_SIZE" --grpo_iteration_num "$GRPO_ITERATION_NUM"
  --repeated_generate_nums "$RESPONSES_PER_PROMPT" --beta "$BETA" --epsilon "$EPSILON"
  --statistical_time "$STATISTICAL_TIME" --num_workers "$NUM_WORKERS" --persistent_workers "$PERSISTENT_WORKERS"
  --eval_interval "$EVAL_INTERVAL" --eval_dataset_path "${EVAL_DATASET_PATH:-}" --eval_split "${EVAL_SPLIT:-test}"
  --log_interval "$LOG_INTERVAL" --seed "$TRAIN_SUBSET_SEED"
  --log_file "$RUN_DIR/logs/metrics.jsonl" --timing_file "$RUN_DIR/logs/timing.csv"
  --summary_file "$RUN_DIR/summary.json" --saved_model_dir "$RUN_DIR/checkpoints/target"
  --saved_statistics_dir "$RUN_DIR/statistics" --checkpoint_dir "$RUN_DIR/checkpoints/resume"
  --resume_checkpoint "$RESUME_CHECKPOINT" --save_checkpoint_steps "$SAVE_CHECKPOINT_STEPS" --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS")
cmd+=(--max_target_optimizer_steps "${MAX_TARGET_OPTIMIZER_STEPS:-0}" --max_rollout_prompts "${MAX_ROLLOUT_PROMPTS:-0}")
cmd+=("$@")
printf 'Run name : %s\nRun dir  : %s\nModel    : %s\nDataset  : %s\nMethod   : puregrpo\nGPUs     : %s\n' \
  "$RUN_NAME" "$RUN_DIR" "$MODEL" "$DATASET_PATH" "$NPROC_PER_NODE"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "${DRY_RUN:-false}" == true ]]; then return 0 2>/dev/null || exit 0; fi
for path in "$MODEL/config.json" "$DATASET_PATH"; do
  [[ -e "$path" ]] || { echo "ERROR: required path not found: $path" >&2; exit 2; }
done
[[ -z "$RESUME_CHECKPOINT" || -f "$RESUME_CHECKPOINT" ]] || { echo 'ERROR: missing resume checkpoint' >&2; exit 2; }
[[ -z "$TARGET_ADAPTER" || -d "$TARGET_ADAPTER" ]] || { echo 'ERROR: missing target adapter' >&2; exit 2; }
if [[ -f "$RUN_DIR/summary.json" && -z "$RESUME_CHECKPOINT" && "${EVAL_ONLY:-false}" != true ]]; then
  echo 'ERROR: completed run exists; choose a new RUN_DIR or explicit RESUME' >&2; exit 2
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --requirements "$PROJECT_DIR/requirements.txt" --require-cuda
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/statistics" "$RUN_DIR/checkpoints" "$TRAIN_MODEL_ROOT"
if [[ "${EVAL_ONLY:-false}" != true ]]; then
  ln -sfn "$RUN_DIR" "$TRAIN_MODEL_ROOT/active_run_puregrpo"
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/write_run_metadata.py" --run-dir "$RUN_DIR" --kind train \
  --item "run_name=$RUN_NAME" --item 'method=puregrpo' --item "model=$MODEL" --item "dataset=$DATASET_PATH" \
  --item "target_adapter=$TARGET_ADAPTER" --item "target_lr=$TARGET_LR" --item "batch_size=$BATCH_SIZE" \
  --item "accumulation_steps=$ACCUMULATION_STEPS" --item "responses_per_prompt=$RESPONSES_PER_PROMPT" \
  --item "temperature=$TEMPERATURE" --item "top_p=$TOP_P" --item "seed=$TRAIN_SUBSET_SEED"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"${cmd[@]}" 2>&1 | tee -a "$RUN_DIR/logs/console.log"
if [[ "${EVAL_ONLY:-false}" != true ]]; then
  ln -sfn "$RUN_DIR" "$TRAIN_MODEL_ROOT/latest_run_puregrpo"
fi
