#!/usr/bin/env bash
set -euo pipefail
# Target-only GRPO; common target settings inherited from SpecNaacl.
MODEL_KEY="qwen25_3b"
METHOD="puregrpo"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-3B-Instruct}"
DATASET="${DATASET:-dapo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TARGET_LR="${TARGET_LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-${REPEATED_GENERATE_NUMS:-8}}"
GEN_MAX_LENGTH="${GEN_MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAMPLE_NUM="${SAMPLE_NUM:-100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"
