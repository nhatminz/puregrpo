#!/usr/bin/env bash
set -euo pipefail
: "${EVAL_DATASET_PATH:?Set EVAL_DATASET_PATH to a real held-out split}"
EVAL_ONLY=true
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/launch/train_model.sh" --eval_only "$@"
