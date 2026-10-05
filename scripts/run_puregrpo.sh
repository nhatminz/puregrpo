#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/launch/train_model.sh" "$@"
