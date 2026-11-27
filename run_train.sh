#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/full_lora_config_custom.yaml}"
PYTHON="${PYTHON:-/home/opervu-user/miniconda3/envs/lang2segtrack/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="./logs/train_${STAMP}.log"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

echo "config : $CONFIG"
echo "device : 0"
echo "python : $PYTHON"
echo "log    : $LOG"
echo

CUDA_LAUNCH_BLOCKING=1 \
TORCH_USE_CUDA_DSA=1 \
PYTHONFAULTHANDLER=1 \
  "$PYTHON" train_sam3_lora_native.py --config "$CONFIG" --device 0 \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo
echo "exit=$status log=$LOG"
exit "$status"
