#!/bin/bash
# Run SAM3 LoRA validation with peak crops (needle_holder prompt).
# Always uses the "lang2segtrack" conda environment and logs full terminal output.
set -eo pipefail

# Always run from the project root regardless of caller's cwd.
cd /data/opervu/ws/sam3/SAM3_LoRA

CONDA_ENV="lang2segtrack"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"

CONFIG="configs/full_lora_config_custom.yaml"
WEIGHTS="/data/opervu/ws/sam3/SAM3_LoRA/outputs/sam3_lora_surgical_instruments/frozen_vision_encode_ep10_7k_20261118_1/best_lora_weights_merged_20261118.pt"
VAL_DIR="/data/dataset/sponge_needle/coco_incision_glove_removed_v2/valid"
#"/data/dataset/sam3_train_coco/valid"
#"/data/dataset/sponge_needle/coco/valid"
PLOTS_DIR="outputs/validation_plots"

TWO_STAGE_THRESHOLD="${TWO_STAGE_THRESHOLD:-0.25}"


CONDA_BIN=""
if [[ -x "$CONDA_BASE/bin/conda" ]]; then
  CONDA_BIN="$CONDA_BASE/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo "ERROR: conda not found (looked in $CONDA_BASE/bin/conda and PATH)" >&2
  exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda environment '$CONDA_ENV' not found" >&2
  exit 1
fi

ENV_PY="$CONDA_BASE/envs/$CONDA_ENV/bin/python3"
if [[ -x "$ENV_PY" ]]; then
  PY_CMD=("$ENV_PY")
else
  PY_CMD=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python3)
fi

mkdir -p "$PLOTS_DIR"
LOG_FILE="$PLOTS_DIR/validation_$(date +%Y%m%d_%H%M%S).log"

echo "==============================================================="
echo "SAM3 LoRA Validation"
echo "==============================================================="
echo "Env:     $CONDA_ENV (${PY_CMD[*]})"
echo "Python:  $("${PY_CMD[@]}" --version 2>&1)"
echo "Config:  $CONFIG"
echo "Weights: $WEIGHTS"
echo "Val dir: $VAL_DIR"
echo "Plots:   $PLOTS_DIR"
echo "Log:     $LOG_FILE"
echo ""

"${PY_CMD[@]}" validate_sam3_lora.py \
  --config "$CONFIG" \
  --weights "$WEIGHTS" \
  --val_data_dir "$VAL_DIR" \
  --plots-dir "$PLOTS_DIR" --no-overlays \
  # --peak-crops --peak-crop-prompt 'needle' --peak-crop-size 768 --peak-crop-topk 2 \
  # --two-stage --two-stage-threshold "$TWO_STAGE_THRESHOLD" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Validation complete. Plots in: $PLOTS_DIR/"
echo "Log saved to: $LOG_FILE"
