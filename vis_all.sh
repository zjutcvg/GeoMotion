#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CUDA_DEVICE=${CUDA_DEVICE:-0}
MODEL_PATH=${MODEL_PATH:-$ROOT_DIR/checkpoint/best_model.pth}
PI3_MODEL_PATH=${PI3_MODEL_PATH:-$ROOT_DIR/checkpoint/model.safetensors}
RAFT_MODEL_PATH=${RAFT_MODEL_PATH:-}
SAM2_CONFIG_PATH=${SAM2_CONFIG_PATH:-$ROOT_DIR/configs/sam2.1/sam2.1_hiera_l.yaml}
SAM2_CHECKPOINT_PATH=${SAM2_CHECKPOINT_PATH:-$ROOT_DIR/sam2-main/checkpoints/sam2.1_hiera_large.pt}
DATASET_DIR=${DATASET_DIR:-$ROOT_DIR/data/test}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/output/test}
THRESHOLD=${THRESHOLD:-0.5}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-32}

append_opt_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    EXTRA_ARGS+=("$flag" "$value")
  fi
}

EXTRA_ARGS=()
append_opt_arg --pi3_model_path "$PI3_MODEL_PATH"
append_opt_arg --raft_model_path "$RAFT_MODEL_PATH"
append_opt_arg --sam2_config_path "$SAM2_CONFIG_PATH"
append_opt_arg --sam2_checkpoint_path "$SAM2_CHECKPOINT_PATH"

CMD=(
  python "$ROOT_DIR/motion_seg_inference.py"
  --model_path "$MODEL_PATH"
  --dataset_dir "$DATASET_DIR"
  --output_dir "$OUTPUT_DIR"
  --sequence_length "$SEQUENCE_LENGTH"
  --threshold "$THRESHOLD"
  "${EXTRA_ARGS[@]}"
)

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${CMD[@]}"
