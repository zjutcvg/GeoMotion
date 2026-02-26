#!/bin/bash
# set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "$0")" && pwd)}
CUDA_DEVICE=${CUDA_DEVICE:-0}
MODEL_NAME=${MODEL_NAME:-pi3_conf_low_35_feature_flow_gotm_verse_stop_all}
MODEL_PATH=${MODEL_PATH:-$ROOT_DIR/logs/${MODEL_NAME}/best_model.pth}
USE_SAM=${USE_SAM:-True}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-32}
DATASETS_CSV=${DATASETS_CSV:-2016-M,2017-M,2016,segtrack,fbms}

# Optional runtime paths (used when provided)
PI3_MODEL_PATH=${PI3_MODEL_PATH:-$ROOT_DIR/checkpoint/model.safetensors}
RAFT_MODEL_PATH=${RAFT_MODEL_PATH:-}
SAM2_CONFIG_PATH=${SAM2_CONFIG_PATH:-$ROOT_DIR/configs/sam2.1/sam2.1_hiera_l.yaml}
SAM2_CHECKPOINT_PATH=${SAM2_CHECKPOINT_PATH:-$ROOT_DIR/sam2-main/checkpoints/sam2.1_hiera_large.pt}

append_opt_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    EXTRA_ARGS+=("$flag" "$value")
  fi
}

resolve_dataset_roots() {
  local dataset="$1"
  case "$dataset" in
    2017-M|2016-M)
      IMG_ROOT="$ROOT_DIR/data/DAVIS2017-M/DAVIS/JPEGImages/480p"
      ANN_ROOT="$ROOT_DIR/data/DAVIS2017-M/DAVIS/Annotations/480p"
      ;;
    2016|2017|davis-all)
      IMG_ROOT="$ROOT_DIR/data/DAVIS/JPEGImages/480p"
      ANN_ROOT="$ROOT_DIR/data/DAVIS/Annotations/480p"
      ;;
    segtrack)
      IMG_ROOT="$ROOT_DIR/data/SegTrackv2/JPEGImages_jpg_standardized"
      ANN_ROOT="$ROOT_DIR/data/SegTrackv2/GroundTruth"
      ;;
    fbms)
      IMG_ROOT="$ROOT_DIR/data/FBMS59_clean/JPEGImages"
      ANN_ROOT="$ROOT_DIR/data/FBMS59_clean/Annotations"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo " Evaluating dataset: $DATASET"
  echo "=========================================="

  resolve_dataset_roots "$DATASET"
  OUTPUT_DIR="$ROOT_DIR/eval/${MODEL_NAME}_${DATASET}"

  EXTRA_ARGS=()
  append_opt_arg --pi3_model_path "$PI3_MODEL_PATH"
  append_opt_arg --raft_model_path "$RAFT_MODEL_PATH"
  append_opt_arg --sam2_config_path "$SAM2_CONFIG_PATH"
  append_opt_arg --sam2_checkpoint_path "$SAM2_CHECKPOINT_PATH"

  CMD=(
    python "$ROOT_DIR/eval.py"
    --model_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --image_root "$IMG_ROOT"
    --annotation_root "$ANN_ROOT"
    --sequence_length "$SEQUENCE_LENGTH"
    --use_sam_refine "$USE_SAM"
    --davis "$DATASET"
    "${EXTRA_ARGS[@]}"
  )

  echo "Running evaluation..."
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${CMD[@]}"

  echo "Finished: $DATASET"
  echo
done

echo "All evaluations completed!"
