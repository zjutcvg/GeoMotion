#!/bin/bash
# =====================================
# Unified Evaluation Script for GeoMotion
# =====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$SCRIPT_DIR}"
DATA_DIR="${DATA_DIR:-$CODE_DIR}"
MODEL_NAME="${MODEL_NAME:-full_geomotion}"
MODEL_PATH="${MODEL_PATH:-$CODE_DIR/checkpoint/Full_GeoMotion.pth}"


DATASETS=("2016-M" "2017-M" "got-test" "segtrack" "fbms")
USE_SAM="${USE_SAM:-True}"
SAM2_CONFIG_PATH="${SAM2_CONFIG_PATH:-$CODE_DIR/sam2-main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml}"
SAM2_CHECKPOINT_PATH="${SAM2_CHECKPOINT_PATH:-$DATA_DIR/sam2-main/checkpoints/sam2.1_hiera_large.pt}"

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo " Evaluating dataset: $DATASET"
    echo "=========================================="


    if [[ "$DATASET" == "2017-M" || "$DATASET" == "2016-M" ]]; then
        IMG_ROOT="$DATA_DIR/data/DAVIS2017-M/DAVIS/JPEGImages/480p"
        ANN_ROOT="$DATA_DIR/data/DAVIS2017-M/DAVIS/Annotations/480p"

    elif [[ "$DATASET" == "2016" || "$DATASET" == "2017" || "$DATASET" == "davis-all" ]]; then
        IMG_ROOT="$DATA_DIR/data/DAVIS/JPEGImages/480p"
        ANN_ROOT="$DATA_DIR/data/DAVIS/Annotations/480p"

    elif [[ "$DATASET" == "segtrack" ]]; then
        IMG_ROOT="$DATA_DIR/data/SegTrackv2/JPEGImages_jpg_standardized"
        ANN_ROOT="$DATA_DIR/data/SegTrackv2/GroundTruth"

    elif [[ "$DATASET" == "fbms" ]]; then
        IMG_ROOT="$DATA_DIR/data/FBMS59_clean/JPEGImages"
        ANN_ROOT="$DATA_DIR/data/FBMS59_clean/Annotations"
    elif [[ "$DATASET" == "got-test" ]]; then
        IMG_ROOT="$DATA_DIR/data/GOT-test/JPEGImages/480p"
        ANN_ROOT="$DATA_DIR/data/GOT-test/Annotations/480p"
    else
        echo "❌ Unknown dataset: $DATASET"
        exit 1
    fi

    OUTPUT_DIR="$CODE_DIR/eval/${MODEL_NAME}_${DATASET}_adapter_raft"

    echo "➡️  Running evaluation..."
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$CODE_DIR/eval_adapter_raft.py" \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --image_root "$IMG_ROOT" \
        --annotation_root "$ANN_ROOT" \
        --sequence_length 32 \
        --use_sam_refine "$USE_SAM" \
        --davis "$DATASET" \
        --sam2_config_path "$SAM2_CONFIG_PATH" \
        --sam2_checkpoint_path "$SAM2_CHECKPOINT_PATH"

    echo "✅ Finished: $DATASET"
    echo ""
done

echo "🎯 All evaluations completed!"
