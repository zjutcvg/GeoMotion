#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=1 python motion_seg_inference.py \
    --model_path checkpoint/best_model.pth \
    --dataset_dir /data1/GOT-10k_Train_split_14 \
    --output_dir /data1/output/GOT-10k_Train_split_14 \
    --threshold 0.1
