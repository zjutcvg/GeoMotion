#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python motion_seg_inference.py \
    --model_path checkpoint/best_model.pth \
    --dataset_dir data/test \
    --output_dir output/test \
    --threshold 0.5