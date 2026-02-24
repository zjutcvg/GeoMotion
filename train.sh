#!/bin/bash
set -euo pipefail

CUDA_DEVICE=${CUDA_DEVICE:-0}
MASTER_PORT=${MASTER_PORT:-29502}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CONFIG_PATH=${CONFIG_PATH:-configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml}

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" torchrun \
  --master-port="$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  train.py "$CONFIG_PATH"
