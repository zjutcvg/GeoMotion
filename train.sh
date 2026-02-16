#!/bin/bash
CUDA_VISIBLE_DEVICES=0 torchrun --master-port=29502 --nproc_per_node=1 train.py configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml