# GeoMotion

<div align="center">
<h1>GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry</h1>

Xiankang He<sup>1,2</sup>, Peile Lin<sup>1,2</sup>, Ying Cui<sup>1,2</sup>, Dongyan Guo<sup>1,2*</sup>, Chunhua Shen<sup>1,2</sup>, Xiaoqin Zhang<sup>1,2</sup>
<br>
<sup>1</sup> College of Computer Science and Technology, Zhejiang University of Technology, Hangzhou 310023
<br>
<sup>2</sup> Zhejiang Key Laboratory of Visual Information Intelligent Processing, Hangzhou 310023
<br>
*Corresponding author
</div>


## Method Overview
We present GeoMotion, a new feed-forward motion segmentation framework that directly infers dynamic masks from latent 4D geometry. It combines 4D geometric priors from a pretrained reconstruction model (pi^3) with local pixel-level motion from optical flow to disentangle object motion from camera motion in a single pass.  Models are available in this repo.


Official codebase for **GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry**.

This repository provides:
- training code (`train.py`)
- evaluation code (`eval.py`)
- inference/visualization code (`motion_seg_inference.py`, `vis_all.sh`)

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n geomotion python=3.10 -y
conda activate geomotion
```

### 1.2 Install PyTorch

Install a PyTorch version matching your CUDA runtime from the official guide:
- https://pytorch.org/get-started/locally/

Example (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Install Project Dependencies

Install **Pi3 official dependencies** first:

```bash
# from Pi3 official repo
# https://github.com/yyfz/Pi3
# pip install -r https://raw.githubusercontent.com/yyfz/Pi3/main/requirements.txt
pip install -r requirements.txt
```

Optional acceleration package:

```bash
pip install xformers
```

## 2. Pretrained Models

Please place checkpoints under `checkpoint/`.

| Checkpoint | Purpose | Expected Path | Download |
|---|---|---|---|
| PI3 backbone (`.safetensors`) | Backbone initialization for train/eval/inference | `checkpoint/model.safetensors` | `TODO: add Hugging Face link` |
| GeoMotion trained model (`.pth`) | Motion segmentation checkpoint | `checkpoint/best_model.pth` | `TODO: add Hugging Face link` |

You can pass PI3 path either by argument or environment variable:

```bash
export PI3_MODEL_PATH=checkpoint/model.safetensors
# Windows PowerShell:
# $env:PI3_MODEL_PATH="checkpoint/model.safetensors"
```

## 3. Dataset Download Guide

### 3.1 Evaluation Datasets

- DAVIS (2016/2017): https://davischallenge.org/davis2017/code.html
- FBMS-59: https://lmb.informatik.uni-freiburg.de/resources/datasets/FBMS/
- SegTrack-v2: https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset/

Recommended local layout:

```text
data/
  DAVIS/
    JPEGImages/480p/<sequence>/*.jpg
    Annotations/480p/<sequence>/*.png
  DAVIS2017-M/
    DAVIS/
      JPEGImages/480p/<sequence>/*.jpg
      Annotations/480p/<sequence>/*.png
  FBMS59_clean/
    JPEGImages/<sequence>/*
    Annotations/<sequence>/*
  SegTrackv2/
    JPEGImages_jpg_standardized/<sequence>/*
    GroundTruth/<sequence>/*
```

### 3.2 Training Datasets (used by current config)

Current training config (`configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml`) uses:
- GOT-10k: https://got-10k.aitestunion.com/
- HOI4D: https://hoi4d.github.io/
- DynamicStereo / GOT-Moving / DynamicVerse: use your prepared copies, then update paths in config.

`train_dataset` is expected to be:

```yaml
train_dataset: ["got10k","hoi4d","dynamic_stereo","gotmoving","dynamicverse"]
```

`train_root` must be in the **same order** as `train_dataset`:

```yaml
train_root:
  - /path/to/GOT-10k_Train_split_01
  - /path/to/HOI4D_clean
  - /path/to/dynamic_stereo_root
  - /path/to/got_train_video_roots_with_masks.txt
  - /path/to/DynamicVerse
```

Dataset structure notes used by current loaders:
- `got10k`: each sequence folder contains `images/` and `masks/`
- `hoi4d`: root contains `images/<seq>/` and `dynamic_masks/<seq>/`
- `dynamic_stereo`: root contains `train/<seq>/images/` and `train/<seq>/dynamic_masks/`
- `gotmoving`: supports either a root directory scan, or a `.txt` file listing sequence roots (recommended)
- `dynamicverse`: recursive scan for `rgb/` and corresponding `mask/` folders

## 4. Inference and Visualization

### 4.1 Quick Start with Script

```bash
bash vis_all.sh
```

Default script command:

```bash
CUDA_VISIBLE_DEVICES=0 python motion_seg_inference.py \
  --model_path checkpoint/best_model.pth \
  --dataset_dir data/test \
  --output_dir output/test \
  --threshold 0.5
```

### 4.2 Single Sequence Inference

```bash
python motion_seg_inference.py \
  --model_path checkpoint/best_model.pth \
  --pi3_model_path checkpoint/model.safetensors \
  --input_dir data/test/<sequence_name> \
  --output_dir output/single \
  --sequence_length 32 \
  --threshold 0.5
```

### 4.3 Dataset Inference

```bash
python motion_seg_inference.py \
  --model_path checkpoint/best_model.pth \
  --pi3_model_path checkpoint/model.safetensors \
  --dataset_dir data/test \
  --output_dir output/test \
  --sequence_length 32 \
  --threshold 0.5
```

## 5. Evaluation

### 5.1 Evaluate One Dataset

```bash
python eval.py \
  --model_path checkpoint/best_model.pth \
  --pi3_model_path checkpoint/model.safetensors \
  --output_dir eval/davis2016 \
  --image_root data/DAVIS/JPEGImages/480p \
  --annotation_root data/DAVIS/Annotations/480p \
  --sequence_length 32 \
  --use_sam_refine True \
  --davis 2016
```

`--davis` supports: `2016`, `2017`, `davis-all`, `2016-M`, `2017-M`, `fbms`, `segtrack`.

### 5.2 Batch Evaluation Script

```bash
bash eval.sh
```

Before running, edit `eval.sh`:
- `CODE_DIR`
- `CUDA_VISIBLE_DEVICES`
- dataset root paths

## 6. Training

### 6.1 Configure Paths

Edit `configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml`:
- `train_dataset` (recommended: `["got10k","hoi4d","dynamic_stereo","gotmoving","dynamicverse"]`)
- `train_root`
- `test_root`
- `log_dir`
- `vggt_model_path` (PI3 `.safetensors` path)

### 6.2 Start Training

Single-GPU (same as `train.sh`):

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master-port=29502 --nproc_per_node=1 train.py configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml
```

Multi-GPU example:

```bash
torchrun --master-port=29502 --nproc_per_node=4 train.py configs/pi3_conf_low_35_feature_flow_gotm_verse_stop_all.yaml
```

Checkpoints and logs are saved to `log_dir` in config.

## 7. Notes

- `eval.py` and `motion_seg_inference.py` now support `--pi3_model_path` and `PI3_MODEL_PATH`.
- If you use custom dataset splits for `2016-M` / `2017-M`, update the sequence lists in `eval.py`.

## 8. Acknowledgements

This project builds on and benefits from the following excellent works:
- Pi3
- VGGT
- SegAnyMotion
- OCLR
- Easi3R

## 9. Citation

```bibtex
@article{geomotion2026,
  title={GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry},
  author={TODO},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```
