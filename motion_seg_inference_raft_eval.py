"""
RAFT-feature 版推理入口。

与 eval_adapter.py 的唯一区别：
1. 使用 pi3_conf_flow_cam_lowfeature_wo_mean_raft 模型（支持 raft_features）
2. predict_motion_mask 中调用 compute_raft_hidden_batch 替代原来的光流幅度
"""
import torch
import torch.nn.functional as F_nn
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Optional

from motion_seg_inference import (
    initialize_raft_model,
    compute_optical_flow_magnitude_batch,
    DEFAULT_RAFT_MODEL_PATH,
    DEFAULT_SAM2_CONFIG_PATH,
    DEFAULT_SAM2_CHECKPOINT_PATH,
)
from motion_seg_inference_raft import compute_raft_hidden_batch


class RaftMotionSegmentationInference:
    """
    RAFT-feature mode inference wrapper.

    Uses the raft-variant model that takes raft_features (hidden_state)
    instead of optical-flow magnitude.
    """

    def __init__(self, model_path, pi3_model_path=None, raft_model_path=None, device='cuda'):
        self.device = device

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Import the raft-variant model
        from pi3.models.pi3_conf_flow_cam_lowfeature_wo_mean_raft import create_pi3_motion_segmentation_model

        if pi3_model_path is None:
            pi3_model_path = os.environ.get("PI3_MODEL_PATH", None)
        self.model = create_pi3_motion_segmentation_model(
            pi3_model_path,
        ).to(device)
        initialize_raft_model(device=device, raft_model_path=raft_model_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded successfully. Best IoU: {checkpoint.get('best_iou', 'N/A')}")

        self.img_size = 518
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def preprocess_video(self, video_frames):
        processed_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            frame = frame.resize((self.img_size, self.img_size), Image.LANCZOS)
            frame_tensor = transforms.ToTensor()(frame)
            frame_tensor = self.normalize(frame_tensor)
            processed_frames.append(frame_tensor)
        video_tensor = torch.stack(processed_frames, dim=0)
        video_tensor = video_tensor.unsqueeze(0)
        return video_tensor

    def predict_motion_mask(self, video_frames, return_confidence=False):
        """
        Predict motion mask using RAFT hidden_state instead of magnitude.

        Returns:
            motion_mask: [S, H, W] numpy array
        """
        video_tensor = self.preprocess_video(video_frames).to(self.device)

        with torch.no_grad():
            # New path: compute RAFT hidden_state features
            raft_features = compute_raft_hidden_batch(
                video_tensor, self.device,
                bidirectional=True,
                fusion_method='mean',
            )  # [B, N, 128, H/8, W/8]

            predictions = self.model(video_tensor, raft_features=raft_features)
            motion_mask = predictions['motion_mask']
            motion_mask = motion_mask.squeeze(0).cpu().numpy()

        return motion_mask

    def load_video_from_directory(self, video_dir, max_frames=None):
        from glob import glob
        from tqdm import tqdm

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(os.path.join(video_dir, ext)))
        img_paths = sorted(img_paths)

        if max_frames:
            img_paths = img_paths[:max_frames]

        images = []
        for path in tqdm(img_paths, desc="Loading frames"):
            img = Image.open(path).convert('RGB')
            images.append(img)
        return images
