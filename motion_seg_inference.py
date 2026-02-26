import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from glob import glob
import tempfile
import shutil
from typing import List, Dict, Tuple
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from code.SegAnyMo import inference
from tqdm import tqdm
from pi3.utils.geometry import process_video_with_improved_sliding_window
from sam2 import build_sam

DEFAULT_RAFT_MODEL_PATH = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DEFAULT_SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT_PATH = "./sam2-main/checkpoints/sam2.1_hiera_large.pt"


def initialize_raft_model(device, raft_model_path=None):
    """Initialize the RAFT optical-flow model."""
    global RAFT_MODEL, RAFT_TRANSFORMS
    if raft_model_path is None:
        raft_model_path = os.environ.get("RAFT_MODEL_PATH", DEFAULT_RAFT_MODEL_PATH)
    
    # Check for torchvision and optical flow models
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        print("Torchvision optical flow models not found (requires torchvision >= 0.13).")
        print("Will fallback to OpenCV optical flow.")
        return False
        
    try:
        # If a specific path is provided, load from it
        if raft_model_path and os.path.exists(raft_model_path):
            print(f"Loading RAFT model from: {raft_model_path}")
            RAFT_MODEL = raft_large(weights=None, progress=False)
            state_dict = torch.load(raft_model_path, map_location=device)
            RAFT_MODEL.load_state_dict(state_dict)
            # When loading custom weights, transforms are often not included, so we create a basic one
            RAFT_TRANSFORMS = lambda img1, img2: (img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)
        else:
            # Otherwise, load the default pre-trained model from torchvision
            print("Loading RAFT model with default torchvision weights...")
            weights = Raft_Large_Weights.DEFAULT
            RAFT_MODEL = raft_large(weights=weights, progress=False)
            RAFT_TRANSFORMS = weights.transforms()
        
        RAFT_MODEL = RAFT_MODEL.to(device).eval()
        print("RAFT model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize RAFT model: {e}")
        print("Will fallback to OpenCV optical flow.")
        RAFT_MODEL = None # Ensure model is None on failure
        return False

def compute_optical_flow_raft(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    使用RAFT计算光流 (输入为 [0,1] 范围的Tensor)
    """
    global RAFT_MODEL, RAFT_TRANSFORMS
    import torchvision.transforms.functional as F
    
    # This function assumes RAFT_MODEL is not None
    original_H, original_W = frame1.shape[1], frame1.shape[2]
    
    # 确保输入是batch格式 (1, 3, H, W)
    if frame1.dim() == 3: frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3: frame2 = frame2.unsqueeze(0)
    
    # RAFT预处理：调整尺寸确保能被8整除
    H, W = frame1.shape[2], frame1.shape[3]
    new_H = ((H + 7) // 8) * 8
    new_W = ((W + 7) // 8) * 8
    
    img1_resized = F.resize(frame1, size=[new_H, new_W], antialias=False)
    img2_resized = F.resize(frame2, size=[new_H, new_W], antialias=False)
    
    # 应用模型特定的变换
    img1_processed, img2_processed = RAFT_TRANSFORMS(img1_resized, img2_resized)
    
    # RAFT推理
    with torch.no_grad():
        flow_predictions = RAFT_MODEL(img1_processed, img2_processed)
        flow_pred = flow_predictions[-1] # 取最后一个（最精细的）预测
    
    # 调整回原始尺寸
    if flow_pred.shape[2] != original_H or flow_pred.shape[3] != original_W:
        # 使用bilinear插值调整光流场大小
        # flow_pred = F.interpolate(flow_pred, size=(original_H, original_W), mode='bilinear', align_corners=False)
        flow_pred = F.resize(flow_pred, size=[original_H, original_W], antialias=False)
                
        # 相应地缩放光流值
        scale_W = original_W / new_W
        scale_H = original_H / new_H
        flow_pred[:, 0] *= scale_W
        flow_pred[:, 1] *= scale_H

    # 返回 (2, H, W)
    return flow_pred.squeeze(0)

def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    将ImageNet标准化的tensor恢复到[0,1]范围
    Args:
        tensor: [..., C, H, W] 标准化后的tensor
    Returns:
        denormalized tensor in [0,1] range
    """
    # ImageNet标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    
    # 反标准化：x = x_norm * std + mean
    denormalized = tensor * std + mean
    
    # 确保在[0,1]范围内
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    return denormalized

def compute_sampson_error_batch(x1, x2, F):
    """
    批量计算Sampson误差(极线几何误差) - GPU加速
    Args:
        x1: [B, N, 2] 第一帧的像素坐标 (normalized [-1,1])
        x2: [B, N, 2] 第二帧的像素坐标 (normalized [-1,1])
        F: [B, 3, 3] 基础矩阵
    Returns:
        error: [B, N] Sampson误差
    """
    # 转换为齐次坐标
    B, N = x1.shape[:2]
    x1_homo = torch.cat([x1, torch.ones(B, N, 1, device=x1.device)], dim=2)  # [B, N, 3]
    x2_homo = torch.cat([x2, torch.ones(B, N, 1, device=x2.device)], dim=2)  # [B, N, 3]
    
    # 计算 x2^T * F * x1 (批量矩阵乘法)
    Fx1 = torch.bmm(x1_homo, F.transpose(1, 2))  # [B, N, 3]
    Ftx2 = torch.bmm(x2_homo, F)                 # [B, N, 3]
    x2Fx1 = torch.sum(x2_homo * Fx1, dim=2)      # [B, N]
    
    # Sampson距离
    denom = Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2
    error = x2Fx1**2 / (denom + 1e-8)
    
    return error


def batch_fundamental_matrix_ransac(x1_batch, x2_batch, confidence=0.99, max_iters=2000):
    """
    批量RANSAC估计基础矩阵 - 使用OpenCV但优化调用
    Args:
        x1_batch: list of [N, 2] numpy arrays
        x2_batch: list of [N, 2] numpy arrays
    Returns:
        F_batch: [B, 3, 3] tensor
        inlier_ratios: [B] tensor
        valid_mask: [B] bool tensor
    """
    B = len(x1_batch)
    F_list = []
    ratio_list = []
    valid_list = []
    
    # 并行处理多个样本 (OpenCV本身会使用多线程)
    for x1, x2 in zip(x1_batch, x2_batch):
        if x1.shape[0] < 8:
            F_list.append(np.eye(3, dtype=np.float32))
            ratio_list.append(0.0)
            valid_list.append(False)
            continue
        
        F, mask = cv2.findFundamentalMat(
            x1, x2, 
            cv2.FM_RANSAC,  # 使用RANSAC代替LMEDS,更快
            ransacReprojThreshold=3.0,
            confidence=confidence,
            maxIters=max_iters
        )
        
        if F is None or mask is None:
            F_list.append(np.eye(3, dtype=np.float32))
            ratio_list.append(0.0)
            valid_list.append(False)
        else:
            F_list.append(F.astype(np.float32))
            ratio_list.append(float(np.sum(mask)) / len(mask))
            valid_list.append(True)
    
    F_batch = torch.from_numpy(np.stack(F_list, axis=0))  # [B, 3, 3]
    ratios = torch.tensor(ratio_list)
    valid = torch.tensor(valid_list)
    
    return F_batch, ratios, valid


def compute_optical_flow_magnitude_batch_with_camera_comp(
    video_tensor: torch.Tensor, 
    device,
    pred_masks: torch.Tensor = None,
    ratio_thresh: float = 0.5,
    error_thresh_high_factor: float = 2.0,
    error_thresh_low_factor: float = 0.01,
    subsample_for_F: int = 4  # 对点进行下采样来加速F矩阵估计
) -> tuple:
    """
    计算光流幅度,并区分相机运动和物体运动 - GPU优化版本
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        pred_masks: [B, N, H, W] 可选的预测mask
        ratio_thresh: 基础矩阵内点比例阈值
        error_thresh_high_factor: 高误差阈值因子
        error_thresh_low_factor: 低误差阈值因子
        subsample_for_F: 点采样步长,越大越快但精度略降
        
    Returns:
        flow_magnitudes: [B, N, H, W] 原始光流幅度
        motion_masks_high: [B, N, H, W] 高置信度物体运动mask
        motion_masks_low: [B, N, H, W] 低误差(静态/相机运动)mask
        trusted_frames: list of (batch_idx, frame_idx) 可信任帧的索引
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized")
        B, N, _, H, W = video_tensor.shape
        return (torch.zeros(B, N, H, W, device=device),
                torch.zeros(B, N, H, W, device=device),
                torch.zeros(B, N, H, W, device=device),
                [])
    
    B, N, C, H, W = video_tensor.shape
    
    # 预分配所有输出张量
    flow_magnitudes = torch.zeros(B, N, H, W, device=device)
    motion_masks_high = torch.zeros(B, N, H, W, device=device)
    motion_masks_low = torch.zeros(B, N, H, W, device=device)
    trusted_frames = []
    
    if pred_masks is None:
        pred_masks = torch.ones(B, N, H, W, device=device)
    
    # 反标准化整个批次
    video_denorm = denormalize_imagenet(video_tensor)
    
    # 预计算归一化网格 (只需计算一次)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    grid = torch.stack([grid_y, grid_x], dim=0).float()  # [2, H, W]
    grid_normalized = torch.stack([
        2.0 * (grid[0] + 0.5) / H - 1,
        2.0 * (grid[1] + 0.5) / W - 1
    ], dim=0)  # [2, H, W]
    
    # 对于F矩阵估计,使用下采样的网格
    if subsample_for_F > 1:
        grid_sub = grid_normalized[:, ::subsample_for_F, ::subsample_for_F]  # [2, H', W']
        H_sub, W_sub = grid_sub.shape[1], grid_sub.shape[2]
    else:
        grid_sub = grid_normalized
        H_sub, W_sub = H, W
    
    # 批量计算所有光流 (主要优化点)
    print("Computing optical flows...")
    all_flows_fwd = {}  # {(b, t): flow}
    all_flows_bwd = {}
    
    # 前向光流: t -> t+1
    for b in range(B):
        for t in range(N-1):
            frame_curr = video_denorm[b, t]
            frame_next = video_denorm[b, t+1]
            flow = compute_optical_flow_raft(frame_curr, frame_next)
            all_flows_fwd[(b, t)] = flow
    
    # 后向光流: t -> t-1
    for b in range(B):
        for t in range(1, N):
            frame_curr = video_denorm[b, t]
            frame_prev = video_denorm[b, t-1]
            flow = compute_optical_flow_raft(frame_curr, frame_prev)
            all_flows_bwd[(b, t)] = flow
    
    print("Computing epipolar geometry constraints...")
    
    # 批量处理每一帧
    for t in range(N):
        # 收集当前时间步所有batch的数据
        batch_flows = []
        batch_has_fwd = []
        batch_has_bwd = []
        
        for b in range(B):
            flows = []
            has_fwd = (b, t) in all_flows_fwd
            has_bwd = (b, t) in all_flows_bwd
            
            if has_fwd:
                flows.append(all_flows_fwd[(b, t)])
            if has_bwd:
                flows.append(all_flows_bwd[(b, t)])
            
            batch_flows.append(flows)
            batch_has_fwd.append(has_fwd)
            batch_has_bwd.append(has_bwd)
        
        # 批量处理F矩阵估计
        x1_batch_for_F = []
        x2_batch_for_F = []
        flow_idx_map = []  # 记录每个flow对应的(batch_idx, flow_type)
        
        for batch_idx in range(B):
            if len(batch_flows[batch_idx]) == 0:
                continue
            
            for flow_idx, flow in enumerate(batch_flows[batch_idx]):
                # 归一化光流
                normalized_flow = torch.stack([
                    2.0 * flow[0] / (H - 1),
                    2.0 * flow[1] / (W - 1)
                ], dim=0).flip(0)  # [2, H, W], flip to (x, y)
                
                # 使用下采样的网格来加速
                if subsample_for_F > 1:
                    flow_sub = F.interpolate(
                        normalized_flow.unsqueeze(0), 
                        size=(H_sub, W_sub), 
                        mode='bilinear', 
                        align_corners=True
                    ).squeeze(0)
                else:
                    flow_sub = normalized_flow
                
                x2 = grid_sub + flow_sub  # [2, H', W']
                
                # 准备点对
                mask = pred_masks[batch_idx, t]
                if subsample_for_F > 1:
                    mask_sub = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H_sub, W_sub),
                        mode='nearest'
                    ).squeeze().bool()
                else:
                    mask_sub = mask.bool()
                
                x1_pts = grid_sub.permute(1, 2, 0).flip(-1).reshape(-1, 2)  # [H'*W', 2]
                x2_pts = x2.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                
                valid = mask_sub.reshape(-1)
                x1_pts = x1_pts[valid].cpu().numpy()
                x2_pts = x2_pts[valid].cpu().numpy()
                
                x1_batch_for_F.append(x1_pts)
                x2_batch_for_F.append(x2_pts)
                flow_idx_map.append((batch_idx, flow_idx))
        
        # 批量估计F矩阵
        if len(x1_batch_for_F) > 0:
            F_batch, ratios, valid = batch_fundamental_matrix_ransac(
                x1_batch_for_F, x2_batch_for_F
            )
            F_batch = F_batch.to(device)
            ratios = ratios.to(device)
            valid = valid.to(device)
            
            # 批量计算Sampson误差
            all_x1 = []
            all_x2 = []
            
            for (batch_idx, flow_idx) in flow_idx_map:
                flow = batch_flows[batch_idx][flow_idx]
                normalized_flow = torch.stack([
                    2.0 * flow[0] / (H - 1),
                    2.0 * flow[1] / (W - 1)
                ], dim=0).flip(0)
                
                x2 = grid_normalized + normalized_flow
                x1_flat = grid_normalized.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                x2_flat = x2.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                
                all_x1.append(x1_flat)
                all_x2.append(x2_flat)
            
            # Stack所有点进行批量计算
            x1_stacked = torch.stack(all_x1, dim=0)  # [num_flows, H*W, 2]
            x2_stacked = torch.stack(all_x2, dim=0)
            
            errors_batch = compute_sampson_error_batch(
                x1_stacked, x2_stacked, F_batch
            )  # [num_flows, H*W]
            
            # 分配回每个样本
            for idx, (batch_idx, flow_idx) in enumerate(flow_idx_map):
                if not valid[idx]:
                    continue
                
                err = errors_batch[idx].reshape(H, W)
                
                # 缩放误差
                fac = (H + W) / 2
                err = err * fac**2
                err = torch.clamp(err, min=0)
                
                # 计算平均光流幅度
                flow = batch_flows[batch_idx][flow_idx]
                flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2)
                avg_flow_norm = flow_mag.mean()
                
                # 存储第一个光流的幅度
                if flow_idx == 0:
                    flow_magnitudes[batch_idx, t] = flow_mag
                
                # 更新或累积误差
                if flow_idx == 0:
                    err_combined = err
                else:
                    err_combined = torch.max(err_combined, err)
                
                # 最后一个flow处理完后生成mask
                if flow_idx == len(batch_flows[batch_idx]) - 1:
                    # 高误差阈值
                    thresh_high = torch.clamp(
                        error_thresh_high_factor * avg_flow_norm, min=0.5
                    )
                    motion_masks_high[batch_idx, t] = (err_combined > thresh_high).float()
                    
                    # 低误差阈值
                    thresh_low = torch.clamp(
                        avg_flow_norm * error_thresh_low_factor, max=0.01
                    )
                    motion_masks_low[batch_idx, t] = (err_combined <= thresh_low).float()
                    
                    # 检查可信度
                    all_valid = True
                    for fi in range(len(batch_flows[batch_idx])):
                        map_idx = flow_idx_map.index((batch_idx, fi))
                        if ratios[map_idx] <= ratio_thresh:
                            all_valid = False
                            break
                    
                    if all_valid and len(batch_flows[batch_idx]) > 0:
                        trusted_frames.append((batch_idx, t))
    
    return flow_magnitudes, motion_masks_high, motion_masks_low, trusted_frames


def compute_optical_flow_magnitude_batch(video_tensor: torch.Tensor, device, 
                                        bidirectional: bool = True,
                                        fusion_method: str = 'max',
                                        return_flow: bool = False):
    """
    为整个视频批次计算光流（支持返回向量）
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        bidirectional: 是否使用双向光流
        fusion_method: 双向光流融合方法
        return_flow: 是否返回光流向量（NEW!）
    
    Returns:
        如果 return_flow=False:
            flow_magnitudes: [B, N, H, W]
        如果 return_flow=True:
            (flow_vectors, flow_magnitudes)
            flow_vectors: [B, N, 2, H, W] 或 [B, N, H, W, 2]
            flow_magnitudes: [B, N, H, W]
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized, using zero flow")
        B, N, _, H, W = video_tensor.shape
        flow_magnitudes = torch.zeros(B, N, H, W, device=device)
        if return_flow:
            flow_vectors = torch.zeros(B, N, 2, H, W, device=device)
            return flow_vectors, flow_magnitudes
        return flow_magnitudes
    
    B, N, C, H, W = video_tensor.shape
    flow_magnitudes = torch.zeros(B, N, H, W, device=device)
    flow_vectors = torch.zeros(B, N, 2, H, W, device=device) if return_flow else None
    
    # 反标准化到[0,1]范围
    video_denorm = denormalize_imagenet(video_tensor)
    
    if not bidirectional:
        for b in range(B):
            for t in range(1, N):
                frame1 = video_denorm[b, t-1]
                frame2 = video_denorm[b, t]
                
                flow = compute_optical_flow_raft(frame1, frame2)  # [2, H, W]
                flow_magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
                
                flow_magnitudes[b, t] = flow_magnitude
                if return_flow:
                    flow_vectors[b, t] = flow  # [2, H, W]
    else:
        # 双向光流
        for b in range(B):
            forward_flows = []
            backward_flows = []
            forward_flow_vecs = [] if return_flow else None
            backward_flow_vecs = [] if return_flow else None
            
            for t in range(N-1):
                frame1 = video_denorm[b, t]
                frame2 = video_denorm[b, t+1]
                
                # 前向光流
                forward_flow = compute_optical_flow_raft(frame1, frame2)
                forward_magnitude = torch.sqrt(forward_flow[0]**2 + forward_flow[1]**2)
                forward_flows.append(forward_magnitude)
                if return_flow:
                    forward_flow_vecs.append(forward_flow)
                
                # 后向光流
                backward_flow = compute_optical_flow_raft(frame2, frame1)
                backward_magnitude = torch.sqrt(backward_flow[0]**2 + backward_flow[1]**2)
                backward_flows.append(backward_magnitude)
                if return_flow:
                    backward_flow_vecs.append(backward_flow)
            
            # 为每一帧分配光流
            for t in range(N):
                flow_values = []
                flow_vecs = [] if return_flow else None
                
                if t > 0:
                    flow_values.append(backward_flows[t-1])
                    if return_flow:
                        flow_vecs.append(backward_flow_vecs[t-1])
                
                if t < N-1:
                    flow_values.append(forward_flows[t])
                    if return_flow:
                        flow_vecs.append(forward_flow_vecs[t])
                
                if len(flow_values) > 0:
                    flow_magnitudes[b, t] = fuse_flow_magnitudes(flow_values, fusion_method)
                    if return_flow:
                        # 融合向量：用 magnitude 做加权平均
                        if fusion_method == 'max':
                            max_idx = torch.argmax(torch.stack([v.max() for v in flow_values]))
                            flow_vectors[b, t] = flow_vecs[max_idx]
                        else:  # mean
                            flow_vectors[b, t] = torch.stack(flow_vecs).mean(dim=0)
    
    if return_flow:
        return flow_vectors, flow_magnitudes
    return flow_magnitudes


def compute_optical_flow_batch_xy(video_tensor: torch.Tensor, device, 
                                  bidirectional: bool = True,
                                  fusion_method: str = 'max') -> torch.Tensor:
    """
    为整个视频批次计算光流（返回完整的x,y分量）
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        bidirectional: 是否使用双向光流
        fusion_method: 双向光流融合方法
    
    Returns:
        flow_xy: [B, N, H, W, 2] 光流x,y分量
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized")
        B, N, _, H, W = video_tensor.shape
        return torch.zeros(B, N, H, W, 2, device=device)
    
    B, N, C, H, W = video_tensor.shape
    flow_xy = torch.zeros(B, N, H, W, 2, device=device)
    
    # 反标准化
    video_denorm = denormalize_imagenet(video_tensor)
    
    if not bidirectional:
        for b in range(B):
            for t in range(1, N):
                frame1 = video_denorm[b, t-1]
                frame2 = video_denorm[b, t]
                
                # 计算光流 [2, H, W]
                flow = compute_optical_flow_raft(frame1, frame2)
                flow_xy[b, t] = flow.permute(1, 2, 0)  # [H, W, 2]
    else:
        # 双向光流
        for b in range(B):
            forward_flows = []  # list of [H, W, 2]
            backward_flows = []
            
            for t in range(N-1):
                frame1 = video_denorm[b, t]
                frame2 = video_denorm[b, t+1]
                
                # 前向光流
                forward_flow = compute_optical_flow_raft(frame1, frame2)  # [2, H, W]
                forward_flows.append(forward_flow.permute(1, 2, 0))  # [H, W, 2]
                
                # 后向光流
                backward_flow = compute_optical_flow_raft(frame2, frame1)
                backward_flows.append(backward_flow.permute(1, 2, 0))  # [H, W, 2]
            
            # 为每一帧分配光流
            for t in range(N):
                flow_values = []
                
                if t > 0:
                    flow_values.append(backward_flows[t-1])
                
                if t < N-1:
                    flow_values.append(forward_flows[t])
                
                if len(flow_values) > 0:
                    flow_xy[b, t] = fuse_flow_xy(flow_values, fusion_method)
    
    return flow_xy

def fuse_flow_xy(flow_list, method='max'):
    """
    融合多个光流（x,y分量）
    
    Args:
        flow_list: list of [H, W, 2]
        method: 融合方法
    
    Returns:
        fused_flow: [H, W, 2]
    """
    flow_stack = torch.stack(flow_list, dim=0)  # [N_flows, H, W, 2]
    flow_magnitude = torch.norm(flow_stack, dim=-1)  # [N_flows, H, W]
    
    if method == 'max':
        max_idx = torch.argmax(flow_magnitude, dim=0)  # [H, W]
        fused = flow_stack[max_idx, torch.arange(flow_stack.shape[1])[:, None], 
                          torch.arange(flow_stack.shape[2])]
    elif method == 'mean':
        fused = flow_stack.mean(dim=0)
    elif method == 'sum':
        fused = flow_stack.sum(dim=0)
    
    return fused

def bilinear_sample(tensor, h_coords, w_coords):
    """
    双线性插值采样
    
    Args:
        tensor: [H, W, C] 或 [B, H, W, C]
        h_coords: [H, W] 或 [B, H, W] 高度坐标（可以是浮点数）
        w_coords: [H, W] 或 [B, H, W] 宽度坐标
    
    Returns:
        sampled: [H, W, C] 或 [B, H, W, C]
    """
    # 处理维度
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        h_coords = h_coords.unsqueeze(0)
        w_coords = w_coords.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W, C = tensor.shape
    
    # 归一化坐标到[-1, 1]范围
    h_norm = 2.0 * h_coords / (H - 1) - 1.0
    w_norm = 2.0 * w_coords / (W - 1) - 1.0
    
    # 创建网格
    grid = torch.stack([w_norm, h_norm], dim=-1)  # [B, H, W, 2]
    
    # 使用grid_sample进行采样
    sampled = F.grid_sample(tensor.permute(0, 3, 1, 2),  # [B, C, H, W]
                           grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=True)
    
    sampled = sampled.permute(0, 2, 3, 1)  # [B, H, W, C]
    
    if squeeze_output:
        sampled = sampled.squeeze(0)
    
    return sampled

def compute_flow_consistency(flow_forward, flow_backward, H, W, threshold=1.0):
    """
    计算前后向光流的一致性掩码
    
    前向：t -> t+1
    后向：t+1 -> t
    
    如果像素(h,w)通过前向光流追踪到(h',w')，
    然后从(h',w')通过后向光流追踪回来，
    如果回到的位置接近(h,w)，说明光流一致且可信
    
    Args:
        flow_forward: [H, W, 2] 前向光流 (t -> t+1)
        flow_backward: [H, W, 2] 后向光流 (t+1 -> t)
        H, W: 图像高宽
        threshold: float 一致性阈值
    
    Returns:
        consistency_mask: [H, W] 布尔掩码，True表示光流可信
        occlusion_mask: [H, W] 布尔掩码，True表示被遮挡的区域
    """
    device = flow_forward.device
    
    # 创建像素坐标网格
    h_grid, w_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                    torch.arange(W, dtype=torch.float32, device=device),
                                    indexing='ij')
    
    # 前向追踪
    h_forward = h_grid + flow_forward[..., 1]
    w_forward = w_grid + flow_forward[..., 0]
    
    # 限制边界
    h_forward = torch.clamp(h_forward, 0, H - 1)
    w_forward = torch.clamp(w_forward, 0, W - 1)
    
    # 采样后向光流
    flow_backward_tracked = bilinear_sample(flow_backward, h_forward, w_forward)  # [H, W, 2]
    
    # 后向追踪
    h_backward = h_forward + flow_backward_tracked[..., 1]
    w_backward = w_forward + flow_backward_tracked[..., 0]
    
    # 计算回到原始位置的偏差
    h_error = torch.abs(h_backward - h_grid)
    w_error = torch.abs(w_backward - w_grid)
    reprojection_error = torch.sqrt(h_error**2 + w_error**2)  # [H, W]
    
    # 一致性掩码：误差小于阈值的像素
    consistency_mask = reprojection_error < threshold  # [H, W]
    
    # 遮挡检测：边界外的像素视为被遮挡
    boundary_mask = (h_forward < 0) | (h_forward >= H) | (w_forward < 0) | (w_forward >= W)
    occlusion_mask = boundary_mask
    
    return consistency_mask, occlusion_mask, reprojection_error

def compute_motion_from_tracked_points(flow_xy, points_world_t, points_world_t1, 
                                       flow_backward=None, H=None, W=None,
                                       consistency_threshold=1.0):
    """
    用光流追踪像素，比较世界坐标的3D点运动
    仅在光流一致的区域计算运动
    
    Args:
        flow_xy: [H, W, 2] 或 [B, H, W, 2] 前向光流的x,y分量
        points_world_t: [H, W, 3] 或 [B, H, W, 3] t时刻的世界坐标
        points_world_t1: [H, W, 3] 或 [B, H, W, 3] t+1时刻的世界坐标
        flow_backward: [H, W, 2] 可选的后向光流，用于一致性检验
        H, W: 图像高宽
        consistency_threshold: float 一致性阈值
    
    Returns:
        point_motion: [H, W] 或 [B, H, W] 3D点的运动量（不可信区域为0）
        tracked_points: [H, W, 3] 或 [B, H, W, 3] 追踪到的t+1时刻的世界坐标
        confidence_mask: [H, W] 或 [B, H, W] 光流可信性掩码
    """
    device = flow_xy.device
    
    # 处理batch维度
    if flow_xy.dim() == 3:
        flow_xy = flow_xy.unsqueeze(0)
        points_world_t = points_world_t.unsqueeze(0)
        points_world_t1 = points_world_t1.unsqueeze(0)
        if flow_backward is not None:
            flow_backward = flow_backward.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B = flow_xy.shape[0]
    if H is None or W is None:
        H, W = flow_xy.shape[1:3]
    
    # 创建像素坐标网格
    h_grid, w_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                    torch.arange(W, dtype=torch.float32, device=device),
                                    indexing='ij')
    
    # 扩展到batch维度
    h_grid = h_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    w_grid = w_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # ===== 计算光流一致性 =====
    consistency_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
    occlusion_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    
    if flow_backward is not None:
        for b in range(B):
            cons_mask, occ_mask, _ = compute_flow_consistency(
                flow_xy[b], flow_backward[b], H, W, 
                threshold=consistency_threshold
            )
            consistency_mask[b] = cons_mask
            occlusion_mask[b] = occ_mask
    
    # ===== 追踪位置 =====
    h_tracked = h_grid + flow_xy[..., 1]  # [B, H, W]
    w_tracked = w_grid + flow_xy[..., 0]  # [B, H, W]
    
    # 限制在图像边界内
    h_tracked = torch.clamp(h_tracked, 0, H - 1)
    w_tracked = torch.clamp(w_tracked, 0, W - 1)
    
    # 双线性插值获取追踪位置的世界坐标
    tracked_world_t1 = bilinear_sample(points_world_t1, h_tracked, w_tracked)  # [B, H, W, 3]
    
    # 计算3D点的运动向量
    point_motion_vec = tracked_world_t1 - points_world_t  # [B, H, W, 3]
    
    # 计算运动量（欧氏距离）
    point_motion = torch.norm(point_motion_vec, dim=-1)  # [B, H, W]
    
    # ===== 仅在可信区域保留运动信号 =====
    # 不可信的区域（遮挡、光流不一致）设为0
    confidence_mask = consistency_mask & (~occlusion_mask)  # [B, H, W]
    point_motion = point_motion * confidence_mask.float()
    
    if squeeze_output:
        point_motion = point_motion.squeeze(0)
        tracked_world_t1 = tracked_world_t1.squeeze(0)
        confidence_mask = confidence_mask.squeeze(0)
    
    return point_motion, tracked_world_t1, confidence_mask

def fuse_flow_magnitudes(flow_values: list, method: str = 'max') -> torch.Tensor:
    """
    融合多个光流幅度值
    Args:
        flow_values: 光流幅度值列表
        method: 融合方法
    Returns:
        融合后的光流幅度
    """
    if len(flow_values) == 1:
        return flow_values[0]
    
    if method == 'max':
        # 取最大值
        return torch.max(torch.stack(flow_values), dim=0)[0]
    elif method == 'mean':
        # 取平均值
        return torch.mean(torch.stack(flow_values), dim=0)
    elif method == 'sum':
        # 求和
        return torch.sum(torch.stack(flow_values), dim=0)
    elif method == 'weighted_mean':
        # 加权平均（给予时间上更近的帧更高权重）
        weights = torch.tensor([1.0] * len(flow_values), device=flow_values[0].device)
        weights = weights / weights.sum()
        weighted_flows = torch.stack([w * f for w, f in zip(weights, flow_values)])
        return torch.sum(weighted_flows, dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

def compute_bidirectional_flow_with_temporal_consistency(video_tensor: torch.Tensor, device,
                                                       temporal_weight: float = 0.8) -> torch.Tensor:
    """
    计算具有时间一致性的双向光流
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        temporal_weight: 时间一致性权重，越大越平滑
    Returns:
        flow_magnitudes: [B, N, H, W] 光流幅度
    """
    # 先获得基础的双向光流
    base_flows = compute_optical_flow_magnitude_batch(
        video_tensor, device, bidirectional=True, fusion_method='mean'
    )
    
    B, N, H, W = base_flows.shape
    smoothed_flows = base_flows.clone()
    
    # 应用时间平滑
    for b in range(B):
        for t in range(1, N-1):  # 跳过边界帧
            # 时间平滑：当前帧 = (1-w) * 当前帧 + w * (前一帧 + 后一帧) / 2
            neighbor_mean = (smoothed_flows[b, t-1] + smoothed_flows[b, t+1]) / 2
            smoothed_flows[b, t] = (1 - temporal_weight) * base_flows[b, t] + \
                                 temporal_weight * neighbor_mean
    
    return smoothed_flows

def normalize_flow_magnitudes(flow_magnitudes: torch.Tensor, 
                            method='percentile', 
                            percentile=95, 
                            min_val=0.0, 
                            max_val=1.0) -> torch.Tensor:
    """
    归一化光流幅度
    Args:
        flow_magnitudes: [B, N, H, W] 光流幅度
        method: 归一化方法 ('minmax', 'percentile', 'zscore')
        percentile: 百分位数（method='percentile'时使用）
        min_val, max_val: 输出范围
    Returns:
        normalized_flow: [B, N, H, W] 归一化后的光流幅度
    """
    if method == 'minmax':
        # Min-Max归一化
        flow_min = flow_magnitudes.min()
        flow_max = flow_magnitudes.max()
        if flow_max > flow_min:
            normalized = (flow_magnitudes - flow_min) / (flow_max - flow_min)
        else:
            normalized = torch.zeros_like(flow_magnitudes)
            
    elif method == 'percentile':
        # 基于百分位数的归一化
        flow_percentile = torch.quantile(flow_magnitudes, percentile/100.0)
        normalized = torch.clamp(flow_magnitudes / (flow_percentile + 1e-8), 0, 1)
        
    elif method == 'zscore':
        # Z-score归一化后sigmoid映射到[0,1]
        flow_mean = flow_magnitudes.mean()
        flow_std = flow_magnitudes.std()
        if flow_std > 1e-8:
            zscore = (flow_magnitudes - flow_mean) / flow_std
            normalized = torch.sigmoid(zscore)
        else:
            normalized = torch.ones_like(flow_magnitudes) * 0.5
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 缩放到指定范围
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized


class MotionSegmentationInference:
    """
    Inference class for motion segmentation using trained VGGT model
    """
    def __init__(self, model_path, pi3_model_path=None, raft_model_path=None, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device
        
        # Load model
        # model_path = '/data0/hexiankang/code/SegAnyMo/logs/motion_pi3_omni/motion_seg_model_epoch_14.pth'
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model (you might need to adjust these parameters)
        from pi3.models.pi3_conf_flow_cam_lowfeature_wo_mean import create_pi3_motion_segmentation_model
        if pi3_model_path is None:
            pi3_model_path = os.environ.get("PI3_MODEL_PATH", None)
        self.model = create_pi3_motion_segmentation_model(
            pi3_model_path=pi3_model_path,
        ).to(device)
        initialize_raft_model(device=device, raft_model_path=raft_model_path)
        
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully. Best IoU: {checkpoint.get('best_iou', 'N/A')}")
        
        # Setup image preprocessing
        self.img_size = 518
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_video(self, video_frames):
        """
        Preprocess video frames for VGGT input
        
        Args:
            video_frames: List of PIL images or numpy arrays
            
        Returns:
            torch.Tensor: Preprocessed video tensor [1, S, 3, H, W]
        """
        processed_frames = []
        
        for frame in video_frames:
            # Convert to PIL if numpy
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            
            # Resize to target size
            frame = frame.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            # Convert to tensor and normalize
            frame_tensor = transforms.ToTensor()(frame)  # [3, H, W] in [0, 1]
            frame_tensor = self.normalize(frame_tensor)  # Normalize for VGGT
            
            processed_frames.append(frame_tensor)
        
        # Stack and add batch dimension
        video_tensor = torch.stack(processed_frames, dim=0)  # [S, 3, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, S, 3, H, W]
        
        return video_tensor
    
    def save_motion_mask(self, masks, save_dir="./motion_mask_vis"):
        """
        保存 motion_mask 可视化结果
        Args:
            masks (torch.Tensor or np.ndarray): 形状 [T, H, W] 或 [1, T, H, W]
            save_dir (str): 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 去掉 batch 维度
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().squeeze(0).numpy()  # [T, H, W]

        num_frames = masks.shape[0]

        for i in range(num_frames):
            mask_np = masks[i]

            # 自动确定 vmax (95% 分位数，避免 outlier)
            vmax = np.percentile(mask_np[mask_np > 0], 95) if np.any(mask_np > 0) else 0.1

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mask_np, cmap="hot", vmin=0, vmax=vmax)
            ax.set_title(f"Motion Mask (Frame {i})")
            ax.axis("off")

            plt.colorbar(im, ax=ax, shrink=0.7, label="Mask Value")
            save_path = os.path.join(save_dir, f"motion_mask_{i:02d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"已保存 {num_frames} 张 motion_mask 到 {os.path.abspath(save_dir)}")


    def visualize_motion_results_improved(
        image: np.ndarray,
        # motion_mask: torch.Tensor,
        # accumulated_mask: torch.Tensor,
        motion_magnitude: torch.Tensor,
        # depth_weights: torch.Tensor,
        # debug_info: Dict,
        save_path: str
    ):
        """改进的可视化函数，包含深度权重信息"""
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        # motion_mask_np = motion_mask.cpu().numpy().astype(np.uint8) * 255
        # accumulated_mask_np = accumulated_mask.cpu().numpy().astype(np.uint8) * 255
        motion_magnitude_np = motion_magnitude.cpu().numpy()
        # depth_weights_np = depth_weights.cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        frame_name = os.path.basename(save_path).split('.')[0]
        
        # debug_text = f"Method: {debug_info.get('method', 'unknown')}"
        # if 'num_valid_comparisons' in debug_info:
            # debug_text += f", Comparisons: {debug_info['num_valid_comparisons']}"
        
        # fig.suptitle(f"{frame_name} - {debug_text}", fontsize=14)

        # 第一行：原图、瞬时运动、累积运动、深度权重
        axes[0,0].imshow(image)
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')

        # 'Motion Magnitude
        vmax = np.percentile(motion_magnitude_np[motion_magnitude_np > 0], 95) if np.any(motion_magnitude_np > 0) else 0.1
        im1 = axes[1,0].imshow(motion_magnitude_np, cmap='hot', vmin=0, vmax=vmax)
        axes[1,0].set_title('Motion Magnitude')
        axes[1,0].axis('off')
        plt.colorbar(im1, ax=axes[1,0], shrink=0.7, label='Motion')

        axes[1,3].set_xlim(0, 1)
        axes[1,3].set_ylim(0, 1)
        axes[1,3].set_title('Statistics')
        axes[1,3].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def predict_with_flow(self, video_frames):
        """
        预测时同时返回光流向量
        
        Returns:
            {
                'motion_prob': (B, H, W),
                'flow_vectors': (B, 2, H, W)  # NEW!
            }
        """
        video_tensor = self.preprocess_video(video_frames).to(self.device)
        # 把你现有的 compute_optical_flow_magnitude_batch 改成：
        with torch.no_grad():
            flow_vectors, flow_mags = compute_optical_flow_magnitude_batch(
                video_tensor, 
                self.device,
                bidirectional=True,
                fusion_method='max',
                return_flow=True  # 关键改动！
            )
        
            # 你的模型仍然只吃 magnitude，不动
            predictions = self.model(video_tensor, flow_mags)
        
        return {
            'motion_prob': predictions['motion_mask'],
            'flow_vectors': flow_vectors  # 新增返回
        }
    
    def predict_motion_mask(self, video_frames, return_confidence=False):
        """
        Predict motion segmentation mask for video frames
        
        Args:
            video_frames: List of PIL images or numpy arrays
            return_confidence: Whether to return additional confidence maps
            
        Returns:
            numpy array: Motion masks [S, H, W] in range [0, 1]
            dict (optional): Additional outputs if return_confidence=True
        """
        from pi3.models.pi3_conf import run_pi3_attention_analysis
        # Preprocess input
        video_tensor = self.preprocess_video(video_frames).to(self.device)

        # Forward pass
        with torch.no_grad():
            flow_magnitudes = compute_optical_flow_magnitude_batch(
                video_tensor, self.device, 
                bidirectional=True,
                fusion_method='max'
            )  # [B, N, H, W]

            # predictions = self.model(video_tensor)
            predictions = self.model(video_tensor, flow_magnitudes)
            motion_mask = predictions['motion_mask']  # [1, S, H, W]
            motion_mask = motion_mask.squeeze(0).cpu().numpy()  # [S, H, W]

        return motion_mask
    
    def load_video_from_directory(self, video_dir, max_frames=None):
        """
        Load video frames from directory
        
        Args:
            video_dir: Directory containing video frames
            max_frames: Maximum number of frames to load
            
        Returns:
            List of PIL images
        """
        # Get image paths
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob(os.path.join(video_dir, ext)))
        
        # img_paths = sorted(img_paths)[::10]
        img_paths = sorted(img_paths)
        
        if max_frames:
            img_paths = img_paths[:max_frames]
        
        # Load images
        images = []
        for path in tqdm(img_paths, desc="Loading frames"):
            img = Image.open(path).convert('RGB')
            images.append(img)
        
        return images
    
    def load_video_from_file(self, video_path, max_frames=None, sample_rate=1):
        """
        Load video from video file (mp4, avi, etc.)
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            sample_rate: Frame sampling rate (1 = every frame)
            
        Returns:
            List of PIL images
        """
        cap = cv2.VideoCapture(video_path)
        
        images = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                images.append(img)
                
                if max_frames and len(images) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Loaded {len(images)} frames from {video_path}")
        
        return images


def visualize_motion_segmentation(images, motion_masks, save_dir, sequence_name="video", 
                                threshold=0.5, overlay_alpha=0.6):
    """
    Create comprehensive visualizations of motion segmentation results
    
    Args:
        images: List of PIL images (original size)
        motion_masks: numpy array [S, H, W] of motion masks
        save_dir: Directory to save visualizations
        sequence_name: Name for the sequence
        threshold: Binary threshold for mask visualization
        overlay_alpha: Alpha for overlay visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_frames = len(images)
    
    # Create individual frame visualizations
    frame_dir = os.path.join(save_dir, f"{sequence_name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    # import pdb; pdb.set_trace()
    
    
    # Create summary grid visualization
    create_summary_grid(images, motion_masks, save_dir, sequence_name, threshold)
    
    # Create video from frames
    # create_motion_video(images, motion_masks, save_dir, sequence_name, threshold)
    
    print(f"Visualizations saved to {save_dir}")


def create_summary_grid(images, motion_masks, save_dir, sequence_name, threshold=0.5):
    """
    Create a grid summary of the entire sequence
    """
    num_frames = len(images)
    grid_size = min(8, num_frames)  # Show up to 8 frames
    
    if grid_size == 0:
        return
    
    # Sample frames evenly
    frame_indices = np.linspace(0, num_frames-1, grid_size, dtype=int)
    
    fig, axes = plt.subplots(3, grid_size, figsize=(2*grid_size, 6))
    if grid_size == 1:
        axes = axes.reshape(3, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        img = images[frame_idx]
        mask = motion_masks[frame_idx]
        
        # Resize mask to match image
        orig_w, orig_h = img.size
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')
        
        # Motion mask
        axes[1, i].imshow(mask_resized, cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title('Motion Mask')
        axes[1, i].axis('off')
        
        # Overlay
        img_array = np.array(img)
        overlay = img_array.copy()
        motion_pixels = mask_resized > threshold
        overlay[motion_pixels] = overlay[motion_pixels] * 0.4 + np.array([255, 0, 0]) * 0.6
        
        axes[2, i].imshow(overlay.astype(np.uint8))
        axes[2, i].set_title('Overlay')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Motion Segmentation Summary - {sequence_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{sequence_name}_summary.png'), 
               dpi=200, bbox_inches='tight')
    plt.close()


def create_motion_video(images, motion_masks, save_dir, sequence_name, threshold=0.5, fps=15):
    """
    Create video visualization of motion segmentation
    """
    if len(images) == 0:
        return
    
    # Get output video path
    video_path = os.path.join(save_dir, f'{sequence_name}_motion.mp4')
    
    # Get frame size from first image
    orig_w, orig_h = images[0].size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (orig_w * 3, orig_h))
    
    for img, mask in zip(images, motion_masks):
        # Resize mask to original image size
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert PIL to numpy
        img_array = np.array(img)
        
        # Create binary mask
        binary_mask = (mask_resized > threshold).astype(np.uint8) * 255
        binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        
        # Create overlay
        overlay = img_array.copy()
        motion_pixels = mask_resized > threshold
        overlay[motion_pixels] = overlay[motion_pixels] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        # Concatenate horizontally: original | binary_mask | overlay
        combined_frame = np.hstack([img_array, binary_mask_3ch, overlay])
        
        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(combined_frame_bgr)
    
    out.release()
    print(f"Motion video saved to {video_path}")

def run_inference_on_video_directory(inference, video_dir, output_dir, 
                                   max_frames=None, sequence_length=16, use_sam_refine=True):
    """
    Run inference on a directory of video frames
    
    Args:
        model_path: Path to trained model
        video_dir: Directory containing video frames
        output_dir: Directory to save results
        max_frames: Maximum frames to process
        sequence_length: Number of frames to process at once
        use_sam_refine: Whether to use SAM2 for refinement
    """
    # Initialize inference
    # inference = MotionSegmentationInference(model_path)
    
    # Load video frames
    print(f"Loading video from {video_dir}")
    video_frames = inference.load_video_from_directory(video_dir, max_frames)
    
    if len(video_frames) == 0:
        print("No frames found in directory")
        return
    
    print(f"Processing {len(video_frames)} frames")
    
    # Process video in chunks if it's too long
    all_motion_masks = []
    all_motion_masks_before = []
    all_motion_masks_after = []
    
    for start_idx in tqdm(range(0, len(video_frames), sequence_length), desc="Processing chunks"):
        end_idx = min(start_idx + sequence_length, len(video_frames))
        chunk_frames = video_frames[start_idx:end_idx]
        
        # ---------- Predict ----------
        motion_masks = inference.predict_motion_mask(chunk_frames)
        
        # ⭐ 保存 refine 前结果（一定要 copy）
        motion_masks_before = motion_masks.copy()
        
        # ---------- SAM refine ----------
        if use_sam_refine:
            print(f"Refining masks with SAM2 for frames {start_idx}-{end_idx}")
            
            mask_h, mask_w = motion_masks.shape[1], motion_masks.shape[2]
            
            frame_arrays = []
            for pil_img in chunk_frames:
                pil_img_resized = pil_img.resize((mask_w, mask_h), Image.BILINEAR)
                frame_array = np.array(pil_img_resized).astype(np.float32) / 255.0
                frame_arrays.append(frame_array)
            
            frame_arrays = np.stack(frame_arrays, axis=0)
            frame_tensors = torch.from_numpy(frame_arrays).float().permute(0, 3, 1, 2)
            
            refined_masks = torch.zeros(
                motion_masks.shape[0],
                motion_masks.shape[1],
                motion_masks.shape[2],
                dtype=torch.float32
            )
            
            mask_list = torch.from_numpy(motion_masks).float()
            refine_sam(frame_tensors, mask_list, refined_masks, offset=0)
            
            motion_masks = refined_masks.cpu().numpy()
        
        # ---------- Collect ----------
        all_motion_masks_before.append(motion_masks_before)
        all_motion_masks_after.append(motion_masks)

    # Concatenate all chunks
    all_motion_masks_before = np.concatenate(all_motion_masks_before, axis=0)
    all_motion_masks_after  = np.concatenate(all_motion_masks_after, axis=0)
    # all_motion_masks = np.concatenate(all_motion_masks, axis=0)
    
    # Save results
    sequence_name = os.path.basename(video_dir)
    # save_motion_masks(all_motion_masks, video_frames, output_dir, sequence_name)
    
    # Create visualizations
    visualize_motion_segmentation(
        video_frames,
        all_motion_masks_before,
        output_dir,
        sequence_name + "_before_sam"
    )

    visualize_motion_segmentation(
        video_frames,
        all_motion_masks_after,
        output_dir,
        sequence_name + "_after_sam"
    )
    

    return all_motion_masks_after

def split_components(mask_t, min_area=64):
    """
    将二进制掩码拆分为独立的连通域。
    Args:
        mask_t: torch tensor [H, W] 或 [1, H, W]
        min_area: 忽略小于此像素数的连通域（去噪）
    Returns:
        List of torch tensors, 每个都是单独的连通域 mask
    """
    # 确保转为 numpy uint8 处理
    if mask_t.ndim == 3:
        mask_t = mask_t[0]
    
    # 确保是在 CPU 上操作 opencv
    m = (mask_t.detach().float().cpu().numpy() > 0.5).astype(np.uint8)

    # 连通域分析
    n, labels = cv2.connectedComponents(m, connectivity=8)
    
    comps = []
    # label 0 是背景，从 1 开始遍历前景
    for k in range(1, n):
        comp = (labels == k).astype(np.uint8)
        if comp.sum() >= min_area:
            # 转回 Tensor 并保持在原设备
            comp_tensor = torch.from_numpy(comp).to(mask_t.device).float()
            # SAM2 add_new_mask 通常不需要 channel 维 (H,W)，如果报错可尝试 unsqueeze(0)
            comps.append(comp_tensor)
            
    return comps


def preprocess_mask(mask_tensor, threshold=0.8):
    # 1. 二值化
    binary_mask = (mask_tensor > threshold).float()
    
    # 2. (可选) 连通域去噪: 如果你想更狠一点，可以把太小的点去掉
    # 注意: 这需要在 CPU numpy 上做，如果不想切设备，只做二值化也足够有效
    # mask_np = binary_mask.cpu().numpy()
    # ... cv2.connectedComponents ...
    
    return binary_mask

def refine_sam(frame_tensors, mask_list, p_masks_sam, offset=0):
  """Refine the final motion masks with SAM2.

  Args:
    frame_tensors: video frames
    mask_list: list of final motion masks
    p_masks_sam: returned SAM2-refined masks
    offset: video frame offset
  """
  model_config = os.environ.get("SAM2_CONFIG_PATH", DEFAULT_SAM2_CONFIG_PATH)
  checkpoint = os.environ.get("SAM2_CHECKPOINT_PATH", DEFAULT_SAM2_CHECKPOINT_PATH)
  predictor = build_sam.build_sam2_video_predictor(
      model_config, checkpoint, device='cuda'
  )
  # make tmp dir for SAMv2
  tmp_dir = tempfile.mkdtemp()
  for i in range(frame_tensors.shape[0]):
    frame = frame_tensors[i].permute(1, 2, 0).cpu().numpy()
    img_path = os.path.join(tmp_dir, f'{i:05}.jpg')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, np.uint8(frame * 255))
  with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    inference_state = predictor.init_state(tmp_dir)
    shutil.rmtree(tmp_dir)
    ann_obj_id = 1
    # Process even frames
    predictor.reset_state(inference_state)

    for idx, mask in enumerate(mask_list):
      if idx % 2 == 1:
        # import pdb;pdb.set_trace()
        clean_mask = preprocess_mask(mask)
        if clean_mask.sum() > 0:
            _, _, _ = predictor.add_new_mask(
                inference_state,
                frame_idx=idx,
                obj_id=ann_obj_id,
                mask=clean_mask,
            )
    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state, start_frame_idx=0):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).float()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
    for out_frame_idx in range(mask_list.shape[0]):
      if out_frame_idx % 2 == 0:
        p_masks_sam[offset + out_frame_idx] = video_segments[out_frame_idx][
            ann_obj_id
        ][0]

    # Process odd frames
    predictor.reset_state(inference_state)
    for idx, mask in enumerate(mask_list):
      if idx % 2 == 0:
        clean_mask = preprocess_mask(mask)
        
        if clean_mask.sum() > 0:
            _, _, _ = predictor.add_new_mask(
                inference_state,
                frame_idx=idx,
                obj_id=ann_obj_id,
                mask=clean_mask,
            )
    video_segments = {}
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(inference_state, start_frame_idx=0):
      video_segments[out_frame_idx] = {
          out_obj_id: (out_mask_logits[i] > 0.0).float()
          for i, out_obj_id in enumerate(out_obj_ids)
      }
    for out_frame_idx in range(mask_list.shape[0]):
      if out_frame_idx % 2 == 1:
        p_masks_sam[offset + out_frame_idx] = video_segments[out_frame_idx][
            ann_obj_id
        ][0]

def run_inference_on_video_file(model_path, video_path, output_dir, 
                               max_frames=None, sample_rate=1, sequence_length=8):
    """
    Run inference on a video file
    
    Args:
        model_path: Path to trained model
        video_path: Path to video file
        output_dir: Directory to save results
        max_frames: Maximum frames to process
        sample_rate: Frame sampling rate
        sequence_length: Number of frames to process at once
    """
    # Initialize inference
    inference = MotionSegmentationInference(model_path)
    
    # Load video frames
    print(f"Loading video from {video_path}")
    video_frames = inference.load_video_from_file(video_path, max_frames, sample_rate)
    
    if len(video_frames) == 0:
        print("No frames found in video")
        return
    
    print(f"Processing {len(video_frames)} frames")
    
    # Process video in chunks
    all_motion_masks = []
    
    for start_idx in tqdm(range(0, len(video_frames), sequence_length), desc="Processing chunks"):
        end_idx = min(start_idx + sequence_length, len(video_frames))
        chunk_frames = video_frames[start_idx:end_idx]
        
        # Predict motion mask for this chunk
        motion_masks = inference.predict_motion_mask(chunk_frames)
        use_sam_refine = True
        # Refine with SAM2 if enabled
        if use_sam_refine:
            print(f"Refining masks with SAM2 for frames {start_idx}-{end_idx}")
            
            # Get mask size
            mask_h, mask_w = motion_masks.shape[1], motion_masks.shape[2]
            
            # Convert PIL Images to tensor format and resize to mask size
            frame_arrays = []
            for pil_img in chunk_frames:
                # Resize PIL image to match mask dimensions
                pil_img_resized = pil_img.resize((mask_w, mask_h), Image.BILINEAR)
                # Convert to numpy array
                frame_array = np.array(pil_img_resized).astype(np.float32) / 255.0
                frame_arrays.append(frame_array)
            
            # Stack into tensor [T, H, W, C]
            frame_arrays = np.stack(frame_arrays, axis=0)
            frame_tensors = torch.from_numpy(frame_arrays).float()
            
            # Convert to [T, C, H, W] format
            frame_tensors = frame_tensors.permute(0, 3, 1, 2)
            
            # print(f"Frame tensors shape: {frame_tensors.shape}, Mask shape: {motion_masks.shape}")
            
            # Prepare SAM-refined masks with correct shape
            refined_masks = torch.zeros(
                motion_masks.shape[0], 
                motion_masks.shape[1], 
                motion_masks.shape[2],
                dtype=torch.float32
            )
            
            # Convert motion_masks to tensor for refine_sam
            mask_list = torch.from_numpy(motion_masks).float()
            
            # Apply SAM refinement
            refine_sam(frame_tensors, mask_list, refined_masks, offset=0)
            
            # Convert back to numpy
            motion_masks = refined_masks.cpu().numpy()
        
        all_motion_masks.append(motion_masks)
    
    # Concatenate all chunks
    all_motion_masks = np.concatenate(all_motion_masks, axis=0)
    
    # Save results
    sequence_name = os.path.splitext(os.path.basename(video_path))[0]
    # save_motion_masks(all_motion_masks, video_frames, output_dir, sequence_name)
    
    # Create visualizations
    visualize_motion_segmentation(video_frames, all_motion_masks, output_dir, sequence_name)
    
    return all_motion_masks


def save_motion_masks(motion_masks, original_images, save_dir, sequence_name):
    """
    Save motion masks as images
    
    Args:
        motion_masks: numpy array [S, H, W]
        original_images: List of original PIL images
        save_dir: Save directory
        sequence_name: Sequence name for filenames
    """
    # Create masks directory
    masks_dir = os.path.join(save_dir, f"{sequence_name}_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, (mask, orig_img) in enumerate(zip(motion_masks, original_images)):
        # Resize mask to original image size
        orig_w, orig_h = orig_img.size
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to 0-255 range
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        
        # Save as image
        mask_path = os.path.join(masks_dir, f'mask_{i:03d}.png')
        cv2.imwrite(mask_path, mask_uint8)
    
    print(f"Motion masks saved to {masks_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Motion Segmentation Inference (directory input only)'
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--pi3_model_path', type=str, default=None,
                        help='Path to pi3 .safetensors checkpoint. If omitted, uses PI3_MODEL_PATH env var.')
    parser.add_argument('--raft_model_path', type=str, default=None,
                        help='Path to RAFT checkpoint. If omitted, uses RAFT_MODEL_PATH env var.')
    parser.add_argument('--sam2_config_path', type=str, default=None,
                        help='Path to SAM2 config yaml. If omitted, uses SAM2_CONFIG_PATH env var.')
    parser.add_argument('--sam2_checkpoint_path', type=str, default=None,
                        help='Path to SAM2 checkpoint. If omitted, uses SAM2_CHECKPOINT_PATH env var.')

    # Single-sequence mode (one RGB frame directory).
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Path to a single rgb frame directory')

    # Multi-sequence mode (e.g., MOSE-0000 style folders).
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset root dir containing */rgb folders')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root directory to save results')

    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum number of frames to process')

    parser.add_argument('--sequence_length', type=int, default=32,
                        help='Number of frames to process at once')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binary threshold for visualization')

    args = parser.parse_args()
    if args.sam2_config_path:
        os.environ["SAM2_CONFIG_PATH"] = args.sam2_config_path
    if args.sam2_checkpoint_path:
        os.environ["SAM2_CHECKPOINT_PATH"] = args.sam2_checkpoint_path

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # Multi-sequence mode (equivalent to the original bash loop).
    # ============================================================
    inference = MotionSegmentationInference(
        model_path=args.model_path,
        pi3_model_path=args.pi3_model_path,
        raft_model_path=args.raft_model_path,
    )
    if args.dataset_dir is not None:
        print(f"Processing dataset: {args.dataset_dir}")

        rgb_dirs = sorted(
            # glob(os.path.join(args.dataset_dir, '*', 'rgb'))
            glob(os.path.join(args.dataset_dir, '*'))
        )
        # import pdb; pdb.set_trace()

        if len(rgb_dirs) == 0:
            raise RuntimeError(f"No rgb folders found in {args.dataset_dir}")

        for rgb_dir in rgb_dirs:
            if not os.listdir(rgb_dir):
                print(f"[Warning] Empty folder: {rgb_dir}, skipping.")
                continue

            # seq_name = os.path.basename(os.path.dirname(rgb_dir))
            seq_name = os.path.basename(os.path.normpath(rgb_dir))
            out_dir = os.path.join(args.output_dir, seq_name)
            # import pdb; pdb.set_trace()
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n=== Processing sequence: {seq_name} ===")
            print(f"Input : {rgb_dir}")
            print(f"Output: {out_dir}")

            motion_masks = run_inference_on_video_directory(
                inference=inference,
                video_dir=rgb_dir,
                output_dir=out_dir,
                max_frames=args.max_frames,
                sequence_length=args.sequence_length,
            )

            if motion_masks is not None:
                motion_ratio = np.mean(motion_masks > args.threshold)
                print(f"[{seq_name}] Motion pixel ratio: {motion_ratio:.2%}")

        print("Dataset inference completed.")
        return

    # ============================================================
    # Single-sequence mode (useful for debugging).
    # ============================================================
    if args.input_dir is not None:
        if not os.path.isdir(args.input_dir):
            raise ValueError(f"Invalid input_dir: {args.input_dir}")

        seq_name = os.path.basename(os.path.dirname(args.input_dir))
        out_dir = os.path.join(args.output_dir, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Processing single sequence: {seq_name}")
        print(f"Input : {args.input_dir}")
        print(f"Output: {out_dir}")

        motion_masks = run_inference_on_video_directory(
            inference=inference,
            video_dir=args.input_dir,
            output_dir=out_dir,
            max_frames=args.max_frames,
            sequence_length=args.sequence_length,
        )

        if motion_masks is not None:
            motion_ratio = np.mean(motion_masks > args.threshold)
            print(f"Motion pixel ratio: {motion_ratio:.2%}")

        print("Inference completed.")
        return

    # ============================================================
    # Argument fallback
    # ============================================================
    raise ValueError("You must specify either --dataset_dir or --input_dir")


def visualize_and_save(img_tensor, attn_map_tensor, title, filename, cmap='inferno', show=True):
    """
    将单帧图像和注意力图叠加显示，并保存到文件。
    （这个函数本身没有 bug，但调用它的地方需要修正）
    """
    H, W = img_tensor.shape[1:]
    
    img_vis = img_tensor.permute(1, 2, 0).numpy()
    # 【改进】 标准化图像以便显示，防止 clipping 警告
    min_val, max_val = img_vis.min(), img_vis.max()
    if max_val > 1.0: # 假设值在 0-255
        img_vis = (img_vis).astype(np.uint8)
    else: # 假设值在 0-1
        img_vis = ((img_vis - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    attn_map_resized = F.interpolate(
        attn_map_tensor.unsqueeze(0).unsqueeze(0), 
        size=(H, W), 
        mode='bilinear', 
        align_corners=False
    ).squeeze().numpy()

    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.imshow(img_vis)
    plt.imshow(attn_map_resized, cmap=cmap, alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.colorbar()
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"🖼️  Saved visualization to: {filename}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def visualize_images_and_motion(images, motion_prior, step=0, save_dir="debug_vis", max_frames=16):
    """
    可视化输入图像和对应的 motion prior

    Args:
        images: [B, N, C, H, W] (已归一化，需要反归一化才能显示)
        motion_prior: [B, N, H, W], 0~1
        step: 当前训练 step，用于文件命名
        save_dir: 保存路径
        max_frames: 每个 batch 只保存前多少帧，避免太多图
    """
    os.makedirs(save_dir, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    B, N, C, H, W = images.shape

    for b in range(min(1, B)):  # 只保存第一个 batch，避免太多图
        for n in range(min(N, max_frames)):
            # ---------------------
            # 反归一化图像
            # ---------------------
            img = images[b, n].detach().cpu() * std + mean
            img = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            # ---------------------
            # motion prior
            # ---------------------
            mp = motion_prior[b, n].detach().cpu().numpy()  # [H, W]

            # ---------------------
            # 可视化
            # ---------------------
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img)
            axs[0].set_title("Input Image")
            axs[0].axis("off")

            im = axs[1].imshow(mp, cmap="viridis")
            axs[1].set_title("Motion Prior")
            axs[1].axis("off")
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

            save_path = os.path.join(save_dir, f"step{step}_b{b}_f{n}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[保存可视化] {save_path}")

def visualize_optical_flow(flow_magnitudes: torch.Tensor, 
                           video_tensor: torch.Tensor, 
                           device, 
                           alpha: float = 0.5, 
                           colormap: str = 'jet',
                           percentile_clip: float = 95.0,
                           gamma: float = 0.5) -> torch.Tensor:
    """
    为视频批次中的每一帧生成光流的可视化图像,并将其返回。
    
    Args:
        flow_magnitudes: [B, N, H, W] 光流幅度
        video_tensor: [B, N, C, H, W] 视频tensor (经过ImageNet标准化的)
        device: 设备
        alpha: 透明度值,范围 [0, 1]
        colormap: 使用的颜色映射方式
        percentile_clip: 百分位裁剪值,用于压缩动态范围 (建议85-98)
        gamma: gamma校正值,用于非线性压缩 (建议0.3-0.7,越小压缩越多)
    Returns:
        vis_images: [B, N, H, W, 3] 可视化的图像
    """
    B, N, H, W = flow_magnitudes.shape
    vis_images = torch.zeros(B, N, H, W, 3, device=device)  # 存储可视化图像,RGB通道

    # 反标准化视频
    video_denorm = denormalize_imagenet(video_tensor)  # [B, N, C, H, W]
    
    for b in range(B):
        for t in range(N):
            flow_magnitude = flow_magnitudes[b, t]  # [H, W]
            frame = video_denorm[b, t]  # [C, H, W], [0,1]

            # ===== 方法1: 百分位裁剪，压缩动态范围 =====
            # 计算百分位值来裁剪极端值
            flow_min = flow_magnitude.min()
            flow_max = torch.quantile(flow_magnitude.flatten(), percentile_clip / 100.0)
            
            # 裁剪并归一化
            flow_magnitude = torch.clamp(flow_magnitude, flow_min, flow_max)
            flow_magnitude = (flow_magnitude - flow_min) / (flow_max - flow_min + 1e-8)
            
            # ===== 方法2: Gamma校正，进行非线性压缩 =====
            # gamma < 1 会压缩高值区域，使前景不那么突出
            flow_magnitude = torch.pow(flow_magnitude, gamma)
            
            # 映射到0-255
            flow_magnitude = (flow_magnitude * 255).byte().cpu().numpy()  # [H, W]
            
            # 应用颜色映射
            flow_color = cv2.applyColorMap(flow_magnitude, cv2.COLORMAP_JET)  # [H, W, 3]
            
            # 将颜色映射与视频帧叠加
            frame = (frame.permute(1, 2, 0) * 255).byte().cpu().numpy()  # [H, W, C]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转为BGR格式
            
            # 使用透明度将光流可视化和原视频帧进行融合
            vis_frame = cv2.addWeighted(frame, 1 - alpha, flow_color, alpha, 0)
            
            # 将结果保存到vis_images
            vis_images[b, t] = torch.from_numpy(vis_frame).to(device)
    
    return vis_images

def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    反标准化ImageNet预处理的图像
    Args:
        tensor: [B, N, C, H, W] 经过ImageNet标准化的tensor
    Returns:
        tensor: [B, N, C, H, W] 反标准化后的tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)
    
    return tensor * std[:, None, None] + mean[:, None, None]


if __name__ == "__main__":
    main()
