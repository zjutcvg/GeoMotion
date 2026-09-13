import torch
import torch.nn.functional as F_nn
import torchvision.transforms.functional as F_vision
from torchvision.models.optical_flow.raft import make_coords_grid

import motion_seg_inference as msi


def compute_raft_hidden_batch(video_tensor: torch.Tensor, device,
                              bidirectional: bool = True,
                              fusion_method: str = 'mean',
                              num_flow_updates: int = 12) -> torch.Tensor:
    """
    替代 compute_optical_flow_magnitude_batch():
    对 batch 视频，每帧取其相邻帧对的 RAFT hidden_state（128-ch, H/8 x W/8）

    Args:
        video_tensor: [B, N, C, H, W] ImageNet 归一化的视频 tensor
        device: 计算设备
        bidirectional: 是否使用双向光流 hidden_state
        fusion_method: 'mean' 或 'max'
        num_flow_updates: RAFT GRU 迭代次数（默认12）

    Returns:
        raft_hidden_states: [B, N, 128, H_8, W_8]
    """
    if msi.RAFT_MODEL is None:
        print("RAFT model not initialized, returning zero hidden states")
        B, N, _, H, W = video_tensor.shape
        H_8 = ((H + 7) // 8)
        W_8 = ((W + 7) // 8)
        return torch.zeros(B, N, 128, H_8, W_8, device=device)

    B, N, C, H, W = video_tensor.shape

    # 反标准化到 [0,1] 范围（RAFT 的输入范围）
    video_denorm = msi.denormalize_imagenet(video_tensor)

    # RAFT 要求 H,W 是 8 的倍数
    raft_H = ((H + 7) // 8) * 8
    raft_W = ((W + 7) // 8) * 8
    H_8 = raft_H // 8
    W_8 = raft_W // 8

    raft_hidden_states = torch.zeros(B, N, 128, H_8, W_8, device=device)

    if not bidirectional:
        # 单向（仅前向）
        for b in range(B):
            for t in range(N - 1):
                hidden = _run_raft_get_hidden(
                    video_denorm[b, t], video_denorm[b, t + 1],
                    H, W, raft_H, raft_W, num_flow_updates,
                )
                raft_hidden_states[b, t + 1] = hidden
            # 第一帧用第二帧的 hidden_state
            if N > 1:
                raft_hidden_states[b, 0] = raft_hidden_states[b, 1].clone()
    else:
        # 双向光流
        for b in range(B):
            forward_hiddens = []
            backward_hiddens = []

            # 先算所有相邻对的 hidden_state
            for t in range(N - 1):
                # 前向：frame[t] -> frame[t+1]
                hidden_fwd = _run_raft_get_hidden(
                    video_denorm[b, t], video_denorm[b, t + 1],
                    H, W, raft_H, raft_W, num_flow_updates,
                )
                forward_hiddens.append(hidden_fwd)  # [1, 128, H/8, W/8]

                # 后向：frame[t+1] -> frame[t]
                hidden_bwd = _run_raft_get_hidden(
                    video_denorm[b, t + 1], video_denorm[b, t],
                    H, W, raft_H, raft_W, num_flow_updates,
                )
                backward_hiddens.append(hidden_bwd)

            # 为每一帧分配融合后的 hidden_state
            for t in range(N):
                hiddens_for_frame = []

                if t > 0:
                    hiddens_for_frame.append(backward_hiddens[t - 1])  # (t-1)->t 的后向
                if t < N - 1:
                    hiddens_for_frame.append(forward_hiddens[t])  # t->(t+1) 的前向

                if len(hiddens_for_frame) == 1:
                    raft_hidden_states[b, t] = hiddens_for_frame[0]
                elif len(hiddens_for_frame) == 2:
                    raft_hidden_states[b, t] = _fuse_hidden_states(
                        hiddens_for_frame, fusion_method
                    )

    return raft_hidden_states


def _run_raft_get_hidden(frame1: torch.Tensor, frame2: torch.Tensor,
                         orig_H: int, orig_W: int,
                         raft_H: int, raft_W: int,
                         num_flow_updates: int = 12) -> torch.Tensor:
    """
    运行 RAFT 子模块，返回最终 hidden_state [128, H/8, W/8]

    Args:
        frame1: [3, H, W] or [1, 3, H, W] 范围 [0,1]
        frame2: [3, H, W] or [1, 3, H, W] 范围 [0,1]
        orig_H, orig_W: 原始分辨率
        raft_H, raft_W: 对齐到8的倍数后的分辨率
        num_flow_updates: GRU 迭代步数
    Returns:
        hidden_state: [128, H_8, W_8]
    """
    if frame1.dim() == 3:
        frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3:
        frame2 = frame2.unsqueeze(0)

    # resize 到 8 的倍数
    img1_resized = F_vision.resize(frame1, size=[raft_H, raft_W], antialias=False)
    img2_resized = F_vision.resize(frame2, size=[raft_H, raft_W], antialias=False)

    # RAFT 预处理变换（输入范围 0~1 → -1~1）
    img1_proc, img2_proc = msi.RAFT_TRANSFORMS(img1_resized, img2_resized)

    with torch.no_grad():
        # 1. Feature encoder
        fmaps = msi.RAFT_MODEL.feature_encoder(torch.cat([img1_proc, img2_proc], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)

        # 2. Correlation pyramid
        msi.RAFT_MODEL.corr_block.build_pyramid(fmap1, fmap2)

        # 3. Context encoder -> split
        context_out = msi.RAFT_MODEL.context_encoder(img1_proc)
        hidden_state, context = torch.split(context_out, [128, 128], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F_nn.relu(context)

        # 4. GRU iterative updates
        B_hid, C_hid, h, w = hidden_state.shape
        coords0 = make_coords_grid(B_hid, h, w).to(fmap1.device)
        coords1 = coords0.clone()

        for _ in range(num_flow_updates):
            corr_features = msi.RAFT_MODEL.corr_block.index_pyramid(centroids_coords=coords1)
            flow = coords1 - coords0
            hidden_state, delta_flow = msi.RAFT_MODEL.update_block(
                hidden_state, context, corr_features, flow
            )
            coords1 = coords1 + delta_flow

    return hidden_state.squeeze(0)  # [128, H/8, W/8]


def _fuse_hidden_states(hidden_list: list, method: str = 'mean') -> torch.Tensor:
    """融合两个 hidden_state"""
    stacked = torch.stack(hidden_list, dim=0)  # [2, 128, H/8, W/8]
    if method == 'max':
        return stacked.max(dim=0)[0]
    elif method == 'mean':
        return stacked.mean(dim=0)
    elif method == 'sum':
        return stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")


def compute_optical_flow_with_hidden(frame1: torch.Tensor, frame2: torch.Tensor) -> tuple:
    """
    同时返回光流和 hidden_state（用于需要两者的情况）
    Returns:
        flow: [2, H, W] 光流场
        hidden_state: [128, H/8, W/8]
    """
    import torchvision.transforms.functional as F

    orig_H, orig_W = frame1.shape[1], frame1.shape[2]

    if frame1.dim() == 3:
        frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3:
        frame2 = frame2.unsqueeze(0)

    H, W = frame1.shape[2], frame1.shape[3]
    raft_H = ((H + 7) // 8) * 8
    raft_W = ((W + 7) // 8) * 8

    img1_resized = F.resize(frame1, size=[raft_H, raft_W], antialias=False)
    img2_resized = F.resize(frame2, size=[raft_H, raft_W], antialias=False)

    img1_proc, img2_proc = msi.RAFT_TRANSFORMS(img1_resized, img2_resized)

    with torch.no_grad():
        # Feature encoder
        fmaps = msi.RAFT_MODEL.feature_encoder(torch.cat([img1_proc, img2_proc], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)

        # Correlation pyramid
        msi.RAFT_MODEL.corr_block.build_pyramid(fmap1, fmap2)

        # Context encoder
        context_out = msi.RAFT_MODEL.context_encoder(img1_proc)
        hidden_state, context = torch.split(context_out, [128, 128], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F_nn.relu(context)

        # GRU iterations
        _, _, h, w = hidden_state.shape
        coords0 = make_coords_grid(1, h, w).to(fmap1.device)
        coords1 = coords0.clone()
        flow_predictions = []

        for _ in range(12):
            corr_features = msi.RAFT_MODEL.corr_block.index_pyramid(centroids_coords=coords1)
            flow = coords1 - coords0
            hidden_state, delta_flow = msi.RAFT_MODEL.update_block(
                hidden_state, context, corr_features, flow
            )
            coords1 = coords1 + delta_flow

            up_mask = None if msi.RAFT_MODEL.mask_predictor is None else msi.RAFT_MODEL.mask_predictor(hidden_state)
            upsampled_flow = _upsample_flow_no_self(flow=(coords1 - coords0), up_mask=up_mask)
            flow_predictions.append(upsampled_flow)

        flow_final = flow_predictions[-1]

    # 恢复到原始分辨率
    if flow_final.shape[2] != orig_H or flow_final.shape[3] != orig_W:
        flow_final = F.resize(flow_final, size=[orig_H, orig_W], antialias=False)
        scale_W = orig_W / raft_W
        scale_H = orig_H / raft_H
        flow_final[:, 0] *= scale_W
        flow_final[:, 1] *= scale_H

    return flow_final.squeeze(0), hidden_state.squeeze(0)


def _upsample_flow_no_self(flow, up_mask=None, factor=8):
    """从 torchvision RAFT 模块引用 upsample_flow（避免使用 self）"""
    from torchvision.models.optical_flow._utils import upsample_flow
    return upsample_flow(flow, up_mask, factor)
