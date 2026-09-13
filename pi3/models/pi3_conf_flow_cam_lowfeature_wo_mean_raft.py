import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
import os

import torch.nn.functional as F
from typing import Optional, Dict, Any

# import pdb;pdb.set_trace()
from .dinov2.layers import Mlp
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from ..utils.geometry import homogenize_points, depth_edge

# Re-export everything from the original module so the raft variant
# can be imported with the same interface.
# The only difference is MotionAwareDecoder and Pi3MotionSeg below.
from .pi3_conf_flow_cam_lowfeature_wo_mean import (
    CombinedLoss,
    compute_iou,
    compute_metrics,
    analyze_motion_consistency,
    example_training_step,
    visualize_flows,
    visualize_imgs_and_hidden,
)


class MotionAwareDecoder(nn.Module):
    """
    RAFT-feature version of MotionAwareDecoder.

    Replaces the flow_encoder (3-layer CNN on magnitude, 1→128) with a
    lightweight raft_feat_adapter that takes the RAFT hidden_state
    (128-ch, H/8×W/8) directly.

    Supports both modes:
      - new: raft_features [B, N, 128, H/8, W/8]
      - old: optical_flow  [B, N, H, W]     (fallback, kept for backward compat)
    """

    def __init__(self, hidden_dim=1024, patch_size=14, raft_feat_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # ===== Feature projection (unchanged) =====
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2048),
            nn.ReLU(inplace=True),
        )

        # ===== RAFT hidden_state adapter (new) =====
        # hidden_state is 128-ch, H/8 × W/8 — only a light refine needed
        self.raft_feat_adapter = nn.Sequential(
            nn.Conv2d(raft_feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ===== Fusion projection (unchanged) =====
        self.fusion_proj = nn.Linear(2048 + 128 + 512, 2048)

    def forward(self, low_hidden, hidden, optical_flow=None, raft_features=None,
                camera_hidden=None, patch_start_idx=None, H=None, W=None, B=None, N=None):
        """
        Args:
            low_hidden:  [B*N, hw, 2*C]  layer-5 + layer-15 features
            hidden:      [B*N, hw, 2*C]  last two layers
            optical_flow:[B, N, H, W]     magnitude (legacy mode)
            raft_features:[B, N, 128, H/8, W/8]  RAFT hidden_state (new mode)
            camera_hidden:[B*N, hw, 512]
        """
        # 1. Patch extraction
        low_hidden_patch = low_hidden[:, patch_start_idx:]   # [B*N, num_patches, 2*C]
        hidden_patch = hidden[:, patch_start_idx:]           # [B*N, num_patches, 2*C]
        camera_patch = camera_hidden[:, patch_start_idx:]    # [B*N, num_patches, 512]

        # 2. Concat low+high → project to 2048
        combined_hidden = torch.cat([low_hidden_patch, hidden_patch], dim=-1)
        projected_hidden = self.feature_proj(combined_hidden)  # [B*N, num_patches, 2048]

        # 3. Motion feature extraction
        if raft_features is not None:
            # New mode: use RAFT hidden_state directly
            B_raft, N_raft, C_raft, H_8, W_8 = raft_features.shape
            flow_feat = self.raft_feat_adapter(
                raft_features.view(B_raft * N_raft, C_raft, H_8, W_8)
            )  # [B*N, 128, H/8, W/8]
        elif optical_flow is not None:
            # Legacy mode: use optical-flow magnitude + CNN encoder
            # (flow_encoder removed in raft version, recompute here if needed)
            flow_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ).to(raft_features.device if raft_features is not None else optical_flow.device)
            flow_feat = flow_encoder(optical_flow.view(B*N, 1, H, W))
        else:
            raise ValueError("Must provide either raft_features or optical_flow")

        # 4. Downsample to patch resolution
        flow_feat_down = F.interpolate(
            flow_feat,
            size=(H // self.patch_size, W // self.patch_size),
            mode='bilinear',
            align_corners=False,
        )  # [B*N, 128, patch_h, patch_w]

        num_patches = flow_feat_down.shape[2] * flow_feat_down.shape[3]
        flow_patches = flow_feat_down.permute(0, 2, 3, 1).reshape(B*N, num_patches, 128)

        # 5. Fusion
        combined = torch.cat([
            projected_hidden,
            flow_patches,
            camera_patch,
        ], dim=-1)  # [B*N, num_patches, 2688]
        fused_patch = self.fusion_proj(combined)  # [B*N, num_patches, 2048]

        # 6. Special tokens
        special_tokens = torch.cat([
            low_hidden[:, :patch_start_idx],
            hidden[:, :patch_start_idx],
        ], dim=-1)
        special_tokens_proj = self.feature_proj(special_tokens)  # [B*N, ps, 2048]

        enhanced_hidden = torch.cat([
            special_tokens_proj,
            fused_patch,
        ], dim=1)  # [B*N, hw, 2048]

        return enhanced_hidden


class Pi3MotionSeg(nn.Module, PyTorchModelHubMixin):
    """
    Pi3 model for motion segmentation — RAFT-hidden-state variant.
    Same as the original but uses MotionAwareDecoder above.
    """

    def __init__(
        self,
        pos_type='rope100',
        decoder_size='large',
        pi3_model_path: Optional[str] = None,
        freeze_backbone: bool = True,
        motion_feature_dim: int = 64,
        raft_feat_dim: int = 128,
    ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope,
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        self.motion_aware_decoder = MotionAwareDecoder(
            hidden_dim=2 * self.dec_embed_dim,
            patch_size=self.patch_size,
            raft_feat_dim=raft_feat_dim,
        )

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False,
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # Load pretrained Pi3 model if provided
        if pi3_model_path:
            self._load_pretrained_pi3(pi3_model_path)

        # Freeze backbone if specified
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_pi3(self, model_path: str):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('mask_') and not k.startswith('motion_')
        }
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded pretrained Pi3 model from {model_path}")

    def _freeze_backbone(self):
        components_to_freeze = [
            'encoder', 'decoder', 'register_token', 'position_getter',
            'point_decoder', 'point_head', 'camera_decoder', 'camera_head',
        ]
        for name, param in self.named_parameters():
            for component in components_to_freeze:
                if name.startswith(component):
                    param.requires_grad = False
                    break

    def decode(self, hidden, N, H, W, imgs):
        """Decoder function adapted from Pi3"""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        selected_layers = [5]
        selected_layers_2 = [15]
        selected_feats = []
        selected_feats_2 = []

        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i in selected_layers:
                if i % 2 == 1:
                    hidden_temp = hidden.reshape(B, N, hw, -1)
                    feat = hidden_temp.reshape(B * N, hw, -1)
                else:
                    feat = hidden
                selected_feats.append(feat)

            if i in selected_layers_2:
                if i % 2 == 1:
                    hidden_temp = hidden.reshape(B, N, hw, -1)
                    feat = hidden_temp.reshape(B * N, hw, -1)
                else:
                    feat = hidden
                selected_feats_2.append(feat)

            if i >= len(self.decoder) - 2:
                final_output.append(hidden.reshape(B * N, hw, -1))

        fused_feat_5 = torch.stack(selected_feats, dim=0).mean(dim=0) if len(selected_feats) > 0 else None
        fused_feat_15 = torch.stack(selected_feats_2, dim=0).mean(dim=0) if len(selected_feats_2) > 0 else None

        if fused_feat_5 is not None and fused_feat_15 is not None:
            cat_output = torch.cat([fused_feat_5, fused_feat_15], dim=-1)
        else:
            cat_output = final_output[-1]

        last_two_layers = torch.cat(final_output, dim=-1)
        return cat_output, last_two_layers, pos.reshape(B * N, hw, -1)

    def forward(self, imgs, flows=None, raft_features=None):
        """
        Forward pass for motion segmentation.

        Args:
            imgs: [B, N, 3, H, W] or [N, 3, H, W]
            flows: [B, N, H, W] optical-flow magnitude (legacy mode)
            raft_features: [B, N, 128, H/8, W/8] RAFT hidden_state (new mode)

        Returns:
            Dict with motion_mask, points, conf, camera_poses
        """
        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(0)
            if flows is not None and len(flows.shape) == 3:
                flows = flows.unsqueeze(0)

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        imgs_reshaped = imgs.reshape(B * N, _, H, W)

        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.encoder(imgs_reshaped, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                low_hidden, hidden, pos = self.decode(hidden, N, H, W, imgs)
        else:
            hidden = self.encoder(imgs_reshaped, is_training=True)
            if isinstance(hidden, dict):
                hidden = hidden["x_norm_patchtokens"]
            low_hidden, hidden, pos = self.decode(hidden, N, H, W)

        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        has_motion_fusion = hasattr(self, 'motion_aware_decoder')
        if raft_features is not None and has_motion_fusion:
            hidden_with_motion = self.motion_aware_decoder(
                low_hidden=low_hidden,
                hidden=hidden,
                raft_features=raft_features,
                camera_hidden=camera_hidden,
                patch_start_idx=self.patch_start_idx,
                H=H, W=W,
                B=B, N=N,
            )
        elif flows is not None and has_motion_fusion:
            hidden_with_motion = self.motion_aware_decoder(
                low_hidden=low_hidden,
                hidden=hidden,
                optical_flow=flows,
                camera_hidden=camera_hidden,
                patch_start_idx=self.patch_start_idx,
                H=H, W=W,
                B=B, N=N,
            )
        else:
            hidden_with_motion = hidden

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden_with_motion, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            motion_mask = torch.sigmoid(conf).squeeze(-1)

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            motion_mask=motion_mask,
        )


# Factory function
def create_pi3_motion_segmentation_model(
    pi3_model_path: Optional[str] = None,
    pos_type: str = 'rope100',
    decoder_size: str = 'large',
    freeze_backbone: bool = True,
    motion_feature_dim: int = 64,
    raft_feat_dim: int = 128,
):
    """
    Factory function to create Pi3-based motion segmentation model
    that accepts RAFT hidden_state features.
    """
    model = Pi3MotionSeg(
        pos_type=pos_type,
        decoder_size=decoder_size,
        pi3_model_path=pi3_model_path,
        freeze_backbone=freeze_backbone,
        motion_feature_dim=motion_feature_dim,
        raft_feat_dim=raft_feat_dim,
    )
    return model
