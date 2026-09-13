from __future__ import annotations

import io
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass(frozen=True)
class MultiInstanceRefineConfig:
    prior_threshold: float = 0.8
    num_prompt_points: int = 8
    prompt_point_min_distance_ratio: float = 0.08
    prompt_box_expand_ratio: float = 0.05
    use_box_prompt: bool = False
    use_box_with_points: bool = False
    multimask_output: bool = True
    max_proposals_per_frame: int = 8
    min_mask_pixels: int = 48
    min_overlap_with_prior: float = 0.35
    min_prior_coverage: float = 0.02
    proposal_nms_iou: float = 0.75
    proposal_merge_mask_iou: float = 0.15
    proposal_merge_box_iou: float = 0.1
    proposal_merge_max_gap: float = 0.04
    proposal_merge_min_appearance: float = 0.35
    max_track_age: int = 4
    min_match_score: float = 0.25
    min_track_length: int = 2
    max_tracks: int = 16
    track_recent_history: int = 3
    track_relink_max_frame_gap: int = 4
    track_relink_min_appearance: float = 0.55
    track_relink_min_center: float = 0.45
    single_mask_threshold: float = 0.8
    single_max_anchor_frames: int = 5
    single_anchor_min_area_ratio: float = 0.45
    single_anchor_min_score_ratio: float = 0.6
    single_anchor_time_diversity_weight: float = 0.35
    fusion_distance_temperature: float = 3.0
    max_anchor_frames: int = 5 # 3
    anchor_min_area_ratio: float = 0.45
    anchor_min_score_ratio: float = 0.6
    anchor_time_diversity_weight: float = 0.35
    weight_mask_iou: float = 0.45
    weight_box_iou: float = 0.2
    weight_center: float = 0.15
    weight_appearance: float = 0.2
    center_sigma: float = 0.15


@dataclass(frozen=True)
class Proposal:
    frame_idx: int
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    area: int
    centroid_xy: Tuple[float, float]
    appearance: np.ndarray
    overlap_with_prior: float
    prior_coverage: float


def preprocess_mask(mask_tensor: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
    return mask_tensor > threshold


def refine_sam(
    frame_tensors: torch.Tensor,
    mask_list: torch.Tensor | np.ndarray,
    p_masks_sam: torch.Tensor,
    offset: int = 0,
    predictor: Any | None = None,
    config: MultiInstanceRefineConfig | None = None,
) -> None:
    if predictor is None:
        raise ValueError("predictor must be provided for single-mask refinement.")

    cfg = config or MultiInstanceRefineConfig()
    if isinstance(mask_list, torch.Tensor):
        mask_tensor = mask_list.detach().cpu().float()
    else:
        mask_tensor = torch.from_numpy(np.asarray(mask_list)).float()

    frame_count = int(mask_tensor.shape[0])
    if frame_count == 0:
        return

    anchor_indices = _select_single_mask_anchors(mask_tensor, cfg)
    if not anchor_indices:
        anchor_indices = [idx for idx in range(frame_count) if preprocess_mask(mask_tensor[idx], cfg.single_mask_threshold).sum() > 0]
    if not anchor_indices:
        return

    frames_rgb = _tensor_frames_to_numpy(frame_tensors)
    tmp_dir = _write_frames_to_tempdir(frames_rgb)
    try:
        with torch.inference_mode(), _sam2_inference_context():
            forward_state = predictor.init_state(tmp_dir)
            _inject_single_mask_anchors(forward_state, predictor, mask_tensor, anchor_indices, cfg)
            forward_outputs = _propagate_all(predictor, forward_state, reverse=False)

            reverse_state = predictor.init_state(tmp_dir)
            _inject_single_mask_anchors(reverse_state, predictor, mask_tensor, anchor_indices, cfg)
            reverse_outputs = _propagate_all(predictor, reverse_state, reverse=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    merged_masks = _merge_directional_logits(
        frame_count=frame_count,
        obj_ids=[1],
        forward_outputs=forward_outputs,
        reverse_outputs=reverse_outputs,
        anchor_frames_by_obj={1: anchor_indices},
        cfg=cfg,
    )
    for frame_idx in range(frame_count):
        p_masks_sam[offset + frame_idx] = merged_masks[frame_idx, 0]


def refine_sam_multi_instance(
    frame_tensors: torch.Tensor,
    mask_list: torch.Tensor | np.ndarray,
    predictor: Any,
    config: MultiInstanceRefineConfig | None = None,
    raw_proposals_vis_dir: str | None = None,
    image_predictor_model: Any | None = None,
    component_labels_per_frame: Sequence[np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Refine binary motion priors as multiple frame-wise SAM2 proposals.

    Args:
        frame_tensors: [T, 3, H, W] in [0, 1].
        mask_list: [T, H, W] motion priors.
        predictor: unused in the current frame-wise merged-only path. Kept for API compatibility.
        config: optional tuning config.
        component_labels_per_frame: optional pre-computed connected-component label
            arrays, one per frame ([H, W] int). Significant components (area >=
            min_component_ratio * max_component_area) each get prompt points.
    """
    cfg = config or MultiInstanceRefineConfig()
    frames_rgb = _tensor_frames_to_numpy(frame_tensors)
    binary_priors = _prepare_binary_priors(mask_list, cfg.prior_threshold)
    component_labels_per_frame = list(component_labels_per_frame) if component_labels_per_frame is not None else [None] * len(frames_rgb)

    with torch.inference_mode(), _sam2_inference_context():
        image_predictor = SAM2ImagePredictor(image_predictor_model or predictor)

        proposals_by_frame = [
            _generate_frame_proposals(
                frame,
                prior,
                frame_idx,
                image_predictor,
                cfg,
                raw_proposals_vis_dir=raw_proposals_vis_dir,
                component_labels=comp_labels,
            )
            for frame_idx, (frame, prior, comp_labels) in enumerate(
                zip(frames_rgb, binary_priors, component_labels_per_frame)
            )
        ]
    proposal_instance_masks, proposal_merged_masks = _build_framewise_merged_masks(
        proposals_by_frame=proposals_by_frame,
        frame_count=len(frames_rgb),
        frame_shape=binary_priors.shape[1:],
    )

    return {
        "instance_masks": proposal_instance_masks,
        "merged_masks": proposal_merged_masks,
        "proposal_instance_masks": proposal_instance_masks,
        "proposal_merged_masks": proposal_merged_masks,
        "track_metadata": [],
        "proposals_per_frame": [len(p) for p in proposals_by_frame],
    }


def _build_framewise_merged_masks(
    proposals_by_frame: Sequence[Sequence[Proposal]],
    frame_count: int,
    frame_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_instances = max((len(proposals) for proposals in proposals_by_frame), default=0)
    height, width = frame_shape
    instance_masks = torch.zeros(
        (frame_count, max_instances, height, width),
        dtype=torch.float32,
    )

    for frame_idx, proposals in enumerate(proposals_by_frame):
        for proposal_idx, proposal in enumerate(proposals):
            instance_masks[frame_idx, proposal_idx] = torch.from_numpy(
                proposal.mask.astype(np.float32)
            )

    merged_masks = (instance_masks.sum(dim=1) > 0).float()
    return instance_masks, merged_masks


def _tensor_frames_to_numpy(frame_tensors: torch.Tensor) -> List[np.ndarray]:
    frames_rgb: List[np.ndarray] = []
    for frame_tensor in frame_tensors:
        frame = frame_tensor.detach().permute(1, 2, 0).cpu().numpy()
        frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frames_rgb.append(frame_uint8)
    return frames_rgb


def _prepare_binary_priors(
    mask_list: torch.Tensor | np.ndarray,
    threshold: float,
) -> np.ndarray:
    if isinstance(mask_list, torch.Tensor):
        priors = mask_list.detach().cpu().numpy()
    else:
        priors = np.asarray(mask_list)
    return priors > threshold


def _generate_frame_proposals(
    frame_rgb: np.ndarray,
    prior_mask: np.ndarray,
    frame_idx: int,
    image_predictor: SAM2ImagePredictor,
    cfg: MultiInstanceRefineConfig,
    raw_proposals_vis_dir: str | None = None,
    component_labels: np.ndarray | None = None,
) -> List[Proposal]:
    if int(prior_mask.sum()) < cfg.min_mask_pixels:
        return []

    with torch.inference_mode(), _sam2_inference_context():
        raw_candidates = _generate_prompt_candidates(
            frame_rgb=frame_rgb,
            prior_mask=prior_mask,
            image_predictor=image_predictor,
            cfg=cfg,
            component_labels=component_labels,
        )
    proposals: List[Proposal] = []
    prior_area = max(int(prior_mask.sum()), 1)
    for seg, proposal_score in raw_candidates:
        seg = seg.astype(bool)
        area = int(seg.sum())
        if area < cfg.min_mask_pixels:
            continue

        overlap = seg & prior_mask
        overlap_area = int(overlap.sum())
        if overlap_area < cfg.min_mask_pixels:
            continue

        overlap_with_prior = overlap_area / float(area)
        prior_coverage = overlap_area / float(prior_area)
        if overlap_with_prior < cfg.min_overlap_with_prior:
            continue
        if prior_coverage < cfg.min_prior_coverage:
            continue

        bbox = _mask_to_box(seg)
        centroid = _mask_centroid(seg)
        appearance = _extract_appearance_feature(frame_rgb, seg)
        score = float(proposal_score * (0.4 + 0.6 * prior_coverage))
        proposals.append(
            Proposal(
                frame_idx=frame_idx,
                mask=seg,
                bbox_xyxy=bbox,
                score=score,
                area=area,
                centroid_xy=centroid,
                appearance=appearance,
                overlap_with_prior=overlap_with_prior,
                prior_coverage=prior_coverage,
            )
        )

    proposals.sort(key=lambda p: p.score, reverse=True)
    merged_proposals = _merge_proposals_to_objects(
        frame_rgb=frame_rgb,
        prior_mask=prior_mask,
        proposals=proposals,
        frame_idx=frame_idx,
        cfg=cfg,
    )
    # import pdb;pdb.set_trace()
    if raw_proposals_vis_dir is not None:
        _save_proposal_debug_visualization(
            frame_rgb=frame_rgb,
            prior_mask=prior_mask,
            raw_proposals=[
                Proposal(
                    frame_idx=frame_idx,
                    mask=seg.astype(bool),
                    bbox_xyxy=_mask_to_box(seg.astype(bool)),
                    score=float(score),
                    area=int(seg.sum()),
                    centroid_xy=_mask_centroid(seg.astype(bool)),
                    appearance=_extract_appearance_feature(frame_rgb, seg.astype(bool)),
                    overlap_with_prior=0.0,
                    prior_coverage=0.0,
                )
                for seg, score in raw_candidates
                if int(seg.sum()) >= cfg.min_mask_pixels
            ],
            filtered_proposals=proposals,
            merged_proposals=merged_proposals,
            save_dir=raw_proposals_vis_dir,
            frame_idx=frame_idx,
        )
    return _deduplicate_proposals(merged_proposals, cfg)


def _save_proposal_debug_visualization(
    frame_rgb: np.ndarray,
    prior_mask: np.ndarray,
    raw_proposals: Sequence[Proposal],
    filtered_proposals: Sequence[Proposal],
    merged_proposals: Sequence[Proposal],
    save_dir: str,
    frame_idx: int,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    original = frame_rgb.astype(np.uint8)
    prior_overlay = _apply_binary_overlay(original, prior_mask, (255, 0, 0), 0.45)
    raw_overlay = _apply_proposal_overlay(original, raw_proposals)
    merged_overlay = _apply_proposal_overlay(original, merged_proposals)
    filtered_overlay = _apply_proposal_overlay(original, filtered_proposals)

    tiles = [
        _annotate_tile(original, f"frame {frame_idx:04d}"),
        _annotate_tile(prior_overlay, "coarse prior"),
        _annotate_tile(raw_overlay, f"raw proposals {len(raw_proposals)}"),
        _annotate_tile(filtered_overlay, f"filtered {len(filtered_proposals)}"),
        _annotate_tile(merged_overlay, f"merged {len(merged_proposals)}"),
    ]
    canvas = np.concatenate(tiles, axis=1)
    out_path = os.path.join(save_dir, f"frame_{frame_idx:05d}_proposal_debug.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _apply_binary_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    if mask_bool.any():
        overlay[mask_bool] = (
            overlay[mask_bool] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_proposal_overlay(
    image_rgb: np.ndarray,
    proposals: Sequence[Proposal],
) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    for proposal_idx, proposal in enumerate(proposals):
        mask = proposal.mask.astype(bool)
        if not mask.any():
            continue
        color = _proposal_color(proposal_idx)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _proposal_color(index: int) -> np.ndarray:
    palette = np.array(
        [
            [255, 99, 71],
            [135, 206, 235],
            [60, 179, 113],
            [255, 215, 0],
            [186, 85, 211],
            [255, 140, 0],
            [0, 191, 255],
            [220, 20, 60],
            [50, 205, 50],
            [123, 104, 238],
        ],
        dtype=np.float32,
    )
    return palette[index % len(palette)]


def _annotate_tile(image_rgb: np.ndarray, label: str) -> np.ndarray:
    tile = image_rgb.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _generate_prompt_candidates(
    frame_rgb: np.ndarray,
    prior_mask: np.ndarray,
    image_predictor: SAM2ImagePredictor,
    cfg: MultiInstanceRefineConfig,
    component_labels: np.ndarray | None = None,
) -> List[Tuple[np.ndarray, float]]:
    image_predictor.set_image(frame_rgb)

    candidates: List[Tuple[np.ndarray, float]] = []
    prompt_box = _expanded_box_from_prior(prior_mask, cfg.prompt_box_expand_ratio)
    prompt_points = _sample_prompt_points(prior_mask, cfg, component_labels=component_labels)
    if cfg.use_box_prompt and prompt_box is not None:
        masks, ious, _ = image_predictor.predict(
            box=np.array(prompt_box, dtype=np.float32),
            multimask_output=cfg.multimask_output,
        )
        candidates.extend((mask > 0.0, float(iou)) for mask, iou in zip(masks, ious))
    for point_xy in prompt_points:
        point_coords = np.array([point_xy], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        box = np.array(prompt_box, dtype=np.float32) if (cfg.use_box_with_points and prompt_box is not None) else None
        masks, ious, _ = image_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=cfg.multimask_output,
        )
        candidates.extend((mask > 0.0, float(iou)) for mask, iou in zip(masks, ious))

    return _deduplicate_raw_candidates(candidates)


def _sample_prompt_points(
    prior_mask: np.ndarray,
    cfg: MultiInstanceRefineConfig,
    component_labels: np.ndarray | None = None,
) -> List[Tuple[float, float]]:
    prior_uint8 = prior_mask.astype(np.uint8)
    if prior_uint8.sum() == 0:
        return []

    height, width = prior_mask.shape

    if component_labels is not None:
        return _sample_prompt_points_from_labels(
            prior_uint8=prior_uint8,
            component_labels=component_labels,
            cfg=cfg,
            height=height,
            width=width,
        )

    distance = cv2.distanceTransform(prior_uint8, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(prior_uint8)
    scores = distance[ys, xs]
    order = np.argsort(scores)[::-1]

    min_distance = cfg.prompt_point_min_distance_ratio * float(np.sqrt(height * height + width * width))
    selected: List[Tuple[float, float]] = []

    for idx in order.tolist():
        point = (float(xs[idx]), float(ys[idx]))
        if all(np.hypot(point[0] - px, point[1] - py) >= min_distance for px, py in selected):
            selected.append(point)
        if len(selected) >= cfg.num_prompt_points:
            break

    if not selected:
        centroid = _mask_centroid(prior_mask)
        selected.append((centroid[0] * width, centroid[1] * height))
    return selected


def _sample_prompt_points_from_labels(
    prior_uint8: np.ndarray,
    component_labels: np.ndarray,
    cfg: MultiInstanceRefineConfig,
    height: int,
    width: int,
) -> List[Tuple[float, float]]:
    unique_labels = np.unique(component_labels)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return _sample_prompt_points_single_component(prior_uint8, cfg, height, width)

    comp_info = []
    for label_id in unique_labels:
        comp_mask = (component_labels == label_id).astype(np.uint8)
        area = int(comp_mask.sum())
        comp_info.append((label_id, area, comp_mask))

    if len(comp_info) == 1:
        return _sample_prompt_points_single_component(prior_uint8, cfg, height, width)

    max_area = max(area for _, area, _ in comp_info)
    min_component_ratio = 0.13
    significant = [
        (label_id, area, comp_mask)
        for label_id, area, comp_mask in comp_info
        if area >= max_area * min_component_ratio
    ]

    if not significant:
        significant = comp_info

    if len(significant) == 1:
        return _sample_points_from_component(
            significant[0][2], cfg.num_prompt_points, height, width
        )

    total_significant_area = sum(area for _, area, _ in significant)
    allocated = []
    remaining = cfg.num_prompt_points
    for i, (_, area, _) in enumerate(significant):
        raw = max(1, round(cfg.num_prompt_points * area / float(total_significant_area)))
        allocated.append(min(raw, remaining - (len(significant) - 1 - i)))
        remaining -= allocated[-1]
    if remaining > 0:
        for i in range(remaining):
            allocated[i] += 1

    selected: List[Tuple[float, float]] = []
    for (_, _, comp_mask), budget in zip(significant, allocated):
        points = _sample_points_from_component(comp_mask, budget, height, width)
        selected.extend(points)

    return selected


def _sample_prompt_points_single_component(
    prior_uint8: np.ndarray,
    cfg: MultiInstanceRefineConfig,
    height: int,
    width: int,
) -> List[Tuple[float, float]]:
    points = _sample_points_from_component(prior_uint8, cfg.num_prompt_points, height, width)
    if not points:
        centroid_xy = _mask_centroid(prior_uint8)
        points.append((centroid_xy[0] * width, centroid_xy[1] * height))
    return points


def _sample_points_from_component(
    comp_uint8: np.ndarray,
    budget: int,
    height: int,
    width: int,
) -> List[Tuple[float, float]]:
    distance = cv2.distanceTransform(comp_uint8, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(comp_uint8)
    scores = distance[ys, xs]
    order = np.argsort(scores)[::-1]

    min_distance = MultiInstanceRefineConfig.prompt_point_min_distance_ratio * float(
        np.sqrt(height * height + width * width)
    )
    selected: List[Tuple[float, float]] = []
    for idx in order.tolist():
        point = (float(xs[idx]), float(ys[idx]))
        if all(np.hypot(point[0] - px, point[1] - py) >= min_distance for px, py in selected):
            selected.append(point)
        if len(selected) >= budget:
            break
    return selected


def _expanded_box_from_prior(
    prior_mask: np.ndarray,
    expand_ratio: float,
) -> Tuple[int, int, int, int] | None:
    box = _mask_to_box(prior_mask)
    if box == (0, 0, 0, 0):
        return None

    x0, y0, x1, y1 = box
    height, width = prior_mask.shape
    expand_x = int((x1 - x0) * expand_ratio)
    expand_y = int((y1 - y0) * expand_ratio)

    return (
        max(0, x0 - expand_x),
        max(0, y0 - expand_y),
        min(width, x1 + expand_x),
        min(height, y1 + expand_y),
    )


def _deduplicate_raw_candidates(
    candidates: Sequence[Tuple[np.ndarray, float]],
) -> List[Tuple[np.ndarray, float]]:
    kept: List[Tuple[np.ndarray, float]] = []
    for mask, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        mask_bool = mask.astype(bool)
        if any(_mask_iou(mask_bool, kept_mask.astype(bool)) > 0.9 for kept_mask, _ in kept):
            continue
        kept.append((mask_bool, score))
    return kept


def _merge_proposals_to_objects(
    frame_rgb: np.ndarray,
    prior_mask: np.ndarray,
    proposals: Sequence[Proposal],
    frame_idx: int,
    cfg: MultiInstanceRefineConfig,
) -> List[Proposal]:
    if len(proposals) <= 1:
        return list(proposals)

    remaining = list(proposals)
    merged: List[Proposal] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]

        changed = True
        while changed:
            changed = False
            next_remaining: List[Proposal] = []
            cluster_mask = _union_masks([proposal.mask for proposal in cluster])
            cluster_box = _mask_to_box(cluster_mask)
            cluster_feature = _extract_appearance_feature(frame_rgb, cluster_mask)

            for candidate in remaining:
                if _should_merge_proposals(
                    cluster_mask=cluster_mask,
                    cluster_box=cluster_box,
                    cluster_feature=cluster_feature,
                    candidate=candidate,
                    cfg=cfg,
                ):
                    cluster.append(candidate)
                    changed = True
                else:
                    next_remaining.append(candidate)
            remaining = next_remaining

        merged.append(
            _build_merged_proposal(
                frame_rgb=frame_rgb,
                prior_mask=prior_mask,
                frame_idx=frame_idx,
                cluster=cluster,
            )
        )

    merged.sort(key=lambda proposal: proposal.score, reverse=True)
    return merged


def _should_merge_proposals(
    cluster_mask: np.ndarray,
    cluster_box: Tuple[int, int, int, int],
    cluster_feature: np.ndarray,
    candidate: Proposal,
    cfg: MultiInstanceRefineConfig,
) -> bool:
    mask_iou = _mask_iou(cluster_mask, candidate.mask)
    box_iou = _box_iou(cluster_box, candidate.bbox_xyxy)
    gap_ratio = _box_gap_ratio(cluster_box, candidate.bbox_xyxy, cluster_mask.shape)
    appearance = _cosine_similarity(cluster_feature, candidate.appearance)

    if mask_iou >= cfg.proposal_merge_mask_iou:
        return True
    if box_iou >= cfg.proposal_merge_box_iou:
        return True
    if gap_ratio <= cfg.proposal_merge_max_gap and appearance >= cfg.proposal_merge_min_appearance:
        return True
    return False


def _build_merged_proposal(
    frame_rgb: np.ndarray,
    prior_mask: np.ndarray,
    frame_idx: int,
    cluster: Sequence[Proposal],
) -> Proposal:
    merged_mask = _union_masks([proposal.mask for proposal in cluster])
    merged_bbox = _mask_to_box(merged_mask)
    merged_area = int(merged_mask.sum())
    merged_centroid = _mask_centroid(merged_mask)
    merged_feature = _extract_appearance_feature(frame_rgb, merged_mask)
    prior_area = max(int(prior_mask.sum()), 1)
    overlap_area = int((merged_mask & prior_mask).sum())
    overlap_with_prior = overlap_area / float(max(merged_area, 1))
    prior_coverage = overlap_area / float(prior_area)
    merged_score = float(max(proposal.score for proposal in cluster))

    return Proposal(
        frame_idx=frame_idx,
        mask=merged_mask,
        bbox_xyxy=merged_bbox,
        score=merged_score,
        area=merged_area,
        centroid_xy=merged_centroid,
        appearance=merged_feature,
        overlap_with_prior=overlap_with_prior,
        prior_coverage=prior_coverage,
    )


def _union_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    union_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union_mask |= mask
    return union_mask


def _deduplicate_proposals(
    proposals: Sequence[Proposal],
    cfg: MultiInstanceRefineConfig,
) -> List[Proposal]:
    kept: List[Proposal] = []
    for proposal in proposals:
        if len(kept) >= cfg.max_proposals_per_frame:
            break
        if any(_mask_iou(proposal.mask, existing.mask) > cfg.proposal_nms_iou for existing in kept):
            continue
        kept.append(proposal)
    return kept


def _build_tracks(
    proposals_by_frame: Sequence[Sequence[Proposal]],
    cfg: MultiInstanceRefineConfig,
) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    next_track_id = 1

    for frame_idx, proposals in enumerate(proposals_by_frame):
        if not proposals:
            continue

        active_indices = [
            idx for idx, track in enumerate(tracks) if frame_idx - track["last_frame"] <= cfg.max_track_age
        ]
        if not active_indices:
            for proposal in proposals:
                tracks.append(_create_track(next_track_id, proposal))
                next_track_id += 1
            continue

        cost_matrix = np.ones((len(active_indices), len(proposals)), dtype=np.float32)
        score_matrix = np.zeros_like(cost_matrix)

        for row, track_idx in enumerate(active_indices):
            for col, proposal in enumerate(proposals):
                match_score = _track_match_score(tracks[track_idx], proposal, cfg)
                score_matrix[row, col] = match_score
                cost_matrix[row, col] = 1.0 - match_score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_proposals: set[int] = set()

        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            if score_matrix[row, col] < cfg.min_match_score:
                continue
            track = tracks[active_indices[row]]
            track["proposals"].append(proposals[col])
            track["last_frame"] = frame_idx
            matched_proposals.add(col)

        for proposal_idx, proposal in enumerate(proposals):
            if proposal_idx in matched_proposals:
                continue
            tracks.append(_create_track(next_track_id, proposal))
            next_track_id += 1

    return tracks


def _create_track(track_id: int, proposal: Proposal) -> Dict[str, Any]:
    return {
        "track_id": track_id,
        "proposals": [proposal],
        "last_frame": proposal.frame_idx,
    }


def _track_match_score(
    track: Dict[str, Any],
    proposal: Proposal,
    cfg: MultiInstanceRefineConfig,
) -> float:
    recent_proposals = track["proposals"][-cfg.track_recent_history :]
    prototype = _build_track_prototype(recent_proposals)

    mask_iou = max(_mask_iou(existing.mask, proposal.mask) for existing in recent_proposals)
    box_iou = max(_box_iou(existing.bbox_xyxy, proposal.bbox_xyxy) for existing in recent_proposals)
    center_score = _center_similarity(prototype["centroid_xy"], proposal.centroid_xy, cfg.center_sigma)
    appearance_score = _cosine_similarity(prototype["appearance"], proposal.appearance)

    score = (
        cfg.weight_mask_iou * mask_iou
        + cfg.weight_box_iou * box_iou
        + cfg.weight_center * center_score
        + cfg.weight_appearance * appearance_score
    )
    return float(score)


def _build_track_prototype(proposals: Sequence[Proposal]) -> Dict[str, Any]:
    appearance = np.mean([proposal.appearance for proposal in proposals], axis=0).astype(np.float32)
    centroid_x = float(np.mean([proposal.centroid_xy[0] for proposal in proposals]))
    centroid_y = float(np.mean([proposal.centroid_xy[1] for proposal in proposals]))
    return {
        "appearance": appearance,
        "centroid_xy": (centroid_x, centroid_y),
    }


def _relink_fragmented_tracks(
    tracks: Sequence[Dict[str, Any]],
    cfg: MultiInstanceRefineConfig,
) -> List[Dict[str, Any]]:
    if len(tracks) <= 1:
        return list(tracks)

    ordered_tracks = sorted(
        [
            {
                "track_id": track["track_id"],
                "proposals": list(track["proposals"]),
                "last_frame": track["last_frame"],
            }
            for track in tracks
        ],
        key=lambda track: track["proposals"][0].frame_idx,
    )

    merged: List[Dict[str, Any]] = []
    for track in ordered_tracks:
        if not merged:
            merged.append(track)
            continue

        prev_track = merged[-1]
        if _should_relink_tracks(prev_track, track, cfg):
            prev_track["proposals"].extend(track["proposals"])
            prev_track["proposals"].sort(key=lambda proposal: proposal.frame_idx)
            prev_track["last_frame"] = prev_track["proposals"][-1].frame_idx
        else:
            merged.append(track)
    return merged


def _should_relink_tracks(
    track_a: Dict[str, Any],
    track_b: Dict[str, Any],
    cfg: MultiInstanceRefineConfig,
) -> bool:
    gap = track_b["proposals"][0].frame_idx - track_a["proposals"][-1].frame_idx
    if gap <= 0 or gap > cfg.track_relink_max_frame_gap:
        return False

    prototype_a = _build_track_prototype(track_a["proposals"][-cfg.track_recent_history :])
    prototype_b = _build_track_prototype(track_b["proposals"][: cfg.track_recent_history])
    appearance = _cosine_similarity(prototype_a["appearance"], prototype_b["appearance"])
    center = _center_similarity(prototype_a["centroid_xy"], prototype_b["centroid_xy"], cfg.center_sigma)
    return appearance >= cfg.track_relink_min_appearance and center >= cfg.track_relink_min_center


def _center_similarity(
    centroid_a: Tuple[float, float],
    centroid_b: Tuple[float, float],
    sigma: float,
) -> float:
    dx = centroid_a[0] - centroid_b[0]
    dy = centroid_a[1] - centroid_b[1]
    dist = np.sqrt(dx * dx + dy * dy)
    return float(np.exp(-(dist * dist) / max(2.0 * sigma * sigma, 1e-6)))


def _cosine_similarity(feature_a: np.ndarray, feature_b: np.ndarray) -> float:
    denom = np.linalg.norm(feature_a) * np.linalg.norm(feature_b)
    if denom < 1e-6:
        return 0.0
    return float(np.clip(np.dot(feature_a, feature_b) / denom, 0.0, 1.0))


def _extract_appearance_feature(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = frame_rgb[mask]
    if pixels.size == 0:
        return np.zeros(6, dtype=np.float32)
    mean_rgb = pixels.mean(axis=0) / 255.0
    std_rgb = pixels.std(axis=0) / 255.0
    return np.concatenate([mean_rgb, std_rgb]).astype(np.float32)


def _mask_to_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    h, w = mask.shape
    return (float(xs.mean() / max(w, 1)), float(ys.mean() / max(h, 1)))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def _box_gap_ratio(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
    mask_shape: Tuple[int, int],
) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    gap_x = max(0, max(ax0 - bx1, bx0 - ax1))
    gap_y = max(0, max(ay0 - by1, by0 - ay1))
    gap = float(np.sqrt(gap_x * gap_x + gap_y * gap_y))

    height, width = mask_shape
    diag = float(np.sqrt(height * height + width * width))
    if diag <= 1e-6:
        return 1.0
    return gap / diag


def _write_frames_to_tempdir(frames_rgb: Sequence[np.ndarray]) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="sam2_multi_instance_")
    for idx, frame_rgb in enumerate(frames_rgb):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        image_path = os.path.join(tmp_dir, f"{idx:05d}.jpg")
        cv2.imwrite(image_path, frame_bgr)
    return tmp_dir


def _inject_tracks_as_objects(
    inference_state: Dict[str, Any],
    predictor: Any,
    tracks: Sequence[Dict[str, Any]],
    cfg: MultiInstanceRefineConfig,
) -> None:
    for track in tracks:
        obj_id = track["track_id"]
        for proposal in _select_track_anchors(track, cfg):
            predictor.add_new_mask(
                inference_state,
                frame_idx=proposal.frame_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(proposal.mask),
            )


def _inject_single_mask_anchors(
    inference_state: Dict[str, Any],
    predictor: Any,
    mask_tensor: torch.Tensor,
    anchor_indices: Sequence[int],
    cfg: MultiInstanceRefineConfig,
) -> None:
    ann_obj_id = 1
    for frame_idx in anchor_indices:
        clean_mask = preprocess_mask(mask_tensor[frame_idx], cfg.single_mask_threshold)
        if clean_mask.sum() == 0:
            continue
        predictor.add_new_mask(
            inference_state,
            frame_idx=int(frame_idx),
            obj_id=ann_obj_id,
            mask=clean_mask,
        )


def _select_track_anchors(
    track: Dict[str, Any],
    cfg: MultiInstanceRefineConfig,
) -> List[Proposal]:
    proposals = list(track["proposals"])
    if not proposals:
        return []

    proposals.sort(key=lambda proposal: proposal.frame_idx)
    if len(proposals) <= cfg.max_anchor_frames:
        return proposals

    areas = np.array([proposal.area for proposal in proposals], dtype=np.float32)
    scores = np.array([proposal.score for proposal in proposals], dtype=np.float32)
    median_area = float(np.median(areas)) if len(areas) > 0 else 0.0
    max_score = float(scores.max()) if len(scores) > 0 else 0.0

    filtered = [
        proposal
        for proposal in proposals
        if proposal.area >= median_area * cfg.anchor_min_area_ratio
        and proposal.score >= max_score * cfg.anchor_min_score_ratio
    ]
    candidate_pool = filtered if filtered else proposals

    scored_candidates = sorted(
        candidate_pool,
        key=lambda proposal: _anchor_quality_score(
            proposal=proposal,
            median_area=median_area,
            max_score=max_score,
        ),
        reverse=True,
    )
    selected = [scored_candidates[0]]

    while len(selected) < min(cfg.max_anchor_frames, len(scored_candidates)):
        best_candidate = None
        best_value = -1.0
        for candidate in scored_candidates:
            if any(candidate.frame_idx == chosen.frame_idx for chosen in selected):
                continue

            quality = _anchor_quality_score(
                proposal=candidate,
                median_area=median_area,
                max_score=max_score,
            )
            time_distance = min(
                abs(candidate.frame_idx - chosen.frame_idx) for chosen in selected
            )
            total_span = proposals[-1].frame_idx - proposals[0].frame_idx
            diversity = time_distance / float(max(total_span, 1))
            value = quality + cfg.anchor_time_diversity_weight * diversity
            if value > best_value:
                best_value = value
                best_candidate = candidate

        if best_candidate is None:
            break
        selected.append(best_candidate)

    return sorted(selected, key=lambda proposal: proposal.frame_idx)


def _anchor_quality_score(
    proposal: Proposal,
    median_area: float,
    max_score: float,
) -> float:
    score_term = proposal.score / max(max_score, 1e-6)
    area_term = proposal.area / max(median_area, 1.0)
    area_term = float(np.clip(area_term, 0.0, 1.5) / 1.5)
    return float(0.5 * score_term + 0.3 * proposal.prior_coverage + 0.2 * area_term)


def _select_single_mask_anchors(
    mask_tensor: torch.Tensor,
    cfg: MultiInstanceRefineConfig,
) -> List[int]:
    frame_count = int(mask_tensor.shape[0])
    if frame_count == 0:
        return []

    stats = []
    for frame_idx in range(frame_count):
        mask = mask_tensor[frame_idx]
        binary_mask = preprocess_mask(mask, cfg.single_mask_threshold)
        area = int(binary_mask.sum().item())
        if area == 0:
            continue
        confidence = float(mask[binary_mask].mean().item()) if binary_mask.any() else 0.0
        stats.append((frame_idx, area, confidence))

    if not stats:
        return []

    areas = np.array([area for _, area, _ in stats], dtype=np.float32)
    confidences = np.array([confidence for _, _, confidence in stats], dtype=np.float32)
    median_area = float(np.median(areas))
    max_confidence = float(confidences.max())

    candidates = [
        (frame_idx, area, confidence)
        for frame_idx, area, confidence in stats
        if area >= median_area * cfg.single_anchor_min_area_ratio
        and confidence >= max_confidence * cfg.single_anchor_min_score_ratio
    ]
    if not candidates:
        candidates = stats

    candidates.sort(
        key=lambda item: _single_mask_anchor_quality(item[1], item[2], median_area, max_confidence),
        reverse=True,
    )
    selected = [candidates[0][0]]

    while len(selected) < min(cfg.single_max_anchor_frames, len(candidates)):
        best_frame_idx = None
        best_value = -1.0
        for frame_idx, area, confidence in candidates:
            if frame_idx in selected:
                continue
            quality = _single_mask_anchor_quality(area, confidence, median_area, max_confidence)
            time_distance = min(abs(frame_idx - chosen_idx) for chosen_idx in selected)
            diversity = time_distance / float(max(frame_count - 1, 1))
            value = quality + cfg.single_anchor_time_diversity_weight * diversity
            if value > best_value:
                best_value = value
                best_frame_idx = frame_idx

        if best_frame_idx is None:
            break
        selected.append(best_frame_idx)

    return sorted(selected)


def _single_mask_anchor_quality(
    area: int,
    confidence: float,
    median_area: float,
    max_confidence: float,
) -> float:
    confidence_term = confidence / max(max_confidence, 1e-6)
    area_term = area / max(median_area, 1.0)
    area_term = float(np.clip(area_term, 0.0, 1.5) / 1.5)
    return float(0.6 * confidence_term + 0.4 * area_term)


def _propagate_all(
    predictor: Any,
    inference_state: Dict[str, Any],
    reverse: bool,
) -> Dict[int, Dict[int, torch.Tensor]]:
    cond_frames = _collect_prompt_frames(inference_state)
    if not cond_frames:
        return {}

    start_frame_idx = max(cond_frames) if reverse else min(cond_frames)
    outputs: Dict[int, Dict[int, torch.Tensor]] = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=start_frame_idx,
        reverse=reverse,
    ):
        outputs[frame_idx] = {
            int(obj_id): mask_logits[obj_offset].float().cpu()
            for obj_offset, obj_id in enumerate(obj_ids)
        }
    return outputs


def _collect_prompt_frames(inference_state: Dict[str, Any]) -> List[int]:
    frame_indices: set[int] = set()

    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        frame_indices.update(obj_output_dict["cond_frame_outputs"].keys())

    for obj_temp_output_dict in inference_state["temp_output_dict_per_obj"].values():
        frame_indices.update(obj_temp_output_dict["cond_frame_outputs"].keys())

    return sorted(frame_indices)


def _sam2_inference_context() -> Any:
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)

def _merge_directional_logits(
    frame_count: int,
    obj_ids: Sequence[int],
    forward_outputs: Dict[int, Dict[int, torch.Tensor]],
    reverse_outputs: Dict[int, Dict[int, torch.Tensor]],
    anchor_frames_by_obj: Dict[int, Sequence[int]],
    cfg: MultiInstanceRefineConfig,
) -> torch.Tensor:
    example_mask = None
    for outputs in (forward_outputs, reverse_outputs):
        for per_obj in outputs.values():
            if per_obj:
                example_mask = next(iter(per_obj.values()))
                break
        if example_mask is not None:
            break
    if example_mask is None:
        raise RuntimeError("SAM2 propagation produced no masks.")

    height, width = example_mask.shape[-2:]
    instance_masks = torch.zeros((frame_count, len(obj_ids), height, width), dtype=torch.float32)

    for frame_idx in range(frame_count):
        for obj_offset, obj_id in enumerate(obj_ids):
            forward_mask = forward_outputs.get(frame_idx, {}).get(obj_id)
            reverse_mask = reverse_outputs.get(frame_idx, {}).get(obj_id)
            anchor_frames = anchor_frames_by_obj.get(obj_id, [])
            if forward_mask is not None and reverse_mask is not None:
                forward_distance = _directional_anchor_distance(anchor_frames, frame_idx, reverse=False)
                reverse_distance = _directional_anchor_distance(anchor_frames, frame_idx, reverse=True)
                forward_weight = _distance_weight(forward_distance, cfg.fusion_distance_temperature)
                reverse_weight = _distance_weight(reverse_distance, cfg.fusion_distance_temperature)
                fused_logits = forward_weight * forward_mask[0] + reverse_weight * reverse_mask[0]
                instance_masks[frame_idx, obj_offset] = (fused_logits > 0.0).float()
            elif forward_mask is not None:
                instance_masks[frame_idx, obj_offset] = (forward_mask[0] > 0.0).float()
            elif reverse_mask is not None:
                instance_masks[frame_idx, obj_offset] = (reverse_mask[0] > 0.0).float()

    return instance_masks


def _directional_anchor_distance(
    anchor_frames: Sequence[int],
    frame_idx: int,
    reverse: bool,
) -> float:
    if reverse:
        candidates = [anchor for anchor in anchor_frames if anchor >= frame_idx]
    else:
        candidates = [anchor for anchor in anchor_frames if anchor <= frame_idx]
    if not candidates:
        return float("inf")
    return float(min(abs(frame_idx - anchor) for anchor in candidates))


def _distance_weight(distance: float, temperature: float) -> float:
    if not np.isfinite(distance):
        return 0.0
    return float(np.exp(-distance / max(temperature, 1e-6)))
