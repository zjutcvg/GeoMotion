from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


@dataclass(frozen=True)
class MultiInstanceRefineConfig:
    prior_threshold: float = 0.8
    points_per_side: int = 24
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    min_mask_region_area: int = 0
    max_proposals_per_frame: int = 8
    min_mask_pixels: int = 48
    min_overlap_with_prior: float = 0.35
    min_prior_coverage: float = 0.02
    proposal_nms_iou: float = 0.75
    max_track_age: int = 2
    min_match_score: float = 0.25
    min_track_length: int = 2
    max_tracks: int = 16
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


def refine_sam_multi_instance(
    frame_tensors: torch.Tensor,
    mask_list: torch.Tensor | np.ndarray,
    predictor: Any,
    config: MultiInstanceRefineConfig | None = None,
) -> Dict[str, Any]:
    """Refine binary motion priors as multiple SAM2 objects.

    Args:
        frame_tensors: [T, 3, H, W] in [0, 1].
        mask_list: [T, H, W] motion priors.
        predictor: built SAM2 video predictor.
        config: optional tuning config.

    Returns:
        A dict with:
            instance_masks: torch.Tensor [T, K, H, W]
            merged_masks: torch.Tensor [T, H, W]
            track_metadata: list[dict]
            proposals_per_frame: list[int]
    """
    cfg = config or MultiInstanceRefineConfig()
    frames_rgb = _tensor_frames_to_numpy(frame_tensors)
    binary_priors = _prepare_binary_priors(mask_list, cfg.prior_threshold)
    
    with torch.inference_mode(), _sam2_inference_context():
        amg = SAM2AutomaticMaskGenerator(
            predictor,
            points_per_side=cfg.points_per_side,
            pred_iou_thresh=cfg.pred_iou_thresh,
            stability_score_thresh=cfg.stability_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            crop_n_layers=cfg.crop_n_layers,
            min_mask_region_area=cfg.min_mask_region_area,
            output_mode="binary_mask",
        )

        proposals_by_frame = [
            _generate_frame_proposals(frame, prior, frame_idx, amg, cfg)
            for frame_idx, (frame, prior) in enumerate(zip(frames_rgb, binary_priors))
        ]

    tracks = _build_tracks(proposals_by_frame, cfg)
    valid_tracks = [track for track in tracks if len(track["proposals"]) >= cfg.min_track_length]
    valid_tracks = valid_tracks[: cfg.max_tracks]

    if not valid_tracks:
        merged_masks = torch.from_numpy(binary_priors.astype(np.float32))
        empty_instances = torch.zeros(
            (merged_masks.shape[0], 0, merged_masks.shape[1], merged_masks.shape[2]),
            dtype=torch.float32,
        )
        return {
            "instance_masks": empty_instances,
            "merged_masks": merged_masks,
            "track_metadata": [],
            "proposals_per_frame": [len(p) for p in proposals_by_frame],
        }

    tmp_dir = _write_frames_to_tempdir(frames_rgb)
    try:
        with torch.inference_mode(), _sam2_inference_context():
            inference_state = predictor.init_state(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    with torch.inference_mode(), _sam2_inference_context():
        _inject_tracks_as_objects(inference_state, predictor, valid_tracks)
        forward_outputs = _propagate_all(predictor, inference_state, reverse=False)

        predictor.reset_state(inference_state)
        _inject_tracks_as_objects(inference_state, predictor, valid_tracks)
        reverse_outputs = _propagate_all(predictor, inference_state, reverse=True)

    obj_ids = [track["track_id"] for track in valid_tracks]
    instance_masks = _merge_propagation_results(
        frame_count=len(frames_rgb),
        obj_ids=obj_ids,
        forward_outputs=forward_outputs,
        reverse_outputs=reverse_outputs,
    )
    merged_masks = (instance_masks.sum(dim=1) > 0).float()

    track_metadata = [
        {
            "track_id": track["track_id"],
            "length": len(track["proposals"]),
            "frames": [proposal.frame_idx for proposal in track["proposals"]],
            "mean_score": float(np.mean([proposal.score for proposal in track["proposals"]])),
        }
        for track in valid_tracks
    ]

    return {
        "instance_masks": instance_masks,
        "merged_masks": merged_masks,
        "track_metadata": track_metadata,
        "proposals_per_frame": [len(p) for p in proposals_by_frame],
    }


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
    amg: SAM2AutomaticMaskGenerator,
    cfg: MultiInstanceRefineConfig,
) -> List[Proposal]:
    if int(prior_mask.sum()) < cfg.min_mask_pixels:
        return []

    with torch.inference_mode(), _sam2_inference_context():
        raw_anns = amg.generate(frame_rgb)
    proposals: List[Proposal] = []
    prior_area = max(int(prior_mask.sum()), 1)

    for ann in raw_anns:
        seg = ann["segmentation"].astype(bool)
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
        score = float(
            ann["predicted_iou"] * ann["stability_score"] * (0.65 * overlap_with_prior + 0.35 * prior_coverage)
        )
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
    return _deduplicate_proposals(proposals, cfg)


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
            last_proposal = tracks[track_idx]["proposals"][-1]
            for col, proposal in enumerate(proposals):
                match_score = _proposal_match_score(last_proposal, proposal, cfg)
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


def _proposal_match_score(
    proposal_a: Proposal,
    proposal_b: Proposal,
    cfg: MultiInstanceRefineConfig,
) -> float:
    mask_iou = _mask_iou(proposal_a.mask, proposal_b.mask)
    box_iou = _box_iou(proposal_a.bbox_xyxy, proposal_b.bbox_xyxy)
    center_score = _center_similarity(proposal_a.centroid_xy, proposal_b.centroid_xy, cfg.center_sigma)
    appearance_score = _cosine_similarity(proposal_a.appearance, proposal_b.appearance)

    score = (
        cfg.weight_mask_iou * mask_iou
        + cfg.weight_box_iou * box_iou
        + cfg.weight_center * center_score
        + cfg.weight_appearance * appearance_score
    )
    return float(score)


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
) -> None:
    for track in tracks:
        obj_id = track["track_id"]
        for proposal in track["proposals"]:
            predictor.add_new_mask(
                inference_state,
                frame_idx=proposal.frame_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(proposal.mask),
            )


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
            int(obj_id): (mask_logits[obj_offset] > 0.0).float().cpu()
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

def _merge_propagation_results(
    frame_count: int,
    obj_ids: Sequence[int],
    forward_outputs: Dict[int, Dict[int, torch.Tensor]],
    reverse_outputs: Dict[int, Dict[int, torch.Tensor]],
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
            if forward_mask is not None and reverse_mask is not None:
                instance_masks[frame_idx, obj_offset] = torch.maximum(forward_mask[0], reverse_mask[0])
            elif forward_mask is not None:
                instance_masks[frame_idx, obj_offset] = forward_mask[0]
            elif reverse_mask is not None:
                instance_masks[frame_idx, obj_offset] = reverse_mask[0]

    return instance_masks
