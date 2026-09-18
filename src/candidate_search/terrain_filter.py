"""Terrain and structural candidate filtering module.

Filters weak, low-texture, or redundant candidate regions using structural complexity,
edge density, variance tests, and Non-Maximum Suppression (NMS).
"""

from typing import Any, Dict, List, Sequence, Tuple
import cv2
import numpy as np

from src.structure.edges import (
    compute_edge_density,
    extract_auto_canny_edges,
    extract_canny_edges,
)
from src.structure.gradients import validate_grayscale


def compute_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox_a: (x, y, w, h)
        bbox_b: (x, y, w, h)

    Returns:
        float: IoU score in range [0.0, 1.0].
    """
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def apply_nms(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.4,
    max_keep: int = 10,
) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression on candidate regions based on correlation scores.

    Args:
        candidates: List of candidate dicts with 'bbox' and 'score'.
        iou_threshold: Overlap threshold above which lower-scored candidates are suppressed.
        max_keep: Maximum number of candidate regions to retain.

    Returns:
        List[Dict[str, Any]]: NMS-filtered candidate list sorted by score descending.
    """
    if not candidates:
        return []

    # Sort descending by score
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_cands:
        bbox = cand["bbox"]
        should_keep = True

        for k in kept:
            if compute_iou(bbox, k["bbox"]) > iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept.append(cand)
            if len(kept) >= max_keep:
                break

    return kept


def filter_candidate_regions(
    candidates: List[Dict[str, Any]],
    reference_image: np.ndarray,
    min_score: float = 0.15,
    min_edge_density: float = 0.005,
    min_variance: float = 5.0,
    nms_iou_threshold: float = 0.4,
    max_candidates: int = 5,
) -> List[Dict[str, Any]]:
    """Filter candidate ROIs by structural score, terrain texture complexity, and NMS.

    Args:
        candidates: Candidate dicts from coarse localization.
        reference_image: Reference lunar image array.
        min_score: Minimum correlation score.
        min_edge_density: Minimum edge density inside ROI to avoid featureless terrain.
        min_variance: Minimum pixel intensity variance to avoid uniform shadows/saturation.
        nms_iou_threshold: Maximum IoU allowed between kept candidates.
        max_candidates: Top-N candidates to keep after filtering.

    Returns:
        List[Dict[str, Any]]: Validated and enriched candidate ROIs.
    """
    ref_gray = validate_grayscale(reference_image)
    ref_h, ref_w = ref_gray.shape[:2]

    valid_candidates: List[Dict[str, Any]] = []

    for cand in candidates:
        score = cand.get("score", 0.0)
        if score < min_score:
            continue

        x, y, w, h = cand["bbox"]

        # Ensure bounding box is within reference boundaries
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(ref_w, x + w)
        y2 = min(ref_h, y + h)

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            continue

        roi = ref_gray[y1:y2, x1:x2]
        var = float(np.var(roi))
        if var < min_variance:
            continue

        # Check edge density in candidate ROI with dynamic auto-thresholds
        roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX) if roi.max() > roi.min() else roi
        roi_edges = extract_auto_canny_edges(roi_norm)
        edge_density = compute_edge_density(roi_edges)
        if edge_density < min_edge_density:
            continue

        cand_copy = dict(cand)
        cand_copy["variance"] = var
        cand_copy["edge_density"] = edge_density
        cand_copy["clipped_bbox"] = (x1, y1, x2 - x1, y2 - y1)
        valid_candidates.append(cand_copy)

    # Apply NMS to suppress overlapping ROIs
    final_candidates = apply_nms(valid_candidates, iou_threshold=nms_iou_threshold, max_keep=max_candidates)

    # Assign 1-indexed rank / ROI IDs
    for idx, c in enumerate(final_candidates, start=1):
        c["roi_id"] = idx

    return final_candidates
