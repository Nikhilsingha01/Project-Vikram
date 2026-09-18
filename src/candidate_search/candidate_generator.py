"""Candidate ROI generator module for terrain-aware lunar image correspondence.

Orchestrates multi-scale coarse localization, structural terrain filtering,
candidate ranking, and ROI extraction.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.candidate_search.coarse_localization import find_coarse_candidates
from src.candidate_search.terrain_filter import filter_candidate_regions
from src.structure.gradients import validate_grayscale
from src.structure.structure_representation import (
    StructuralRepresentation,
    extract_structure_representation,
)


def generate_candidate_rois(
    source_image: Union[np.ndarray, StructuralRepresentation],
    reference_image: Union[np.ndarray, StructuralRepresentation],
    scales: Sequence[float] = (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0),
    top_k_per_scale: int = 5,
    min_score: float = 0.15,
    min_edge_density: float = 0.005,
    min_variance: float = 5.0,
    nms_iou_threshold: float = 0.4,
    max_candidates: int = 5,
) -> Dict[str, Any]:
    """Generate and rank candidate Regions of Interest (ROIs) from reference imagery.

    Args:
        source_image: Source image or StructuralRepresentation.
        reference_image: Reference image or StructuralRepresentation.
        scales: Scale factors to test during coarse search.
        top_k_per_scale: Peaks to detect per scale.
        min_score: Minimum correlation threshold.
        min_edge_density: Minimum edge density inside candidate ROI.
        min_variance: Minimum pixel variance inside candidate ROI.
        nms_iou_threshold: NMS overlap threshold.
        max_candidates: Maximum number of ranked candidate ROIs to return.

    Returns:
        Dict[str, Any]: Structured search output containing:
            - 'candidates': Ranked list of candidate ROI dicts with 'roi_id', 'bbox', 'score', 'scale', etc.
            - 'total_raw_candidates': Total candidates found across all scales before filtering.
            - 'num_candidates_retained': Number of candidates surviving filtering.
            - 'best_candidate': Top-ranked candidate ROI dict (or None if no candidates found).
    """
    # 1. Ensure structural representations
    if isinstance(source_image, StructuralRepresentation):
        src_rep = source_image
        src_raw = None
    else:
        src_raw = validate_grayscale(source_image)
        src_rep = extract_structure_representation(src_raw)

    if isinstance(reference_image, StructuralRepresentation):
        ref_rep = reference_image
        ref_raw = None
    else:
        ref_raw = validate_grayscale(reference_image)
        ref_rep = extract_structure_representation(ref_raw)

    # 2. Coarse localization
    raw_candidates = find_coarse_candidates(
        src_rep,
        ref_rep,
        scales=scales,
        top_k_per_scale=top_k_per_scale,
    )

    # Reference array for terrain texture validation
    ref_eval_img = ref_raw if ref_raw is not None else ref_rep.composite

    # 3. Terrain and structural filtering
    filtered_candidates = filter_candidate_regions(
        raw_candidates,
        ref_eval_img,
        min_score=min_score,
        min_edge_density=min_edge_density,
        min_variance=min_variance,
        nms_iou_threshold=nms_iou_threshold,
        max_candidates=max_candidates,
    )

    best_candidate = filtered_candidates[0] if filtered_candidates else None

    return {
        "candidates": filtered_candidates,
        "total_raw_candidates": len(raw_candidates),
        "num_candidates_retained": len(filtered_candidates),
        "best_candidate": best_candidate,
    }


def extract_candidate_crop(
    reference_image: np.ndarray,
    candidate: Dict[str, Any],
) -> np.ndarray:
    """Extract the image sub-array corresponding to a candidate ROI.

    Args:
        reference_image: Full reference image.
        candidate: Candidate dictionary with 'bbox' or 'clipped_bbox'.

    Returns:
        np.ndarray: Cropped ROI sub-image.
    """
    gray = validate_grayscale(reference_image)
    bbox = candidate.get("clipped_bbox", candidate.get("bbox"))
    if bbox is None:
        raise ValueError("Candidate does not contain 'bbox' or 'clipped_bbox'.")

    x, y, w, h = bbox
    return gray[y : y + h, x : x + w]
