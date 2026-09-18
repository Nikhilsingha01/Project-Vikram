"""Coarse localization module for lunar candidate ROI search.

Performs multi-scale structural cross-correlation between source and reference
structural representations to identify promising candidate regions of interest.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.structure.gradients import validate_grayscale
from src.structure.structure_representation import (
    StructuralRepresentation,
    extract_structure_representation,
)


def extract_search_representation(
    image_or_rep: Union[np.ndarray, StructuralRepresentation],
) -> np.ndarray:
    """Extract 2D structural composite array from either an image array or StructuralRepresentation.

    Args:
        image_or_rep: Image array or StructuralRepresentation instance.

    Returns:
        np.ndarray: 2D uint8 structural composite matrix.
    """
    if isinstance(image_or_rep, StructuralRepresentation):
        return image_or_rep.composite
    elif isinstance(image_or_rep, np.ndarray):
        gray = validate_grayscale(image_or_rep)
        rep = extract_structure_representation(gray)
        return rep.composite
    else:
        raise TypeError(
            f"Expected np.ndarray or StructuralRepresentation, got {type(image_or_rep).__name__}"
        )


def find_coarse_candidates(
    source_rep: Union[np.ndarray, StructuralRepresentation],
    reference_rep: Union[np.ndarray, StructuralRepresentation],
    scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    top_k_per_scale: int = 5,
    match_method: int = cv2.TM_CCOEFF_NORMED,
) -> List[Dict[str, Any]]:
    """Perform multi-scale structural matching to find candidate ROIs in the reference image.

    Args:
        source_rep: Source image or StructuralRepresentation (e.g. Chandrayaan-2 patch).
        reference_rep: Reference image or StructuralRepresentation (e.g. LRO/SELENE mosaic).
        scales: Scale factors to resize the source structural template for scale-invariance.
        top_k_per_scale: Number of top candidate peaks to extract per scale level.
        match_method: OpenCV template matching method (default cv2.TM_CCOEFF_NORMED).

    Returns:
        List[Dict[str, Any]]: Unfiltered candidate regions, each containing:
            - 'bbox': Tuple (x, y, w, h) in reference image coordinates
            - 'score': Normalized correlation score [-1.0, 1.0]
            - 'scale': Applied scale factor
            - 'source_size': (src_h, src_w)
    """
    src_comp = extract_search_representation(source_rep)
    ref_comp = extract_search_representation(reference_rep)

    ref_h, ref_w = ref_comp.shape[:2]
    orig_src_h, orig_src_w = src_comp.shape[:2]

    candidates: List[Dict[str, Any]] = []

    for s in scales:
        if s <= 0:
            continue

        target_w = int(orig_src_w * s)
        target_h = int(orig_src_h * s)

        # Template cannot be larger than the reference image
        if target_w >= ref_w or target_h >= ref_h or target_w < 16 or target_h < 16:
            continue

        resized_src = cv2.resize(
            src_comp, (target_w, target_h), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        )

        res = cv2.matchTemplate(ref_comp, resized_src, match_method)

        # Flatten and extract top-k peak coordinates
        res_copy = res.copy()
        for _ in range(top_k_per_scale):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_copy)

            if match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                score = 1.0 - min_val
                best_loc = min_loc
            else:
                score = max_val
                best_loc = max_loc

            if np.isnan(score) or np.isinf(score):
                break

            bx, by = best_loc
            candidates.append({
                "bbox": (int(bx), int(by), int(target_w), int(target_h)),
                "score": float(score),
                "scale": float(s),
                "source_size": (orig_src_h, orig_src_w),
            })

            # Suppress local neighborhood around this peak in correlation map to find distinct peaks
            suppress_rad_x = max(4, target_w // 4)
            suppress_rad_y = max(4, target_h // 4)
            y1 = max(0, by - suppress_rad_y)
            y2 = min(res_copy.shape[0], by + suppress_rad_y + 1)
            x1 = max(0, bx - suppress_rad_x)
            x2 = min(res_copy.shape[1], bx + suppress_rad_x + 1)

            if match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
                res_copy[y1:y2, x1:x2] = float("inf")
            else:
                res_copy[y1:y2, x1:x2] = -float("inf")

    return candidates
