"""Classical SIFT feature matching and correspondence module.

Implements scale-invariant feature extraction (SIFT), brute-force descriptor matching,
Lowe's ratio test filtering, coordinate mapping, and structured correspondence outputs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.structure.gradients import validate_grayscale


def match_sift_features(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    max_features: int = 5000,
    ratio_threshold: float = 0.75,
    cross_check: bool = False,
) -> Dict[str, Any]:
    """Extract SIFT features and perform correspondence using Lowe's ratio test.

    Args:
        source_image: Source lunar image array (e.g., Chandrayaan-2 patch).
        reference_image: Reference lunar image array (or ROI sub-image).
        max_features: Maximum number of SIFT keypoints to detect.
        ratio_threshold: Distance ratio threshold for Lowe's second-neighbor test (0.7 - 0.8).
        cross_check: If True, enforces mutual nearest neighbors (only if knnMatch is not used).

    Returns:
        Dict[str, Any]: Structured correspondence record containing:
            - 'source_points': (N, 2) float32 array of [x, y] coordinates in source image
            - 'reference_points': (N, 2) float32 array of [x, y] coordinates in reference image
            - 'confidence': Overall correspondence quality score in [0.0, 1.0]
            - 'matches': Detailed list of match items with distances and coordinates
            - 'num_matches': Number of accepted matches
            - 'num_keypoints_source': Keypoints detected in source
            - 'num_keypoints_reference': Keypoints detected in reference
    """
    src_gray = validate_grayscale(source_image)
    ref_gray = validate_grayscale(reference_image)

    # Ensure 8-bit representation for OpenCV SIFT compatibility
    if src_gray.dtype != np.uint8:
        src_gray = cv2.normalize(src_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_gray.dtype != np.uint8:
        ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Initialize SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6,
    )

    # Detect keypoints and compute descriptors
    kp_src, desc_src = sift.detectAndCompute(src_gray, None)
    kp_ref, desc_ref = sift.detectAndCompute(ref_gray, None)

    num_kp_src = len(kp_src) if kp_src is not None else 0
    num_kp_ref = len(kp_ref) if kp_ref is not None else 0

    empty_points = np.empty((0, 2), dtype=np.float32)

    # Handle edge case: insufficient keypoints or descriptors
    if desc_src is None or desc_ref is None or len(desc_src) < 2 or len(desc_ref) < 2:
        return {
            "source_points": empty_points,
            "reference_points": empty_points,
            "confidence": 0.0,
            "matches": [],
            "num_matches": 0,
            "num_keypoints_source": num_kp_src,
            "num_keypoints_reference": num_kp_ref,
        }

    # Match descriptors using BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc_src, desc_ref, k=2)

    good_matches_list: List[Dict[str, Any]] = []
    src_pts: List[Tuple[float, float]] = []
    ref_pts: List[Tuple[float, float]] = []
    distances: List[float] = []

    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            # Lowe's ratio test: distance to best match vs second best match
            if m.distance < ratio_threshold * n.distance:
                src_pt = kp_src[m.queryIdx].pt
                ref_pt = kp_ref[m.trainIdx].pt

                src_pts.append(src_pt)
                ref_pts.append(ref_pt)
                distances.append(m.distance)

                good_matches_list.append({
                    "source_idx": int(m.queryIdx),
                    "reference_idx": int(m.trainIdx),
                    "source_pt": (float(src_pt[0]), float(src_pt[1])),
                    "reference_pt": (float(ref_pt[0]), float(ref_pt[1])),
                    "distance": float(m.distance),
                    "ratio": float(m.distance / max(n.distance, 1e-6)),
                })

    num_matches = len(src_pts)

    if num_matches > 0:
        source_points = np.array(src_pts, dtype=np.float32)
        reference_points = np.array(ref_pts, dtype=np.float32)

        # Compute match confidence score based on match count, ratio test strictness, and mean distance
        # Ratio of good matches relative to detected keypoints
        ratio_coverage = min(1.0, num_matches / max(10, min(num_kp_src, num_kp_ref)))
        # Mean distance confidence (lower distance is better)
        avg_dist = float(np.mean(distances))
        dist_confidence = max(0.0, 1.0 - (avg_dist / 300.0))

        confidence = float(np.clip(0.6 * ratio_coverage + 0.4 * dist_confidence, 0.0, 1.0))
    else:
        source_points = empty_points
        reference_points = empty_points
        confidence = 0.0

    return {
        "source_points": source_points,
        "reference_points": reference_points,
        "confidence": confidence,
        "matches": good_matches_list,
        "num_matches": num_matches,
        "num_keypoints_source": num_kp_src,
        "num_keypoints_reference": num_kp_ref,
    }


def match_with_candidate_roi(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    candidate_roi: Dict[str, Any],
    max_features: int = 5000,
    ratio_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Perform correspondence between source image and a candidate reference ROI,

    mapping match coordinates back to the global reference image coordinate frame.

    Args:
        source_image: Source lunar image array.
        reference_image: Full reference image.
        candidate_roi: Candidate dictionary with 'bbox' or 'clipped_bbox' (x, y, w, h).
        max_features: Maximum SIFT features.
        ratio_threshold: Lowe's ratio test threshold.

    Returns:
        Dict[str, Any]: Structured correspondence record with global reference coordinates.
    """
    ref_gray = validate_grayscale(reference_image)
    bbox = candidate_roi.get("clipped_bbox", candidate_roi.get("bbox"))
    if bbox is None:
        raise ValueError("candidate_roi must contain 'bbox' or 'clipped_bbox'.")

    rx, ry, rw, rh = bbox
    roi_crop = ref_gray[ry : ry + rh, rx : rx + rw]

    match_result = match_sift_features(
        source_image=source_image,
        reference_image=roi_crop,
        max_features=max_features,
        ratio_threshold=ratio_threshold,
    )

    # Offset reference points by ROI origin (rx, ry) to map to full reference frame
    if match_result["num_matches"] > 0:
        global_ref_pts = match_result["reference_points"].copy()
        global_ref_pts[:, 0] += rx
        global_ref_pts[:, 1] += ry
        match_result["reference_points"] = global_ref_pts

        for m in match_result["matches"]:
            local_x, local_y = m["reference_pt"]
            m["reference_pt"] = (local_x + rx, local_y + ry)

    match_result["candidate_roi"] = candidate_roi
    return match_result
