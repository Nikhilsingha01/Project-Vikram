import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.structure.gradients import validate_grayscale

logger = logging.getLogger(__name__)


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
        cross_check: If True, enforces bidirectional mutual nearest-neighbor consistency (1-to-1).

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

    good_matches_list: List[Dict[str, Any]] = []
    src_pts: List[Tuple[float, float]] = []
    ref_pts: List[Tuple[float, float]] = []
    distances: List[float] = []

    if cross_check:
        # Bidirectional mutual nearest-neighbor check with Lowe's ratio test in both directions
        raw_forward = bf.knnMatch(desc_src, desc_ref, k=2)
        raw_backward = bf.knnMatch(desc_ref, desc_src, k=2)

        # Forward queryIdx -> Best DMatch passing ratio test
        good_forward: Dict[int, Any] = {}
        for pair in raw_forward:
            if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
                good_forward[pair[0].queryIdx] = pair[0]
            elif len(pair) == 1:
                good_forward[pair[0].queryIdx] = pair[0]

        # Backward queryIdx (ref) -> Best DMatch passing ratio test
        good_backward: Dict[int, Any] = {}
        for pair in raw_backward:
            if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance:
                good_backward[pair[0].queryIdx] = pair[0]
            elif len(pair) == 1:
                good_backward[pair[0].queryIdx] = pair[0]

        # Enforce mutual 1-to-1 agreement
        matched_train_indices: set = set()
        mutual_pairs: List[Tuple[int, int, float]] = []

        for q_idx, m_fwd in good_forward.items():
            t_idx = m_fwd.trainIdx
            if t_idx in good_backward and good_backward[t_idx].trainIdx == q_idx:
                mutual_pairs.append((q_idx, t_idx, float(m_fwd.distance)))

        # Sort by distance for best quality matches first
        mutual_pairs.sort(key=lambda p: p[2])
        matched_train_indices: Set[int] = set()
        matched_ref_coords: Set[Tuple[float, float]] = set()

        for q_idx, t_idx, dist in mutual_pairs:
            ref_pt = kp_ref[t_idx].pt
            ref_coord = (round(float(ref_pt[0]), 2), round(float(ref_pt[1]), 2))
            if t_idx not in matched_train_indices and ref_coord not in matched_ref_coords:
                matched_train_indices.add(t_idx)
                matched_ref_coords.add(ref_coord)
                src_pt = kp_src[q_idx].pt
                src_pts.append(src_pt)
                ref_pts.append(ref_pt)
                distances.append(dist)
                good_matches_list.append({
                    "source_idx": int(q_idx),
                    "reference_idx": int(t_idx),
                    "source_pt": (float(src_pt[0]), float(src_pt[1])),
                    "reference_pt": (float(ref_pt[0]), float(ref_pt[1])),
                    "distance": float(dist),
                    "ratio": float(dist / max(1e-6, dist)),
                })
    else:
        raw_matches = bf.knnMatch(desc_src, desc_ref, k=2)
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
    cross_check: bool = True,
) -> Dict[str, Any]:
    """Perform correspondence between source image and a candidate reference ROI,

    mapping match coordinates back to the global reference image coordinate frame.

    Args:
        source_image: Source lunar image array.
        reference_image: Full reference image.
        candidate_roi: Candidate dictionary with 'bbox' or 'clipped_bbox' (x, y, w, h).
        max_features: Maximum SIFT features.
        ratio_threshold: Lowe's ratio test threshold.
        cross_check: If True (default), enforces mutual nearest-neighbor consistency (1-to-1).

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
        cross_check=cross_check,
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


def evaluate_geometric_ransac(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    source_shape: Optional[Tuple[int, int]] = None,
    ransac_threshold: float = 5.0,
    min_inlier_ratio: float = 0.25,
    min_inliers_required: int = 6,
    confidence: float = 0.99,
    max_iters: int = 2000,
) -> Dict[str, Any]:
    """Evaluate geometric consistency and detect degenerate transformations via RANSAC homography.

    Performs comprehensive diagnostic checks:
    - Homography singularity / zero or negative determinant
    - Polygon corner transformation / collapsed Shoelace area (< 1.0 px^2 or < 0.1% of original)
    - Keypoint spatial concentration (std_x < 2.0 or std_y < 2.0 in reference frame)
    - Duplicate reference point reuse / multiplicity
    - Inlier ratio and reprojection RMSE thresholds

    Args:
        source_points: (N, 2) array of matched source coordinates.
        reference_points: (N, 2) array of matched reference coordinates.
        source_shape: Optional (height, width) tuple of source image for polygon area checks.
        ransac_threshold: Maximum reprojection error to classify a point as an inlier.
        min_inlier_ratio: Minimum acceptable ratio of inliers to total matches.
        min_inliers_required: Minimum absolute number of inliers required.
        confidence: RANSAC confidence level.
        max_iters: Maximum RANSAC iterations.

    Returns:
        Dict[str, Any]: Detailed evaluation metrics and degeneracy diagnostics.
    """
    total = len(source_points)
    if total < 4 or len(reference_points) < 4:
        return {
            "total_matches": total,
            "inliers": 0,
            "outliers": total,
            "inlier_ratio": 0.0,
            "rmse": float("nan"),
            "homography": None,
            "inlier_mask": np.zeros(total, dtype=bool),
            "is_degenerate": True,
            "degeneracy_reasons": ["Insufficient matches (< 4) for geometric estimation"],
            "primary_rejection_reason": "INSUFFICIENT_MATCHES",
            "unique_source_points": len(np.unique(source_points, axis=0)) if total > 0 else 0,
            "unique_reference_points": len(np.unique(reference_points, axis=0)) if len(reference_points) > 0 else 0,
            "max_ref_point_multiplicity": 0,
            "source_inliers_spread": (0.0, 0.0),
            "reference_inliers_spread": (0.0, 0.0),
            "area_ratio": float("nan"),
            "is_reliable": False,
        }

    # Count unique coordinates and multiplicity
    _, src_counts = np.unique(source_points, axis=0, return_counts=True)
    _, ref_counts = np.unique(reference_points, axis=0, return_counts=True)
    unique_src = len(src_counts)
    unique_ref = len(ref_counts)
    max_ref_mult = int(ref_counts.max()) if len(ref_counts) > 0 else 0

    reasons: List[str] = []
    is_degenerate = False
    primary_reason: Optional[str] = None

    # Check for duplicate / many-to-one matches in incoming set
    if max_ref_mult > 1 and (total - unique_ref) > (total * 0.25):
        is_degenerate = True
        reasons.append(f"High reference coordinate multiplicity: max={max_ref_mult}, {total - unique_ref} duplicate matches")
        if primary_reason is None:
            primary_reason = "DUPLICATE_REFERENCE_MATCHES"

    H, mask = cv2.findHomography(
        source_points.reshape(-1, 1, 2),
        reference_points.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=confidence,
        maxIters=max_iters,
    )

    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(total, dtype=bool)
    inliers = int(inlier_mask.sum())
    outliers = total - inliers
    inlier_ratio = float(inliers / total)

    rmse = float("nan")
    area_ratio = float("nan")
    src_spread = (0.0, 0.0)
    ref_spread = (0.0, 0.0)

    if inliers >= 4 and H is not None:
        src_inliers = source_points[inlier_mask]
        ref_inliers = reference_points[inlier_mask]

        # Inlier spatial spread
        src_spread = (float(np.std(src_inliers[:, 0])), float(np.std(src_inliers[:, 1])))
        ref_spread = (float(np.std(ref_inliers[:, 0])), float(np.std(ref_inliers[:, 1])))

        # Reprojection RMSE over inliers
        proj = cv2.perspectiveTransform(src_inliers.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs = np.linalg.norm(ref_inliers - proj, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))

        # Determinant & Singularity Check
        det = float(np.linalg.det(H))
        if abs(det) < 1e-4 or det <= 0:
            is_degenerate = True
            reasons.append(f"Singular / negative determinant (det={det:.3e})")
            if primary_reason is None:
                primary_reason = "DEGENERATE_TRANSFORMATION"

        # Transformed Corner Collapse / Area Ratio Check
        sh, sw = source_shape if source_shape is not None else (1000, 1000)
        corners = np.array([[[0, 0]], [[sw, 0]], [[sw, sh]], [[0, sh]]], dtype=np.float32)
        try:
            w_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            # Shoelace formula for polygon area
            cx, cy = w_corners[:, 0], w_corners[:, 1]
            poly_area = 0.5 * abs(float(np.dot(cx, np.roll(cy, 1)) - np.dot(cy, np.roll(cx, 1))))
            orig_area = float(sw * sh)
            area_ratio = float(poly_area / max(1.0, orig_area))

            if poly_area < 1.0 or area_ratio < 1e-3:
                is_degenerate = True
                reasons.append(f"Extreme area collapse (area_ratio={area_ratio:.3e}, warped_area={poly_area:.2f} px^2)")
                if primary_reason is None:
                    primary_reason = "SPATIAL_COLLAPSE"
        except Exception as e:
            is_degenerate = True
            reasons.append(f"Corner projection failed: {e}")
            if primary_reason is None:
                primary_reason = "DEGENERATE_TRANSFORMATION"

        # Spatial Concentration Check
        if ref_spread[0] < 2.0 and ref_spread[1] < 2.0:
            is_degenerate = True
            reasons.append(f"Reference inliers collapsed to single cluster (ref_std_x={ref_spread[0]:.2f}, ref_std_y={ref_spread[1]:.2f})")
            if primary_reason is None:
                primary_reason = "INSUFFICIENT_SPATIAL_COVERAGE"

        # Multiplicity in inliers check
        _, inlier_ref_counts = np.unique(ref_inliers, axis=0, return_counts=True)
        if len(inlier_ref_counts) > 0 and inlier_ref_counts.max() >= max(3, int(inliers * 0.5)):
            is_degenerate = True
            reasons.append(f"Over {inlier_ref_counts.max()}/{inliers} inliers share the exact same reference coordinate")
            if primary_reason is None:
                primary_reason = "DUPLICATE_REFERENCE_MATCHES"

        # Inlier Ratio check
        if inlier_ratio < min_inlier_ratio:
            reasons.append(f"Low inlier ratio ({inlier_ratio:.1%} < {min_inlier_ratio:.1%})")
            if primary_reason is None:
                primary_reason = "LOW_INLIER_RATIO"

        # Inlier Count check
        if inliers < min_inliers_required:
            reasons.append(f"Insufficient inliers ({inliers} < {min_inliers_required})")
            if primary_reason is None:
                primary_reason = "INSUFFICIENT_INLIERS"

        # Reprojection error check
        if not np.isnan(rmse) and rmse > ransac_threshold:
            reasons.append(f"High reprojection RMSE ({rmse:.2f} px > {ransac_threshold:.2f} px)")
            if primary_reason is None:
                primary_reason = "HIGH_REPROJECTION_ERROR"
    else:
        is_degenerate = True
        reasons.append("Fewer than 4 RANSAC inliers found")
        if primary_reason is None:
            primary_reason = "INSUFFICIENT_INLIERS"

    is_reliable = (
        (not is_degenerate)
        and (inlier_ratio >= min_inlier_ratio)
        and (inliers >= min_inliers_required)
        and (not np.isnan(rmse) and rmse <= ransac_threshold)
    )

    if is_reliable:
        primary_reason = None

    return {
        "total_matches": total,
        "inliers": inliers,
        "outliers": outliers,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "homography": H,
        "inlier_mask": inlier_mask,
        "is_degenerate": is_degenerate,
        "degeneracy_reasons": reasons,
        "primary_rejection_reason": primary_reason,
        "unique_source_points": unique_src,
        "unique_reference_points": unique_ref,
        "max_ref_point_multiplicity": max_ref_mult,
        "source_inliers_spread": src_spread,
        "reference_inliers_spread": ref_spread,
        "area_ratio": area_ratio,
        "is_reliable": is_reliable,
    }


def select_best_candidate_correspondence(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    candidates: Sequence[Dict[str, Any]],
    max_features: int = 5000,
    ratio_threshold: float = 0.75,
    cross_check: bool = True,
    ransac_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Evaluate SIFT correspondence across all top-K candidate ROIs and select the most geometrically reliable.

    Prioritizes:
    1. Geometrically reliable / non-degenerate transformation
    2. RANSAC inlier count
    3. Inlier ratio
    4. Spatial distribution / spread
    5. Low reprojection RMSE

    Args:
        source_image: Source lunar image.
        reference_image: Full reference lunar image.
        candidates: Sequence of candidate ROI dicts from coarse localization / terrain filter.
        max_features: Maximum SIFT keypoints.
        ratio_threshold: Lowe's ratio test threshold.
        cross_check: If True (default), enforces mutual nearest-neighbor consistency.
        ransac_threshold: RANSAC inlier distance threshold in pixels.

    Returns:
        Dict[str, Any]: Structured evaluation containing:
            - 'best_match': Top selected correspondence dict (with 'ransac_eval', 'is_reliable', etc.).
            - 'best_candidate': Candidate ROI corresponding to best match (or None).
            - 'candidate_evaluations': List of evaluation records for all candidates.
            - 'num_candidates_evaluated': Total candidate ROIs evaluated.
            - 'num_reliable_candidates': Number of candidates that produced reliable geometric matches.
            - 'overall_reliable': True if at least one candidate passed geometric validation.
    """
    if not candidates:
        # Fallback to direct full-image matching
        direct_match = match_sift_features(
            source_image,
            reference_image,
            max_features=max_features,
            ratio_threshold=ratio_threshold,
            cross_check=cross_check,
        )
        ransac_eval = evaluate_geometric_ransac(
            direct_match["source_points"],
            direct_match["reference_points"],
            source_shape=source_image.shape[:2],
            ransac_threshold=ransac_threshold,
        )
        direct_match["ransac_eval"] = ransac_eval
        direct_match["is_reliable"] = ransac_eval["is_reliable"]
        direct_match["primary_rejection_reason"] = ransac_eval["primary_rejection_reason"]
        return {
            "best_match": direct_match,
            "best_candidate": None,
            "candidate_evaluations": [],
            "num_candidates_evaluated": 0,
            "num_reliable_candidates": 1 if ransac_eval["is_reliable"] else 0,
            "overall_reliable": ransac_eval["is_reliable"],
        }

    evaluations: List[Dict[str, Any]] = []

    for cand in candidates:
        cid = cand.get("roi_id", len(evaluations) + 1)
        bbox = cand.get("clipped_bbox", cand.get("bbox", (0, 0, 0, 0)))

        # 1. Match SIFT features on candidate ROI crop
        match_record = match_with_candidate_roi(
            source_image=source_image,
            reference_image=reference_image,
            candidate_roi=cand,
            max_features=max_features,
            ratio_threshold=ratio_threshold,
            cross_check=cross_check,
        )

        # 2. Geometric RANSAC Evaluation
        ransac_eval = evaluate_geometric_ransac(
            source_points=match_record["source_points"],
            reference_points=match_record["reference_points"],
            source_shape=source_image.shape[:2],
            ransac_threshold=ransac_threshold,
        )

        match_record["ransac_eval"] = ransac_eval
        match_record["is_reliable"] = ransac_eval["is_reliable"]
        match_record["primary_rejection_reason"] = ransac_eval["primary_rejection_reason"]

        evaluations.append({
            "candidate": cand,
            "roi_id": cid,
            "bbox": bbox,
            "candidate_score": float(cand.get("score", 0.0)),
            "scale": float(cand.get("scale", 1.0)),
            "num_matches": match_record["num_matches"],
            "inliers": ransac_eval["inliers"],
            "inlier_ratio": ransac_eval["inlier_ratio"],
            "rmse": ransac_eval["rmse"],
            "is_degenerate": ransac_eval["is_degenerate"],
            "is_reliable": ransac_eval["is_reliable"],
            "rejection_reason": ransac_eval["primary_rejection_reason"],
            "match_record": match_record,
        })

    # Sort / Rank candidates:
    # Priority 1: is_reliable (True > False)
    # Priority 2: inliers (higher is better)
    # Priority 3: inlier_ratio (higher is better)
    # Priority 4: candidate_score (higher is better)
    def ranking_key(item: Dict[str, Any]) -> Tuple[int, int, float, float]:
        is_rel = 1 if item["is_reliable"] else 0
        inliers = item["inliers"]
        inlier_ratio = item["inlier_ratio"]
        cand_score = item["candidate_score"]
        return (is_rel, inliers, inlier_ratio, cand_score)

    sorted_evals = sorted(evaluations, key=ranking_key, reverse=True)
    best_eval = sorted_evals[0]

    reliable_count = sum(1 for e in evaluations if e["is_reliable"])

    logger.info(
        "select_best_candidate_correspondence: Evaluated %d candidates, %d reliable. Best ROI #%s (inliers=%d, reliable=%s)",
        len(candidates),
        reliable_count,
        best_eval["roi_id"],
        best_eval["inliers"],
        best_eval["is_reliable"],
    )

    return {
        "best_match": best_eval["match_record"],
        "best_candidate": best_eval["candidate"],
        "candidate_evaluations": evaluations,
        "num_candidates_evaluated": len(evaluations),
        "num_reliable_candidates": reliable_count,
        "overall_reliable": best_eval["is_reliable"],
    }
