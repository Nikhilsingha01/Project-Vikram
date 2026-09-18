"""M2 Pipeline Test & Demonstration Script.

Executes the complete Member 2 (M2) pipeline on input lunar imagery:
1. Image Ingestion & Dimensions Check
2. Structure Representation Extraction (gradients, edges, terrain relief)
3. Terrain-Aware Candidate Search (multi-scale localization, filtering, ranking)
4. Fine SIFT Correspondence on Best Candidate ROI
5. Visualization Artifact Generation
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.candidate_search import (
    extract_candidate_crop,
    generate_candidate_rois,
)
from src.correspondence import (
    evaluate_geometric_ransac,
    match_sift_features,
    match_with_candidate_roi,
    select_best_candidate_correspondence,
)
from src.ingestion.metadata import (
    GSDValidationResult,
    ImageMetadata,
    calculate_physical_scale_ratio,
)
from src.structure import (
    StructuralRepresentation,
    extract_structure_representation,
)
from src.validation.evidence_gate import (
    EvidenceGate,
    EvidenceGateConfig,
)


def draw_candidate_rois(
    reference_image: np.ndarray,
    candidates: List[Dict[str, Any]],
    best_candidate_id: Optional[int] = None,
) -> np.ndarray:
    """Draw bounding boxes and scores for all candidate ROIs onto reference image.

    Args:
        reference_image: Grayscale or BGR reference image.
        candidates: List of candidate ROI dicts.
        best_candidate_id: Optional ROI ID of the selected best candidate.

    Returns:
        np.ndarray: BGR annotated reference image.
    """
    if reference_image.ndim == 2:
        vis = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = reference_image.copy()

    for cand in candidates:
        roi_id = cand.get("roi_id", 1)
        score = cand.get("score", 0.0)
        bbox = cand.get("clipped_bbox", cand.get("bbox"))
        if bbox is None:
            continue

        x, y, w, h = bbox
        is_best = (roi_id == best_candidate_id) if best_candidate_id is not None else (roi_id == 1)

        # Best ROI in bright green, others in yellow/cyan
        color = (0, 255, 0) if is_best else (255, 180, 0)
        thickness = 2 if is_best else 1

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label = f"ROI #{roi_id} ({score:.2f})"
        cv2.putText(
            vis,
            label,
            (x + 4, max(y + 16, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def draw_sift_correspondences(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    max_draw: int = 50,
) -> np.ndarray:
    """Draw side-by-side matching lines between source and reference points.

    Args:
        source_image: 2D grayscale source image.
        reference_image: 2D grayscale reference image.
        source_points: (N, 2) array of coordinates in source image.
        reference_points: (N, 2) array of coordinates in reference image.
        max_draw: Maximum number of match lines to draw.

    Returns:
        np.ndarray: BGR stitched match visualization.
    """
    h1, w1 = source_image.shape[:2]
    h2, w2 = reference_image.shape[:2]

    canvas_h = max(h1, h2)
    canvas_w = w1 + w2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place source image on left
    if source_image.ndim == 2:
        canvas[:h1, :w1] = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h1, :w1] = source_image

    # Place reference image on right
    if reference_image.ndim == 2:
        canvas[:h2, w1 : w1 + w2] = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h2, w1 : w1 + w2] = reference_image

    num_matches = len(source_points)
    if num_matches == 0:
        return canvas

    step = max(1, num_matches // max_draw) if num_matches > max_draw else 1

    for i in range(0, num_matches, step):
        pt1 = (int(round(source_points[i][0])), int(round(source_points[i][1])))
        pt2 = (int(round(reference_points[i][0] + w1)), int(round(reference_points[i][1])))

        # Draw circle on keypoints
        cv2.circle(canvas, pt1, 3, (0, 255, 255), -1)
        cv2.circle(canvas, pt2, 3, (0, 255, 255), -1)

        # Draw connecting line
        cv2.line(canvas, pt1, pt2, (0, 220, 100), 1, cv2.LINE_AA)

    return canvas


def draw_ransac_matches(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    inlier_mask: np.ndarray,
    mode: str = "all",
    title: Optional[str] = None,
    max_draw: int = 100,
) -> np.ndarray:
    """Draw side-by-side matches partitioned by RANSAC inliers/outliers.

    Args:
        source_image: 2D grayscale or BGR source image.
        reference_image: 2D grayscale or BGR reference image.
        source_points: (N, 2) array of coordinates in source image.
        reference_points: (N, 2) array of coordinates in reference image.
        inlier_mask: (N,) boolean array where True indicates inlier.
        mode: "inliers", "outliers", or "all".
        title: Optional banner title text.
        max_draw: Maximum lines to draw per subset.

    Returns:
        np.ndarray: BGR stitched visualization canvas.
    """
    h1, w1 = source_image.shape[:2]
    h2, w2 = reference_image.shape[:2]

    header_h = 36 if title else 0
    canvas_h = max(h1, h2) + header_h
    canvas_w = w1 + w2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place source image on left
    src_bgr = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR) if source_image.ndim == 2 else source_image
    ref_bgr = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR) if reference_image.ndim == 2 else reference_image

    canvas[header_h : header_h + h1, :w1] = src_bgr
    canvas[header_h : header_h + h2, w1 : w1 + w2] = ref_bgr

    if title:
        cv2.putText(
            canvas,
            title,
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    num_total = len(source_points)
    if num_total == 0:
        return canvas

    mask = inlier_mask if inlier_mask is not None else np.zeros(num_total, dtype=bool)

    # Determine indices to draw based on mode
    if mode == "inliers":
        indices = [i for i in range(num_total) if mask[i]]
    elif mode == "outliers":
        indices = [i for i in range(num_total) if not mask[i]]
    else:  # "all"
        # First draw outliers, then inliers on top
        indices = [i for i in range(num_total) if not mask[i]] + [i for i in range(num_total) if mask[i]]

    step = max(1, len(indices) // max_draw) if len(indices) > max_draw else 1

    for idx in indices[::step]:
        is_inlier = bool(mask[idx])
        pt1 = (int(round(source_points[idx][0])), int(round(source_points[idx][1])) + header_h)
        pt2 = (int(round(reference_points[idx][0] + w1)), int(round(reference_points[idx][1])) + header_h)

        if is_inlier:
            line_color = (0, 255, 0)      # Bright green for inliers
            pt_color = (0, 255, 255)      # Yellow for inlier keypoints
            thickness = 2
            radius = 4
        else:
            line_color = (0, 0, 255)      # Red for outliers
            pt_color = (100, 100, 255)    # Light red for outlier keypoints
            thickness = 1
            radius = 3

        cv2.circle(canvas, pt1, radius, pt_color, -1)
        cv2.circle(canvas, pt2, radius, pt_color, -1)
        cv2.line(canvas, pt1, pt2, line_color, thickness, cv2.LINE_AA)

    return canvas


def load_lunar_image(image_path: Path) -> np.ndarray:
    """Load a lunar image supporting PNG, JPG, TIFF, GeoTIFF (8-bit, 16-bit, float32).

    Converts multi-channel to single-channel grayscale and normalizes 16-bit/float
    data to 8-bit uint8 for OpenCV/SIFT processing.

    Args:
        image_path: Path to image file.

    Returns:
        np.ndarray: 2D uint8 grayscale image array.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file cannot be decoded by OpenCV.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    # Load with IMREAD_UNCHANGED to preserve bit depth (e.g. 16-bit TIFFs)
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(
            f"Failed to read image at: {image_path}. Format may be unsupported or corrupted."
        )

    # Convert color / multi-band to grayscale
    if raw.ndim == 3:
        if raw.shape[2] == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        elif raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        elif raw.shape[2] == 1:
            gray = raw.squeeze(axis=2)
        else:
            gray = raw[:, :, 0]  # Take first band for multi-spectral imagery
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported array dimension: {raw.ndim}")

    # Handle 16-bit / 32-bit float satellite bands
    if gray.dtype == np.uint16:
        min_v, max_v = float(gray.min()), float(gray.max())
        if max_v > min_v:
            gray_u8 = ((gray.astype(np.float32) - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            gray_u8 = (gray // 256).astype(np.uint8)
    elif gray.dtype in (np.float32, np.float64):
        min_v, max_v = float(gray.min()), float(gray.max())
        if max_v > min_v:
            gray_u8 = ((gray - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
        else:
            gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_u8 = gray

    return gray_u8


def run_m2_pipeline(
    source_path: Path,
    reference_path: Path,
    output_dir: Path,
    ratio_threshold: float = 0.75,
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
) -> None:
    """Execute complete M2 pipeline and save visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Images (supports PNG, JPG, TIFF, 8-bit / 16-bit)
    source_img = load_lunar_image(source_path)
    reference_img = load_lunar_image(reference_path)

    src_h, src_w = source_img.shape[:2]
    ref_h, ref_w = reference_img.shape[:2]

    # 2. GSD Metadata Validation & Physical Scale Analysis
    gsd_val = calculate_physical_scale_ratio(source_gsd, reference_gsd)

    # 3. Structure Representation Extraction
    src_structure = extract_structure_representation(source_img)
    ref_structure = extract_structure_representation(reference_img)

    # 4. Terrain-Aware Candidate Search
    search_result = generate_candidate_rois(
        source_image=src_structure,
        reference_image=ref_structure,
        scales=(0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0),
        top_k_per_scale=5,
        min_score=0.10,
        min_edge_density=0.001,
        min_variance=1.0,
        nms_iou_threshold=0.4,
        max_candidates=5,
        gsd_validation=gsd_val,
    )

    candidates = search_result["candidates"]

    # 5. Multi-Candidate Correspondence & RANSAC Geometric Selection
    selection_result = select_best_candidate_correspondence(
        source_image=source_img,
        reference_image=reference_img,
        candidates=candidates,
        ratio_threshold=ratio_threshold,
        cross_check=True,
    )

    best_match = selection_result["best_match"]
    best_candidate = selection_result["best_candidate"]
    candidate_evals = selection_result["candidate_evaluations"]

    # Extract metrics from best selected match
    source_points = best_match["source_points"]
    reference_points = best_match["reference_points"]
    num_matches = best_match["num_matches"]
    confidence = best_match["confidence"]
    num_kp_src = best_match["num_keypoints_source"]
    num_kp_ref = best_match["num_keypoints_reference"]

    best_bbox_str = str(best_candidate["bbox"]) if best_candidate else "None"
    best_score = best_candidate["score"] if best_candidate else 0.0
    best_roi_id = best_candidate.get("roi_id", 1) if best_candidate else None

    # RANSAC Diagnostic summary from best candidate
    ransac_diag = best_match.get("ransac_eval")
    if ransac_diag is None:
        ransac_diag = evaluate_geometric_ransac(
            source_points=source_points,
            reference_points=reference_points,
            source_shape=(src_h, src_w),
            ransac_threshold=5.0,
        )

    n_inliers = ransac_diag["inliers"]
    n_outliers = ransac_diag["outliers"]
    inlier_ratio = ransac_diag["inlier_ratio"]
    rmse = ransac_diag["rmse"]
    H_matrix = ransac_diag["homography"]
    inlier_mask = ransac_diag["inlier_mask"]
    is_degenerate = ransac_diag["is_degenerate"]
    degeneracy_reasons = ransac_diag["degeneracy_reasons"]
    is_reliable = ransac_diag["is_reliable"]

    # 6. M3 EvidenceGate Evaluation
    evidence_gate = EvidenceGate(EvidenceGateConfig())
    gate_result = evidence_gate.evaluate(best_match)

    # Compute full image SIFT keypoints for diagnostic baseline
    sift_diag = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    kp_src_full, _ = sift_diag.detectAndCompute(source_img, None)
    kp_ref_full, _ = sift_diag.detectAndCompute(reference_img, None)
    n_kp_src_full = len(kp_src_full) if kp_src_full is not None else 0
    n_kp_ref_full = len(kp_ref_full) if kp_ref_full is not None else 0

    # Gradient magnitude statistics
    src_mag = src_structure.gradient_magnitude
    ref_mag = ref_structure.gradient_magnitude

    # Edge map non-zero pixel counts
    src_edge_nz = int(np.count_nonzero(src_structure.edges))
    ref_edge_nz = int(np.count_nonzero(ref_structure.edges))

    # 7. Print Comprehensive Results & Diagnostic Table
    print("=" * 80)
    print("FLUX - M2 Multi-Candidate Pipeline Execution & Diagnostic Report")
    print("=" * 80)
    print(f"Source image dimensions        : {src_w} x {src_h} (dtype: {source_img.dtype})")
    print(f"Reference image dimensions     : {ref_w} x {ref_h} (dtype: {reference_img.dtype})")
    print("-" * 80)
    print("DIAGNOSTIC 0: Sensor & Physical GSD Validation")
    if gsd_val.is_valid:
        print(f"  Source GSD                     : {gsd_val.source_gsd:.4f} m/pixel")
        print(f"  Reference GSD                  : {gsd_val.reference_gsd:.4f} m/pixel")
        print(f"  Physical scale ratio (src/ref) : {gsd_val.scale_ratio:.4e} ({1.0/gsd_val.scale_ratio:.1f}x resolution gap)")
        print(f"  Physical scale suitability     : {'SUITABLE' if gsd_val.is_physically_suitable else 'UNSUITABLE / EXTREME GAP'}")
        print(f"  GSD Validation Status          : {gsd_val.status}")
        if gsd_val.warning_message:
            print(f"  [GSD WARNING]                  : {gsd_val.warning_message}")
    else:
        print(f"  GSD Validation Status          : {gsd_val.status}")
        print(f"  Metadata note                  : {gsd_val.warning_message}")
    print("-" * 80)
    print("DIAGNOSTIC 1: Intensity & Dynamic Range Statistics")
    print(f"  Source intensity   : min={source_img.min()}, max={source_img.max()}, mean={source_img.mean():.2f}, std={source_img.std():.2f}, var={np.var(source_img):.2f}")
    print(f"  Reference intensity: min={reference_img.min()}, max={reference_img.max()}, mean={reference_img.mean():.2f}, std={reference_img.std():.2f}, var={np.var(reference_img):.2f}")
    print("-" * 80)
    print("DIAGNOSTIC 2: Edge & Gradient Structural Metrics")
    print(f"  Source non-zero edge pixels    : {src_edge_nz} / {source_img.size} (density: {src_structure.edge_density:.6f})")
    print(f"  Reference non-zero edge pixels : {ref_edge_nz} / {reference_img.size} (density: {ref_structure.edge_density:.6f})")
    print(f"  Source gradient magnitude      : min={src_mag.min():.2f}, max={src_mag.max():.2f}, mean={src_mag.mean():.2f}, std={src_mag.std():.2f}")
    print(f"  Reference gradient magnitude   : min={ref_mag.min():.2f}, max={ref_mag.max():.2f}, mean={ref_mag.mean():.2f}, std={ref_mag.std():.2f}")
    print("-" * 80)
    print("DIAGNOSTIC 3: Candidate ROI Multi-Scale Search Summary")
    print(f"  Number of candidate ROIs found : {len(candidates)}")
    print(f"  Selected candidate ROI ID      : #{best_roi_id} (Score: {best_score:.4f}, BBox: {best_bbox_str})")
    print("-" * 80)
    print("DIAGNOSTIC 4: Candidate ROIs Evaluation & Degeneracy Comparison Table")
    print(f"{'ROI ID':<8} {'Score':<8} {'Scale':<8} {'BBox (x,y,w,h)':<22} {'Matches':<9} {'Inliers':<9} {'Ratio':<8} {'Reliable':<10} {'Rejection Reason'}")
    print("-" * 80)
    for ceval in candidate_evals:
        cid_str = f"#{ceval['roi_id']}"
        score_str = f"{ceval['candidate_score']:.4f}"
        scale_str = f"{ceval['scale']:.3f}"
        bbox_str = str(ceval['bbox'])
        n_m = str(ceval['num_matches'])
        n_inl = str(ceval['inliers'])
        ratio_str = f"{ceval['inlier_ratio']:.1%}"
        rel_str = "YES" if ceval['is_reliable'] else "NO"
        rej_str = str(ceval['rejection_reason']) if ceval['rejection_reason'] else "ACCEPTED"
        print(f"{cid_str:<8} {score_str:<8} {scale_str:<8} {bbox_str:<22} {n_m:<9} {n_inl:<9} {ratio_str:<8} {rel_str:<10} {rej_str}")
    print("-" * 80)
    print("DIAGNOSTIC 5: SIFT Keypoint & Selected Correspondence Analysis")
    print(f"  Source keypoints (full image)  : {n_kp_src_full}")
    print(f"  Reference keypoints (full img) : {n_kp_ref_full}")
    print(f"  Source keypoints (matching)    : {num_kp_src}")
    print(f"  Reference keypoints (ROI/match): {num_kp_ref}")
    print(f"  Mutual 1-to-1 matches accepted : {num_matches}")
    print(f"  Correspondence confidence      : {confidence:.6f}")
    print(f"  Unique source keypoint coords  : {ransac_diag['unique_source_points']}")
    print(f"  Unique reference keypoint coords: {ransac_diag['unique_reference_points']}")
    print(f"  Max ref point multiplicity     : {ransac_diag['max_ref_point_multiplicity']}")
    print("-" * 80)
    print("DIAGNOSTIC 6: RANSAC Geometric Validation & Degeneracy Analysis")
    print(f"  Total matches evaluated        : {ransac_diag['total_matches']}")
    print(f"  RANSAC inliers                 : {n_inliers}")
    print(f"  RANSAC outliers                : {n_outliers}")
    print(f"  Inlier ratio                   : {inlier_ratio:.4f} ({inlier_ratio:.1%})")
    print(f"  Reprojection RMSE (inliers)    : {rmse:.4f} px" if not np.isnan(rmse) else "  Reprojection RMSE (inliers)    : NaN")
    print(f"  Source inliers spatial spread  : std_x={ransac_diag['source_inliers_spread'][0]:.2f}, std_y={ransac_diag['source_inliers_spread'][1]:.2f}")
    print(f"  Reference inliers spatial spread: std_x={ransac_diag['reference_inliers_spread'][0]:.2f}, std_y={ransac_diag['reference_inliers_spread'][1]:.2f}")
    print(f"  Transformed area compression   : {ransac_diag['area_ratio']:.3e}" if not np.isnan(ransac_diag['area_ratio']) else "  Transformed area compression   : NaN")
    print(f"  Degenerate homography flag     : {is_degenerate}")
    if degeneracy_reasons:
        print("  Degeneracy / anomaly reasons   :")
        for r in degeneracy_reasons:
            print(f"    - {r}")
    print(f"  Geometric reliability status   : {'RELIABLE (ACCEPTED)' if is_reliable else 'UNRELIABLE (REJECTED)'}")
    print("  Estimated Homography Matrix H  :")
    if H_matrix is not None:
        for row in H_matrix:
            print(f"    [{row[0]:14.6e}, {row[1]:14.6e}, {row[2]:14.6e}]")
    else:
        print("    None (estimation failed / rejected)")
    print("-" * 80)
    print("DIAGNOSTIC 7: M3 EvidenceGate Validation Outcome")
    print(f"  EvidenceGate Passed            : {gate_result.passed}")
    print(f"  Rejection Reason               : {gate_result.rejection_reason}")
    print(f"  Composite Evidence Score       : {gate_result.evidence_score:.4f}")
    print(f"  Gate Diagnostics               : {gate_result.diagnostics}")
    print("=" * 80)

    # 8. Save Visualizations
    out_source = output_dir / "01_source.png"
    out_ref = output_dir / "02_reference.png"
    out_src_struct = output_dir / "03_source_structure.png"
    out_ref_struct = output_dir / "04_reference_structure.png"
    out_candidate_rois = output_dir / "05_candidate_rois.png"
    out_correspondence = output_dir / "06_sift_correspondence.png"
    out_ransac_inliers = output_dir / "07_ransac_inliers.png"
    out_ransac_outliers = output_dir / "08_ransac_outliers.png"
    out_ransac_all = output_dir / "09_ransac_all_matches.png"

    cv2.imwrite(str(out_source), source_img)
    cv2.imwrite(str(out_ref), reference_img)
    cv2.imwrite(str(out_src_struct), src_structure.composite)
    cv2.imwrite(str(out_ref_struct), ref_structure.composite)

    vis_rois = draw_candidate_rois(reference_img, candidates, best_candidate_id=best_roi_id)
    cv2.imwrite(str(out_candidate_rois), vis_rois)

    vis_corr = draw_sift_correspondences(
        source_img, reference_img, source_points, reference_points
    )
    cv2.imwrite(str(out_correspondence), vis_corr)

    # 07_ransac_inliers.png
    vis_inliers = draw_ransac_matches(
        source_img,
        reference_img,
        source_points,
        reference_points,
        inlier_mask,
        mode="inliers",
        title=f"RANSAC Inliers ({n_inliers}/{len(source_points)}, RMSE: {rmse:.2f}px)" if not np.isnan(rmse) else f"RANSAC Inliers ({n_inliers})",
    )
    cv2.imwrite(str(out_ransac_inliers), vis_inliers)

    # 08_ransac_outliers.png
    vis_outliers = draw_ransac_matches(
        source_img,
        reference_img,
        source_points,
        reference_points,
        inlier_mask,
        mode="outliers",
        title=f"RANSAC Outliers ({n_outliers}/{len(source_points)})",
    )
    cv2.imwrite(str(out_ransac_outliers), vis_outliers)

    # 09_ransac_all_matches.png
    vis_all = draw_ransac_matches(
        source_img,
        reference_img,
        source_points,
        reference_points,
        inlier_mask,
        mode="all",
        title=f"All Matches (Green: {n_inliers} Inliers, Red: {n_outliers} Outliers, Inlier Ratio: {inlier_ratio:.1%})",
    )
    cv2.imwrite(str(out_ransac_all), vis_all)

    print(f"Visualizations saved to: {output_dir.resolve()}")
    print(f" - {out_source.name}")
    print(f" - {out_ref.name}")
    print(f" - {out_src_struct.name}")
    print(f" - {out_ref_struct.name}")
    print(f" - {out_candidate_rois.name}")
    print(f" - {out_correspondence.name}")
    print(f" - {out_ransac_inliers.name}")
    print(f" - {out_ransac_outliers.name}")
    print(f" - {out_ransac_all.name}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run Member 2 (M2) Pipeline: Structure Analysis, Candidate Search, and SIFT Correspondence."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/test/source_lunar_test.png",
        help="Path to source lunar image",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/test/reference_lunar_test.png",
        help="Path to reference lunar image",
    )
    parser.add_argument(
        "--source-gsd",
        type=float,
        default=None,
        help="Ground Sample Distance of source image in meters/pixel (e.g. 0.23 for OHRC)",
    )
    parser.add_argument(
        "--reference-gsd",
        type=float,
        default=None,
        help="Ground Sample Distance of reference image in meters/pixel (e.g. 200.0 for LRO WAC)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/m2_test",
        help="Directory where output visualizations will be saved",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for SIFT matching (default: 0.75)",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    reference_path = Path(args.reference)
    if not reference_path.is_absolute():
        reference_path = PROJECT_ROOT / reference_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # Auto-detect sensible default GSD if known files are passed and GSD not specified
    source_gsd = args.source_gsd
    reference_gsd = args.reference_gsd
    if source_gsd is None and ("ohrc" in source_path.name.lower() or "chandrayaan" in source_path.name.lower()):
        source_gsd = 0.23
    if reference_gsd is None and ("lro" in reference_path.name.lower() or "quickmap" in reference_path.name.lower()):
        reference_gsd = 200.0

    run_m2_pipeline(
        source_path=source_path,
        reference_path=reference_path,
        output_dir=output_dir,
        ratio_threshold=args.ratio_threshold,
        source_gsd=source_gsd,
        reference_gsd=reference_gsd,
    )


if __name__ == "__main__":
    main()

