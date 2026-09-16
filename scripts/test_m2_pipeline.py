"""M2 Pipeline Test & Demonstration Script.

Executes the complete Member 2 (M2) pipeline on input lunar imagery:
1. Image Ingestion & Dimensions Check
2. Structure Representation Extraction (gradients, edges, terrain relief)
3. Terrain-Aware Candidate Search (multi-scale localization, filtering, ranking)
4. Fine SIFT Correspondence on Best Candidate ROI
5. Visualization Artifact Generation
"""

import argparse
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
    match_sift_features,
    match_with_candidate_roi,
)
from src.structure import (
    StructuralRepresentation,
    extract_structure_representation,
)


def draw_candidate_rois(
    reference_image: np.ndarray,
    candidates: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw bounding boxes and scores for all candidate ROIs onto reference image.

    Args:
        reference_image: Grayscale or BGR reference image.
        candidates: List of candidate ROI dicts.

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
        is_best = (roi_id == 1)

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
) -> None:
    """Execute complete M2 pipeline and save visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Images (supports PNG, JPG, TIFF, 8-bit / 16-bit)
    source_img = load_lunar_image(source_path)
    reference_img = load_lunar_image(reference_path)

    src_h, src_w = source_img.shape[:2]
    ref_h, ref_w = reference_img.shape[:2]

    # 2. Structure Representation Extraction
    src_structure = extract_structure_representation(source_img)
    ref_structure = extract_structure_representation(reference_img)

    # 3. Terrain-Aware Candidate Search
    search_result = generate_candidate_rois(
        source_image=src_structure,
        reference_image=ref_structure,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        top_k_per_scale=5,
        min_score=0.10,
        min_edge_density=0.001,
        min_variance=1.0,
        nms_iou_threshold=0.4,
        max_candidates=5,
    )

    candidates = search_result["candidates"]
    best_candidate = search_result["best_candidate"]

    # 4. Fine SIFT Correspondence
    if best_candidate is not None:
        match_result = match_with_candidate_roi(
            source_image=source_img,
            reference_image=reference_img,
            candidate_roi=best_candidate,
            ratio_threshold=ratio_threshold,
        )
    else:
        # Fallback to direct full-image matching if no candidate ROI passed filtering
        match_result = match_sift_features(
            source_image=source_img,
            reference_image=reference_img,
            ratio_threshold=ratio_threshold,
        )

    # Extract metrics
    source_points = match_result["source_points"]
    reference_points = match_result["reference_points"]
    num_matches = match_result["num_matches"]
    confidence = match_result["confidence"]
    num_kp_src = match_result["num_keypoints_source"]
    num_kp_ref = match_result["num_keypoints_reference"]

    best_bbox_str = str(best_candidate["bbox"]) if best_candidate else "None"
    best_score = best_candidate["score"] if best_candidate else 0.0

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

    # 5. Print Results & Diagnostic Statistics
    print("=" * 65)
    print("FLUX - M2 Pipeline Execution & Diagnostic Report")
    print("=" * 65)
    print(f"Source image dimensions        : {src_w} x {src_h} (dtype: {source_img.dtype})")
    print(f"Reference image dimensions     : {ref_w} x {ref_h} (dtype: {reference_img.dtype})")
    print("-" * 65)
    print("DIAGNOSTIC 1: Intensity & Dynamic Range Statistics")
    print(f"  Source intensity   : min={source_img.min()}, max={source_img.max()}, mean={source_img.mean():.2f}, std={source_img.std():.2f}, var={np.var(source_img):.2f}")
    print(f"  Reference intensity: min={reference_img.min()}, max={reference_img.max()}, mean={reference_img.mean():.2f}, std={reference_img.std():.2f}, var={np.var(reference_img):.2f}")
    print("-" * 65)
    print("DIAGNOSTIC 2: Edge & Gradient Structural Metrics")
    print(f"  Source non-zero edge pixels    : {src_edge_nz} / {source_img.size} (density: {src_structure.edge_density:.6f})")
    print(f"  Reference non-zero edge pixels : {ref_edge_nz} / {reference_img.size} (density: {ref_structure.edge_density:.6f})")
    print(f"  Source gradient magnitude      : min={src_mag.min():.2f}, max={src_mag.max():.2f}, mean={src_mag.mean():.2f}, std={src_mag.std():.2f}")
    print(f"  Reference gradient magnitude   : min={ref_mag.min():.2f}, max={ref_mag.max():.2f}, mean={ref_mag.mean():.2f}, std={ref_mag.std():.2f}")
    print("-" * 65)
    print("DIAGNOSTIC 3: Candidate Search & ROI Localization")
    print(f"  Number of candidate ROIs found : {len(candidates)}")
    print(f"  Best candidate bbox            : {best_bbox_str}")
    print(f"  Best candidate score           : {best_score:.6f}")
    if best_candidate:
        print(f"  Best candidate scale           : {best_candidate.get('scale', 1.0)}")
        print(f"  Best candidate ROI variance    : {best_candidate.get('variance', 0.0):.2f}")
        print(f"  Best candidate ROI edge density: {best_candidate.get('edge_density', 0.0):.6f}")
    print("-" * 65)
    print("DIAGNOSTIC 4: SIFT Keypoint & Correspondence Analysis")
    print(f"  Source keypoints (full image)  : {n_kp_src_full}")
    print(f"  Reference keypoints (full img) : {n_kp_ref_full}")
    print(f"  Source keypoints (matching)    : {num_kp_src}")
    print(f"  Reference keypoints (ROI/match): {num_kp_ref}")
    print(f"  Good matches accepted          : {num_matches}")
    print(f"  Correspondence confidence      : {confidence:.6f}")
    print(f"  source_points shape            : {source_points.shape}")
    print(f"  reference_points shape         : {reference_points.shape}")
    print("=" * 65)

    # 6. Save Visualizations
    out_source = output_dir / "01_source.png"
    out_ref = output_dir / "02_reference.png"
    out_src_struct = output_dir / "03_source_structure.png"
    out_ref_struct = output_dir / "04_reference_structure.png"
    out_candidate_rois = output_dir / "05_candidate_rois.png"
    out_correspondence = output_dir / "06_sift_correspondence.png"

    cv2.imwrite(str(out_source), source_img)
    cv2.imwrite(str(out_ref), reference_img)
    cv2.imwrite(str(out_src_struct), src_structure.composite)
    cv2.imwrite(str(out_ref_struct), ref_structure.composite)

    vis_rois = draw_candidate_rois(reference_img, candidates)
    cv2.imwrite(str(out_candidate_rois), vis_rois)

    vis_corr = draw_sift_correspondences(
        source_img, reference_img, source_points, reference_points
    )
    cv2.imwrite(str(out_correspondence), vis_corr)

    print(f"Visualizations saved to: {output_dir.resolve()}")
    print(f" - {out_source.name}")
    print(f" - {out_ref.name}")
    print(f" - {out_src_struct.name}")
    print(f" - {out_ref_struct.name}")
    print(f" - {out_candidate_rois.name}")
    print(f" - {out_correspondence.name}")


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

    run_m2_pipeline(
        source_path=source_path,
        reference_path=reference_path,
        output_dir=output_dir,
        ratio_threshold=args.ratio_threshold,
    )


if __name__ == "__main__":
    main()
