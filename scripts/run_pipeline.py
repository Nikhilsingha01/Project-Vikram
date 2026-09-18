"""FLUX End-to-End Lunar Image Correspondence & Registration Pipeline.

Connects and executes all stages:
1. M1: Data Ingestion, Image Normalization & GSD Physical Scale Validation
2. M2: Structure Representation Extraction (gradients, edge maps, relief)
3. M2: Terrain-Aware Candidate Search (multi-scale structural matching & NMS filtering)
4. M2: Bidirectional Mutual Nearest-Neighbor SIFT Correspondence & RANSAC evaluation
5. M3: EvidenceGate Go/No-Go Decision
6. M3: Geometric Estimation, Inlier Analysis, & Reprojection Error Quantification
7. M3 / Registration: Image Warping, Overlap Calculation & Subpixel Alignment
8. M3: Comprehensive Validation Metrics & Final ACCEPT/REJECT Decision
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# M1 Ingestion & Metadata
from src.ingestion.metadata import (
    GSDValidationResult,
    ImageMetadata,
    calculate_physical_scale_ratio,
)

# M2 Structure Analysis
from src.structure import (
    StructuralRepresentation,
    extract_structure_representation,
)

# M2 Candidate Search
from src.candidate_search import (
    extract_candidate_crop,
    generate_candidate_rois,
)

# M2 Fine Correspondence
from src.correspondence import (
    evaluate_geometric_ransac,
    match_sift_features,
    match_with_candidate_roi,
    select_best_candidate_correspondence,
)

# M3 Validation & Registration
from src.validation.evidence_gate import (
    EvidenceGate,
    EvidenceGateConfig,
    EvidenceGateResult,
)
from src.validation.geometric_estimation import (
    EstimationResult,
    GeometricEstimator,
    GeometricEstimatorConfig,
    TransformModel,
)
from src.validation.inlier_analysis import (
    InlierAnalyzer,
    InlierReport,
)
from src.validation.reprojection import (
    ReprojectionReport,
    compute_reprojection_errors,
)
from src.validation.registration import (
    RegistrationConfig,
    RegistrationResult,
    register_image,
)
from src.validation.metrics import (
    ACCEPT,
    REJECT,
    DecisionThresholds,
    MetricsWeights,
    ValidationMetrics,
    compute_validation_metrics,
)

logger = logging.getLogger("flux.pipeline")


def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load and normalize an image to 8-bit uint8 grayscale."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV failed to decode image: {image_path}")

    if raw.ndim == 3:
        if raw.shape[2] == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        elif raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        elif raw.shape[2] == 1:
            gray = raw.squeeze(axis=2)
        else:
            gray = raw[:, :, 0]
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported image dimension: {raw.ndim}")

    if gray.dtype == np.uint16:
        min_v, max_v = float(gray.min()), float(gray.max())
        gray = ((gray.astype(np.float32) - min_v) / max(1e-6, max_v - min_v) * 255.0).astype(np.uint8)
    elif gray.dtype in (np.float32, np.float64):
        min_v, max_v = float(gray.min()), float(gray.max())
        gray = ((gray - min_v) / max(1e-6, max_v - min_v) * 255.0).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def draw_matches_visualization(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 100,
    title: Optional[str] = None,
) -> np.ndarray:
    """Draw side-by-side matches annotated with green/red lines."""
    h1, w1 = src_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    header_h = 36 if title else 0
    canvas_h = max(h1, h2) + header_h
    canvas_w = w1 + w2

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[header_h : header_h + h1, :w1] = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR) if src_img.ndim == 2 else src_img
    canvas[header_h : header_h + h2, w1:] = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR) if ref_img.ndim == 2 else ref_img

    if title:
        cv2.putText(canvas, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

    n_pts = len(src_pts)
    if n_pts == 0:
        return canvas

    mask = inlier_mask if inlier_mask is not None else np.ones(n_pts, dtype=bool)
    indices = [i for i in range(n_pts) if not mask[i]] + [i for i in range(n_pts) if mask[i]]
    step = max(1, len(indices) // max_draw) if len(indices) > max_draw else 1

    for idx in indices[::step]:
        is_inlier = bool(mask[idx])
        p1 = (int(round(src_pts[idx][0])), int(round(src_pts[idx][1])) + header_h)
        p2 = (int(round(ref_pts[idx][0] + w1)), int(round(ref_pts[idx][1])) + header_h)
        color = (0, 255, 0) if is_inlier else (0, 0, 255)
        cv2.circle(canvas, p1, 3, (0, 255, 255) if is_inlier else (100, 100, 255), -1)
        cv2.circle(canvas, p2, 3, (0, 255, 255) if is_inlier else (100, 100, 255), -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    return canvas


def run_e2e_pipeline(
    source_path: Path,
    reference_path: Path,
    output_dir: Path,
    source_gsd: Optional[float] = None,
    reference_gsd: Optional[float] = None,
    ratio_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Execute complete FLUX End-to-End Pipeline."""
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    src_img = load_grayscale_image(source_path)
    ref_img = load_grayscale_image(reference_path)
    src_h, src_w = src_img.shape[:2]
    ref_h, ref_w = ref_img.shape[:2]

    # GSD Validation
    gsd_result = calculate_physical_scale_ratio(source_gsd, reference_gsd)

    # 2. Structure Representation
    src_struct = extract_structure_representation(src_img)
    ref_struct = extract_structure_representation(ref_img)

    # 3. Candidate Search
    search_result = generate_candidate_rois(
        source_image=src_struct,
        reference_image=ref_struct,
        scales=(0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0),
        top_k_per_scale=5,
        min_score=0.10,
        min_edge_density=0.001,
        min_variance=1.0,
        nms_iou_threshold=0.4,
        max_candidates=5,
        gsd_validation=gsd_result,
    )
    candidates = search_result["candidates"]

    # 4. Fine SIFT Correspondence & Multi-Candidate Selection
    selection_result = select_best_candidate_correspondence(
        source_image=src_img,
        reference_image=ref_img,
        candidates=candidates,
        ratio_threshold=ratio_threshold,
        cross_check=True,
    )

    best_match = selection_result["best_match"]
    best_candidate = selection_result["best_candidate"]
    candidate_evals = selection_result["candidate_evaluations"]

    # 5. M3 Evidence Gate
    gate = EvidenceGate(EvidenceGateConfig())
    gate_result = gate.evaluate(best_match)

    # 6. M3 Geometric Estimation, Inlier Analysis & Reprojection Error
    estimation_result: Optional[EstimationResult] = None
    inlier_report: Optional[InlierReport] = None
    reprojection_report: Optional[ReprojectionReport] = None
    registration_result: Optional[RegistrationResult] = None

    if gate_result.passed:
        estimator = GeometricEstimator(
            GeometricEstimatorConfig(
                model=TransformModel.AFFINE,
                ransac_reproj_threshold=5.0,
                min_inliers=6,
                min_inlier_ratio=0.20,
            )
        )
        estimation_result = estimator.estimate(best_match)

        if estimation_result.success:
            analyzer = InlierAnalyzer(grid_divisions=4)
            inlier_report = analyzer.analyse(
                inlier_mask=estimation_result.inlier_mask,
                source_points=best_match["source_points"],
                reference_points=best_match["reference_points"],
                image_shape=src_img.shape[:2],
            )

            reprojection_report = compute_reprojection_errors(
                source_points=np.asarray(best_match["source_points"], dtype=np.float64),
                reference_points=np.asarray(best_match["reference_points"], dtype=np.float64),
                transform_matrix=estimation_result.transform_matrix,
                inlier_mask=estimation_result.inlier_mask,
                model_type="affine",
                compute_symmetric=True,
            )

            registration_result = register_image(
                source_image=src_img,
                transform_matrix=estimation_result.transform_matrix,
                reference_shape=ref_img.shape[:2],
                config=RegistrationConfig(model_type="affine", compute_overlap=True),
            )

    # 7. M3 Validation Metrics & Decision
    validation_metrics: ValidationMetrics = compute_validation_metrics(
        gate_result=gate_result,
        estimation_result=estimation_result,
        inlier_report=inlier_report,
        reprojection_report=reprojection_report,
        registration_result=registration_result,
    )

    t_elapsed = time.perf_counter() - t_start

    # 8. Print Executive Diagnostic Report
    print("=" * 80)
    print("FLUX - END-TO-END LUNAR REGISTRATION & VALIDATION REPORT")
    print("=" * 80)
    print(f"Source image                   : {source_path.name} ({src_w}x{src_h} px)")
    print(f"Reference image                : {reference_path.name} ({ref_w}x{ref_h} px)")
    print(f"Execution time                 : {t_elapsed:.3f} seconds")
    print("-" * 80)
    print("STAGE 1: Sensor & Physical GSD Validation")
    if gsd_result.is_valid:
        print(f"  Source GSD                   : {gsd_result.source_gsd:.4f} m/px")
        print(f"  Reference GSD                : {gsd_result.reference_gsd:.4f} m/px")
        print(f"  Physical scale ratio         : {gsd_result.scale_ratio:.4e} ({1.0/gsd_result.scale_ratio:.1f}x resolution gap)")
        print(f"  Suitability Status           : {gsd_result.status}")
        if gsd_result.warning_message:
            print(f"  [GSD Note]                   : {gsd_result.warning_message}")
    else:
        print(f"  GSD Status                   : {gsd_result.status} ({gsd_result.warning_message})")
    print("-" * 80)
    print("STAGE 2: Terrain-Aware Candidate Search & Selection")
    print(f"  Candidate ROIs evaluated     : {len(candidate_evals)}")
    if best_candidate:
        print(f"  Selected Best Candidate      : ROI #{best_candidate.get('roi_id')} (Score: {best_candidate.get('score', 0.0):.4f}, BBox: {best_candidate.get('bbox')})")
    else:
        print("  Selected Candidate           : None (Fallback to global matching)")
    print("-" * 80)
    print("STAGE 3: Mutual SIFT Correspondence")
    print(f"  Mutual 1-to-1 Matches        : {best_match['num_matches']}")
    print(f"  Confidence Score             : {best_match['confidence']:.4f}")
    print("-" * 80)
    print("STAGE 4: M3 EvidenceGate Validation")
    print(f"  Gate Passed                  : {gate_result.passed}")
    print(f"  Evidence Score               : {gate_result.evidence_score:.4f}")
    if not gate_result.passed:
        print(f"  Gate Rejection Reason        : {gate_result.rejection_reason}")
    print("-" * 80)
    print("STAGE 5: Geometric Estimation & Reprojection")
    if estimation_result and estimation_result.success:
        print(f"  Estimation Success           : True (Model: {estimation_result.transform_type})")
        if inlier_report:
            print(f"  RANSAC Inliers               : {inlier_report.inlier_count}/{inlier_report.total_count} ({inlier_report.inlier_ratio:.1%})")
            print(f"  Spatial Coverage             : {inlier_report.spatial_coverage:.4f}")
        if reprojection_report:
            print(f"  Inlier Reprojection RMSE     : {reprojection_report.inlier_rmse:.3f} px")
    else:
        print("  Estimation Status            : SKIPPED / REJECTED (Gate rejected or insufficient inliers)")
    print("-" * 80)
    print("STAGE 6: Final Validation Decision")
    print(f"  Quality Score                : {validation_metrics.quality_score:.4f}")
    print(f"  FINAL DECISION               : {'[ACCEPT]' if validation_metrics.passed else '[REJECT]'} {validation_metrics.decision}")
    if not validation_metrics.passed:
        print(f"  Failure Stage                : {validation_metrics.failure_stage}")
        print(f"  Failure Reason               : {validation_metrics.failure_reason}")
    print("=" * 80)

    # 9. Save Visualizations
    cv2.imwrite(str(output_dir / "01_source.png"), src_img)
    cv2.imwrite(str(output_dir / "02_reference.png"), ref_img)
    cv2.imwrite(str(output_dir / "03_source_structure.png"), src_struct.composite)
    cv2.imwrite(str(output_dir / "04_reference_structure.png"), ref_struct.composite)

    inlier_mask = estimation_result.inlier_mask if (estimation_result and estimation_result.success) else None
    vis_corr = draw_matches_visualization(
        src_img,
        ref_img,
        best_match["source_points"],
        best_match["reference_points"],
        inlier_mask=inlier_mask,
        title=f"FLUX Correspondence (Matches: {best_match['num_matches']})",
    )
    cv2.imwrite(str(output_dir / "05_correspondences.png"), vis_corr)

    if registration_result and registration_result.registered_image is not None:
        cv2.imwrite(str(output_dir / "06_registered_image.png"), registration_result.registered_image)

    # Save structured JSON summary
    summary_data = {
        "source": str(source_path),
        "reference": str(reference_path),
        "execution_time_seconds": t_elapsed,
        "gsd_validation": {
            "source_gsd": gsd_result.source_gsd,
            "reference_gsd": gsd_result.reference_gsd,
            "scale_ratio": gsd_result.scale_ratio,
            "status": gsd_result.status,
            "warning": gsd_result.warning_message,
        },
        "candidate_search": {
            "num_candidates": len(candidate_evals),
            "best_candidate_bbox": best_candidate.get("bbox") if best_candidate else None,
            "best_candidate_score": best_candidate.get("score") if best_candidate else 0.0,
        },
        "correspondence": {
            "num_matches": best_match["num_matches"],
            "confidence": best_match["confidence"],
        },
        "evidence_gate": {
            "passed": gate_result.passed,
            "evidence_score": gate_result.evidence_score,
            "rejection_reason": gate_result.rejection_reason,
        },
        "validation_metrics": {
            "passed": validation_metrics.passed,
            "quality_score": validation_metrics.quality_score,
            "decision": validation_metrics.decision,
            "failure_stage": validation_metrics.failure_stage,
            "failure_reason": validation_metrics.failure_reason,
        },
    }

    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Artifacts successfully saved to: {output_dir.resolve()}")
    return summary_data


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="FLUX End-to-End Lunar Registration & Validation Pipeline.")
    parser.add_argument("--source", type=str, default="data/test/OHRC/ohrc_test_tile.png", help="Source image path")
    parser.add_argument("--reference", type=str, default="data/prepared/lunar_reference/LRO_gray.png", help="Reference image path")
    parser.add_argument("--source-gsd", type=float, default=None, help="Source GSD (m/px)")
    parser.add_argument("--reference-gsd", type=float, default=None, help="Reference GSD (m/px)")
    parser.add_argument("--output-dir", type=str, default="outputs/e2e_pipeline", help="Output directory")
    parser.add_argument("--ratio-threshold", type=float, default=0.75, help="Lowe's ratio test threshold")

    args = parser.parse_args()

    src_p = Path(args.source)
    if not src_p.is_absolute():
        src_p = PROJECT_ROOT / src_p

    ref_p = Path(args.reference)
    if not ref_p.is_absolute():
        ref_p = PROJECT_ROOT / ref_p

    out_p = Path(args.output_dir)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p

    source_gsd = args.source_gsd
    reference_gsd = args.reference_gsd
    if source_gsd is None and ("ohrc" in src_p.name.lower() or "chandrayaan" in src_p.name.lower()):
        source_gsd = 0.23
    if reference_gsd is None and ("lro" in ref_p.name.lower() or "quickmap" in ref_p.name.lower()):
        reference_gsd = 200.0

    run_e2e_pipeline(
        source_path=src_p,
        reference_path=ref_p,
        output_dir=out_p,
        source_gsd=source_gsd,
        reference_gsd=reference_gsd,
        ratio_threshold=args.ratio_threshold,
    )


if __name__ == "__main__":
    main()
