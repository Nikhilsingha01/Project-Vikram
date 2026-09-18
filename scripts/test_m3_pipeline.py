"""M3 Pipeline Test & Demonstration Script.

Executes the complete Member 3 (M3) Evidence-Gated Validation pipeline on
the output of the M2 correspondence stage:

    Stage 1 — Evidence Gate        (EvidenceGate)
    Stage 2 — Geometric Estimation (GeometricEstimator / RANSAC)
    Stage 3 — Inlier Analysis      (InlierAnalyzer)
    Stage 4 — Reprojection         (compute_reprojection_errors)
    Stage 5 — Registration         (register_image)
    Stage 6 — Metrics              (compute_validation_metrics)

Usage
-----
Run from the project root:

    python scripts/test_m3_pipeline.py \\
        --source  data/test/source_lunar_test.png \\
        --reference data/test/reference_lunar_test.png \\
        --output-dir outputs/m3_test

All M2 pipeline stages (structure extraction, candidate search, SIFT
correspondence) are invoked exactly as in test_m2_pipeline.py — this
script consumes their output without modifying any M2 module.

NOTE: M3 algorithm stubs raise NotImplementedError.  This script is
      structured so that each stage's NotImplementedError is caught and
      reported gracefully, allowing the pipeline skeleton to be executed
      end-to-end for structural verification even before implementation.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# M2 imports (READ-ONLY — do NOT modify M2 modules)
# ---------------------------------------------------------------------------
from src.candidate_search import generate_candidate_rois
from src.correspondence import match_sift_features, match_with_candidate_roi
from src.structure import extract_structure_representation

# ---------------------------------------------------------------------------
# M3 imports
# ---------------------------------------------------------------------------
from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig
from src.validation.geometric_estimation import (
    GeometricEstimator,
    GeometricEstimatorConfig,
    TransformModel,
)
from src.validation.inlier_analysis import InlierAnalyzer, analyse_inliers
from src.validation.reprojection import compute_reprojection_errors
from src.validation.registration import RegistrationConfig, register_image
from src.validation.metrics import MetricsWeights, compute_validation_metrics


# ---------------------------------------------------------------------------
# Image loading utility (mirrors test_m2_pipeline.py — not duplicated there)
# ---------------------------------------------------------------------------


def load_lunar_image(image_path: Path) -> np.ndarray:
    """Load a lunar image (PNG/JPG/TIFF, 8-bit or 16-bit) as uint8 grayscale.

    Args:
        image_path: Path to source or reference image.

    Returns:
        np.ndarray: 2D uint8 grayscale image.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Failed to read image: {image_path}")

    if raw.ndim == 3:
        if raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported array ndim: {raw.ndim}")

    if gray.dtype == np.uint16:
        mn, mx = float(gray.min()), float(gray.max())
        gray = ((gray.astype(np.float32) - mn) / max(mx - mn, 1e-6) * 255).astype(np.uint8)
    elif gray.dtype in (np.float32, np.float64):
        mn, mx = float(gray.min()), float(gray.max())
        gray = ((gray - mn) / max(mx - mn, 1e-6) * 255).astype(np.uint8)
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


# ---------------------------------------------------------------------------
# M3 stage runners (each wrapped to handle NotImplementedError gracefully)
# ---------------------------------------------------------------------------


def run_stage_1_evidence_gate(
    correspondence: Dict[str, Any],
    config: Optional[EvidenceGateConfig] = None,
) -> Optional[Any]:
    """Run M3 Stage 1: Evidence Gate.

    Args:
        correspondence: M2 correspondence dict.
        config: Optional EvidenceGateConfig.

    Returns:
        EvidenceGateResult or None if not yet implemented.
    """
    # TODO (M3-SCRIPT-S1): Remove the NotImplementedError guard when
    #   EvidenceGate.evaluate() is implemented.
    print("\n[M3 Stage 1] Evidence Gate")
    try:
        gate = EvidenceGate(config=config)
        result = gate.evaluate(correspondence)
        print(f"  Passed         : {result.passed}")
        print(f"  Evidence score : {result.evidence_score:.4f}")
        if not result.passed:
            print(f"  Rejection      : {result.rejection_reason}")
        return result
    except NotImplementedError:
        print("  [STUB] EvidenceGate.evaluate() not yet implemented — skipping.")
        return None


def run_stage_2_geometric_estimation(
    correspondence: Dict[str, Any],
    config: Optional[GeometricEstimatorConfig] = None,
) -> Optional[Any]:
    """Run M3 Stage 2: Robust Geometric Estimation.

    Args:
        correspondence: M2 correspondence dict.
        config: Optional GeometricEstimatorConfig.

    Returns:
        EstimationResult or None if not yet implemented.
    """
    # TODO (M3-SCRIPT-S2): Remove guard when GeometricEstimator.estimate() is implemented.
    print("\n[M3 Stage 2] Geometric Estimation")
    try:
        estimator = GeometricEstimator(config=config)
        result = estimator.estimate(correspondence)
        print(f"  Success        : {result.success}")
        if result.success:
            print(f"  Inlier count   : {result.inlier_count}")
            print(f"  Inlier ratio   : {result.inlier_ratio:.4f}")
            print(f"  Model          : {result.model.name}")
        else:
            print(f"  Failure reason : {result.failure_reason}")
        return result
    except NotImplementedError:
        print("  [STUB] GeometricEstimator.estimate() not yet implemented — skipping.")
        return None


def run_stage_3_inlier_analysis(
    correspondence: Dict[str, Any],
    estimation_result: Optional[Any],
) -> Optional[Any]:
    """Run M3 Stage 3: Inlier Analysis.

    Args:
        correspondence: M2 correspondence dict.
        estimation_result: EstimationResult from Stage 2.

    Returns:
        InlierReport or None if not yet implemented or estimation failed.
    """
    # TODO (M3-SCRIPT-S3): Remove guard when InlierAnalyzer.analyse() is implemented.
    print("\n[M3 Stage 3] Inlier Analysis")
    if estimation_result is None or not estimation_result.success:
        print("  Skipped (no valid estimation result from Stage 2).")
        return None
    try:
        analyser = InlierAnalyzer(grid_divisions=4)
        report = analyser.analyse(
            inlier_mask=estimation_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            transform_matrix=estimation_result.transform_matrix,
        )
        print(f"  Inlier count      : {report.inlier_count}")
        print(f"  Outlier count     : {report.outlier_count}")
        print(f"  Inlier ratio      : {report.inlier_ratio:.4f}")
        print(f"  Spatial uniformity: {report.spatial_uniformity:.4f}")
        return report
    except NotImplementedError:
        print("  [STUB] InlierAnalyzer.analyse() not yet implemented — skipping.")
        return None


def run_stage_4_reprojection(
    correspondence: Dict[str, Any],
    estimation_result: Optional[Any],
) -> Optional[Any]:
    """Run M3 Stage 4: Reprojection Error Computation.

    Args:
        correspondence: M2 correspondence dict.
        estimation_result: EstimationResult from Stage 2.

    Returns:
        ReprojectionReport or None if not yet implemented or estimation failed.
    """
    # TODO (M3-SCRIPT-S4): Remove guard when compute_reprojection_errors() is implemented.
    print("\n[M3 Stage 4] Reprojection Error")
    if estimation_result is None or not estimation_result.success:
        print("  Skipped (no valid estimation result from Stage 2).")
        return None
    try:
        report = compute_reprojection_errors(
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            transform_matrix=estimation_result.transform_matrix,
            inlier_mask=estimation_result.inlier_mask,
            model_type=estimation_result.model.name.lower(),
            compute_symmetric=True,
        )
        print(f"  Overall RMSE      : {report.overall_rmse:.4f} px")
        print(f"  Inlier mean error : {report.inlier_mean_error:.4f} px")
        print(f"  Inlier median err : {report.inlier_median_error:.4f} px")
        print(f"  Outlier mean error: {report.outlier_mean_error:.4f} px")
        return report
    except NotImplementedError:
        print("  [STUB] compute_reprojection_errors() not yet implemented — skipping.")
        return None


def run_stage_5_registration(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    estimation_result: Optional[Any],
    output_dir: Path,
) -> Optional[Any]:
    """Run M3 Stage 5: Image Registration (warping).

    Args:
        source_image: Source lunar image (uint8 grayscale).
        reference_image: Reference lunar image (uint8 grayscale).
        estimation_result: EstimationResult from Stage 2.
        output_dir: Directory for saving the registered image.

    Returns:
        RegistrationResult or None if not yet implemented or estimation failed.
    """
    # TODO (M3-SCRIPT-S5): Remove guard when register_image() is implemented.
    print("\n[M3 Stage 5] Image Registration")
    if estimation_result is None or not estimation_result.success:
        print("  Skipped (no valid estimation result from Stage 2).")
        return None
    try:
        config = RegistrationConfig(
            model_type=estimation_result.model.name.lower(),
            compute_overlap=True,
        )
        result = register_image(
            source_image=source_image,
            transform_matrix=estimation_result.transform_matrix,
            reference_shape=reference_image.shape[:2],
            config=config,
        )
        print(f"  Warp succeeded : {result.quality_flags.get('warp_succeeded', False)}")
        print(f"  Overlap bbox   : {result.overlap_bbox}")

        # Save registered image
        out_path = output_dir / "07_registered_source.png"
        cv2.imwrite(str(out_path), result.registered_image)
        print(f"  Saved          : {out_path.name}")
        return result
    except NotImplementedError:
        print("  [STUB] register_image() not yet implemented — skipping.")
        return None


def run_stage_6_metrics(
    gate_result: Optional[Any],
    estimation_result: Optional[Any],
    inlier_report: Optional[Any],
    reprojection_report: Optional[Any],
    registration_result: Optional[Any],
) -> Optional[Any]:
    """Run M3 Stage 6: Validation Metrics Aggregation.

    Args:
        gate_result: EvidenceGateResult from Stage 1.
        estimation_result: EstimationResult from Stage 2.
        inlier_report: InlierReport from Stage 3.
        reprojection_report: ReprojectionReport from Stage 4.
        registration_result: RegistrationResult from Stage 5.

    Returns:
        ValidationMetrics or None if not yet implemented or gate_result is None.
    """
    # TODO (M3-SCRIPT-S6): Remove guard when compute_validation_metrics() is implemented.
    print("\n[M3 Stage 6] Validation Metrics")
    if gate_result is None:
        print("  Skipped (no gate result from Stage 1).")
        return None
    try:
        metrics = compute_validation_metrics(
            gate_result=gate_result,
            estimation_result=estimation_result,
            inlier_report=inlier_report,
            reprojection_report=reprojection_report,
            registration_result=registration_result,
        )
        print(f"  Quality score  : {metrics.quality_score:.4f}")
        print(f"  Passed         : {metrics.passed}")
        if not metrics.passed:
            print(f"  Failure stage  : {metrics.failure_stage}")
            print(f"  Failure reason : {metrics.failure_reason}")
        return metrics
    except NotImplementedError:
        print("  [STUB] compute_validation_metrics() not yet implemented — skipping.")
        return None


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------


def run_m3_pipeline(
    source_path: Path,
    reference_path: Path,
    output_dir: Path,
    ratio_threshold: float = 0.75,
    min_matches: int = 10,
    transform_model: str = "homography",
) -> None:
    """Execute complete M2 + M3 pipeline and report results.

    M2 stages are run first to produce the correspondence dict, then M3
    stages are applied sequentially.

    Args:
        source_path: Path to source lunar image.
        reference_path: Path to reference lunar image.
        output_dir: Directory for output visualisations.
        ratio_threshold: Lowe's ratio test threshold for M2 SIFT matching.
        min_matches: Minimum matches required by EvidenceGate.
        transform_model: Transform model for geometric estimation.
            One of 'homography', 'affine', 'similarity', 'translation'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # M2: Load images and run correspondence pipeline
    # ------------------------------------------------------------------
    print("=" * 65)
    print("FLUX — M3 Pipeline Execution Report")
    print("=" * 65)

    source_img = load_lunar_image(source_path)
    reference_img = load_lunar_image(reference_path)
    print(f"Source    : {source_path.name}  {source_img.shape}")
    print(f"Reference : {reference_path.name}  {reference_img.shape}")

    # M2 Stage 1+2: Structure extraction
    src_structure = extract_structure_representation(source_img)
    ref_structure = extract_structure_representation(reference_img)

    # M2 Stage 3: Candidate search
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
    best_candidate = search_result["best_candidate"]

    # M2 Stage 4: Fine SIFT correspondence
    if best_candidate is not None:
        correspondence = match_with_candidate_roi(
            source_image=source_img,
            reference_image=reference_img,
            candidate_roi=best_candidate,
            ratio_threshold=ratio_threshold,
        )
    else:
        correspondence = match_sift_features(
            source_image=source_img,
            reference_image=reference_img,
            ratio_threshold=ratio_threshold,
        )

    print(f"\nM2 Correspondence Output:")
    print(f"  num_matches : {correspondence['num_matches']}")
    print(f"  confidence  : {correspondence['confidence']:.4f}")
    print(f"  source_points shape : {correspondence['source_points'].shape}")
    print(f"  reference_points shape: {correspondence['reference_points'].shape}")

    # ------------------------------------------------------------------
    # M3: Validation pipeline
    # ------------------------------------------------------------------
    model_enum = {
        "homography": TransformModel.HOMOGRAPHY,
        "affine": TransformModel.AFFINE,
        "similarity": TransformModel.SIMILARITY,
        "translation": TransformModel.TRANSLATION,
    }.get(transform_model.lower(), TransformModel.HOMOGRAPHY)

    # Stage 1: Evidence Gate
    gate_cfg = EvidenceGateConfig(min_matches=min_matches)
    gate_result = run_stage_1_evidence_gate(correspondence, config=gate_cfg)

    # Stage 2: Geometric Estimation
    est_cfg = GeometricEstimatorConfig(
        model=model_enum,
        ransac_reproj_threshold=5.0,
        ransac_confidence=0.995,
        min_inliers=8,
    )
    estimation_result = run_stage_2_geometric_estimation(correspondence, config=est_cfg)

    # Stage 3: Inlier Analysis
    inlier_report = run_stage_3_inlier_analysis(correspondence, estimation_result)

    # Stage 4: Reprojection
    reprojection_report = run_stage_4_reprojection(correspondence, estimation_result)

    # Stage 5: Registration
    registration_result = run_stage_5_registration(
        source_img, reference_img, estimation_result, output_dir
    )

    # Stage 6: Metrics
    metrics = run_stage_6_metrics(
        gate_result=gate_result,
        estimation_result=estimation_result,
        inlier_report=inlier_report,
        reprojection_report=reprojection_report,
        registration_result=registration_result,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("M3 Pipeline Summary")
    print("=" * 65)
    stages = [
        ("Stage 1 Evidence Gate",       gate_result),
        ("Stage 2 Geometric Estimation", estimation_result),
        ("Stage 3 Inlier Analysis",      inlier_report),
        ("Stage 4 Reprojection",         reprojection_report),
        ("Stage 5 Registration",         registration_result),
        ("Stage 6 Metrics",              metrics),
    ]
    for name, result in stages:
        status = "OK (stub)" if result is None else "COMPLETED"
        print(f"  {name:<35}: {status}")

    if metrics is not None:
        print(f"\n  Overall quality score : {metrics.quality_score:.4f}")
        print(f"  Pipeline passed       : {metrics.passed}")

    print("=" * 65)
    print(f"Outputs saved to: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the M3 pipeline test script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Member 3 (M3) Pipeline: Evidence-Gated Validation, Robust Geometric "
            "Estimation, Registration, and Metrics."
        )
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/test/source_lunar_test.png",
        help="Path to source lunar image.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/test/reference_lunar_test.png",
        help="Path to reference lunar image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/m3_test",
        help="Directory for output visualisations.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for M2 SIFT matching.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="Minimum matches required by M3 Evidence Gate.",
    )
    parser.add_argument(
        "--transform-model",
        type=str,
        default="homography",
        choices=["homography", "affine", "similarity", "translation"],
        help="Geometric transform model for M3 estimation.",
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

    run_m3_pipeline(
        source_path=source_path,
        reference_path=reference_path,
        output_dir=output_dir,
        ratio_threshold=args.ratio_threshold,
        min_matches=args.min_matches,
        transform_model=args.transform_model,
    )


if __name__ == "__main__":
    main()
