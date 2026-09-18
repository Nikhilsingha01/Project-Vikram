"""M3 End-to-End Validation Pipeline -- Synthetic Integration Test.

NOTE: THIS IS A SYNTHETIC INTEGRATION TEST.
All correspondence points in the primary test case are generated from a
*known ground-truth affine transformation* applied to a deterministic
synthetic lunar-like image pair.  This guarantees full control over the
expected outcome and makes the pipeline behaviour reproducible without
requiring real Chandrayaan-2 imagery.

Pipeline stages exercised
--------------------------
  M2 Interface (simulated)
    --> Stage 1 : Evidence Gate         (EvidenceGate)
    --> Stage 2 : Geometric Estimation  (GeometricEstimator / RANSAC)
    --> Stage 3 : Inlier Analysis       (InlierAnalyzer)
    --> Stage 4 : Reprojection Error    (compute_reprojection_errors)
    --> Stage 5 : Image Registration    (register_image)
    --> Stage 6 : Validation Metrics    (compute_validation_metrics)
    --> Final   : ACCEPT / REJECT decision

Failure cases demonstrated
--------------------------
  A. Insufficient correspondences  -- gate rejects before RANSAC
  B. Outlier-heavy correspondences -- RANSAC fails or low inlier ratio
  C. Invalid / degenerate transform -- estimation returns failure

Usage
-----
Run from the project root::

    python scripts/test_m3_pipeline.py

Outputs are saved under ``outputs/m3_validation/``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# M3 imports (all sub-modules fully implemented)
# ---------------------------------------------------------------------------
from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig, EvidenceGateResult
from src.validation.geometric_estimation import (
    GeometricEstimator,
    GeometricEstimatorConfig,
    TransformModel,
)
from src.validation.inlier_analysis import InlierAnalyzer
from src.validation.reprojection import compute_reprojection_errors
from src.validation.registration import RegistrationConfig, register_image
from src.validation.metrics import (
    DecisionThresholds,
    MetricsWeights,
    compute_validation_metrics,
)


# ===========================================================================
# Synthetic image and correspondence generation
# ===========================================================================

def _make_synthetic_image(seed: int = 0) -> np.ndarray:
    """Generate a structured 400x400 uint8 grayscale synthetic lunar image.

    The image contains crater-like rings, linear ridges, and Gaussian blobs
    to give SIFT meaningful texture.  It is fully deterministic given ``seed``.

    Args:
        seed: Random seed for noise.

    Returns:
        (400, 400) uint8 grayscale array.
    """
    rng = np.random.RandomState(seed)
    H, W = 400, 400
    img = np.full((H, W), 80, dtype=np.uint8)

    # Crater rings (bright rim + dark interior)
    craters: List[Tuple[int, int, int]] = [
        (80,  80,  35), (200,  60,  28), (320,  90,  22),
        (60, 200,  30), (200, 200,  40), (340, 200,  25),
        (90, 330,  20), (230, 340,  32), (350, 340,  18),
    ]
    for cx, cy, r in craters:
        cv2.circle(img, (cx, cy), r,       220, thickness=3)
        cv2.circle(img, (cx, cy), max(1, r - 8), 40, thickness=-1)

    # Linear ridge features
    for y in range(0, H, 60):
        cv2.line(img, (0, y + 15), (W, y + 10), 160, 1)

    # Gaussian blobs for texture
    for _ in range(20):
        bx, by = rng.randint(10, W - 10), rng.randint(10, H - 10)
        br = rng.randint(8, 20)
        cv2.circle(img, (bx, by), br, int(rng.uniform(130, 200)), -1)

    # Mild Gaussian noise
    noise = rng.randint(-12, 12, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _build_known_affine(
    angle_deg: float = 8.0,
    scale: float = 1.05,
    tx: float = 15.0,
    ty: float = -12.0,
    image_size: Tuple[int, int] = (400, 400),
) -> np.ndarray:
    """Build a 2x3 affine matrix encoding rotation + uniform scale + translation.

    Args:
        angle_deg: Rotation angle in degrees.
        scale: Uniform scale factor.
        tx: X translation in pixels.
        ty: Y translation in pixels.
        image_size: (H, W) of the source image (used to rotate around centre).

    Returns:
        (2, 3) float64 affine matrix.
    """
    H, W = image_size
    cx, cy = W / 2.0, H / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    M = np.array([
        [scale * cos_t, -scale * sin_t, tx + cx * (1 - scale * cos_t) + cy * scale * sin_t],
        [scale * sin_t,  scale * cos_t, ty + cy * (1 - scale * cos_t) - cx * scale * sin_t],
    ], dtype=np.float64)
    return M


def _make_synthetic_correspondence(
    M_affine: np.ndarray,
    image_shape: Tuple[int, int] = (400, 400),
    n_clean: int = 60,
    n_outliers: int = 10,
    noise_std: float = 0.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build an M2-format correspondence dict from a known ground-truth transform.

    Source points are sampled on a grid that avoids image edges.  Reference
    points are computed by applying the affine transform exactly, then
    small Gaussian noise is added to inlier pairs.  Outlier pairs have
    reference points drawn uniformly at random.

    NOTE: This function deliberately bypasses the M2 SIFT module to give
    a fully deterministic, noise-controlled test case.  The resulting dict
    matches the M2 data contract exactly.

    Args:
        M_affine: (2, 3) float64 ground-truth affine.
        image_shape: (H, W) of the image.
        n_clean: Number of inlier (correctly transformed) point pairs.
        n_outliers: Number of random outlier pairs.
        noise_std: Standard deviation of inlier coordinate noise in pixels.
        seed: Random seed.

    Returns:
        M2-compatible correspondence dict with keys:
            source_points, reference_points, confidence, matches,
            num_matches, num_keypoints_source, num_keypoints_reference.
    """
    rng = np.random.RandomState(seed)
    H, W = image_shape

    # Grid source points, 20px margin from edges
    margin = 20
    xs = np.linspace(margin, W - margin, int(np.sqrt(n_clean)) + 2)
    ys = np.linspace(margin, H - margin, int(np.sqrt(n_clean)) + 2)
    grid_x, grid_y = np.meshgrid(xs, ys)
    all_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # Subsample to exactly n_clean
    idx = rng.choice(len(all_pts), size=n_clean, replace=False)
    src_clean = all_pts[idx].astype(np.float32)

    # Apply affine transform: [x'; y'] = M[:, :2] @ [x; y]^T + M[:, 2]
    ones = np.ones((n_clean, 1), dtype=np.float64)
    src_h = np.hstack([src_clean.astype(np.float64), ones])  # (N, 3)
    ref_clean = (M_affine @ src_h.T).T.astype(np.float32)    # (N, 2)

    # Add small Gaussian noise to inlier reference points
    ref_clean += rng.normal(0, noise_std, ref_clean.shape).astype(np.float32)

    # Outlier pairs: random source, completely random reference
    if n_outliers > 0:
        src_out = rng.uniform([margin, margin], [W - margin, H - margin],
                               (n_outliers, 2)).astype(np.float32)
        ref_out = rng.uniform([margin, margin], [W - margin, H - margin],
                               (n_outliers, 2)).astype(np.float32)
        source_points = np.vstack([src_clean, src_out])
        reference_points = np.vstack([ref_clean, ref_out])
    else:
        source_points = src_clean
        reference_points = ref_clean

    n_total = source_points.shape[0]
    # Confidence: based on inlier fraction (synthetic proxy, not from SIFT)
    confidence = float(n_clean / n_total)

    matches = [
        {
            "source_pt": (float(source_points[i, 0]), float(source_points[i, 1])),
            "reference_pt": (float(reference_points[i, 0]), float(reference_points[i, 1])),
            "distance": float(rng.uniform(50, 150)),
            "ratio": float(rng.uniform(0.4, 0.75)),
        }
        for i in range(n_total)
    ]

    return {
        "source_points": source_points,
        "reference_points": reference_points,
        "confidence": confidence,
        "matches": matches,
        "num_matches": n_total,
        "num_keypoints_source": n_total + rng.randint(20, 60),
        "num_keypoints_reference": n_total + rng.randint(20, 60),
    }


# ===========================================================================
# Output image utilities
# ===========================================================================

def _draw_correspondences(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    max_draw: int = 80,
    title: str = "Correspondences",
) -> np.ndarray:
    """Draw side-by-side match visualisation coloured by inlier/outlier status.

    Args:
        src_img: Source image (grayscale).
        ref_img: Reference image (grayscale).
        src_pts: (N, 2) source coordinates.
        ref_pts: (N, 2) reference coordinates.
        inlier_mask: Optional (N,) bool.  If None all lines are drawn in cyan.
        max_draw: Maximum match lines to render.
        title: Text label rendered on the canvas.

    Returns:
        BGR canvas image with annotated match lines.
    """
    h1, w1 = src_img.shape[:2]
    h2, w2 = ref_img.shape[:2]
    ch = max(h1, h2)
    cw = w1 + w2 + 4  # 4-px gap

    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1 + 4:] = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    n = len(src_pts)
    step = max(1, n // max_draw) if n > max_draw else 1
    for i in range(0, n, step):
        p1 = (int(round(float(src_pts[i, 0]))), int(round(float(src_pts[i, 1]))))
        p2 = (int(round(float(ref_pts[i, 0]))) + w1 + 4,
              int(round(float(ref_pts[i, 1]))))
        if inlier_mask is not None:
            color = (0, 220, 0) if inlier_mask[i] else (0, 0, 220)
        else:
            color = (0, 220, 220)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _make_overlay(
    warped_img: np.ndarray,
    ref_img: np.ndarray,
    alpha: float = 0.5,
    title: str = "Overlay",
) -> np.ndarray:
    """Blend registered (warped) source over reference for visual alignment check.

    Args:
        warped_img: Warped source image (same shape as ref_img, grayscale).
        ref_img: Reference image (grayscale).
        alpha: Blend weight for warped (1-alpha for reference).
        title: Label text.

    Returns:
        BGR blended image.
    """
    warp_bgr = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    ref_bgr  = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    blend = np.clip(alpha * warp_bgr + (1.0 - alpha) * ref_bgr, 0, 255).astype(np.uint8)
    # Tint the warped channel green, reference red for false-colour overlay
    false_color = np.zeros_like(blend)
    false_color[:, :, 1] = warped_img  # green = source
    false_color[:, :, 2] = ref_img     # red   = reference
    cv2.putText(blend, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 200), 1, cv2.LINE_AA)
    return blend


# ===========================================================================
# Pipeline report printer
# ===========================================================================

def _section(title: str) -> None:
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def _subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def _row(label: str, value: Any, indent: int = 4) -> None:
    pad = " " * indent
    if isinstance(value, float):
        print(f"{pad}{label:<35}: {value:.4f}")
    else:
        print(f"{pad}{label:<35}: {value}")


def print_pipeline_report(
    scenario_name: str,
    correspondence: Dict[str, Any],
    gate_result: Any,
    estimation_result: Optional[Any],
    inlier_report: Optional[Any],
    reprojection_report: Optional[Any],
    registration_result: Optional[Any],
    metrics: Any,
) -> None:
    """Print the concise, human-readable pipeline report to stdout.

    Args:
        scenario_name: Display name for the scenario.
        correspondence: M2 correspondence dict.
        gate_result: EvidenceGateResult.
        estimation_result: EstimationResult or None.
        inlier_report: InlierReport or None.
        reprojection_report: ReprojectionReport or None.
        registration_result: RegistrationResult or None.
        metrics: ValidationMetrics.
    """
    _section(f"Scenario: {scenario_name}")

    _subsection("Correspondence (M2 output)")
    _row("Correspondence count", correspondence["num_matches"])
    _row("M2 confidence", correspondence["confidence"])

    _subsection("Stage 1: Evidence Gate")
    _row("Gate passed", gate_result.passed)
    _row("Evidence score", gate_result.evidence_score)
    if not gate_result.passed:
        _row("Rejection reason", gate_result.rejection_reason)

    if estimation_result is not None:
        _subsection("Stage 2: Geometric Estimation")
        _row("Transform model", estimation_result.transform_type)
        _row("Estimation success", estimation_result.success)
        if not estimation_result.success:
            _row("Failure reason", estimation_result.reason)

    if inlier_report is not None:
        _subsection("Stage 3: Inlier / Outlier Analysis")
        _row("Inlier count", inlier_report.inlier_count)
        _row("Outlier count", inlier_report.outlier_count)
        _row("Inlier ratio", inlier_report.inlier_ratio)
        _row("Spatial coverage (0-1)", inlier_report.spatial_coverage)

    if reprojection_report is not None:
        _subsection("Stage 4: Reprojection Error (inlier subset)")
        _row("Mean  (px)", reprojection_report.inlier_mean_error)
        _row("Median (px)", reprojection_report.inlier_median_error)
        _row("RMSE  (px)", reprojection_report.inlier_rmse)
        _row("Max   (px)", reprojection_report.inlier_max_error)

    if registration_result is not None:
        _subsection("Stage 5: Image Registration")
        _row("Warp succeeded", registration_result.quality_flags.get("warp_succeeded"))
        _row("Overlap nonzero", registration_result.quality_flags.get("overlap_nonzero"))
        _row("Overlap sufficient", registration_result.quality_flags.get("overlap_sufficient"))
        _row("Overlap bbox (x,y,w,h)", registration_result.overlap_bbox)

    _subsection("Stage 6: Validation Metrics")
    _row("Quality score (0-1)", metrics.quality_score)

    decision_str = f"{'[ACCEPT]' if metrics.passed else '[REJECT]'}  {metrics.decision}"
    print(f"\n  {'FINAL DECISION':<35}: {decision_str}")
    if not metrics.passed:
        _row("Failure stage", metrics.failure_stage)
        _row("Failure reason", metrics.failure_reason)


# ===========================================================================
# Individual pipeline stage runners
# ===========================================================================

def _run_full_pipeline(
    correspondence: Dict[str, Any],
    src_img: np.ndarray,
    ref_img: np.ndarray,
    transform_model: TransformModel = TransformModel.AFFINE,
    gate_config: Optional[EvidenceGateConfig] = None,
    est_config: Optional[GeometricEstimatorConfig] = None,
    thresholds: Optional[Any] = None,
    weights: Optional[MetricsWeights] = None,
) -> Dict[str, Any]:
    """Execute all 6 M3 stages in sequence and return all results.

    Args:
        correspondence: M2 correspondence dict (real or synthetic).
        src_img: Source grayscale image.
        ref_img: Reference grayscale image.
        transform_model: TransformModel enum to use for estimation.
        gate_config: Optional EvidenceGateConfig.
        est_config: Optional GeometricEstimatorConfig.
        thresholds: Optional DecisionThresholds.
        weights: Optional MetricsWeights.

    Returns:
        Dict with keys: gate, estimation, inlier, reprojection,
        registration, metrics, correspondence.
    """
    # --- Stage 1: Evidence Gate -------------------------------------------
    gate = EvidenceGate(config=gate_config or EvidenceGateConfig())
    gate_result = gate.evaluate(correspondence)

    # --- Stage 2: Geometric Estimation ------------------------------------
    estimation_result = None
    if gate_result.passed:
        cfg = est_config or GeometricEstimatorConfig(
            model=transform_model,
            ransac_reproj_threshold=5.0,
            ransac_confidence=0.995,
            min_inliers=6,
            min_inlier_ratio=0.15,
            refine_with_lm=True,
        )
        estimator = GeometricEstimator(config=cfg)
        estimation_result = estimator.estimate(correspondence)

    # --- Stage 3: Inlier Analysis -----------------------------------------
    inlier_report = None
    if estimation_result is not None and estimation_result.success:
        analyser = InlierAnalyzer(grid_divisions=4)
        inlier_report = analyser.analyse(
            inlier_mask=estimation_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            image_shape=src_img.shape[:2],
        )

    # --- Stage 4: Reprojection Error --------------------------------------
    reprojection_report = None
    if estimation_result is not None and estimation_result.success:
        reprojection_report = compute_reprojection_errors(
            source_points=np.asarray(correspondence["source_points"], dtype=np.float64),
            reference_points=np.asarray(correspondence["reference_points"], dtype=np.float64),
            transform_matrix=estimation_result.transform_matrix,
            inlier_mask=estimation_result.inlier_mask,
            model_type=estimation_result.transform_type.lower(),
            compute_symmetric=True,
        )

    # --- Stage 5: Image Registration --------------------------------------
    registration_result = None
    if estimation_result is not None and estimation_result.success:
        reg_config = RegistrationConfig(
            model_type=estimation_result.transform_type.lower(),
            compute_overlap=True,
        )
        registration_result = register_image(
            source_image=src_img,
            transform_matrix=estimation_result.transform_matrix,
            reference_shape=ref_img.shape[:2],
            config=reg_config,
        )

    # --- Stage 6: Validation Metrics --------------------------------------
    metrics = compute_validation_metrics(
        gate_result=gate_result,
        estimation_result=estimation_result,
        inlier_report=inlier_report,
        reprojection_report=reprojection_report,
        registration_result=registration_result,
        weights=weights,
        thresholds=thresholds,
    )

    return {
        "correspondence": correspondence,
        "gate": gate_result,
        "estimation": estimation_result,
        "inlier": inlier_report,
        "reprojection": reprojection_report,
        "registration": registration_result,
        "metrics": metrics,
    }


# ===========================================================================
# Scenario A -- Happy path: clean synthetic correspondence
# ===========================================================================

def run_scenario_happy_path(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    output_dir: Path,
) -> None:
    """Scenario: deterministic synthetic inlier-rich correspondence → ACCEPT.

    Generates 60 inlier pairs + 10 outliers from a known affine transform.
    All M3 stages run to completion; the expected decision is ACCEPT.
    """
    M_gt = _build_known_affine(angle_deg=8.0, scale=1.05, tx=15.0, ty=-12.0,
                                image_size=src_img.shape[:2])
    correspondence = _make_synthetic_correspondence(
        M_affine=M_gt,
        image_shape=src_img.shape[:2],
        n_clean=60,
        n_outliers=10,
        noise_std=0.5,
        seed=42,
    )

    results = _run_full_pipeline(
        correspondence=correspondence,
        src_img=src_img,
        ref_img=ref_img,
        transform_model=TransformModel.AFFINE,
    )

    print_pipeline_report(
        "Happy Path -- Known Affine (60 inliers + 10 outliers, noise=0.5px)",
        correspondence=correspondence,
        gate_result=results["gate"],
        estimation_result=results["estimation"],
        inlier_report=results["inlier"],
        reprojection_report=results["reprojection"],
        registration_result=results["registration"],
        metrics=results["metrics"],
    )

    # ---- Save output images ------------------------------------------------
    cv2.imwrite(str(output_dir / "01_source.png"), src_img)
    cv2.imwrite(str(output_dir / "02_reference.png"), ref_img)

    # Correspondence visualisation (no inlier mask yet at M2 stage)
    vis_all = _draw_correspondences(
        src_img, ref_img,
        correspondence["source_points"],
        correspondence["reference_points"],
        inlier_mask=None,
        title="All Correspondences (M2 output)",
    )
    cv2.imwrite(str(output_dir / "03_correspondences_all.png"), vis_all)

    # Inlier/outlier visualisation (after RANSAC)
    er = results["estimation"]
    if er is not None and er.inlier_mask is not None:
        vis_io = _draw_correspondences(
            src_img, ref_img,
            correspondence["source_points"],
            correspondence["reference_points"],
            inlier_mask=er.inlier_mask,
            title="Inliers (green) vs Outliers (red)",
        )
        cv2.imwrite(str(output_dir / "04_inlier_outlier.png"), vis_io)

    rr = results["registration"]
    if rr is not None:
        cv2.imwrite(str(output_dir / "05_registered_source.png"), rr.registered_image)
        overlay = _make_overlay(rr.registered_image, ref_img,
                                title="Registration Overlay (green=src, red=ref)")
        cv2.imwrite(str(output_dir / "06_registration_overlay.png"), overlay)

    print(f"\n  [Outputs] saved to {output_dir.resolve()}")


# ===========================================================================
# Scenario B -- Failure: insufficient correspondences
# ===========================================================================

def run_scenario_insufficient_matches(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    output_dir: Path,
) -> None:
    """Scenario: only 3 match pairs -- EvidenceGate should REJECT immediately.

    The gate is configured with min_matches=10 (default).  Three points are
    far below the hard gate threshold, so Stage 1 rejects and no downstream
    processing occurs.
    """
    M_gt = _build_known_affine(angle_deg=5.0, scale=1.0, tx=10.0, ty=5.0,
                                image_size=src_img.shape[:2])
    correspondence = _make_synthetic_correspondence(
        M_affine=M_gt,
        image_shape=src_img.shape[:2],
        n_clean=3,
        n_outliers=0,
        noise_std=0.5,
        seed=99,
    )

    results = _run_full_pipeline(
        correspondence=correspondence,
        src_img=src_img,
        ref_img=ref_img,
        transform_model=TransformModel.AFFINE,
        gate_config=EvidenceGateConfig(min_matches=10),
    )

    print_pipeline_report(
        "Failure Case A -- Insufficient Correspondences (3 matches, min=10)",
        correspondence=correspondence,
        gate_result=results["gate"],
        estimation_result=results["estimation"],
        inlier_report=results["inlier"],
        reprojection_report=results["reprojection"],
        registration_result=results["registration"],
        metrics=results["metrics"],
    )


# ===========================================================================
# Scenario C -- Failure: outlier-heavy (geometrically inconsistent)
# ===========================================================================

def run_scenario_outlier_heavy(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    output_dir: Path,
) -> None:
    """Scenario: 80% random outliers -- RANSAC should fail or produce low inlier ratio.

    The DecisionThresholds.min_inlier_ratio=0.25 means anything below 25%
    inlier fraction is rejected as geometrically inconsistent.
    """
    M_gt = _build_known_affine(angle_deg=3.0, scale=1.02, tx=8.0, ty=-5.0,
                                image_size=src_img.shape[:2])
    # 15 clean + 60 pure random outliers → ~20% inlier ratio if RANSAC works at all
    correspondence = _make_synthetic_correspondence(
        M_affine=M_gt,
        image_shape=src_img.shape[:2],
        n_clean=15,
        n_outliers=60,
        noise_std=1.0,
        seed=77,
    )

    results = _run_full_pipeline(
        correspondence=correspondence,
        src_img=src_img,
        ref_img=ref_img,
        transform_model=TransformModel.AFFINE,
        thresholds=DecisionThresholds(
            min_inlier_ratio=0.40,  # strict threshold to force rejection
            min_inlier_count=20,
        ),
    )

    print_pipeline_report(
        "Failure Case B -- Outlier-Heavy (15 inliers + 60 outliers, strict threshold)",
        correspondence=correspondence,
        gate_result=results["gate"],
        estimation_result=results["estimation"],
        inlier_report=results["inlier"],
        reprojection_report=results["reprojection"],
        registration_result=results["registration"],
        metrics=results["metrics"],
    )


# ===========================================================================
# Scenario D -- Failure: degenerate / invalid transform
# ===========================================================================

def run_scenario_degenerate_transform(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    output_dir: Path,
) -> None:
    """Scenario: all source points identical (collinear) -- estimation should fail.

    When all n source points map to essentially the same location, RANSAC
    cannot estimate a valid affine / homography and returns failure.
    We force this with n_clean=0 and all outlier points clustered around one spot.
    """
    H, W = src_img.shape[:2]

    # All source and reference points at the exact same pixel -- fully degenerate
    n = 20
    clustered_src = np.tile([[200.0, 200.0]], (n, 1)).astype(np.float32)
    clustered_ref = np.tile([[210.0, 210.0]], (n, 1)).astype(np.float32)

    correspondence: Dict[str, Any] = {
        "source_points": clustered_src,
        "reference_points": clustered_ref,
        "confidence": 0.55,
        "matches": [],
        "num_matches": n,
        "num_keypoints_source": n + 10,
        "num_keypoints_reference": n + 10,
    }

    results = _run_full_pipeline(
        correspondence=correspondence,
        src_img=src_img,
        ref_img=ref_img,
        transform_model=TransformModel.AFFINE,
        gate_config=EvidenceGateConfig(
            min_matches=10,
            min_spatial_spread=0.001,  # very low so gate passes
            min_confidence=0.1,
            min_evidence_score=0.01,
        ),
    )

    print_pipeline_report(
        "Failure Case C -- Degenerate (all points co-located, zero spatial spread)",
        correspondence=correspondence,
        gate_result=results["gate"],
        estimation_result=results["estimation"],
        inlier_report=results["inlier"],
        reprojection_report=results["reprojection"],
        registration_result=results["registration"],
        metrics=results["metrics"],
    )


# ===========================================================================
# Main entry point
# ===========================================================================

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "m3_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("#  FLUX -- M3 End-to-End Synthetic Validation Pipeline")
    print("#")
    print("#  NOTE: This is a SYNTHETIC INTEGRATION TEST.")
    print("#  Correspondence points are generated from a KNOWN ground-truth")
    print("#  affine transform applied to deterministic synthetic imagery.")
    print("#  No accuracy percentages from synthetic data represent scientific")
    print("#  performance claims for real Chandrayaan-2 imagery.")
    print("#" * 70)

    t0 = time.perf_counter()

    # Generate synthetic images (deterministic)
    src_img = _make_synthetic_image(seed=0)
    ref_img = _make_synthetic_image(seed=1)  # slightly different texture / noise

    # -----------------------------------------------------------------------
    # Scenario 1 -- Happy path
    # -----------------------------------------------------------------------
    run_scenario_happy_path(src_img, ref_img, output_dir)

    # -----------------------------------------------------------------------
    # Scenario 2 -- Insufficient correspondences
    # -----------------------------------------------------------------------
    run_scenario_insufficient_matches(src_img, ref_img, output_dir)

    # -----------------------------------------------------------------------
    # Scenario 3 -- Outlier-heavy
    # -----------------------------------------------------------------------
    run_scenario_outlier_heavy(src_img, ref_img, output_dir)

    # -----------------------------------------------------------------------
    # Scenario 4 -- Degenerate / invalid transform
    # -----------------------------------------------------------------------
    run_scenario_degenerate_transform(src_img, ref_img, output_dir)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  All scenarios completed in {elapsed:.2f}s")
    print(f"  Output images: {output_dir.resolve()}")
    print(f"{'=' * 70}\n")

    # List saved files
    saved = sorted(output_dir.glob("*.png"))
    if saved:
        print("  Output images generated:")
        for f in saved:
            print(f"    {f.name}")
    else:
        print("  (No output images; all scenarios rejected before registration)")


if __name__ == "__main__":
    main()
