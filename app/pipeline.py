"""Application-facing orchestration for the real M1 → M2 → M3 → M4 pipeline.

This module is the integration layer only: it validates image inputs, invokes the
existing M1/M2/M3/M4 implementations from the src package, and publishes the
resulting metrics and visual artifacts for the MVP/demo entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.candidate_search import generate_candidate_rois
from src.correspondence import match_sift_features, match_with_candidate_roi
from src.registration.subpixel_refinement import refine_subpixel_points
from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig
from src.validation.geometric_estimation import (
    GeometricEstimator,
    GeometricEstimatorConfig,
    TransformModel,
)
from src.validation.inlier_analysis import analyse_inliers
from src.validation.metrics import compute_validation_metrics
from src.validation.reprojection import compute_reprojection_errors
from src.validation.registration import RegistrationConfig, register_image


LOGGER = logging.getLogger("app.pipeline")

DEFAULT_SOURCE_IMAGE = Path("data/raw/lunar_reference/LRO/quickmap-lroc.png")
DEFAULT_REFERENCE_IMAGE = Path("data/prepared/lunar_reference/LRO_gray.png")
DEFAULT_OUTPUT_DIR = Path("data/outputs")


@dataclass
class PipelineConfig:
    """Centralized configuration for the integrated M1 → M2 → M3 → M4 pipeline."""

    m1_scales: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    m1_top_k_per_scale: int = 3
    m1_min_score: float = 0.05
    m1_max_candidates: int = 2
    m2_ratio_threshold: float = 0.75
    min_matches: int = 5
    min_confidence: float = 0.05
    min_spatial_spread: float = 0.01
    min_evidence_score: float = 0.05
    model: TransformModel = TransformModel.HOMOGRAPHY
    ransac_reproj_threshold: float = 10.0
    ransac_max_iters: int = 1000
    min_inliers: int = 4
    min_inlier_ratio: float = 0.05
    registration_model_type: str = "homography"
    m4_window_size: Tuple[int, int] = (5, 5)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    visualization_name: str = "mvp_pipeline_visualization.png"
    result_name: str = "mvp_pipeline_result.json"

    def evidence_gate_config(self) -> EvidenceGateConfig:
        return EvidenceGateConfig(
            min_matches=self.min_matches,
            min_confidence=self.min_confidence,
            min_spatial_spread=self.min_spatial_spread,
            min_evidence_score=self.min_evidence_score,
        )

    def geometric_config(self) -> GeometricEstimatorConfig:
        return GeometricEstimatorConfig(
            model=self.model,
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
            ransac_reproj_threshold=self.ransac_reproj_threshold,
            ransac_max_iters=self.ransac_max_iters,
            refine_with_lm=True,
        )


@dataclass
class PipelineInput:
    """Validated input for the orchestration layer."""

    source_path: Optional[Path] = None
    reference_path: Optional[Path] = None
    source_image: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None


@dataclass
class StageResult:
    """Per-stage outcome for the pipeline orchestrator."""

    name: str
    executed: bool
    success: bool
    status: str
    counts: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M1Result:
    executed: bool
    success: bool
    candidate_count: int
    total_raw_candidates: int
    best_candidate: Optional[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M2Result:
    executed: bool
    success: bool
    num_matches: int
    confidence: float
    source_points: Optional[np.ndarray]
    reference_points: Optional[np.ndarray]
    candidate_roi: Optional[Dict[str, Any]]
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.source_points is not None:
            data["source_points"] = self.source_points.tolist()
        if self.reference_points is not None:
            data["reference_points"] = self.reference_points.tolist()
        return data


@dataclass
class M3Result:
    executed: bool
    success: bool
    gate_passed: bool
    estimation_success: bool
    num_matches: int
    num_inliers: int
    inlier_ratio: Optional[float]
    decision: Optional[str]
    metrics: Optional[Dict[str, Any]]
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class M4Result:
    executed: bool
    success: bool
    num_refined_points: int
    refined_source_points: Optional[np.ndarray]
    refined_reference_points: Optional[np.ndarray]
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.refined_source_points is not None:
            data["refined_source_points"] = self.refined_source_points.tolist()
        if self.refined_reference_points is not None:
            data["refined_reference_points"] = self.refined_reference_points.tolist()
        return data


@dataclass
class PipelineResult:
    """Final result object returned by the real M1->M4 pipeline."""

    success: bool
    status: str
    source_path: Optional[str]
    reference_path: Optional[str]
    source_shape: Optional[Tuple[int, int]]
    reference_shape: Optional[Tuple[int, int]]
    final_matches: int = 0
    final_inliers: int = 0
    final_inlier_ratio: Optional[float] = None
    final_confidence: Optional[float] = None
    final_rmse: Optional[float] = None
    final_mean_error: Optional[float] = None
    final_max_error: Optional[float] = None
    final_quality_score: Optional[float] = None
    failure_reason: Optional[str] = None
    processing_steps: List[Dict[str, str]] = field(default_factory=list)
    stage_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    visualization_path: Optional[str] = None
    result_path: Optional[str] = None
    total_processing_time_ms: Optional[int] = None
    m1: Optional[M1Result] = None
    m2: Optional[M2Result] = None
    m3: Optional[M3Result] = None
    m4: Optional[M4Result] = None
    registered_image: Optional[np.ndarray] = None
    demo_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.registered_image is not None:
            data["registered_image"] = None
        return data


def run_pipeline(
    source: Union[str, Path, np.ndarray],
    reference: Union[str, Path, np.ndarray],
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Execute the end-to-end M1→M2→M3→M4 pipeline for two real images."""
    cfg = config or PipelineConfig()
    total_start = time.perf_counter()
    input_data = _resolve_inputs(source, reference)

    result = PipelineResult(
        success=False,
        status="rejected",
        source_path=str(input_data.source_path) if input_data.source_path else None,
        reference_path=str(input_data.reference_path) if input_data.reference_path else None,
        source_shape=input_data.source_image.shape[:2] if input_data.source_image is not None else None,
        reference_shape=input_data.reference_image.shape[:2] if input_data.reference_image is not None else None,
        processing_steps=[],
        stage_results={},
        total_processing_time_ms=None,
        demo_mode=False,
    )

    if input_data.source_image is None or input_data.reference_image is None:
        raise ValueError("Both source and reference images must be valid and readable.")

    source_img = _to_grayscale(input_data.source_image, "source_image")
    reference_img = _to_grayscale(input_data.reference_image, "reference_image")
    result.source_shape = source_img.shape[:2]
    result.reference_shape = reference_img.shape[:2]

    # M1: candidate generation
    m1_start = time.perf_counter()
    m1_result = _run_m1(source_img, reference_img, cfg)
    result.m1 = m1_result
    result.stage_results["m1"] = {"executed": m1_result.executed, "success": m1_result.success, "details": m1_result.to_dict()}
    result.processing_steps.append({"name": "M1", "status": "completed" if m1_result.success else "failed", "detail": m1_result.error or "Coarse candidate search complete."})
    if not m1_result.success:
        result.failure_reason = f"M1 failed: {m1_result.error}"
        result.status = "rejected"
        result.success = False
        result.processing_steps.append({"name": "PIPELINE", "status": "failed", "detail": result.failure_reason})
        result.total_processing_time_ms = int((time.perf_counter() - total_start) * 1000)
        return result

    # M2: fine correspondence on best candidate ROI
    m2_start = time.perf_counter()
    m2_result = _run_m2(source_img, reference_img, m1_result, cfg)
    result.m2 = m2_result
    result.stage_results["m2"] = {"executed": m2_result.executed, "success": m2_result.success, "details": m2_result.to_dict()}
    result.processing_steps.append({"name": "M2", "status": "completed" if m2_result.success else "failed", "detail": m2_result.error or f"Matched {m2_result.num_matches} feature pairs."})
    if not m2_result.success:
        result.failure_reason = f"M2 failed: {m2_result.error}"
        result.status = "rejected"
        result.success = False
        result.processing_steps.append({"name": "PIPELINE", "status": "failed", "detail": result.failure_reason})
        result.total_processing_time_ms = int((time.perf_counter() - total_start) * 1000)
        return result

    # M3: validation/registration pipeline
    m3_start = time.perf_counter()
    m3_result, registration_result = _run_m3(source_img, reference_img, m2_result, cfg)
    result.m3 = m3_result
    result.stage_results["m3"] = {"executed": m3_result.executed, "success": m3_result.success, "details": m3_result.to_dict()}
    result.processing_steps.append({"name": "M3", "status": "completed" if m3_result.success else "failed", "detail": m3_result.error or f"Validation decision: {m3_result.decision}."})
    if not m3_result.success:
        result.failure_reason = f"M3 failed: {m3_result.error}"
        result.status = "rejected"
        result.success = False
        result.processing_steps.append({"name": "PIPELINE", "status": "failed", "detail": result.failure_reason})
        result.total_processing_time_ms = int((time.perf_counter() - total_start) * 1000)
        return result

    # M4: sub-pixel refinement
    m4_start = time.perf_counter()
    m4_result = _run_m4(source_img, reference_img, m2_result, registration_result, cfg)
    result.m4 = m4_result
    result.stage_results["m4"] = {"executed": m4_result.executed, "success": m4_result.success, "details": m4_result.to_dict()}
    result.processing_steps.append({"name": "M4", "status": "completed" if m4_result.success else "failed", "detail": m4_result.error or f"Refined {m4_result.num_refined_points} points."})
    if not m4_result.success:
        result.failure_reason = f"M4 failed: {m4_result.error}"
        result.status = "rejected"
        result.success = False
        result.processing_steps.append({"name": "PIPELINE", "status": "failed", "detail": result.failure_reason})
        result.total_processing_time_ms = int((time.perf_counter() - total_start) * 1000)
        return result

    result.success = True
    result.status = "accepted"
    result.final_matches = int(m2_result.num_matches)
    result.final_inliers = int(m3_result.num_inliers)
    result.final_inlier_ratio = (
        float(m3_result.num_inliers) / float(m2_result.num_matches)
        if m2_result.num_matches > 0
        else None
    )
    result.final_confidence = float(m2_result.confidence)
    if m3_result.metrics:
        result.final_rmse = m3_result.metrics.get("reprojection_rmse_px")
        result.final_mean_error = m3_result.metrics.get("reprojection_mean_px")
        result.final_max_error = m3_result.metrics.get("reprojection_max_px")
    result.final_quality_score = float(m3_result.metrics.get("quality_score")) if m3_result.metrics else None
    result.registered_image = registration_result.registered_image if registration_result else None
    result.total_processing_time_ms = int((time.perf_counter() - total_start) * 1000)
    result.processing_steps.append({"name": "PIPELINE", "status": "completed", "detail": "M1 → M2 → M3 → M4 pipeline completed successfully."})
    result.m1.duration_ms = int((time.perf_counter() - m1_start) * 1000)
    result.m2.duration_ms = int((time.perf_counter() - m2_start) * 1000)
    result.m3.duration_ms = int((time.perf_counter() - m3_start) * 1000)
    result.m4.duration_ms = int((time.perf_counter() - m4_start) * 1000)
    return result


def _resolve_inputs(
    source: Union[str, Path, np.ndarray],
    reference: Union[str, Path, np.ndarray],
) -> PipelineInput:
    if isinstance(source, np.ndarray):
        source_img = source
        source_path = None
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        source_img = _load_image(source_path)

    if isinstance(reference, np.ndarray):
        reference_img = reference
        reference_path = None
    else:
        reference_path = Path(reference)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        reference_img = _load_image(reference_path)

    return PipelineInput(source_path=source_path, reference_path=reference_path, source_image=source_img, reference_image=reference_img)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image at: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return _to_uint8(image)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D grayscale or 3D color; got shape {array.shape}.")
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.uint16:
        minimum = array.min()
        maximum = array.max()
        if maximum > minimum:
            return ((array - minimum) / float(maximum - minimum) * 255.0).astype(np.uint8)
        return np.zeros_like(array, dtype=np.uint8)
    if array.dtype in (np.float32, np.float64):
        minimum = array.min()
        maximum = array.max()
        if maximum > minimum:
            norm = (array - minimum) / float(maximum - minimum)
            return (norm * 255.0).astype(np.uint8)
        return (array * 255.0).astype(np.uint8)
    return cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _to_grayscale(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8)
    if array.ndim == 3 and array.shape[2] in (1, 3, 4):
        if array.shape[2] == 1:
            return array[:, :, 0].astype(np.uint8)
        code = cv2.COLOR_BGR2GRAY if array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(array, code)
    raise ValueError(f"{name} must be a 2-D grayscale or 3/4-channel color image.")


def _run_m1(source_img: np.ndarray, reference_img: np.ndarray, config: PipelineConfig) -> M1Result:
    start = time.perf_counter()
    try:
        result = generate_candidate_rois(
            source_image=source_img,
            reference_image=reference_img,
            scales=config.m1_scales,
            top_k_per_scale=config.m1_top_k_per_scale,
            min_score=config.m1_min_score,
            max_candidates=config.m1_max_candidates,
        )
        candidates = result.get("candidates", [])
        best_candidate = result.get("best_candidate")
        count = result.get("num_candidates_retained", len(candidates))
        success = bool(candidates or best_candidate is not None)
        return M1Result(
            executed=True,
            success=success,
            candidate_count=count,
            total_raw_candidates=int(result.get("total_raw_candidates", 0)),
            best_candidate=best_candidate,
            candidates=candidates,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=None if success else "No candidate regions survived M1 filtering.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("M1 candidate generation failed")
        return M1Result(
            executed=True,
            success=False,
            candidate_count=0,
            total_raw_candidates=0,
            best_candidate=None,
            candidates=[],
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc),
        )


def _run_m2(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    m1_result: M1Result,
    config: PipelineConfig,
) -> M2Result:
    start = time.perf_counter()
    try:
        if m1_result.best_candidate is not None:
            result = match_with_candidate_roi(
                source_image=source_img,
                reference_image=reference_img,
                candidate_roi=m1_result.best_candidate,
                ratio_threshold=config.m2_ratio_threshold,
            )
            candidate_roi = m1_result.best_candidate
        else:
            result = match_sift_features(
                source_image=source_img,
                reference_image=reference_img,
                max_features=5000,
                ratio_threshold=config.m2_ratio_threshold,
            )
            candidate_roi = None

        source_points = result.get("source_points")
        reference_points = result.get("reference_points")
        num_matches = int(result.get("num_matches", 0))
        confidence = float(result.get("confidence", 0.0))
        success = num_matches > 0
        return M2Result(
            executed=True,
            success=success,
            num_matches=num_matches,
            confidence=confidence,
            source_points=source_points,
            reference_points=reference_points,
            candidate_roi=candidate_roi,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=None if success else "M2 did not produce any valid matches.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("M2 feature matching failed")
        return M2Result(
            executed=True,
            success=False,
            num_matches=0,
            confidence=0.0,
            source_points=None,
            reference_points=None,
            candidate_roi=None,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc),
        )


def _run_m3(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    m2_result: M2Result,
    config: PipelineConfig,
):
    start = time.perf_counter()
    try:
        if m2_result.source_points is None or m2_result.reference_points is None:
            return M3Result(
                executed=True,
                success=False,
                gate_passed=False,
                estimation_success=False,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                decision=None,
                metrics=None,
                duration_ms=int((time.perf_counter() - start) * 1000),
                error="M2 did not produce source/reference points.",
            ), None

        correspondence = {
            "source_points": m2_result.source_points,
            "reference_points": m2_result.reference_points,
            "confidence": float(m2_result.confidence),
            "matches": [{"source_pt": tuple(map(float, row))} for row in m2_result.source_points],
            "num_matches": int(m2_result.num_matches),
            "num_keypoints_source": int(m2_result.num_matches),
            "num_keypoints_reference": int(m2_result.num_matches),
        }

        evidence_gate = EvidenceGate(config=config.evidence_gate_config())
        gate_result = evidence_gate.evaluate(correspondence)
        if not gate_result.passed:
            metrics = compute_validation_metrics(gate_result=gate_result)
            return M3Result(
                executed=True,
                success=False,
                gate_passed=False,
                estimation_success=False,
                num_matches=int(m2_result.num_matches),
                num_inliers=0,
                inlier_ratio=0.0,
                decision=metrics.decision,
                metrics=metrics.to_dict(),
                duration_ms=int((time.perf_counter() - start) * 1000),
                error=gate_result.rejection_reason or "Evidence gate rejected the match set.",
            ), None

        estimator = GeometricEstimator(config=config.geometric_config())
        estimation_result = estimator.estimate(correspondence)
        if not estimation_result.success:
            metrics = compute_validation_metrics(gate_result=gate_result, estimation_result=estimation_result)
            return M3Result(
                executed=True,
                success=False,
                gate_passed=True,
                estimation_success=False,
                num_matches=int(m2_result.num_matches),
                num_inliers=0,
                inlier_ratio=0.0,
                decision=metrics.decision,
                metrics=metrics.to_dict(),
                duration_ms=int((time.perf_counter() - start) * 1000),
                error=estimation_result.reason or "Geometric estimation failed.",
            ), None

        inlier_report = analyse_inliers(
            inlier_mask=estimation_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            image_shape=reference_img.shape[:2],
        )
        reprojection_report = compute_reprojection_errors(
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            transform_matrix=estimation_result.transform_matrix,
            inlier_mask=estimation_result.inlier_mask,
            model_type="homography",
        )
        registration_result = register_image(
            source_image=source_img,
            transform_matrix=estimation_result.transform_matrix,
            reference_shape=reference_img.shape[:2],
            config=RegistrationConfig(model_type=config.registration_model_type),
        )
        metrics = compute_validation_metrics(
            gate_result=gate_result,
            estimation_result=estimation_result,
            inlier_report=inlier_report,
            reprojection_report=reprojection_report,
            registration_result=registration_result,
        )
        m3 = M3Result(
            executed=True,
            success=bool(metrics.passed),
            gate_passed=gate_result.passed,
            estimation_success=estimation_result.success,
            num_matches=int(m2_result.num_matches),
            num_inliers=int(estimation_result.num_inliers),
            inlier_ratio=float(estimation_result.inlier_ratio),
            decision=metrics.decision,
            metrics=metrics.to_dict(),
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=None if metrics.passed else metrics.failure_reason,
        )
        return m3, registration_result
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("M3 validation failed")
        return M3Result(
            executed=True,
            success=False,
            gate_passed=False,
            estimation_success=False,
            num_matches=int(m2_result.num_matches),
            num_inliers=0,
            inlier_ratio=0.0,
            decision=None,
            metrics=None,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc),
        ), None


def _run_m4(
    source_img: np.ndarray,
    reference_img: np.ndarray,
    m2_result: M2Result,
    registration_result: Optional[Any],
    config: PipelineConfig,
) -> M4Result:
    start = time.perf_counter()
    try:
        if m2_result.source_points is None or m2_result.reference_points is None:
            return M4Result(
                executed=True,
                success=False,
                num_refined_points=0,
                refined_source_points=None,
                refined_reference_points=None,
                duration_ms=int((time.perf_counter() - start) * 1000),
                error="M2 produced no points for M4 refinement.",
            )

        source_points = np.asarray(m2_result.source_points, dtype=np.float32)
        if source_points.size == 0:
            return M4Result(
                executed=True,
                success=False,
                num_refined_points=0,
                refined_source_points=None,
                refined_reference_points=None,
                duration_ms=int((time.perf_counter() - start) * 1000),
                error="No source points available for sub-pixel refinement.",
            )

        refined_source = refine_subpixel_points(
            source_img,
            source_points,
            window_size=config.m4_window_size,
            max_iterations=30,
        )
        refined_reference = refine_subpixel_points(
            reference_img,
            np.asarray(m2_result.reference_points, dtype=np.float32),
            window_size=config.m4_window_size,
            max_iterations=30,
        )
        num_refined = int(len(refined_source))
        success = num_refined > 0
        return M4Result(
            executed=True,
            success=success,
            num_refined_points=num_refined,
            refined_source_points=refined_source,
            refined_reference_points=refined_reference,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=None if success else "Sub-pixel refinement produced no valid points.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("M4 refinement failed")
        return M4Result(
            executed=True,
            success=False,
            num_refined_points=0,
            refined_source_points=None,
            refined_reference_points=None,
            duration_ms=int((time.perf_counter() - start) * 1000),
            error=str(exc),
        )


def _save_pipeline_outputs(
    result: PipelineResult,
    source_img: np.ndarray,
    reference_img: np.ndarray,
    config: PipelineConfig,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / config.result_name
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, default=str)
    result.result_path = str(result_path)

    visual_path = output_dir / config.visualization_name
    draw_matches = _build_visualization(source_img, reference_img, result)
    cv2.imwrite(str(visual_path), draw_matches)
    result.visualization_path = str(visual_path)


def _build_visualization(source_img: np.ndarray, reference_img: np.ndarray, result: PipelineResult) -> np.ndarray:
    canvas_h = max(source_img.shape[0], reference_img.shape[0])
    canvas_w = source_img.shape[1] + reference_img.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[: source_img.shape[0], : source_img.shape[1]] = cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR)
    canvas[: reference_img.shape[0], source_img.shape[1] : source_img.shape[1] + reference_img.shape[1]] = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)

    if result.m1 and result.m1.best_candidate is not None:
        bbox = result.m1.best_candidate.get("clipped_bbox") or result.m1.best_candidate.get("bbox")
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(canvas, (source_img.shape[1] + x, y), (source_img.shape[1] + x + w, y + h), (0, 255, 0), 2)

    if result.m2 and result.m2.source_points is not None and result.m2.reference_points is not None:
        src_points = result.m2.source_points
        ref_points = result.m2.reference_points
        for idx in range(min(len(src_points), 80)):
            p1 = (int(round(src_points[idx, 0])), int(round(src_points[idx, 1])))
            p2 = (int(round(source_img.shape[1] + ref_points[idx, 0])), int(round(ref_points[idx, 1])))
            cv2.circle(canvas, p1, 2, (0, 255, 255), -1)
            cv2.circle(canvas, p2, 2, (0, 255, 255), -1)
            cv2.line(canvas, p1, p2, (0, 220, 100), 1, cv2.LINE_AA)

    return canvas


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FLUX M1 → M2 → M3 → M4 end-to-end pipeline on the bundled lunar prototype images.")
    parser.add_argument("--image1", type=str, default=str(DEFAULT_SOURCE_IMAGE), help="Path to source image (default: bundled lunar source image).")
    parser.add_argument("--image2", type=str, default=str(DEFAULT_REFERENCE_IMAGE), help="Path to reference image (default: bundled lunar reference image).")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory where JSON result and visual outputs are saved.")
    args = parser.parse_args(argv)

    config = PipelineConfig(output_dir=Path(args.output_dir))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    LOGGER.info("[INPUT] Loading image 1: %s", args.image1)
    LOGGER.info("[INPUT] Loading image 2: %s", args.image2)
    result = run_pipeline(args.image1, args.image2, config=config)
    _save_pipeline_outputs(result, _load_image(Path(args.image1)), _load_image(Path(args.image2)), config)

    LOGGER.info("[PIPELINE] Final status: %s", result.status)
    LOGGER.info("[PIPELINE] M1 candidates: %s", result.m1.candidate_count if result.m1 else 0)
    LOGGER.info("[PIPELINE] M2 matches: %s", result.m2.num_matches if result.m2 else 0)
    LOGGER.info("[PIPELINE] M3 inliers: %s", result.m3.num_inliers if result.m3 else 0)
    LOGGER.info("[PIPELINE] M4 refined points: %s", result.m4.num_refined_points if result.m4 else 0)
    LOGGER.info("[PIPELINE] Result JSON: %s", result.result_path)
    LOGGER.info("[PIPELINE] Visualization: %s", result.visualization_path)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
