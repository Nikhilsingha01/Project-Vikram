"""Aggregate validation metrics and structured reporting for M3 pipeline.

This module is the final stage of the M3 pipeline.  It collects all
intermediate outputs — evidence gate result, geometric estimation result,
inlier analysis, reprojection report, and registration result — and
aggregates them into a single, richly-annotated ValidationMetrics object
that represents the full quality picture of one source→reference match attempt.

Design Goals
------------
- One canonical quality object per match attempt (ValidationMetrics).
- Single numeric quality score in [0, 1] for downstream ranking/filtering.
- Structured pass/fail flags with human-readable failure reasons at every stage.
- JSON-serialisable diagnostics for logging and experiment tracking.
- Decoupled from any single stage: can be constructed even when early stages
  fail (partial metrics).

Composite Quality Score
-----------------------
The overall quality score is a weighted combination of stage-level scores:

    quality_score = w1 * evidence_score
                  + w2 * inlier_ratio
                  + w3 * (1 − normalised_rmse)
                  + w4 * spatial_uniformity

Default weights (configurable):
    w1 = 0.20  (evidence gate)
    w2 = 0.35  (inlier ratio — most important geometric signal)
    w3 = 0.25  (reprojection quality)
    w4 = 0.20  (spatial uniformity of inliers)

Output
------
ValidationMetrics — the top-level aggregated result containing all stage
outputs and the composite quality_score.

compute_validation_metrics() — convenience factory function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

# Forward references — stages are imported lazily to avoid circular imports.
# When sub-modules are implemented, replace TYPE_CHECKING guard with direct imports.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.validation.evidence_gate import EvidenceGateResult
    from src.validation.geometric_estimation import EstimationResult, TransformModel
    from src.validation.inlier_analysis import InlierReport
    from src.validation.reprojection import ReprojectionReport
    from src.validation.registration import RegistrationResult


# ---------------------------------------------------------------------------
# Composite score weights
# ---------------------------------------------------------------------------


@dataclass
class MetricsWeights:
    """Weights for the composite quality score computation.

    All weights are expected to be non-negative; they are normalised
    internally so they need not sum to 1.0.

    Attributes:
        evidence_weight: Weight for the EvidenceGate evidence_score.
        inlier_ratio_weight: Weight for RANSAC inlier_ratio.
        reprojection_weight: Weight for normalised reprojection quality
            (1 − normalised_rmse).
        spatial_uniformity_weight: Weight for inlier spatial uniformity.
        rmse_saturation_px: RMSE value (pixels) at which the reprojection
            quality signal saturates at 0.  Below 1 px → quality = 1.
    """

    evidence_weight: float = 0.20
    inlier_ratio_weight: float = 0.35
    reprojection_weight: float = 0.25
    spatial_uniformity_weight: float = 0.20
    rmse_saturation_px: float = 20.0


# ---------------------------------------------------------------------------
# Top-level metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetrics:
    """Aggregated quality metrics for one M3 validation run.

    All stage-level result objects are stored alongside the computed
    scalar signals and the composite quality_score.

    Attributes:
        quality_score: Composite quality score ∈ [0, 1].  Higher is better.
        passed: True if all mandatory stages passed (gate + estimation).
        failure_stage: Name of the first stage that failed, or None.
        failure_reason: Human-readable failure code from the failing stage.

        evidence_score: Evidence gate composite signal ∈ [0, 1].
        inlier_count: Number of RANSAC inliers.
        inlier_ratio: Inlier fraction ∈ [0, 1].
        inlier_rmse: Root-mean-square reprojection error over inliers (px).
        spatial_uniformity: Inlier spatial uniformity ∈ [0, 1].
        transform_model: Name of the estimated transform model.

        gate_result: Full EvidenceGateResult from Stage 1.
        estimation_result: Full EstimationResult from Stage 2.
        inlier_report: Full InlierReport from Stage 3.
        reprojection_report: Full ReprojectionReport from Stage 4.
        registration_result: Full RegistrationResult from Stage 5 (may be None
            if the pipeline terminated before registration).

        diagnostics: Flat dict of all key scalar metrics for JSON logging.
    """

    quality_score: float
    passed: bool
    failure_stage: Optional[str]
    failure_reason: Optional[str]

    # Scalar signals (copies for quick access without traversing nested objects)
    evidence_score: float = 0.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    inlier_rmse: float = float("inf")
    spatial_uniformity: float = 0.0
    transform_model: Optional[str] = None

    # Stage outputs (stored by reference)
    gate_result: Optional[Any] = None          # EvidenceGateResult
    estimation_result: Optional[Any] = None    # EstimationResult
    inlier_report: Optional[Any] = None        # InlierReport
    reprojection_report: Optional[Any] = None  # ReprojectionReport
    registration_result: Optional[Any] = None  # RegistrationResult

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scalar metrics to a flat JSON-compatible dictionary.

        Returns:
            Dict containing all scalar metric fields and diagnostics.
            Stage result objects are excluded (not JSON-serialisable).
        """
        # TODO (M3-MET-DICT-01): Build flat dict from scalar fields.
        # TODO (M3-MET-DICT-02): Merge diagnostics sub-dict.
        # TODO (M3-MET-DICT-03): Replace np.float32/int64 values with Python
        #   native types for JSON compatibility.
        raise NotImplementedError("ValidationMetrics.to_dict() not yet implemented.")

    def summary_string(self) -> str:
        """Return a human-readable one-line summary of the metrics.

        Returns:
            String like:
            'PASS  quality=0.82  inliers=47/60 (78%)  RMSE=2.31px  model=HOMOGRAPHY'
            or:
            'FAIL  stage=evidence_gate  reason=insufficient_matches'
        """
        # TODO (M3-MET-STR-01): Branch on self.passed.
        # TODO (M3-MET-STR-02): Format PASS branch with quality, inlier, RMSE, model.
        # TODO (M3-MET-STR-03): Format FAIL branch with stage and reason.
        raise NotImplementedError("ValidationMetrics.summary_string() not yet implemented.")


# ---------------------------------------------------------------------------
# Factory / aggregation function
# ---------------------------------------------------------------------------


def compute_validation_metrics(
    gate_result: Any,                        # EvidenceGateResult
    estimation_result: Optional[Any] = None, # EstimationResult
    inlier_report: Optional[Any] = None,     # InlierReport
    reprojection_report: Optional[Any] = None,# ReprojectionReport
    registration_result: Optional[Any] = None,# RegistrationResult
    weights: Optional[MetricsWeights] = None,
) -> ValidationMetrics:
    """Aggregate all M3 stage outputs into a single ValidationMetrics object.

    This function is the canonical entry point for creating a ValidationMetrics
    instance.  It handles partial results gracefully: if the pipeline terminated
    early (e.g. gate rejected), later-stage arguments will be None and the
    function computes a failure ValidationMetrics accordingly.

    Args:
        gate_result: EvidenceGateResult from EvidenceGate.evaluate().
        estimation_result: EstimationResult from GeometricEstimator.estimate().
            None if gate rejected.
        inlier_report: InlierReport from InlierAnalyzer.analyse().
            None if estimation failed.
        reprojection_report: ReprojectionReport from compute_reprojection_errors().
            None if earlier stage failed.
        registration_result: RegistrationResult from register_image().
            None if pipeline terminated before registration.
        weights: MetricsWeights for composite score.  Defaults to MetricsWeights().

    Returns:
        ValidationMetrics — fully populated (with zeros/None for absent stages).
    """
    # TODO (M3-MET-01): Set weights = MetricsWeights() if None.

    # TODO (M3-MET-02): Extract evidence_score from gate_result.
    #   If gate_result.passed is False, set failure_stage='evidence_gate'
    #   and failure_reason from gate_result.rejection_reason.

    # TODO (M3-MET-03): Extract inlier_ratio and inlier_count from
    #   estimation_result (if not None), else default to 0.

    # TODO (M3-MET-04): Extract inlier_rmse from reprojection_report.inlier_mean_error
    #   (if not None), else default to inf.

    # TODO (M3-MET-05): Extract spatial_uniformity from inlier_report (if not None).

    # TODO (M3-MET-06): Extract transform_model name string from estimation_result.

    # TODO (M3-MET-07): Compute normalised_rmse = min(1, rmse / weights.rmse_saturation_px).
    #   reprojection_quality = max(0, 1 - normalised_rmse).

    # TODO (M3-MET-08): Compute weighted composite score using weights.
    #   Normalise weights by their sum.

    # TODO (M3-MET-09): Determine passed = gate_result.passed AND
    #   (estimation_result is not None) AND estimation_result.success.

    # TODO (M3-MET-10): Build diagnostics dict with all scalar signals.

    # TODO (M3-MET-11): Return ValidationMetrics(...).

    raise NotImplementedError(
        "compute_validation_metrics() is not yet implemented (M3 TODO)."
    )


# ---------------------------------------------------------------------------
# Batch utility (stub)
# ---------------------------------------------------------------------------


def summarise_batch_metrics(
    metrics_list: list,
    sort_by: str = "quality_score",
) -> Dict[str, Any]:
    """Summarise a list of ValidationMetrics from a batch run.

    Computes aggregate statistics (mean, median, pass rate, etc.) across
    multiple match attempts — useful for evaluation runs.

    Args:
        metrics_list: List of ValidationMetrics instances.
        sort_by: Key to sort results by.  One of 'quality_score',
            'inlier_ratio', 'inlier_rmse'.

    Returns:
        Dict with:
            'pass_count'    : int
            'fail_count'    : int
            'pass_rate'     : float
            'mean_quality'  : float
            'median_quality': float
            'mean_inlier_ratio' : float
            'mean_rmse'     : float
            'per_result'    : List[Dict]  — sorted per-result summaries
    """
    # TODO (M3-MET-BATCH-01): Validate metrics_list is non-empty.
    # TODO (M3-MET-BATCH-02): Extract scalar signals from each ValidationMetrics.
    # TODO (M3-MET-BATCH-03): Compute aggregate stats using numpy.
    # TODO (M3-MET-BATCH-04): Sort per_result list by sort_by key.
    # TODO (M3-MET-BATCH-05): Return summary dict.
    raise NotImplementedError(
        "summarise_batch_metrics() is not yet implemented (M3 TODO)."
    )
