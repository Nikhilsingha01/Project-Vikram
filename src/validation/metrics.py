"""Aggregate validation metrics and final Accept/Reject decision for M3 pipeline.

This module is the sixth and final stage of the M3 pipeline.  It collects the
structured outputs from every preceding stage — evidence gate, geometric
estimation, inlier analysis, reprojection error analysis, and image
registration — and aggregates them into a single ``ValidationMetrics`` object
that represents the complete quality picture of one source→reference match
attempt.

Design Goals
------------
- One canonical quality object per match attempt (``ValidationMetrics``).
- Structured ACCEPT / REJECT decision backed by configurable, documented
  thresholds — never hard-coded magic numbers.
- A composite quality score ∈ [0, 1] for downstream ranking / filtering.
- Clear separation between missing/unavailable metrics and zero values.
- JSON-serialisable ``to_dict()`` output suitable for logging and the M4 pipeline.
- Decoupled from any single stage: can be constructed even when early stages
  fail (partial metrics).

Decision Logic
--------------
The final decision (ACCEPT / REJECT) is evidence-based and gated through
every mandatory stage in sequence:

  1. Evidence Gate must have passed.
  2. Geometric estimation must have succeeded.
  3. Inlier ratio  ≥ ``DecisionThresholds.min_inlier_ratio``.
  4. Inlier count  ≥ ``DecisionThresholds.min_inlier_count``.
  5. Inlier RMSE   ≤ ``DecisionThresholds.max_inlier_rmse_px`` (if available).
  6. Registration warp must have succeeded (if registration result present).
  7. Overlap sufficient flag must be True (if registration result present).

Any stage failure produces REJECT with a named ``failure_stage`` and
human-readable ``failure_reason``.

All thresholds live in ``DecisionThresholds`` and are configurable.  No
scientifically unsupported fixed values are used.

Composite Quality Score
-----------------------
    quality_score = w1 * evidence_score
                  + w2 * inlier_ratio
                  + w3 * reprojection_quality   (= max(0, 1 − RMSE/saturation))
                  + w4 * spatial_uniformity

Weights and RMSE saturation live in ``MetricsWeights`` (configurable).
Weights are normalised internally so they need not sum to 1.0.

Output
------
``ValidationMetrics`` — fully populated object with all stage sub-results,
scalar signals, decision flags, composite score, and a flat diagnostics dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Weights for the composite quality score
# ---------------------------------------------------------------------------


@dataclass
class MetricsWeights:
    """Weights for the composite quality score computation.

    All weights must be non-negative.  They are normalised internally so they
    need not sum to 1.0.

    Attributes:
        evidence_weight: Weight for the EvidenceGate ``evidence_score``.
        inlier_ratio_weight: Weight for RANSAC ``inlier_ratio``.
        reprojection_weight: Weight for reprojection quality
            (= max(0, 1 − RMSE / rmse_saturation_px)).
        spatial_uniformity_weight: Weight for inlier spatial uniformity.
        rmse_saturation_px: RMSE (pixels) at which reprojection quality
            saturates to 0.  Values below this give proportional quality.
    """

    evidence_weight: float = 0.20
    inlier_ratio_weight: float = 0.35
    reprojection_weight: float = 0.25
    spatial_uniformity_weight: float = 0.20
    rmse_saturation_px: float = 20.0


# ---------------------------------------------------------------------------
# Configurable ACCEPT/REJECT decision thresholds
# ---------------------------------------------------------------------------


@dataclass
class DecisionThresholds:
    """Configurable thresholds used to produce the ACCEPT / REJECT decision.

    Every threshold is documented with its rationale so the decision is fully
    auditable.  Change them per-deployment; never hard-code into business logic.

    Attributes:
        min_inlier_ratio: Minimum RANSAC inlier fraction required to ACCEPT.
            Below this the geometric model is considered unreliable.
            Default 0.25 — requires at least 1-in-4 correspondences to agree.
        min_inlier_count: Absolute minimum number of RANSAC inliers.
            Affine estimation needs ≥ 3; homography needs ≥ 4.  Default 6 gives
            a comfortable margin above the degenerate minimum.
        max_inlier_rmse_px: Maximum allowed inlier reprojection RMSE in pixels.
            None means the RMSE check is skipped (useful when reprojection
            report is unavailable).  Default None (not enforced by default).
        require_warp_success: If True and a registration result is present,
            REJECT when warp_succeeded=False.  Default True.
        require_overlap_sufficient: If True and a registration result is
            present, REJECT when overlap_sufficient=False.  Default False
            (overlap check is advisory by default).
    """

    min_inlier_ratio: float = 0.25
    min_inlier_count: int = 6
    max_inlier_rmse_px: Optional[float] = None
    require_warp_success: bool = True
    require_overlap_sufficient: bool = False


# ---------------------------------------------------------------------------
# Decision enumeration
# ---------------------------------------------------------------------------

ACCEPT = "ACCEPT"
REJECT = "REJECT"


# ---------------------------------------------------------------------------
# Top-level metrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationMetrics:
    """Aggregated quality metrics for one M3 validation run.

    Scalar signals are extracted from stage results and stored here for
    fast access without traversing nested objects.

    Attributes:
        decision: 'ACCEPT' or 'REJECT'.
        quality_score: Composite quality score ∈ [0, 1].  Higher is better.
        passed: True if decision == ACCEPT.
        failure_stage: Name of the first stage that caused REJECT, or None.
        failure_reason: Human-readable failure code / description, or None.

        evidence_score: EvidenceGate composite signal ∈ [0, 1].
            None if gate result was not provided.
        gate_passed: Whether the evidence gate passed.

        num_correspondences: Total number of M2 correspondence pairs.
        num_inliers: Number of RANSAC inliers.
        num_outliers: Number of RANSAC outliers.
        inlier_ratio: Inlier fraction ∈ [0, 1].  None if estimation absent.
        spatial_uniformity: Inlier spatial uniformity ∈ [0, 1].
            None if inlier report absent.

        reprojection_mean_px: Mean inlier reprojection error (px).  None if absent.
        reprojection_median_px: Median inlier reprojection error (px).  None if absent.
        reprojection_rmse_px: Inlier RMSE (px).  None if absent.
        reprojection_max_px: Maximum inlier reprojection error (px).  None if absent.

        transform_model: String name of the estimated transform model.
        transform_valid: True if estimation succeeded and matrix is finite.

        warp_succeeded: True if registration warping ran without error.
            None if registration result absent.
        overlap_sufficient: True if valid-pixel overlap meets minimum ratio.
            None if registration result absent.

        gate_result: Full EvidenceGateResult from Stage 1.
        estimation_result: Full EstimationResult from Stage 2.
        inlier_report: Full InlierReport from Stage 3.
        reprojection_report: Full ReprojectionReport from Stage 4.
        registration_result: Full RegistrationResult from Stage 5 (may be None).

        diagnostics: Flat dict of all key scalar metrics for JSON logging.
    """

    # Top-level decision
    decision: str
    quality_score: float
    passed: bool
    failure_stage: Optional[str]
    failure_reason: Optional[str]

    # Evidence gate
    evidence_score: Optional[float] = None
    gate_passed: bool = False

    # Correspondence / inlier signals
    num_correspondences: int = 0
    num_inliers: int = 0
    num_outliers: int = 0
    inlier_ratio: Optional[float] = None
    spatial_uniformity: Optional[float] = None

    # Reprojection error signals
    reprojection_mean_px: Optional[float] = None
    reprojection_median_px: Optional[float] = None
    reprojection_rmse_px: Optional[float] = None
    reprojection_max_px: Optional[float] = None

    # Transform
    transform_model: Optional[str] = None
    transform_valid: bool = False

    # Registration
    warp_succeeded: Optional[bool] = None
    overlap_sufficient: Optional[bool] = None

    # Stage results (full objects)
    gate_result: Optional[Any] = None
    estimation_result: Optional[Any] = None
    inlier_report: Optional[Any] = None
    reprojection_report: Optional[Any] = None
    registration_result: Optional[Any] = None

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise scalar metrics to a flat JSON-compatible dictionary.

        Stage result objects (gate_result, estimation_result, etc.) are
        excluded because they are not JSON-serialisable.  All numpy scalar
        types are converted to Python-native types.

        Returns:
            Dict[str, Any] suitable for ``json.dumps()``.
        """
        def _native(v: Any) -> Any:
            """Convert numpy scalars to Python-native types."""
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, float) and math.isnan(v):
                return None   # NaN → null in JSON
            if isinstance(v, float) and math.isinf(v):
                return None   # Inf → null in JSON
            return v

        base = {
            "decision": self.decision,
            "quality_score": _native(self.quality_score),
            "passed": bool(self.passed),
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            # Evidence
            "evidence_score": _native(self.evidence_score),
            "gate_passed": bool(self.gate_passed),
            # Correspondences
            "num_correspondences": int(self.num_correspondences),
            "num_inliers": int(self.num_inliers),
            "num_outliers": int(self.num_outliers),
            "inlier_ratio": _native(self.inlier_ratio),
            "spatial_uniformity": _native(self.spatial_uniformity),
            # Reprojection
            "reprojection_mean_px": _native(self.reprojection_mean_px),
            "reprojection_median_px": _native(self.reprojection_median_px),
            "reprojection_rmse_px": _native(self.reprojection_rmse_px),
            "reprojection_max_px": _native(self.reprojection_max_px),
            # Transform
            "transform_model": self.transform_model,
            "transform_valid": bool(self.transform_valid),
            # Registration
            "warp_succeeded": _native(self.warp_succeeded),
            "overlap_sufficient": _native(self.overlap_sufficient),
        }
        # Merge flat diagnostics (scalar values only)
        for k, v in self.diagnostics.items():
            if k not in base:
                base[k] = _native(v)
        return base

    def summary_string(self) -> str:
        """Return a human-readable one-line summary of the metrics.

        Examples:
            'ACCEPT  quality=0.82  inliers=47/60 (78%)  RMSE=2.31px  model=AFFINE'
            'REJECT  stage=geometric_estimation  reason=ransac_failed'
        """
        if self.passed:
            inlier_total = self.num_correspondences or (
                self.num_inliers + (self.num_outliers or 0)
            )
            pct = (
                f"({self.inlier_ratio * 100:.0f}%)"
                if self.inlier_ratio is not None
                else ""
            )
            rmse_str = (
                f"RMSE={self.reprojection_rmse_px:.2f}px"
                if self.reprojection_rmse_px is not None
                else "RMSE=N/A"
            )
            model_str = self.transform_model or "N/A"
            return (
                f"ACCEPT  quality={self.quality_score:.2f}"
                f"  inliers={self.num_inliers}/{inlier_total} {pct}"
                f"  {rmse_str}"
                f"  model={model_str}"
            )
        else:
            return (
                f"REJECT  stage={self.failure_stage or 'unknown'}"
                f"  reason={self.failure_reason or 'unknown'}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert v to Python float, returning default on None / NaN / Inf."""
    if v is None:
        return default
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(fv) or math.isinf(fv):
        return default
    return fv


def _compute_quality_score(
    evidence_score: float,
    inlier_ratio: float,
    reprojection_rmse: Optional[float],
    spatial_uniformity: float,
    weights: MetricsWeights,
) -> float:
    """Compute the composite quality score ∈ [0, 1].

    Normalises weights by their sum so the caller need not ensure they sum
    to 1.0.  Clamps the final result to [0, 1].

    Args:
        evidence_score: Gate composite score ∈ [0, 1].
        inlier_ratio: RANSAC inlier fraction ∈ [0, 1].
        reprojection_rmse: Inlier RMSE in pixels.  None treated as worst-case.
        spatial_uniformity: Inlier spatial uniformity ∈ [0, 1].
        weights: MetricsWeights instance.

    Returns:
        Composite quality score ∈ [0, 1].
    """
    w1 = max(0.0, weights.evidence_weight)
    w2 = max(0.0, weights.inlier_ratio_weight)
    w3 = max(0.0, weights.reprojection_weight)
    w4 = max(0.0, weights.spatial_uniformity_weight)
    total_w = w1 + w2 + w3 + w4
    if total_w <= 0.0:
        return 0.0

    # Reprojection quality: 1 at RMSE=0, 0 at RMSE=saturation
    if reprojection_rmse is None:
        reproj_quality = 0.0
    else:
        sat = max(weights.rmse_saturation_px, 1e-9)
        reproj_quality = max(0.0, 1.0 - reprojection_rmse / sat)

    raw = (
        w1 * float(evidence_score)
        + w2 * float(inlier_ratio)
        + w3 * reproj_quality
        + w4 * float(spatial_uniformity)
    )
    return float(max(0.0, min(1.0, raw / total_w)))


def _decide(
    gate_result: Any,
    estimation_result: Optional[Any],
    inlier_report: Optional[Any],
    reprojection_report: Optional[Any],
    registration_result: Optional[Any],
    thresholds: DecisionThresholds,
) -> tuple:
    """Return (decision, failure_stage, failure_reason).

    Evaluates each mandatory criterion in order.  The first failure terminates
    evaluation and returns REJECT with the failing stage name and reason.
    If all criteria pass, returns (ACCEPT, None, None).
    """
    # ---- 1. Evidence gate ---------------------------------------------------
    if gate_result is None or not gate_result.passed:
        reason = (
            getattr(gate_result, "rejection_reason", None)
            or "gate_result_none"
        )
        return REJECT, "evidence_gate", reason

    # ---- 2. Geometric estimation -------------------------------------------
    if estimation_result is None or not estimation_result.success:
        reason = (
            getattr(estimation_result, "reason", None)
            or getattr(estimation_result, "failure_reason", None)
            or "estimation_result_none"
        ) if estimation_result is not None else "estimation_result_none"
        return REJECT, "geometric_estimation", reason

    # ---- 3. Inlier ratio threshold -----------------------------------------
    inlier_ratio = getattr(estimation_result, "inlier_ratio", None)
    if inlier_ratio is None:
        # Compute from counts if ratio not directly present
        n_in = getattr(estimation_result, "num_inliers", 0) or 0
        n_tot = getattr(estimation_result, "total_matches", 0) or 0
        inlier_ratio = n_in / max(n_tot, 1)

    if float(inlier_ratio) < thresholds.min_inlier_ratio:
        return (
            REJECT,
            "inlier_ratio",
            f"inlier_ratio={inlier_ratio:.3f} < min={thresholds.min_inlier_ratio}",
        )

    # ---- 4. Minimum inlier count -------------------------------------------
    n_inliers = getattr(estimation_result, "num_inliers", 0) or 0
    if n_inliers < thresholds.min_inlier_count:
        return (
            REJECT,
            "inlier_count",
            f"num_inliers={n_inliers} < min={thresholds.min_inlier_count}",
        )

    # ---- 5. Reprojection RMSE threshold (optional) -------------------------
    if thresholds.max_inlier_rmse_px is not None and reprojection_report is not None:
        rmse = _safe_float(getattr(reprojection_report, "inlier_rmse", None))
        if rmse is not None and rmse > thresholds.max_inlier_rmse_px:
            return (
                REJECT,
                "reprojection_error",
                (
                    f"inlier_rmse={rmse:.2f}px"
                    f" > max={thresholds.max_inlier_rmse_px}px"
                ),
            )

    # ---- 6. Registration warp success (if present and required) -----------
    if registration_result is not None and thresholds.require_warp_success:
        qf = getattr(registration_result, "quality_flags", {})
        if not qf.get("warp_succeeded", True):
            return REJECT, "registration", "warp_succeeded=False"

    # ---- 7. Overlap sufficient (if present and required) -------------------
    if registration_result is not None and thresholds.require_overlap_sufficient:
        qf = getattr(registration_result, "quality_flags", {})
        if not qf.get("overlap_sufficient", True):
            return REJECT, "registration", "overlap_sufficient=False"

    return ACCEPT, None, None


# ---------------------------------------------------------------------------
# Factory / aggregation function
# ---------------------------------------------------------------------------


def compute_validation_metrics(
    gate_result: Any,
    estimation_result: Optional[Any] = None,
    inlier_report: Optional[Any] = None,
    reprojection_report: Optional[Any] = None,
    registration_result: Optional[Any] = None,
    weights: Optional[MetricsWeights] = None,
    thresholds: Optional[DecisionThresholds] = None,
) -> ValidationMetrics:
    """Aggregate all M3 stage outputs into a single ValidationMetrics object.

    Handles partial results gracefully: if the pipeline terminated early
    (e.g. gate rejected), later-stage arguments will be None and the function
    computes a REJECT ValidationMetrics accordingly.

    Args:
        gate_result: EvidenceGateResult from EvidenceGate.evaluate().
            Required (must not be None).
        estimation_result: EstimationResult from GeometricEstimator.estimate().
            None if gate rejected.
        inlier_report: InlierReport from InlierAnalyzer / analyse_inliers().
            None if estimation failed.
        reprojection_report: ReprojectionReport from compute_reprojection_errors().
            None if earlier stage failed.
        registration_result: RegistrationResult from register_image().
            None if pipeline terminated before registration.
        weights: MetricsWeights for composite score.  Defaults to
            MetricsWeights() (all defaults).
        thresholds: DecisionThresholds for ACCEPT/REJECT logic.  Defaults to
            DecisionThresholds() (all defaults).

    Returns:
        ValidationMetrics — fully populated.

    Raises:
        TypeError: If gate_result is None (it is always required).
    """
    if gate_result is None:
        raise TypeError("gate_result must not be None.")

    if weights is None:
        weights = MetricsWeights()
    if thresholds is None:
        thresholds = DecisionThresholds()

    # ---- 1. Extract evidence gate signals -----------------------------------
    evidence_score = _safe_float(
        getattr(gate_result, "evidence_score", 0.0), default=0.0
    )
    gate_passed = bool(getattr(gate_result, "passed", False))

    # ---- 2. Extract estimation signals --------------------------------------
    num_inliers: int = 0
    num_outliers: int = 0
    num_correspondences: int = 0
    inlier_ratio: Optional[float] = None
    transform_model: Optional[str] = None
    transform_valid: bool = False

    if estimation_result is not None:
        num_inliers = int(
            getattr(estimation_result, "num_inliers", 0)
            or getattr(estimation_result, "inlier_count", 0)
            or 0
        )
        num_outliers = int(getattr(estimation_result, "num_outliers", 0) or 0)
        total_matches = int(getattr(estimation_result, "total_matches", 0) or 0)
        num_correspondences = total_matches or (num_inliers + num_outliers)

        # inlier_ratio — prefer the field if present, otherwise compute
        _ratio = getattr(estimation_result, "inlier_ratio", None)
        if _ratio is not None:
            inlier_ratio = _safe_float(_ratio)
        elif num_correspondences > 0:
            inlier_ratio = num_inliers / num_correspondences
        else:
            inlier_ratio = 0.0

        # Transform model name
        model_obj = getattr(estimation_result, "model", None)
        if model_obj is not None:
            transform_model = str(model_obj.name if hasattr(model_obj, "name") else model_obj)
        elif getattr(estimation_result, "transform_type", None):
            transform_model = str(estimation_result.transform_type)

        # Transform validity: success AND matrix is finite
        if getattr(estimation_result, "success", False):
            M = getattr(estimation_result, "transform_matrix", None)
            if M is not None and np.isfinite(np.asarray(M)).all():
                transform_valid = True

    # ---- 3. Extract inlier analysis signals --------------------------------
    spatial_uniformity: Optional[float] = None

    if inlier_report is not None:
        _su = (
            getattr(inlier_report, "spatial_coverage", None)
            or getattr(inlier_report, "spatial_uniformity", None)
        )
        spatial_uniformity = _safe_float(_su)
        # Overwrite inlier counts from InlierReport if available (more precise)
        _ic = getattr(inlier_report, "inlier_count", None)
        _oc = getattr(inlier_report, "outlier_count", None)
        if _ic is not None:
            num_inliers = int(_ic)
        if _oc is not None:
            num_outliers = int(_oc)

    # ---- 4. Extract reprojection signals -----------------------------------
    reprojection_mean_px: Optional[float] = None
    reprojection_median_px: Optional[float] = None
    reprojection_rmse_px: Optional[float] = None
    reprojection_max_px: Optional[float] = None

    if reprojection_report is not None:
        reprojection_mean_px = _safe_float(
            getattr(reprojection_report, "inlier_mean_error", None)
        )
        reprojection_median_px = _safe_float(
            getattr(reprojection_report, "inlier_median_error", None)
        )
        reprojection_rmse_px = _safe_float(
            getattr(reprojection_report, "inlier_rmse", None)
        )
        reprojection_max_px = _safe_float(
            getattr(reprojection_report, "inlier_max_error", None)
        )

    # ---- 5. Extract registration signals -----------------------------------
    warp_succeeded: Optional[bool] = None
    overlap_sufficient: Optional[bool] = None

    if registration_result is not None:
        qf = getattr(registration_result, "quality_flags", {})
        _ws = qf.get("warp_succeeded")
        _os = qf.get("overlap_sufficient")
        warp_succeeded = bool(_ws) if _ws is not None else None
        overlap_sufficient = bool(_os) if _os is not None else None

    # ---- 6. ACCEPT / REJECT decision ---------------------------------------
    decision, failure_stage, failure_reason = _decide(
        gate_result, estimation_result,
        inlier_report, reprojection_report,
        registration_result, thresholds,
    )
    passed = decision == ACCEPT

    # ---- 7. Composite quality score ----------------------------------------
    quality_score = _compute_quality_score(
        evidence_score=evidence_score or 0.0,
        inlier_ratio=inlier_ratio or 0.0,
        reprojection_rmse=reprojection_rmse_px,
        spatial_uniformity=spatial_uniformity or 0.0,
        weights=weights,
    )

    # ---- 8. Build diagnostics dict -----------------------------------------
    diagnostics: Dict[str, Any] = {
        "decision": decision,
        "quality_score": quality_score,
        "evidence_score": evidence_score,
        "gate_passed": gate_passed,
        "num_correspondences": num_correspondences,
        "num_inliers": num_inliers,
        "num_outliers": num_outliers,
        "inlier_ratio": inlier_ratio,
        "spatial_uniformity": spatial_uniformity,
        "reprojection_mean_px": reprojection_mean_px,
        "reprojection_median_px": reprojection_median_px,
        "reprojection_rmse_px": reprojection_rmse_px,
        "reprojection_max_px": reprojection_max_px,
        "transform_model": transform_model,
        "transform_valid": transform_valid,
        "warp_succeeded": warp_succeeded,
        "overlap_sufficient": overlap_sufficient,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
    }

    return ValidationMetrics(
        decision=decision,
        quality_score=quality_score,
        passed=passed,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        evidence_score=evidence_score,
        gate_passed=gate_passed,
        num_correspondences=num_correspondences,
        num_inliers=num_inliers,
        num_outliers=num_outliers,
        inlier_ratio=inlier_ratio,
        spatial_uniformity=spatial_uniformity,
        reprojection_mean_px=reprojection_mean_px,
        reprojection_median_px=reprojection_median_px,
        reprojection_rmse_px=reprojection_rmse_px,
        reprojection_max_px=reprojection_max_px,
        transform_model=transform_model,
        transform_valid=transform_valid,
        warp_succeeded=warp_succeeded,
        overlap_sufficient=overlap_sufficient,
        gate_result=gate_result,
        estimation_result=estimation_result,
        inlier_report=inlier_report,
        reprojection_report=reprojection_report,
        registration_result=registration_result,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------------


def summarise_batch_metrics(
    metrics_list: List[ValidationMetrics],
    sort_by: str = "quality_score",
) -> Dict[str, Any]:
    """Summarise a list of ValidationMetrics from a batch run.

    Computes aggregate statistics across multiple match attempts — useful
    for evaluation and experiment tracking.

    Args:
        metrics_list: List of ValidationMetrics instances.  Must be non-empty.
        sort_by: Key to sort per-result summaries by.  One of
            'quality_score', 'inlier_ratio', 'inlier_rmse'.

    Returns:
        Dict with keys:
            'pass_count'      : int
            'fail_count'      : int
            'pass_rate'       : float ∈ [0, 1]
            'mean_quality'    : float
            'median_quality'  : float
            'mean_inlier_ratio': float
            'mean_rmse'       : float (NaN-excluded)
            'per_result'      : List[Dict] — per-result summary dicts sorted
                                 by sort_by (descending).

    Raises:
        ValueError: If metrics_list is empty.
    """
    if not metrics_list:
        raise ValueError("metrics_list must not be empty.")

    scores = np.array([m.quality_score for m in metrics_list], dtype=np.float64)
    pass_flags = np.array([m.passed for m in metrics_list], dtype=bool)
    inlier_ratios = np.array(
        [m.inlier_ratio if m.inlier_ratio is not None else 0.0 for m in metrics_list],
        dtype=np.float64,
    )
    rmses = np.array(
        [
            m.reprojection_rmse_px
            if m.reprojection_rmse_px is not None
            else np.nan
            for m in metrics_list
        ],
        dtype=np.float64,
    )
    valid_rmses = rmses[np.isfinite(rmses)]

    # Per-result summaries
    sort_keys = {
        "quality_score": lambda m: m.quality_score,
        "inlier_ratio": lambda m: m.inlier_ratio or 0.0,
        "inlier_rmse": lambda m: -(m.reprojection_rmse_px or float("inf")),
    }
    key_fn = sort_keys.get(sort_by, sort_keys["quality_score"])
    sorted_metrics = sorted(metrics_list, key=key_fn, reverse=True)

    per_result = [m.to_dict() for m in sorted_metrics]

    return {
        "pass_count": int(pass_flags.sum()),
        "fail_count": int((~pass_flags).sum()),
        "pass_rate": float(pass_flags.mean()),
        "mean_quality": float(scores.mean()),
        "median_quality": float(np.median(scores)),
        "mean_inlier_ratio": float(inlier_ratios.mean()),
        "mean_rmse": float(np.mean(valid_rmses)) if valid_rmses.size > 0 else float("nan"),
        "per_result": per_result,
    }
