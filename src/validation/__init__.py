"""M3 Evidence-Gated Validation package for Project Vikram / FLUX.

Provides a multi-stage validation pipeline that sits between M2 correspondence
output and the final registration result.  The pipeline is structured as:

    M2 Correspondence Output
        │
        ▼
    EvidenceGate          (evidence_gate.py)
        │  Applies confidence thresholds, minimum-match counts, and
        │  multi-signal evidence scoring to decide whether a correspondence
        │  set is worthy of further geometric processing.
        ▼
    GeometricEstimation   (geometric_estimation.py)
        │  Robust homography / affine / similarity estimation via RANSAC
        │  (or MAGSAC++) using only the gate-approved point pairs.
        ▼
    InlierAnalysis        (inlier_analysis.py)
        │  Classifies each correspondence as inlier / outlier based on
        │  the estimated model; computes inlier ratio and spatial statistics.
        ▼
    Reprojection          (reprojection.py)
        │  Projects source keypoints through the estimated transform and
        │  computes per-match reprojection error in pixel space.
        ▼
    Registration          (registration.py)
        │  Applies the validated transform to produce a registered image
        │  and associated geographic metadata.
        ▼
    Metrics               (metrics.py)
            Aggregates all stage outputs into a structured ValidationResult
            with quantitative quality scores for downstream reporting.

M2 Data Contract (READ-ONLY — do NOT modify M2 modules):
    The correspondence dict produced by src.correspondence.feature_matching
    has the following guaranteed keys:

        source_points       : np.ndarray, shape (N, 2), dtype float32 — [x, y]
        reference_points    : np.ndarray, shape (N, 2), dtype float32 — [x, y]
        confidence          : float  ∈ [0.0, 1.0]
        matches             : List[Dict]  — per-match detail records
        num_matches         : int
        num_keypoints_source    : int
        num_keypoints_reference : int

    The dict may optionally contain:
        candidate_roi       : Dict — present when match_with_candidate_roi() was used
"""

# ---------------------------------------------------------------------------
# Public API — populated incrementally as sub-modules are implemented.
# ---------------------------------------------------------------------------

# TODO (M3): uncomment and extend as each sub-module is implemented.
# from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig
# from src.validation.geometric_estimation import (
#     GeometricEstimator,
#     EstimationResult,
#     TransformModel,
# )
# from src.validation.inlier_analysis import InlierAnalyzer, InlierReport
# from src.validation.reprojection import compute_reprojection_errors, ReprojectionReport
# from src.validation.registration import register_image, RegistrationResult
# from src.validation.metrics import ValidationMetrics, compute_validation_metrics

__all__: list = [
    # Populated once sub-modules are implemented.
]
