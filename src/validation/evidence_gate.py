"""Evidence-Gated entry filter for M3 Validation pipeline.

The EvidenceGate is the first stage of the M3 pipeline.  It inspects the raw
correspondence output from M2 and decides — based on a configurable set of
evidence signals — whether the match set is sufficient to proceed with
geometric estimation.

Rationale
---------
Downstream RANSAC-based estimation is meaningless if the incoming correspondence
set is degenerate (too few points, near-zero confidence, uniform spatial
distribution, etc.).  The gate provides a single authoritative go / no-go
decision with a structured rejection reason so the pipeline can skip expensive
computation and report a clean failure mode.

Evidence Signals (planned)
--------------------------
1. Minimum match count            — hard lower bound on N.
2. M2 confidence threshold        — scalar gate on the upstream confidence score.
3. Spatial spread (coverage)      — measures whether points cover the image area
                                    (low spread ⇒ degenerate / textureless patch).
4. Mutual consistency score       — optional cross-check ratio.
5. Descriptor quality proxy       — mean ratio-test distance if available.

M2 Input Contract
-----------------
The gate accepts the correspondence dict produced by:
    src.correspondence.feature_matching.match_sift_features()
    src.correspondence.feature_matching.match_with_candidate_roi()

Required keys:
    source_points       : np.ndarray (N, 2) float32
    reference_points    : np.ndarray (N, 2) float32
    confidence          : float ∈ [0, 1]
    num_matches         : int
    num_keypoints_source    : int
    num_keypoints_reference : int

Output
------
EvidenceGateResult — a dataclass carrying:
    passed              : bool
    rejection_reason    : Optional[str]
    evidence_score      : float ∈ [0, 1]   — composite evidence quality
    diagnostics         : Dict             — per-signal breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvidenceGateConfig:
    """Hyperparameters controlling the evidence gate decision thresholds.

    Attributes:
        min_matches: Absolute minimum number of correspondence pairs required.
            Below this the gate always rejects regardless of other signals.
        min_confidence: Minimum M2 correspondence confidence score in [0, 1].
        min_spatial_spread: Minimum normalised spatial spread of source points
            (0 = all co-located, 1 = uniformly distributed across image).
        min_evidence_score: Composite evidence score threshold for passing.
            The composite is a weighted combination of all signal scores.
        confidence_weight: Weight of M2 confidence in composite score.
        spread_weight: Weight of spatial spread in composite score.
        match_count_weight: Weight of the normalised match count in score.
        match_count_saturation: Number of matches at which the count-based
            signal saturates (reaches 1.0).
    """

    min_matches: int = 10
    min_confidence: float = 0.1
    min_spatial_spread: float = 0.05
    min_evidence_score: float = 0.15
    confidence_weight: float = 0.4
    spread_weight: float = 0.3
    match_count_weight: float = 0.3
    match_count_saturation: int = 50


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvidenceGateResult:
    """Structured output from the EvidenceGate.

    Attributes:
        passed: True if the correspondence set cleared all thresholds.
        rejection_reason: Human-readable explanation when passed=False,
            None when passed=True.
        evidence_score: Composite evidence quality score ∈ [0, 1].
        diagnostics: Dictionary with per-signal sub-scores and metadata.
    """

    passed: bool
    rejection_reason: Optional[str]
    evidence_score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main gate class
# ---------------------------------------------------------------------------


class EvidenceGate:
    """Multi-signal evidence gate for M3 validation.

    Usage::

        gate = EvidenceGate(config=EvidenceGateConfig(min_matches=8))
        result = gate.evaluate(correspondence_dict)
        if not result.passed:
            logger.warning("Gate rejected: %s", result.rejection_reason)
            return  # skip expensive estimation

    Args:
        config: EvidenceGateConfig instance.  Defaults to EvidenceGateConfig().
    """

    def __init__(self, config: Optional[EvidenceGateConfig] = None) -> None:
        self.config = config or EvidenceGateConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, correspondence: Dict[str, Any]) -> EvidenceGateResult:
        """Evaluate whether a correspondence dict passes the evidence gate.

        Args:
            correspondence: Correspondence dict from M2 (match_sift_features or
                match_with_candidate_roi).  Must contain source_points,
                reference_points, confidence, and num_matches.

        Returns:
            EvidenceGateResult with pass/fail decision and diagnostics.

        Raises:
            ValueError: If required keys are missing from correspondence.
        """
        # ---- 1. Validate required M2 keys (M3-EG-01) -----------------------
        self._validate_correspondence_keys(correspondence)

        num_matches: int = int(correspondence.get("num_matches", 0))
        confidence: float = float(correspondence.get("confidence", 0.0))
        source_points: np.ndarray = np.asarray(
            correspondence["source_points"], dtype=np.float64
        )

        # ---- 2. Hard: minimum match count (M3-EG-02) -----------------------
        if num_matches < self.config.min_matches:
            return EvidenceGateResult(
                passed=False,
                rejection_reason="insufficient_matches",
                evidence_score=0.0,
                diagnostics={
                    "num_matches": num_matches,
                    "min_matches": self.config.min_matches,
                },
            )

        # ---- 3. Hard: M2 confidence threshold (M3-EG-03) -------------------
        if confidence < self.config.min_confidence:
            return EvidenceGateResult(
                passed=False,
                rejection_reason="low_m2_confidence",
                evidence_score=0.0,
                diagnostics={
                    "confidence": confidence,
                    "min_confidence": self.config.min_confidence,
                },
            )

        # ---- 4. Normalised match-count signal (M3-EG-04) -------------------
        count_score: float = min(
            1.0, num_matches / max(self.config.match_count_saturation, 1)
        )

        # ---- 5. Spatial spread signal (M3-EG-05) ---------------------------
        spread_score: float = self._compute_spatial_spread(source_points)
        if spread_score < self.config.min_spatial_spread:
            return EvidenceGateResult(
                passed=False,
                rejection_reason="insufficient_spatial_spread",
                evidence_score=0.0,
                diagnostics={
                    "spread_score": spread_score,
                    "min_spatial_spread": self.config.min_spatial_spread,
                },
            )

        # ---- 6. Composite evidence score (M3-EG-06) ------------------------
        w_conf  = max(0.0, self.config.confidence_weight)
        w_spread = max(0.0, self.config.spread_weight)
        w_count = max(0.0, self.config.match_count_weight)
        total_w = w_conf + w_spread + w_count
        if total_w <= 0.0:
            total_w = 1.0  # guard against all-zero config

        evidence_score: float = float(
            (w_conf * confidence + w_spread * spread_score + w_count * count_score)
            / total_w
        )
        evidence_score = max(0.0, min(1.0, evidence_score))

        # ---- 7. Final composite threshold gate (M3-EG-07) ------------------
        if evidence_score < self.config.min_evidence_score:
            return EvidenceGateResult(
                passed=False,
                rejection_reason="low_composite_evidence",
                evidence_score=evidence_score,
                diagnostics={
                    "evidence_score": evidence_score,
                    "min_evidence_score": self.config.min_evidence_score,
                    "count_score": count_score,
                    "confidence": confidence,
                    "spread_score": spread_score,
                },
            )

        # ---- 8. Pass (M3-EG-08) -------------------------------------------
        return EvidenceGateResult(
            passed=True,
            rejection_reason=None,
            evidence_score=evidence_score,
            diagnostics={
                "num_matches": num_matches,
                "confidence": confidence,
                "count_score": count_score,
                "spread_score": spread_score,
                "evidence_score": evidence_score,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers (stubs)
    # ------------------------------------------------------------------

    def _compute_spatial_spread(
        self, points: np.ndarray, image_shape: Optional[tuple] = None
    ) -> float:
        """Compute a normalised spatial spread score for a set of 2-D points.

        Args:
            points: (N, 2) float32 array of [x, y] coordinates.
            image_shape: Optional (H, W) tuple for normalisation.  If None,
                uses the bounding box of the points themselves.

        Returns:
            Spread score ∈ [0, 1].  0 = all points coincident, 1 = maximum
            spread covering the normalisation area.
        """
        # M3-EG-SPREAD-01: Handle degenerate cases
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            return 0.0

        # M3-EG-SPREAD-02: Compute bounding box dimensions
        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
        x_range = x_max - x_min
        y_range = y_max - y_min

        # M3-EG-SPREAD-03: Normalise spread by image diagonal (or bbox diagonal)
        if image_shape is not None:
            h, w = float(image_shape[0]), float(image_shape[1])
            diagonal = float(np.sqrt(h ** 2 + w ** 2))
        else:
            diagonal = float(np.sqrt(x_range ** 2 + y_range ** 2))

        if diagonal < 1e-9:
            return 0.0

        # M3-EG-SPREAD-04: Use the mean std of x and y coordinates, normalised
        # by half the diagonal (std of a uniform distribution over [0, D] ≈ D/sqrt(12))
        std_x = float(np.std(points[:, 0]))
        std_y = float(np.std(points[:, 1]))
        mean_std = (std_x + std_y) / 2.0
        # Normalise: max mean_std for uniform spread ≈ diagonal / (2 * sqrt(12))
        normaliser = diagonal / (2.0 * float(np.sqrt(12.0)))
        spread = float(np.clip(mean_std / max(normaliser, 1e-9), 0.0, 1.0))
        return spread

    def _validate_correspondence_keys(self, correspondence: Dict[str, Any]) -> None:
        """Raise ValueError if required M2 keys are absent.

        Args:
            correspondence: M2 correspondence dict to validate.

        Raises:
            ValueError: With list of missing keys.
        """
        # M3-EG-KEYS-01: Required M2 output keys
        REQUIRED_KEYS = (
            "source_points",
            "reference_points",
            "confidence",
            "num_matches",
        )
        # M3-EG-KEYS-02: Find missing keys
        missing = [k for k in REQUIRED_KEYS if k not in correspondence]
        # M3-EG-KEYS-03: Raise with descriptive message if any are absent
        if missing:
            raise ValueError(
                f"correspondence dict is missing required M2 keys: {missing}. "
                f"Got keys: {sorted(correspondence.keys())}"
            )
