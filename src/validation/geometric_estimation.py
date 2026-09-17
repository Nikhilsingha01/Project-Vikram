"""Robust geometric model estimation for M3 Validation pipeline.

This module is the second stage of the M3 pipeline, executed after the
EvidenceGate has approved the correspondence set.  It estimates a geometric
transform (homography or affine) from the correspondence point pairs using
RANSAC-based robust estimation via OpenCV.

Supported Transform Models
--------------------------
TransformModel.HOMOGRAPHY
    Full projective transform (8 DOF).  Estimated via cv2.findHomography
    with RANSAC.  Requires ≥ 4 non-collinear point pairs.  Correct for
    planar scenes viewed from varying viewpoints.

TransformModel.AFFINE
    Full affine transform (6 DOF) — rotation, scale, shear, translation.
    Estimated via cv2.estimateAffine2D with RANSAC.  Requires ≥ 3 non-
    collinear pairs.  Preferred for near-nadir imagery.

TransformModel.SIMILARITY
    Similarity transform (4 DOF) — isotropic scale + rotation + translation.
    Estimated via cv2.estimateAffinePartial2D with RANSAC.  Requires ≥ 2
    pairs.  Appropriate when scale is the dominant unknown.

TransformModel.TRANSLATION
    Pure translation (2 DOF).  Estimated as the robust median of
    displacement vectors.  Requires ≥ 1 pair.  Use when images are
    known to be same-scale and near-nadir.

M2 Input Contract
-----------------
The estimator accepts a correspondence dict produced by:
    src.correspondence.feature_matching.match_sift_features()
    src.correspondence.feature_matching.match_with_candidate_roi()

Required keys:
    source_points    : np.ndarray (N, 2) float32 — [x, y] in source image
    reference_points : np.ndarray (N, 2) float32 — [x, y] in reference image
    num_matches      : int

Output
------
EstimationResult — dataclass containing:
    success          : bool
    transform_type   : str   — model name string
    transform_matrix : Optional[np.ndarray]  — shape depends on model
    inlier_mask      : Optional[np.ndarray]  — (N,) bool
    num_inliers      : int
    num_outliers     : int
    total_matches    : int
    inlier_ratio     : float ∈ [0, 1]
    model            : TransformModel
    reason           : Optional[str]
    diagnostics      : Dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum point counts per model (for pre-RANSAC gate)
_MIN_POINTS: Dict[str, int] = {
    "HOMOGRAPHY": 4,
    "AFFINE": 3,
    "SIMILARITY": 2,
    "TRANSLATION": 1,
}

# Minimum determinant magnitude for a valid homography (near-singular check)
_HOMOGRAPHY_DET_EPS: float = 1e-6

# Minimum scale factor magnitude (catches degenerate shrink-to-point)
_AFFINE_SCALE_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransformModel(Enum):
    """Supported geometric transform models for estimation."""

    HOMOGRAPHY = auto()   # 8-DOF full projective
    AFFINE = auto()       # 6-DOF affine
    SIMILARITY = auto()   # 4-DOF similarity (isotropic scale + rotation)
    TRANSLATION = auto()  # 2-DOF pure translation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GeometricEstimatorConfig:
    """Configuration for the robust geometric estimator.

    Attributes:
        model: Transform model to estimate (TransformModel enum).
        ransac_reproj_threshold: RANSAC inlier threshold in pixels.  Points
            with reprojection error below this are classified as inliers.
        ransac_confidence: Desired probability that the estimated model is
            correct (influences number of RANSAC iterations).
        ransac_max_iters: Hard cap on RANSAC iterations.
        min_inliers: Minimum absolute inlier count for a valid estimation.
        min_inlier_ratio: Minimum inlier ratio (inliers / total matches)
            for a valid estimation.
        use_magsac: If True, attempt MAGSAC++ estimator for homography
            (requires OpenCV ≥ 4.5).  Falls back to standard RANSAC if
            cv2.USAC_MAGSAC is not available.
        refine_with_lm: If True, refine the estimated model by re-estimating
            on the RANSAC inlier subset only (equivalent to LM refinement).
    """

    model: TransformModel = TransformModel.HOMOGRAPHY
    ransac_reproj_threshold: float = 5.0
    ransac_confidence: float = 0.995
    ransac_max_iters: int = 2000
    min_inliers: int = 8
    min_inlier_ratio: float = 0.15
    use_magsac: bool = False
    refine_with_lm: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EstimationResult:
    """Structured output from the GeometricEstimator.

    Attributes:
        success: True if a valid model was found.
        transform_type: Human-readable model name string (e.g. 'HOMOGRAPHY').
        transform_matrix: Estimated transform matrix.
            - HOMOGRAPHY   → (3, 3) float64
            - AFFINE       → (2, 3) float64
            - SIMILARITY   → (2, 3) float64
            - TRANSLATION  → (2, 3) float64  (identity + translation columns)
            None on failure.
        inlier_mask: Boolean array of shape (N,) marking RANSAC inliers.
            None on failure.
        num_inliers: Number of RANSAC inlier matches.
        num_outliers: Number of RANSAC outlier matches.
        total_matches: Total N (num_inliers + num_outliers).
        inlier_ratio: num_inliers / total_matches.
        model: The TransformModel enum used.
        reason: Human-readable failure code when success=False, None on success.
        diagnostics: Additional key/value pairs for debugging and downstream
            stages (e.g. determinant, condition number).
    """

    success: bool
    transform_type: str
    transform_matrix: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    num_inliers: int
    num_outliers: int
    total_matches: int
    inlier_ratio: float
    model: TransformModel
    reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Aliases for backward compatibility with Phase 1 stubs
    @property
    def inlier_count(self) -> int:
        """Alias for num_inliers (backward-compat with Phase 1 stub)."""
        return self.num_inliers

    @property
    def failure_reason(self) -> Optional[str]:
        """Alias for reason (backward-compat with Phase 1 stub)."""
        return self.reason


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_failure(
    model: TransformModel,
    total: int,
    reason: str,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> EstimationResult:
    """Construct a failed EstimationResult with standard fields."""
    return EstimationResult(
        success=False,
        transform_type=model.name,
        transform_matrix=None,
        inlier_mask=None,
        num_inliers=0,
        num_outliers=total,
        total_matches=total,
        inlier_ratio=0.0,
        model=model,
        reason=reason,
        diagnostics=diagnostics or {},
    )


def _sanitise_points(
    src: np.ndarray,
    ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter out rows where either src or ref contains NaN / Inf values.

    Args:
        src: (N, 2) source point array.
        ref: (N, 2) reference point array.

    Returns:
        Tuple (clean_src, clean_ref, valid_mask) where valid_mask is a
        boolean (N,) array indicating which original rows are valid.
    """
    finite_src = np.isfinite(src).all(axis=1)
    finite_ref = np.isfinite(ref).all(axis=1)
    valid_mask = finite_src & finite_ref
    return src[valid_mask], ref[valid_mask], valid_mask


def _parse_mask(
    raw_mask: Optional[np.ndarray],
    n: int,
    default_all_inliers: bool = False,
) -> np.ndarray:
    """Convert an OpenCV mask (N,1) uint8 or None to a (N,) bool array.

    Args:
        raw_mask: Mask returned by cv2 estimation functions.
        n: Number of point pairs.
        default_all_inliers: If True and raw_mask is None, return all-True
            array.  Used for translation (no mask returned).

    Returns:
        (N,) bool numpy array.
    """
    if raw_mask is None:
        return np.ones(n, dtype=bool) if default_all_inliers else np.zeros(n, dtype=bool)
    return raw_mask.ravel().astype(bool)


def _validate_matrix_finite(matrix: np.ndarray) -> bool:
    """Return True if all elements of matrix are finite."""
    return bool(np.isfinite(matrix).all())


# ---------------------------------------------------------------------------
# Main estimator class
# ---------------------------------------------------------------------------


class GeometricEstimator:
    """Robust geometric model estimator using RANSAC.

    Consumes the M2 correspondence dict (approved by EvidenceGate) and
    produces a validated transform matrix with an inlier mask.

    Usage::

        estimator = GeometricEstimator(
            config=GeometricEstimatorConfig(model=TransformModel.HOMOGRAPHY)
        )
        result = estimator.estimate(correspondence_dict)
        if result.success:
            H = result.transform_matrix     # (3, 3) float64
            mask = result.inlier_mask       # (N,) bool

    Args:
        config: GeometricEstimatorConfig.  Defaults to GeometricEstimatorConfig().
    """

    def __init__(self, config: Optional[GeometricEstimatorConfig] = None) -> None:
        self.config = config or GeometricEstimatorConfig()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(self, correspondence: Dict[str, Any]) -> EstimationResult:
        """Estimate a geometric transform from an M2 correspondence dict.

        Performs the following steps:
        1. Validate required keys are present.
        2. Extract and sanitise point arrays (reject NaN/Inf rows).
        3. Check minimum point count for the configured model.
        4. Dispatch to the appropriate RANSAC backend.
        5. Validate the returned matrix (non-None, finite, non-singular).
        6. Check inlier count and ratio thresholds.
        7. Optionally refine on inlier subset.
        8. Return a populated EstimationResult.

        Args:
            correspondence: Correspondence dict from M2 or upstream M3 gate.
                Must contain 'source_points' (N,2) and 'reference_points' (N,2).

        Returns:
            EstimationResult.  success=True only if all checks pass.

        Raises:
            ValueError: If required keys are missing from correspondence.
        """
        model = self.config.model

        # ---- 1. Key validation --------------------------------------------------
        self._validate_required_keys(correspondence)

        src_raw = correspondence["source_points"]
        ref_raw = correspondence["reference_points"]

        # ---- 2. Shape validation ------------------------------------------------
        src_raw = np.asarray(src_raw, dtype=np.float64)
        ref_raw = np.asarray(ref_raw, dtype=np.float64)

        if src_raw.ndim != 2 or src_raw.shape[1] != 2:
            raise ValueError(
                f"source_points must be shape (N, 2), got {src_raw.shape}."
            )
        if ref_raw.ndim != 2 or ref_raw.shape[1] != 2:
            raise ValueError(
                f"reference_points must be shape (N, 2), got {ref_raw.shape}."
            )
        if src_raw.shape[0] != ref_raw.shape[0]:
            raise ValueError(
                f"source_points ({src_raw.shape[0]}) and reference_points "
                f"({ref_raw.shape[0]}) must have the same number of rows."
            )

        total_raw = src_raw.shape[0]

        # ---- 3. NaN / Inf sanitisation ------------------------------------------
        src_clean, ref_clean, valid_mask = _sanitise_points(src_raw, ref_raw)
        n_valid = src_clean.shape[0]
        n_dropped = total_raw - n_valid

        if n_valid == 0:
            return _make_failure(
                model, total_raw, "all_points_invalid",
                {"dropped_invalid": n_dropped},
            )

        # ---- 4. Minimum point count gate ----------------------------------------
        min_pts = _MIN_POINTS[model.name]
        if n_valid < min_pts:
            return _make_failure(
                model, n_valid,
                f"insufficient_correspondences (need {min_pts}, got {n_valid})",
                {"min_required": min_pts, "n_valid": n_valid,
                 "dropped_invalid": n_dropped},
            )

        # ---- 5. Dispatch to backend ---------------------------------------------
        result = self._dispatch(model, src_clean, ref_clean, n_valid)

        # If backend already failed, pass through
        if not result.success:
            return result

        # ---- 6. Post-estimation threshold checks --------------------------------
        if result.num_inliers < self.config.min_inliers:
            return _make_failure(
                model, n_valid,
                f"insufficient_inliers (got {result.num_inliers}, "
                f"need {self.config.min_inliers})",
                {**result.diagnostics, "num_inliers": result.num_inliers},
            )

        if result.inlier_ratio < self.config.min_inlier_ratio:
            return _make_failure(
                model, n_valid,
                f"low_inlier_ratio (got {result.inlier_ratio:.3f}, "
                f"need {self.config.min_inlier_ratio:.3f})",
                {**result.diagnostics, "inlier_ratio": result.inlier_ratio},
            )

        # ---- 7. Optional inlier-subset refinement --------------------------------
        if self.config.refine_with_lm and result.inlier_mask is not None:
            result = self._refine_on_inliers(model, src_clean, ref_clean, result)

        return result

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        model: TransformModel,
        src: np.ndarray,
        ref: np.ndarray,
        n: int,
    ) -> EstimationResult:
        """Route to the correct backend method."""
        if model == TransformModel.HOMOGRAPHY:
            return self._estimate_homography(src, ref)
        elif model == TransformModel.AFFINE:
            return self._estimate_affine(src, ref)
        elif model == TransformModel.SIMILARITY:
            return self._estimate_similarity(src, ref)
        elif model == TransformModel.TRANSLATION:
            return self._estimate_translation(src, ref)
        else:
            raise ValueError(f"Unsupported TransformModel: {model}")

    # ------------------------------------------------------------------
    # Estimation backends
    # ------------------------------------------------------------------

    def _estimate_homography(
        self,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
    ) -> EstimationResult:
        """Estimate homography via cv2.findHomography with RANSAC / MAGSAC++.

        Args:
            src_pts: (N, 2) float64 source points (pre-sanitised).
            ref_pts: (N, 2) float64 reference points (pre-sanitised).

        Returns:
            EstimationResult with (3, 3) transform_matrix on success.
        """
        model = TransformModel.HOMOGRAPHY
        n = src_pts.shape[0]

        # Select RANSAC method — try MAGSAC++ if requested and available
        method = cv2.RANSAC
        if self.config.use_magsac:
            magsac_flag = getattr(cv2, "USAC_MAGSAC", None)
            if magsac_flag is not None:
                method = magsac_flag

        try:
            H, raw_mask = cv2.findHomography(
                src_pts.reshape(-1, 1, 2).astype(np.float32),
                ref_pts.reshape(-1, 1, 2).astype(np.float32),
                method=method,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
                confidence=self.config.ransac_confidence,
                maxIters=self.config.ransac_max_iters,
            )
        except cv2.error as exc:
            return _make_failure(model, n, f"cv2_error: {exc}")

        # Degenerate: OpenCV returned None
        if H is None:
            return _make_failure(model, n, "ransac_failed_no_solution")

        # Non-finite values in matrix
        if not _validate_matrix_finite(H):
            return _make_failure(model, n, "invalid_matrix_nonfinite")

        # Near-singular homography check
        det = float(np.linalg.det(H))
        if abs(det) < _HOMOGRAPHY_DET_EPS:
            return _make_failure(
                model, n, "degenerate_homography_near_singular",
                {"det": det},
            )

        mask = _parse_mask(raw_mask, n)
        num_inliers = int(mask.sum())
        num_outliers = n - num_inliers

        return EstimationResult(
            success=True,
            transform_type=model.name,
            transform_matrix=H.astype(np.float64),
            inlier_mask=mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            total_matches=n,
            inlier_ratio=num_inliers / n if n > 0 else 0.0,
            model=model,
            reason=None,
            diagnostics={"det": det, "method": method},
        )

    def _estimate_affine(
        self,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
    ) -> EstimationResult:
        """Estimate full affine transform (6 DOF) via cv2.estimateAffine2D.

        Args:
            src_pts: (N, 2) float64 source points (pre-sanitised).
            ref_pts: (N, 2) float64 reference points (pre-sanitised).

        Returns:
            EstimationResult with (2, 3) transform_matrix on success.
        """
        model = TransformModel.AFFINE
        n = src_pts.shape[0]

        try:
            M, raw_mask = cv2.estimateAffine2D(
                src_pts.reshape(-1, 1, 2).astype(np.float32),
                ref_pts.reshape(-1, 1, 2).astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
                confidence=self.config.ransac_confidence,
                maxIters=self.config.ransac_max_iters,
            )
        except cv2.error as exc:
            return _make_failure(model, n, f"cv2_error: {exc}")

        if M is None:
            return _make_failure(model, n, "ransac_failed_no_solution")

        if not _validate_matrix_finite(M):
            return _make_failure(model, n, "invalid_matrix_nonfinite")

        # Check affine submatrix scale (2×2 linear part)
        A = M[:2, :2]
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        if scale < _AFFINE_SCALE_EPS:
            return _make_failure(
                model, n, "degenerate_affine_zero_scale",
                {"scale": scale},
            )

        mask = _parse_mask(raw_mask, n)
        num_inliers = int(mask.sum())
        num_outliers = n - num_inliers

        return EstimationResult(
            success=True,
            transform_type=model.name,
            transform_matrix=M.astype(np.float64),
            inlier_mask=mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            total_matches=n,
            inlier_ratio=num_inliers / n if n > 0 else 0.0,
            model=model,
            reason=None,
            diagnostics={"scale": scale},
        )

    def _estimate_similarity(
        self,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
    ) -> EstimationResult:
        """Estimate similarity transform (4 DOF) via cv2.estimateAffinePartial2D.

        Args:
            src_pts: (N, 2) float64 source points (pre-sanitised).
            ref_pts: (N, 2) float64 reference points (pre-sanitised).

        Returns:
            EstimationResult with (2, 3) transform_matrix on success.
        """
        model = TransformModel.SIMILARITY
        n = src_pts.shape[0]

        try:
            M, raw_mask = cv2.estimateAffinePartial2D(
                src_pts.reshape(-1, 1, 2).astype(np.float32),
                ref_pts.reshape(-1, 1, 2).astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.ransac_reproj_threshold,
                confidence=self.config.ransac_confidence,
                maxIters=self.config.ransac_max_iters,
            )
        except cv2.error as exc:
            return _make_failure(model, n, f"cv2_error: {exc}")

        if M is None:
            return _make_failure(model, n, "ransac_failed_no_solution")

        if not _validate_matrix_finite(M):
            return _make_failure(model, n, "invalid_matrix_nonfinite")

        # Scale extracted from similarity matrix [a, -b, tx; b, a, ty]
        a = float(M[0, 0])
        b = float(M[1, 0])
        scale = float(np.sqrt(a * a + b * b))
        if scale < _AFFINE_SCALE_EPS:
            return _make_failure(
                model, n, "degenerate_similarity_zero_scale",
                {"scale": scale},
            )

        mask = _parse_mask(raw_mask, n)
        num_inliers = int(mask.sum())
        num_outliers = n - num_inliers

        return EstimationResult(
            success=True,
            transform_type=model.name,
            transform_matrix=M.astype(np.float64),
            inlier_mask=mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            total_matches=n,
            inlier_ratio=num_inliers / n if n > 0 else 0.0,
            model=model,
            reason=None,
            diagnostics={"scale": scale},
        )

    def _estimate_translation(
        self,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
    ) -> EstimationResult:
        """Estimate pure translation as the robust median of displacement vectors.

        Each point pair contributes a displacement (dx, dy) = ref - src.
        The median is used as the robust translation estimate.  Inliers are
        pairs whose displacement is within ransac_reproj_threshold of the
        median.

        Args:
            src_pts: (N, 2) float64 source points.
            ref_pts: (N, 2) float64 reference points.

        Returns:
            EstimationResult with (2, 3) affine-compatible matrix encoding
            the translation: [[1, 0, dx], [0, 1, dy]].
        """
        model = TransformModel.TRANSLATION
        n = src_pts.shape[0]

        displacements = ref_pts - src_pts          # (N, 2)
        tx = float(np.median(displacements[:, 0]))
        ty = float(np.median(displacements[:, 1]))

        # Classify inliers: distance from median displacement < threshold
        dist = np.sqrt(
            (displacements[:, 0] - tx) ** 2
            + (displacements[:, 1] - ty) ** 2
        )
        mask = dist < self.config.ransac_reproj_threshold
        num_inliers = int(mask.sum())
        num_outliers = n - num_inliers

        if num_inliers == 0:
            return _make_failure(model, n, "ransac_failed_no_solution")

        # Build affine-compatible (2, 3) matrix
        M = np.array([[1.0, 0.0, tx],
                      [0.0, 1.0, ty]], dtype=np.float64)

        return EstimationResult(
            success=True,
            transform_type=model.name,
            transform_matrix=M,
            inlier_mask=mask,
            num_inliers=num_inliers,
            num_outliers=num_outliers,
            total_matches=n,
            inlier_ratio=num_inliers / n if n > 0 else 0.0,
            model=model,
            reason=None,
            diagnostics={"tx": tx, "ty": ty},
        )

    # ------------------------------------------------------------------
    # Inlier-subset refinement
    # ------------------------------------------------------------------

    def _refine_on_inliers(
        self,
        model: TransformModel,
        src_pts: np.ndarray,
        ref_pts: np.ndarray,
        initial: EstimationResult,
    ) -> EstimationResult:
        """Re-estimate the transform using only RANSAC inliers as input.

        This is equivalent to a non-iterative LM refinement step:
        re-running the linear estimation on the clean inlier set gives a
        better-conditioned solution without RANSAC overhead.

        If re-estimation fails (degenerate inlier set), the original
        EstimationResult is returned unchanged.

        Args:
            model: TransformModel enum.
            src_pts: Full (N, 2) source point array.
            ref_pts: Full (N, 2) reference point array.
            initial: EstimationResult from the RANSAC pass.

        Returns:
            Refined EstimationResult, or initial on failure.
        """
        mask = initial.inlier_mask
        if mask is None or mask.sum() < _MIN_POINTS[model.name]:
            return initial

        src_in = src_pts[mask]
        ref_in = ref_pts[mask]
        n_in = src_in.shape[0]

        try:
            if model == TransformModel.HOMOGRAPHY:
                M_refined, _ = cv2.findHomography(
                    src_in.reshape(-1, 1, 2).astype(np.float32),
                    ref_in.reshape(-1, 1, 2).astype(np.float32),
                    method=0,   # Least-squares (no RANSAC)
                )
                if M_refined is None or not _validate_matrix_finite(M_refined):
                    return initial
                det = float(np.linalg.det(M_refined))
                if abs(det) < _HOMOGRAPHY_DET_EPS:
                    return initial
                refined_matrix = M_refined.astype(np.float64)
                diag = {**initial.diagnostics, "refined": True, "det_refined": det}

            elif model in (TransformModel.AFFINE, TransformModel.SIMILARITY):
                fn = (cv2.estimateAffine2D if model == TransformModel.AFFINE
                      else cv2.estimateAffinePartial2D)
                M_refined, _ = fn(
                    src_in.reshape(-1, 1, 2).astype(np.float32),
                    ref_in.reshape(-1, 1, 2).astype(np.float32),
                    method=cv2.LMEDS,   # Least-median-of-squares on clean set
                )
                if M_refined is None or not _validate_matrix_finite(M_refined):
                    return initial
                refined_matrix = M_refined.astype(np.float64)
                diag = {**initial.diagnostics, "refined": True}

            else:
                # Translation: re-compute mean on inliers (more accurate than median)
                displacements = ref_in - src_in
                tx = float(displacements[:, 0].mean())
                ty = float(displacements[:, 1].mean())
                refined_matrix = np.array([[1.0, 0.0, tx],
                                           [0.0, 1.0, ty]], dtype=np.float64)
                diag = {**initial.diagnostics, "refined": True,
                        "tx_refined": tx, "ty_refined": ty}

        except (cv2.error, np.linalg.LinAlgError):
            return initial

        return EstimationResult(
            success=True,
            transform_type=initial.transform_type,
            transform_matrix=refined_matrix,
            inlier_mask=initial.inlier_mask,
            num_inliers=initial.num_inliers,
            num_outliers=initial.num_outliers,
            total_matches=initial.total_matches,
            inlier_ratio=initial.inlier_ratio,
            model=model,
            reason=None,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_keys(correspondence: Dict[str, Any]) -> None:
        """Raise ValueError if required M2 keys are absent.

        Args:
            correspondence: M2 correspondence dict to validate.

        Raises:
            ValueError: Listing all missing required keys.
        """
        required = ("source_points", "reference_points")
        missing = [k for k in required if k not in correspondence]
        if missing:
            raise ValueError(
                f"correspondence dict is missing required keys: {missing}. "
                f"Got keys: {list(correspondence.keys())}"
            )
