"""Reprojection error computation for M3 Validation pipeline.

This module is the fourth stage of the M3 pipeline.  It computes per-point
reprojection errors for all correspondence pairs (inliers and outliers alike)
using the transform matrix estimated by GeometricEstimator (Phase 2).

Reprojection Error Definition
------------------------------
For a transform T and a match (src_i, ref_i):

    forward_error_i = || project(T, src_i) − ref_i ||₂

where `project` performs the model-appropriate projection:
  - Homography (3×3): perspective division  w = T·src; p = w[:2] / w[2]
  - Affine / Similarity / Translation (2×3): linear multiply + add

Symmetric reprojection error (optional):
    sym_error_i = 0.5 · (forward_error_i + || project(T⁻¹, ref_i) − src_i ||₂)

Scope (Phase 4)
---------------
This module deliberately does NOT:
  - Apply image warping / registration (Phase 5 — registration.py).
  - Compute a final quality score or accuracy percentage (Phase 6 — metrics.py).
  - Require ground-truth data for any computed value.

All inputs are the correspondence arrays from M2 and the transform produced by
M3 Phase 2 (GeometricEstimator) together with the inlier mask from Phase 2/3.

M3 Input Contract
-----------------
    source_points    : np.ndarray (N, 2) float  — source [x, y] coords
    reference_points : np.ndarray (N, 2) float  — reference [x, y] coords
    transform_matrix : np.ndarray               — (3,3) homography or (2,3) affine
    inlier_mask      : np.ndarray (N,) bool     — RANSAC inlier flags (optional)
    model_type       : str                       — 'homography' | 'affine' |
                                                   'similarity' | 'translation'

Output
------
ReprojectionReport — dataclass containing:
    per_match_errors      : np.ndarray (N,) float64  — per-point forward errors
    per_match_sym_errors  : Optional[np.ndarray] (N,) — symmetric errors
    inlier_errors         : np.ndarray (M,) float64  — errors for inliers only
    outlier_errors        : np.ndarray (K,) float64  — errors for outliers only
    inlier_mean_error     : float
    inlier_median_error   : float
    inlier_rmse           : float
    inlier_max_error      : float
    outlier_mean_error    : float
    overall_mean_error    : float
    overall_median_error  : float
    overall_rmse          : float
    overall_max_error     : float
    diagnostics           : Dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported model type strings (lower-cased for comparison)
_AFFINE_MODELS = frozenset({"affine", "similarity", "translation"})
_HOMOGRAPHY_MODELS = frozenset({"homography"})
_ALL_MODELS = _AFFINE_MODELS | _HOMOGRAPHY_MODELS

# Guard for near-zero w in homography perspective division
_HOMOGRAPHY_W_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReprojectionReport:
    """Structured reprojection error report for all N correspondence pairs.

    Attributes:
        per_match_errors: Forward reprojection error for every correspondence
            pair, including both inliers and outliers. Shape (N,) float64.
        per_match_sym_errors: Symmetric reprojection error averaged over
            forward and backward directions. None if not requested.
        inlier_errors: Per-match forward errors for RANSAC inlier subset.
            Shape (M,) float64.  Empty array when no inliers or no mask.
        outlier_errors: Per-match forward errors for RANSAC outlier subset.
            Shape (K,) float64.  Empty array when no outliers or no mask.
        inlier_mean_error: Mean forward error over inliers. 0.0 if no inliers.
        inlier_median_error: Median forward error over inliers.
        inlier_rmse: Root-mean-square error over inliers.
        inlier_max_error: Maximum forward error over inliers.
        outlier_mean_error: Mean forward error over outliers. 0.0 if no outliers.
        overall_mean_error: Mean forward error over all N matches.
        overall_median_error: Median forward error over all N matches.
        overall_rmse: Root-mean-square error over all N matches.
        overall_max_error: Maximum forward error over all N matches.
        diagnostics: Additional data — percentile breakdown, model_type, N, M, K.
    """

    per_match_errors: np.ndarray
    per_match_sym_errors: Optional[np.ndarray]

    # Split error arrays
    inlier_errors: np.ndarray
    outlier_errors: np.ndarray

    # Aggregate statistics — inlier subset
    inlier_mean_error: float
    inlier_median_error: float
    inlier_rmse: float
    inlier_max_error: float

    # Aggregate statistics — outlier subset
    outlier_mean_error: float

    # Aggregate statistics — all N matches
    overall_mean_error: float
    overall_median_error: float
    overall_rmse: float
    overall_max_error: float

    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transform_matrix: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    model_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Validate and coerce all inputs to canonical dtypes.

    Returns:
        (src_f64, ref_f64, M_f64, mask_bool_or_None)

    Raises:
        TypeError: If any required argument is not an ndarray.
        ValueError: For shape mismatches, wrong dimensionality, unknown model.
    """
    # Type checks
    for name, arr in [
        ("source_points", source_points),
        ("reference_points", reference_points),
        ("transform_matrix", transform_matrix),
    ]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"'{name}' must be a numpy ndarray, got {type(arr).__name__}."
            )

    # Model type validation
    mt = model_type.lower()
    if mt not in _ALL_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {sorted(_ALL_MODELS)}."
        )

    # Coerce to float64
    src = np.asarray(source_points, dtype=np.float64)
    ref = np.asarray(reference_points, dtype=np.float64)
    M = np.asarray(transform_matrix, dtype=np.float64)

    # Point array shape checks
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            f"source_points must be shape (N, 2), got {source_points.shape}."
        )
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(
            f"reference_points must be shape (N, 2), got {reference_points.shape}."
        )
    if src.shape[0] != ref.shape[0]:
        raise ValueError(
            f"source_points ({src.shape[0]}) and reference_points ({ref.shape[0]}) "
            "must have the same number of rows."
        )

    # Transform matrix shape check
    if mt in _HOMOGRAPHY_MODELS:
        if M.shape != (3, 3):
            raise ValueError(
                f"Homography transform_matrix must be (3, 3), got {M.shape}."
            )
    else:  # affine / similarity / translation
        if M.shape != (2, 3):
            raise ValueError(
                f"Affine/similarity/translation transform_matrix must be (2, 3), "
                f"got {M.shape}."
            )

    # Validate transform matrix is finite
    if not np.isfinite(M).all():
        raise ValueError("transform_matrix contains NaN or Inf values.")

    # Inlier mask validation
    mask: Optional[np.ndarray] = None
    if inlier_mask is not None:
        if not isinstance(inlier_mask, np.ndarray):
            raise TypeError(
                f"'inlier_mask' must be a numpy ndarray, got {type(inlier_mask).__name__}."
            )
        mask = np.asarray(inlier_mask).ravel().astype(bool)
        if mask.shape[0] != src.shape[0]:
            raise ValueError(
                f"inlier_mask length ({mask.shape[0]}) does not match "
                f"source_points length ({src.shape[0]})."
            )

    return src, ref, M, mask


def _project_points(
    points: np.ndarray,
    transform_matrix: np.ndarray,
    model_type: str,
) -> np.ndarray:
    """Project (N, 2) points through a transform matrix.

    Args:
        points: (N, 2) float64 [x, y] input points.
        transform_matrix: (3,3) for homography or (2,3) for affine-family.
        model_type: Lowercase model string.

    Returns:
        (N, 2) float64 projected [x', y'] coordinates.
        Rows where projection is degenerate (w ≈ 0 for homography) are
        set to NaN so downstream distance computation produces NaN rather
        than incorrect values.
    """
    n = points.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Homogeneous source coordinates (N, 3)
    ones = np.ones((n, 1), dtype=np.float64)
    pts_h = np.hstack([points, ones])  # (N, 3)

    mt = model_type.lower()

    if mt in _HOMOGRAPHY_MODELS:
        # (3,3) @ (3,N) → (3,N), then transpose
        projected_h = (transform_matrix @ pts_h.T).T   # (N, 3)
        w = projected_h[:, 2:3]                         # (N, 1)
        # Guard against near-zero w (degenerate perspective division)
        safe = np.abs(w) > _HOMOGRAPHY_W_EPS
        projected_xy = np.where(
            safe,
            projected_h[:, :2] / np.where(safe, w, 1.0),
            np.nan,
        )
    else:
        # Affine-family: (2,3) @ (3,N) → (2,N), then transpose
        projected_xy = (transform_matrix @ pts_h.T).T   # (N, 2)

    return projected_xy.astype(np.float64)


def _euclidean_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-row Euclidean distance between two (N,2) arrays.

    NaN propagates naturally: if either a or b has NaN in row i, the
    distance for row i will be NaN.

    Returns:
        (N,) float64 distance array.
    """
    diff = a - b
    return np.sqrt((diff ** 2).sum(axis=1))


def _aggregate_stats(errors: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute (mean, median, rmse, max) over a 1-D error array.

    NaN values are excluded from all statistics.  If the array is empty
    or all-NaN, returns (0.0, 0.0, 0.0, 0.0).
    """
    valid = errors[np.isfinite(errors)]
    if valid.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean_ = float(np.mean(valid))
    median_ = float(np.median(valid))
    rmse_ = float(np.sqrt(np.mean(valid ** 2)))
    max_ = float(np.max(valid))
    return mean_, median_, rmse_, max_


def _percentile_diag(errors: np.ndarray) -> Dict[str, float]:
    """Return a dict of error percentiles for diagnostics."""
    valid = errors[np.isfinite(errors)]
    if valid.size == 0:
        return {f"p{p}": 0.0 for p in (25, 50, 75, 90, 95, 99)}
    return {
        f"p{p}": float(np.percentile(valid, p))
        for p in (25, 50, 75, 90, 95, 99)
    }


def _sanitise_input_points(
    src: np.ndarray, ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows containing NaN or Inf from src/ref, return a valid_mask.

    Returns:
        (clean_src, clean_ref, valid_mask) where valid_mask is (N,) bool.
    """
    finite_src = np.isfinite(src).all(axis=1)
    finite_ref = np.isfinite(ref).all(axis=1)
    valid = finite_src & finite_ref
    return src[valid], ref[valid], valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reprojection_errors(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transform_matrix: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    model_type: str = "homography",
    compute_symmetric: bool = False,
) -> ReprojectionReport:
    """Compute per-match reprojection errors for all N correspondences.

    Projects each source point through the transform and measures the
    Euclidean distance to its reference point.  Provides inlier/outlier
    split statistics and overall aggregate statistics.

    Args:
        source_points: (N, 2) float array of [x, y] source coordinates.
        reference_points: (N, 2) float array of [x, y] reference coordinates.
        transform_matrix: Estimated transform matrix from GeometricEstimator.
            Shape: (3, 3) for homography, (2, 3) for affine/similarity/translation.
        inlier_mask: Optional (N,) bool array from RANSAC / InlierAnalyzer.
            If None, all matches are treated as inliers for aggregate statistics.
        model_type: Transform model string.
            One of 'homography', 'affine', 'similarity', 'translation'.
        compute_symmetric: If True, also compute symmetric reprojection error
            using the matrix inverse.

    Returns:
        ReprojectionReport with per-match errors and aggregate statistics.

    Raises:
        TypeError: If any required argument is not a numpy ndarray.
        ValueError: If shapes are inconsistent, model_type is unknown, or
            transform_matrix is not finite or has wrong shape.
    """
    mt = model_type.lower()

    # ---- 1. Validate and coerce inputs --------------------------------------
    src, ref, M, mask = _validate_inputs(
        source_points, reference_points, transform_matrix, inlier_mask, model_type
    )
    n_total = src.shape[0]

    # ---- 2. Handle empty input -----------------------------------------------
    empty1d = np.empty(0, dtype=np.float64)
    empty2d = np.empty((0, 2), dtype=np.float64)

    if n_total == 0:
        return ReprojectionReport(
            per_match_errors=empty1d,
            per_match_sym_errors=empty1d if compute_symmetric else None,
            inlier_errors=empty1d,
            outlier_errors=empty1d,
            inlier_mean_error=0.0,
            inlier_median_error=0.0,
            inlier_rmse=0.0,
            inlier_max_error=0.0,
            outlier_mean_error=0.0,
            overall_mean_error=0.0,
            overall_median_error=0.0,
            overall_rmse=0.0,
            overall_max_error=0.0,
            diagnostics={"note": "empty_input", "model_type": mt, "N": 0},
        )

    # ---- 3. Forward projection -----------------------------------------------
    projected = _project_points(src, M, mt)    # (N, 2) — may contain NaN for
                                               # degenerate homography rows

    # Per-match forward errors (N,): NaN if projection was degenerate
    per_errors = _euclidean_distances(projected, ref)

    # ---- 4. Symmetric reprojection (optional) --------------------------------
    per_sym: Optional[np.ndarray] = None
    if compute_symmetric and mt in _HOMOGRAPHY_MODELS:
        try:
            M_inv = np.linalg.inv(M)
            back_projected = _project_points(ref, M_inv, mt)
            back_errors = _euclidean_distances(back_projected, src)
            per_sym = 0.5 * (per_errors + back_errors)
        except np.linalg.LinAlgError:
            per_sym = None  # singular matrix — skip symmetric
    elif compute_symmetric and mt in _AFFINE_MODELS:
        # Affine inverse exists analytically
        try:
            # Extend (2,3) to (3,3), invert, slice back to (2,3)
            M33 = np.eye(3, dtype=np.float64)
            M33[:2, :] = M
            M33_inv = np.linalg.inv(M33)
            M_inv_23 = M33_inv[:2, :]
            back_projected = _project_points(ref, M_inv_23, mt)
            back_errors = _euclidean_distances(back_projected, src)
            per_sym = 0.5 * (per_errors + back_errors)
        except np.linalg.LinAlgError:
            per_sym = None

    # ---- 5. Inlier / outlier split ------------------------------------------
    if mask is not None:
        outlier_mask = ~mask
        inlier_errors = per_errors[mask]
        outlier_errors = per_errors[outlier_mask]
    else:
        # No mask provided — treat all as inliers
        inlier_errors = per_errors.copy()
        outlier_errors = empty1d

    # ---- 6. Aggregate statistics --------------------------------------------
    in_mean, in_med, in_rmse, in_max = _aggregate_stats(inlier_errors)
    out_mean, _, _, _ = _aggregate_stats(outlier_errors)
    all_mean, all_med, all_rmse, all_max = _aggregate_stats(per_errors)

    # ---- 7. Diagnostics ------------------------------------------------------
    n_inliers = int(mask.sum()) if mask is not None else n_total
    n_outliers = n_total - n_inliers
    diag: Dict[str, Any] = {
        "model_type": mt,
        "N": n_total,
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "n_nan_errors": int(np.isnan(per_errors).sum()),
        **_percentile_diag(per_errors),
    }

    return ReprojectionReport(
        per_match_errors=per_errors,
        per_match_sym_errors=per_sym,
        inlier_errors=inlier_errors,
        outlier_errors=outlier_errors,
        inlier_mean_error=in_mean,
        inlier_median_error=in_med,
        inlier_rmse=in_rmse,
        inlier_max_error=in_max,
        outlier_mean_error=out_mean,
        overall_mean_error=all_mean,
        overall_median_error=all_med,
        overall_rmse=all_rmse,
        overall_max_error=all_max,
        diagnostics=diag,
    )
