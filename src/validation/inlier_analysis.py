"""Inlier classification and spatial analysis for M3 Validation pipeline.

This module is the third stage of the M3 pipeline.  It takes the RANSAC
inlier mask produced by GeometricEstimator (Phase 2) and the raw M2
correspondence points, then produces a fully-structured InlierReport with:

1. Separated inlier source / reference point arrays.
2. Separated outlier source / reference point arrays.
3. Inlier count, outlier count, and inlier ratio.
4. Spatial coverage — a measure of how evenly inlier points cover the image
   area, computed via a grid-based occupancy entropy approach.
5. Input validation with safe handling of empty and invalid inputs.

Scope (Phase 3)
---------------
This module deliberately does NOT:
  - Compute reprojection errors (Phase 4 — reprojection.py).
  - Estimate a registration transform (Phase 5 — registration.py).
  - Compute accuracy metrics that require ground truth.

All results are derived only from the inlier mask and correspondence points
produced by M2 and Phase 2.

Spatial Coverage Definition
----------------------------
The image plane (or point bounding box) is divided into a G×G grid of cells.
The occupancy distribution p[i] = (points in cell i) / N is computed and its
normalised Shannon entropy is used as the coverage score:

    coverage = H(p) / log(G²)

where H(p) = -Σ p[i] * log(p[i]+ε).  This gives:
    0.0  — all inliers clustered in a single cell (worst)
    1.0  — perfectly uniform distribution across all cells (best)

M2 / M3 Input Contract
-----------------------
    source_points    : np.ndarray (N, 2) float32/float64  [x, y]
    reference_points : np.ndarray (N, 2) float32/float64  [x, y]
    inlier_mask      : np.ndarray (N,) bool from GeometricEstimator

Output
------
InlierReport — dataclass with:
    inlier_mask              : np.ndarray (N,) bool
    inlier_source_points     : np.ndarray (M, 2)  — inlier source coords
    inlier_reference_points  : np.ndarray (M, 2)  — inlier reference coords
    outlier_source_points    : np.ndarray (K, 2)  — outlier source coords
    outlier_reference_points : np.ndarray (K, 2)  — outlier reference coords
    inlier_count             : int  (M)
    outlier_count            : int  (K)
    total_count              : int  (N = M + K)
    inlier_ratio             : float ∈ [0, 1]
    spatial_coverage         : float ∈ [0, 1]
    diagnostics              : Dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENTROPY_EPS: float = 1e-10          # prevents log(0) in entropy calculation
_DEFAULT_GRID_DIVISIONS: int = 4      # 4×4 = 16 cells by default


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class InlierReport:
    """Structured inlier / outlier analysis output.

    Attributes:
        inlier_mask: Original (N,) bool mask from RANSAC.
        inlier_source_points: (M, 2) source coordinates for inlier matches.
        inlier_reference_points: (M, 2) reference coordinates for inlier matches.
        outlier_source_points: (K, 2) source coordinates for outlier matches.
        outlier_reference_points: (K, 2) reference coordinates for outlier matches.
        inlier_count: M — number of RANSAC inliers.
        outlier_count: K — number of RANSAC outliers.
        total_count: N = M + K — total correspondence pairs analysed.
        inlier_ratio: M / N  ∈ [0, 1].  0.0 when N=0.
        spatial_coverage: Grid-occupancy entropy of inlier source points
            normalised to [0, 1].  1.0 = perfectly uniform; 0.0 = all
            points in one cell.  0.0 when no inliers.
        diagnostics: Additional metadata (grid_divisions, bounding box, etc.).

        # Legacy aliases (backward-compatible with Phase 1 stub)
        residual_stats: Empty dict by default.  Reserved for Phase 4.
    """

    inlier_mask: np.ndarray

    # Separated point arrays
    inlier_source_points: np.ndarray
    inlier_reference_points: np.ndarray
    outlier_source_points: np.ndarray
    outlier_reference_points: np.ndarray

    # Scalar statistics
    inlier_count: int
    outlier_count: int
    total_count: int
    inlier_ratio: float
    spatial_coverage: float

    # Reserved for Phase 4 (not computed here)
    residual_stats: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------------
    # Backward-compat alias — Phase 1 stub used 'spatial_uniformity'
    # ---------------------------------------------------------------------------

    @property
    def spatial_uniformity(self) -> float:
        """Alias for spatial_coverage (backward-compat with Phase 1 scaffold)."""
        return self.spatial_coverage


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    inlier_mask: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce inputs to canonical dtypes.

    Args:
        inlier_mask: Input mask — will be coerced to (N,) bool.
        source_points: Source point array — must be (N, 2).
        reference_points: Reference point array — must be (N, 2).

    Returns:
        Tuple (mask_bool, src_f64, ref_f64) all coerced and validated.

    Raises:
        TypeError: If any input is not a numpy array.
        ValueError: If shapes are inconsistent or wrong dimensionality.
    """
    # Type checks
    for name, arr in [
        ("inlier_mask", inlier_mask),
        ("source_points", source_points),
        ("reference_points", reference_points),
    ]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"'{name}' must be a numpy ndarray, got {type(arr).__name__}."
            )

    # Coerce source / reference to float64
    src = np.asarray(source_points, dtype=np.float64)
    ref = np.asarray(reference_points, dtype=np.float64)
    mask = np.asarray(inlier_mask).ravel().astype(bool)

    # Shape checks for src / ref
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            f"source_points must be shape (N, 2), got {source_points.shape}."
        )
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError(
            f"reference_points must be shape (N, 2), got {reference_points.shape}."
        )

    n_src = src.shape[0]
    n_ref = ref.shape[0]
    n_mask = mask.shape[0]

    # Consistent N
    if n_src != n_ref:
        raise ValueError(
            f"source_points ({n_src}) and reference_points ({n_ref}) "
            "must have the same number of rows."
        )
    if n_mask != n_src:
        raise ValueError(
            f"inlier_mask length ({n_mask}) does not match "
            f"source_points length ({n_src})."
        )

    return mask, src, ref


def _compute_spatial_coverage(
    points: np.ndarray,
    grid_divisions: int,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute grid-occupancy normalised entropy as a spatial coverage score.

    Args:
        points: (M, 2) float64 [x, y] inlier coordinates.
        grid_divisions: G — number of cells per axis (G×G total cells).
        image_shape: Optional (H, W) for grid extent.  If None, uses the
            bounding box of the point set.

    Returns:
        Tuple (coverage_score, diag_dict).
        coverage_score ∈ [0, 1].  diag_dict contains bbox, cell_counts, etc.
    """
    m = points.shape[0]
    g = max(1, grid_divisions)
    n_cells = g * g

    # Degenerate: no points → 0 coverage
    if m == 0:
        return 0.0, {"grid_divisions": g, "n_cells": n_cells, "n_points": 0}

    # Single point → minimum coverage
    if m == 1:
        return 0.0, {"grid_divisions": g, "n_cells": n_cells, "n_points": 1}

    # Determine grid extent
    if image_shape is not None:
        h, w = float(image_shape[0]), float(image_shape[1])
        x_min, x_max = 0.0, w
        y_min, y_max = 0.0, h
    else:
        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())

    # Guard against degenerate (all identical) bounding box
    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range < 1e-9:
        x_range = 1.0
        x_min -= 0.5
    if y_range < 1e-9:
        y_range = 1.0
        y_min -= 0.5

    # Bin points into G×G grid using digitize
    x_bins = np.linspace(x_min, x_max, g + 1)
    y_bins = np.linspace(y_min, y_max, g + 1)

    # np.digitize returns 1-indexed; clip to [0, g-1]
    col_idx = np.clip(np.digitize(points[:, 0], x_bins) - 1, 0, g - 1)
    row_idx = np.clip(np.digitize(points[:, 1], y_bins) - 1, 0, g - 1)
    cell_idx = row_idx * g + col_idx  # linear cell index in [0, G²-1]

    # Count occupancy per cell
    counts = np.bincount(cell_idx, minlength=n_cells).astype(np.float64)

    # Normalised probability distribution
    p = counts / counts.sum()

    # Shannon entropy H(p) = -Σ p * log(p)  (ignoring zero-probability cells)
    entropy = -float(np.sum(p * np.log(p + _ENTROPY_EPS)))

    # Maximum possible entropy: uniform distribution → log(n_cells)
    max_entropy = float(np.log(n_cells))

    if max_entropy < _ENTROPY_EPS:
        # Only 1 cell (g=1) → trivially "covered"
        coverage = 1.0
    else:
        coverage = float(np.clip(entropy / max_entropy, 0.0, 1.0))

    diag = {
        "grid_divisions": g,
        "n_cells": n_cells,
        "n_points": m,
        "occupied_cells": int((counts > 0).sum()),
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "entropy": entropy,
        "max_entropy": max_entropy,
    }
    return coverage, diag


def _empty_point_array() -> np.ndarray:
    """Return an empty (0, 2) float64 array."""
    return np.empty((0, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------


class InlierAnalyzer:
    """Inlier / outlier separation and spatial analysis for M3 Phase 3.

    Usage::

        analyser = InlierAnalyzer(grid_divisions=4)
        report = analyser.analyse(
            inlier_mask=estimation_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
        )

        # Access separated arrays
        inlier_src = report.inlier_source_points   # (M, 2)
        inlier_ref = report.inlier_reference_points
        outlier_src = report.outlier_source_points  # (K, 2)

        # Scalar stats
        print(report.inlier_count, report.inlier_ratio, report.spatial_coverage)

    Args:
        grid_divisions: G — grid cells per axis for spatial coverage.
            A G×G grid is used; default is 4 (16 cells).
    """

    def __init__(self, grid_divisions: int = _DEFAULT_GRID_DIVISIONS) -> None:
        if grid_divisions < 1:
            raise ValueError(f"grid_divisions must be ≥ 1, got {grid_divisions}.")
        self.grid_divisions = int(grid_divisions)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(
        self,
        inlier_mask: np.ndarray,
        source_points: np.ndarray,
        reference_points: np.ndarray,
        transform_matrix: Optional[np.ndarray] = None,   # reserved for Phase 4
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> InlierReport:
        """Perform inlier / outlier separation and spatial coverage analysis.

        Args:
            inlier_mask: (N,) bool array from RANSAC (GeometricEstimator).
            source_points: (N, 2) float source coordinates from M2.
            reference_points: (N, 2) float reference coordinates from M2.
            transform_matrix: Not used in Phase 3.  Accepted for API
                compatibility with later phases; ignored here.
            image_shape: Optional (H, W).  If provided, spatial coverage is
                computed over the full image extent rather than the point
                bounding box.

        Returns:
            InlierReport with all inlier/outlier arrays, scalar statistics,
            and spatial coverage.

        Raises:
            TypeError: If any array argument is not a numpy ndarray.
            ValueError: If shapes are inconsistent.
        """
        # ---- 1. Validate & coerce inputs ------------------------------------
        mask, src, ref = _validate_inputs(inlier_mask, source_points, reference_points)
        n_total = src.shape[0]

        # ---- 2. Handle empty input ------------------------------------------
        if n_total == 0:
            return InlierReport(
                inlier_mask=mask,
                inlier_source_points=_empty_point_array(),
                inlier_reference_points=_empty_point_array(),
                outlier_source_points=_empty_point_array(),
                outlier_reference_points=_empty_point_array(),
                inlier_count=0,
                outlier_count=0,
                total_count=0,
                inlier_ratio=0.0,
                spatial_coverage=0.0,
                diagnostics={"note": "empty_input"},
            )

        # ---- 3. Separate inliers and outliers --------------------------------
        outlier_mask = ~mask

        inlier_src = src[mask]
        inlier_ref = ref[mask]
        outlier_src = src[outlier_mask]
        outlier_ref = ref[outlier_mask]

        inlier_count = int(mask.sum())
        outlier_count = int(outlier_mask.sum())
        inlier_ratio = inlier_count / n_total

        # ---- 4. Spatial coverage of inlier source points --------------------
        coverage, cov_diag = _compute_spatial_coverage(
            inlier_src, self.grid_divisions, image_shape
        )

        diag: Dict[str, Any] = {
            "grid_divisions": self.grid_divisions,
            "total_count": n_total,
            **cov_diag,
        }

        return InlierReport(
            inlier_mask=mask,
            inlier_source_points=inlier_src,
            inlier_reference_points=inlier_ref,
            outlier_source_points=outlier_src,
            outlier_reference_points=outlier_ref,
            inlier_count=inlier_count,
            outlier_count=outlier_count,
            total_count=n_total,
            inlier_ratio=inlier_ratio,
            spatial_coverage=coverage,
            diagnostics=diag,
        )


# ---------------------------------------------------------------------------
# Standalone function interface
# ---------------------------------------------------------------------------


def analyse_inliers(
    inlier_mask: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    transform_matrix: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    grid_divisions: int = _DEFAULT_GRID_DIVISIONS,
) -> InlierReport:
    """Analyse inlier / outlier separation from M3 Phase 2 estimation output.

    Convenience wrapper around InlierAnalyzer.analyse().  Preferred entry
    point for one-shot usage without caching.

    Args:
        inlier_mask: (N,) bool inlier mask from GeometricEstimator.
        source_points: (N, 2) float source point coordinates from M2.
        reference_points: (N, 2) float reference point coordinates from M2.
        transform_matrix: Not used in Phase 3.  Accepted for API compatibility.
        image_shape: Optional (H, W) for grid extent normalisation.
        grid_divisions: G for G×G spatial grid (default 4).

    Returns:
        InlierReport with separated point arrays and scalar statistics.

    Raises:
        TypeError: If any array argument is not a numpy ndarray.
        ValueError: If shapes are inconsistent.
    """
    analyser = InlierAnalyzer(grid_divisions=grid_divisions)
    return analyser.analyse(
        inlier_mask=inlier_mask,
        source_points=source_points,
        reference_points=reference_points,
        transform_matrix=transform_matrix,
        image_shape=image_shape,
    )
