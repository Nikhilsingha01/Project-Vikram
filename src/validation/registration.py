"""Image registration (warping) for M3 Validation pipeline.

This module is the fifth stage of the M3 pipeline.  It applies the validated
geometric transform — produced by GeometricEstimator (Phase 2) and confirmed
by InlierAnalyzer (Phase 3) and ReprojectionReport (Phase 4) — to warp a
source image into the reference coordinate frame.

Warp Strategy
-------------
- Homography (3×3) : ``cv2.warpPerspective``
- Affine / Similarity / Translation (2×3) : ``cv2.warpAffine``

Border handling: ``BORDER_CONSTANT`` with configurable fill value (default 0).
Interpolation: ``INTER_LINEAR`` by default; overridable via ``RegistrationConfig``.

Overlap Detection
-----------------
After warping, the valid-pixel bounding box (the region that contains warped
source pixels, i.e. pixel value != fill_value) is computed via ``cv2.boundingRect``
on the non-fill pixel mask.

Geographic Localisation (Optional)
-----------------------------------
If a reference ground sampling distance (GSD) is provided, the centre of the
source image is projected through the transform to give its location in the
reference pixel frame, and then converted to a metric offset.

M3 Input Contract
-----------------
    source_image     : np.ndarray (H, W) or (H, W, C) — source image
    transform_matrix : np.ndarray (3,3) homography or (2,3) affine
    reference_shape  : Tuple[int, int] (H_ref, W_ref)
    config           : RegistrationConfig (optional)
    reference_gsd    : float (optional) — metres per pixel

Output
------
RegistrationResult — dataclass with:
    registered_image    : np.ndarray — warped source in reference frame
    transform_matrix    : np.ndarray — the applied transform
    output_size         : Tuple[int, int] — (W, H) of output canvas
    overlap_bbox        : Optional[(x, y, w, h)] — valid-pixel bounding box
    estimated_location  : Optional[Dict] — geographic metadata
    quality_flags       : Dict[str, bool]
    diagnostics         : Dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AFFINE_MODELS = frozenset({"affine", "similarity", "translation"})
_HOMOGRAPHY_MODELS = frozenset({"homography"})
_ALL_MODELS = _AFFINE_MODELS | _HOMOGRAPHY_MODELS

# Minimum fraction of output canvas that must contain valid (non-fill) pixels
# for the 'overlap_sufficient' quality flag to be True.
_MIN_OVERLAP_RATIO: float = 0.01


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegistrationConfig:
    """Configuration for the image registration (warping) step.

    Attributes:
        interpolation: OpenCV interpolation flag.  Default cv2.INTER_LINEAR (1).
            Use cv2.INTER_LANCZOS4 (4) for highest quality.
        border_mode: OpenCV border extrapolation mode.  Default BORDER_CONSTANT (0).
        border_value: Fill value for pixels outside the source image boundary.
        model_type: Transform model type — 'homography', 'affine',
            'similarity', or 'translation'.  Determines warp function.
        output_size: Optional (W, H) canvas size.  If None, uses reference
            image dimensions.
        compute_overlap: If True, compute the valid-pixel overlap bounding box.
    """

    interpolation: int = 1          # cv2.INTER_LINEAR
    border_mode: int = 0            # cv2.BORDER_CONSTANT
    border_value: int = 0
    model_type: str = "homography"
    output_size: Optional[Tuple[int, int]] = None   # (W, H)
    compute_overlap: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Structured output from the registration (warping) step.

    Attributes:
        registered_image: Source image warped into the reference coordinate
            frame.  Same dtype as input source_image.  Shape (H_ref, W_ref)
            or (H_ref, W_ref, C) for colour inputs.
        transform_matrix: The exact transform matrix that was applied.
        output_size: (W, H) dimensions of the output canvas.
        overlap_bbox: (x, y, w, h) bounding box of valid (non-fill) pixels
            in the warped image.  None if compute_overlap was False or the
            warped image is all fill.
        estimated_location: Optional geographic metadata dict containing
            'reference_pixel_x', 'reference_pixel_y',
            'estimated_x_offset_m', 'estimated_y_offset_m', 'gsd_m_per_px'.
            None if reference_gsd was not provided.
        quality_flags: Sanity-check boolean flags:
            - 'warp_succeeded'     : warpPerspective/warpAffine ran without error
            - 'overlap_nonzero'    : at least one valid pixel in warped output
            - 'overlap_sufficient' : valid-pixel area > MIN_OVERLAP_RATIO of canvas
        diagnostics: Additional metadata (canvas size, fill ratio, model_type, …)
    """

    registered_image: np.ndarray
    transform_matrix: np.ndarray
    output_size: Tuple[int, int]           # (W, H)
    overlap_bbox: Optional[Tuple[int, int, int, int]]
    estimated_location: Optional[Dict[str, Any]]
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    source_image: np.ndarray,
    transform_matrix: np.ndarray,
    reference_shape: Tuple[int, int],
    model_type: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Validate and coerce warp inputs.

    Returns:
        (source_image_validated, transform_f64, model_type_lower)

    Raises:
        TypeError: If source_image or transform_matrix is not ndarray.
        ValueError: If source_image is not 2-D or 3-D, transform has wrong
            shape for the model, reference_shape is invalid, or model unknown.
    """
    if not isinstance(source_image, np.ndarray):
        raise TypeError(
            f"source_image must be numpy ndarray, got {type(source_image).__name__}."
        )
    if not isinstance(transform_matrix, np.ndarray):
        raise TypeError(
            f"transform_matrix must be numpy ndarray, got {type(transform_matrix).__name__}."
        )

    if source_image.ndim not in (2, 3):
        raise ValueError(
            f"source_image must be 2-D (H,W) or 3-D (H,W,C), "
            f"got ndim={source_image.ndim}."
        )

    mt = model_type.lower()
    if mt not in _ALL_MODELS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: {sorted(_ALL_MODELS)}."
        )

    M = np.asarray(transform_matrix, dtype=np.float64)

    if mt in _HOMOGRAPHY_MODELS:
        if M.shape != (3, 3):
            raise ValueError(
                f"Homography transform_matrix must be (3, 3), got {M.shape}."
            )
    else:
        if M.shape != (2, 3):
            raise ValueError(
                f"Affine/similarity/translation transform_matrix must be "
                f"(2, 3), got {M.shape}."
            )

    if not np.isfinite(M).all():
        raise ValueError("transform_matrix contains NaN or Inf values.")

    if len(reference_shape) < 2 or reference_shape[0] <= 0 or reference_shape[1] <= 0:
        raise ValueError(
            f"reference_shape must be (H, W) with positive integers, "
            f"got {reference_shape}."
        )

    return source_image, M, mt


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_overlap_bbox(
    warped_image: np.ndarray,
    fill_value: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of valid (non-fill) pixels in a warped image.

    Args:
        warped_image: 2-D or 3-D warped output image.
        fill_value: Border fill value used during warping.

    Returns:
        (x, y, w, h) bounding box tuple, or None if no valid pixels exist.
    """
    # Collapse colour channels: a pixel is 'valid' if ANY channel != fill
    if warped_image.ndim == 3:
        mask = (warped_image != fill_value).any(axis=2).astype(np.uint8)
    else:
        mask = (warped_image != fill_value).astype(np.uint8)

    if mask.sum() == 0:
        return None

    # cv2.boundingRect expects a single-channel binary image
    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, w, h)


def _estimate_geographic_location(
    transform_matrix: np.ndarray,
    source_shape: Tuple[int, int],
    reference_gsd: float,
    model_type: str = "homography",
) -> Dict[str, Any]:
    """Estimate the geographic offset of the source image centre in the reference frame.

    Projects the centre point of the source image through the transform to
    get its location in reference pixel coordinates, then converts the pixel
    offset to metres using reference_gsd.

    Args:
        transform_matrix: (3,3) homography or (2,3) affine.
        source_shape: (H_src, W_src).
        reference_gsd: Ground sampling distance in metres/pixel.
        model_type: Lowercase model string.

    Returns:
        Dict with keys: 'reference_pixel_x', 'reference_pixel_y',
        'estimated_x_offset_m', 'estimated_y_offset_m', 'gsd_m_per_px'.
    """
    h_src, w_src = source_shape
    cx = w_src / 2.0
    cy = h_src / 2.0

    mt = model_type.lower()
    pt = np.array([cx, cy, 1.0], dtype=np.float64)

    if mt in _HOMOGRAPHY_MODELS:
        projected = transform_matrix @ pt
        w = projected[2]
        if abs(w) < 1e-9:
            ref_px, ref_py = 0.0, 0.0
        else:
            ref_px = projected[0] / w
            ref_py = projected[1] / w
    else:
        # (2, 3) affine
        M33 = np.eye(3, dtype=np.float64)
        M33[:2, :] = transform_matrix
        projected = M33 @ pt
        ref_px = projected[0]
        ref_py = projected[1]

    return {
        "reference_pixel_x": float(ref_px),
        "reference_pixel_y": float(ref_py),
        "estimated_x_offset_m": float(ref_px * reference_gsd),
        "estimated_y_offset_m": float(ref_py * reference_gsd),
        "gsd_m_per_px": float(reference_gsd),
    }


# ---------------------------------------------------------------------------
# Main registration function
# ---------------------------------------------------------------------------


def register_image(
    source_image: np.ndarray,
    transform_matrix: np.ndarray,
    reference_shape: Tuple[int, int],
    config: Optional[RegistrationConfig] = None,
    reference_gsd: Optional[float] = None,
) -> RegistrationResult:
    """Warp source_image into the reference coordinate frame.

    Applies either ``cv2.warpPerspective`` (homography) or ``cv2.warpAffine``
    (affine / similarity / translation) depending on ``config.model_type`` and
    the shape of ``transform_matrix``.

    Args:
        source_image: 2-D (H_src, W_src) or 3-D (H_src, W_src, C) image.
            Any numpy dtype is accepted (uint8, float32, etc.).
        transform_matrix: Validated transform from GeometricEstimator.
            Shape (3, 3) for homography, (2, 3) for affine/similarity/translation.
        reference_shape: (H_ref, W_ref) of the reference image.  Used as the
            output canvas size unless ``config.output_size`` overrides it.
        config: RegistrationConfig instance.  Defaults to RegistrationConfig()
            (homography model, INTER_LINEAR, BORDER_CONSTANT fill=0).
        reference_gsd: Optional ground sampling distance in metres per pixel.
            When provided, geographic offset metadata is computed.

    Returns:
        RegistrationResult with:
            - registered_image : warped source in reference frame
            - output_size      : (W, H) of the canvas
            - overlap_bbox     : (x, y, w, h) or None
            - estimated_location : geographic dict or None
            - quality_flags    : dict of sanity booleans
            - diagnostics      : additional metadata dict

    Raises:
        TypeError: If source_image or transform_matrix is not ndarray.
        ValueError: If source_image is not 2-D/3-D, transform has wrong shape,
            reference_shape is invalid, or model_type is unknown.
    """
    if config is None:
        config = RegistrationConfig()

    # ---- 1. Validate inputs -------------------------------------------------
    img, M, mt = _validate_inputs(
        source_image, transform_matrix, reference_shape, config.model_type
    )

    # ---- 2. Determine output canvas size ------------------------------------
    if config.output_size is not None:
        out_w, out_h = int(config.output_size[0]), int(config.output_size[1])
    else:
        out_h, out_w = int(reference_shape[0]), int(reference_shape[1])

    dsize = (out_w, out_h)  # cv2 convention: (width, height)

    # ---- 3. Warp -----------------------------------------------------------
    warp_succeeded = False
    registered: Optional[np.ndarray] = None

    try:
        if mt in _HOMOGRAPHY_MODELS:
            # (3,3) perspective warp
            registered = cv2.warpPerspective(
                img, M,
                dsize=dsize,
                flags=config.interpolation,
                borderMode=config.border_mode,
                borderValue=config.border_value,
            )
        else:
            # (2,3) affine warp
            M23 = M.astype(np.float64)
            registered = cv2.warpAffine(
                img, M23,
                dsize=dsize,
                flags=config.interpolation,
                borderMode=config.border_mode,
                borderValue=config.border_value,
            )
        warp_succeeded = True
    except cv2.error as exc:
        # Return a black canvas on warp failure so callers can inspect quality_flags
        registered = np.full(
            (out_h, out_w) if img.ndim == 2 else (out_h, out_w, img.shape[2]),
            config.border_value,
            dtype=img.dtype,
        )
        warp_succeeded = False

    # ---- 4. Overlap bounding box -------------------------------------------
    overlap_bbox: Optional[Tuple[int, int, int, int]] = None
    if config.compute_overlap and warp_succeeded:
        overlap_bbox = _compute_overlap_bbox(registered, config.border_value)

    # ---- 5. Quality flags --------------------------------------------------
    canvas_area = out_w * out_h
    overlap_area = (overlap_bbox[2] * overlap_bbox[3]) if overlap_bbox is not None else 0
    overlap_ratio = overlap_area / max(canvas_area, 1)

    quality_flags: Dict[str, bool] = {
        "warp_succeeded": warp_succeeded,
        "overlap_nonzero": overlap_area > 0,
        "overlap_sufficient": overlap_ratio >= _MIN_OVERLAP_RATIO,
    }

    # ---- 6. Geographic localisation (optional) -----------------------------
    estimated_location: Optional[Dict[str, Any]] = None
    if reference_gsd is not None and warp_succeeded:
        try:
            estimated_location = _estimate_geographic_location(
                M, img.shape[:2], reference_gsd, mt
            )
        except Exception:
            estimated_location = None

    # ---- 7. Diagnostics ----------------------------------------------------
    fill_count = int((registered == config.border_value).sum()
                     if registered.ndim == 2
                     else (registered == config.border_value).all(axis=2).sum())
    total_pixels = out_w * out_h

    diagnostics: Dict[str, Any] = {
        "model_type": mt,
        "source_shape": tuple(img.shape),
        "output_size_wh": dsize,
        "output_canvas_pixels": total_pixels,
        "fill_pixels": fill_count,
        "fill_ratio": fill_count / max(total_pixels, 1),
        "overlap_area_px": overlap_area,
        "overlap_ratio": overlap_ratio,
    }

    return RegistrationResult(
        registered_image=registered,
        transform_matrix=M,
        output_size=dsize,
        overlap_bbox=overlap_bbox,
        estimated_location=estimated_location,
        quality_flags=quality_flags,
        diagnostics=diagnostics,
    )
