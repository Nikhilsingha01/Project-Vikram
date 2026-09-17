"""Image registration and coordinate transformation for M3 Validation pipeline.

This module is the fifth stage of the M3 pipeline.  It applies the validated
geometric transform (from GeometricEstimator, confirmed by InlierAnalyzer and
ReprojectionReport) to produce a registered output image and associated
coordinate metadata.

Purpose
-------
After evidence gating, estimation, and reprojection analysis have all passed,
this module performs the final spatial warping operation:

1. Warp the source image into the reference coordinate frame using the
   estimated transform matrix.
2. Compute the registration bounding box (the overlap region between source
   and reference after warping).
3. Derive the estimated geographic location (pixel offsets → geographic
   coordinate estimate if reference GSD/origin are available).
4. Provide a RegistrationResult that bundles the warped image, quality
   metadata, and the transform for downstream reporting.

Transform Application
---------------------
For a homography H (3×3):
    cv2.warpPerspective(source_image, H, dsize=(ref_w, ref_h))

For an affine/similarity transform M (2×3):
    cv2.warpAffine(source_image, M, dsize=(ref_w, ref_h))

Border handling strategy: BORDER_CONSTANT with fill_value=0 (black fill).
Interpolation: INTER_LINEAR by default; INTER_LANCZOS4 for high-quality mode.

Output
------
RegistrationResult — dataclass containing:
    registered_image    : np.ndarray — warped source image in reference frame
    transform_matrix    : np.ndarray — the applied transform
    overlap_bbox        : Optional[Tuple[int, int, int, int]] — (x, y, w, h)
    estimated_location  : Optional[Dict]  — geographic metadata if GSD provided
    quality_flags       : Dict[str, bool] — sanity checks on warp output
    diagnostics         : Dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RegistrationConfig:
    """Configuration for the image registration (warping) step.

    Attributes:
        interpolation: OpenCV interpolation flag.  cv2.INTER_LINEAR by default.
            Use cv2.INTER_LANCZOS4 for highest quality.
        border_mode: OpenCV border extrapolation mode (BORDER_CONSTANT=0).
        border_value: Fill value for pixels outside the source image boundary.
        model_type: Transform model type used — 'homography', 'affine',
            'similarity', or 'translation'.  Determines warp function selection.
        output_size: Optional (width, height) output canvas size.  If None,
            defaults to reference image dimensions.
        compute_overlap: If True, compute the valid-pixel overlap bounding box
            after warping.
    """

    interpolation: int = 1          # cv2.INTER_LINEAR
    border_mode: int = 0            # cv2.BORDER_CONSTANT
    border_value: int = 0
    model_type: str = "homography"
    output_size: Optional[Tuple[int, int]] = None
    compute_overlap: bool = True


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    """Structured output from the registration (warping) step.

    Attributes:
        registered_image: Source image warped into the reference coordinate
            frame.  Same dtype as input source_image.
        transform_matrix: The applied transform matrix.
        overlap_bbox: (x, y, w, h) bounding box of valid (non-fill) pixels
            in the warped image.  None if compute_overlap was False.
        estimated_location: Optional geographic metadata dict.  Contains
            'estimated_x_offset', 'estimated_y_offset', 'gsd_m_per_px' if
            reference GSD was provided.
        quality_flags: Sanity check boolean flags:
            - 'warp_succeeded'    : True if warpPerspective/warpAffine ran.
            - 'overlap_nonzero'   : True if valid pixel area > 0.
            - 'overlap_sufficient': True if overlap area > min_overlap_ratio.
        diagnostics: Additional metadata (output canvas size, fill ratio, etc.)
    """

    registered_image: np.ndarray
    transform_matrix: np.ndarray
    overlap_bbox: Optional[Tuple[int, int, int, int]]
    estimated_location: Optional[Dict[str, Any]]
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


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

    Args:
        source_image: 2-D (H_src, W_src) uint8 grayscale source image.
        transform_matrix: Validated transform from GeometricEstimator.
            Shape (3, 3) for homography, (2, 3) for affine/similarity.
        reference_shape: (H_ref, W_ref) dimensions of the reference image,
            used as the output canvas size unless config.output_size is set.
        config: RegistrationConfig instance.  Defaults to RegistrationConfig().
        reference_gsd: Optional ground sampling distance in metres per pixel
            for the reference image.  Used to compute geographic location.

    Returns:
        RegistrationResult with warped image and metadata.

    Raises:
        ValueError: If source_image is not 2-D or transform_matrix has
            unexpected shape.
    """
    # TODO (M3-REG-01): Validate source_image is 2-D uint8.
    # TODO (M3-REG-02): Determine output canvas size from reference_shape
    #   or config.output_size.
    # TODO (M3-REG-03): Dispatch to cv2.warpPerspective (homography) or
    #   cv2.warpAffine (affine/similarity/translation) based on config.model_type.
    # TODO (M3-REG-04): Set quality_flags['warp_succeeded'] = True/False.
    # TODO (M3-REG-05): If config.compute_overlap, call _compute_overlap_bbox()
    #   on the warped image to find the valid-pixel bounding box.
    # TODO (M3-REG-06): If reference_gsd is provided, call
    #   _estimate_geographic_location() to populate estimated_location.
    # TODO (M3-REG-07): Build diagnostics dict with canvas size, fill ratio.
    # TODO (M3-REG-08): Return RegistrationResult.
    raise NotImplementedError(
        "register_image() is not yet implemented (M3 TODO)."
    )


# ---------------------------------------------------------------------------
# Private helper stubs
# ---------------------------------------------------------------------------


def _compute_overlap_bbox(
    warped_image: np.ndarray,
    fill_value: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box of valid (non-fill) pixels in a warped image.

    Args:
        warped_image: 2-D warped output image.
        fill_value: Border fill value used during warping.

    Returns:
        (x, y, w, h) bounding box, or None if no valid pixels exist.
    """
    # TODO (M3-REG-OVL-01): Create binary mask of pixels != fill_value.
    # TODO (M3-REG-OVL-02): Use cv2.boundingRect on non-zero pixel coordinates.
    # TODO (M3-REG-OVL-03): Return None if mask is empty.
    raise NotImplementedError("_compute_overlap_bbox() not yet implemented.")


def _estimate_geographic_location(
    transform_matrix: np.ndarray,
    source_shape: Tuple[int, int],
    reference_gsd: float,
    model_type: str = "homography",
) -> Dict[str, Any]:
    """Estimate the geographic offset of source image centre in reference frame.

    Projects the centre point of the source image through the transform to
    get its location in reference pixel coordinates, then converts to a
    geographic offset using reference_gsd.

    Args:
        transform_matrix: Estimated transform (3×3 or 2×3).
        source_shape: (H_src, W_src) of the source image.
        reference_gsd: Ground sampling distance in metres/pixel.
        model_type: Transform model type.

    Returns:
        Dict with keys:
            'reference_pixel_x'  : float
            'reference_pixel_y'  : float
            'estimated_x_offset_m': float  (relative offset in metres)
            'estimated_y_offset_m': float
            'gsd_m_per_px'       : float
    """
    # TODO (M3-REG-GEO-01): Compute source image centre (cx, cy).
    # TODO (M3-REG-GEO-02): Project centre through transform_matrix.
    # TODO (M3-REG-GEO-03): Convert projected pixel location to geographic
    #   offset using reference_gsd.
    raise NotImplementedError("_estimate_geographic_location() not yet implemented.")
