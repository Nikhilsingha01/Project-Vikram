"""GSD-Aware Metadata Validation and Physical Scale Calculation for FLUX.

Provides structured metadata extraction, validation of Ground Sample Distance (GSD),
physical scale ratio computation, and scale recommendations for terrain-aware search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image Metadata Structure
# ---------------------------------------------------------------------------


@dataclass
class ImageMetadata:
    """Metadata container for satellite/lunar imagery.

    Attributes:
        image_path: Optional path to the image file.
        width: Image width in pixels.
        height: Image height in pixels.
        gsd: Ground Sample Distance in meters/pixel (must be > 0).
        coordinate_system: Coordinate Reference System string (e.g. 'Moon 2000').
        projection: Map projection (e.g. 'Polar Stereographic').
        sensor_name: Name of sensor/instrument (e.g. 'Chandrayaan-2 OHRC', 'LROC WAC').
        bounds: Optional geographic or projected bounding box.
        is_available: True if metadata was successfully retrieved/specified.
    """

    image_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    gsd: Optional[float] = None
    coordinate_system: Optional[str] = None
    projection: Optional[str] = None
    sensor_name: Optional[str] = None
    bounds: Optional[Dict[str, Any]] = None
    is_available: bool = True

    def has_valid_gsd(self) -> bool:
        """Check if GSD is present, positive, finite, and non-NaN."""
        if self.gsd is None:
            return False
        try:
            val = float(self.gsd)
            return val > 0.0 and math.isfinite(val) and not math.isnan(val)
        except (ValueError, TypeError):
            return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImageMetadata:
        """Construct ImageMetadata safely from a dictionary."""
        if not data:
            return cls(is_available=False)

        raw_gsd = data.get("gsd", data.get("pixel_resolution", data.get("resolution")))
        parsed_gsd: Optional[float] = None
        if raw_gsd is not None:
            try:
                # Handle string like '0.23 m/pixel' or numerical float
                if isinstance(raw_gsd, str):
                    clean_str = raw_gsd.lower().replace("m/pixel", "").replace("m/px", "").replace("m", "").strip()
                    parsed_gsd = float(clean_str)
                else:
                    parsed_gsd = float(raw_gsd)
            except (ValueError, TypeError):
                parsed_gsd = None

        return cls(
            image_path=data.get("image_path"),
            width=int(data["width"]) if "width" in data and data["width"] is not None else None,
            height=int(data["height"]) if "height" in data and data["height"] is not None else None,
            gsd=parsed_gsd,
            coordinate_system=data.get("coordinate_system", data.get("srs")),
            projection=data.get("projection"),
            sensor_name=data.get("sensor_name", data.get("instrument")),
            bounds=data.get("bounds"),
            is_available=True,
        )


# ---------------------------------------------------------------------------
# GSD Validation Result Structure
# ---------------------------------------------------------------------------


@dataclass
class GSDValidationResult:
    """Structured result from GSD validation and physical scale ratio analysis.

    Attributes:
        source_gsd: Validated source Ground Sample Distance (m/px).
        reference_gsd: Validated reference Ground Sample Distance (m/px).
        scale_ratio: Physical scale factor (source_gsd / reference_gsd).
        is_valid: True if both GSDs are valid positive real numbers.
        is_physically_suitable: True if scale ratio is within realistic fine-matching bounds.
        is_extreme_scale_gap: True if scale ratio indicates extreme resolution mismatch (> 10x or < 0.1x).
        status: Categorical status string.
        warning_message: User-facing explanatory message for warnings (does not indicate pipeline error).
        recommended_scales: Optional sequence of candidate search scales guided by physical GSD.
        diagnostics: Per-signal metadata dictionary.
    """

    source_gsd: Optional[float] = None
    reference_gsd: Optional[float] = None
    scale_ratio: Optional[float] = None
    is_valid: bool = False
    is_physically_suitable: bool = False
    is_extreme_scale_gap: bool = False
    status: str = "MISSING_METADATA"
    warning_message: Optional[str] = None
    recommended_scales: Optional[Tuple[float, ...]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scale Calculation and Validation Logic
# ---------------------------------------------------------------------------


def calculate_physical_scale_ratio(
    source_gsd: Optional[Union[float, ImageMetadata]],
    reference_gsd: Optional[Union[float, ImageMetadata]],
    min_suitable_ratio: float = 0.2,
    max_suitable_ratio: float = 5.0,
    extreme_gap_threshold: float = 10.0,
) -> GSDValidationResult:
    """Calculate and validate the physical scale ratio between source and reference imagery.

    Formula:
        scale_ratio = source_gsd / reference_gsd

    Physical interpretation:
        - scale_ratio = 1.0 : Same spatial resolution (1 source pixel = 1 reference pixel).
        - scale_ratio = 0.5 : Source has 2x higher resolution (1 ref pixel = 2 source pixels).
        - scale_ratio = 0.00115 (e.g. 0.23m / 200m) : Source has 870x higher resolution.

    Args:
        source_gsd: Source GSD in m/px or ImageMetadata object.
        reference_gsd: Reference GSD in m/px or ImageMetadata object.
        min_suitable_ratio: Minimum scale ratio considered suitable for direct fine feature matching (default: 0.2).
        max_suitable_ratio: Maximum scale ratio considered suitable for direct fine feature matching (default: 5.0).
        extreme_gap_threshold: Threshold factor defining extreme resolution discrepancy (default: 10.0).

    Returns:
        GSDValidationResult containing validated scale ratio, suitability flags, and diagnostics.
    """
    # Extract numerical GSDs if ImageMetadata instances were provided
    src_val: Optional[float] = None
    if isinstance(source_gsd, ImageMetadata):
        src_val = source_gsd.gsd if source_gsd.has_valid_gsd() else None
    elif source_gsd is not None:
        try:
            src_val = float(source_gsd)
        except (ValueError, TypeError):
            src_val = None

    ref_val: Optional[float] = None
    if isinstance(reference_gsd, ImageMetadata):
        ref_val = reference_gsd.gsd if reference_gsd.has_valid_gsd() else None
    elif reference_gsd is not None:
        try:
            ref_val = float(reference_gsd)
        except (ValueError, TypeError):
            ref_val = None

    # Handle missing metadata safely
    if src_val is None or ref_val is None:
        missing = []
        if src_val is None:
            missing.append("source_gsd")
        if ref_val is None:
            missing.append("reference_gsd")
        logger.info("GSD metadata incomplete: %s not provided. Skipping physical scale validation.", missing)
        return GSDValidationResult(
            source_gsd=src_val,
            reference_gsd=ref_val,
            scale_ratio=None,
            is_valid=False,
            is_physically_suitable=False,
            is_extreme_scale_gap=False,
            status="MISSING_METADATA",
            warning_message=f"GSD metadata unavailable for: {', '.join(missing)}.",
            recommended_scales=None,
            diagnostics={"missing_fields": missing},
        )

    # Validate strictly positive real numbers
    if src_val <= 0.0 or ref_val <= 0.0 or not math.isfinite(src_val) or not math.isfinite(ref_val):
        logger.warning(
            "Invalid non-positive or non-finite GSD values encountered: source_gsd=%s, reference_gsd=%s",
            src_val,
            ref_val,
        )
        return GSDValidationResult(
            source_gsd=src_val,
            reference_gsd=ref_val,
            scale_ratio=None,
            is_valid=False,
            is_physically_suitable=False,
            is_extreme_scale_gap=False,
            status="INVALID_GSD",
            warning_message="Invalid GSD: Ground Sample Distance must be a strictly positive real number.",
            recommended_scales=None,
            diagnostics={"source_gsd": src_val, "reference_gsd": ref_val},
        )

    # Calculate physical scale ratio
    scale_ratio = float(src_val / ref_val)
    is_suitable = bool(min_suitable_ratio <= scale_ratio <= max_suitable_ratio)
    is_extreme = bool(
        scale_ratio < (1.0 / extreme_gap_threshold) or scale_ratio > extreme_gap_threshold
    )

    recommended_scales: Optional[Tuple[float, ...]] = None
    warning_msg: Optional[str] = None
    status: str

    if is_extreme:
        status = "VALID_EXTREME_GAP"
        gap_factor = (1.0 / scale_ratio) if scale_ratio < 1.0 else scale_ratio
        warning_msg = (
            f"Extreme GSD disparity detected: Source GSD ({src_val:.4f} m/px) vs "
            f"Reference GSD ({ref_val:.4f} m/px) yields physical scale ratio {scale_ratio:.4e} "
            f"({gap_factor:.1f}x resolution gap). Direct SIFT feature correspondence across this gap "
            f"is physically unfeasible without intermediate multi-resolution scale bridging."
        )
        logger.warning(warning_msg)
    elif not is_suitable:
        status = "VALID_LARGE_GAP"
        gap_factor = (1.0 / scale_ratio) if scale_ratio < 1.0 else scale_ratio
        warning_msg = (
            f"Moderate GSD disparity detected: physical scale ratio {scale_ratio:.4f} "
            f"({gap_factor:.1f}x resolution gap). Multi-scale candidate search recommended."
        )
        logger.info(warning_msg)
        # Suggest scales centered around physical scale ratio
        recommended_scales = (
            round(scale_ratio * 0.5, 4),
            round(scale_ratio * 0.75, 4),
            round(scale_ratio * 1.0, 4),
            round(scale_ratio * 1.25, 4),
            round(scale_ratio * 1.5, 4),
        )
    else:
        status = "VALID_NORMAL"
        logger.info(
            "Physical GSD validation passed: source=%.4f m/px, reference=%.4f m/px, ratio=%.4f (suitable)",
            src_val,
            ref_val,
            scale_ratio,
        )
        recommended_scales = (
            round(scale_ratio * 0.67, 4),
            round(scale_ratio * 0.8, 4),
            round(scale_ratio * 1.0, 4),
            round(scale_ratio * 1.25, 4),
            round(scale_ratio * 1.5, 4),
        )

    return GSDValidationResult(
        source_gsd=src_val,
        reference_gsd=ref_val,
        scale_ratio=scale_ratio,
        is_valid=True,
        is_physically_suitable=is_suitable,
        is_extreme_scale_gap=is_extreme,
        status=status,
        warning_message=warning_msg,
        recommended_scales=recommended_scales,
        diagnostics={
            "source_gsd_m": src_val,
            "reference_gsd_m": ref_val,
            "scale_ratio": scale_ratio,
            "gap_factor": (1.0 / scale_ratio) if scale_ratio < 1.0 else scale_ratio,
            "min_suitable_ratio": min_suitable_ratio,
            "max_suitable_ratio": max_suitable_ratio,
        },
    )
