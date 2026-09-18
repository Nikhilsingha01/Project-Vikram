"""Application-facing pipeline contract.

The demo implementation intentionally does not perform localization,
matching, or registration. It provides the result shape that the real
M1-to-M4 pipeline will later implement.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


PIPELINE_STAGES = (
    "Image Preparation",
    "Terrain / Structure Analysis",
    "Candidate Search",
    "Feature Matching",
    "Evidence Validation",
    "Geometric Registration",
    "Sub-pixel Refinement",
)


@dataclass
class PipelineResult:
    """Standard result exchanged between the app and a FLUX pipeline."""

    success: bool
    status: str
    source_image: Any = None
    reference_image: Any = None
    registered_image: Any = None
    confidence: Optional[float] = None
    matches: Optional[int] = None
    inliers: Optional[int] = None
    inlier_ratio: Optional[float] = None
    rmse: Optional[float] = None
    mean_error: Optional[float] = None
    max_error: Optional[float] = None
    failure_reason: Optional[str] = None
    processing_steps: Optional[List[Dict[str, str]]] = None
    demo_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a dictionary for UI and adapter use."""
        return asdict(self)


def run_flux(source_image: np.ndarray, reference_image: np.ndarray) -> Dict[str, Any]:
    """Run the current FLUX demo pipeline.

    This function is the stable UI boundary. It deliberately reports that
    scientific processing is unavailable rather than inventing measurements.
    Replace its implementation with the real adapter when M1-M4 are wired.
    """
    _validate_image(source_image, "source_image")
    _validate_image(reference_image, "reference_image")

    steps = [
        {
            "name": stage,
            "status": "completed" if index == 0 else "not_implemented",
            "detail": (
                "Images accepted by the demo pipeline."
                if index == 0
                else "Waiting for the corresponding FLUX module."
            ),
        }
        for index, stage in enumerate(PIPELINE_STAGES)
    ]
    return PipelineResult(
        success=True,
        status="rejected",
        source_image=source_image,
        reference_image=reference_image,
        failure_reason=(
            "Demo mode completed the UI flow only. "
            "No scientific match or registration was performed."
        ),
        processing_steps=steps,
        demo_mode=True,
    ).to_dict()


def _validate_image(image: np.ndarray, name: str) -> None:
    """Validate the minimal image contract required by the app layer."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError(f"{name} must be a non-empty NumPy image array.")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be a 2-D grayscale or 3-D color image.")
