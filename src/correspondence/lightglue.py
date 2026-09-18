"""LightGlue (Local Feature Matching at Light Speed) interface (experimental placeholder).

LightGlue matches sparse keypoints with neural network attention and adaptive stopping.
This module defines the interface for future integration in experimental phases.
"""

from typing import Any, Dict, Optional
import numpy as np


class LightGlueMatcher:
    """LightGlue feature matcher wrapper."""

    def __init__(self, weights_path: Optional[str] = None, feature_type: str = "superpoint"):
        """Initialize LightGlue matcher interface.

        Args:
            weights_path: Path to pretrained LightGlue weights.
            feature_type: Type of upstream features ('superpoint', 'sift', 'disk').
        """
        self.weights_path = weights_path
        self.feature_type = feature_type
        self.is_available = False

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform LightGlue sparse feature matching.

        Args:
            source_image: Source lunar image.
            reference_image: Reference lunar image.

        Returns:
            Dict[str, Any]: Structured correspondence record.
        """
        raise NotImplementedError(
            "LightGlue is an experimental deep learning module reserved for Phase 5. "
            "Please use the classical SIFT baseline in `src/correspondence/feature_matching.py`."
        )
