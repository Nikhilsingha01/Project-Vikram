"""SuperPoint deep feature extractor interface (experimental placeholder).

SuperPoint is a self-supervised deep learning feature detector and descriptor.
This module defines the interface for future integration in experimental phases.
"""

from typing import Any, Dict, Optional
import numpy as np


class SuperPointMatcher:
    """SuperPoint feature detection and descriptor extraction wrapper."""

    def __init__(self, weights_path: Optional[str] = None, max_keypoints: int = 2048):
        """Initialize SuperPoint matcher interface.

        Args:
            weights_path: Path to pretrained SuperPoint weights.
            max_keypoints: Maximum number of keypoints to extract.
        """
        self.weights_path = weights_path
        self.max_keypoints = max_keypoints
        self.is_available = False

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform SuperPoint feature matching.

        Args:
            source_image: Source lunar image.
            reference_image: Reference lunar image.

        Returns:
            Dict[str, Any]: Structured correspondence record.
        """
        raise NotImplementedError(
            "SuperPoint is an experimental deep learning module reserved for Phase 5. "
            "Please use the classical SIFT baseline in `src/correspondence/feature_matching.py`."
        )
