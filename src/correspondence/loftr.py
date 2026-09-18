"""LoFTR (Detector-Free Local Feature Matching with Transformers) interface (experimental placeholder).

LoFTR performs dense correspondence using self and cross-attention layers.
This module defines the interface for future integration in experimental phases.
"""

from typing import Any, Dict, Optional
import numpy as np


class LoFTRMatcher:
    """LoFTR detector-free feature matcher wrapper."""

    def __init__(self, weights_path: Optional[str] = None, match_threshold: float = 0.2):
        """Initialize LoFTR matcher interface.

        Args:
            weights_path: Path to pretrained LoFTR weights.
            match_threshold: Minimum confidence threshold for correspondence.
        """
        self.weights_path = weights_path
        self.match_threshold = match_threshold
        self.is_available = False

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform LoFTR dense feature matching.

        Args:
            source_image: Source lunar image.
            reference_image: Reference lunar image.

        Returns:
            Dict[str, Any]: Structured correspondence record.
        """
        raise NotImplementedError(
            "LoFTR is an experimental deep learning module reserved for Phase 5. "
            "Please use the classical SIFT baseline in `src/correspondence/feature_matching.py`."
        )
