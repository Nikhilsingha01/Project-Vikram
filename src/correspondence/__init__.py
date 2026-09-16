"""Fine Correspondence package for Project Vikram / FLUX.

Provides classical SIFT feature matching baseline with Lowe's ratio test,
ROI coordinate mapping, and interfaces for experimental deep correspondence models.
"""

from src.correspondence.feature_matching import (
    match_sift_features,
    match_with_candidate_roi,
)
from src.correspondence.lightglue import LightGlueMatcher
from src.correspondence.loftr import LoFTRMatcher
from src.correspondence.superpoint import SuperPointMatcher

__all__ = [
    "match_sift_features",
    "match_with_candidate_roi",
    "SuperPointMatcher",
    "LoFTRMatcher",
    "LightGlueMatcher",
]
