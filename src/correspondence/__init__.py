"""Fine Correspondence package for Project Vikram / FLUX.

Provides classical SIFT feature matching baseline with Lowe's ratio test,
ROI coordinate mapping, and interfaces for experimental deep correspondence models.
"""

from src.correspondence.feature_matching import (
    evaluate_geometric_ransac,
    match_sift_features,
    match_with_candidate_roi,
    select_best_candidate_correspondence,
)
from src.correspondence.lightglue import LightGlueMatcher
from src.correspondence.loftr import LoFTRMatcher
from src.correspondence.superpoint import SuperPointMatcher

__all__ = [
    "match_sift_features",
    "match_with_candidate_roi",
    "evaluate_geometric_ransac",
    "select_best_candidate_correspondence",
    "SuperPointMatcher",
    "LoFTRMatcher",
    "LightGlueMatcher",
]
