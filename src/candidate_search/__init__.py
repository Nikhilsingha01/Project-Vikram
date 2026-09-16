"""Terrain-Aware Candidate Search package for Project Vikram / FLUX.

Provides multi-scale coarse localization, structural terrain filtering,
and candidate Region of Interest (ROI) generation for lunar image registration.
"""

from src.candidate_search.candidate_generator import (
    extract_candidate_crop,
    generate_candidate_rois,
)
from src.candidate_search.coarse_localization import (
    extract_search_representation,
    find_coarse_candidates,
)
from src.candidate_search.terrain_filter import (
    apply_nms,
    compute_iou,
    filter_candidate_regions,
)

__all__ = [
    "extract_search_representation",
    "find_coarse_candidates",
    "compute_iou",
    "apply_nms",
    "filter_candidate_regions",
    "generate_candidate_rois",
    "extract_candidate_crop",
]
