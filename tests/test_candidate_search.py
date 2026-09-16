"""Unit tests for M2 Terrain-Aware Candidate Search module."""

import unittest
import numpy as np
import cv2

from src.candidate_search.coarse_localization import (
    find_coarse_candidates,
    extract_search_representation,
)
from src.candidate_search.terrain_filter import (
    compute_iou,
    apply_nms,
    filter_candidate_regions,
)
from src.candidate_search.candidate_generator import (
    generate_candidate_rois,
    extract_candidate_crop,
)
from src.structure.structure_representation import extract_structure_representation


class TestCandidateSearch(unittest.TestCase):
    """Test suite for candidate search and filtering."""

    def setUp(self):
        # Create a 256x256 reference mosaic with distinct craters
        self.ref_img = np.full((256, 256), 100, dtype=np.uint8)
        cv2.circle(self.ref_img, (80, 80), 20, 230, thickness=2)
        cv2.circle(self.ref_img, (78, 78), 16, 30, thickness=-1)

        cv2.circle(self.ref_img, (180, 180), 25, 240, thickness=3)
        cv2.circle(self.ref_img, (178, 178), 20, 20, thickness=-1)

        # Create source image as a crop of the crater at (80, 80)
        self.src_img = self.ref_img[50:114, 50:114].copy()

    def test_compute_iou(self):
        box1 = (10, 10, 50, 50)
        box2 = (10, 10, 50, 50)
        self.assertAlmostEqual(compute_iou(box1, box2), 1.0)

        box3 = (100, 100, 50, 50)
        self.assertAlmostEqual(compute_iou(box1, box3), 0.0)

        box4 = (35, 10, 50, 50)
        iou = compute_iou(box1, box4)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_apply_nms(self):
        cands = [
            {"bbox": (10, 10, 50, 50), "score": 0.9},
            {"bbox": (12, 12, 50, 50), "score": 0.85},  # Highly overlapping with first
            {"bbox": (100, 100, 50, 50), "score": 0.7}, # Non-overlapping
        ]
        kept = apply_nms(cands, iou_threshold=0.4, max_keep=5)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["score"], 0.9)
        self.assertEqual(kept[1]["score"], 0.7)

    def test_find_coarse_candidates(self):
        cands = find_coarse_candidates(
            self.src_img,
            self.ref_img,
            scales=(1.0,),
            top_k_per_scale=3,
        )
        self.assertGreater(len(cands), 0)
        best = max(cands, key=lambda c: c["score"])
        # The crop was taken at (50, 50)
        bx, by, bw, bh = best["bbox"]
        self.assertAlmostEqual(bx, 50, delta=10)
        self.assertAlmostEqual(by, 50, delta=10)

    def test_generate_candidate_rois(self):
        result = generate_candidate_rois(
            self.src_img,
            self.ref_img,
            scales=(0.8, 1.0, 1.2),
            top_k_per_scale=3,
            min_score=0.2,
            max_candidates=3,
        )
        self.assertIn("candidates", result)
        self.assertIn("best_candidate", result)
        self.assertIsNotNone(result["best_candidate"])
        self.assertGreater(result["num_candidates_retained"], 0)

        # Test crop extraction
        crop = extract_candidate_crop(self.ref_img, result["best_candidate"])
        self.assertIsInstance(crop, np.ndarray)
        self.assertGreater(crop.size, 0)


if __name__ == "__main__":
    unittest.main()
