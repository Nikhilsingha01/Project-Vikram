"""Unit tests for M2 Fine Correspondence module."""

import unittest
import numpy as np
import cv2

from src.correspondence.feature_matching import (
    evaluate_geometric_ransac,
    match_sift_features,
    match_with_candidate_roi,
    select_best_candidate_correspondence,
)
from src.correspondence.superpoint import SuperPointMatcher
from src.correspondence.loftr import LoFTRMatcher
from src.correspondence.lightglue import LightGlueMatcher


class TestCorrespondence(unittest.TestCase):
    """Test suite for SIFT matching, RANSAC evaluation, and experimental correspondence placeholders."""

    def setUp(self):
        # Create structured image with multiple textured lunar craters
        np.random.seed(42)
        self.img1 = np.full((200, 200), 120, dtype=np.uint8)

        # Draw features
        for x, y, r in [(50, 50, 20), (140, 60, 25), (70, 150, 30), (150, 140, 18)]:
            cv2.circle(self.img1, (x, y), r, 240, thickness=2)
            cv2.circle(self.img1, (x - 2, y - 2), max(1, r - 5), 40, thickness=-1)

        # Add minor texture/speckle
        noise = np.random.randint(-10, 10, self.img1.shape, dtype=np.int16)
        self.img1 = np.clip(self.img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Create img2 via affine transformation (simulating small shift/rotation/scale)
        rows, cols = self.img1.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
        M[0, 2] += 10  # dx
        M[1, 2] += -5  # dy
        self.img2 = cv2.warpAffine(self.img1, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    def test_sift_matching(self):
        result = match_sift_features(
            self.img1,
            self.img2,
            max_features=2000,
            ratio_threshold=0.8,
        )

        self.assertIn("source_points", result)
        self.assertIn("reference_points", result)
        self.assertIn("confidence", result)
        self.assertIn("matches", result)

        src_pts = result["source_points"]
        ref_pts = result["reference_points"]

        self.assertIsInstance(src_pts, np.ndarray)
        self.assertIsInstance(ref_pts, np.ndarray)
        self.assertEqual(src_pts.shape, ref_pts.shape)
        if len(src_pts) > 0:
            self.assertEqual(src_pts.shape[1], 2)
            self.assertEqual(ref_pts.shape[1], 2)
            self.assertGreater(result["confidence"], 0.0)

    def test_sift_matching_with_cross_check(self):
        result = match_sift_features(
            self.img1,
            self.img2,
            max_features=2000,
            ratio_threshold=0.8,
            cross_check=True,
        )
        self.assertIn("source_points", result)
        self.assertIn("reference_points", result)
        # All source and reference descriptor indices should be 1-to-1 unique
        if result["num_matches"] > 0:
            src_indices = [m["source_idx"] for m in result["matches"]]
            ref_indices = [m["reference_idx"] for m in result["matches"]]
            self.assertEqual(len(src_indices), len(set(src_indices)))
            self.assertEqual(len(ref_indices), len(set(ref_indices)))

    def test_sift_matching_empty_or_blank_image(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        result = match_sift_features(blank, blank)
        self.assertEqual(result["num_matches"], 0)
        self.assertEqual(len(result["source_points"]), 0)
        self.assertEqual(result["confidence"], 0.0)

    def test_match_with_candidate_roi(self):
        # Create a larger reference image embedding img2
        ref_full = np.full((300, 300), 100, dtype=np.uint8)
        ref_full[50:250, 50:250] = self.img2

        roi = {"bbox": (50, 50, 200, 200), "score": 0.95}
        result = match_with_candidate_roi(self.img1, ref_full, roi, ratio_threshold=0.8)

        self.assertIn("candidate_roi", result)
        if result["num_matches"] > 0:
            ref_pts = result["reference_points"]
            # All reference points should be offset by at least roi origin (50, 50)
            self.assertTrue(np.all(ref_pts[:, 0] >= 50))
            self.assertTrue(np.all(ref_pts[:, 1] >= 50))

    # --- 9 Specific Tests for Many-to-One, RANSAC Degeneracy & Candidate Selection ---

    # 1. Many-to-one reference point matching
    def test_many_to_one_reference_point_matching(self):
        src_pts = np.array([
            [10.0, 10.0], [20.0, 30.0], [40.0, 50.0], [60.0, 70.0], [80.0, 90.0]
        ], dtype=np.float32)
        # All map to the same reference coordinate
        ref_pts = np.full_like(src_pts, [100.0, 100.0])

        eval_result = evaluate_geometric_ransac(src_pts, ref_pts, source_shape=(100, 100))
        self.assertEqual(eval_result["unique_reference_points"], 1)
        self.assertEqual(eval_result["max_ref_point_multiplicity"], 5)
        self.assertTrue(eval_result["is_degenerate"])
        self.assertFalse(eval_result["is_reliable"])

    # 2. Mutual nearest-neighbor filtering
    def test_mutual_nearest_neighbor_filtering(self):
        # Test that cross_check=True strictly prevents multiple source points matching one reference point
        match_res = match_sift_features(
            self.img1, self.img2, max_features=1000, ratio_threshold=0.85, cross_check=True
        )
        if match_res["num_matches"] > 0:
            ref_pts = match_res["reference_points"]
            unique_ref_pts = np.unique(ref_pts, axis=0)
            self.assertEqual(len(ref_pts), len(unique_ref_pts))

    # 3. Duplicate reference point rejection
    def test_duplicate_reference_point_rejection(self):
        src_pts = np.array([
            [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0],
            [50.0, 50.0], [60.0, 60.0], [70.0, 70.0], [80.0, 80.0],
        ], dtype=np.float32)
        # 6 out of 8 points share point (120, 120)
        ref_pts = np.array([
            [120.0, 120.0], [120.0, 120.0], [120.0, 120.0], [120.0, 120.0],
            [120.0, 120.0], [120.0, 120.0], [150.0, 160.0], [170.0, 180.0],
        ], dtype=np.float32)

        eval_result = evaluate_geometric_ransac(src_pts, ref_pts, source_shape=(100, 100))
        self.assertTrue(eval_result["is_degenerate"])
        self.assertFalse(eval_result["is_reliable"])
        self.assertIn("DUPLICATE_REFERENCE_MATCHES", eval_result["primary_rejection_reason"])

    # 4. Collapsed homography rejection
    def test_collapsed_homography_rejection(self):
        # Points along a single line (collinear)
        src_pts = np.array([
            [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0],
            [50.0, 50.0], [60.0, 60.0], [70.0, 70.0], [80.0, 80.0],
        ], dtype=np.float32)
        ref_pts = np.array([
            [100.0, 10.0], [100.0, 20.0], [100.0, 30.0], [100.0, 40.0],
            [100.0, 50.0], [100.0, 60.0], [100.0, 70.0], [100.0, 80.0],
        ], dtype=np.float32)

        eval_result = evaluate_geometric_ransac(src_pts, ref_pts, source_shape=(100, 100))
        self.assertTrue(eval_result["is_degenerate"])
        self.assertFalse(eval_result["is_reliable"])

    # 5. Near-zero determinant rejection
    def test_near_zero_determinant_rejection(self):
        # Degenerate mapping leading to rank-deficient transformation
        src_pts = np.array([
            [10.0, 10.0], [10.0, 90.0], [90.0, 10.0], [90.0, 90.0],
            [50.0, 50.0], [30.0, 70.0], [70.0, 30.0], [40.0, 60.0],
        ], dtype=np.float32)
        # All mapped onto a horizontal line y = 50.0
        ref_pts = np.array([
            [10.0, 50.0], [20.0, 50.0], [30.0, 50.0], [40.0, 50.0],
            [50.0, 50.0], [60.0, 50.0], [70.0, 50.0], [80.0, 50.0],
        ], dtype=np.float32)

        eval_result = evaluate_geometric_ransac(src_pts, ref_pts, source_shape=(100, 100))
        self.assertTrue(eval_result["is_degenerate"])
        self.assertFalse(eval_result["is_reliable"])

    # 6. Low inlier ratio rejection
    def test_low_inlier_ratio_rejection(self):
        # 6 valid inliers + 24 completely random outliers -> inlier ratio = 6/30 = 20% (< 25%)
        np.random.seed(42)
        good_src = np.array([
            [10.0, 10.0], [50.0, 10.0], [90.0, 10.0],
            [10.0, 90.0], [50.0, 90.0], [90.0, 90.0],
        ], dtype=np.float32)
        good_ref = good_src + 50.0

        bad_src = np.random.uniform(0, 100, (24, 2)).astype(np.float32)
        bad_ref = np.random.uniform(0, 500, (24, 2)).astype(np.float32)

        src_pts = np.vstack([good_src, bad_src])
        ref_pts = np.vstack([good_ref, bad_ref])

        eval_result = evaluate_geometric_ransac(
            src_pts, ref_pts, source_shape=(100, 100), min_inlier_ratio=0.25
        )
        self.assertFalse(eval_result["is_reliable"])
        self.assertEqual(eval_result["primary_rejection_reason"], "LOW_INLIER_RATIO")

    # 7. Spatially concentrated inlier rejection
    def test_spatially_concentrated_inlier_rejection(self):
        # Inliers clustered tightly in tiny sub-region with std < 2.0
        src_pts = np.array([
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 11.0],
            [10.5, 10.5], [10.2, 10.8], [10.8, 10.2], [10.6, 10.6],
        ], dtype=np.float32)
        # Clean translation (det = 1.0, area preserved, but spread < 2.0)
        ref_pts = src_pts + 100.0

        eval_result = evaluate_geometric_ransac(src_pts, ref_pts, source_shape=(100, 100))
        self.assertTrue(eval_result["is_degenerate"])
        self.assertFalse(eval_result["is_reliable"])
        self.assertEqual(eval_result["primary_rejection_reason"], "INSUFFICIENT_SPATIAL_COVERAGE")

    # 8. Top-K candidate evaluation
    def test_top_k_candidate_evaluation(self):
        ref_full = np.full((300, 300), 100, dtype=np.uint8)
        ref_full[50:250, 50:250] = self.img2

        candidates = [
            {"roi_id": 1, "bbox": (50, 50, 200, 200), "score": 0.85, "scale": 1.0},
            {"roi_id": 2, "bbox": (0, 0, 100, 100), "score": 0.50, "scale": 0.5},
            {"roi_id": 3, "bbox": (100, 100, 100, 100), "score": 0.30, "scale": 0.5},
        ]

        selection = select_best_candidate_correspondence(
            self.img1, ref_full, candidates, ratio_threshold=0.8, cross_check=True
        )

        self.assertEqual(selection["num_candidates_evaluated"], 3)
        self.assertEqual(len(selection["candidate_evaluations"]), 3)
        self.assertIsNotNone(selection["best_candidate"])
        self.assertIn("best_match", selection)

    # 9. Selection of geometrically valid candidate over raw high-score candidate
    def test_selection_of_geometrically_valid_candidate_over_raw_high_score(self):
        # Synthetic test: Candidate 1 is textureless / wrong region (high coarse score, 0 inliers)
        # Candidate 2 contains true transformed pattern (lower coarse score, high inliers)
        ref_full = np.zeros((400, 400), dtype=np.uint8)
        # Place true match in ROI 2 location (200, 200)
        ref_full[200:400, 200:400] = self.img2

        candidates = [
            {"roi_id": 1, "bbox": (0, 0, 200, 200), "score": 0.99, "scale": 1.0},  # Blank region, high false score
            {"roi_id": 2, "bbox": (200, 200, 200, 200), "score": 0.25, "scale": 1.0},  # True region, lower score
        ]

        selection = select_best_candidate_correspondence(
            self.img1, ref_full, candidates, ratio_threshold=0.8, cross_check=True
        )

        # Should pick candidate 2 because candidate 1 produces 0 inliers
        self.assertEqual(selection["best_candidate"]["roi_id"], 2)

    def test_deep_learning_placeholders(self):
        sp = SuperPointMatcher()
        with self.assertRaises(NotImplementedError):
            sp.match(self.img1, self.img2)

        loftr = LoFTRMatcher()
        with self.assertRaises(NotImplementedError):
            loftr.match(self.img1, self.img2)

        lg = LightGlueMatcher()
        with self.assertRaises(NotImplementedError):
            lg.match(self.img1, self.img2)


if __name__ == "__main__":
    unittest.main()
