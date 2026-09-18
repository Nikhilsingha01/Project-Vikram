"""Unit tests for M2 Fine Correspondence module."""

import unittest
import numpy as np
import cv2

from src.correspondence.feature_matching import (
    match_sift_features,
    match_with_candidate_roi,
)
from src.correspondence.superpoint import SuperPointMatcher
from src.correspondence.loftr import LoFTRMatcher
from src.correspondence.lightglue import LightGlueMatcher


class TestCorrespondence(unittest.TestCase):
    """Test suite for SIFT matching and experimental correspondence placeholders."""

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
