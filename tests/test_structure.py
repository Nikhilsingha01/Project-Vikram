"""Unit tests for M2 Structure Analysis module."""

import unittest
import numpy as np
import cv2

from src.structure.gradients import (
    validate_grayscale,
    compute_gradients,
    compute_gradient_magnitude,
    compute_gradient_orientation,
    compute_gradient_field,
)
from src.structure.edges import (
    extract_canny_edges,
    extract_auto_canny_edges,
    compute_edge_density,
)
from src.structure.terrain_features import (
    extract_morphological_relief,
    extract_crater_candidates,
    extract_terrain_descriptor_map,
)
from src.structure.structure_representation import (
    StructuralRepresentation,
    extract_structure_representation,
)


class TestStructureAnalysis(unittest.TestCase):
    """Test suite for structure analysis functions."""

    def setUp(self):
        # Create synthetic lunar crater-like pattern (128x128)
        self.img = np.zeros((128, 128), dtype=np.uint8)
        # Background lunar regolith intensity
        self.img[:] = 100
        # Draw crater rim (bright circle) and shadow (dark center)
        cv2.circle(self.img, (64, 64), 30, 220, thickness=3)
        cv2.circle(self.img, (60, 60), 25, 40, thickness=-1)

    def test_validate_grayscale(self):
        # 2D Grayscale
        gray = validate_grayscale(self.img)
        self.assertEqual(gray.shape, (128, 128))

        # 3D BGR
        bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        gray_from_bgr = validate_grayscale(bgr)
        self.assertEqual(gray_from_bgr.shape, (128, 128))

        # Invalid input
        with self.assertRaises(TypeError):
            validate_grayscale("not an array")

        with self.assertRaises(ValueError):
            validate_grayscale(np.array([]))

    def test_compute_gradients(self):
        gx, gy = compute_gradients(self.img, ksize=3)
        self.assertEqual(gx.shape, (128, 128))
        self.assertEqual(gy.shape, (128, 128))
        self.assertTrue(np.any(gx != 0))
        self.assertTrue(np.any(gy != 0))

    def test_gradient_magnitude_and_orientation(self):
        gx, gy = compute_gradients(self.img, ksize=3)
        mag_norm = compute_gradient_magnitude(gx, gy, normalize=True)
        self.assertEqual(mag_norm.dtype, np.uint8)
        self.assertEqual(mag_norm.shape, (128, 128))
        self.assertGreater(np.max(mag_norm), 0)

        ori_deg = compute_gradient_orientation(gx, gy, in_degrees=True)
        self.assertEqual(ori_deg.shape, (128, 128))
        self.assertTrue(np.all(ori_deg >= 0.0) and np.all(ori_deg <= 360.0))

        field = compute_gradient_field(self.img)
        self.assertIn("gx", field)
        self.assertIn("gy", field)
        self.assertIn("magnitude", field)
        self.assertIn("orientation", field)

    def test_canny_edges(self):
        edges = extract_canny_edges(self.img, low_threshold=50, high_threshold=150)
        self.assertEqual(edges.shape, (128, 128))
        self.assertEqual(edges.dtype, np.uint8)

        density = compute_edge_density(edges)
        self.assertGreater(density, 0.0)
        self.assertLessEqual(density, 1.0)

        auto_edges = extract_auto_canny_edges(self.img)
        self.assertEqual(auto_edges.shape, (128, 128))

    def test_terrain_features(self):
        relief = extract_morphological_relief(self.img, kernel_size=5, morph_type="relief")
        self.assertEqual(relief.shape, (128, 128))
        self.assertEqual(relief.dtype, np.uint8)

        descriptor_map = extract_terrain_descriptor_map(self.img, kernel_size=5)
        self.assertEqual(descriptor_map.shape, (128, 128))

        craters = extract_crater_candidates(self.img, min_radius=10, max_radius=40)
        self.assertIsInstance(craters, list)

    def test_structural_representation(self):
        rep = extract_structure_representation(self.img)
        self.assertIsInstance(rep, StructuralRepresentation)
        self.assertEqual(rep.original_shape, (128, 128))
        self.assertEqual(rep.composite.shape, (128, 128))
        self.assertEqual(rep.composite.dtype, np.uint8)
        self.assertGreater(rep.edge_density, 0.0)

        rep_dict = rep.to_dict()
        self.assertIn("composite", rep_dict)
        self.assertIn("edges", rep_dict)


if __name__ == "__main__":
    unittest.main()
