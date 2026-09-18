"""Unit tests for M3 Evidence-Gated Validation pipeline.

Test Strategy
-------------
All M3 module tests use unittest.TestCase and are structured in the same
pattern as the existing M2 test suite (tests/test_structure.py,
tests/test_correspondence.py).

Phase 1 (scaffold) tests verify:
  1. Module-level imports resolve without error.
  2. Dataclass and Enum definitions are correct (fields, types, defaults).
  3. Unimplemented stubs raise NotImplementedError.

Phase 2 (M3 Geometric Estimation implemented) tests verify:
  4. Successful affine estimation from synthetic points with a known transform.
  5. Successful homography estimation from synthetic points with a known transform.
  6. Graceful handling of insufficient correspondences.
  7. Graceful rejection of NaN / Inf coordinates.
  8. Graceful handling of RANSAC failure (random noise correspondences).
  9. Configurable RANSAC threshold and confidence settings.
  10. Missing required key raises ValueError.

Synthetic Test Data
--------------------
All algorithmic tests use self-contained synthetic point sets or images —
no external data files are required.

M2 Correspondence Dict Mock
-----------------------------
_make_correspondence() produces a minimal dict matching the M2 data contract.
_make_transformed_correspondence() produces point pairs with a known affine
transform plus configurable Gaussian noise and outlier fraction.
"""

import unittest
from typing import Any, Dict, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Import guards — validate that all M3 modules are importable
# ---------------------------------------------------------------------------


class TestM3Imports(unittest.TestCase):
    """Verify that all M3 sub-modules import without error."""

    def test_import_evidence_gate(self):
        """EvidenceGate module imports cleanly."""
        # TODO (M3-TEST-IMP-01): Replace with direct assertion once implemented.
        from src.validation import evidence_gate  # noqa: F401

    def test_import_geometric_estimation(self):
        """GeometricEstimation module imports cleanly."""
        from src.validation import geometric_estimation  # noqa: F401

    def test_import_inlier_analysis(self):
        """InlierAnalysis module imports cleanly."""
        from src.validation import inlier_analysis  # noqa: F401

    def test_import_reprojection(self):
        """Reprojection module imports cleanly."""
        from src.validation import reprojection  # noqa: F401

    def test_import_registration(self):
        """Registration module imports cleanly."""
        from src.validation import registration  # noqa: F401

    def test_import_metrics(self):
        """Metrics module imports cleanly."""
        from src.validation import metrics  # noqa: F401

    def test_import_validation_package(self):
        """Top-level src.validation package imports cleanly."""
        import src.validation  # noqa: F401


# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------


def _make_synthetic_images():
    """Create a pair of structured synthetic 200×200 grayscale images.

    Returns:
        Tuple (img_src, img_ref, M) where M is the 2×3 affine transform
        that maps img_src → img_ref.
    """
    np.random.seed(0)
    img_src = np.full((200, 200), 120, dtype=np.uint8)

    # Draw structured crater-like features
    for cx, cy, r in [(50, 50, 20), (140, 60, 25), (70, 150, 30), (150, 140, 18)]:
        cv2.circle(img_src, (cx, cy), r, 240, thickness=2)
        cv2.circle(img_src, (cx - 2, cy - 2), max(1, r - 5), 40, thickness=-1)

    noise = np.random.randint(-8, 8, img_src.shape, dtype=np.int16)
    img_src = np.clip(img_src.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    rows, cols = img_src.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=5, scale=1.0)
    M[0, 2] += 10.0   # dx
    M[1, 2] += -5.0   # dy
    img_ref = cv2.warpAffine(img_src, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    return img_src, img_ref, M


def _make_correspondence(n: int = 30) -> Dict[str, Any]:
    """Create a minimal M2 correspondence dict with N synthetic matches.

    The source/reference points are generated as random pairs with a small
    perturbation so they are not perfectly collinear.

    Args:
        n: Number of match pairs.

    Returns:
        Dict matching the M2 data contract.
    """
    np.random.seed(42)
    src_pts = np.random.uniform(10, 190, (n, 2)).astype(np.float32)
    # Reference points = source + small random displacement (simulates inliers)
    ref_pts = src_pts + np.random.uniform(-5, 5, (n, 2)).astype(np.float32)

    return {
        "source_points": src_pts,
        "reference_points": ref_pts,
        "confidence": 0.65,
        "matches": [{"source_idx": i, "reference_idx": i} for i in range(n)],
        "num_matches": n,
        "num_keypoints_source": n + 20,
        "num_keypoints_reference": n + 15,
    }


# ---------------------------------------------------------------------------
# EvidenceGate tests
# ---------------------------------------------------------------------------


class TestEvidenceGate(unittest.TestCase):
    """Tests for EvidenceGateConfig, EvidenceGateResult, and EvidenceGate."""

    def setUp(self):
        from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig
        self.EvidenceGate = EvidenceGate
        self.EvidenceGateConfig = EvidenceGateConfig
        self.correspondence = _make_correspondence(n=30)

    def test_config_defaults(self):
        """EvidenceGateConfig instantiates with correct default values."""
        cfg = self.EvidenceGateConfig()
        self.assertEqual(cfg.min_matches, 10)
        self.assertGreater(cfg.min_confidence, 0.0)
        self.assertLessEqual(cfg.min_confidence, 1.0)
        self.assertGreater(cfg.confidence_weight, 0.0)

    def test_config_custom(self):
        """EvidenceGateConfig accepts custom values."""
        cfg = self.EvidenceGateConfig(min_matches=5, min_confidence=0.2)
        self.assertEqual(cfg.min_matches, 5)
        self.assertAlmostEqual(cfg.min_confidence, 0.2)

    def test_gate_instantiation(self):
        """EvidenceGate can be instantiated with default and custom config."""
        gate_default = self.EvidenceGate()
        self.assertIsNotNone(gate_default.config)

        gate_custom = self.EvidenceGate(config=self.EvidenceGateConfig(min_matches=4))
        self.assertEqual(gate_custom.config.min_matches, 4)

    def test_evaluate_raises_not_implemented(self):
        """EvidenceGate.evaluate() raises NotImplementedError (pre-implementation)."""
        gate = self.EvidenceGate()
        with self.assertRaises(NotImplementedError):
            gate.evaluate(self.correspondence)

    def test_gate_result_dataclass(self):
        """EvidenceGateResult can be constructed directly."""
        from src.validation.evidence_gate import EvidenceGateResult
        result = EvidenceGateResult(
            passed=True,
            rejection_reason=None,
            evidence_score=0.75,
            diagnostics={"count_score": 0.6},
        )
        self.assertTrue(result.passed)
        self.assertIsNone(result.rejection_reason)
        self.assertAlmostEqual(result.evidence_score, 0.75)


# ---------------------------------------------------------------------------
# GeometricEstimation tests
# ---------------------------------------------------------------------------


class TestGeometricEstimation(unittest.TestCase):
    """Tests for TransformModel, GeometricEstimatorConfig, and GeometricEstimator."""

    def setUp(self):
        from src.validation.geometric_estimation import (
            GeometricEstimator,
            GeometricEstimatorConfig,
            TransformModel,
            EstimationResult,
        )
        self.GeometricEstimator = GeometricEstimator
        self.GeometricEstimatorConfig = GeometricEstimatorConfig
        self.TransformModel = TransformModel
        self.EstimationResult = EstimationResult
        self.correspondence = _make_correspondence(n=30)

    def test_transform_model_enum_values(self):
        """TransformModel has all required members."""
        TM = self.TransformModel
        self.assertTrue(hasattr(TM, "HOMOGRAPHY"))
        self.assertTrue(hasattr(TM, "AFFINE"))
        self.assertTrue(hasattr(TM, "SIMILARITY"))
        self.assertTrue(hasattr(TM, "TRANSLATION"))

    def test_config_defaults(self):
        """GeometricEstimatorConfig has sensible defaults."""
        cfg = self.GeometricEstimatorConfig()
        self.assertEqual(cfg.model, self.TransformModel.HOMOGRAPHY)
        self.assertGreater(cfg.ransac_reproj_threshold, 0.0)
        self.assertGreater(cfg.ransac_confidence, 0.0)
        self.assertLessEqual(cfg.ransac_confidence, 1.0)
        self.assertGreater(cfg.min_inliers, 0)

    def test_estimator_instantiation(self):
        """GeometricEstimator instantiates with default and custom config."""
        est_default = self.GeometricEstimator()
        self.assertIsNotNone(est_default.config)

        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
        )
        est_custom = self.GeometricEstimator(config=cfg)
        self.assertEqual(est_custom.config.model, self.TransformModel.AFFINE)

    def test_estimate_is_callable(self):
        """GeometricEstimator.estimate() is callable and returns EstimationResult."""
        estimator = self.GeometricEstimator(
            config=self.GeometricEstimatorConfig(
                model=self.TransformModel.HOMOGRAPHY,
                min_inliers=1,
                min_inlier_ratio=0.0,
            )
        )
        result = estimator.estimate(self.correspondence)
        self.assertIsInstance(result, self.EstimationResult)

    def test_estimation_result_dataclass(self):
        """EstimationResult can be constructed directly with new field names."""
        mask = np.ones(30, dtype=bool)
        result = self.EstimationResult(
            success=True,
            transform_type="HOMOGRAPHY",
            transform_matrix=np.eye(3, dtype=np.float64),
            inlier_mask=mask,
            num_inliers=30,
            num_outliers=0,
            total_matches=30,
            inlier_ratio=1.0,
            model=self.TransformModel.HOMOGRAPHY,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_inliers, 30)
        self.assertEqual(result.inlier_count, 30)   # backward-compat alias
        self.assertIsNone(result.failure_reason)     # backward-compat alias
        self.assertIsNone(result.reason)


# ---------------------------------------------------------------------------
# GeometricEstimation algorithmic tests (Phase 2 — implemented)
# ---------------------------------------------------------------------------


def _make_transformed_correspondence(
    n_inliers: int = 50,
    n_outliers: int = 10,
    transform_matrix: Optional[np.ndarray] = None,
    noise_sigma: float = 0.5,
    seed: int = 7,
) -> tuple:
    """Generate synthetic correspondence dict with a known affine transform.

    Creates N inlier pairs related by the given affine matrix and adds
    n_outliers pairs with random reference coordinates.

    Args:
        n_inliers: Number of inlier pairs.
        n_outliers: Number of outlier pairs (random, not on transform).
        transform_matrix: (2, 3) affine matrix.  Defaults to a 5° rotation
            + (10, -5) translation transform.
        noise_sigma: Std of Gaussian noise added to reference inlier points.
        seed: Random seed.

    Returns:
        Tuple (correspondence_dict, true_matrix) where true_matrix is the
        ground-truth (2, 3) affine transform used to generate inliers.
    """
    rng = np.random.RandomState(seed)

    if transform_matrix is None:
        # 5° rotation + (10, -5) translation
        angle = np.deg2rad(5.0)
        c, s = np.cos(angle), np.sin(angle)
        transform_matrix = np.array([
            [c, -s, 10.0],
            [s,  c, -5.0],
        ], dtype=np.float64)

    # Generate inlier source points in [20, 180] range
    src_in = rng.uniform(20, 180, (n_inliers, 2)).astype(np.float64)
    # Apply affine transform: ref = A * src + t
    A = transform_matrix[:, :2]
    t = transform_matrix[:, 2]
    ref_in = (src_in @ A.T) + t
    # Add Gaussian noise
    ref_in += rng.normal(0, noise_sigma, ref_in.shape)

    # Generate outlier pairs — random source and random reference
    src_out = rng.uniform(20, 180, (n_outliers, 2)).astype(np.float64)
    ref_out = rng.uniform(20, 180, (n_outliers, 2)).astype(np.float64)

    src_pts = np.vstack([src_in, src_out]).astype(np.float32)
    ref_pts = np.vstack([ref_in, ref_out]).astype(np.float32)
    n_total = n_inliers + n_outliers

    correspondence = {
        "source_points": src_pts,
        "reference_points": ref_pts,
        "confidence": 0.75,
        "matches": [{"source_idx": i, "reference_idx": i} for i in range(n_total)],
        "num_matches": n_total,
        "num_keypoints_source": n_total + 20,
        "num_keypoints_reference": n_total + 15,
    }
    return correspondence, transform_matrix




class TestGeometricEstimationImpl(unittest.TestCase):
    """Algorithmic tests for GeometricEstimator (M3 Phase 2 — implemented).

    All tests use self-contained synthetic point sets and do not depend on
    any M2 module or external data files.
    """

    def setUp(self):
        from src.validation.geometric_estimation import (
            GeometricEstimator,
            GeometricEstimatorConfig,
            TransformModel,
            EstimationResult,
        )
        self.GeometricEstimator = GeometricEstimator
        self.GeometricEstimatorConfig = GeometricEstimatorConfig
        self.TransformModel = TransformModel
        self.EstimationResult = EstimationResult

        # Known affine: 5° rotation + (10, -5) translation
        angle = np.deg2rad(5.0)
        c, s = np.cos(angle), np.sin(angle)
        self.known_affine = np.array([
            [c, -s, 10.0],
            [s,  c, -5.0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Test 1: Successful affine estimation
    # ------------------------------------------------------------------

    def test_affine_estimation_success(self):
        """Affine estimation succeeds on synthetic inlier-majority data.

        With 50 inliers (low noise) and 10 outliers, RANSAC should recover
        the transform and report success with high inlier ratio.
        """
        correspondence, true_M = _make_transformed_correspondence(
            n_inliers=50, n_outliers=10, transform_matrix=self.known_affine,
            noise_sigma=0.3,
        )
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
            refine_with_lm=True,
        )
        estimator = self.GeometricEstimator(config=cfg)
        result = estimator.estimate(correspondence)

        self.assertIsInstance(result, self.EstimationResult)
        self.assertTrue(result.success, f"Expected success, got reason: {result.reason}")
        self.assertEqual(result.transform_type, "AFFINE")
        self.assertIsNotNone(result.transform_matrix)
        self.assertIsNotNone(result.inlier_mask)
        self.assertEqual(result.transform_matrix.shape, (2, 3))
        self.assertGreaterEqual(result.num_inliers, 30)  # majority recovered
        self.assertGreater(result.inlier_ratio, 0.5)
        self.assertIsNone(result.reason)
        # num_inliers + num_outliers == total_matches
        self.assertEqual(result.num_inliers + result.num_outliers, result.total_matches)
        # inlier_mask length matches total cleaned input
        self.assertEqual(len(result.inlier_mask), result.total_matches)

    def test_affine_matrix_accuracy(self):
        """Recovered affine matrix is close to the ground-truth transform.

        With very low noise (sigma=0.1), the estimated matrix should match
        the known affine within a reasonable tolerance.
        """
        correspondence, true_M = _make_transformed_correspondence(
            n_inliers=80, n_outliers=5, transform_matrix=self.known_affine,
            noise_sigma=0.1,
        )
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.AFFINE,
            ransac_reproj_threshold=2.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
            refine_with_lm=True,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertTrue(result.success, f"reason: {result.reason}")
        M_est = result.transform_matrix
        # The linear (rotation+scale) part should be close
        np.testing.assert_allclose(
            M_est[:, :2], true_M[:, :2], atol=0.05,
            err_msg="Linear part of affine matrix deviates too much from ground truth",
        )
        # Translation part should be close
        np.testing.assert_allclose(
            M_est[:, 2], true_M[:, 2], atol=1.5,
            err_msg="Translation part of affine matrix deviates too much",
        )

    # ------------------------------------------------------------------
    # Test 2: Successful homography estimation
    # ------------------------------------------------------------------

    def test_homography_estimation_success(self):
        """Homography estimation succeeds on synthetic inlier-majority data.

        Uses a near-affine homography (planar scene approximation).
        """
        # Build homography from the affine matrix
        H_true = np.eye(3, dtype=np.float64)
        H_true[:2, :] = self.known_affine

        # Generate inlier pairs using full homography projection
        rng = np.random.RandomState(13)
        n_in, n_out = 60, 10
        src_in = rng.uniform(20, 180, (n_in, 2)).astype(np.float64)
        # Project through H
        src_h = np.hstack([src_in, np.ones((n_in, 1))])
        ref_h = (H_true @ src_h.T).T
        ref_in = (ref_h[:, :2] / ref_h[:, 2:3]) + rng.normal(0, 0.3, (n_in, 2))

        src_out = rng.uniform(20, 180, (n_out, 2)).astype(np.float64)
        ref_out = rng.uniform(20, 180, (n_out, 2)).astype(np.float64)

        src_pts = np.vstack([src_in, src_out]).astype(np.float32)
        ref_pts = np.vstack([ref_in, ref_out]).astype(np.float32)

        correspondence = {
            "source_points": src_pts,
            "reference_points": ref_pts,
            "confidence": 0.8,
            "num_matches": len(src_pts),
            "matches": [],
            "num_keypoints_source": 100,
            "num_keypoints_reference": 100,
        }

        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.HOMOGRAPHY,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
            refine_with_lm=True,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertIsInstance(result, self.EstimationResult)
        self.assertTrue(result.success, f"reason: {result.reason}")
        self.assertEqual(result.transform_type, "HOMOGRAPHY")
        self.assertEqual(result.transform_matrix.shape, (3, 3))
        self.assertGreaterEqual(result.num_inliers, 30)
        self.assertGreater(result.inlier_ratio, 0.4)
        self.assertIsNone(result.reason)

    # ------------------------------------------------------------------
    # Test 3: Insufficient correspondences
    # ------------------------------------------------------------------

    def test_insufficient_correspondences_homography(self):
        """Homography requires ≥ 4 points; fewer → success=False."""
        src = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)  # only 3
        ref = src + 5.0
        correspondence = {
            "source_points": src,
            "reference_points": ref,
            "confidence": 0.9,
            "num_matches": 3,
            "matches": [],
            "num_keypoints_source": 3,
            "num_keypoints_reference": 3,
        }
        cfg = self.GeometricEstimatorConfig(model=self.TransformModel.HOMOGRAPHY)
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.reason)
        self.assertIn("insufficient", result.reason.lower())
        self.assertIsNone(result.transform_matrix)
        self.assertIsNone(result.inlier_mask)

    def test_insufficient_correspondences_affine(self):
        """Affine requires ≥ 3 points; 2 points → success=False."""
        src = np.array([[10, 20], [30, 40]], dtype=np.float32)  # only 2
        ref = src + 5.0
        correspondence = {
            "source_points": src,
            "reference_points": ref,
            "confidence": 0.9,
            "num_matches": 2,
            "matches": [],
            "num_keypoints_source": 2,
            "num_keypoints_reference": 2,
        }
        cfg = self.GeometricEstimatorConfig(model=self.TransformModel.AFFINE)
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertFalse(result.success)
        self.assertIn("insufficient", result.reason.lower())

    def test_empty_correspondence(self):
        """Zero points → success=False regardless of model."""
        empty = np.empty((0, 2), dtype=np.float32)
        correspondence = {
            "source_points": empty,
            "reference_points": empty,
            "confidence": 0.0,
            "num_matches": 0,
            "matches": [],
            "num_keypoints_source": 0,
            "num_keypoints_reference": 0,
        }
        for model in (self.TransformModel.HOMOGRAPHY, self.TransformModel.AFFINE):
            cfg = self.GeometricEstimatorConfig(model=model)
            result = self.GeometricEstimator(config=cfg).estimate(correspondence)
            self.assertFalse(result.success, f"Expected failure for model {model}")

    # ------------------------------------------------------------------
    # Test 4: NaN / Inf input rejection
    # ------------------------------------------------------------------

    def test_nan_coordinates_rejected(self):
        """Rows containing NaN are silently dropped; if all invalid → failure."""
        nan_src = np.full((5, 2), np.nan, dtype=np.float32)
        nan_ref = np.full((5, 2), np.nan, dtype=np.float32)
        correspondence = {
            "source_points": nan_src,
            "reference_points": nan_ref,
            "confidence": 0.5,
            "num_matches": 5,
            "matches": [],
            "num_keypoints_source": 5,
            "num_keypoints_reference": 5,
        }
        cfg = self.GeometricEstimatorConfig(model=self.TransformModel.AFFINE)
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertFalse(result.success)
        self.assertIn("invalid", result.reason.lower())

    def test_inf_coordinates_rejected(self):
        """Rows containing Inf are silently dropped; if all invalid → failure."""
        inf_src = np.full((4, 2), np.inf, dtype=np.float32)
        inf_ref = np.full((4, 2), np.inf, dtype=np.float32)
        correspondence = {
            "source_points": inf_src,
            "reference_points": inf_ref,
            "confidence": 0.5,
            "num_matches": 4,
            "matches": [],
            "num_keypoints_source": 4,
            "num_keypoints_reference": 4,
        }
        cfg = self.GeometricEstimatorConfig(model=self.TransformModel.HOMOGRAPHY)
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertFalse(result.success)

    def test_mixed_valid_nan_rows(self):
        """When some rows are NaN and the valid remainder meets minimum,
        estimation continues on the clean subset."""
        correspondence, _ = _make_transformed_correspondence(
            n_inliers=40, n_outliers=5, transform_matrix=self.known_affine,
            noise_sigma=0.3, seed=3,
        )
        # Inject NaN into the first 5 rows of source_points
        src = correspondence["source_points"].copy()
        src[:5] = np.nan
        correspondence["source_points"] = src

        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)
        # Should still succeed since enough valid rows remain
        self.assertTrue(result.success, f"Expected success after NaN drop, got: {result.reason}")

    # ------------------------------------------------------------------
    # Test 5: RANSAC failure on purely random correspondences
    # ------------------------------------------------------------------

    def test_ransac_failure_random_correspondences(self):
        """Purely random (uncorrelated) correspondences produce a failure result.

        With 30 random pairs and a tight reproj threshold (1.0 px), RANSAC
        will find at most a handful of chance inliers — below min_inliers=20.
        """
        rng = np.random.RandomState(99)
        n = 30
        src = rng.uniform(0, 200, (n, 2)).astype(np.float32)
        ref = rng.uniform(0, 200, (n, 2)).astype(np.float32)  # completely random

        correspondence = {
            "source_points": src,
            "reference_points": ref,
            "confidence": 0.5,
            "num_matches": n,
            "matches": [],
            "num_keypoints_source": n,
            "num_keypoints_reference": n,
        }
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.HOMOGRAPHY,
            ransac_reproj_threshold=1.0,   # tight threshold
            min_inliers=20,                # high requirement
            min_inlier_ratio=0.5,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        # Either RANSAC returned None OR inlier check failed
        self.assertFalse(result.success)
        self.assertIsNotNone(result.reason)

    # ------------------------------------------------------------------
    # Test 6: Missing required keys raise ValueError
    # ------------------------------------------------------------------

    def test_missing_source_points_raises_value_error(self):
        """Missing 'source_points' key raises ValueError."""
        bad = {"reference_points": np.zeros((10, 2), dtype=np.float32)}
        estimator = self.GeometricEstimator()
        with self.assertRaises(ValueError) as ctx:
            estimator.estimate(bad)
        self.assertIn("source_points", str(ctx.exception))

    def test_missing_reference_points_raises_value_error(self):
        """Missing 'reference_points' key raises ValueError."""
        bad = {"source_points": np.zeros((10, 2), dtype=np.float32)}
        estimator = self.GeometricEstimator()
        with self.assertRaises(ValueError) as ctx:
            estimator.estimate(bad)
        self.assertIn("reference_points", str(ctx.exception))

    # ------------------------------------------------------------------
    # Test 7: Result field completeness and types
    # ------------------------------------------------------------------

    def test_result_field_types_on_success(self):
        """On success, all EstimationResult fields have correct types."""
        correspondence, _ = _make_transformed_correspondence(
            n_inliers=50, n_outliers=5, transform_matrix=self.known_affine,
            noise_sigma=0.2, seed=5,
        )
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)
        self.assertTrue(result.success)

        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.transform_type, str)
        self.assertIsInstance(result.transform_matrix, np.ndarray)
        self.assertIsInstance(result.inlier_mask, np.ndarray)
        self.assertEqual(result.inlier_mask.dtype, bool)
        self.assertIsInstance(result.num_inliers, int)
        self.assertIsInstance(result.num_outliers, int)
        self.assertIsInstance(result.total_matches, int)
        self.assertIsInstance(result.inlier_ratio, float)
        self.assertIsInstance(result.model, self.TransformModel)
        self.assertIsNone(result.reason)
        self.assertIsInstance(result.diagnostics, dict)
        # Sanity: counts are consistent
        self.assertEqual(result.num_inliers + result.num_outliers, result.total_matches)
        self.assertAlmostEqual(
            result.inlier_ratio,
            result.num_inliers / result.total_matches,
            places=6,
        )

    def test_result_field_types_on_failure(self):
        """On failure, transform_matrix and inlier_mask are None."""
        src = np.array([[10, 20], [30, 40]], dtype=np.float32)  # < 3 for AFFINE
        bad = {
            "source_points": src,
            "reference_points": src + 5,
            "confidence": 0.5,
            "num_matches": 2,
            "matches": [],
            "num_keypoints_source": 2,
            "num_keypoints_reference": 2,
        }
        result = self.GeometricEstimator(
            config=self.GeometricEstimatorConfig(model=self.TransformModel.AFFINE)
        ).estimate(bad)

        self.assertFalse(result.success)
        self.assertIsNone(result.transform_matrix)
        self.assertIsNone(result.inlier_mask)
        self.assertEqual(result.num_inliers, 0)
        self.assertIsNotNone(result.reason)
        self.assertIsInstance(result.reason, str)

    # ------------------------------------------------------------------
    # Test 8: Configurable RANSAC threshold sensitivity
    # ------------------------------------------------------------------

    def test_tight_threshold_yields_fewer_inliers(self):
        """A tighter RANSAC threshold should yield equal or fewer inliers."""
        correspondence, _ = _make_transformed_correspondence(
            n_inliers=50, n_outliers=10, transform_matrix=self.known_affine,
            noise_sigma=1.5,   # higher noise → more points near threshold boundary
        )
        base_cfg = dict(
            model=self.TransformModel.AFFINE,
            min_inliers=1,
            min_inlier_ratio=0.0,
            refine_with_lm=False,
        )
        loose = self.GeometricEstimator(
            config=self.GeometricEstimatorConfig(ransac_reproj_threshold=10.0, **base_cfg)
        ).estimate(correspondence)
        tight = self.GeometricEstimator(
            config=self.GeometricEstimatorConfig(ransac_reproj_threshold=1.0, **base_cfg)
        ).estimate(correspondence)

        if loose.success and tight.success:
            self.assertLessEqual(tight.num_inliers, loose.num_inliers)

    # ------------------------------------------------------------------
    # Test 9: Similarity model runs without error
    # ------------------------------------------------------------------

    def test_similarity_estimation(self):
        """Similarity estimation runs and returns a result (success or graceful failure)."""
        correspondence, _ = _make_transformed_correspondence(
            n_inliers=40, n_outliers=10, transform_matrix=self.known_affine,
            noise_sigma=0.5,
        )
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.SIMILARITY,
            ransac_reproj_threshold=5.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)
        # Must return an EstimationResult regardless
        self.assertIsInstance(result, self.EstimationResult)
        self.assertIn(result.transform_type, ("SIMILARITY",))
        if result.success:
            self.assertEqual(result.transform_matrix.shape, (2, 3))

    # ------------------------------------------------------------------
    # Test 10: Translation model
    # ------------------------------------------------------------------

    def test_translation_estimation(self):
        """Translation-only estimation recovers median displacement."""
        rng = np.random.RandomState(21)
        n = 40
        dx, dy = 15.0, -8.0
        src = rng.uniform(20, 180, (n, 2)).astype(np.float32)
        # Pure translation + small noise
        ref = (src + np.array([dx, dy])) + rng.normal(0, 0.5, (n, 2)).astype(np.float32)

        correspondence = {
            "source_points": src,
            "reference_points": ref,
            "confidence": 0.9,
            "num_matches": n,
            "matches": [],
            "num_keypoints_source": n,
            "num_keypoints_reference": n,
        }
        cfg = self.GeometricEstimatorConfig(
            model=self.TransformModel.TRANSLATION,
            ransac_reproj_threshold=2.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        result = self.GeometricEstimator(config=cfg).estimate(correspondence)

        self.assertTrue(result.success, f"reason: {result.reason}")
        M = result.transform_matrix
        self.assertEqual(M.shape, (2, 3))
        # Translation values should be close to dx, dy
        self.assertAlmostEqual(M[0, 2], dx, delta=1.5)
        self.assertAlmostEqual(M[1, 2], dy, delta=1.5)


# ---------------------------------------------------------------------------
# InlierAnalysis tests
# ---------------------------------------------------------------------------


class TestInlierAnalysis(unittest.TestCase):
    """Tests for InlierReport, InlierAnalyzer, and analyse_inliers()."""

    def setUp(self):
        from src.validation.inlier_analysis import (
            InlierAnalyzer,
            InlierReport,
            analyse_inliers,
        )
        self.InlierAnalyzer = InlierAnalyzer
        self.InlierReport = InlierReport
        self.analyse_inliers = analyse_inliers

        n = 30
        self.mask = np.array([True] * 22 + [False] * 8, dtype=bool)
        self.src_pts = np.random.uniform(10, 190, (n, 2)).astype(np.float32)
        self.ref_pts = self.src_pts + 5.0

    def test_analyser_instantiation(self):
        """InlierAnalyzer instantiates with default and custom grid_divisions."""
        analyser_default = self.InlierAnalyzer()
        self.assertEqual(analyser_default.grid_divisions, 4)

        analyser_custom = self.InlierAnalyzer(grid_divisions=8)
        self.assertEqual(analyser_custom.grid_divisions, 8)

    def test_analyse_is_callable(self):
        """InlierAnalyzer.analyse() is callable and returns an InlierReport."""
        analyser = self.InlierAnalyzer()
        report = analyser.analyse(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertIsInstance(report, self.InlierReport)

    def test_standalone_analyse_is_callable(self):
        """analyse_inliers() is callable and returns an InlierReport."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertIsInstance(report, self.InlierReport)

    def test_inlier_report_dataclass(self):
        """InlierReport can be constructed directly with new field names."""
        in_src  = self.src_pts[:22].astype(np.float64)
        in_ref  = self.ref_pts[:22].astype(np.float64)
        out_src = self.src_pts[22:].astype(np.float64)
        out_ref = self.ref_pts[22:].astype(np.float64)
        report = self.InlierReport(
            inlier_mask=self.mask,
            inlier_source_points=in_src,
            inlier_reference_points=in_ref,
            outlier_source_points=out_src,
            outlier_reference_points=out_ref,
            inlier_count=22,
            outlier_count=8,
            total_count=30,
            inlier_ratio=22 / 30,
            spatial_coverage=0.70,
            residual_stats={"mean": 2.1, "median": 1.9, "max": 5.3, "std": 0.8},
        )
        self.assertEqual(report.inlier_count, 22)
        self.assertAlmostEqual(report.inlier_ratio, 22 / 30)
        self.assertAlmostEqual(report.spatial_coverage, 0.70)
        # backward-compat alias
        self.assertAlmostEqual(report.spatial_uniformity, 0.70)


# ---------------------------------------------------------------------------
# InlierAnalysis algorithmic tests (Phase 3 — implemented)
# ---------------------------------------------------------------------------


class TestInlierAnalysisImpl(unittest.TestCase):
    """Algorithmic tests for InlierAnalyzer and analyse_inliers() (Phase 3).

    All tests are self-contained and require no external data files.
    """

    def setUp(self):
        from src.validation.inlier_analysis import InlierAnalyzer, InlierReport, analyse_inliers
        self.InlierAnalyzer = InlierAnalyzer
        self.InlierReport = InlierReport
        self.analyse_inliers = analyse_inliers

        # Standard 30-point fixture: 22 inliers, 8 outliers
        rng = np.random.RandomState(0)
        self.n = 30
        self.n_inliers = 22
        self.n_outliers = 8
        self.mask = np.array([True] * self.n_inliers + [False] * self.n_outliers, dtype=bool)
        self.src_pts = rng.uniform(10, 190, (self.n, 2)).astype(np.float32)
        self.ref_pts = (self.src_pts + rng.uniform(2, 8, (self.n, 2))).astype(np.float32)

    # ------------------------------------------------------------------
    # Test 1: Normal inlier / outlier separation
    # ------------------------------------------------------------------

    def test_normal_separation_counts(self):
        """Correct inlier/outlier counts for a mixed mask."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertEqual(report.inlier_count, self.n_inliers)
        self.assertEqual(report.outlier_count, self.n_outliers)
        self.assertEqual(report.total_count, self.n)
        self.assertAlmostEqual(report.inlier_ratio, self.n_inliers / self.n, places=6)

    def test_normal_separation_shapes(self):
        """Separated point arrays have correct shapes."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertEqual(report.inlier_source_points.shape, (self.n_inliers, 2))
        self.assertEqual(report.inlier_reference_points.shape, (self.n_inliers, 2))
        self.assertEqual(report.outlier_source_points.shape, (self.n_outliers, 2))
        self.assertEqual(report.outlier_reference_points.shape, (self.n_outliers, 2))

    def test_normal_separation_values(self):
        """Inlier points match source_points[mask] exactly."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        expected_src_in = self.src_pts[self.mask].astype(np.float64)
        expected_ref_in = self.ref_pts[self.mask].astype(np.float64)
        expected_src_out = self.src_pts[~self.mask].astype(np.float64)
        expected_ref_out = self.ref_pts[~self.mask].astype(np.float64)

        np.testing.assert_array_almost_equal(report.inlier_source_points, expected_src_in)
        np.testing.assert_array_almost_equal(report.inlier_reference_points, expected_ref_in)
        np.testing.assert_array_almost_equal(report.outlier_source_points, expected_src_out)
        np.testing.assert_array_almost_equal(report.outlier_reference_points, expected_ref_out)

    # ------------------------------------------------------------------
    # Test 2: All inliers (mask = all True)
    # ------------------------------------------------------------------

    def test_all_inliers(self):
        """All-True mask: outlier arrays are empty, inlier_ratio = 1.0."""
        all_true = np.ones(self.n, dtype=bool)
        report = self.analyse_inliers(
            inlier_mask=all_true,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertEqual(report.inlier_count, self.n)
        self.assertEqual(report.outlier_count, 0)
        self.assertAlmostEqual(report.inlier_ratio, 1.0)
        self.assertEqual(report.outlier_source_points.shape[0], 0)
        self.assertEqual(report.outlier_reference_points.shape[0], 0)
        self.assertEqual(report.inlier_source_points.shape, (self.n, 2))

    # ------------------------------------------------------------------
    # Test 3: All outliers (mask = all False)
    # ------------------------------------------------------------------

    def test_all_outliers(self):
        """All-False mask: inlier arrays are empty, inlier_ratio = 0.0."""
        all_false = np.zeros(self.n, dtype=bool)
        report = self.analyse_inliers(
            inlier_mask=all_false,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertEqual(report.inlier_count, 0)
        self.assertEqual(report.outlier_count, self.n)
        self.assertAlmostEqual(report.inlier_ratio, 0.0)
        self.assertEqual(report.inlier_source_points.shape[0], 0)
        self.assertEqual(report.inlier_reference_points.shape[0], 0)
        self.assertEqual(report.spatial_coverage, 0.0)

    # ------------------------------------------------------------------
    # Test 4: Invalid mask size
    # ------------------------------------------------------------------

    def test_mask_size_mismatch_raises_value_error(self):
        """Mask length != N raises ValueError."""
        bad_mask = np.ones(self.n + 5, dtype=bool)  # wrong length
        with self.assertRaises(ValueError) as ctx:
            self.analyse_inliers(
                inlier_mask=bad_mask,
                source_points=self.src_pts,
                reference_points=self.ref_pts,
            )
        self.assertIn("inlier_mask", str(ctx.exception).lower())

    def test_src_ref_size_mismatch_raises_value_error(self):
        """source_points and reference_points with different N raises ValueError."""
        with self.assertRaises(ValueError):
            self.analyse_inliers(
                inlier_mask=self.mask,
                source_points=self.src_pts,
                reference_points=self.ref_pts[:10],   # truncated
            )

    def test_wrong_shape_raises_value_error(self):
        """source_points with shape (N, 3) raises ValueError."""
        bad_pts = np.ones((self.n, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.analyse_inliers(
                inlier_mask=self.mask,
                source_points=bad_pts,
                reference_points=self.ref_pts,
            )

    def test_non_array_raises_type_error(self):
        """Passing a list instead of ndarray raises TypeError."""
        with self.assertRaises(TypeError):
            self.analyse_inliers(
                inlier_mask=list(self.mask),        # list, not ndarray
                source_points=self.src_pts,
                reference_points=self.ref_pts,
            )

    # ------------------------------------------------------------------
    # Test 5: Empty input
    # ------------------------------------------------------------------

    def test_empty_input(self):
        """Zero-length arrays return an InlierReport with all-zero counts."""
        empty_mask = np.array([], dtype=bool)
        empty_pts = np.empty((0, 2), dtype=np.float32)
        report = self.analyse_inliers(
            inlier_mask=empty_mask,
            source_points=empty_pts,
            reference_points=empty_pts,
        )
        self.assertIsInstance(report, self.InlierReport)
        self.assertEqual(report.inlier_count, 0)
        self.assertEqual(report.outlier_count, 0)
        self.assertEqual(report.total_count, 0)
        self.assertAlmostEqual(report.inlier_ratio, 0.0)
        self.assertAlmostEqual(report.spatial_coverage, 0.0)
        self.assertEqual(report.inlier_source_points.shape, (0, 2))
        self.assertEqual(report.outlier_source_points.shape, (0, 2))

    # ------------------------------------------------------------------
    # Test 6: Spatial coverage
    # ------------------------------------------------------------------

    def test_spatial_coverage_range(self):
        """spatial_coverage is always in [0, 1]."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertGreaterEqual(report.spatial_coverage, 0.0)
        self.assertLessEqual(report.spatial_coverage, 1.0)

    def test_spatial_coverage_uniform_points_is_high(self):
        """Uniformly spread points should produce high spatial coverage."""
        rng = np.random.RandomState(42)
        # Create a grid of points spread across [0, 200] x [0, 200]
        n = 64
        xs = np.tile(np.linspace(10, 190, 8), 8).astype(np.float32)
        ys = np.repeat(np.linspace(10, 190, 8), 8).astype(np.float32)
        src = np.column_stack([xs, ys])
        ref = src + 2.0
        mask = np.ones(n, dtype=bool)

        report = self.analyse_inliers(
            inlier_mask=mask,
            source_points=src,
            reference_points=ref,
            grid_divisions=4,
        )
        # Uniform grid points should produce coverage close to 1.0
        self.assertGreater(report.spatial_coverage, 0.80)

    def test_spatial_coverage_clustered_points_is_low(self):
        """All inliers clustered in one corner → low spatial coverage."""
        rng = np.random.RandomState(7)
        n = 20
        # All points tightly clustered near (5, 5)
        src = rng.normal(5, 0.5, (n, 2)).astype(np.float32)
        src = np.clip(src, 0, 200)
        ref = src + 1.0
        mask = np.ones(n, dtype=bool)

        report = self.analyse_inliers(
            inlier_mask=mask,
            source_points=src,
            reference_points=ref,
            image_shape=(200, 200),  # full image extent
            grid_divisions=4,
        )
        # Clustered in one cell of 16 → low coverage
        self.assertLess(report.spatial_coverage, 0.50)

    def test_spatial_coverage_with_image_shape(self):
        """image_shape parameter is accepted and affects coverage computation."""
        report_no_shape = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        report_with_shape = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
            image_shape=(256, 256),
        )
        # Both should return valid InlierReport
        self.assertIsInstance(report_no_shape, self.InlierReport)
        self.assertIsInstance(report_with_shape, self.InlierReport)
        # With a larger image extent, coverage may be lower (points are
        # smaller fraction of grid) — just verify it's in range
        self.assertGreaterEqual(report_with_shape.spatial_coverage, 0.0)
        self.assertLessEqual(report_with_shape.spatial_coverage, 1.0)

    # ------------------------------------------------------------------
    # Test 7: Consistency checks on InlierReport fields
    # ------------------------------------------------------------------

    def test_counts_are_consistent(self):
        """inlier_count + outlier_count == total_count for any mask."""
        for seed in [0, 1, 2, 3]:
            rng = np.random.RandomState(seed)
            mask = rng.random(self.n) > 0.4
            report = self.analyse_inliers(
                inlier_mask=mask,
                source_points=self.src_pts,
                reference_points=self.ref_pts,
            )
            self.assertEqual(
                report.inlier_count + report.outlier_count,
                report.total_count,
                msg=f"seed={seed}: count consistency failed",
            )
            if report.total_count > 0:
                self.assertAlmostEqual(
                    report.inlier_ratio,
                    report.inlier_count / report.total_count,
                    places=6,
                )

    def test_inlier_mask_preserved(self):
        """InlierReport.inlier_mask matches the input mask exactly."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        np.testing.assert_array_equal(report.inlier_mask, self.mask)

    def test_backward_compat_alias(self):
        """spatial_uniformity property returns same value as spatial_coverage."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertEqual(report.spatial_uniformity, report.spatial_coverage)

    def test_field_types_on_success(self):
        """All InlierReport fields have correct types on a normal run."""
        report = self.analyse_inliers(
            inlier_mask=self.mask,
            source_points=self.src_pts,
            reference_points=self.ref_pts,
        )
        self.assertIsInstance(report.inlier_mask, np.ndarray)
        self.assertEqual(report.inlier_mask.dtype, bool)
        self.assertIsInstance(report.inlier_source_points, np.ndarray)
        self.assertIsInstance(report.inlier_reference_points, np.ndarray)
        self.assertIsInstance(report.outlier_source_points, np.ndarray)
        self.assertIsInstance(report.outlier_reference_points, np.ndarray)
        self.assertIsInstance(report.inlier_count, int)
        self.assertIsInstance(report.outlier_count, int)
        self.assertIsInstance(report.total_count, int)
        self.assertIsInstance(report.inlier_ratio, float)
        self.assertIsInstance(report.spatial_coverage, float)
        self.assertIsInstance(report.diagnostics, dict)

    # ------------------------------------------------------------------
    # Test 8: End-to-end — Phase 2 → Phase 3 integration
    # ------------------------------------------------------------------

    def test_end_to_end_phase2_to_phase3(self):
        """Phase 2 EstimationResult feeds directly into Phase 3 analyse_inliers()."""
        from src.validation.geometric_estimation import (
            GeometricEstimator, GeometricEstimatorConfig, TransformModel,
        )
        correspondence, _ = _make_transformed_correspondence(
            n_inliers=40, n_outliers=10, seed=11,
        )
        cfg = GeometricEstimatorConfig(
            model=TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        est_result = GeometricEstimator(config=cfg).estimate(correspondence)
        self.assertTrue(est_result.success, f"Phase 2 failed: {est_result.reason}")

        report = self.analyse_inliers(
            inlier_mask=est_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
        )
        self.assertIsInstance(report, self.InlierReport)
        # Counts from Phase 2 and Phase 3 must agree
        self.assertEqual(report.inlier_count, est_result.num_inliers)
        self.assertEqual(report.outlier_count, est_result.num_outliers)
        self.assertEqual(report.total_count, est_result.total_matches)
        self.assertAlmostEqual(report.inlier_ratio, est_result.inlier_ratio, places=5)
        # Separated arrays must reconstruct the original point set
        all_src = np.vstack([report.inlier_source_points, report.outlier_source_points])
        self.assertEqual(all_src.shape[0], report.total_count)



# ---------------------------------------------------------------------------


class TestReprojection(unittest.TestCase):
    """Tests for ReprojectionReport and compute_reprojection_errors()."""

    def setUp(self):
        from src.validation.reprojection import (
            ReprojectionReport,
            compute_reprojection_errors,
        )
        self.ReprojectionReport = ReprojectionReport
        self.compute_reprojection_errors = compute_reprojection_errors

        n = 30
        self.src_pts = np.random.uniform(10, 190, (n, 2)).astype(np.float32)
        self.ref_pts = self.src_pts + 3.0
        self.mask = np.ones(n, dtype=bool)
        self.H = np.eye(3, dtype=np.float64)  # Identity homography

    def test_compute_reprojection_errors_is_callable(self):
        """compute_reprojection_errors() is callable and returns ReprojectionReport."""
        report = self.compute_reprojection_errors(
            source_points=self.src_pts,
            reference_points=self.ref_pts,
            transform_matrix=self.H,
            inlier_mask=self.mask,
        )
        self.assertIsInstance(report, self.ReprojectionReport)

    def test_reprojection_report_dataclass(self):
        """ReprojectionReport can be constructed directly with new fields."""
        n = 30
        errors = np.random.uniform(0, 5, n).astype(np.float64)
        report = self.ReprojectionReport(
            per_match_errors=errors,
            per_match_sym_errors=None,
            inlier_errors=errors,
            outlier_errors=np.empty(0, dtype=np.float64),
            inlier_mean_error=float(errors.mean()),
            inlier_median_error=float(np.median(errors)),
            inlier_rmse=float(np.sqrt(np.mean(errors ** 2))),
            inlier_max_error=float(errors.max()),
            outlier_mean_error=0.0,
            overall_mean_error=float(errors.mean()),
            overall_median_error=float(np.median(errors)),
            overall_rmse=float(np.sqrt(np.mean(errors ** 2))),
            overall_max_error=float(errors.max()),
        )
        self.assertEqual(len(report.per_match_errors), n)
        self.assertIsNone(report.per_match_sym_errors)
        self.assertGreaterEqual(report.overall_rmse, 0.0)


# ---------------------------------------------------------------------------
# Reprojection algorithmic tests (Phase 4 — implemented)
# ---------------------------------------------------------------------------


class TestReprojectionImpl(unittest.TestCase):
    """Algorithmic tests for compute_reprojection_errors() (M3 Phase 4).

    All tests are self-contained and require no external data files.
    """

    def setUp(self):
        from src.validation.reprojection import (
            ReprojectionReport, compute_reprojection_errors, _project_points,
        )
        self.ReprojectionReport = ReprojectionReport
        self.compute_reprojection_errors = compute_reprojection_errors
        self._project_points = _project_points

        # Known affine: 5° rotation + (10, -5) translation
        angle = np.deg2rad(5.0)
        c, s = np.cos(angle), np.sin(angle)
        self.affine_M = np.array([
            [c, -s, 10.0],
            [s,  c, -5.0],
        ], dtype=np.float64)
        # Near-affine homography (last row = [0,0,1])
        self.homography_H = np.eye(3, dtype=np.float64)
        self.homography_H[:2, :] = self.affine_M

        rng = np.random.RandomState(42)
        self.n = 30
        self.src = rng.uniform(20, 180, (self.n, 2)).astype(np.float32)
        # Apply known affine to get exact reference points
        A, t = self.affine_M[:, :2], self.affine_M[:, 2]
        self.ref_affine = (self.src.astype(np.float64) @ A.T + t).astype(np.float32)
        self.all_inliers = np.ones(self.n, dtype=bool)

    # ------------------------------------------------------------------
    # Test 1: Identity transform → zero reprojection error
    # ------------------------------------------------------------------

    def test_identity_affine_zero_error(self):
        """Identity (2,3) affine matrix on src=ref gives exactly 0 error."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        rng = np.random.RandomState(0)
        pts = rng.uniform(10, 190, (20, 2)).astype(np.float32)

        report = self.compute_reprojection_errors(
            source_points=pts,
            reference_points=pts,   # ref = src → error should be ~0
            transform_matrix=I,
            model_type="affine",
        )
        np.testing.assert_allclose(
            report.per_match_errors, 0.0, atol=1e-10,
            err_msg="Identity affine must yield zero reprojection errors.",
        )
        self.assertAlmostEqual(report.overall_rmse, 0.0, places=10)
        self.assertAlmostEqual(report.overall_max_error, 0.0, places=10)

    def test_identity_homography_zero_error(self):
        """Identity (3,3) homography matrix on src=ref gives exactly 0 error."""
        H = np.eye(3, dtype=np.float64)
        rng = np.random.RandomState(1)
        pts = rng.uniform(10, 190, (20, 2)).astype(np.float32)

        report = self.compute_reprojection_errors(
            source_points=pts,
            reference_points=pts,
            transform_matrix=H,
            model_type="homography",
        )
        np.testing.assert_allclose(
            report.per_match_errors, 0.0, atol=1e-10,
        )

    # ------------------------------------------------------------------
    # Test 2: Known affine transform → near-zero reprojection error
    # ------------------------------------------------------------------

    def test_affine_known_transform_near_zero_error(self):
        """When ref = affine(src), reprojection error should be near zero."""
        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=self.ref_affine,
            transform_matrix=self.affine_M,
            inlier_mask=self.all_inliers,
            model_type="affine",
        )
        self.assertIsInstance(report, self.ReprojectionReport)
        # Forward projection error should be near machine epsilon
        np.testing.assert_allclose(
            report.per_match_errors, 0.0, atol=1e-5,
            err_msg="Reprojection errors should be ~0 for exact transform.",
        )
        self.assertAlmostEqual(report.inlier_mean_error, 0.0, places=5)
        self.assertAlmostEqual(report.inlier_rmse, 0.0, places=5)
        self.assertAlmostEqual(report.overall_rmse, 0.0, places=5)

    # ------------------------------------------------------------------
    # Test 3: Known homography transform → near-zero reprojection error
    # ------------------------------------------------------------------

    def test_homography_known_transform_near_zero_error(self):
        """When ref = H(src), reprojection error should be near zero."""
        H = self.homography_H
        # Generate reference from exact homography projection
        src_h = np.hstack([self.src.astype(np.float64), np.ones((self.n, 1))])
        ref_h = (H @ src_h.T).T
        ref_exact = (ref_h[:, :2] / ref_h[:, 2:3]).astype(np.float32)

        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=ref_exact,
            transform_matrix=H,
            inlier_mask=self.all_inliers,
            model_type="homography",
        )
        np.testing.assert_allclose(
            report.per_match_errors, 0.0, atol=1e-5,
        )
        self.assertAlmostEqual(report.overall_rmse, 0.0, places=5)

    # ------------------------------------------------------------------
    # Test 4: Non-zero reprojection error with known magnitude
    # ------------------------------------------------------------------

    def test_known_nonzero_error(self):
        """Translate reference by known amount and verify mean error matches."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        dx, dy = 3.0, 4.0   # displacement magnitude = 5.0
        rng = np.random.RandomState(5)
        pts = rng.uniform(10, 190, (self.n, 2)).astype(np.float32)
        # ref is uniformly shifted by (dx, dy) → error = sqrt(dx^2 + dy^2) = 5
        ref_shifted = (pts + np.array([dx, dy], dtype=np.float32))

        report = self.compute_reprojection_errors(
            source_points=pts,
            reference_points=ref_shifted,
            transform_matrix=I,
            model_type="affine",
        )
        expected_error = float(np.sqrt(dx**2 + dy**2))
        np.testing.assert_allclose(
            report.per_match_errors, expected_error, atol=1e-5,
            err_msg=f"All errors should equal {expected_error}.",
        )
        self.assertAlmostEqual(report.overall_mean_error, expected_error, places=5)
        self.assertAlmostEqual(report.overall_median_error, expected_error, places=5)
        self.assertAlmostEqual(report.overall_rmse, expected_error, places=5)
        self.assertAlmostEqual(report.overall_max_error, expected_error, places=5)

    # ------------------------------------------------------------------
    # Test 5: Mean / Median / RMSE / Max calculations
    # ------------------------------------------------------------------

    def test_stat_calculations_all_inliers(self):
        """Mean, median, RMSE, max are computed correctly when all are inliers."""
        # Use identity transform so errors = distances in original space
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        rng = np.random.RandomState(99)
        src = rng.uniform(0, 100, (50, 2)).astype(np.float64)
        # Non-uniform displacement to create varied errors
        displacements = rng.uniform(0, 10, (50, 2)).astype(np.float64)
        ref = (src + displacements).astype(np.float64)
        expected_errors = np.sqrt((displacements**2).sum(axis=1))

        report = self.compute_reprojection_errors(
            source_points=src.astype(np.float32),
            reference_points=ref.astype(np.float32),
            transform_matrix=I,
            model_type="affine",
        )
        np.testing.assert_allclose(report.per_match_errors, expected_errors, atol=1e-5)
        self.assertAlmostEqual(report.overall_mean_error, float(expected_errors.mean()), places=5)
        self.assertAlmostEqual(report.overall_median_error, float(np.median(expected_errors)), places=5)
        self.assertAlmostEqual(
            report.overall_rmse, float(np.sqrt(np.mean(expected_errors**2))), places=5
        )
        self.assertAlmostEqual(report.overall_max_error, float(expected_errors.max()), places=5)

    def test_inlier_outlier_split_stats(self):
        """Inlier and outlier stats are computed over the correct subsets."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        n_in, n_out = 20, 10
        n = n_in + n_out

        # Inliers: small error (1.0 px)
        src_in = np.zeros((n_in, 2), dtype=np.float64)
        ref_in = np.full((n_in, 2), 1.0 / np.sqrt(2), dtype=np.float64)  # error=1
        # Outliers: large error (10.0 px)
        src_out = np.zeros((n_out, 2), dtype=np.float64)
        ref_out = np.full((n_out, 2), 10.0 / np.sqrt(2), dtype=np.float64)  # error=10

        src = np.vstack([src_in, src_out]).astype(np.float32)
        ref = np.vstack([ref_in, ref_out]).astype(np.float32)
        mask = np.array([True] * n_in + [False] * n_out, dtype=bool)

        report = self.compute_reprojection_errors(
            source_points=src,
            reference_points=ref,
            transform_matrix=I,
            inlier_mask=mask,
            model_type="affine",
        )
        self.assertAlmostEqual(report.inlier_mean_error, 1.0, places=4)
        self.assertAlmostEqual(report.outlier_mean_error, 10.0, places=4)
        self.assertEqual(len(report.inlier_errors), n_in)
        self.assertEqual(len(report.outlier_errors), n_out)
        # Overall stats cover all N
        self.assertEqual(len(report.per_match_errors), n)

    # ------------------------------------------------------------------
    # Test 6: Empty input
    # ------------------------------------------------------------------

    def test_empty_input(self):
        """Zero points return all-zero ReprojectionReport."""
        empty = np.empty((0, 2), dtype=np.float32)
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        report = self.compute_reprojection_errors(
            source_points=empty,
            reference_points=empty,
            transform_matrix=I,
            model_type="affine",
        )
        self.assertIsInstance(report, self.ReprojectionReport)
        self.assertEqual(len(report.per_match_errors), 0)
        self.assertAlmostEqual(report.overall_rmse, 0.0)
        self.assertAlmostEqual(report.overall_mean_error, 0.0)

    # ------------------------------------------------------------------
    # Test 7: Invalid transformation matrix
    # ------------------------------------------------------------------

    def test_wrong_homography_shape_raises_value_error(self):
        """(2, 3) matrix for homography model raises ValueError."""
        bad_M = np.eye(2, 3, dtype=np.float64)  # wrong for homography
        with self.assertRaises(ValueError) as ctx:
            self.compute_reprojection_errors(
                source_points=self.src,
                reference_points=self.ref_affine,
                transform_matrix=bad_M,
                model_type="homography",
            )
        self.assertIn("3, 3", str(ctx.exception))

    def test_wrong_affine_shape_raises_value_error(self):
        """(3, 3) matrix for affine model raises ValueError."""
        bad_M = np.eye(3, dtype=np.float64)  # wrong for affine
        with self.assertRaises(ValueError) as ctx:
            self.compute_reprojection_errors(
                source_points=self.src,
                reference_points=self.ref_affine,
                transform_matrix=bad_M,
                model_type="affine",
            )
        self.assertIn("2, 3", str(ctx.exception))

    def test_unknown_model_type_raises_value_error(self):
        """Unknown model_type string raises ValueError."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            self.compute_reprojection_errors(
                source_points=self.src,
                reference_points=self.ref_affine,
                transform_matrix=I,
                model_type="invalid_model",
            )

    def test_nan_in_transform_matrix_raises_value_error(self):
        """NaN in transform_matrix raises ValueError."""
        bad = np.array([[1, 0, 0], [0, np.nan, 0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            self.compute_reprojection_errors(
                source_points=self.src,
                reference_points=self.ref_affine,
                transform_matrix=bad,
                model_type="affine",
            )

    # ------------------------------------------------------------------
    # Test 8: NaN / Inf in source or reference points
    # ------------------------------------------------------------------

    def test_nan_in_source_points_yields_nan_errors(self):
        """NaN in source_points propagates to NaN in per-match errors for that row."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        src_with_nan = self.src.copy().astype(np.float64)
        src_with_nan[0, 0] = np.nan  # corrupt first point

        report = self.compute_reprojection_errors(
            source_points=src_with_nan.astype(np.float32),
            reference_points=self.ref_affine,
            transform_matrix=I,
            model_type="affine",
        )
        # First error must be NaN; overall stats should exclude it
        self.assertTrue(np.isnan(report.per_match_errors[0]))
        # Aggregate stats should still be finite (NaN excluded)
        self.assertTrue(np.isfinite(report.overall_mean_error))
        self.assertTrue(np.isfinite(report.overall_rmse))

    def test_inf_in_reference_points_yields_inf_or_nan_errors(self):
        """Inf in reference_points propagates to Inf/NaN in per-match errors."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        ref_with_inf = self.ref_affine.copy().astype(np.float64)
        ref_with_inf[5] = np.inf

        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=ref_with_inf.astype(np.float32),
            transform_matrix=I,
            model_type="affine",
        )
        # Row 5 error should be Inf or NaN
        self.assertFalse(np.isfinite(report.per_match_errors[5]))

    # ------------------------------------------------------------------
    # Test 9: Symmetric reprojection error
    # ------------------------------------------------------------------

    def test_symmetric_reprojection_computed_for_homography(self):
        """compute_symmetric=True populates per_match_sym_errors for homography."""
        H = self.homography_H
        src_h = np.hstack([self.src.astype(np.float64), np.ones((self.n, 1))])
        ref_h = (H @ src_h.T).T
        ref_exact = (ref_h[:, :2] / ref_h[:, 2:3]).astype(np.float32)

        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=ref_exact,
            transform_matrix=H,
            model_type="homography",
            compute_symmetric=True,
        )
        self.assertIsNotNone(report.per_match_sym_errors)
        self.assertEqual(len(report.per_match_sym_errors), self.n)
        # Symmetric errors ≥ 0
        valid_sym = report.per_match_sym_errors[np.isfinite(report.per_match_sym_errors)]
        self.assertTrue((valid_sym >= 0).all())

    def test_no_symmetric_when_not_requested(self):
        """compute_symmetric=False (default) leaves per_match_sym_errors as None."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=self.ref_affine,
            transform_matrix=I,
            model_type="affine",
            compute_symmetric=False,
        )
        self.assertIsNone(report.per_match_sym_errors)

    # ------------------------------------------------------------------
    # Test 10: Result field types and consistency
    # ------------------------------------------------------------------

    def test_result_field_types(self):
        """All ReprojectionReport fields have correct types."""
        mask = np.array([True] * 20 + [False] * 10, dtype=bool)
        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=self.ref_affine,
            transform_matrix=self.affine_M,
            inlier_mask=mask,
            model_type="affine",
        )
        self.assertIsInstance(report.per_match_errors, np.ndarray)
        self.assertEqual(report.per_match_errors.dtype, np.float64)
        self.assertIsInstance(report.inlier_errors, np.ndarray)
        self.assertIsInstance(report.outlier_errors, np.ndarray)
        self.assertIsInstance(report.inlier_mean_error, float)
        self.assertIsInstance(report.inlier_median_error, float)
        self.assertIsInstance(report.inlier_rmse, float)
        self.assertIsInstance(report.inlier_max_error, float)
        self.assertIsInstance(report.outlier_mean_error, float)
        self.assertIsInstance(report.overall_mean_error, float)
        self.assertIsInstance(report.overall_median_error, float)
        self.assertIsInstance(report.overall_rmse, float)
        self.assertIsInstance(report.overall_max_error, float)
        self.assertIsInstance(report.diagnostics, dict)
        # Sizes
        self.assertEqual(len(report.per_match_errors), self.n)
        self.assertEqual(len(report.inlier_errors), 20)
        self.assertEqual(len(report.outlier_errors), 10)

    def test_no_inlier_mask_treats_all_as_inliers(self):
        """When inlier_mask=None, all matches are treated as inliers."""
        I = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        report = self.compute_reprojection_errors(
            source_points=self.src,
            reference_points=self.ref_affine,
            transform_matrix=I,
            inlier_mask=None,
            model_type="affine",
        )
        # inlier_errors covers all points
        self.assertEqual(len(report.inlier_errors), self.n)
        self.assertEqual(len(report.outlier_errors), 0)
        # Overall and inlier stats should be identical
        self.assertAlmostEqual(report.overall_rmse, report.inlier_rmse, places=10)

    # ------------------------------------------------------------------
    # Test 11: Phase 2 → Phase 3 → Phase 4 integration
    # ------------------------------------------------------------------

    def test_end_to_end_phase2_phase3_phase4(self):
        """Full pipeline: GeometricEstimator → analyse_inliers → compute_reprojection_errors."""
        from src.validation.geometric_estimation import (
            GeometricEstimator, GeometricEstimatorConfig, TransformModel,
        )
        from src.validation.inlier_analysis import analyse_inliers

        correspondence, _ = _make_transformed_correspondence(
            n_inliers=40, n_outliers=10, seed=77,
        )

        # Phase 2
        cfg = GeometricEstimatorConfig(
            model=TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        est = GeometricEstimator(config=cfg).estimate(correspondence)
        self.assertTrue(est.success, f"Phase 2 failed: {est.reason}")

        # Phase 3
        p3 = analyse_inliers(
            inlier_mask=est.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
        )

        # Phase 4
        report = self.compute_reprojection_errors(
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            transform_matrix=est.transform_matrix,
            inlier_mask=est.inlier_mask,
            model_type="affine",
        )
        self.assertIsInstance(report, self.ReprojectionReport)
        # Inlier error count must match Phase 3 inlier count
        self.assertEqual(len(report.inlier_errors), p3.inlier_count)
        self.assertEqual(len(report.outlier_errors), p3.outlier_count)
        # Inlier reprojection error must be small (RANSAC threshold was 3 px)
        self.assertLess(report.inlier_mean_error, 3.5)
        self.assertGreaterEqual(report.overall_rmse, 0.0)


# ---------------------------------------------------------------------------
# Registration helpers and tests
# ---------------------------------------------------------------------------


def _make_synthetic_images(
    height: int = 200,
    width: int = 200,
    seed: int = 0,
) -> tuple:
    """Create a pair of uint8 synthetic grayscale images and a (2,3) affine matrix.

    Returns:
        Tuple (source_image, reference_image, affine_M) where:
            source_image  : (H, W) uint8 ndarray
            reference_image: (H, W) uint8 ndarray (random but same shape)
            affine_M      : (2, 3) float64 affine matrix (5° rotation + 8px shift)
    """
    rng = np.random.RandomState(seed)
    # Source: gradient pattern so warped content is visually distinct from fill
    src = np.zeros((height, width), dtype=np.uint8)
    for i in range(0, height, 20):
        src[i:i + 10, :] = 128
    src[40:160, 40:160] = 200  # bright square in centre

    ref = rng.randint(50, 200, (height, width), dtype=np.uint8)

    # 5° rotation + (8, -4) translation
    angle = np.deg2rad(5.0)
    c, s = np.cos(angle), np.sin(angle)
    M = np.array([
        [c, -s, 8.0],
        [s,  c, -4.0],
    ], dtype=np.float64)
    return src, ref, M


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    """Tests for RegistrationConfig, RegistrationResult, and register_image()."""

    def setUp(self):
        from src.validation.registration import (
            RegistrationConfig,
            RegistrationResult,
            register_image,
        )
        self.RegistrationConfig = RegistrationConfig
        self.RegistrationResult = RegistrationResult
        self.register_image = register_image

        self.img_src, self.img_ref, self.M = _make_synthetic_images()
        # Extend to (3,3) homography form from (2,3) affine
        self.H = np.vstack([self.M, [0.0, 0.0, 1.0]])

    def test_config_defaults(self):
        """RegistrationConfig has sensible defaults."""
        cfg = self.RegistrationConfig()
        self.assertEqual(cfg.model_type, "homography")
        self.assertTrue(cfg.compute_overlap)
        self.assertIsNone(cfg.output_size)

    def test_register_image_is_callable(self):
        """register_image() is callable and returns a RegistrationResult."""
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.H,
            reference_shape=self.img_ref.shape[:2],
        )
        self.assertIsInstance(result, self.RegistrationResult)

    def test_registration_result_dataclass(self):
        """RegistrationResult can be constructed directly with all fields."""
        dummy_warped = np.zeros_like(self.img_ref)
        result = self.RegistrationResult(
            registered_image=dummy_warped,
            transform_matrix=self.H,
            output_size=(200, 200),
            overlap_bbox=(10, 10, 180, 180),
            estimated_location=None,
            quality_flags={"warp_succeeded": True, "overlap_nonzero": True},
        )
        self.assertEqual(result.registered_image.shape, self.img_ref.shape)
        self.assertEqual(result.output_size, (200, 200))
        self.assertTrue(result.quality_flags["warp_succeeded"])


# ---------------------------------------------------------------------------
# Registration algorithmic tests (Phase 5 — implemented)
# ---------------------------------------------------------------------------


class TestRegistrationImpl(unittest.TestCase):
    """Algorithmic tests for register_image() (M3 Phase 5)."""

    def setUp(self):
        from src.validation.registration import (
            RegistrationConfig, RegistrationResult, register_image,
            _compute_overlap_bbox, _estimate_geographic_location,
        )
        self.RegistrationConfig = RegistrationConfig
        self.RegistrationResult = RegistrationResult
        self.register_image = register_image
        self._compute_overlap_bbox = _compute_overlap_bbox
        self._estimate_geographic_location = _estimate_geographic_location

        self.img_src, self.img_ref, self.affine_M = _make_synthetic_images()
        self.H = np.vstack([self.affine_M, [0.0, 0.0, 1.0]])
        self.ref_shape = self.img_ref.shape[:2]   # (H, W)

    # ------------------------------------------------------------------
    # Test 1: Affine warp produces correct output shape
    # ------------------------------------------------------------------

    def test_affine_warp_output_shape(self):
        """warpAffine output matches reference_shape."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertIsInstance(result, self.RegistrationResult)
        self.assertEqual(result.registered_image.shape, self.img_ref.shape)
        self.assertEqual(result.output_size, (self.ref_shape[1], self.ref_shape[0]))
        self.assertTrue(result.quality_flags["warp_succeeded"])

    def test_affine_warp_preserves_dtype(self):
        """Warped image dtype matches source image dtype."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertEqual(result.registered_image.dtype, self.img_src.dtype)

    # ------------------------------------------------------------------
    # Test 2: Homography warp produces correct output shape
    # ------------------------------------------------------------------

    def test_homography_warp_output_shape(self):
        """warpPerspective output matches reference_shape."""
        cfg = self.RegistrationConfig(model_type="homography")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.H,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertEqual(result.registered_image.shape, self.img_ref.shape)
        self.assertTrue(result.quality_flags["warp_succeeded"])

    # ------------------------------------------------------------------
    # Test 3: Identity transform → warped image == source
    # ------------------------------------------------------------------

    def test_identity_affine_preserves_image(self):
        """Identity (2,3) affine leaves source image unchanged."""
        I_affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=I_affine,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        # With identity, registered image should equal source (same shape)
        np.testing.assert_array_equal(
            result.registered_image, self.img_src,
            err_msg="Identity affine must produce pixel-identical output.",
        )

    def test_identity_homography_preserves_image(self):
        """Identity (3,3) homography leaves source image unchanged."""
        I_hom = np.eye(3, dtype=np.float64)
        cfg = self.RegistrationConfig(model_type="homography")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=I_hom,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        np.testing.assert_array_equal(result.registered_image, self.img_src)

    # ------------------------------------------------------------------
    # Test 4: Custom output_size override
    # ------------------------------------------------------------------

    def test_custom_output_size(self):
        """config.output_size overrides reference_shape for canvas dimensions."""
        target_w, target_h = 320, 240
        cfg = self.RegistrationConfig(
            model_type="affine",
            output_size=(target_w, target_h),
        )
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertEqual(result.registered_image.shape[0], target_h)
        self.assertEqual(result.registered_image.shape[1], target_w)
        self.assertEqual(result.output_size, (target_w, target_h))

    # ------------------------------------------------------------------
    # Test 5: Overlap bounding box
    # ------------------------------------------------------------------

    def test_overlap_bbox_present_on_valid_warp(self):
        """overlap_bbox is not None when the warped image has non-fill pixels."""
        cfg = self.RegistrationConfig(model_type="affine", compute_overlap=True)
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertIsNotNone(result.overlap_bbox)
        x, y, w, h = result.overlap_bbox
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)
        # Bbox must be inside canvas
        out_w, out_h = result.output_size
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, out_w)
        self.assertLessEqual(y + h, out_h)

    def test_overlap_bbox_none_when_disabled(self):
        """overlap_bbox is None when compute_overlap=False."""
        cfg = self.RegistrationConfig(model_type="affine", compute_overlap=False)
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertIsNone(result.overlap_bbox)

    def test_overlap_bbox_helper_none_on_black_image(self):
        """_compute_overlap_bbox returns None for an all-zero (fill) image."""
        black = np.zeros((100, 100), dtype=np.uint8)
        self.assertIsNone(self._compute_overlap_bbox(black, fill_value=0))

    def test_overlap_bbox_helper_detects_valid_region(self):
        """_compute_overlap_bbox correctly locates a bright rectangle."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:60, 30:80] = 255
        bbox = self._compute_overlap_bbox(img, fill_value=0)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertEqual(y, 20)
        self.assertEqual(x, 30)
        self.assertEqual(h, 40)
        self.assertEqual(w, 50)

    # ------------------------------------------------------------------
    # Test 6: Invalid inputs raise appropriate errors
    # ------------------------------------------------------------------

    def test_non_array_source_raises_type_error(self):
        """Passing a list as source_image raises TypeError."""
        with self.assertRaises(TypeError):
            self.register_image(
                source_image=[[0, 1], [2, 3]],  # list, not ndarray
                transform_matrix=self.H,
                reference_shape=self.ref_shape,
            )

    def test_wrong_homography_shape_raises_value_error(self):
        """(2,3) matrix for homography raises ValueError."""
        cfg = self.RegistrationConfig(model_type="homography")
        with self.assertRaises(ValueError):
            self.register_image(
                source_image=self.img_src,
                transform_matrix=self.affine_M,   # (2,3), wrong for homography
                reference_shape=self.ref_shape,
                config=cfg,
            )

    def test_wrong_affine_shape_raises_value_error(self):
        """(3,3) matrix for affine raises ValueError."""
        cfg = self.RegistrationConfig(model_type="affine")
        with self.assertRaises(ValueError):
            self.register_image(
                source_image=self.img_src,
                transform_matrix=self.H,   # (3,3), wrong for affine
                reference_shape=self.ref_shape,
                config=cfg,
            )

    def test_unknown_model_type_raises_value_error(self):
        """Unknown model_type string raises ValueError."""
        cfg = self.RegistrationConfig(model_type="unknown_model")
        with self.assertRaises(ValueError):
            self.register_image(
                source_image=self.img_src,
                transform_matrix=self.affine_M,
                reference_shape=self.ref_shape,
                config=cfg,
            )

    def test_nan_transform_raises_value_error(self):
        """NaN in transform_matrix raises ValueError."""
        bad = self.affine_M.copy()
        bad[0, 0] = np.nan
        cfg = self.RegistrationConfig(model_type="affine")
        with self.assertRaises(ValueError):
            self.register_image(
                source_image=self.img_src,
                transform_matrix=bad,
                reference_shape=self.ref_shape,
                config=cfg,
            )

    def test_1d_source_raises_value_error(self):
        """1-D source_image raises ValueError."""
        bad_img = np.zeros(100, dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.register_image(
                source_image=bad_img,
                transform_matrix=self.H,
                reference_shape=self.ref_shape,
            )

    # ------------------------------------------------------------------
    # Test 7: Geographic localisation
    # ------------------------------------------------------------------

    def test_geographic_location_populated_when_gsd_provided(self):
        """estimated_location is not None when reference_gsd is provided."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
            reference_gsd=0.5,  # 0.5 m/px
        )
        self.assertIsNotNone(result.estimated_location)
        loc = result.estimated_location
        self.assertIn("reference_pixel_x", loc)
        self.assertIn("reference_pixel_y", loc)
        self.assertIn("estimated_x_offset_m", loc)
        self.assertIn("estimated_y_offset_m", loc)
        self.assertAlmostEqual(loc["gsd_m_per_px"], 0.5)

    def test_geographic_location_none_when_no_gsd(self):
        """estimated_location is None when reference_gsd is not provided."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertIsNone(result.estimated_location)

    def test_gsd_helper_identity_projects_centre(self):
        """_estimate_geographic_location: identity homography maps centre to itself."""
        h, w = 200, 200
        loc = self._estimate_geographic_location(
            np.eye(3, dtype=np.float64),
            source_shape=(h, w),
            reference_gsd=1.0,
            model_type="homography",
        )
        self.assertAlmostEqual(loc["reference_pixel_x"], w / 2, places=6)
        self.assertAlmostEqual(loc["reference_pixel_y"], h / 2, places=6)
        self.assertAlmostEqual(loc["estimated_x_offset_m"], w / 2, places=6)

    # ------------------------------------------------------------------
    # Test 8: Quality flags
    # ------------------------------------------------------------------

    def test_quality_flags_structure(self):
        """quality_flags contains the three expected boolean keys."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        for key in ("warp_succeeded", "overlap_nonzero", "overlap_sufficient"):
            self.assertIn(key, result.quality_flags)
            self.assertIsInstance(result.quality_flags[key], bool)
        self.assertTrue(result.quality_flags["warp_succeeded"])
        self.assertTrue(result.quality_flags["overlap_nonzero"])

    # ------------------------------------------------------------------
    # Test 9: Diagnostics dict
    # ------------------------------------------------------------------

    def test_diagnostics_populated(self):
        """diagnostics dict is returned with key metadata."""
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        diag = result.diagnostics
        self.assertIn("model_type", diag)
        self.assertIn("fill_ratio", diag)
        self.assertIn("overlap_ratio", diag)
        self.assertGreaterEqual(diag["fill_ratio"], 0.0)
        self.assertLessEqual(diag["fill_ratio"], 1.0)

    # ------------------------------------------------------------------
    # Test 10: Colour (3-channel) image
    # ------------------------------------------------------------------

    def test_colour_image_affine_warp(self):
        """register_image works with (H, W, 3) colour images."""
        colour_src = np.stack([self.img_src] * 3, axis=2)  # (H, W, 3)
        cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=colour_src,
            transform_matrix=self.affine_M,
            reference_shape=self.ref_shape,
            config=cfg,
        )
        self.assertEqual(result.registered_image.ndim, 3)
        self.assertEqual(result.registered_image.shape[2], 3)
        self.assertEqual(
            result.registered_image.shape[:2], (self.ref_shape[0], self.ref_shape[1])
        )

    # ------------------------------------------------------------------
    # Test 11: End-to-end Phase 2 → Phase 5
    # ------------------------------------------------------------------

    def test_end_to_end_phase2_to_phase5(self):
        """Full pipeline: GeometricEstimator → register_image."""
        from src.validation.geometric_estimation import (
            GeometricEstimator, GeometricEstimatorConfig, TransformModel,
        )
        correspondence, true_M = _make_transformed_correspondence(
            n_inliers=50, n_outliers=5, seed=33,
        )
        cfg_est = GeometricEstimatorConfig(
            model=TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4,
            min_inlier_ratio=0.0,
        )
        est = GeometricEstimator(config=cfg_est).estimate(correspondence)
        self.assertTrue(est.success, f"Phase 2 failed: {est.reason}")

        reg_cfg = self.RegistrationConfig(model_type="affine")
        result = self.register_image(
            source_image=self.img_src,
            transform_matrix=est.transform_matrix,
            reference_shape=self.ref_shape,
            config=reg_cfg,
        )
        self.assertIsInstance(result, self.RegistrationResult)
        self.assertTrue(result.quality_flags["warp_succeeded"])
        self.assertEqual(result.registered_image.shape, self.img_src.shape)
        self.assertIsNotNone(result.overlap_bbox)



# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics(unittest.TestCase):
    """Tests for MetricsWeights, ValidationMetrics, and compute_validation_metrics()."""

    def setUp(self):
        from src.validation.metrics import (
            MetricsWeights,
            ValidationMetrics,
            compute_validation_metrics,
            summarise_batch_metrics,
        )
        self.MetricsWeights = MetricsWeights
        self.ValidationMetrics = ValidationMetrics
        self.compute_validation_metrics = compute_validation_metrics
        self.summarise_batch_metrics = summarise_batch_metrics

    def test_weights_defaults(self):
        """MetricsWeights defaults sum to 1.0."""
        w = self.MetricsWeights()
        total = (
            w.evidence_weight
            + w.inlier_ratio_weight
            + w.reprojection_weight
            + w.spatial_uniformity_weight
        )
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_weights_custom(self):
        """MetricsWeights accepts custom values."""
        w = self.MetricsWeights(evidence_weight=0.5, inlier_ratio_weight=0.5,
                                reprojection_weight=0.0, spatial_uniformity_weight=0.0)
        self.assertAlmostEqual(w.evidence_weight, 0.5)

    def test_compute_metrics_is_callable(self):
        """compute_validation_metrics() is callable with a gate_result."""
        from src.validation.evidence_gate import EvidenceGateResult
        gate_result = EvidenceGateResult(
            passed=True, rejection_reason=None, evidence_score=0.7,
        )
        # Without estimation_result gate passes but no estimation → REJECT
        vm = self.compute_validation_metrics(gate_result=gate_result)
        self.assertIsInstance(vm, self.ValidationMetrics)

    def test_summarise_batch_raises_on_empty(self):
        """summarise_batch_metrics() raises ValueError on empty list."""
        with self.assertRaises(ValueError):
            self.summarise_batch_metrics(metrics_list=[])

    def test_validation_metrics_dataclass(self):
        """ValidationMetrics can be constructed directly with all fields."""
        vm = self.ValidationMetrics(
            decision="ACCEPT",
            quality_score=0.82,
            passed=True,
            failure_stage=None,
            failure_reason=None,
            evidence_score=0.70,
            num_inliers=45,
            inlier_ratio=0.75,
            reprojection_rmse_px=2.3,
            spatial_uniformity=0.68,
            transform_model="HOMOGRAPHY",
        )
        self.assertTrue(vm.passed)
        self.assertAlmostEqual(vm.quality_score, 0.82)
        self.assertEqual(vm.transform_model, "HOMOGRAPHY")
        self.assertEqual(vm.decision, "ACCEPT")

    def test_to_dict_returns_dict(self):
        """ValidationMetrics.to_dict() returns a flat dict."""
        from src.validation.evidence_gate import EvidenceGateResult
        gate_result = EvidenceGateResult(
            passed=True, rejection_reason=None, evidence_score=0.7,
        )
        vm = self.compute_validation_metrics(gate_result=gate_result)
        d = vm.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("decision", d)
        self.assertIn("quality_score", d)
        self.assertIn("passed", d)

    def test_summary_string_returns_string(self):
        """ValidationMetrics.summary_string() returns a non-empty string."""
        from src.validation.evidence_gate import EvidenceGateResult
        gate_result = EvidenceGateResult(
            passed=False, rejection_reason="insufficient_matches", evidence_score=0.1,
        )
        vm = self.compute_validation_metrics(gate_result=gate_result)
        s = vm.summary_string()
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)
        self.assertIn("REJECT", s)


# ---------------------------------------------------------------------------
# Metrics algorithmic tests (Phase 6 — implemented)
# ---------------------------------------------------------------------------


def _make_gate_result(passed=True, evidence_score=0.8, rejection_reason=None):
    """Build a minimal EvidenceGateResult for testing."""
    from src.validation.evidence_gate import EvidenceGateResult
    return EvidenceGateResult(
        passed=passed,
        rejection_reason=rejection_reason,
        evidence_score=evidence_score,
    )


def _make_estimation_result(
    success=True, n_inliers=40, n_outliers=10, inlier_ratio=0.80,
    model_name="AFFINE", transform_matrix=None,
):
    """Build a minimal EstimationResult-like namespace for testing."""
    import types
    r = types.SimpleNamespace()
    r.success = success
    r.num_inliers = n_inliers
    r.num_outliers = n_outliers
    r.total_matches = n_inliers + n_outliers
    r.inlier_ratio = inlier_ratio
    r.reason = None if success else "ransac_failed"
    r.failure_reason = r.reason
    r.inlier_mask = np.ones(n_inliers + n_outliers, dtype=bool)
    r.inlier_mask[n_inliers:] = False

    # Model as simple namespace with .name attribute
    r.model = types.SimpleNamespace(name=model_name)
    if transform_matrix is None:
        r.transform_matrix = np.eye(3, dtype=np.float64)
    else:
        r.transform_matrix = transform_matrix
    return r


def _make_inlier_report(inlier_count=40, outlier_count=10, spatial_coverage=0.75):
    """Build a minimal InlierReport-like namespace for testing."""
    import types
    r = types.SimpleNamespace()
    r.inlier_count = inlier_count
    r.outlier_count = outlier_count
    r.inlier_ratio = inlier_count / max(inlier_count + outlier_count, 1)
    r.spatial_coverage = spatial_coverage
    return r


def _make_reprojection_report(
    inlier_mean=1.5, inlier_median=1.2, inlier_rmse=1.8, inlier_max=4.0,
):
    """Build a minimal ReprojectionReport-like namespace for testing."""
    import types
    r = types.SimpleNamespace()
    r.inlier_mean_error = inlier_mean
    r.inlier_median_error = inlier_median
    r.inlier_rmse = inlier_rmse
    r.inlier_max_error = inlier_max
    r.outlier_mean_error = 15.0
    r.overall_rmse = 5.0
    r.per_match_errors = np.ones(50, dtype=np.float64)
    return r


def _make_registration_result(warp_succeeded=True, overlap_sufficient=True):
    """Build a minimal RegistrationResult-like namespace for testing."""
    import types
    r = types.SimpleNamespace()
    r.registered_image = np.zeros((200, 200), dtype=np.uint8)
    r.transform_matrix = np.eye(3, dtype=np.float64)
    r.output_size = (200, 200)
    r.overlap_bbox = (10, 10, 180, 180)
    r.estimated_location = None
    r.quality_flags = {
        "warp_succeeded": warp_succeeded,
        "overlap_nonzero": warp_succeeded,
        "overlap_sufficient": overlap_sufficient,
    }
    r.diagnostics = {}
    return r


class TestMetricsImpl(unittest.TestCase):
    """Algorithmic tests for compute_validation_metrics() (M3 Phase 6).

    All tests are self-contained and do not depend on external data files.
    They use simple namespace objects to simulate stage results so the tests
    remain isolated from upstream implementation details.
    """

    def setUp(self):
        from src.validation.metrics import (
            MetricsWeights, DecisionThresholds, ValidationMetrics,
            compute_validation_metrics, summarise_batch_metrics,
            ACCEPT, REJECT,
        )
        self.MetricsWeights = MetricsWeights
        self.DecisionThresholds = DecisionThresholds
        self.ValidationMetrics = ValidationMetrics
        self.compute = compute_validation_metrics
        self.summarise = summarise_batch_metrics
        self.ACCEPT = ACCEPT
        self.REJECT = REJECT

        # Default thresholds: lenient so basic happy-path tests pass
        self.lenient = DecisionThresholds(
            min_inlier_ratio=0.20,
            min_inlier_count=4,
            max_inlier_rmse_px=None,
            require_warp_success=True,
            require_overlap_sufficient=False,
        )

    # ------------------------------------------------------------------
    # Test 1: Gate required
    # ------------------------------------------------------------------

    def test_none_gate_result_raises_type_error(self):
        """compute_validation_metrics raises TypeError when gate_result is None."""
        with self.assertRaises(TypeError):
            self.compute(gate_result=None)

    # ------------------------------------------------------------------
    # Test 2: Gate rejected → REJECT
    # ------------------------------------------------------------------

    def test_gate_rejected_produces_reject(self):
        """When the evidence gate fails, decision is REJECT at stage evidence_gate."""
        gate = _make_gate_result(passed=False, evidence_score=0.05,
                                 rejection_reason="insufficient_matches")
        vm = self.compute(gate_result=gate, thresholds=self.lenient)
        self.assertEqual(vm.decision, self.REJECT)
        self.assertFalse(vm.passed)
        self.assertEqual(vm.failure_stage, "evidence_gate")
        self.assertEqual(vm.failure_reason, "insufficient_matches")

    # ------------------------------------------------------------------
    # Test 3: Estimation failed → REJECT
    # ------------------------------------------------------------------

    def test_estimation_failure_produces_reject(self):
        """When geometric estimation fails, decision is REJECT at stage geometric_estimation."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=False, n_inliers=2, n_outliers=8)
        vm = self.compute(
            gate_result=gate,
            estimation_result=est,
            thresholds=self.lenient,
        )
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "geometric_estimation")

    # ------------------------------------------------------------------
    # Test 4: No estimation result (gate passed, nothing else) → REJECT
    # ------------------------------------------------------------------

    def test_no_estimation_result_produces_reject(self):
        """No estimation_result with a passed gate still produces REJECT."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        vm = self.compute(gate_result=gate, thresholds=self.lenient)
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "geometric_estimation")

    # ------------------------------------------------------------------
    # Test 5: Low inlier ratio → REJECT
    # ------------------------------------------------------------------

    def test_low_inlier_ratio_produces_reject(self):
        """Inlier ratio below threshold produces REJECT at stage inlier_ratio."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(
            success=True, n_inliers=5, n_outliers=45, inlier_ratio=0.10,
        )
        thresh = self.DecisionThresholds(min_inlier_ratio=0.25, min_inlier_count=4)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=thresh)
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "inlier_ratio")

    # ------------------------------------------------------------------
    # Test 6: Insufficient inlier count → REJECT
    # ------------------------------------------------------------------

    def test_insufficient_inlier_count_produces_reject(self):
        """Inlier count below threshold produces REJECT at stage inlier_count."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(
            success=True, n_inliers=3, n_outliers=7, inlier_ratio=0.30,
        )
        thresh = self.DecisionThresholds(min_inlier_ratio=0.10, min_inlier_count=6)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=thresh)
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "inlier_count")

    # ------------------------------------------------------------------
    # Test 7: Excessive reprojection error → REJECT
    # ------------------------------------------------------------------

    def test_excessive_rmse_produces_reject(self):
        """RMSE above max_inlier_rmse_px produces REJECT at stage reprojection_error."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        rep = _make_reprojection_report(inlier_rmse=15.0)
        thresh = self.DecisionThresholds(
            min_inlier_ratio=0.20, min_inlier_count=4,
            max_inlier_rmse_px=5.0,   # 15 px exceeds 5 px limit
        )
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            reprojection_report=rep, thresholds=thresh,
        )
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "reprojection_error")

    # ------------------------------------------------------------------
    # Test 8: Failed warp → REJECT
    # ------------------------------------------------------------------

    def test_failed_warp_produces_reject(self):
        """warp_succeeded=False with require_warp_success=True produces REJECT."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        reg = _make_registration_result(warp_succeeded=False)
        thresh = self.DecisionThresholds(
            min_inlier_ratio=0.20, min_inlier_count=4,
            require_warp_success=True,
        )
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            registration_result=reg, thresholds=thresh,
        )
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "registration")

    # ------------------------------------------------------------------
    # Test 9: Insufficient overlap → REJECT (when required)
    # ------------------------------------------------------------------

    def test_insufficient_overlap_produces_reject_when_required(self):
        """overlap_sufficient=False with require_overlap_sufficient=True → REJECT."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        reg = _make_registration_result(warp_succeeded=True, overlap_sufficient=False)
        thresh = self.DecisionThresholds(
            min_inlier_ratio=0.20, min_inlier_count=4,
            require_warp_success=True, require_overlap_sufficient=True,
        )
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            registration_result=reg, thresholds=thresh,
        )
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "registration")

    # ------------------------------------------------------------------
    # Test 10: Successful full pipeline → ACCEPT
    # ------------------------------------------------------------------

    def test_full_pipeline_accept(self):
        """All stages pass with good metrics → ACCEPT decision."""
        gate = _make_gate_result(passed=True, evidence_score=0.85)
        est = _make_estimation_result(success=True, n_inliers=48, n_outliers=12,
                                      inlier_ratio=0.80)
        inlier = _make_inlier_report(inlier_count=48, outlier_count=12,
                                     spatial_coverage=0.75)
        rep = _make_reprojection_report(inlier_rmse=1.8)
        reg = _make_registration_result(warp_succeeded=True, overlap_sufficient=True)
        thresh = self.DecisionThresholds(
            min_inlier_ratio=0.25, min_inlier_count=4,
            max_inlier_rmse_px=5.0,
            require_warp_success=True, require_overlap_sufficient=False,
        )
        vm = self.compute(
            gate_result=gate,
            estimation_result=est,
            inlier_report=inlier,
            reprojection_report=rep,
            registration_result=reg,
            thresholds=thresh,
        )
        self.assertEqual(vm.decision, self.ACCEPT)
        self.assertTrue(vm.passed)
        self.assertIsNone(vm.failure_stage)
        self.assertIsNone(vm.failure_reason)
        self.assertGreater(vm.quality_score, 0.0)
        self.assertLessEqual(vm.quality_score, 1.0)

    # ------------------------------------------------------------------
    # Test 11: Missing optional metrics (partial pipeline)
    # ------------------------------------------------------------------

    def test_missing_reprojection_report_still_accepts(self):
        """Without reprojection report, RMSE-based fields are None (not zero)."""
        gate = _make_gate_result(passed=True, evidence_score=0.85)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        vm = self.compute(
            gate_result=gate, estimation_result=est, thresholds=self.lenient,
        )
        # Decision can be ACCEPT or REJECT depending on inlier counts
        self.assertIsNone(vm.reprojection_rmse_px,
                          "RMSE must be None when reprojection_report is absent.")
        self.assertIsNone(vm.reprojection_mean_px)
        self.assertIsNone(vm.reprojection_median_px)
        self.assertIsNone(vm.reprojection_max_px)

    def test_missing_inlier_report_spatial_uniformity_is_none(self):
        """Without inlier_report, spatial_uniformity is None."""
        gate = _make_gate_result(passed=True, evidence_score=0.85)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        vm = self.compute(
            gate_result=gate, estimation_result=est, thresholds=self.lenient,
        )
        self.assertIsNone(vm.spatial_uniformity)

    def test_missing_registration_result_warp_flags_are_none(self):
        """Without registration_result, warp_succeeded and overlap_sufficient are None."""
        gate = _make_gate_result(passed=True, evidence_score=0.85)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        vm = self.compute(
            gate_result=gate, estimation_result=est, thresholds=self.lenient,
        )
        self.assertIsNone(vm.warp_succeeded)
        self.assertIsNone(vm.overlap_sufficient)

    # ------------------------------------------------------------------
    # Test 12: Composite quality score properties
    # ------------------------------------------------------------------

    def test_quality_score_in_unit_interval(self):
        """quality_score is always within [0, 1]."""
        for evidence in (0.0, 0.5, 1.0):
            for ir in (0.0, 0.5, 1.0):
                gate = _make_gate_result(passed=True, evidence_score=evidence)
                est = _make_estimation_result(
                    success=True, n_inliers=int(40 * ir), n_outliers=int(40 * (1 - ir)),
                    inlier_ratio=ir,
                )
                vm = self.compute(
                    gate_result=gate, estimation_result=est, thresholds=self.lenient,
                )
                self.assertGreaterEqual(vm.quality_score, 0.0)
                self.assertLessEqual(vm.quality_score, 1.0)

    def test_quality_score_increases_with_better_inlier_ratio(self):
        """Higher inlier_ratio produces higher quality_score (all else equal)."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est_low = _make_estimation_result(
            success=True, n_inliers=10, n_outliers=40, inlier_ratio=0.20,
        )
        est_high = _make_estimation_result(
            success=True, n_inliers=45, n_outliers=5, inlier_ratio=0.90,
        )
        vm_low = self.compute(
            gate_result=gate, estimation_result=est_low, thresholds=self.lenient,
        )
        vm_high = self.compute(
            gate_result=gate, estimation_result=est_high, thresholds=self.lenient,
        )
        self.assertGreater(vm_high.quality_score, vm_low.quality_score)

    def test_custom_weights_change_score(self):
        """Custom MetricsWeights change the composite score."""
        gate = _make_gate_result(passed=True, evidence_score=0.0)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10,
                                      inlier_ratio=0.80)
        # With all weight on inlier_ratio, score should be ~0.80
        w_inlier_only = self.MetricsWeights(
            evidence_weight=0.0, inlier_ratio_weight=1.0,
            reprojection_weight=0.0, spatial_uniformity_weight=0.0,
        )
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            weights=w_inlier_only, thresholds=self.lenient,
        )
        self.assertAlmostEqual(vm.quality_score, 0.80, places=3)

    # ------------------------------------------------------------------
    # Test 13: Field values extracted correctly
    # ------------------------------------------------------------------

    def test_evidence_score_extracted(self):
        """evidence_score is extracted from gate_result.evidence_score."""
        gate = _make_gate_result(passed=True, evidence_score=0.72)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=self.lenient)
        self.assertAlmostEqual(vm.evidence_score, 0.72, places=6)

    def test_num_correspondences_extracted(self):
        """num_correspondences is n_inliers + n_outliers."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=30, n_outliers=20)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=self.lenient)
        self.assertEqual(vm.num_correspondences, 50)

    def test_inlier_count_from_inlier_report_overrides_estimation(self):
        """If inlier_report provides inlier_count, it overrides the estimation value."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        inlier = _make_inlier_report(inlier_count=38, outlier_count=12)  # slightly different
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            inlier_report=inlier, thresholds=self.lenient,
        )
        self.assertEqual(vm.num_inliers, 38)
        self.assertEqual(vm.num_outliers, 12)

    def test_reprojection_fields_extracted(self):
        """All reprojection error fields are extracted from reprojection_report."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        rep = _make_reprojection_report(
            inlier_mean=1.1, inlier_median=0.9, inlier_rmse=1.5, inlier_max=3.8,
        )
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            reprojection_report=rep, thresholds=self.lenient,
        )
        self.assertAlmostEqual(vm.reprojection_mean_px, 1.1, places=5)
        self.assertAlmostEqual(vm.reprojection_median_px, 0.9, places=5)
        self.assertAlmostEqual(vm.reprojection_rmse_px, 1.5, places=5)
        self.assertAlmostEqual(vm.reprojection_max_px, 3.8, places=5)

    def test_warp_flags_extracted_from_registration(self):
        """warp_succeeded and overlap_sufficient are extracted from registration_result."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        reg = _make_registration_result(warp_succeeded=True, overlap_sufficient=False)
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            registration_result=reg, thresholds=self.lenient,
        )
        self.assertTrue(vm.warp_succeeded)
        self.assertFalse(vm.overlap_sufficient)

    # ------------------------------------------------------------------
    # Test 14: to_dict() serialisation
    # ------------------------------------------------------------------

    def test_to_dict_contains_all_scalar_keys(self):
        """to_dict() returns dict with all mandatory scalar keys."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=self.lenient)
        d = vm.to_dict()
        required_keys = [
            "decision", "quality_score", "passed", "failure_stage", "failure_reason",
            "evidence_score", "gate_passed", "num_correspondences",
            "num_inliers", "num_outliers", "inlier_ratio",
            "reprojection_rmse_px", "transform_model", "transform_valid",
            "warp_succeeded",
        ]
        for key in required_keys:
            self.assertIn(key, d, f"Missing key in to_dict(): {key}")

    def test_to_dict_values_are_python_native(self):
        """to_dict() returns Python-native (not numpy) values."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=self.lenient)
        d = vm.to_dict()
        for key, val in d.items():
            if val is None:
                continue
            self.assertNotIsInstance(
                val, (np.integer, np.floating, np.bool_),
                f"Key '{key}' has numpy type {type(val).__name__}",
            )

    def test_to_dict_is_json_serialisable(self):
        """to_dict() output can be serialised with json.dumps()."""
        import json
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        rep = _make_reprojection_report(inlier_rmse=2.0)
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            reprojection_report=rep, thresholds=self.lenient,
        )
        d = vm.to_dict()
        # Should not raise
        serialised = json.dumps(d)
        self.assertIsInstance(serialised, str)

    # ------------------------------------------------------------------
    # Test 15: summary_string()
    # ------------------------------------------------------------------

    def test_summary_string_accept_format(self):
        """ACCEPT summary_string contains ACCEPT, quality, inliers, RMSE, model."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        rep = _make_reprojection_report(inlier_rmse=2.3)
        vm = self.compute(
            gate_result=gate, estimation_result=est,
            reprojection_report=rep, thresholds=self.lenient,
        )
        s = vm.summary_string()
        self.assertIn("ACCEPT", s)
        self.assertIn("quality=", s)
        self.assertIn("inliers=", s)
        self.assertIn("RMSE=", s)

    def test_summary_string_reject_format(self):
        """REJECT summary_string contains REJECT, stage, reason."""
        gate = _make_gate_result(passed=False, evidence_score=0.05,
                                 rejection_reason="low_confidence")
        vm = self.compute(gate_result=gate, thresholds=self.lenient)
        s = vm.summary_string()
        self.assertIn("REJECT", s)
        self.assertIn("stage=", s)
        self.assertIn("reason=", s)
        self.assertIn("low_confidence", s)

    # ------------------------------------------------------------------
    # Test 16: summarise_batch_metrics()
    # ------------------------------------------------------------------

    def test_batch_summarise_basic(self):
        """summarise_batch_metrics returns correct counts and stats."""
        gate_ok = _make_gate_result(passed=True, evidence_score=0.8)
        gate_fail = _make_gate_result(passed=False, evidence_score=0.05,
                                      rejection_reason="no_matches")
        est_ok = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)

        vm1 = self.compute(gate_result=gate_ok, estimation_result=est_ok,
                           thresholds=self.lenient)
        vm2 = self.compute(gate_result=gate_fail, thresholds=self.lenient)
        vm3 = self.compute(gate_result=gate_ok, estimation_result=est_ok,
                           thresholds=self.lenient)

        summary = self.summarise([vm1, vm2, vm3])
        self.assertEqual(summary["pass_count"] + summary["fail_count"], 3)
        self.assertAlmostEqual(
            summary["pass_rate"],
            summary["pass_count"] / 3, places=6
        )
        self.assertGreaterEqual(summary["mean_quality"], 0.0)
        self.assertLessEqual(summary["mean_quality"], 1.0)
        self.assertIn("per_result", summary)
        self.assertEqual(len(summary["per_result"]), 3)

    def test_batch_empty_raises_value_error(self):
        """summarise_batch_metrics raises ValueError on empty list."""
        with self.assertRaises(ValueError):
            self.summarise([])

    def test_batch_single_item(self):
        """summarise_batch_metrics works with a single-item list."""
        gate = _make_gate_result(passed=True, evidence_score=0.8)
        est = _make_estimation_result(success=True, n_inliers=40, n_outliers=10)
        vm = self.compute(gate_result=gate, estimation_result=est, thresholds=self.lenient)
        summary = self.summarise([vm])
        self.assertEqual(len(summary["per_result"]), 1)
        self.assertAlmostEqual(summary["pass_rate"], float(vm.passed))

    # ------------------------------------------------------------------
    # Test 17: End-to-end Phase 1 → Phase 6 with real M3 modules
    # ------------------------------------------------------------------

    def test_end_to_end_full_pipeline_accept(self):
        """Full pipeline using real M3 modules: Phase 2→Phase 6 → ACCEPT.

        Phase 1 (EvidenceGate.evaluate) is not yet implemented, so the gate
        result is constructed directly from EvidenceGateResult.
        """
        from src.validation.evidence_gate import EvidenceGateResult
        from src.validation.geometric_estimation import (
            GeometricEstimator, GeometricEstimatorConfig, TransformModel,
        )
        from src.validation.inlier_analysis import analyse_inliers
        from src.validation.reprojection import compute_reprojection_errors
        from src.validation.registration import register_image, RegistrationConfig

        # Build a strong synthetic correspondence (50 inliers, 5 outliers)
        correspondence, true_M = _make_transformed_correspondence(
            n_inliers=50, n_outliers=5, seed=42,
        )

        # Phase 1: Construct gate result directly (evaluate() not yet implemented)
        gate_result = EvidenceGateResult(
            passed=True, rejection_reason=None, evidence_score=0.85,
        )

        # Phase 2: Geometric estimation
        est_cfg = GeometricEstimatorConfig(
            model=TransformModel.AFFINE,
            ransac_reproj_threshold=3.0,
            min_inliers=4, min_inlier_ratio=0.0,
        )
        est_result = GeometricEstimator(config=est_cfg).estimate(correspondence)
        self.assertTrue(est_result.success)

        # Phase 3: Inlier analysis
        inlier_report = analyse_inliers(
            inlier_mask=est_result.inlier_mask,
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
        )

        # Phase 4: Reprojection
        rep_report = compute_reprojection_errors(
            source_points=correspondence["source_points"],
            reference_points=correspondence["reference_points"],
            transform_matrix=est_result.transform_matrix,
            inlier_mask=est_result.inlier_mask,
            model_type="affine",
        )

        # Phase 5: Registration
        img_src, img_ref, _ = _make_synthetic_images()
        reg_result = register_image(
            source_image=img_src,
            transform_matrix=est_result.transform_matrix,
            reference_shape=img_ref.shape[:2],
            config=RegistrationConfig(model_type="affine"),
        )

        # Phase 6: Metrics
        thresh = self.DecisionThresholds(
            min_inlier_ratio=0.20, min_inlier_count=4,
            max_inlier_rmse_px=None, require_warp_success=True,
        )
        vm = self.compute(
            gate_result=gate_result,
            estimation_result=est_result,
            inlier_report=inlier_report,
            reprojection_report=rep_report,
            registration_result=reg_result,
            thresholds=thresh,
        )

        self.assertIsInstance(vm, self.ValidationMetrics)
        self.assertEqual(vm.decision, self.ACCEPT)
        self.assertTrue(vm.passed)
        self.assertIsNone(vm.failure_stage)
        self.assertIsNotNone(vm.inlier_ratio)
        self.assertGreater(vm.inlier_ratio, 0.20)
        self.assertIsNotNone(vm.reprojection_rmse_px)
        self.assertLess(vm.reprojection_rmse_px, 3.5)
        self.assertTrue(vm.warp_succeeded)
        self.assertGreater(vm.quality_score, 0.0)

        # Check summary string
        s = vm.summary_string()
        self.assertIn("ACCEPT", s)

        # Check to_dict and JSON serialisability
        import json
        d = vm.to_dict()
        self.assertEqual(d["decision"], self.ACCEPT)
        json.dumps(d)  # must not raise

    def test_end_to_end_gate_reject_pipeline(self):
        """Pipeline with gate rejection propagates correctly through Phase 6.

        Phase 1 (EvidenceGate.evaluate) is not yet implemented, so the gate
        result is constructed directly from EvidenceGateResult.
        """
        from src.validation.evidence_gate import EvidenceGateResult

        # Simulate a gate that rejected due to insufficient matches
        gate_result = EvidenceGateResult(
            passed=False,
            rejection_reason="insufficient_matches",
            evidence_score=0.05,
        )
        self.assertFalse(gate_result.passed)

        vm = self.compute(gate_result=gate_result)
        self.assertEqual(vm.decision, self.REJECT)
        self.assertEqual(vm.failure_stage, "evidence_gate")
        s = vm.summary_string()
        self.assertIn("REJECT", s)


class TestM3PipelineSkeleton(unittest.TestCase):
    """Smoke test verifying the M3 pipeline skeleton can be imported and
    that all stage classes are discoverable through the expected import paths."""

    def test_all_stage_classes_importable(self):
        """All M3 stage classes are importable by name."""
        from src.validation.evidence_gate import EvidenceGate, EvidenceGateConfig
        from src.validation.geometric_estimation import (
            GeometricEstimator,
            GeometricEstimatorConfig,
            TransformModel,
        )
        from src.validation.inlier_analysis import InlierAnalyzer, analyse_inliers
        from src.validation.reprojection import compute_reprojection_errors
        from src.validation.registration import register_image, RegistrationConfig
        from src.validation.metrics import (
            compute_validation_metrics,
            ValidationMetrics,
            MetricsWeights,
            summarise_batch_metrics,
        )

        # Verify none are None
        for obj in [
            EvidenceGate, EvidenceGateConfig,
            GeometricEstimator, GeometricEstimatorConfig, TransformModel,
            InlierAnalyzer, analyse_inliers,
            compute_reprojection_errors,
            register_image, RegistrationConfig,
            compute_validation_metrics, ValidationMetrics,
            MetricsWeights, summarise_batch_metrics,
        ]:
            self.assertIsNotNone(obj)


if __name__ == "__main__":
    unittest.main(verbosity=2)
