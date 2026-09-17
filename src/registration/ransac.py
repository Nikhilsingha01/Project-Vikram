"""Robust geometric-model estimation helpers."""

from typing import Tuple

import cv2
import numpy as np


def estimate_affine(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reprojection_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate an affine transform and return its OpenCV inlier mask."""
    matrix, mask = cv2.estimateAffine2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=max_iterations,
        confidence=confidence,
        refineIters=10,
    )
    if matrix is None or mask is None:
        raise ValueError("Unable to estimate an affine transform from the correspondences.")
    return matrix.astype(np.float64), mask.ravel().astype(bool)


def estimate_projective(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    reprojection_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a homography and return its OpenCV inlier mask."""
    matrix, mask = cv2.findHomography(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=max_iterations,
        confidence=confidence,
    )
    if matrix is None or mask is None:
        raise ValueError("Unable to estimate a homography from the correspondences.")
    return matrix.astype(np.float64), mask.ravel().astype(bool)


def validate_ransac_parameters(
    reprojection_threshold: float,
    confidence: float,
    max_iterations: int,
) -> None:
    """Validate shared robust-estimation parameters."""
    if reprojection_threshold <= 0:
        raise ValueError("reprojection_threshold must be positive.")
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in the interval (0, 1].")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")