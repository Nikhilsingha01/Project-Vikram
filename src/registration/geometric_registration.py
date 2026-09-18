"""Feature-detector-independent geometric image registration."""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .ransac import estimate_affine, estimate_projective, validate_ransac_parameters
from .subpixel_refinement import refine_subpixel_points


def estimate_affine_transform(source_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """Estimate a robust 2-D affine transform mapping source to reference."""
    source, reference = _validate_correspondences(source_points, reference_points, minimum=3)
    matrix, _ = estimate_affine(source, reference)
    return matrix


def estimate_homography(source_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """Estimate a robust projective transform mapping source to reference."""
    source, reference = _validate_correspondences(source_points, reference_points, minimum=4)
    matrix, _ = estimate_projective(source, reference)
    return matrix


def register_affine(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate and apply an affine transform in reference-image coordinates."""
    return _register(source_image, reference_image, source_points, reference_points, "affine", **kwargs)


def register_homography(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Estimate and apply a homography in reference-image coordinates."""
    return _register(source_image, reference_image, source_points, reference_points, "homography", **kwargs)


def register_images(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    method: str = "affine",
    refine: bool = False,
    reprojection_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 2000,
    refinement_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Estimate, warp, and measure registration using validated correspondences."""
    return _register(
        source_image,
        reference_image,
        source_points,
        reference_points,
        method.lower(),
        refine=refine,
        reprojection_threshold=reprojection_threshold,
        confidence=confidence,
        max_iterations=max_iterations,
        refinement_kwargs=refinement_kwargs,
    )


def reprojection_errors(
    source_points: np.ndarray, reference_points: np.ndarray, transformation_matrix: np.ndarray
) -> np.ndarray:
    """Return Euclidean reprojection error for each correspondence."""
    source, reference = _validate_correspondences(source_points, reference_points, minimum=0)
    transformed = _transform_points(source, transformation_matrix)
    return np.linalg.norm(transformed - reference, axis=1)


def _register(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    method: str,
    reprojection_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iterations: int = 2000,
    refine: bool = False,
    refinement_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    source_image = _validate_image(source_image, "source_image")
    reference_image = _validate_image(reference_image, "reference_image")
    minimum = {"affine": 3, "homography": 4}.get(method)
    if minimum is None:
        raise ValueError("method must be either 'affine' or 'homography'.")
    source, reference = _validate_correspondences(source_points, reference_points, minimum)
    validate_ransac_parameters(reprojection_threshold, confidence, max_iterations)
    estimator = estimate_affine if method == "affine" else estimate_projective
    matrix, mask = estimator(
        source, reference, reprojection_threshold, confidence, max_iterations
    )
    refined_source = source.copy()
    refined_reference = reference.copy()
    if refine:
        options = refinement_kwargs or {}
        refined_source = refine_subpixel_points(source_image, refined_source, **options)
        refined_reference = refine_subpixel_points(reference_image, refined_reference, **options)
        matrix, mask = estimator(
            refined_source, refined_reference, reprojection_threshold, confidence, max_iterations
        )
    errors = reprojection_errors(refined_source, refined_reference, matrix)
    inliers = np.asarray(mask, dtype=bool)
    height, width = reference_image.shape[:2]
    if method == "affine":
        registered = cv2.warpAffine(source_image, matrix[:2], (width, height))
    else:
        registered = cv2.warpPerspective(source_image, matrix, (width, height))
    return {
        "success": True,
        "method": method,
        "transformation_matrix": matrix,
        "registered_image": registered,
        "source_points": source,
        "reference_points": reference,
        "refined_source_points": refined_source,
        "refined_reference_points": refined_reference,
        "reprojection_errors": errors,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "num_matches": int(len(source)),
        "inlier_mask": inliers,
        "num_inliers": int(np.count_nonzero(inliers)),
        "inlier_ratio": float(np.mean(inliers)),
    }


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3) or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D or 3-D image array.")
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels.")
    return array


def _validate_correspondences(
    source_points: np.ndarray, reference_points: np.ndarray, minimum: int
) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float32)
    reference = np.asarray(reference_points, dtype=np.float32)
    if source.size == 0 or reference.size == 0:
        raise ValueError("Point correspondences must not be empty.")
    if source.ndim != 2 or reference.ndim != 2 or source.shape[1:] != (2,) or reference.shape[1:] != (2,):
        raise ValueError("source_points and reference_points must have shape (N, 2).")
    if len(source) != len(reference):
        raise ValueError("source_points and reference_points must contain the same number of points.")
    if len(source) < minimum:
        raise ValueError(f"At least {minimum} point correspondences are required.")
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError("Point correspondences must contain only finite coordinates.")
    return source, reference


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (2, 3):
        return cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)