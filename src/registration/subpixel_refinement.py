"""Image-based sub-pixel refinement for established point correspondences."""

from typing import Tuple

import cv2
import numpy as np


def refine_subpixel_points(
    image: np.ndarray,
    points: np.ndarray,
    window_size: Tuple[int, int] = (5, 5),
    zero_zone: Tuple[int, int] = (-1, -1),
    max_iterations: int = 30,
    epsilon: float = 0.01,
    min_corner_response: float = 1e-6,
) -> np.ndarray:
    """Refine suitable points using local image gradients.

    Points without a measurable corner response, or points too close to the
    boundary for the requested window, are returned unchanged.
    """
    gray = _as_grayscale(image)
    input_points = _validate_points(points, "points")
    refined = input_points.copy()
    if not len(refined) or gray.size == 0:
        return refined
    if window_size[0] < 1 or window_size[1] < 1:
        raise ValueError("window_size values must be positive.")
    if max_iterations < 1 or epsilon <= 0:
        raise ValueError("max_iterations must be positive and epsilon must be positive.")

    try:
        response = cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)
    except cv2.error:
        return refined
    half_w, half_h = window_size
    height, width = gray.shape[:2]
    valid_indices = []
    for index, (x, y) in enumerate(refined):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if (
            half_w <= x < width - half_w
            and half_h <= y < height - half_h
            and 0 <= xi < width
            and 0 <= yi < height
            and response[yi, xi] > min_corner_response
        ):
            valid_indices.append(index)
    if not valid_indices:
        return refined

    candidates = refined[valid_indices].reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
    try:
        cv2.cornerSubPix(gray, candidates, window_size, zero_zone, criteria)
    except cv2.error:
        return refined
    refined[valid_indices] = candidates.reshape(-1, 2)
    return refined


def refine_points(*args, **kwargs) -> np.ndarray:
    """Backward-compatible short alias for :func:`refine_subpixel_points`."""
    return refine_subpixel_points(*args, **kwargs)


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] in (1, 3, 4):
        if array.shape[2] == 1:
            gray = array[:, :, 0]
        else:
            code = cv2.COLOR_BGR2GRAY if array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            gray = cv2.cvtColor(array, code)
    else:
        raise ValueError("image must be a 2-D grayscale or 3/4-channel color array.")
    if gray.size == 0:
        raise ValueError("image must not be empty.")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array