"""Edge extraction module for lunar image structural representation.

Provides functions to extract edge maps using Canny algorithms, auto-tuned thresholds,
and edge density computations for structural terrain analysis.
"""

from typing import Tuple, Union
import cv2
import numpy as np

from src.structure.gradients import validate_grayscale


def extract_canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Extract edge map using Canny edge detector.

    Args:
        image: Input 2D grayscale image.
        low_threshold: Lower threshold for the hysteresis procedure.
        high_threshold: Upper threshold for the hysteresis procedure.
        aperture_size: Aperture size for the Sobel operator (3, 5, or 7).
        l2_gradient: If True, uses more accurate L2 norm sqrt((dI/dx)^2 + (dI/dy)^2).

    Returns:
        np.ndarray: Binary uint8 edge map where edge pixels are 255 and others are 0.
    """
    gray = validate_grayscale(image)

    if gray.dtype != np.uint8:
        # Scale to uint8 if needed
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_u8 = norm_gray.astype(np.uint8)
    else:
        gray_u8 = gray

    if aperture_size not in (3, 5, 7):
        raise ValueError(f"aperture_size must be 3, 5, or 7, got {aperture_size}")

    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Thresholds must be non-negative integers.")

    if low_threshold > high_threshold:
        low_threshold, high_threshold = high_threshold, low_threshold

    edges = cv2.Canny(
        gray_u8,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges


def extract_auto_canny_edges(
    image: np.ndarray,
    sigma: float = 0.33,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Extract Canny edges using automated threshold determination based on median intensity.

    Args:
        image: Input 2D grayscale image.
        sigma: Tuning parameter for threshold interval (default 0.33).
        aperture_size: Aperture size for the Sobel operator (3, 5, or 7).
        l2_gradient: If True, uses L2 norm for gradient computation.

    Returns:
        np.ndarray: Binary uint8 edge map.
    """
    gray = validate_grayscale(image)

    if gray.dtype != np.uint8:
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_u8 = norm_gray.astype(np.uint8)
    else:
        gray_u8 = gray

    v = np.median(gray_u8)
    low_thresh = int(max(0, (1.0 - sigma) * v))
    high_thresh = int(min(255, (1.0 + sigma) * v))

    # Ensure low_thresh < high_thresh even if image has uniform illumination
    if low_thresh == high_thresh:
        low_thresh = max(0, low_thresh - 10)
        high_thresh = min(255, high_thresh + 10)

    return extract_canny_edges(
        gray_u8,
        low_threshold=low_thresh,
        high_threshold=high_thresh,
        aperture_size=aperture_size,
        l2_gradient=l2_gradient,
    )


def compute_edge_density(edge_map: np.ndarray) -> float:
    """Compute edge density (ratio of non-zero edge pixels to total pixels).

    Args:
        edge_map: Binary edge map.

    Returns:
        float: Edge density value in range [0.0, 1.0].
    """
    if edge_map.size == 0:
        return 0.0
    edge_pixels = np.count_nonzero(edge_map)
    return float(edge_pixels / edge_map.size)
