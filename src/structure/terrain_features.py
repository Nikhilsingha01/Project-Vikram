"""Lunar terrain features module for structural correspondence.

Extracts illumination-invariant local morphological relief, crater rim indicators,
and multi-scale structural descriptors from lunar surface imagery.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.structure.gradients import validate_grayscale


def extract_morphological_relief(
    image: np.ndarray,
    kernel_size: int = 5,
    morph_type: str = "gradient",
) -> np.ndarray:
    """Extract morphological terrain relief map.

    Morphological operations highlight surface elevations, crater boundaries, and ridges
    while reducing low-frequency sun illumination variations.

    Args:
        image: Input 2D grayscale image.
        kernel_size: Structuring element diameter (odd integer >= 3).
        morph_type: Type of operation ('gradient', 'tophat', 'blackhat', 'relief').
            - 'gradient': Morphological gradient (Dilation - Erosion).
            - 'tophat': Top-Hat (Image - Opening), highlights bright peaks/rims.
            - 'blackhat': Black-Hat (Closing - Image), highlights dark crater floors/shadows.
            - 'relief': Combined Top-Hat + Black-Hat contrast enhancement.

    Returns:
        np.ndarray: Single-channel uint8 morphological feature map.
    """
    gray = validate_grayscale(image)

    if gray.dtype != np.uint8:
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_u8 = norm_gray.astype(np.uint8)
    else:
        gray_u8 = gray

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if morph_type == "gradient":
        result = cv2.morphologyEx(gray_u8, cv2.MORPH_GRADIENT, kernel)
    elif morph_type == "tophat":
        result = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)
    elif morph_type == "blackhat":
        result = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel)
    elif morph_type == "relief":
        tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, kernel)
        result = cv2.addWeighted(tophat, 0.5, blackhat, 0.5, 0)
    else:
        raise ValueError(f"Unknown morph_type: {morph_type}. Choose from gradient, tophat, blackhat, relief.")

    return result


def extract_crater_candidates(
    image: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 60,
    param1: int = 50,
    param2: int = 25,
) -> List[Dict[str, float]]:
    """Detect circular/elliptical crater candidate structures using Hough Circle transform.

    Args:
        image: Input 2D grayscale image.
        min_radius: Minimum crater radius in pixels.
        max_radius: Maximum crater radius in pixels.
        param1: Gradient threshold passed to the Canny edge detector in Hough.
        param2: Accumulator threshold for circle center detection.

    Returns:
        List[Dict[str, float]]: List of detected craters with keys 'x', 'y', 'radius', 'confidence'.
    """
    gray = validate_grayscale(image)

    if gray.dtype != np.uint8:
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_u8 = norm_gray.astype(np.uint8)
    else:
        gray_u8 = gray

    # Smooth image to reduce false alarms from sensor grain
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(min_radius, 10),
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    craters: List[Dict[str, float]] = []
    if circles is not None:
        circles_arr = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles_arr:
            craters.append({
                "x": float(x),
                "y": float(y),
                "radius": float(r),
                "confidence": float(max(0.1, min(1.0, 1.0 - (r / max(max_radius, 1)) * 0.2))),
            })

    return craters


def extract_terrain_descriptor_map(
    image: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Compute a multi-cue structural terrain descriptor map.

    Combines morphological gradient with Laplacian edge response to produce
    an illumination-resilient terrain representation suitable for candidate correlation.

    Args:
        image: Input 2D grayscale image.
        kernel_size: Kernel size for morphological filtering.

    Returns:
        np.ndarray: uint8 2D terrain descriptor map in range [0, 255].
    """
    gray = validate_grayscale(image)

    if gray.dtype != np.uint8:
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray_u8 = norm_gray.astype(np.uint8)
    else:
        gray_u8 = gray

    # 1. Morphological relief
    morph_grad = extract_morphological_relief(gray_u8, kernel_size=kernel_size, morph_type="relief")

    # 2. Laplacian of Gaussian for terrain ridge/rim features
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian)
    lap_u8 = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Fuse morphological relief and Laplacian contours
    fused = cv2.addWeighted(morph_grad, 0.6, lap_u8, 0.4, 0)
    return fused
