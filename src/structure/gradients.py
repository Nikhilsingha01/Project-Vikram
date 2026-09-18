"""Gradient computation module for lunar image structural analysis.

Provides functions to compute Sobel directional gradients (Gx, Gy),
gradient magnitude, and gradient orientation fields for lunar terrain imagery.
"""

from typing import Dict, Tuple, Union
import cv2
import numpy as np


def validate_grayscale(image: np.ndarray) -> np.ndarray:
    """Validate and convert input image to a standard single-channel 2D uint8/float32 array.

    Args:
        image: Input image array (2D grayscale or 3D BGR/RGB).

    Returns:
        np.ndarray: 2D single-channel grayscale image.

    Raises:
        ValueError: If input is None, empty, or has unsupported dimensions.
        TypeError: If input is not a numpy ndarray.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be numpy.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Input image is empty.")

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 1:
            image = image.squeeze(axis=2)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    elif image.ndim != 2:
        raise ValueError(f"Expected 2D or 3D image, got array with {image.ndim} dimensions.")

    return image


def compute_gradients(
    image: np.ndarray,
    ksize: int = 3,
    dtype: int = cv2.CV_64F,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute directional Sobel gradients in X and Y directions.

    Args:
        image: Input 2D grayscale image.
        ksize: Size of the extended Sobel kernel; must be 1, 3, 5, or 7.
        dtype: Output image depth (default cv2.CV_64F to prevent clipping).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (gx, gy) gradient matrices.
    """
    gray = validate_grayscale(image)
    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"Sobel ksize must be 1, 3, 5, or 7, got {ksize}")

    gx = cv2.Sobel(gray, dtype, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, dtype, 0, 1, ksize=ksize)
    return gx, gy


def compute_gradient_magnitude(
    gx: np.ndarray,
    gy: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute gradient magnitude from Sobel X and Y gradients.

    Args:
        gx: Sobel gradient in X direction.
        gy: Sobel gradient in Y direction.
        normalize: If True, normalizes output to [0, 255] uint8.
                   If False, returns raw float magnitude.

    Returns:
        np.ndarray: Gradient magnitude array.
    """
    if gx.shape != gy.shape:
        raise ValueError(f"Shape mismatch between gx {gx.shape} and gy {gy.shape}")

    magnitude = cv2.magnitude(gx.astype(np.float64), gy.astype(np.float64))

    if normalize:
        max_val = np.max(magnitude)
        if max_val > 1e-8:
            norm_mag = (magnitude / max_val * 255.0).astype(np.uint8)
        else:
            norm_mag = np.zeros_like(magnitude, dtype=np.uint8)
        return norm_mag

    return magnitude


def compute_gradient_orientation(
    gx: np.ndarray,
    gy: np.ndarray,
    in_degrees: bool = True,
) -> np.ndarray:
    """Compute gradient orientation angle from Sobel X and Y gradients.

    Args:
        gx: Sobel gradient in X direction.
        gy: Sobel gradient in Y direction.
        in_degrees: If True, returns angles in [0, 360) degrees.
                    If False, returns angles in [-pi, pi] radians.

    Returns:
        np.ndarray: Gradient orientation array.
    """
    if gx.shape != gy.shape:
        raise ValueError(f"Shape mismatch between gx {gx.shape} and gy {gy.shape}")

    if in_degrees:
        orientation = cv2.phase(gx.astype(np.float64), gy.astype(np.float64), angleInDegrees=True)
    else:
        orientation = np.arctan2(gy.astype(np.float64), gx.astype(np.float64))

    return orientation


def compute_gradient_field(
    image: np.ndarray,
    ksize: int = 3,
    normalize_magnitude: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute complete gradient field representation (gx, gy, magnitude, orientation).

    Args:
        image: Input 2D grayscale image.
        ksize: Size of the Sobel kernel (1, 3, 5, or 7).
        normalize_magnitude: Whether to normalize magnitude to [0, 255] uint8.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'gx': Horizontal gradient
            - 'gy': Vertical gradient
            - 'magnitude': Gradient magnitude
            - 'orientation': Orientation in degrees [0, 360)
    """
    gx, gy = compute_gradients(image, ksize=ksize)
    magnitude = compute_gradient_magnitude(gx, gy, normalize=normalize_magnitude)
    orientation = compute_gradient_orientation(gx, gy, in_degrees=True)

    return {
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "orientation": orientation,
    }
