"""Comprehensive structural representation module for lunar imagery.

Fuses gradient fields, Canny edge boundaries, orientation patterns, and morphological
terrain descriptors into a unified structural representation for illumination-invariant
matching and candidate search.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.structure.edges import compute_edge_density, extract_canny_edges
from src.structure.gradients import (
    compute_gradient_field,
    validate_grayscale,
)
from src.structure.terrain_features import extract_terrain_descriptor_map


@dataclass
class StructuralRepresentation:
    """Dataclass holding all extracted structural layers for an image."""

    original_shape: Tuple[int, int]
    gradient_magnitude: np.ndarray
    gradient_orientation: np.ndarray
    edges: np.ndarray
    terrain_map: np.ndarray
    composite: np.ndarray
    edge_density: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert structural representation to dictionary format."""
        return {
            "original_shape": self.original_shape,
            "gradient_magnitude": self.gradient_magnitude,
            "gradient_orientation": self.gradient_orientation,
            "edges": self.edges,
            "terrain_map": self.terrain_map,
            "composite": self.composite,
            "edge_density": self.edge_density,
            "metadata": self.metadata,
        }


def extract_structure_representation(
    image: np.ndarray,
    edge_low_threshold: int = 50,
    edge_high_threshold: int = 150,
    sobel_ksize: int = 3,
    morph_kernel_size: int = 5,
) -> StructuralRepresentation:
    """Extract and fuse all structural layers for an input lunar image.

    Args:
        image: Input 2D grayscale or BGR image.
        edge_low_threshold: Canny edge lower threshold.
        edge_high_threshold: Canny edge upper threshold.
        sobel_ksize: Sobel kernel size (1, 3, 5, or 7).
        morph_kernel_size: Morphological structuring element size.

    Returns:
        StructuralRepresentation: Object containing all extracted structural components.
    """
    gray = validate_grayscale(image)
    h, w = gray.shape[:2]

    # 1. Gradient Analysis
    grad_field = compute_gradient_field(gray, ksize=sobel_ksize, normalize_magnitude=True)
    grad_mag = grad_field["magnitude"]
    grad_ori = grad_field["orientation"]

    # 2. Edge Extraction
    edges = extract_canny_edges(
        gray,
        low_threshold=edge_low_threshold,
        high_threshold=edge_high_threshold,
        aperture_size=sobel_ksize,
    )
    edge_dens = compute_edge_density(edges)

    # 3. Terrain Relif & Morphological Feature Map
    terrain_map = extract_terrain_descriptor_map(gray, kernel_size=morph_kernel_size)

    # 4. Fused Structural Composite (Single-channel uint8 normalized)
    # Blends normalized gradient magnitude (40%), Canny edges (30%), and terrain relief (30%)
    composite = cv2.addWeighted(grad_mag, 0.4, terrain_map, 0.3, 0)
    composite = cv2.addWeighted(composite, 1.0, edges, 0.3, 0)
    composite = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    metadata = {
        "edge_low_threshold": edge_low_threshold,
        "edge_high_threshold": edge_high_threshold,
        "sobel_ksize": sobel_ksize,
        "morph_kernel_size": morph_kernel_size,
    }

    return StructuralRepresentation(
        original_shape=(h, w),
        gradient_magnitude=grad_mag,
        gradient_orientation=grad_ori,
        edges=edges,
        terrain_map=terrain_map,
        composite=composite,
        edge_density=edge_dens,
        metadata=metadata,
    )
