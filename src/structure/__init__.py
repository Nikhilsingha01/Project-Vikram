"""Structure Analysis package for Project Vikram / FLUX.

Provides gradient analysis, edge extraction, terrain feature detection,
and unified structural representations for lunar image correspondence.
"""

from src.structure.edges import (
    compute_edge_density,
    extract_auto_canny_edges,
    extract_canny_edges,
)
from src.structure.gradients import (
    compute_gradient_field,
    compute_gradient_magnitude,
    compute_gradient_orientation,
    compute_gradients,
    validate_grayscale,
)
from src.structure.structure_representation import (
    StructuralRepresentation,
    extract_structure_representation,
)
from src.structure.terrain_features import (
    extract_crater_candidates,
    extract_morphological_relief,
    extract_terrain_descriptor_map,
)

__all__ = [
    "validate_grayscale",
    "compute_gradients",
    "compute_gradient_magnitude",
    "compute_gradient_orientation",
    "compute_gradient_field",
    "extract_canny_edges",
    "extract_auto_canny_edges",
    "compute_edge_density",
    "extract_morphological_relief",
    "extract_crater_candidates",
    "extract_terrain_descriptor_map",
    "StructuralRepresentation",
    "extract_structure_representation",
]
