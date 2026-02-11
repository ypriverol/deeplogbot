"""Models package for bot detection."""

from .isoforest import train_isolation_forest, compute_feature_importances
from .classification import (
    classify_locations,  # Legacy function for backward compatibility
    classify_locations_hierarchical,
    classify_locations_deep,
)

__all__ = [
    "train_isolation_forest",
    "compute_feature_importances",
    "classify_locations",  # Legacy function
    "classify_locations_hierarchical",
    "classify_locations_deep",
]

