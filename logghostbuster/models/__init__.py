"""Models package for bot detection."""

from .isoforest import train_isolation_forest, compute_feature_importances
from .classification import (
    classify_locations, 
    classify_locations_ml, 
    classify_locations_deep
)

__all__ = [
    "train_isolation_forest",
    "compute_feature_importances",
    "classify_locations",
    "classify_locations_ml",
    "classify_locations_deep",
]

