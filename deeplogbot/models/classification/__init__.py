"""Classification methods for bot and download hub detection."""

from .rules import classify_locations_hierarchical, classify_locations
from .deep_architecture import classify_locations_deep
from .post_classification import (
    apply_hub_protection,
    log_prediction_summary,
    log_hierarchical_summary,
)

__all__ = [
    "classify_locations",  # Legacy function for backward compatibility
    "classify_locations_hierarchical",
    "classify_locations_deep",
    "apply_hub_protection",
    "log_prediction_summary",
    "log_hierarchical_summary",
]
