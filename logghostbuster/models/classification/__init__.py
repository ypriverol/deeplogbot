"""Classification methods for bot and download hub detection."""

from .rules import classify_locations_hierarchical, classify_locations
from .deep_architecture import classify_locations_deep

__all__ = [
    "classify_locations",  # Legacy function for backward compatibility
    "classify_locations_hierarchical",
    "classify_locations_deep",
]

