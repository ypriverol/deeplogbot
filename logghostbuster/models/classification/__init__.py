"""Classification methods for bot and download hub detection."""

from .rules import classify_locations
from .ml_supervised import classify_locations_ml
from .deep_architecture import classify_locations_deep

__all__ = [
    "classify_locations",
    "classify_locations_ml",
    "classify_locations_deep",
]

