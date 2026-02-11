"""Isolation Forest model package for anomaly detection."""

from .models import train_isolation_forest, compute_feature_importances

__all__ = [
    "train_isolation_forest",
    "compute_feature_importances",
]

