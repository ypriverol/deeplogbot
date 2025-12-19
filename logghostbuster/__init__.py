"""
LogGhostbuster: A hybrid Isolation Forest-based system for detecting bot behavior in log data.
"""

__version__ = "0.1.0"
__author__ = "LogGhostbuster Contributors"

from .main import run_bot_annotator
from .models import train_isolation_forest, compute_feature_importances, classify_locations
from .features import (
    extract_location_features,
    extract_location_features_ebi,
    BaseFeatureExtractor,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)
from .reports import generate_report, ReportGenerator, annotate_downloads
from .features.schema import LogSchema, EBI_SCHEMA, get_schema, register_schema, SCHEMA_REGISTRY

__all__ = [
    "run_bot_annotator",
    "train_isolation_forest",
    "compute_feature_importances",
    "classify_locations",
    "extract_location_features",
    "extract_location_features_ebi",
    "annotate_downloads",
    "generate_report",
    "LogSchema",
    "EBI_SCHEMA",
    "get_schema",
    "register_schema",
    "SCHEMA_REGISTRY",
    "BaseFeatureExtractor",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
]

