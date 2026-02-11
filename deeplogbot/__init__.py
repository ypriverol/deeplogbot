"""
DeepLogBot: A hybrid Isolation Forest-based system for detecting bot behavior in log data.

This package provides tools for detecting bots and download hubs in log data,
with support for multiple log providers and extensible classification rules.

Provider System:
    DeepLogBot supports multiple log providers (EBI, custom, etc.). Each provider
    defines its own schema, feature extractors, and classification rules.

    Available providers can be listed with:
        from deeplogbot import list_available_providers
        print(list_available_providers())

    Set the active provider with:
        from deeplogbot import set_active_provider
        set_active_provider('ebi')  # or 'custom', etc.

Example:
    from deeplogbot import run_bot_annotator

    # Run with EBI provider (default)
    results = run_bot_annotator(
        input_parquet='data.parquet',
        output_dir='output/',
        classification_method='deep',
        provider='ebi'
    )
"""

__version__ = "0.1.0"
__author__ = "DeepLogBot Contributors"

from .main import run_bot_annotator
from .models import (
    train_isolation_forest,
    compute_feature_importances,
    classify_locations,
    classify_locations_hierarchical
)
from .features import (
    extract_location_features,
    extract_location_features_ebi,
    BaseFeatureExtractor,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)
from .reports import generate_report, ReportGenerator, annotate_downloads
from .features import LogSchema, EBI_SCHEMA, get_schema, register_schema, SCHEMA_REGISTRY

# Provider management
from .config import (
    set_active_provider,
    get_active_provider_name,
    list_available_providers,
    get_provider_config,
    get_provider_taxonomy,
)

__all__ = [
    # Main pipeline
    "run_bot_annotator",
    # Models
    "train_isolation_forest",
    "compute_feature_importances",
    "classify_locations",
    "classify_locations_hierarchical",
    # Feature extraction
    "extract_location_features",
    "extract_location_features_ebi",
    "BaseFeatureExtractor",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
    # Reports
    "annotate_downloads",
    "generate_report",
    # Schema
    "LogSchema",
    "EBI_SCHEMA",
    "get_schema",
    "register_schema",
    "SCHEMA_REGISTRY",
    # Provider management
    "set_active_provider",
    "get_active_provider_name",
    "list_available_providers",
    "get_provider_config",
    "get_provider_taxonomy",
]

