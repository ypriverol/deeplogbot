"""EBI-specific feature extractors and utilities.

This package contains all feature extraction logic specific to EBI log data:
- Schema definitions (LogEbiSchema, EBI_SCHEMA)
- Feature extractors (YearlyPatternExtractor, TimeOfDayExtractor, etc.)
- Behavioral feature extraction (including new timing/session features)
- Discriminative feature extraction (including new access/statistical features)
- Core EBI extraction functions

New in this version:
- 24 new bot detection features across 6 categories
- Centralized feature registry with documentation
"""

from ...schema import LogSchema, get_schema, register_schema, SCHEMA_REGISTRY
from .schema import LogEbiSchema, EBI_SCHEMA
from .ebi import (
    extract_location_features_ebi,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
    NewBotDetectionFeaturesExtractor,
    TimeSeriesFeaturesExtractor,
    TimeWindowExtractor,
    extract_location_features,  # Alias for backward compatibility
)
from .timeseries import (
    extract_outburst_features,
    extract_periodicity_features,
    extract_trend_features,
    extract_recency_features,
    extract_distribution_shape_features,
    extract_all_timeseries_features,
    OUTBURST_FEATURES,
    PERIODICITY_FEATURES,
    TREND_FEATURES,
    RECENCY_FEATURES,
    DISTRIBUTION_SHAPE_FEATURES,
    ALL_TIMESERIES_FEATURES,
)
from .behavioral import (
    extract_behavioral_features,
    extract_advanced_behavioral_features,
    add_bot_interaction_features,
    add_bot_signature_features,
    # New behavioral feature functions
    extract_timing_precision_features,
    extract_user_distribution_features,
    extract_session_behavior_features,
    ADVANCED_BEHAVIORAL_FEATURES,
    NEW_BEHAVIORAL_FEATURES,
)
from .discriminative import (
    extract_discriminative_features,
    # New discriminative feature functions
    extract_access_pattern_features,
    extract_statistical_anomaly_features,
    extract_comparative_features,
    DISCRIMINATIVE_FEATURES,
    NEW_DISCRIMINATIVE_FEATURES,
)

__all__ = [
    # Schema
    "LogSchema",
    "LogEbiSchema",
    "EBI_SCHEMA",
    "get_schema",
    "register_schema",
    "SCHEMA_REGISTRY",
    # EBI Extractors
    "extract_location_features",
    "extract_location_features_ebi",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
    "NewBotDetectionFeaturesExtractor",
    "TimeSeriesFeaturesExtractor",
    "TimeWindowExtractor",
    # Behavioral - existing
    "extract_behavioral_features",
    "extract_advanced_behavioral_features",
    "add_bot_interaction_features",
    "add_bot_signature_features",
    "ADVANCED_BEHAVIORAL_FEATURES",
    # Behavioral - new
    "extract_timing_precision_features",
    "extract_user_distribution_features",
    "extract_session_behavior_features",
    "NEW_BEHAVIORAL_FEATURES",
    # Discriminative - existing
    "extract_discriminative_features",
    "DISCRIMINATIVE_FEATURES",
    # Discriminative - new
    "extract_access_pattern_features",
    "extract_statistical_anomaly_features",
    "extract_comparative_features",
    "NEW_DISCRIMINATIVE_FEATURES",
    # Time Series
    "extract_outburst_features",
    "extract_periodicity_features",
    "extract_trend_features",
    "extract_recency_features",
    "extract_distribution_shape_features",
    "extract_all_timeseries_features",
    "OUTBURST_FEATURES",
    "PERIODICITY_FEATURES",
    "TREND_FEATURES",
    "RECENCY_FEATURES",
    "DISTRIBUTION_SHAPE_FEATURES",
    "ALL_TIMESERIES_FEATURES",
]
