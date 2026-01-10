"""EBI-specific feature extractors and utilities.

This package contains all feature extraction logic specific to EBI log data:
- Schema definitions (LogEbiSchema, EBI_SCHEMA)
- Feature extractors (YearlyPatternExtractor, TimeOfDayExtractor, etc.)
- Behavioral feature extraction
- Discriminative feature extraction
- Core EBI extraction functions
"""

from ...schema import LogSchema, get_schema, register_schema, SCHEMA_REGISTRY
from .schema import LogEbiSchema, EBI_SCHEMA
from .ebi import (
    extract_location_features_ebi,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
    TimeWindowExtractor,
    extract_location_features,  # Alias for backward compatibility
)
from .behavioral import (
    extract_behavioral_features,
    extract_advanced_behavioral_features,
    add_bot_interaction_features,
    add_bot_signature_features,
    ADVANCED_BEHAVIORAL_FEATURES,
)
from .discriminative import (
    extract_discriminative_features,
    DISCRIMINATIVE_FEATURES,
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
    "TimeWindowExtractor",
    # Behavioral
    "extract_behavioral_features",
    "extract_advanced_behavioral_features",
    "add_bot_interaction_features",
    "add_bot_signature_features",
    "ADVANCED_BEHAVIORAL_FEATURES",
    # Discriminative
    "extract_discriminative_features",
    "DISCRIMINATIVE_FEATURES",
]
