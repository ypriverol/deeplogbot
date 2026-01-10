"""Feature extraction package for logghostbuster."""

from .base import BaseFeatureExtractor
from .schema import LogSchema, get_schema, register_schema, SCHEMA_REGISTRY
from .providers.ebi import (
    # EBI Schema
    LogEbiSchema,
    EBI_SCHEMA,
    # EBI Extractors
    extract_location_features,
    extract_location_features_ebi,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
    TimeWindowExtractor,
    # Behavioral
    extract_behavioral_features,
    extract_advanced_behavioral_features,
    add_bot_interaction_features,
    add_bot_signature_features,
    ADVANCED_BEHAVIORAL_FEATURES,
    # Discriminative
    extract_discriminative_features,
    DISCRIMINATIVE_FEATURES,
)

__all__ = [
    "BaseFeatureExtractor",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
    "extract_location_features",
    "extract_location_features_ebi",
    "extract_behavioral_features",
    "extract_advanced_behavioral_features",
    "add_bot_interaction_features",
    "add_bot_signature_features",
    "ADVANCED_BEHAVIORAL_FEATURES",
    "extract_discriminative_features",
    "DISCRIMINATIVE_FEATURES",
    "LogSchema",
    "LogEbiSchema",
    "EBI_SCHEMA",
    "get_schema",
    "register_schema",
    "SCHEMA_REGISTRY",
]
