"""Feature extraction package for logghostbuster."""

from .base import BaseFeatureExtractor
from .providers.ebi import (
    extract_location_features_ebi,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)
from .schema import LogSchema, EBI_SCHEMA, get_schema, register_schema, SCHEMA_REGISTRY

# For backward compatibility, export extract_location_features as alias to EBI version
from .providers.ebi import extract_location_features

__all__ = [
    "BaseFeatureExtractor",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
    "extract_location_features",
    "extract_location_features_ebi",
    "LogSchema",
    "EBI_SCHEMA",
    "get_schema",
    "register_schema",
    "SCHEMA_REGISTRY",
]
