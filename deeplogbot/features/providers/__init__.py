"""Provider-specific feature extractors and convenience functions."""

from .ebi import (
    extract_location_features_ebi,
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)

# This package contains provider-specific feature extractors and convenience functions
# Examples:
# - EBI-specific extractors and convenience functions
# - AWS CloudWatch extractors
# - Custom provider extractors

__all__ = [
    "extract_location_features_ebi",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
]
