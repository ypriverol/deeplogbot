"""Schema configuration for different log formats."""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LogSchema:
    """
    Schema definition for log data fields.
    
    This allows the feature extraction to work with different log formats
    by specifying field mappings.
    """
    # Location/geographic fields
    location_field: str = "geo_location"
    country_field: str = "country"
    city_field: Optional[str] = "geoip_city_name"
    
    # User/entity fields
    user_field: str = "user"
    project_field: Optional[str] = "accession"  # Optional, can be None for logs without projects
    
    # Temporal fields
    timestamp_field: str = "timestamp"
    year_field: Optional[str] = "year"  # Optional, will be extracted from timestamp if None
    
    # Filtering thresholds
    min_location_downloads: int = 1  # Minimum downloads for a location to be considered (default: 1 to include all users)
    min_year: int = 2020  # Minimum year to include in analysis
    
    # Time zone settings
    working_hours_start: int = 9  # UTC hour for working hours start (0-23)
    working_hours_end: int = 17  # UTC hour for working hours end (0-23)
    night_hours_start: int = 22  # UTC hour for night hours start (0-23)
    night_hours_end: int = 6  # UTC hour for night hours end (0-23)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for serialization."""
        return {
            "location_field": self.location_field,
            "country_field": self.country_field,
            "city_field": self.city_field,
            "user_field": self.user_field,
            "project_field": self.project_field,
            "timestamp_field": self.timestamp_field,
            "year_field": self.year_field,
            "min_location_downloads": self.min_location_downloads,
            "min_year": self.min_year,
            "working_hours_start": self.working_hours_start,
            "working_hours_end": self.working_hours_end,
            "night_hours_start": self.night_hours_start,
            "night_hours_end": self.night_hours_end,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogSchema":
        """Create schema from dictionary."""
        return cls(**data)


# Predefined schemas for common log formats
EBI_SCHEMA = LogSchema(
    location_field="geo_location",
    country_field="country",
    city_field="geoip_city_name",
    user_field="user",
    project_field="accession",
    timestamp_field="timestamp",
    year_field="year",
    min_location_downloads=1,  # Include all locations (can be increased to filter noise)
    min_year=2020,
)


# Registry of available schemas
SCHEMA_REGISTRY: Dict[str, LogSchema] = {
    "ebi": EBI_SCHEMA,
}


def get_schema(name: str) -> LogSchema:
    """Get a schema by name from the registry."""
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown schema: {name}. Available schemas: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[name]


def register_schema(name: str, schema: LogSchema):
    """Register a custom schema."""
    SCHEMA_REGISTRY[name] = schema
