"""EBI-specific schema definitions extending the base LogSchema."""

from ...schema import LogSchema


class LogEbiSchema(LogSchema):
    """
    EBI-specific log schema.
    
    Extends the base LogSchema with EBI-specific field mappings and defaults.
    This schema is tailored for EBI download logs.
    """
    
    def __init__(self):
        """Initialize EBI schema with EBI-specific defaults."""
        super().__init__(
            location_field="geo_location",
            country_field="country",
            city_field="geoip_city_name",
            user_field="user",
            project_field="accession",
            method_field="method",
            timestamp_field="timestamp",
            year_field=None,  # Will be extracted from timestamp
            min_location_downloads=1,
            min_year=2020,
            working_hours_start=9,
            working_hours_end=17,
            night_hours_start=22,
            night_hours_end=6,
        )


# Create the default EBI schema instance
EBI_SCHEMA = LogEbiSchema()

# Register in the global schema registry
from ...schema import register_schema
register_schema("ebi", EBI_SCHEMA)
