"""
Example: Using custom schemas and feature extractors for different log formats.

This example shows how to:
1. Define a custom LogSchema for a different log format
2. Create custom feature extractors
3. Use them with the bot detection pipeline
"""

from logghostbuster import (
    LogSchema, 
    BaseFeatureExtractor, 
    run_bot_annotator,
    register_schema
)
import pandas as pd


# Example 1: Define a custom schema for a different log format
# Suppose your logs have different field names
custom_log_schema = LogSchema(
    location_field="ip_geo_coordinates",  # Instead of "geo_location"
    country_field="country_code",         # Instead of "country"
    city_field="city_name",               # Instead of "geoip_city_name"
    user_field="user_id",                 # Instead of "user"
    project_field=None,                   # No project field in this log format
    timestamp_field="event_timestamp",    # Instead of "timestamp"
    year_field="event_year",              # Pre-computed year field
    min_location_downloads=50,            # Lower threshold
    min_year=2020,                        # Different year threshold
    working_hours_start=8,                # Different working hours (8 AM - 5 PM)
    working_hours_end=17,
)

# Register it for easy reuse
register_schema("custom_logs", custom_log_schema)


# Example 2: Create a custom feature extractor
class CustomBandwidthExtractor(BaseFeatureExtractor):
    """Extract bandwidth-related features if available in logs."""
    
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """
        Add bandwidth features if bandwidth field exists in the log.
        """
        # Check if bandwidth field exists (would need to query schema or try/except)
        # For this example, assume we add a feature based on existing data
        
        # Example: Add a feature based on download patterns
        # This is just a placeholder - actual implementation would query the data
        df['avg_download_size'] = 0.0  # Would calculate from actual data
        
        return df


# Example 3: Use custom schema and extractors
def run_with_custom_schema():
    """Run bot detection with custom schema and extractors."""
    
    custom_extractors = [
        CustomBandwidthExtractor(custom_log_schema),
    ]
    
    results = run_bot_annotator(
        input_parquet='path/to/your/custom_logs.parquet',
        output_parquet='path/to/annotated_output.parquet',
        output_dir='output/custom_analysis',
        schema=custom_log_schema,
        custom_extractors=custom_extractors,
        contamination=0.15,
    )
    
    return results


# Example 4: Using predefined schemas
from logghostbuster import get_schema, EBI_SCHEMA

def run_with_ebi_schema():
    """Run with EBI schema (default)."""
    results = run_bot_annotator(
        input_parquet='path/to/ebi_logs.parquet',
        schema=EBI_SCHEMA,  # Explicitly use EBI schema
    )
    return results


def run_with_registered_schema():
    """Run with a schema from the registry."""
    custom_schema = get_schema("custom_logs")
    
    results = run_bot_annotator(
        input_parquet='path/to/custom_logs.parquet',
        schema=custom_schema,
    )
    return results


if __name__ == "__main__":
    # Run examples
    print("Example usage of custom schemas and extractors")
    print("See code comments for details")
