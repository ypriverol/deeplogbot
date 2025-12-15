"""Main feature extraction function."""

import os
import pandas as pd
import numpy as np
from typing import Optional, List

from ..utils import logger
from ..schema import LogSchema, EBI_SCHEMA
from .base import BaseFeatureExtractor
from .standard import (
    YearlyPatternExtractor,
    TimeOfDayExtractor,
    CountryLevelExtractor,
)


def extract_location_features(
    conn, 
    input_parquet: str,
    schema: Optional[LogSchema] = None,
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None
):
    """
    Extract behavioral features per location for ML analysis.
    
    Features capture patterns that distinguish:
    - Bots: many users, low downloads/user, high hourly user density, irregular time patterns
    - Mirrors: few users, high downloads/user, systematic patterns, regular time patterns
    
    Args:
        conn: Database connection (DuckDB)
        input_parquet: Path to input parquet file
        schema: LogSchema defining field mappings (defaults to EBI_SCHEMA)
        custom_extractors: Optional list of custom feature extractors to apply
        
    Returns:
        DataFrame with extracted features
    """
    if schema is None:
        schema = EBI_SCHEMA
    
    logger.info("Extracting location-level features...")
    logger.info(f"  Using schema: {schema.__class__.__name__}")
    
    escaped_path = os.path.abspath(input_parquet).replace("'", "''")
    
    # Step 1: Get basic location stats
    logger.info("  Step 1/4: Basic location statistics...")
    
    # Build year expression
    year_expr = f"EXTRACT(YEAR FROM CAST({schema.timestamp_field} AS TIMESTAMP))"
    if schema.year_field:
        year_expr = schema.year_field
    
    # Build city field expression
    city_expr = f"MAX({schema.city_field})" if schema.city_field else "NULL"
    
    # Build project field expression
    project_expr = ""
    if schema.project_field:
        project_expr = f", COUNT(DISTINCT {schema.project_field}) as unique_projects"
    else:
        project_expr = ", 0 as unique_projects"
    
    location_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        {schema.country_field} as country,
        {city_expr} as city,
        COUNT(DISTINCT {schema.user_field}) as unique_users,
        COUNT(*) as total_downloads,
        CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT {schema.user_field}), 0) as downloads_per_user
        {project_expr},
        COUNT(DISTINCT DATE_TRUNC('hour', CAST({schema.timestamp_field} AS TIMESTAMP))) as active_hours,
        COUNT(DISTINCT {year_expr}) as years_active,
        MIN({year_expr}) as first_year,
        MAX({year_expr}) as last_year
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    AND {year_expr} >= {schema.min_year}
    GROUP BY {schema.location_field}, {schema.country_field}
    HAVING COUNT(*) >= {schema.min_location_downloads}
    """
    
    location_df = conn.execute(location_query).df()
    logger.info(f"  Found {len(location_df):,} locations")
    
    # Step 2: Get hourly patterns
    logger.info("  Step 2/4: Hourly user patterns...")
    hourly_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        AVG(users_per_hour) as avg_users_per_hour,
        MAX(users_per_hour) as max_users_per_hour
    FROM (
        SELECT 
            {schema.location_field} as geo_location,
            DATE_TRUNC('hour', CAST({schema.timestamp_field} AS TIMESTAMP)) as hour_window,
            COUNT(DISTINCT {schema.user_field}) as users_per_hour
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        AND {year_expr} >= {schema.min_year}
        GROUP BY {schema.location_field}, hour_window
    )
    GROUP BY {schema.location_field}
    """
    
    hourly_df = conn.execute(hourly_query).df()
    
    # Merge basic stats
    df = location_df.merge(hourly_df, on='geo_location', how='left')
    
    # Calculate derived features from basic stats
    df['user_cv'] = df['avg_users_per_hour'].fillna(0) / (df['avg_users_per_hour'].mean() + 1e-10)
    df['users_per_active_hour'] = df['unique_users'] / df['active_hours'].replace(0, 1)
    if schema.project_field:
        df['projects_per_user'] = df['unique_projects'] / df['unique_users'].replace(0, 1)
    else:
        df['projects_per_user'] = 0
    
    # Step 3: Apply standard feature extractors
    logger.info("  Step 3/4: Applying feature extractors...")
    
    # Standard extractors (order matters - yearly must come before country-level)
    standard_extractors = [
        YearlyPatternExtractor(schema),
        TimeOfDayExtractor(schema),
        CountryLevelExtractor(schema),
    ]
    
    # Combine with custom extractors if provided
    all_extractors = standard_extractors
    if custom_extractors:
        all_extractors = standard_extractors + custom_extractors
        logger.info(f"  Using {len(custom_extractors)} custom feature extractor(s)")
    
    # Apply extractors in sequence
    for extractor in all_extractors:
        logger.info(f"    Running {extractor.get_name()}...")
        df = extractor.extract(df, escaped_path, conn)
    
    logger.info(f"Extracted features for {len(df):,} locations")
    
    return df


def extract_location_features_ebi(conn, input_parquet):
    """
    Convenience function for EBI log format (backward compatibility).
    
    This is a wrapper that uses the default EBI schema.
    """
    return extract_location_features(conn, input_parquet, schema=EBI_SCHEMA)
