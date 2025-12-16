"""
EBI-specific feature extractors and extraction functions.

This module contains all EBI-specific code:
- Feature extractors (YearlyPatternExtractor, TimeOfDayExtractor, CountryLevelExtractor)
- Core extraction logic
- Convenience functions
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, List
from scipy import stats

from ..base import BaseFeatureExtractor
from ..schema import LogSchema, EBI_SCHEMA
from ...utils import logger


# ============================================================================
# Feature Extractors
# ============================================================================

class YearlyPatternExtractor(BaseFeatureExtractor):
    """Extract yearly temporal patterns."""
    
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """Extract yearly pattern features."""
        logger.info("  Extracting yearly pattern features...")
        
        escaped_path = input_parquet_path
        
        # Build yearly query based on schema
        year_expr = f"EXTRACT(YEAR FROM CAST({self.schema.timestamp_field} AS TIMESTAMP))"
        if self.schema.year_field:
            year_expr = self.schema.year_field
        
        yearly_query = f"""
        SELECT 
            {self.schema.location_field} as geo_location,
            {year_expr} as year,
            COUNT(*) as downloads_in_year
        FROM read_parquet('{escaped_path}')
        WHERE {self.schema.location_field} IS NOT NULL
        AND {self.schema.timestamp_field} IS NOT NULL
        AND {year_expr} >= {self.schema.min_year}
        GROUP BY {self.schema.location_field}, {year_expr}
        """
        
        yearly_df = conn.execute(yearly_query).df()
        
        if len(yearly_df) == 0:
            # No yearly data, return with zero-filled features
            df['yearly_entropy'] = 0
            df['peak_year_concentration'] = 0
            df['years_span'] = 0
            df['downloads_per_year'] = 0
            df['year_over_year_cv'] = 0
            df['fraction_latest_year'] = 0
            df['is_new_location'] = 0
            df['spike_ratio'] = 0
            df['years_before_latest'] = 0
            df['latest_year_downloads'] = 0
            return df
        
        # Calculate yearly features
        latest_year_global = yearly_df['year'].max()
        
        yearly_agg = yearly_df.groupby('geo_location').agg({
            'year': ['min', 'max', 'count'],
            'downloads_in_year': ['sum', 'max']
        }).reset_index()
        yearly_agg.columns = ['geo_location', 'first_year', 'last_year', 'years_count', 
                              'total_downloads', 'max_year_downloads']
        
        latest_year_data = yearly_df[yearly_df['year'] == latest_year_global].groupby('geo_location')['downloads_in_year'].sum().reset_index()
        latest_year_data.columns = ['geo_location', 'latest_year_downloads']
        yearly_agg = yearly_agg.merge(latest_year_data, on='geo_location', how='left')
        yearly_agg['latest_year_downloads'] = yearly_agg['latest_year_downloads'].fillna(0)
        
        # Merge with main df
        yearly_agg = yearly_agg.merge(
            df[['geo_location', 'years_active', 'total_downloads', 'first_year']], 
            on='geo_location', 
            how='left', 
            suffixes=('', '_main')
        )
        
        # Calculate previous years stats
        prev_years_stats = yearly_df[yearly_df['year'] < latest_year_global].groupby('geo_location').agg({
            'downloads_in_year': ['sum', 'count']
        }).reset_index()
        prev_years_stats.columns = ['geo_location', 'prev_years_dl', 'prev_years_count']
        yearly_agg = yearly_agg.merge(prev_years_stats, on='geo_location', how='left')
        yearly_agg['prev_years_dl'] = yearly_agg['prev_years_dl'].fillna(0)
        yearly_agg['prev_years_count'] = yearly_agg['prev_years_count'].fillna(0)
        yearly_agg['avg_prev_years'] = yearly_agg['prev_years_dl'] / yearly_agg['prev_years_count'].replace(0, 1)
        
        # Calculate features
        yearly_agg['fraction_latest_year'] = np.where(
            yearly_agg['total_downloads'] > 0,
            yearly_agg['latest_year_downloads'] / yearly_agg['total_downloads'],
            0
        )
        yearly_agg['is_new_location'] = np.where(
            (yearly_agg['first_year'] == latest_year_global) & (yearly_agg['years_count'] == 1),
            1, 0
        )
        yearly_agg['spike_ratio'] = np.where(
            yearly_agg['avg_prev_years'] > 0,
            yearly_agg['latest_year_downloads'] / (yearly_agg['avg_prev_years'] + 1e-10),
            np.where(yearly_agg['latest_year_downloads'] > 0, yearly_agg['latest_year_downloads'] / 1000, 0)
        )
        yearly_agg['years_before_latest'] = yearly_agg['prev_years_count']
        
        # Calculate entropy and CV
        yearly_pivot = yearly_df.pivot_table(
            index='geo_location', 
            columns='year', 
            values='downloads_in_year', 
            fill_value=0
        )
        yearly_props = yearly_pivot.div(yearly_pivot.sum(axis=1).replace(0, 1), axis=0).fillna(0)
        yearly_pivot = yearly_pivot.reindex(yearly_agg['geo_location'], fill_value=0)
        yearly_props = yearly_pivot.div(yearly_pivot.sum(axis=1).replace(0, 1), axis=0).fillna(0)
        
        yearly_agg['yearly_entropy'] = yearly_props.apply(
            lambda row: stats.entropy(row.values + 1e-10) if row.sum() > 0 else 0, axis=1
        ).values
        yearly_agg['peak_year_concentration'] = yearly_props.max(axis=1).values
        yearly_agg['year_over_year_cv'] = yearly_props.apply(
            lambda row: (row.std() / (row.mean() + 1e-10)) if len(row[row > 0]) > 1 else 0, axis=1
        ).values
        
        yearly_agg['years_span'] = yearly_agg['years_active'].fillna(yearly_agg['years_count'])
        yearly_agg['downloads_per_year'] = np.where(
            yearly_agg['years_span'] > 0,
            yearly_agg['total_downloads'] / yearly_agg['years_span'],
            0
        )
        
        # Merge back to main df
        yearly_features_df = yearly_agg[[
            'geo_location', 'yearly_entropy', 'peak_year_concentration', 'years_span',
            'downloads_per_year', 'year_over_year_cv', 'fraction_latest_year',
            'is_new_location', 'spike_ratio', 'years_before_latest', 'latest_year_downloads'
        ]].copy()
        
        df = df.merge(yearly_features_df, on='geo_location', how='left')
        
        # Fill NaN values
        yearly_cols = ['yearly_entropy', 'peak_year_concentration', 'years_span',
                       'downloads_per_year', 'year_over_year_cv', 'fraction_latest_year',
                       'is_new_location', 'spike_ratio', 'years_before_latest', 'latest_year_downloads']
        for col in yearly_cols:
            df[col] = df[col].fillna(0)
        
        return df


class TimeOfDayExtractor(BaseFeatureExtractor):
    """Extract time-of-day pattern features."""
    
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """Extract time-of-day features."""
        logger.info("  Extracting time-of-day features...")
        
        escaped_path = input_parquet_path
        
        time_of_day_query = f"""
        SELECT 
            {self.schema.location_field} as geo_location,
            EXTRACT(HOUR FROM CAST({self.schema.timestamp_field} AS TIMESTAMP)) as hour_of_day,
            COUNT(*) as downloads_at_hour
        FROM read_parquet('{escaped_path}')
        WHERE {self.schema.location_field} IS NOT NULL
        AND {self.schema.timestamp_field} IS NOT NULL
        AND EXTRACT(YEAR FROM CAST({self.schema.timestamp_field} AS TIMESTAMP)) >= {self.schema.min_year}
        GROUP BY {self.schema.location_field}, hour_of_day
        """
        
        time_of_day_df = conn.execute(time_of_day_query).df()
        
        time_features = []
        total_locs = len(df)
        for idx, geo_loc in enumerate(df['geo_location']):
            if (idx + 1) % 2000 == 0:
                logger.info(f"    Progress: {idx + 1:,}/{total_locs:,} locations ({100*(idx+1)/total_locs:.1f}%)")
            location_tod = time_of_day_df[time_of_day_df['geo_location'] == geo_loc]
            
            if len(location_tod) == 0:
                time_features.append({
                    'hourly_download_std': 0,
                    'peak_hour_concentration': 0,
                    'working_hours_ratio': 0,
                    'hourly_entropy': 0,
                    'night_activity_ratio': 0
                })
                continue
            
            hourly_dist = location_tod.set_index('hour_of_day')['downloads_at_hour'].reindex(
                range(24), fill_value=0
            ).values
            
            total_dl = hourly_dist.sum()
            if total_dl == 0:
                hourly_props = hourly_dist
            else:
                hourly_props = hourly_dist / total_dl
            
            hourly_download_std = np.std(hourly_dist)
            peak_hour_concentration = np.max(hourly_props)
            
            # Use schema-defined working hours
            working_hours = hourly_dist[self.schema.working_hours_start:self.schema.working_hours_end].sum()
            working_hours_ratio = working_hours / total_dl if total_dl > 0 else 0
            
            hourly_entropy = stats.entropy(hourly_props + 1e-10)
            
            # Use schema-defined night hours
            if self.schema.night_hours_start <= self.schema.night_hours_end:
                night_hours = hourly_dist[self.schema.night_hours_start:self.schema.night_hours_end].sum()
            else:
                # Wraps around midnight
                night_hours = np.concatenate([
                    hourly_dist[self.schema.night_hours_start:24],
                    hourly_dist[0:self.schema.night_hours_end]
                ]).sum()
            night_activity_ratio = night_hours / total_dl if total_dl > 0 else 0
            
            time_features.append({
                'hourly_download_std': hourly_download_std,
                'peak_hour_concentration': peak_hour_concentration,
                'working_hours_ratio': working_hours_ratio,
                'hourly_entropy': hourly_entropy,
                'night_activity_ratio': night_activity_ratio
            })
        
        time_features_df = pd.DataFrame(time_features)
        time_features_df.insert(0, 'geo_location', df['geo_location'].values)
        df = df.merge(time_features_df, on='geo_location', how='left')
        
        time_cols = ['hourly_download_std', 'peak_hour_concentration', 'working_hours_ratio', 
                     'hourly_entropy', 'night_activity_ratio']
        for col in time_cols:
            df[col] = df[col].fillna(0)
        
        return df


class CountryLevelExtractor(BaseFeatureExtractor):
    """Extract country-level aggregated features."""
    
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        """Extract country-level features for coordinated bot detection."""
        logger.info("  Calculating country-level features for coordinated bot detection...")
        
        # Note: country_field is renamed to 'country' in feature_extraction.py
        country_stats = df.groupby('country').agg({
            'geo_location': 'count',
            'latest_year_downloads': 'sum',
            'total_downloads': 'sum',
            'fraction_latest_year': 'mean',
            'is_new_location': 'sum',
            'spike_ratio': lambda x: (x > 1.5).sum(),
            'downloads_per_user': lambda x: (x < 30).sum(),
            'unique_users': 'sum',
        }).reset_index()
        country_stats.columns = [
            'country', 'locations_per_country', 'country_latest_year_dl', 'country_total_dl',
            'country_avg_fraction_latest', 'country_new_locations', 'country_high_spike_locations',
            'country_low_dl_user_locations', 'country_total_users'
        ]
        
        country_stats['country_fraction_latest'] = np.where(
            country_stats['country_total_dl'] > 0,
            country_stats['country_latest_year_dl'] / country_stats['country_total_dl'],
            0
        )
        country_stats['country_new_location_ratio'] = np.where(
            country_stats['locations_per_country'] > 0,
            country_stats['country_new_locations'] / country_stats['locations_per_country'],
            0
        )
        country_stats['country_suspicious_location_ratio'] = np.where(
            country_stats['locations_per_country'] > 0,
            (country_stats['country_high_spike_locations'] + country_stats['country_low_dl_user_locations']) / 
            (country_stats['locations_per_country'] * 2),
            0
        )
        
        df = df.merge(country_stats, on='country', how='left')
        
        country_cols = ['locations_per_country', 'country_latest_year_dl', 'country_total_dl',
                        'country_avg_fraction_latest', 'country_new_locations', 'country_high_spike_locations',
                        'country_low_dl_user_locations', 'country_total_users', 'country_fraction_latest',
                        'country_new_location_ratio', 'country_suspicious_location_ratio']
        for col in country_cols:
            df[col] = df[col].fillna(0)
        
        return df


# ============================================================================
# Core Extraction Function (Internal)
# ============================================================================

def _extract_location_features_core(
    conn, 
    input_parquet: str,
    schema: LogSchema,
    extractors: List[BaseFeatureExtractor],
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None
):
    """
    EBI core feature extraction function (internal).
    
    This function implements EBI-specific extraction logic:
    - Groups records by geo_location + country
    - Calculates basic location statistics
    - Extracts hourly user patterns
    - Applies EBI-specific feature extractors
    
    Args:
        conn: Database connection (DuckDB)
        input_parquet: Path to input parquet file
        schema: LogSchema defining field mappings (EBI_SCHEMA)
        extractors: List of EBI feature extractors to apply
        custom_extractors: Optional additional custom extractors
    
    Returns:
        DataFrame with extracted features
    """
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
    
    # Step 3: Apply feature extractors
    logger.info("  Step 3/4: Applying feature extractors...")
    
    # Combine extractors
    all_extractors = extractors
    if custom_extractors:
        all_extractors = extractors + custom_extractors
        logger.info(f"  Using {len(custom_extractors)} custom feature extractor(s)")
    
    # Apply extractors in sequence
    for extractor in all_extractors:
        logger.info(f"    Running {extractor.get_name()}...")
        df = extractor.extract(df, escaped_path, conn)
    
    logger.info(f"Extracted features for {len(df):,} locations")
    
    return df


# ============================================================================
# Public Extraction Functions
# ============================================================================

def extract_location_features(
    conn, 
    input_parquet: str,
    schema: Optional[LogSchema] = None,
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None
):
    """
    EBI-specific feature extraction function.
    
    Extracts behavioral features per location using EBI-tailored extractors:
    - YearlyPatternExtractor: Yearly temporal patterns
    - TimeOfDayExtractor: Time-of-day patterns
    - CountryLevelExtractor: Country-level aggregations
    
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
    
    # EBI-specific extractors (order matters - yearly must come before country-level)
    ebi_extractors = [
        YearlyPatternExtractor(schema),
        TimeOfDayExtractor(schema),
        CountryLevelExtractor(schema),
    ]
    
    # Use EBI core extraction function with EBI extractors
    return _extract_location_features_core(conn, input_parquet, schema, ebi_extractors, custom_extractors)


def extract_location_features_ebi(conn, input_parquet):
    """
    Convenience function for EBI log format (backward compatibility).
    
    This uses EBI-specific extraction logic with EBI schema and EBI-tailored extractors.
    
    Args:
        conn: Database connection (DuckDB)
        input_parquet: Path to input parquet file
    
    Returns:
        DataFrame with extracted features using EBI schema and EBI extractors
    """
    return extract_location_features(conn, input_parquet, schema=EBI_SCHEMA)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "extract_location_features",
    "extract_location_features_ebi",
    "YearlyPatternExtractor",
    "TimeOfDayExtractor",
    "CountryLevelExtractor",
]
