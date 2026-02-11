"""Behavioral feature extraction for bot detection.

This module contains functions to compute behavioral features that capture
intrinsic download patterns rather than rule-based thresholds. These features
are more stable across time and require less maintenance.

Key behavioral features:
- Temporal patterns (entropy, regularity, concentration)
- Access patterns (velocity, session anomalies)
- User patterns (coordination, diversity)
- File patterns (diversity, concentration)
"""

import pandas as pd
import numpy as np

from ....utils import logger
from .schema import EBI_SCHEMA


# Constants for feature computation
ANOMALY_SCORE_OFFSET = 0.5
LOG_USERS_OFFSET = 2
EPSILON = 1e-10

# Composite score weights
COMPOSITE_SCORE_WEIGHTS = {
    'DL_USER_PER_LOG_USERS': 0.3,
    'USER_SCARCITY': 0.25,
    'DOWNLOAD_CONCENTRATION': 0.25,
    'ANOMALY_SCORE': 0.2,
}

# Bot thresholds (for feature computation)
BOT_THRESHOLDS = {
    'FEW_USERS': 100,
    'MODERATE_DL_PER_USER': 20,
}


def extract_behavioral_features(df: pd.DataFrame, input_parquet: str, conn) -> pd.DataFrame:
    """
    Extract behavioral features from raw download data.
    
    These features capture behavioral signatures that distinguish patterns like:
    - Pipeline/CI: Regular intervals, same files
    - Research group: Working hours, diverse files
    - Automated sync: Daily/weekly patterns
    
    Args:
        df: DataFrame with basic features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        
    Returns:
        DataFrame with behavioral features added
    """
    schema = EBI_SCHEMA
    escaped_path = input_parquet.replace("'", "''")
    
    # Feature 1: Temporal regularity (how mechanical are the download times?)
    logger.info("  Extracting temporal regularity features...")
    
    temporal_query = f"""
    WITH sampled_downloads AS (
        SELECT 
            {schema.location_field} as geo_location,
            {schema.timestamp_field} as download_time,
            ROW_NUMBER() OVER (PARTITION BY {schema.location_field} ORDER BY RANDOM()) as rn
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    limited_downloads AS (
        -- Sample up to 50 downloads per location to reduce memory/disk usage
        SELECT geo_location, download_time 
        FROM sampled_downloads 
        WHERE rn <= 50
    ),
    download_intervals AS (
        SELECT 
            geo_location,
            EPOCH(CAST(download_time AS TIMESTAMP) - 
                  LAG(CAST(download_time AS TIMESTAMP)) 
                  OVER (PARTITION BY geo_location ORDER BY download_time)) as interval_seconds
        FROM limited_downloads
    )
    SELECT 
        geo_location,
        AVG(interval_seconds) as mean_interval,
        STDDEV(interval_seconds) as std_interval,
        -- Coefficient of variation: low = mechanical, high = random
        CASE 
            WHEN AVG(interval_seconds) > 0 
            THEN STDDEV(interval_seconds) / AVG(interval_seconds) 
            ELSE 1.0 
        END as interval_cv,
        -- Regularity score: inverse of CV (high = mechanical)
        CASE 
            WHEN STDDEV(interval_seconds) > 0 
            THEN AVG(interval_seconds) / (STDDEV(interval_seconds) + 1) 
            ELSE 0 
        END as regularity_score
    FROM download_intervals
    WHERE interval_seconds IS NOT NULL AND interval_seconds > 0
    GROUP BY geo_location
    """
    
    try:
        temporal_df = conn.execute(temporal_query).df()
        df = df.merge(temporal_df, on='geo_location', how='left')
        df['regularity_score'] = df['regularity_score'].fillna(0)
        df['interval_cv'] = df['interval_cv'].fillna(1)
    except Exception as e:
        logger.warning(f"  Temporal features extraction failed: {e}")
        df['regularity_score'] = 0
        df['interval_cv'] = 1
    
    # Feature 2: Day-of-week patterns (weekday vs weekend)
    logger.info("  Extracting day-of-week patterns...")
    
    dow_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        -- Weekend ratio: 0 = all weekday, 1 = all weekend
        AVG(CASE 
            WHEN DAYOFWEEK(CAST({schema.timestamp_field} AS TIMESTAMP)) IN (0, 6) 
            THEN 1.0 ELSE 0.0 
        END) as weekend_ratio,
        -- Weekday concentration: how concentrated on specific days?
        COUNT(DISTINCT DAYOFWEEK(CAST({schema.timestamp_field} AS TIMESTAMP))) as unique_days_of_week
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY {schema.location_field}
    """
    
    try:
        dow_df = conn.execute(dow_query).df()
        df = df.merge(dow_df, on='geo_location', how='left')
        df['weekend_ratio'] = df['weekend_ratio'].fillna(0.3)
        df['unique_days_of_week'] = df['unique_days_of_week'].fillna(5)
    except Exception as e:
        logger.warning(f"  Day-of-week features extraction failed: {e}")
        df['weekend_ratio'] = 0.3
        df['unique_days_of_week'] = 5
    
    # Feature 3: File diversity (same files vs diverse files)
    logger.info("  Extracting file diversity features...")
    
    file_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        COUNT(DISTINCT filename) as unique_files,
        COUNT(*) as total_downloads,
        -- File concentration: 1/unique_files (high = few files, low = many files)
        1.0 / NULLIF(COUNT(DISTINCT filename), 0) as file_concentration,
        -- File diversity ratio: unique_files / total_downloads
        CAST(COUNT(DISTINCT filename) AS DOUBLE) / COUNT(*) as file_diversity_ratio
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    GROUP BY {schema.location_field}
    """
    
    try:
        file_df = conn.execute(file_query).df()
        df = df.merge(
            file_df[['geo_location', 'unique_files', 'file_concentration', 'file_diversity_ratio']], 
            on='geo_location', 
            how='left'
        )
        df['file_diversity_ratio'] = df['file_diversity_ratio'].fillna(1)
        df['file_concentration'] = df['file_concentration'].fillna(1)
    except Exception as e:
        logger.warning(f"  File diversity features extraction failed: {e}")
        df['file_diversity_ratio'] = 1
        df['file_concentration'] = 1
    
    # Feature 4: Session patterns (do downloads come in bursts/sessions?)
    logger.info("  Extracting session pattern features...")
    
    # Define a session as downloads within 30 minutes of each other
    session_query = f"""
    WITH numbered_downloads AS (
        SELECT 
            {schema.location_field} as geo_location,
            CAST({schema.timestamp_field} AS TIMESTAMP) as ts,
            ROW_NUMBER() OVER (PARTITION BY {schema.location_field} ORDER BY {schema.timestamp_field}) as rn
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    session_breaks AS (
        SELECT 
            a.geo_location,
            CASE 
                WHEN EPOCH(a.ts - b.ts) > 1800 THEN 1  -- 30 min gap = new session
                ELSE 0 
            END as is_new_session
        FROM numbered_downloads a
        LEFT JOIN numbered_downloads b ON a.geo_location = b.geo_location AND a.rn = b.rn + 1
    )
    SELECT 
        geo_location,
        SUM(is_new_session) + 1 as num_sessions,
        COUNT(*) as total_in_sessions
    FROM session_breaks
    GROUP BY geo_location
    """
    
    try:
        session_df = conn.execute(session_query).df()
        session_df['downloads_per_session'] = session_df['total_in_sessions'] / session_df['num_sessions'].replace(0, 1)
        df = df.merge(
            session_df[['geo_location', 'num_sessions', 'downloads_per_session']], 
            on='geo_location', 
            how='left'
        )
        df['num_sessions'] = df['num_sessions'].fillna(1)
        # Use total_downloads if available, otherwise use 0
        if 'total_downloads' in df.columns:
            df['downloads_per_session'] = df['downloads_per_session'].fillna(df['total_downloads'])
        else:
            df['downloads_per_session'] = df['downloads_per_session'].fillna(0)
    except Exception as e:
        logger.warning(f"  Session pattern features extraction failed: {e}")
        df['num_sessions'] = 1
        df['downloads_per_session'] = df['total_downloads']
    
    # Derived behavioral features
    df['is_mechanical'] = (df['regularity_score'] > df['regularity_score'].quantile(0.75)).astype(float)
    df['is_weekday_biased'] = (df['weekend_ratio'] < 0.15).astype(float)
    df['is_single_file'] = (df.get('file_diversity_ratio', 0) < 0.1).astype(float)
    df['is_bursty'] = (df['downloads_per_session'] > df['downloads_per_session'].quantile(0.75)).astype(float)
    
    logger.info("  Behavioral features extracted successfully")
    
    return df


def extract_advanced_behavioral_features(
    df: pd.DataFrame, 
    input_parquet: str, 
    conn,
    schema=None
) -> pd.DataFrame:
    """
    Extract advanced behavioral features for bot detection.
    
    These features capture intrinsic behavioral patterns that remain stable
    over time and don't require manual threshold tuning.
    
    Features added:
    - burst_pattern_score: Detects spike → silence cycles (bot behavior)
    - circadian_rhythm_deviation: Measures deviation from human activity patterns
    - user_coordination_score: Detects synchronized bot farm activity
    
    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings (defaults to EBI_SCHEMA)
        
    Returns:
        DataFrame with advanced behavioral features added
    """
    if schema is None:
        schema = EBI_SCHEMA
    
    escaped_path = input_parquet.replace("'", "''")
    
    logger.info("  Extracting advanced behavioral features...")
    
    # =========================================================================
    # Feature 1: Burst Pattern Score
    # =========================================================================
    # Detects "spike then silence" patterns characteristic of bots
    logger.info("    Extracting burst pattern features...")
    
    burst_query = f"""
    WITH hourly_activity AS (
        SELECT 
            {schema.location_field} as geo_location,
            DATE_TRUNC('hour', CAST({schema.timestamp_field} AS TIMESTAMP)) as hour,
            COUNT(*) as downloads_in_hour
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
    ),
    activity_stats AS (
        SELECT 
            geo_location,
            AVG(downloads_in_hour) as mean_hourly,
            STDDEV(downloads_in_hour) as std_hourly,
            MAX(downloads_in_hour) as max_hourly,
            MIN(downloads_in_hour) as min_hourly,
            COUNT(*) as active_hours
        FROM hourly_activity
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        -- Burst score: (max - mean) / std (high = bursty behavior)
        CASE 
            WHEN std_hourly > 0 
            THEN (max_hourly - mean_hourly) / std_hourly 
            ELSE 0 
        END as burst_pattern_score,
        -- Coefficient of variation (high = irregular bursts)
        CASE 
            WHEN mean_hourly > 0 
            THEN std_hourly / mean_hourly 
            ELSE 0 
        END as hourly_cv_burst,
        -- Spike intensity: max / mean (high = concentrated spikes)
        CASE 
            WHEN mean_hourly > 0 
            THEN max_hourly / mean_hourly 
            ELSE 1 
        END as spike_intensity
    FROM activity_stats
    WHERE active_hours > 1  -- Need multiple hours for meaningful stats
    """
    
    try:
        burst_df = conn.execute(burst_query).df()
        df = df.merge(burst_df, on='geo_location', how='left')
        df['burst_pattern_score'] = df['burst_pattern_score'].fillna(0)
        df['hourly_cv_burst'] = df['hourly_cv_burst'].fillna(1)
        df['spike_intensity'] = df['spike_intensity'].fillna(1)
        logger.info(f"      ✓ Burst patterns extracted for {len(burst_df)} locations")
    except Exception as e:
        logger.warning(f"      ✗ Burst pattern extraction failed: {e}")
        df['burst_pattern_score'] = 0
        df['hourly_cv_burst'] = 1
        df['spike_intensity'] = 1
    
    # =========================================================================
    # Feature 2: Circadian Rhythm Deviation
    # =========================================================================
    # Measures how much activity deviates from human circadian rhythms
    logger.info("    Extracting circadian rhythm features...")
    
    circadian_query = f"""
    WITH hourly_dist AS (
        SELECT 
            {schema.location_field} as geo_location,
            EXTRACT(HOUR FROM CAST({schema.timestamp_field} AS TIMESTAMP)) as hour,
            COUNT(*) as count
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
    ),
    time_period_counts AS (
        SELECT 
            geo_location,
            -- Night (0-5): Humans sleep, bots don't
            SUM(CASE WHEN hour BETWEEN 0 AND 5 THEN count ELSE 0 END) as night_count,
            -- Morning (6-8): Humans wake up
            SUM(CASE WHEN hour BETWEEN 6 AND 8 THEN count ELSE 0 END) as morning_count,
            -- Work hours (9-17): Peak human activity
            SUM(CASE WHEN hour BETWEEN 9 AND 17 THEN count ELSE 0 END) as work_count,
            -- Evening (18-23): Humans wind down
            SUM(CASE WHEN hour BETWEEN 18 AND 23 THEN count ELSE 0 END) as evening_count,
            SUM(count) as total_count
        FROM hourly_dist
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        -- Expected human pattern: ~10% night, ~50% work, ~25% evening, ~15% morning
        -- Deviation = sum of absolute differences from expected proportions
        ABS((night_count::FLOAT / total_count) - 0.10) + 
        ABS((work_count::FLOAT / total_count) - 0.50) + 
        ABS((evening_count::FLOAT / total_count) - 0.25) + 
        ABS((morning_count::FLOAT / total_count) - 0.15) as circadian_rhythm_deviation,
        -- Individual ratios for analysis
        (night_count::FLOAT / total_count) as night_ratio_advanced,
        (work_count::FLOAT / total_count) as work_ratio_advanced,
        (evening_count::FLOAT / total_count) as evening_ratio,
        (morning_count::FLOAT / total_count) as morning_ratio
    FROM time_period_counts
    WHERE total_count > 0
    """
    
    try:
        circadian_df = conn.execute(circadian_query).df()
        df = df.merge(circadian_df, on='geo_location', how='left')
        # High deviation = non-human pattern
        df['circadian_rhythm_deviation'] = df['circadian_rhythm_deviation'].fillna(0.5)
        df['night_ratio_advanced'] = df['night_ratio_advanced'].fillna(0.1)
        df['work_ratio_advanced'] = df['work_ratio_advanced'].fillna(0.5)
        df['evening_ratio'] = df['evening_ratio'].fillna(0.25)
        df['morning_ratio'] = df['morning_ratio'].fillna(0.15)
        logger.info(f"      ✓ Circadian rhythms extracted for {len(circadian_df)} locations")
    except Exception as e:
        logger.warning(f"      ✗ Circadian rhythm extraction failed: {e}")
        df['circadian_rhythm_deviation'] = 0.5
        df['night_ratio_advanced'] = 0.1
        df['work_ratio_advanced'] = 0.5
        df['evening_ratio'] = 0.25
        df['morning_ratio'] = 0.15
    
    # =========================================================================
    # Feature 3: User Coordination Score
    # =========================================================================
    # Detects synchronized activity across multiple users (bot farm signature)
    logger.info("    Extracting user coordination features...")
    
    coordination_query = f"""
    WITH user_hourly_activity AS (
        SELECT 
            {schema.location_field} as geo_location,
            {schema.user_field} as user_id,
            DATE_TRUNC('hour', CAST({schema.timestamp_field} AS TIMESTAMP)) as hour,
            COUNT(*) as downloads
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.user_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    concurrent_users AS (
        SELECT 
            geo_location,
            hour,
            COUNT(DISTINCT user_id) as concurrent_users
        FROM user_hourly_activity
        GROUP BY 1, 2
    ),
    coordination_stats AS (
        SELECT 
            geo_location,
            -- Low stddev with many users = synchronized (bot farm)
            STDDEV(concurrent_users) as user_coordination_std,
            AVG(concurrent_users) as avg_concurrent_users,
            MAX(concurrent_users) as max_concurrent_users,
            COUNT(DISTINCT hour) as active_hours
        FROM concurrent_users
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        user_coordination_std,
        avg_concurrent_users,
        max_concurrent_users,
        -- Coordination score: high average with low variance = coordinated
        CASE 
            WHEN user_coordination_std > 0 
            THEN avg_concurrent_users / user_coordination_std 
            ELSE 0 
        END as user_coordination_score,
        -- Peak ratio: max / avg (high = synchronized spikes)
        CASE 
            WHEN avg_concurrent_users > 0 
            THEN max_concurrent_users / avg_concurrent_users 
            ELSE 1 
        END as user_peak_ratio
    FROM coordination_stats
    WHERE active_hours > 1
    """
    
    try:
        coord_df = conn.execute(coordination_query).df()
        df = df.merge(coord_df, on='geo_location', how='left')
        df['user_coordination_score'] = df['user_coordination_score'].fillna(0)
        df['user_peak_ratio'] = df['user_peak_ratio'].fillna(1)
        df['user_coordination_std'] = df['user_coordination_std'].fillna(0)
        df['avg_concurrent_users'] = df['avg_concurrent_users'].fillna(0)
        df['max_concurrent_users'] = df['max_concurrent_users'].fillna(0)
        logger.info(f"      ✓ User coordination extracted for {len(coord_df)} locations")
    except Exception as e:
        logger.warning(f"      ✗ User coordination extraction failed: {e}")
        df['user_coordination_score'] = 0
        df['user_peak_ratio'] = 1
        df['user_coordination_std'] = 0
        df['avg_concurrent_users'] = 0
        df['max_concurrent_users'] = 0
    
    # =========================================================================
    # Derived Boolean Flags (for interpretability)
    # =========================================================================
    if 'burst_pattern_score' in df.columns:
        df['is_bursty_advanced'] = (df['burst_pattern_score'] > df['burst_pattern_score'].quantile(0.75)).astype(float)
    else:
        df['is_bursty_advanced'] = 0.0
    
    if 'night_ratio_advanced' in df.columns:
        df['is_nocturnal'] = (df['night_ratio_advanced'] > 0.25).astype(float)
    else:
        df['is_nocturnal'] = 0.0
    
    if 'user_coordination_score' in df.columns:
        df['is_coordinated'] = (df['user_coordination_score'] > df['user_coordination_score'].quantile(0.75)).astype(float)
    else:
        df['is_coordinated'] = 0.0
    
    logger.info("    ✓ Advanced behavioral features extraction complete")
    
    # Summary statistics
    if 'is_bursty_advanced' in df.columns:
        logger.info(f"      Bursty locations: {df['is_bursty_advanced'].sum():,} ({df['is_bursty_advanced'].mean()*100:.1f}%)")
    if 'is_nocturnal' in df.columns:
        logger.info(f"      Nocturnal locations: {df['is_nocturnal'].sum():,} ({df['is_nocturnal'].mean()*100:.1f}%)")
    if 'is_coordinated' in df.columns:
        logger.info(f"      Coordinated locations: {df['is_coordinated'].sum():,} ({df['is_coordinated'].mean()*100:.1f}%)")
    
    return df


# List of advanced behavioral features for reference
ADVANCED_BEHAVIORAL_FEATURES = [
    'burst_pattern_score',
    'hourly_cv_burst',
    'spike_intensity',
    'circadian_rhythm_deviation',
    'night_ratio_advanced',
    'work_ratio_advanced',
    'evening_ratio',
    'morning_ratio',
    'user_coordination_score',
    'user_peak_ratio',
    'user_coordination_std',
    'avg_concurrent_users',
    'max_concurrent_users',
    'is_bursty_advanced',
    'is_nocturnal',
    'is_coordinated',
]


def add_bot_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features for better bot detection.
    
    These features combine multiple base features to capture complex patterns.
    
    Args:
        df: DataFrame with location features
        
    Returns:
        DataFrame with additional interaction features
    """
    def _has_required_columns(df, *cols):
        """Check if all required columns exist."""
        return all(col in df.columns for col in cols)
    
    # Core bot pattern: High DL/user with few users
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        df['dl_user_per_log_users'] = df['downloads_per_user'] / np.log(df['unique_users'] + LOG_USERS_OFFSET)
        
        df['user_scarcity_score'] = np.where(
            df['downloads_per_user'] > BOT_THRESHOLDS['MODERATE_DL_PER_USER'],
            np.exp(-df['unique_users'] / BOT_THRESHOLDS['FEW_USERS']),
            0
        )
        
        df['download_concentration'] = df['downloads_per_user'] * (1 / (df['unique_users'] + 1))
    
    # Anomaly-weighted features
    if 'anomaly_score' in df.columns and 'downloads_per_user' in df.columns:
        df['anomaly_dl_interaction'] = (df['anomaly_score'] + ANOMALY_SCORE_OFFSET).clip(0, 1) * df['downloads_per_user']
    
    # Temporal features
    if _has_required_columns(df, 'hourly_entropy', 'downloads_per_user'):
        df['temporal_irregularity'] = (1 / (df['hourly_entropy'] + 0.1)) * np.log(df['downloads_per_user'] + 1)
    
    # Composite bot score
    score_components = []
    weights = []
    
    if 'dl_user_per_log_users' in df.columns:
        max_val = df['dl_user_per_log_users'].quantile(0.95)
        if max_val > EPSILON:
            score_components.append(np.clip(df['dl_user_per_log_users'] / max_val, 0, 1))
            weights.append(COMPOSITE_SCORE_WEIGHTS['DL_USER_PER_LOG_USERS'])
    
    if 'user_scarcity_score' in df.columns:
        score_components.append(df['user_scarcity_score'])
        weights.append(COMPOSITE_SCORE_WEIGHTS['USER_SCARCITY'])
    
    if 'download_concentration' in df.columns:
        max_val = df['download_concentration'].quantile(0.95)
        if max_val > EPSILON:
            score_components.append(np.clip(df['download_concentration'] / max_val, 0, 1))
            weights.append(COMPOSITE_SCORE_WEIGHTS['DOWNLOAD_CONCENTRATION'])
    
    if 'anomaly_score' in df.columns:
        score_components.append((df['anomaly_score'] + ANOMALY_SCORE_OFFSET).clip(0, 1))
        weights.append(COMPOSITE_SCORE_WEIGHTS['ANOMALY_SCORE'])
    
    if score_components:
        weights = np.array(weights) / np.sum(weights)
        df['bot_composite_score'] = sum(w * s for w, s in zip(weights, score_components))
    
    return df


def extract_timing_precision_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema=None
) -> pd.DataFrame:
    """
    Extract features that detect mechanical/scheduled bot timing patterns.

    These features capture timing precision that distinguishes bots from humans:
    - Bots often operate on fixed schedules (every 60s, on round seconds)
    - Bots may lack sub-second timestamp precision
    - Bots have very regular intervals between requests

    Features computed:
    - request_interval_mode: Most common interval between requests
    - round_second_ratio: Fraction of requests on round seconds (:00, :15, :30, :45)
    - millisecond_variance: Variance of millisecond component
    - interval_entropy: Entropy of interval distribution

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with timing precision features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting timing precision features...")

    # Query for timing precision analysis
    timing_query = f"""
    WITH sampled_downloads AS (
        SELECT
            {schema.location_field} as geo_location,
            CAST({schema.timestamp_field} AS TIMESTAMP) as ts,
            ROW_NUMBER() OVER (PARTITION BY {schema.location_field} ORDER BY RANDOM()) as rn
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    limited_downloads AS (
        SELECT geo_location, ts
        FROM sampled_downloads
        WHERE rn <= 100  -- Sample for efficiency
    ),
    timing_analysis AS (
        SELECT
            geo_location,
            ts,
            EXTRACT(SECOND FROM ts) as second_of_minute,
            EXTRACT(MILLISECOND FROM ts) as millisecond,
            EPOCH(ts - LAG(ts) OVER (PARTITION BY geo_location ORDER BY ts)) as interval_seconds
        FROM limited_downloads
    ),
    interval_stats AS (
        SELECT
            geo_location,
            -- Mode approximation: most common rounded interval
            MODE(ROUND(interval_seconds / 10) * 10) as interval_mode_10s,
            -- Round second detection (:00, :15, :30, :45)
            AVG(CASE
                WHEN second_of_minute IN (0, 15, 30, 45) THEN 1.0
                ELSE 0.0
            END) as round_second_ratio,
            -- Millisecond variance (0 means no sub-second precision)
            VARIANCE(millisecond) as millisecond_variance,
            -- Interval statistics for entropy
            AVG(interval_seconds) as mean_interval,
            STDDEV(interval_seconds) as std_interval,
            COUNT(*) as sample_count
        FROM timing_analysis
        WHERE interval_seconds IS NOT NULL
        AND interval_seconds > 0
        AND interval_seconds < 86400  -- Filter out > 1 day gaps
        GROUP BY geo_location
    )
    SELECT
        geo_location,
        COALESCE(interval_mode_10s, 0) as request_interval_mode,
        COALESCE(round_second_ratio, 0.25) as round_second_ratio,
        COALESCE(millisecond_variance, 250000) as millisecond_variance,
        -- Entropy approximation: CV-based (lower CV = lower entropy = more mechanical)
        CASE
            WHEN mean_interval > 0 AND std_interval IS NOT NULL
            THEN LOG2(GREATEST(std_interval / mean_interval + 1, 1))
            ELSE 1.0
        END as interval_entropy
    FROM interval_stats
    WHERE sample_count >= 5
    """

    try:
        timing_df = conn.execute(timing_query).df()
        df = df.merge(timing_df, on='geo_location', how='left')

        # Fill defaults for locations without timing data
        df['request_interval_mode'] = df['request_interval_mode'].fillna(0)
        df['round_second_ratio'] = df['round_second_ratio'].fillna(0.25)  # Random baseline
        df['millisecond_variance'] = df['millisecond_variance'].fillna(250000)  # High variance = random
        df['interval_entropy'] = df['interval_entropy'].fillna(1.0)

        logger.info(f"    ✓ Timing precision features extracted for {len(timing_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Timing precision extraction failed: {e}")
        df['request_interval_mode'] = 0
        df['round_second_ratio'] = 0.25
        df['millisecond_variance'] = 250000
        df['interval_entropy'] = 1.0

    return df


def extract_user_distribution_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema=None
) -> pd.DataFrame:
    """
    Extract features that detect bot farms vs legitimate users.

    Bot farms create many fake users with similar behavior patterns.
    Legitimate usage shows natural diversity in user download patterns.

    Features computed:
    - user_entropy: Shannon entropy of downloads across users
    - user_gini_coefficient: Gini coefficient of download distribution
    - single_download_user_ratio: Fraction of users with only 1 download
    - power_user_ratio: Fraction of downloads from top 10% users

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with user distribution features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting user distribution features...")

    user_dist_query = f"""
    WITH user_downloads AS (
        SELECT
            {schema.location_field} as geo_location,
            {schema.user_field} as user_id,
            COUNT(*) as download_count
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.user_field} IS NOT NULL
        GROUP BY 1, 2
    ),
    location_stats AS (
        SELECT
            geo_location,
            COUNT(DISTINCT user_id) as total_users,
            SUM(download_count) as total_downloads,
            -- Single download users
            SUM(CASE WHEN download_count = 1 THEN 1 ELSE 0 END) as single_dl_users,
            -- For Gini: we need sorted cumulative sums
            ARRAY_AGG(download_count ORDER BY download_count) as sorted_downloads
        FROM user_downloads
        GROUP BY geo_location
    ),
    entropy_calc AS (
        SELECT
            ud.geo_location,
            -- Shannon entropy
            -SUM(
                (ud.download_count::FLOAT / ls.total_downloads) *
                LOG2(ud.download_count::FLOAT / ls.total_downloads + 1e-10)
            ) as user_entropy
        FROM user_downloads ud
        JOIN location_stats ls ON ud.geo_location = ls.geo_location
        WHERE ls.total_downloads > 0
        GROUP BY ud.geo_location
    ),
    power_user_calc AS (
        SELECT
            geo_location,
            -- Top 10% users contribution
            SUM(CASE
                WHEN user_rank <= GREATEST(total_users * 0.1, 1)
                THEN download_count
                ELSE 0
            END)::FLOAT / SUM(download_count) as power_user_ratio
        FROM (
            SELECT
                geo_location,
                download_count,
                total_users,
                ROW_NUMBER() OVER (PARTITION BY geo_location ORDER BY download_count DESC) as user_rank
            FROM user_downloads ud
            JOIN location_stats ls USING (geo_location)
        )
        GROUP BY geo_location
    )
    SELECT
        ls.geo_location,
        COALESCE(ec.user_entropy, 0) as user_entropy,
        ls.single_dl_users::FLOAT / NULLIF(ls.total_users, 0) as single_download_user_ratio,
        COALESCE(pc.power_user_ratio, 1.0) as power_user_ratio,
        ls.total_users,
        ls.sorted_downloads
    FROM location_stats ls
    LEFT JOIN entropy_calc ec ON ls.geo_location = ec.geo_location
    LEFT JOIN power_user_calc pc ON ls.geo_location = pc.geo_location
    """

    try:
        user_dist_df = conn.execute(user_dist_query).df()

        # Calculate Gini coefficient from sorted downloads
        def calculate_gini(sorted_downloads):
            """Calculate Gini coefficient from sorted download counts."""
            if sorted_downloads is None or len(sorted_downloads) < 2:
                return 0.5
            arr = np.array(sorted_downloads, dtype=float)
            if arr.sum() == 0:
                return 0.5
            n = len(arr)
            indices = np.arange(1, n + 1)
            return (2 * np.sum(indices * arr) / (n * np.sum(arr))) - (n + 1) / n

        user_dist_df['user_gini_coefficient'] = user_dist_df['sorted_downloads'].apply(calculate_gini)

        # Merge with main df
        df = df.merge(
            user_dist_df[['geo_location', 'user_entropy', 'single_download_user_ratio',
                          'power_user_ratio', 'user_gini_coefficient']],
            on='geo_location',
            how='left'
        )

        # Fill defaults
        df['user_entropy'] = df['user_entropy'].fillna(0)
        df['user_gini_coefficient'] = df['user_gini_coefficient'].fillna(0.5)
        df['single_download_user_ratio'] = df['single_download_user_ratio'].fillna(0.5)
        df['power_user_ratio'] = df['power_user_ratio'].fillna(0.5)

        logger.info(f"    ✓ User distribution features extracted for {len(user_dist_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ User distribution extraction failed: {e}")
        df['user_entropy'] = 0
        df['user_gini_coefficient'] = 0.5
        df['single_download_user_ratio'] = 0.5
        df['power_user_ratio'] = 0.5

    return df


def extract_session_behavior_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema=None
) -> pd.DataFrame:
    """
    Extract features that distinguish human browsing sessions from bot sessions.

    Human sessions have natural variation in duration, timing, and intensity.
    Bot sessions are often mechanically consistent.

    Features computed:
    - session_duration_cv: Coefficient of variation of session lengths
    - inter_session_regularity: Regularity of gaps between sessions
    - downloads_per_session_cv: CV of downloads per session
    - session_start_hour_entropy: Entropy of session start times

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with session behavior features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting session behavior features...")

    # Session detection with 30-minute gap threshold
    session_query = f"""
    WITH ordered_downloads AS (
        SELECT
            {schema.location_field} as geo_location,
            CAST({schema.timestamp_field} AS TIMESTAMP) as ts,
            LAG(CAST({schema.timestamp_field} AS TIMESTAMP))
                OVER (PARTITION BY {schema.location_field} ORDER BY {schema.timestamp_field}) as prev_ts
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    session_markers AS (
        SELECT
            geo_location,
            ts,
            EXTRACT(HOUR FROM ts) as start_hour,
            CASE
                WHEN prev_ts IS NULL OR EPOCH(ts - prev_ts) > 1800
                THEN 1
                ELSE 0
            END as is_session_start,
            CASE
                WHEN prev_ts IS NOT NULL AND EPOCH(ts - prev_ts) > 1800
                THEN EPOCH(ts - prev_ts)
                ELSE NULL
            END as inter_session_gap
        FROM ordered_downloads
    ),
    sessions AS (
        SELECT
            geo_location,
            ts,
            start_hour,
            inter_session_gap,
            SUM(is_session_start) OVER (
                PARTITION BY geo_location
                ORDER BY ts
            ) as session_id
        FROM session_markers
    ),
    session_stats AS (
        SELECT
            geo_location,
            session_id,
            MIN(start_hour) as session_start_hour,
            COUNT(*) as downloads_in_session,
            EPOCH(MAX(ts) - MIN(ts)) as session_duration_seconds
        FROM sessions
        GROUP BY geo_location, session_id
    ),
    location_session_stats AS (
        SELECT
            geo_location,
            -- Session duration CV
            CASE
                WHEN AVG(session_duration_seconds) > 0
                THEN STDDEV(session_duration_seconds) / AVG(session_duration_seconds)
                ELSE 0
            END as session_duration_cv,
            -- Downloads per session CV
            CASE
                WHEN AVG(downloads_in_session) > 0
                THEN STDDEV(downloads_in_session) / AVG(downloads_in_session)
                ELSE 0
            END as downloads_per_session_cv,
            -- Session count
            COUNT(*) as num_sessions
        FROM session_stats
        GROUP BY geo_location
    ),
    inter_session_stats AS (
        SELECT
            geo_location,
            -- Inter-session regularity (1 / CV of gaps)
            CASE
                WHEN STDDEV(inter_session_gap) > 0
                THEN AVG(inter_session_gap) / STDDEV(inter_session_gap)
                ELSE 0
            END as inter_session_regularity
        FROM session_markers
        WHERE inter_session_gap IS NOT NULL
        GROUP BY geo_location
    ),
    hour_entropy AS (
        SELECT
            geo_location,
            -- Entropy of session start hours
            -SUM(
                (hour_count::FLOAT / total_sessions) *
                LOG2(hour_count::FLOAT / total_sessions + 1e-10)
            ) as session_start_hour_entropy
        FROM (
            SELECT
                geo_location,
                session_start_hour,
                COUNT(*) as hour_count,
                SUM(COUNT(*)) OVER (PARTITION BY geo_location) as total_sessions
            FROM session_stats
            GROUP BY geo_location, session_start_hour
        )
        GROUP BY geo_location
    )
    SELECT
        lss.geo_location,
        COALESCE(lss.session_duration_cv, 0) as session_duration_cv,
        COALESCE(iss.inter_session_regularity, 0) as inter_session_regularity,
        COALESCE(lss.downloads_per_session_cv, 0) as downloads_per_session_cv,
        COALESCE(he.session_start_hour_entropy, 0) as session_start_hour_entropy
    FROM location_session_stats lss
    LEFT JOIN inter_session_stats iss ON lss.geo_location = iss.geo_location
    LEFT JOIN hour_entropy he ON lss.geo_location = he.geo_location
    WHERE lss.num_sessions >= 2
    """

    try:
        session_df = conn.execute(session_query).df()
        df = df.merge(session_df, on='geo_location', how='left')

        # Fill defaults
        df['session_duration_cv'] = df['session_duration_cv'].fillna(1.0)  # High CV = varied
        df['inter_session_regularity'] = df['inter_session_regularity'].fillna(0)
        df['downloads_per_session_cv'] = df['downloads_per_session_cv'].fillna(1.0)
        df['session_start_hour_entropy'] = df['session_start_hour_entropy'].fillna(2.0)  # Moderate entropy

        logger.info(f"    ✓ Session behavior features extracted for {len(session_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Session behavior extraction failed: {e}")
        df['session_duration_cv'] = 1.0
        df['inter_session_regularity'] = 0
        df['downloads_per_session_cv'] = 1.0
        df['session_start_hour_entropy'] = 2.0

    return df


# List of new timing/session features for reference
NEW_BEHAVIORAL_FEATURES = [
    # Timing precision features
    'request_interval_mode',
    'round_second_ratio',
    'millisecond_variance',
    'interval_entropy',
    # User distribution features
    'user_entropy',
    'user_gini_coefficient',
    'single_download_user_ratio',
    'power_user_ratio',
    # Session behavior features
    'session_duration_cv',
    'inter_session_regularity',
    'downloads_per_session_cv',
    'session_start_hour_entropy',
]


def add_bot_signature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add discriminative bot signature features.
    
    These features capture behavioral signatures that are highly discriminative
    for bot detection, focusing on temporal and access patterns.
    
    Features added:
    - access_regularity: Inverse of hourly entropy (low entropy = regular = bot-like)
    - ua_per_user: User-Agent diversity per user
    - request_velocity: Downloads per active hour
    - ip_concentration: 1 - IP entropy
    - session_anomaly: Deviation from median session length
    - request_pattern_anomaly: 1 / file request entropy
    - weekend_weekday_imbalance: Deviation from expected 2/7 ratio
    
    Args:
        df: DataFrame with location features
        
    Returns:
        DataFrame with additional signature features
    """
    logger.info("    Adding bot signature features...")
    
    # 1. Access regularity (temporal pattern consistency)
    if 'hourly_entropy' in df.columns:
        # High entropy = irregular, low entropy = regular (bot-like)
        df['access_regularity'] = 1.0 / (df['hourly_entropy'] + 0.1)
    else:
        df['access_regularity'] = 0.0
    
    # 2. User-Agent diversity (if available in raw data, use proxy)
    # Proxy: locations with many users but low diversity are suspicious
    if 'unique_users' in df.columns and 'total_downloads' in df.columns:
        df['ua_per_user'] = 1.0 / (df['unique_users'] / (df['total_downloads'] + 1) + 0.01)
    else:
        df['ua_per_user'] = 1.0
    
    # 3. Request velocity (downloads per active time)
    if 'total_downloads' in df.columns:
        # Estimate active hours from entropy and working hours ratio
        if 'working_hours_ratio' in df.columns:
            active_hours = df['working_hours_ratio'] * 24 * 7  # Hours per week
            df['request_velocity'] = df['total_downloads'] / (active_hours + 1)
        else:
            df['request_velocity'] = df['total_downloads'] / 168  # Assume full week
    else:
        df['request_velocity'] = 0.0
    
    # 4. IP concentration (1 - entropy)
    # Proxy: high users with low diversity suggests IP cycling
    if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
        # Estimate IP entropy from user distribution
        user_entropy = np.log1p(df['unique_users']) / np.log1p(df['unique_users'].max() + 1)
        df['ip_concentration'] = 1.0 - user_entropy
    else:
        df['ip_concentration'] = 0.0
    
    # 5. Session anomaly (deviation from median)
    if 'downloads_per_user' in df.columns:
        median_dl_per_user = df['downloads_per_user'].median()
        df['session_anomaly'] = np.abs(df['downloads_per_user'] - median_dl_per_user) / (median_dl_per_user + 1)
    else:
        df['session_anomaly'] = 0.0
    
    # 6. Request pattern anomaly (file diversity)
    # Bots often request same files repeatedly
    # Proxy: low DL/user with high users suggests coordinated same-file requests
    if 'downloads_per_user' in df.columns and 'unique_users' in df.columns:
        # Low DL/user = likely requesting same files
        file_entropy_proxy = df['downloads_per_user'] / (np.log1p(df['unique_users']) + 1)
        df['request_pattern_anomaly'] = 1.0 / (file_entropy_proxy + 0.1)
    else:
        df['request_pattern_anomaly'] = 0.0
    
    # 7. Working hours imbalance
    # Bots work 24/7, humans tend to work during business hours
    if 'working_hours_ratio' in df.columns:
        # Expected ratio: ~0.5-0.6 for humans (working hours = 9-17, ~8/24 hours)
        # Bots: closer to uniform (~0.33) or inverted (night activity)
        expected_working_hours_ratio = 0.5
        df['weekend_weekday_imbalance'] = np.abs(df['working_hours_ratio'] - expected_working_hours_ratio)
    else:
        df['weekend_weekday_imbalance'] = 0.0
    
    return df
