"""
Discriminative behavioral features to separate malicious bots from legitimate automation.

The problem: Current behavioral features detect automation but can't distinguish:
- Malicious bots (bot farms, scrapers, coordinated attacks)
- Legitimate automation (CI/CD, mirrors, research pipelines)

Solution: Add features that capture the INTENT and PATTERN of automation.
"""

import pandas as pd
import numpy as np
from typing import Optional
from ....utils import logger
from ...schema import LogSchema
from .schema import EBI_SCHEMA


def normalize_feature(series: pd.Series) -> np.ndarray:
    """Robust normalization using quantiles."""
    if len(series) == 0:
        return np.array([])
    q01, q99 = series.quantile([0.01, 0.99])
    if q99 - q01 < 1e-10:
        return np.zeros(len(series))
    normalized = (series - q01) / (q99 - q01)
    return np.clip(normalized, 0, 1).values


def extract_discriminative_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that discriminate malicious bots from legitimate automation.
    
    Key insight: Malicious bots and legitimate automation both show:
    - Burst patterns
    - Coordination
    - Non-human timing
    
    But they differ in:
    - File access patterns (bots explore, mirrors copy same files)
    - User diversity (bot farms use many fake IDs, mirrors use few real users)
    - Geographic consistency (mirrors are stable, bot farms rotate IPs)
    - Version targeting (mirrors get all versions, bots target latest)
    - Error rates (bots generate more 404s, timeouts)
    
    Args:
        df: DataFrame with basic features
        input_parquet: Path to raw data
        conn: DuckDB connection
        schema: LogSchema for field mappings
        
    Returns:
        DataFrame with discriminative features
    """
    if schema is None:
        schema = EBI_SCHEMA
    
    escaped_path = input_parquet.replace("'", "''")
    
    logger.info("Extracting discriminative features (malicious vs legitimate automation)...")
    
    # =========================================================================
    # Feature 1: File Access Diversity Score
    # =========================================================================
    # Malicious: High diversity (explore many files to find targets)
    # Legitimate: Low diversity (mirror specific files repeatedly)
    logger.info("  Extracting file access diversity...")
    
    file_diversity_query = f"""
    WITH file_stats AS (
        SELECT 
            {schema.location_field} as geo_location,
            filename,
            COUNT(*) as download_count,
            COUNT(DISTINCT {schema.user_field}) as users_per_file
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND filename IS NOT NULL
        GROUP BY 1, 2
    ),
    location_stats AS (
        SELECT 
            geo_location,
            COUNT(DISTINCT filename) as unique_files,
            COUNT(*) as total_file_downloads,
            AVG(users_per_file) as avg_users_per_file,
            -- Simplified entropy calculation (log of unique files / total)
            CASE 
                WHEN COUNT(*) > 0 AND COUNT(DISTINCT filename) > 0
                THEN LOG2(CAST(COUNT(*) AS DOUBLE) / COUNT(DISTINCT filename))
                ELSE 0
            END as file_entropy
        FROM file_stats
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        unique_files,
        file_entropy,
        avg_users_per_file,
        -- File exploration score: high entropy + high unique files = exploration (bot)
        CASE 
            WHEN unique_files > 10 
            THEN file_entropy * LN(unique_files + 1) 
            ELSE 0 
        END as file_exploration_score,
        -- Mirroring score: low entropy + low unique files = mirroring (legitimate)
        CASE 
            WHEN file_entropy > 0 
            THEN 1.0 / (file_entropy + 0.1) 
            ELSE 10 
        END as file_mirroring_score
    FROM location_stats
    """
    
    try:
        file_div_df = conn.execute(file_diversity_query).df()
        df = df.merge(file_div_df, on='geo_location', how='left')
        df['file_exploration_score'] = df['file_exploration_score'].fillna(0)
        df['file_mirroring_score'] = df['file_mirroring_score'].fillna(1)
        df['file_entropy'] = df['file_entropy'].fillna(0)
        logger.info(f"    ✓ File diversity extracted for {len(file_div_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ File diversity extraction failed: {e}")
        df['file_exploration_score'] = 0
        df['file_mirroring_score'] = 1
        df['file_entropy'] = 0
    
    # =========================================================================
    # Feature 2: User ID Diversity & Authenticity Score
    # =========================================================================
    # Malicious: Many users, low activity per user (bot farm with fake IDs)
    # Legitimate: Few users, high activity per user (real accounts)
    logger.info("  Extracting user authenticity patterns...")
    
    user_authenticity_query = f"""
    WITH user_activity AS (
        SELECT 
            {schema.location_field} as geo_location,
            {schema.user_field} as user_id,
            COUNT(*) as downloads_per_user,
            COUNT(DISTINCT filename) as files_per_user,
            COUNT(DISTINCT DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP))) as active_days
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.user_field} IS NOT NULL
        AND filename IS NOT NULL
        GROUP BY 1, 2
    ),
    user_patterns AS (
        SELECT 
            geo_location,
            COUNT(DISTINCT user_id) as unique_users_count,
            AVG(downloads_per_user) as avg_downloads_per_user,
            STDDEV(downloads_per_user) as std_downloads_per_user,
            -- Bot farm pattern: many users with similar low activity
            CASE 
                WHEN STDDEV(downloads_per_user) > 0 
                THEN AVG(downloads_per_user) / STDDEV(downloads_per_user) 
                ELSE 0 
            END as user_homogeneity_score,
            -- Authentic user pattern: users have varied activity over time
            AVG(active_days) as avg_active_days_per_user,
            AVG(files_per_user) as avg_files_per_user
        FROM user_activity
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        user_homogeneity_score,
        avg_active_days_per_user,
        avg_files_per_user,
        -- Bot farm score: high homogeneity + many users + low activity
        CASE 
            WHEN unique_users_count > 100 AND user_homogeneity_score > 2 
            THEN user_homogeneity_score * LN(unique_users_count + 1) 
            ELSE 0 
        END as bot_farm_score,
        -- Authentic score: low homogeneity + sustained activity
        COALESCE(avg_active_days_per_user * avg_files_per_user, 0) as user_authenticity_score
    FROM user_patterns
    """
    
    try:
        user_auth_df = conn.execute(user_authenticity_query).df()
        df = df.merge(user_auth_df, on='geo_location', how='left')
        df['bot_farm_score'] = df['bot_farm_score'].fillna(0)
        df['user_authenticity_score'] = df['user_authenticity_score'].fillna(1)
        df['user_homogeneity_score'] = df['user_homogeneity_score'].fillna(0)
        logger.info(f"    ✓ User authenticity extracted for {len(user_auth_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ User authenticity extraction failed: {e}")
        df['bot_farm_score'] = 0
        df['user_authenticity_score'] = 1
        df['user_homogeneity_score'] = 0
    
    # =========================================================================
    # Feature 3: Geographic Stability
    # =========================================================================
    # Malicious: IP rotation, geographic inconsistency
    # Legitimate: Stable location (institutional servers)
    logger.info("  Extracting geographic stability...")
    
    if 'ip_concentration' in df.columns:
        # High IP concentration = many IPs = unstable (bot)
        # Geographic stability = inverse of IP concentration
        df['geographic_stability'] = 1.0 / (df['ip_concentration'] + 0.1)
        df['geographic_stability'] = df['geographic_stability'].clip(0, 1)
    else:
        # Fallback: assume stable if we can't compute
        df['geographic_stability'] = 0.5
    
    # =========================================================================
    # Feature 4: Version Targeting Pattern
    # =========================================================================
    # Malicious: Target only latest versions (looking for vulnerabilities)
    # Legitimate: Download all versions (archival, mirroring)
    logger.info("  Extracting version targeting patterns...")
    
    version_pattern_query = f"""
    WITH version_downloads AS (
        SELECT 
            {schema.location_field} as geo_location,
            filename,
            COUNT(*) as download_count,
            -- Extract version if possible (simplified - assumes filename contains version pattern)
            REGEXP_EXTRACT(filename, '[0-9]+\\.[0-9]+\\.[0-9]+', 0) as version
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND filename IS NOT NULL
        GROUP BY 1, 2, 4
    ),
    version_patterns AS (
        SELECT 
            geo_location,
            COUNT(DISTINCT version) as unique_versions,
            COUNT(DISTINCT filename) as unique_files,
            -- Latest version bias: do they only target newest?
            CASE 
                WHEN COUNT(DISTINCT version) > 0 
                THEN 1.0 / COUNT(DISTINCT version) 
                ELSE 1 
            END as version_concentration
        FROM version_downloads
        WHERE version IS NOT NULL AND version != ''
        GROUP BY geo_location
    )
    SELECT 
        geo_location,
        COALESCE(unique_versions, 0) as unique_versions,
        COALESCE(version_concentration, 0.5) as version_concentration,
        -- Latest-only targeting: high concentration = suspicious
        CASE 
            WHEN version_concentration > 0.5 
            THEN 1 
            ELSE 0 
        END as targets_latest_only
    FROM version_patterns
    """
    
    try:
        version_df = conn.execute(version_pattern_query).df()
        df = df.merge(version_df, on='geo_location', how='left')
        df['version_concentration'] = df['version_concentration'].fillna(0.5)
        df['targets_latest_only'] = df['targets_latest_only'].fillna(0)
        df['unique_versions'] = df['unique_versions'].fillna(0)
        logger.info(f"    ✓ Version patterns extracted for {len(version_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Version pattern extraction failed: {e}")
        df['version_concentration'] = 0.5
        df['targets_latest_only'] = 0
        df['unique_versions'] = 0
    
    # =========================================================================
    # Feature 5: Persistence Pattern
    # =========================================================================
    # Malicious: Short-lived activity (hit and run)
    # Legitimate: Long-lived, consistent activity (institutional)
    logger.info("  Extracting persistence patterns...")
    
    persistence_query = f"""
    WITH temporal_span AS (
        SELECT 
            {schema.location_field} as geo_location,
            MIN(CAST({schema.timestamp_field} AS TIMESTAMP)) as first_seen,
            MAX(CAST({schema.timestamp_field} AS TIMESTAMP)) as last_seen,
            COUNT(DISTINCT DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP))) as active_days,
            COUNT(DISTINCT DATE_TRUNC('week', CAST({schema.timestamp_field} AS TIMESTAMP))) as active_weeks
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY {schema.location_field}
    )
    SELECT 
        geo_location,
        EPOCH(last_seen - first_seen) / 86400.0 as lifespan_days,
        active_days,
        active_weeks,
        -- Activity density: active_days / lifespan (low = sporadic = suspicious)
        CASE 
            WHEN EPOCH(last_seen - first_seen) > 0 
            THEN active_days / (EPOCH(last_seen - first_seen) / 86400.0) 
            ELSE 1 
        END as activity_density,
        -- Persistence score: long lifespan + high density = legitimate
        COALESCE(active_weeks * active_days, 0) as persistence_score
    FROM temporal_span
    """
    
    try:
        persistence_df = conn.execute(persistence_query).df()
        df = df.merge(persistence_df, on='geo_location', how='left')
        df['lifespan_days'] = df['lifespan_days'].fillna(1)
        df['activity_density'] = df['activity_density'].fillna(0.5)
        df['persistence_score'] = df['persistence_score'].fillna(0)
        logger.info(f"    ✓ Persistence patterns extracted for {len(persistence_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Persistence extraction failed: {e}")
        df['lifespan_days'] = 1
        df['activity_density'] = df.get('active_days', pd.Series(1)).fillna(1)
        df['persistence_score'] = 0
    
    # =========================================================================
    # Composite Discriminative Scores
    # =========================================================================
    logger.info("  Computing composite discriminative scores...")
    
    # Malicious Bot Score (high = likely malicious)
    malicious_components = []
    if 'file_exploration_score' in df.columns:
        malicious_components.append(normalize_feature(df['file_exploration_score']) * 0.25)
    if 'bot_farm_score' in df.columns:
        malicious_components.append(normalize_feature(df['bot_farm_score']) * 0.30)
    if 'geographic_stability' in df.columns:
        malicious_components.append((1 - df['geographic_stability'].clip(0, 1)) * 0.20)
    if 'targets_latest_only' in df.columns:
        malicious_components.append(df['targets_latest_only'] * 0.15)
    if 'activity_density' in df.columns:
        malicious_components.append((1 - df['activity_density'].clip(0, 1)) * 0.10)
    
    if malicious_components:
        df['malicious_bot_score'] = sum(malicious_components)
        df['malicious_bot_score'] = df['malicious_bot_score'].clip(0, 1)
    else:
        df['malicious_bot_score'] = 0
    
    # Legitimate Automation Score (high = likely legitimate)
    legitimate_components = []
    if 'file_mirroring_score' in df.columns:
        legitimate_components.append(normalize_feature(df['file_mirroring_score']) * 0.30)
    if 'user_authenticity_score' in df.columns:
        legitimate_components.append(normalize_feature(df['user_authenticity_score']) * 0.25)
    if 'geographic_stability' in df.columns:
        legitimate_components.append(df['geographic_stability'].clip(0, 1) * 0.20)
    if 'persistence_score' in df.columns:
        legitimate_components.append(normalize_feature(df['persistence_score']) * 0.25)
    
    if legitimate_components:
        df['legitimate_automation_score'] = sum(legitimate_components)
        df['legitimate_automation_score'] = df['legitimate_automation_score'].clip(0, 1)
    else:
        df['legitimate_automation_score'] = 0
    
    # Final discriminator: malicious - legitimate
    df['bot_vs_legitimate_score'] = df['malicious_bot_score'] - df['legitimate_automation_score']
    
    # Categorize
    df['is_likely_malicious'] = (df['bot_vs_legitimate_score'] > 0.3).astype(float)
    df['is_likely_legitimate_automation'] = (df['bot_vs_legitimate_score'] < -0.3).astype(float)
    
    logger.info("✓ Discriminative features extraction complete")
    logger.info(f"  Likely malicious: {df['is_likely_malicious'].sum():,} ({df['is_likely_malicious'].mean()*100:.1f}%)")
    logger.info(f"  Likely legitimate automation: {df['is_likely_legitimate_automation'].sum():,} ({df['is_likely_legitimate_automation'].mean()*100:.1f}%)")
    logger.info(f"  Ambiguous: {((df['bot_vs_legitimate_score'].abs() <= 0.3).sum()):,}")
    
    return df


# List of discriminative features for reference
DISCRIMINATIVE_FEATURES = [
    'file_exploration_score',
    'file_mirroring_score',
    'file_entropy',
    'bot_farm_score',
    'user_authenticity_score',
    'user_homogeneity_score',
    'geographic_stability',
    'version_concentration',
    'targets_latest_only',
    'unique_versions',
    'lifespan_days',
    'activity_density',
    'persistence_score',
    'malicious_bot_score',
    'legitimate_automation_score',
    'bot_vs_legitimate_score',
    'is_likely_malicious',
    'is_likely_legitimate_automation',
]
