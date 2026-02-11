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
    if 'protocol_legitimacy_score' in df.columns:
        legitimate_components.append(df['protocol_legitimacy_score'].clip(0, 1) * 0.15)

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


def extract_access_pattern_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that detect systematic crawling/scraping patterns.

    Bots often access files in predictable patterns:
    - Alphabetical order (directory listing crawlers)
    - Sequential numbering (automated scrapers)
    - Directory tree traversal (depth-first crawlers)
    - High retry rates (failing scrapers)

    Features computed:
    - alphabetical_access_score: Correlation with alphabetical order
    - sequential_file_ratio: Fraction of sequential file accesses
    - directory_traversal_score: Following directory structure
    - retry_ratio: Files accessed multiple times
    - unique_file_ratio: Unique files / total downloads

    Args:
        df: DataFrame with basic features
        input_parquet: Path to raw data
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with access pattern features
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting access pattern features...")

    access_pattern_query = f"""
    WITH file_access AS (
        SELECT
            {schema.location_field} as geo_location,
            filename,
            CAST({schema.timestamp_field} AS TIMESTAMP) as ts,
            ROW_NUMBER() OVER (
                PARTITION BY {schema.location_field}
                ORDER BY {schema.timestamp_field}
            ) as access_order,
            -- Extract numeric portion from filename for sequential detection
            REGEXP_EXTRACT(filename, '([0-9]+)', 1) as file_number
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND filename IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    alphabetical_ranks AS (
        SELECT
            geo_location,
            filename,
            access_order,
            file_number,
            DENSE_RANK() OVER (
                PARTITION BY geo_location
                ORDER BY filename
            ) as alpha_rank
        FROM file_access
    ),
    location_file_stats AS (
        SELECT
            geo_location,
            COUNT(DISTINCT filename) as unique_files,
            COUNT(*) as total_downloads,
            -- Files with multiple accesses
            SUM(CASE WHEN file_count > 1 THEN 1 ELSE 0 END) as repeated_files
        FROM (
            SELECT
                geo_location,
                filename,
                COUNT(*) as file_count
            FROM file_access
            GROUP BY geo_location, filename
        )
        GROUP BY geo_location
    ),
    sequential_analysis AS (
        SELECT
            a.geo_location,
            -- Sequential file detection (consecutive numbered files)
            AVG(CASE
                WHEN a.file_number IS NOT NULL AND a.file_number != ''
                AND b.file_number IS NOT NULL AND b.file_number != ''
                AND TRY_CAST(a.file_number AS INTEGER) IS NOT NULL
                AND TRY_CAST(b.file_number AS INTEGER) IS NOT NULL
                AND ABS(TRY_CAST(a.file_number AS INTEGER) - TRY_CAST(b.file_number AS INTEGER)) = 1
                THEN 1.0
                ELSE 0.0
            END) as sequential_file_ratio
        FROM alphabetical_ranks a
        JOIN alphabetical_ranks b
            ON a.geo_location = b.geo_location
            AND a.access_order = b.access_order - 1
        WHERE a.file_number IS NOT NULL AND a.file_number != ''
        GROUP BY a.geo_location
    ),
    correlation_stats AS (
        SELECT
            geo_location,
            -- Correlation between access order and alphabetical order
            -- Using Pearson correlation approximation
            CASE
                WHEN STDDEV(access_order) > 0 AND STDDEV(alpha_rank) > 0
                THEN (
                    AVG(access_order * alpha_rank) - AVG(access_order) * AVG(alpha_rank)
                ) / (STDDEV(access_order) * STDDEV(alpha_rank))
                ELSE 0
            END as alphabetical_access_score
        FROM alphabetical_ranks
        GROUP BY geo_location
    )
    SELECT
        lfs.geo_location,
        COALESCE(cs.alphabetical_access_score, 0) as alphabetical_access_score,
        COALESCE(sa.sequential_file_ratio, 0) as sequential_file_ratio,
        -- Directory traversal score: simplified as alphabetical correlation * file diversity
        COALESCE(cs.alphabetical_access_score, 0) *
            (lfs.unique_files::FLOAT / NULLIF(lfs.total_downloads, 0)) as directory_traversal_score,
        -- Retry ratio
        lfs.repeated_files::FLOAT / NULLIF(lfs.unique_files, 0) as retry_ratio,
        -- Unique file ratio
        lfs.unique_files::FLOAT / NULLIF(lfs.total_downloads, 0) as unique_file_ratio
    FROM location_file_stats lfs
    LEFT JOIN correlation_stats cs ON lfs.geo_location = cs.geo_location
    LEFT JOIN sequential_analysis sa ON lfs.geo_location = sa.geo_location
    """

    try:
        access_df = conn.execute(access_pattern_query).df()
        df = df.merge(access_df, on='geo_location', how='left')

        # Fill defaults
        df['alphabetical_access_score'] = df['alphabetical_access_score'].fillna(0)
        df['sequential_file_ratio'] = df['sequential_file_ratio'].fillna(0)
        df['directory_traversal_score'] = df['directory_traversal_score'].fillna(0)
        df['retry_ratio'] = df['retry_ratio'].fillna(0)
        df['unique_file_ratio'] = df['unique_file_ratio'].fillna(1.0)

        logger.info(f"    ✓ Access pattern features extracted for {len(access_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Access pattern extraction failed: {e}")
        df['alphabetical_access_score'] = 0
        df['sequential_file_ratio'] = 0
        df['directory_traversal_score'] = 0
        df['retry_ratio'] = 0
        df['unique_file_ratio'] = 1.0

    return df


def extract_statistical_anomaly_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that detect statistically impossible patterns.

    Real download data follows certain statistical laws. Bots often
    generate traffic that violates these patterns.

    Features computed:
    - benford_deviation: Deviation from Benford's Law (first digit distribution)
    - hourly_uniformity_score: How uniform hourly distribution is
    - weekday_pattern_score: Deviation from expected weekday ratio

    Args:
        df: DataFrame with basic features
        input_parquet: Path to raw data
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with statistical anomaly features
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting statistical anomaly features...")

    # Benford's Law expected distribution for first digits 1-9
    benford_expected = np.array([0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046])

    statistical_query = f"""
    WITH daily_downloads AS (
        SELECT
            {schema.location_field} as geo_location,
            DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
            COUNT(*) as daily_count
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
    ),
    first_digits AS (
        SELECT
            geo_location,
            CAST(SUBSTR(CAST(daily_count AS VARCHAR), 1, 1) AS INTEGER) as first_digit,
            COUNT(*) as digit_count
        FROM daily_downloads
        WHERE daily_count >= 1
        GROUP BY geo_location, first_digit
    ),
    first_digit_totals AS (
        SELECT
            geo_location,
            SUM(digit_count) as total_days,
            -- Get array of digit proportions
            SUM(CASE WHEN first_digit = 1 THEN digit_count ELSE 0 END)::FLOAT as d1,
            SUM(CASE WHEN first_digit = 2 THEN digit_count ELSE 0 END)::FLOAT as d2,
            SUM(CASE WHEN first_digit = 3 THEN digit_count ELSE 0 END)::FLOAT as d3,
            SUM(CASE WHEN first_digit = 4 THEN digit_count ELSE 0 END)::FLOAT as d4,
            SUM(CASE WHEN first_digit = 5 THEN digit_count ELSE 0 END)::FLOAT as d5,
            SUM(CASE WHEN first_digit = 6 THEN digit_count ELSE 0 END)::FLOAT as d6,
            SUM(CASE WHEN first_digit = 7 THEN digit_count ELSE 0 END)::FLOAT as d7,
            SUM(CASE WHEN first_digit = 8 THEN digit_count ELSE 0 END)::FLOAT as d8,
            SUM(CASE WHEN first_digit = 9 THEN digit_count ELSE 0 END)::FLOAT as d9
        FROM first_digits
        GROUP BY geo_location
    ),
    hourly_uniformity AS (
        SELECT
            {schema.location_field} as geo_location,
            -- Chi-squared statistic for uniformity
            -- Lower variance from uniform = higher uniformity score
            1.0 - (VARIANCE(hour_count) / (AVG(hour_count) * AVG(hour_count) + 0.01)) as hourly_uniformity_raw
        FROM (
            SELECT
                {schema.location_field},
                EXTRACT(HOUR FROM CAST({schema.timestamp_field} AS TIMESTAMP)) as hour,
                COUNT(*) as hour_count
            FROM read_parquet('{escaped_path}')
            WHERE {schema.location_field} IS NOT NULL
            AND {schema.timestamp_field} IS NOT NULL
            GROUP BY 1, 2
        )
        GROUP BY {schema.location_field}
    ),
    weekday_pattern AS (
        SELECT
            {schema.location_field} as geo_location,
            -- Weekday ratio (expecting ~5/7 = 0.714 for uniform distribution)
            AVG(CASE
                WHEN DAYOFWEEK(CAST({schema.timestamp_field} AS TIMESTAMP)) NOT IN (0, 6)
                THEN 1.0
                ELSE 0.0
            END) as weekday_ratio
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY {schema.location_field}
    )
    SELECT
        fdt.geo_location,
        fdt.total_days,
        -- Raw digit proportions for Python processing
        fdt.d1 / NULLIF(fdt.total_days, 0) as p1,
        fdt.d2 / NULLIF(fdt.total_days, 0) as p2,
        fdt.d3 / NULLIF(fdt.total_days, 0) as p3,
        fdt.d4 / NULLIF(fdt.total_days, 0) as p4,
        fdt.d5 / NULLIF(fdt.total_days, 0) as p5,
        fdt.d6 / NULLIF(fdt.total_days, 0) as p6,
        fdt.d7 / NULLIF(fdt.total_days, 0) as p7,
        fdt.d8 / NULLIF(fdt.total_days, 0) as p8,
        fdt.d9 / NULLIF(fdt.total_days, 0) as p9,
        COALESCE(hu.hourly_uniformity_raw, 0) as hourly_uniformity_raw,
        -- Weekday pattern score: deviation from expected 5/7
        ABS(COALESCE(wp.weekday_ratio, 0.714) - 0.714) / 0.714 as weekday_pattern_score
    FROM first_digit_totals fdt
    LEFT JOIN hourly_uniformity hu ON fdt.geo_location = hu.geo_location
    LEFT JOIN weekday_pattern wp ON fdt.geo_location = wp.geo_location
    """

    try:
        stats_df = conn.execute(statistical_query).df()

        # Calculate Benford deviation (chi-squared style)
        def calculate_benford_deviation(row):
            if row['total_days'] < 10:
                return 0  # Not enough data
            observed = np.array([
                row['p1'] or 0, row['p2'] or 0, row['p3'] or 0,
                row['p4'] or 0, row['p5'] or 0, row['p6'] or 0,
                row['p7'] or 0, row['p8'] or 0, row['p9'] or 0
            ])
            # Chi-squared-like deviation
            deviation = np.sum((observed - benford_expected) ** 2 / (benford_expected + 0.01))
            return deviation

        stats_df['benford_deviation'] = stats_df.apply(calculate_benford_deviation, axis=1)

        # Normalize hourly uniformity to 0-1
        stats_df['hourly_uniformity_score'] = stats_df['hourly_uniformity_raw'].clip(0, 1)

        # Merge with main df
        df = df.merge(
            stats_df[['geo_location', 'benford_deviation', 'hourly_uniformity_score',
                      'weekday_pattern_score']],
            on='geo_location',
            how='left'
        )

        # Fill defaults
        df['benford_deviation'] = df['benford_deviation'].fillna(0)
        df['hourly_uniformity_score'] = df['hourly_uniformity_score'].fillna(0.5)
        df['weekday_pattern_score'] = df['weekday_pattern_score'].fillna(0)

        logger.info(f"    ✓ Statistical anomaly features extracted for {len(stats_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Statistical anomaly extraction failed: {e}")
        df['benford_deviation'] = 0
        df['hourly_uniformity_score'] = 0.5
        df['weekday_pattern_score'] = 0

    return df


def extract_comparative_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that compare locations to their context/peers.

    Anomalous locations stand out from:
    - Their country's average behavior
    - Similar-sized locations
    - Historical patterns

    Features computed:
    - country_zscore: Z-score vs country average
    - temporal_trend_anomaly: Deviation from historical trend
    - peer_similarity: Similarity to peer locations
    - global_rank_percentile: Percentile by download volume

    Args:
        df: DataFrame with basic features
        input_parquet: Path to raw data
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with comparative features
    """
    if schema is None:
        schema = EBI_SCHEMA

    logger.info("  Extracting comparative features...")

    try:
        # Country-level Z-scores
        if 'country' in df.columns and 'downloads_per_user' in df.columns:
            country_stats = df.groupby('country')['downloads_per_user'].agg(['mean', 'std']).reset_index()
            country_stats.columns = ['country', 'country_dpu_mean', 'country_dpu_std']
            df = df.merge(country_stats, on='country', how='left')

            df['country_zscore'] = np.where(
                df['country_dpu_std'] > 0,
                (df['downloads_per_user'] - df['country_dpu_mean']) / df['country_dpu_std'],
                0
            )
            df = df.drop(columns=['country_dpu_mean', 'country_dpu_std'])
        else:
            df['country_zscore'] = 0

        # Temporal trend anomaly (based on latest year vs historical)
        if 'spike_ratio' in df.columns and 'years_before_latest' in df.columns:
            # Locations with high spike ratio but previous history are anomalous
            df['temporal_trend_anomaly'] = np.where(
                df['years_before_latest'] > 0,
                np.log1p(df['spike_ratio']) / np.log1p(df['years_before_latest'] + 1),
                0
            )
        else:
            df['temporal_trend_anomaly'] = 0

        # Peer similarity (compare to locations with similar user counts)
        if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
            # Bin by user count and calculate median behavior
            df['user_bin'] = pd.qcut(df['unique_users'], q=10, labels=False, duplicates='drop')
            peer_medians = df.groupby('user_bin').agg({
                'downloads_per_user': 'median',
                'hourly_entropy': 'median' if 'hourly_entropy' in df.columns else lambda x: 0
            }).reset_index()
            peer_medians.columns = ['user_bin', 'peer_dpu_median', 'peer_entropy_median']
            df = df.merge(peer_medians, on='user_bin', how='left')

            # Calculate similarity as inverse of normalized deviation
            dpu_dev = np.abs(df['downloads_per_user'] - df['peer_dpu_median']) / (df['peer_dpu_median'] + 1)
            df['peer_similarity'] = 1 / (1 + dpu_dev)
            df = df.drop(columns=['user_bin', 'peer_dpu_median', 'peer_entropy_median'], errors='ignore')
        else:
            df['peer_similarity'] = 0.5

        # Global rank percentile
        if 'total_downloads' in df.columns:
            df['global_rank_percentile'] = df['total_downloads'].rank(pct=True)
        else:
            df['global_rank_percentile'] = 0.5

        # Fill any NaN values
        df['country_zscore'] = df['country_zscore'].fillna(0)
        df['temporal_trend_anomaly'] = df['temporal_trend_anomaly'].fillna(0)
        df['peer_similarity'] = df['peer_similarity'].fillna(0.5)
        df['global_rank_percentile'] = df['global_rank_percentile'].fillna(0.5)

        logger.info(f"    ✓ Comparative features extracted for {len(df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Comparative feature extraction failed: {e}")
        df['country_zscore'] = 0
        df['temporal_trend_anomaly'] = 0
        df['peer_similarity'] = 0.5
        df['global_rank_percentile'] = 0.5

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

# New discriminative features
NEW_DISCRIMINATIVE_FEATURES = [
    # Access pattern features
    'alphabetical_access_score',
    'sequential_file_ratio',
    'directory_traversal_score',
    'retry_ratio',
    'unique_file_ratio',
    # Statistical anomaly features
    'benford_deviation',
    'hourly_uniformity_score',
    'weekday_pattern_score',
    # Comparative features
    'country_zscore',
    'temporal_trend_anomaly',
    'peer_similarity',
    'global_rank_percentile',
]
