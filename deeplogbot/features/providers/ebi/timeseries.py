"""
Time Series Features for Bot Detection.

This module extracts advanced time series features that capture temporal dynamics:
- Outburst/Spike Detection: Sudden changes in activity
- Periodicity Detection: Regular cycling patterns (weekly, monthly)
- Trend Analysis: Long-term direction and acceleration
- Recency Weighting: Recent behavior emphasis
- Distribution Shape: Higher-order statistics

These features complement the existing TimeWindowExtractor by providing
aggregate-level temporal metrics useful for both ML and deep learning models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy import stats
from scipy.fft import rfft, rfftfreq
from ....utils import logger
from ...schema import LogSchema
from .schema import EBI_SCHEMA


# =============================================================================
# OUTBURST/SPIKE DETECTION FEATURES
# =============================================================================

def extract_outburst_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that detect sudden spikes/outbursts in activity.

    Bots often show characteristic spike patterns:
    - Sudden onset of activity
    - Short intense bursts followed by silence
    - Regular spike patterns

    Features computed:
    - outburst_count: Number of significant spikes (>2 std from mean)
    - outburst_intensity: Average magnitude of outbursts
    - max_outburst_zscore: Highest Z-score across time windows
    - outburst_ratio: Fraction of activity in outburst periods
    - longest_outburst_streak: Maximum consecutive high-activity periods
    - time_since_last_outburst: Recency of latest spike (normalized)

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with outburst features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting outburst/spike detection features...")

    # Get daily download counts per location
    daily_query = f"""
    WITH daily_counts AS (
        SELECT
            {schema.location_field} as geo_location,
            DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
            COUNT(*) as daily_downloads
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    ),
    location_stats AS (
        SELECT
            geo_location,
            AVG(daily_downloads) as mean_daily,
            STDDEV(daily_downloads) as std_daily,
            MAX(daily_downloads) as max_daily,
            COUNT(*) as num_days
        FROM daily_counts
        GROUP BY geo_location
    ),
    zscore_analysis AS (
        SELECT
            dc.geo_location,
            dc.day,
            dc.daily_downloads,
            (dc.daily_downloads - ls.mean_daily) / NULLIF(ls.std_daily, 0) as zscore,
            ROW_NUMBER() OVER (PARTITION BY dc.geo_location ORDER BY dc.day DESC) as day_rank
        FROM daily_counts dc
        JOIN location_stats ls ON dc.geo_location = ls.geo_location
        WHERE ls.std_daily > 0
    ),
    outburst_metrics AS (
        SELECT
            geo_location,
            -- Count of outbursts (|z| > 2)
            SUM(CASE WHEN ABS(zscore) > 2 THEN 1 ELSE 0 END) as outburst_count,
            -- Average intensity of outbursts
            AVG(CASE WHEN zscore > 2 THEN zscore ELSE NULL END) as outburst_intensity,
            -- Max Z-score
            MAX(zscore) as max_outburst_zscore,
            -- Fraction of downloads during outbursts
            SUM(CASE WHEN zscore > 2 THEN daily_downloads ELSE 0 END)::FLOAT /
                NULLIF(SUM(daily_downloads), 0) as outburst_ratio,
            -- Time since last outburst (normalized by total days)
            MIN(CASE WHEN zscore > 2 THEN day_rank ELSE NULL END)::FLOAT /
                NULLIF(MAX(day_rank), 0) as time_since_last_outburst,
            COUNT(*) as total_days
        FROM zscore_analysis
        GROUP BY geo_location
    )
    SELECT
        geo_location,
        COALESCE(outburst_count, 0) as outburst_count,
        COALESCE(outburst_intensity, 0) as outburst_intensity,
        COALESCE(max_outburst_zscore, 0) as max_outburst_zscore,
        COALESCE(outburst_ratio, 0) as outburst_ratio,
        COALESCE(time_since_last_outburst, 1) as time_since_last_outburst,
        total_days
    FROM outburst_metrics
    """

    try:
        outburst_df = conn.execute(daily_query).df()

        # Calculate longest outburst streak using a more complex query
        streak_query = f"""
        WITH daily_counts AS (
            SELECT
                {schema.location_field} as geo_location,
                DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
                COUNT(*) as daily_downloads
            FROM read_parquet('{escaped_path}')
            WHERE {schema.location_field} IS NOT NULL
            AND {schema.timestamp_field} IS NOT NULL
            GROUP BY 1, 2
        ),
        location_stats AS (
            SELECT
                geo_location,
                AVG(daily_downloads) as mean_daily,
                STDDEV(daily_downloads) as std_daily
            FROM daily_counts
            GROUP BY geo_location
        ),
        flagged_days AS (
            SELECT
                dc.geo_location,
                dc.day,
                CASE
                    WHEN ls.std_daily > 0 AND
                         (dc.daily_downloads - ls.mean_daily) / ls.std_daily > 2
                    THEN 1 ELSE 0
                END as is_outburst
            FROM daily_counts dc
            JOIN location_stats ls ON dc.geo_location = ls.geo_location
        ),
        streak_groups AS (
            SELECT
                geo_location,
                day,
                is_outburst,
                SUM(CASE WHEN is_outburst = 0 THEN 1 ELSE 0 END)
                    OVER (PARTITION BY geo_location ORDER BY day) as streak_group
            FROM flagged_days
        )
        SELECT
            geo_location,
            MAX(streak_length) as longest_outburst_streak
        FROM (
            SELECT
                geo_location,
                streak_group,
                SUM(is_outburst) as streak_length
            FROM streak_groups
            WHERE is_outburst = 1
            GROUP BY geo_location, streak_group
        )
        GROUP BY geo_location
        """

        streak_df = conn.execute(streak_query).df()

        # Merge results
        outburst_df = outburst_df.merge(streak_df, on='geo_location', how='left')
        outburst_df['longest_outburst_streak'] = outburst_df['longest_outburst_streak'].fillna(0)

        df = df.merge(
            outburst_df[['geo_location', 'outburst_count', 'outburst_intensity',
                         'max_outburst_zscore', 'outburst_ratio',
                         'time_since_last_outburst', 'longest_outburst_streak']],
            on='geo_location',
            how='left'
        )

        # Fill defaults
        df['outburst_count'] = df['outburst_count'].fillna(0)
        df['outburst_intensity'] = df['outburst_intensity'].fillna(0)
        df['max_outburst_zscore'] = df['max_outburst_zscore'].fillna(0)
        df['outburst_ratio'] = df['outburst_ratio'].fillna(0)
        df['time_since_last_outburst'] = df['time_since_last_outburst'].fillna(1)
        df['longest_outburst_streak'] = df['longest_outburst_streak'].fillna(0)

        logger.info(f"    ✓ Outburst features extracted for {len(outburst_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Outburst feature extraction failed: {e}")
        df['outburst_count'] = 0
        df['outburst_intensity'] = 0
        df['max_outburst_zscore'] = 0
        df['outburst_ratio'] = 0
        df['time_since_last_outburst'] = 1
        df['longest_outburst_streak'] = 0

    return df


# =============================================================================
# PERIODICITY DETECTION FEATURES
# =============================================================================

def extract_periodicity_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that detect periodic/cyclical patterns.

    Bots often operate on strict schedules (daily, weekly sync jobs).
    Legitimate automation may also show periodicity but with different patterns.

    Features computed:
    - weekly_autocorr: Autocorrelation at 7-day lag
    - dominant_period_days: Most significant period detected (FFT)
    - periodicity_strength: Strength of the dominant period
    - period_regularity: How consistent the period is

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with periodicity features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting periodicity detection features...")

    # Get daily download counts
    daily_query = f"""
    SELECT
        {schema.location_field} as geo_location,
        DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
        COUNT(*) as daily_downloads
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    try:
        daily_df = conn.execute(daily_query).df()

        def compute_periodicity(group):
            """Compute periodicity features for a location."""
            downloads = group['daily_downloads'].values
            n = len(downloads)

            if n < 14:  # Need at least 2 weeks of data
                return pd.Series({
                    'weekly_autocorr': 0,
                    'dominant_period_days': 0,
                    'periodicity_strength': 0,
                    'period_regularity': 0
                })

            # Normalize
            downloads_norm = downloads - np.mean(downloads)
            std = np.std(downloads_norm)
            if std == 0:
                return pd.Series({
                    'weekly_autocorr': 0,
                    'dominant_period_days': 0,
                    'periodicity_strength': 0,
                    'period_regularity': 0
                })

            downloads_norm = downloads_norm / std

            # 1. Weekly autocorrelation (lag=7)
            if n > 7:
                autocorr_7 = np.correlate(downloads_norm[7:], downloads_norm[:-7])[0] / (n - 7)
            else:
                autocorr_7 = 0

            # 2. FFT for dominant period
            try:
                fft_result = rfft(downloads_norm)
                power = np.abs(fft_result) ** 2
                freqs = rfftfreq(n, d=1)  # d=1 day

                # Exclude DC component and very low frequencies
                valid_idx = freqs > 1/n
                if np.sum(valid_idx) > 0:
                    power_valid = power[valid_idx]
                    freqs_valid = freqs[valid_idx]

                    # Find dominant frequency
                    max_idx = np.argmax(power_valid)
                    dominant_freq = freqs_valid[max_idx]
                    dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0

                    # Periodicity strength: ratio of dominant power to total
                    periodicity_strength = power_valid[max_idx] / (np.sum(power_valid) + 1e-10)

                    # Period regularity: how sharp the peak is (kurtosis-like)
                    if len(power_valid) > 3:
                        sorted_power = np.sort(power_valid)[::-1]
                        period_regularity = sorted_power[0] / (np.mean(sorted_power[1:4]) + 1e-10)
                    else:
                        period_regularity = 1.0
                else:
                    dominant_period = 0
                    periodicity_strength = 0
                    period_regularity = 0
            except Exception:
                dominant_period = 0
                periodicity_strength = 0
                period_regularity = 0

            return pd.Series({
                'weekly_autocorr': float(autocorr_7),
                'dominant_period_days': float(dominant_period),
                'periodicity_strength': float(min(periodicity_strength, 1.0)),
                'period_regularity': float(min(period_regularity, 10.0))
            })

        # Apply to each location
        periodicity_df = daily_df.groupby('geo_location').apply(compute_periodicity).reset_index()

        df = df.merge(periodicity_df, on='geo_location', how='left')

        # Fill defaults
        df['weekly_autocorr'] = df['weekly_autocorr'].fillna(0)
        df['dominant_period_days'] = df['dominant_period_days'].fillna(0)
        df['periodicity_strength'] = df['periodicity_strength'].fillna(0)
        df['period_regularity'] = df['period_regularity'].fillna(0)

        logger.info(f"    ✓ Periodicity features extracted for {len(periodicity_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Periodicity feature extraction failed: {e}")
        df['weekly_autocorr'] = 0
        df['dominant_period_days'] = 0
        df['periodicity_strength'] = 0
        df['period_regularity'] = 0

    return df


# =============================================================================
# TREND ANALYSIS FEATURES
# =============================================================================

def extract_trend_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that capture long-term trends and acceleration.

    Different patterns for different behaviors:
    - Bots: Often flat or sudden jumps
    - Growing services: Steady upward trend
    - Attacks: Spike then decline

    Features computed:
    - trend_slope: Linear trend direction (normalized)
    - trend_strength: R² of linear fit (how linear is the trend)
    - trend_acceleration: Second derivative (speeding up or slowing down)
    - detrended_volatility: Volatility after removing trend
    - trend_direction: Categorical (-1, 0, +1)

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with trend features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting trend analysis features...")

    # Get weekly download counts for smoother trends
    weekly_query = f"""
    SELECT
        {schema.location_field} as geo_location,
        DATE_TRUNC('week', CAST({schema.timestamp_field} AS TIMESTAMP)) as week,
        COUNT(*) as weekly_downloads
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    try:
        weekly_df = conn.execute(weekly_query).df()

        def compute_trends(group):
            """Compute trend features for a location."""
            downloads = group['weekly_downloads'].values
            n = len(downloads)

            if n < 4:  # Need at least 4 weeks
                return pd.Series({
                    'trend_slope': 0,
                    'trend_strength': 0,
                    'trend_acceleration': 0,
                    'detrended_volatility': 0,
                    'trend_direction': 0
                })

            x = np.arange(n)
            mean_dl = np.mean(downloads)

            # 1. Linear regression for trend
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, downloads)
                trend_strength = r_value ** 2

                # Normalize slope by mean
                trend_slope = slope / (mean_dl + 1e-10)

                # Trend direction
                if abs(trend_slope) < 0.01:
                    trend_direction = 0
                elif trend_slope > 0:
                    trend_direction = 1
                else:
                    trend_direction = -1

            except Exception:
                trend_slope = 0
                trend_strength = 0
                trend_direction = 0

            # 2. Acceleration (second derivative)
            if n >= 5:
                try:
                    # Fit quadratic
                    coeffs = np.polyfit(x, downloads, 2)
                    acceleration = coeffs[0] * 2  # Second derivative of ax² + bx + c
                    trend_acceleration = acceleration / (mean_dl + 1e-10)
                except Exception:
                    trend_acceleration = 0
            else:
                trend_acceleration = 0

            # 3. Detrended volatility
            try:
                predicted = slope * x + intercept
                residuals = downloads - predicted
                detrended_volatility = np.std(residuals) / (mean_dl + 1e-10)
            except Exception:
                detrended_volatility = 0

            return pd.Series({
                'trend_slope': float(np.clip(trend_slope, -10, 10)),
                'trend_strength': float(trend_strength),
                'trend_acceleration': float(np.clip(trend_acceleration, -10, 10)),
                'detrended_volatility': float(min(detrended_volatility, 10)),
                'trend_direction': int(trend_direction)
            })

        # Apply to each location
        trend_df = weekly_df.groupby('geo_location').apply(compute_trends).reset_index()

        df = df.merge(trend_df, on='geo_location', how='left')

        # Fill defaults
        df['trend_slope'] = df['trend_slope'].fillna(0)
        df['trend_strength'] = df['trend_strength'].fillna(0)
        df['trend_acceleration'] = df['trend_acceleration'].fillna(0)
        df['detrended_volatility'] = df['detrended_volatility'].fillna(0)
        df['trend_direction'] = df['trend_direction'].fillna(0)

        logger.info(f"    ✓ Trend features extracted for {len(trend_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Trend feature extraction failed: {e}")
        df['trend_slope'] = 0
        df['trend_strength'] = 0
        df['trend_acceleration'] = 0
        df['detrended_volatility'] = 0
        df['trend_direction'] = 0

    return df


# =============================================================================
# RECENCY-WEIGHTED FEATURES
# =============================================================================

def extract_recency_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract features that emphasize recent behavior.

    Recent activity is often more predictive for bot detection:
    - Recent trend may differ from overall trend
    - Recent volatility indicates current state
    - Momentum shows acceleration direction

    Features computed:
    - recent_activity_ratio: Last 30 days / historical daily average
    - recent_volatility_ratio: Recent CV / historical CV
    - momentum_score: Exponentially-weighted trend
    - recency_concentration: Fraction of activity in last 30 days

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with recency features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting recency-weighted features...")

    recency_query = f"""
    WITH daily_counts AS (
        SELECT
            {schema.location_field} as geo_location,
            DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
            COUNT(*) as daily_downloads
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
    ),
    date_bounds AS (
        SELECT
            geo_location,
            MIN(day) as first_day,
            MAX(day) as last_day
        FROM daily_counts
        GROUP BY geo_location
    ),
    recency_stats AS (
        SELECT
            dc.geo_location,
            -- Historical stats (all data)
            AVG(dc.daily_downloads) as hist_mean,
            STDDEV(dc.daily_downloads) as hist_std,
            SUM(dc.daily_downloads) as total_downloads,
            COUNT(*) as total_days,
            -- Recent stats (last 30 days)
            AVG(CASE
                WHEN dc.day >= db.last_day - INTERVAL '30 days'
                THEN dc.daily_downloads
                ELSE NULL
            END) as recent_mean,
            STDDEV(CASE
                WHEN dc.day >= db.last_day - INTERVAL '30 days'
                THEN dc.daily_downloads
                ELSE NULL
            END) as recent_std,
            SUM(CASE
                WHEN dc.day >= db.last_day - INTERVAL '30 days'
                THEN dc.daily_downloads
                ELSE 0
            END) as recent_downloads,
            COUNT(CASE
                WHEN dc.day >= db.last_day - INTERVAL '30 days'
                THEN 1
                ELSE NULL
            END) as recent_days
        FROM daily_counts dc
        JOIN date_bounds db ON dc.geo_location = db.geo_location
        GROUP BY dc.geo_location
    )
    SELECT
        geo_location,
        -- Recent activity ratio
        CASE
            WHEN hist_mean > 0
            THEN recent_mean / hist_mean
            ELSE 1
        END as recent_activity_ratio,
        -- Recent volatility ratio
        CASE
            WHEN hist_std > 0 AND hist_mean > 0 AND recent_mean > 0
            THEN (recent_std / NULLIF(recent_mean, 0)) / (hist_std / hist_mean)
            ELSE 1
        END as recent_volatility_ratio,
        -- Recency concentration
        CASE
            WHEN total_downloads > 0
            THEN recent_downloads::FLOAT / total_downloads
            ELSE 0
        END as recency_concentration,
        total_days,
        recent_days
    FROM recency_stats
    """

    try:
        recency_df = conn.execute(recency_query).df()

        # Calculate momentum using exponential weighting on weekly data
        weekly_query = f"""
        SELECT
            {schema.location_field} as geo_location,
            DATE_TRUNC('week', CAST({schema.timestamp_field} AS TIMESTAMP)) as week,
            COUNT(*) as weekly_downloads
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """

        weekly_df = conn.execute(weekly_query).df()

        def compute_momentum(group):
            """Compute exponentially-weighted momentum."""
            downloads = group['weekly_downloads'].values
            n = len(downloads)

            if n < 3:
                return 0

            # Exponential weights (recent = more weight)
            alpha = 0.3
            weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
            weights = weights / weights.sum()

            # Weighted mean
            weighted_mean = np.sum(weights * downloads)

            # Recent average (last 3 weeks)
            recent_avg = np.mean(downloads[-3:])

            # Historical average (before last 3 weeks)
            if n > 3:
                hist_avg = np.mean(downloads[:-3])
            else:
                hist_avg = weighted_mean

            # Momentum: recent vs historical, weighted
            momentum = (recent_avg - hist_avg) / (hist_avg + 1e-10)

            return float(np.clip(momentum, -5, 5))

        momentum_df = weekly_df.groupby('geo_location').apply(
            lambda g: pd.Series({'momentum_score': compute_momentum(g)})
        ).reset_index()

        # Merge all recency features
        recency_df = recency_df.merge(momentum_df, on='geo_location', how='left')

        df = df.merge(
            recency_df[['geo_location', 'recent_activity_ratio', 'recent_volatility_ratio',
                        'recency_concentration', 'momentum_score']],
            on='geo_location',
            how='left'
        )

        # Fill defaults
        df['recent_activity_ratio'] = df['recent_activity_ratio'].fillna(1)
        df['recent_volatility_ratio'] = df['recent_volatility_ratio'].fillna(1)
        df['recency_concentration'] = df['recency_concentration'].fillna(0)
        df['momentum_score'] = df['momentum_score'].fillna(0)

        logger.info(f"    ✓ Recency features extracted for {len(recency_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Recency feature extraction failed: {e}")
        df['recent_activity_ratio'] = 1
        df['recent_volatility_ratio'] = 1
        df['recency_concentration'] = 0
        df['momentum_score'] = 0

    return df


# =============================================================================
# DISTRIBUTION SHAPE FEATURES
# =============================================================================

def extract_distribution_shape_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract higher-order statistics of download distribution.

    Distribution shape reveals behavior patterns:
    - Bots: Often left-skewed (many low-activity periods, few high)
    - Legitimate: More symmetric or right-skewed
    - Heavy tails: Extreme events (attacks)

    Features computed:
    - download_skewness: Skewness of daily download distribution
    - download_kurtosis: Kurtosis (tail heaviness)
    - tail_heaviness_ratio: Extreme values / median
    - zero_day_ratio: Fraction of days with no activity

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with distribution shape features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting distribution shape features...")

    daily_query = f"""
    SELECT
        {schema.location_field} as geo_location,
        DATE_TRUNC('day', CAST({schema.timestamp_field} AS TIMESTAMP)) as day,
        COUNT(*) as daily_downloads
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY 1, 2
    """

    try:
        daily_df = conn.execute(daily_query).df()

        def compute_shape_stats(group):
            """Compute distribution shape statistics."""
            downloads = group['daily_downloads'].values
            n = len(downloads)

            if n < 5:
                return pd.Series({
                    'download_skewness': 0,
                    'download_kurtosis': 0,
                    'tail_heaviness_ratio': 1,
                    'zero_day_ratio': 0
                })

            # Skewness and kurtosis
            try:
                skewness = stats.skew(downloads)
                kurtosis = stats.kurtosis(downloads)
            except Exception:
                skewness = 0
                kurtosis = 0

            # Tail heaviness: 99th percentile / median
            median_dl = np.median(downloads)
            p99 = np.percentile(downloads, 99)
            tail_heaviness = p99 / (median_dl + 1e-10)

            # Zero-day ratio (if we had a full date range)
            zero_day_ratio = np.sum(downloads == 0) / n

            return pd.Series({
                'download_skewness': float(np.clip(skewness, -10, 10)),
                'download_kurtosis': float(np.clip(kurtosis, -10, 50)),
                'tail_heaviness_ratio': float(min(tail_heaviness, 100)),
                'zero_day_ratio': float(zero_day_ratio)
            })

        shape_df = daily_df.groupby('geo_location').apply(compute_shape_stats).reset_index()

        df = df.merge(shape_df, on='geo_location', how='left')

        # Fill defaults
        df['download_skewness'] = df['download_skewness'].fillna(0)
        df['download_kurtosis'] = df['download_kurtosis'].fillna(0)
        df['tail_heaviness_ratio'] = df['tail_heaviness_ratio'].fillna(1)
        df['zero_day_ratio'] = df['zero_day_ratio'].fillna(0)

        logger.info(f"    ✓ Distribution shape features extracted for {len(shape_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Distribution shape feature extraction failed: {e}")
        df['download_skewness'] = 0
        df['download_kurtosis'] = 0
        df['tail_heaviness_ratio'] = 1
        df['zero_day_ratio'] = 0

    return df


# =============================================================================
# BOT SIGNATURE TEMPORAL FEATURES
# =============================================================================

def extract_bot_signature_temporal_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract temporal features specifically designed for bot vs human discrimination.

    These features capture signatures that are almost impossible for bots to mimic:
    - autocorrelation_lag1: Day-to-day correlation (bots=high, humans=low)
    - circadian_deviation: Distance from human circadian rhythm
    - request_timing_entropy: Entropy of request timing within hours

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with bot signature temporal features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    escaped_path = input_parquet.replace("'", "''")

    logger.info("  Extracting bot signature temporal features...")

    # Query for hourly patterns per location
    hourly_query = f"""
    SELECT
        {schema.location_field} as geo_location,
        EXTRACT(HOUR FROM CAST({schema.timestamp_field} AS TIMESTAMP)) as hour,
        COUNT(*) as hourly_downloads
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY 1, 2
    ORDER BY 1, 2
    """

    try:
        hourly_df = conn.execute(hourly_query).df()

        def compute_bot_signatures(group):
            """Compute bot signature features for a location."""
            hours = group['hour'].values
            counts = group['hourly_downloads'].values

            # Build 24-hour distribution
            hour_dist = np.zeros(24)
            for h, c in zip(hours, counts):
                hour_dist[int(h)] = c
            total = hour_dist.sum()
            if total > 0:
                hour_dist = hour_dist / total

            # 1. Circadian deviation: how far from human circadian rhythm
            # Human pattern: peak at 10-11am, dip at 3-4am
            # Using a cosine model: peak at hour 14 (2pm), trough at hour 3 (3am)
            human_pattern = np.array([
                0.01, 0.005, 0.003, 0.002, 0.005, 0.01,   # 0-5am (low)
                0.02, 0.04, 0.06, 0.07, 0.08, 0.08,        # 6-11am (rising)
                0.07, 0.07, 0.08, 0.07, 0.06, 0.05,        # 12-5pm (plateau)
                0.04, 0.04, 0.04, 0.03, 0.02, 0.015        # 6-11pm (declining)
            ])
            human_pattern = human_pattern / human_pattern.sum()

            # Jensen-Shannon divergence from human pattern
            from scipy.spatial.distance import jensenshannon
            circadian_dev = float(jensenshannon(hour_dist + 1e-10, human_pattern + 1e-10))

            # 2. Request timing entropy: how uniform is the distribution
            # Bots tend to have more uniform or very concentrated distributions
            # Humans have moderate entropy (not uniform, not concentrated)
            entropy = -np.sum(hour_dist * np.log(hour_dist + 1e-10))
            max_entropy = np.log(24)
            timing_entropy = entropy / max_entropy if max_entropy > 0 else 0

            # 3. Autocorrelation lag 1: sequential hour correlation
            # Bots: high autocorrelation (predictable patterns)
            # Humans: moderate autocorrelation (some structure but variable)
            if len(hour_dist) >= 2:
                autocorr = np.corrcoef(hour_dist[:-1], hour_dist[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0

            return pd.Series({
                'autocorrelation_lag1': float(autocorr),
                'circadian_deviation': float(circadian_dev),
                'request_timing_entropy': float(timing_entropy),
            })

        sig_df = hourly_df.groupby('geo_location').apply(compute_bot_signatures).reset_index()

        df = df.merge(sig_df, on='geo_location', how='left')

        df['autocorrelation_lag1'] = df['autocorrelation_lag1'].fillna(0)
        df['circadian_deviation'] = df['circadian_deviation'].fillna(0.5)
        df['request_timing_entropy'] = df['request_timing_entropy'].fillna(0.5)

        logger.info(f"    ✓ Bot signature temporal features extracted for {len(sig_df)} locations")
    except Exception as e:
        logger.warning(f"    ✗ Bot signature temporal feature extraction failed: {e}")
        df['autocorrelation_lag1'] = 0
        df['circadian_deviation'] = 0.5
        df['request_timing_entropy'] = 0.5

    return df


# =============================================================================
# COMBINED TIME SERIES EXTRACTION (updated to include bot signatures)
# =============================================================================

def extract_all_timeseries_features(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    schema: Optional[LogSchema] = None
) -> pd.DataFrame:
    """
    Extract all time series features in one call.

    This is a convenience function that runs all time series extractors:
    - Outburst detection (6 features)
    - Periodicity detection (4 features)
    - Trend analysis (5 features)
    - Recency weighting (4 features)
    - Distribution shape (4 features)
    - Bot signature temporal (3 features)

    Total: 26 time series features

    Args:
        df: DataFrame with basic location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        schema: LogSchema for field mappings

    Returns:
        DataFrame with all time series features added
    """
    if schema is None:
        schema = EBI_SCHEMA

    logger.info("Extracting advanced time series features...")

    df = extract_outburst_features(df, input_parquet, conn, schema)
    df = extract_periodicity_features(df, input_parquet, conn, schema)
    df = extract_trend_features(df, input_parquet, conn, schema)
    df = extract_recency_features(df, input_parquet, conn, schema)
    df = extract_distribution_shape_features(df, input_parquet, conn, schema)
    df = extract_bot_signature_temporal_features(df, input_parquet, conn, schema)

    logger.info("✓ Advanced time series features extraction complete")

    return df


# Feature lists for reference
OUTBURST_FEATURES = [
    'outburst_count',
    'outburst_intensity',
    'max_outburst_zscore',
    'outburst_ratio',
    'time_since_last_outburst',
    'longest_outburst_streak',
]

PERIODICITY_FEATURES = [
    'weekly_autocorr',
    'dominant_period_days',
    'periodicity_strength',
    'period_regularity',
]

TREND_FEATURES = [
    'trend_slope',
    'trend_strength',
    'trend_acceleration',
    'detrended_volatility',
    'trend_direction',
]

RECENCY_FEATURES = [
    'recent_activity_ratio',
    'recent_volatility_ratio',
    'recency_concentration',
    'momentum_score',
]

DISTRIBUTION_SHAPE_FEATURES = [
    'download_skewness',
    'download_kurtosis',
    'tail_heaviness_ratio',
    'zero_day_ratio',
]

BOT_SIGNATURE_TEMPORAL_FEATURES = [
    'autocorrelation_lag1',
    'circadian_deviation',
    'request_timing_entropy',
]

ALL_TIMESERIES_FEATURES = (
    OUTBURST_FEATURES +
    PERIODICITY_FEATURES +
    TREND_FEATURES +
    RECENCY_FEATURES +
    DISTRIBUTION_SHAPE_FEATURES +
    BOT_SIGNATURE_TEMPORAL_FEATURES
)
