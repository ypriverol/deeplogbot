"""Temporal consistency checks for bot detection.

Detects locations whose recent behavior diverges from their own history,
catching spikes and behavior shifts without fixed thresholds.
Uses modified z-scores (MAD-based) for robustness to outliers.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_temporal_anomaly(df: pd.DataFrame,
                             conn=None,
                             input_parquet: str = None) -> pd.DataFrame:
    """Add temporal consistency scores to location DataFrame.

    For each location, compares recent behavior (last 3 months) against
    its own historical baseline. Locations with sudden behavior shifts
    get high temporal_anomaly_score values.

    Args:
        df: DataFrame with location features (must have geo_location)
        conn: Optional DuckDB connection for querying parquet
        input_parquet: Path to parquet file

    Returns:
        DataFrame with added columns:
          temporal_anomaly_score: 0-1, higher = more temporally anomalous
          has_temporal_spike: bool, True if a significant spike was detected
    """
    df = df.copy()
    df['temporal_anomaly_score'] = 0.0
    df['has_temporal_spike'] = False

    # Use feature-based temporal signals (available without parquet query)
    _compute_feature_based_temporal(df)

    # If we have parquet access, compute monthly spike detection
    if conn is not None and input_parquet is not None:
        try:
            _compute_monthly_spikes(df, conn, input_parquet)
        except Exception as e:
            logger.warning(f"Monthly spike detection failed: {e}")

    n_spikes = df['has_temporal_spike'].sum()
    logger.info(f"  Temporal consistency: {n_spikes} locations with spikes detected")
    return df


def _compute_feature_based_temporal(df: pd.DataFrame):
    """Compute temporal anomaly from existing features.

    Uses fraction_latest_year, spike_ratio, recent_activity_ratio,
    and year_over_year_cv as signals of temporal inconsistency.
    """
    signals = []

    # Signal 1: Fraction of activity in latest year (spiky if > 0.8)
    if 'fraction_latest_year' in df.columns:
        s = df['fraction_latest_year'].fillna(0).clip(0, 1)
        # Transform: 0.5 = normal, 1.0 = all in latest year
        signals.append(np.clip((s - 0.5) * 2, 0, 1))

    # Signal 2: Spike ratio (high = bursty behavior)
    if 'spike_ratio' in df.columns:
        s = df['spike_ratio'].fillna(0)
        # Normalize using sigmoid-like transform
        signals.append(1 - 1 / (1 + s / 5))

    # Signal 3: Recent activity ratio (recent 30d vs historical)
    if 'recent_activity_ratio' in df.columns:
        s = df['recent_activity_ratio'].fillna(1).clip(0, 100)
        # > 3x historical = suspicious
        signals.append(np.clip((s - 1) / 10, 0, 1))

    # Signal 4: Year-over-year CV (high = inconsistent across years)
    if 'year_over_year_cv' in df.columns:
        s = df['year_over_year_cv'].fillna(0).clip(0, 10)
        signals.append(np.clip(s / 3, 0, 1))

    # Signal 5: Momentum (exponentially-weighted trend)
    if 'momentum_score' in df.columns:
        s = df['momentum_score'].fillna(0).abs()
        signals.append(np.clip(s / 5, 0, 1))

    if signals:
        # Average all available signals
        temporal_score = np.mean(signals, axis=0)
        df['temporal_anomaly_score'] = temporal_score

        # Flag locations with high temporal anomaly
        df['has_temporal_spike'] = temporal_score > 0.6


def _compute_monthly_spikes(df: pd.DataFrame, conn, input_parquet: str):
    """Detect monthly download spikes per location using modified z-scores.

    A spike is defined as a month with modified z-score > 3.5 relative
    to the location's own monthly download history.
    """
    p = str(input_parquet).replace("'", "''")

    # Get monthly downloads per location (sampled for performance)
    try:
        monthly = conn.execute(f"""
            SELECT geo_location, year, month, COUNT(*) as downloads
            FROM read_parquet('{p}')
            WHERE geo_location IS NOT NULL
            GROUP BY geo_location, year, month
        """).df()
    except Exception:
        # If full query is too heavy, sample
        monthly = conn.execute(f"""
            SELECT geo_location, year, month, COUNT(*) as downloads
            FROM (SELECT * FROM read_parquet('{p}') USING SAMPLE 10 PERCENT (system))
            WHERE geo_location IS NOT NULL
            GROUP BY geo_location, year, month
        """).df()

    if monthly.empty:
        return

    # Compute per-location spike scores
    spike_scores = {}
    for loc, group in monthly.groupby('geo_location'):
        dls = group['downloads'].values
        if len(dls) < 3:
            continue

        # Modified z-score using MAD (robust to outliers)
        median = np.median(dls)
        mad = np.median(np.abs(dls - median))
        if mad < 1:
            mad = 1  # Prevent division by zero

        modified_z = 0.6745 * (dls - median) / mad
        max_z = np.max(np.abs(modified_z))

        if max_z > 3.5:
            # Normalize to 0-1, capped at z=10
            spike_scores[loc] = min(max_z / 10, 1.0)

    if spike_scores:
        score_series = pd.Series(spike_scores)
        matching = df.index[df['geo_location'].isin(score_series.index)]
        if len(matching) > 0:
            geo_to_score = df.loc[matching, 'geo_location'].map(score_series)
            # Combine with feature-based score (max of both)
            df.loc[matching, 'temporal_anomaly_score'] = np.maximum(
                df.loc[matching, 'temporal_anomaly_score'].values,
                geo_to_score.fillna(0).values,
            )
            df.loc[matching, 'has_temporal_spike'] |= (geo_to_score.fillna(0) > 0).values

        logger.info(f"  Monthly spike detection: {len(spike_scores)} locations with spikes")
