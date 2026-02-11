#!/usr/bin/env python3
"""Aggregate 159M download records to location-level features using DuckDB.

This script performs out-of-core aggregation of the full PRIDE download
dataset (159M records, 4.7GB) into ~71K unique geographic locations with
computed features ready for classification.

Usage:
    python scripts/aggregate_locations.py \
        --input pride_data/data_downloads_parquet.parquet \
        --output output/phase3_classification/locations_aggregated.parquet
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load config directly to avoid torch import chain
config_path = project_root / 'deeplogbot' / 'config.yaml'
with open(config_path, 'r') as f:
    APP_CONFIG = yaml.safe_load(f)

logger = logging.getLogger('deeplogbot')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def configure_duckdb(conn):
    """Configure DuckDB for large dataset processing."""
    duckdb_config = APP_CONFIG.get('duckdb', {})
    memory_limit = duckdb_config.get('memory_limit', '10GB')
    temp_dir = os.path.abspath(duckdb_config.get('temp_directory', './duckdb-tmp/'))
    max_temp = duckdb_config.get('max_temp_directory_size', '10GiB')

    os.makedirs(temp_dir, exist_ok=True)
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
    conn.execute(f"PRAGMA max_temp_directory_size='{max_temp}'")
    conn.execute("PRAGMA threads=4")

    logger.info(f"  DuckDB config: memory={memory_limit}, temp={temp_dir}")


def get_dataset_overview(conn, parquet_path: str) -> dict:
    """Get overview statistics of the full dataset."""
    escaped = os.path.abspath(parquet_path).replace("'", "''")

    logger.info("Getting dataset overview...")

    stats = {}

    # Total records
    total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped}')").fetchone()[0]
    stats['total_records'] = int(total)
    logger.info(f"  Total records: {total:,}")

    # Date range
    try:
        date_range = conn.execute(f"""
            SELECT MIN(CAST(timestamp AS TIMESTAMP)) as min_date,
                   MAX(CAST(timestamp AS TIMESTAMP)) as max_date
            FROM read_parquet('{escaped}')
            WHERE timestamp IS NOT NULL
        """).fetchone()
        stats['date_range'] = {
            'start': str(date_range[0]),
            'end': str(date_range[1]),
        }
        logger.info(f"  Date range: {date_range[0]} to {date_range[1]}")
    except Exception as e:
        logger.warning(f"  Could not get date range: {e}")

    # Countries
    try:
        n_countries = conn.execute(f"""
            SELECT COUNT(DISTINCT country) FROM read_parquet('{escaped}')
            WHERE country IS NOT NULL
        """).fetchone()[0]
        stats['n_countries'] = int(n_countries)
        logger.info(f"  Countries: {n_countries}")
    except Exception:
        pass

    # Unique locations
    try:
        n_locations = conn.execute(f"""
            SELECT COUNT(DISTINCT geo_location) FROM read_parquet('{escaped}')
            WHERE geo_location IS NOT NULL
        """).fetchone()[0]
        stats['n_unique_locations'] = int(n_locations)
        logger.info(f"  Unique locations: {n_locations:,}")
    except Exception:
        pass

    # Protocols
    try:
        protocols = conn.execute(f"""
            SELECT method, COUNT(*) as cnt
            FROM read_parquet('{escaped}')
            WHERE method IS NOT NULL
            GROUP BY method
            ORDER BY cnt DESC
        """).fetchdf()
        stats['protocols'] = dict(zip(protocols['method'], protocols['cnt'].astype(int)))
        for _, row in protocols.iterrows():
            logger.info(f"  Protocol {row['method']}: {row['cnt']:,}")
    except Exception:
        pass

    # Unique projects
    try:
        n_projects = conn.execute(f"""
            SELECT COUNT(DISTINCT accession) FROM read_parquet('{escaped}')
            WHERE accession IS NOT NULL
        """).fetchone()[0]
        stats['n_projects'] = int(n_projects)
        logger.info(f"  Unique projects: {n_projects:,}")
    except Exception:
        pass

    return stats


def aggregate_to_locations(conn, parquet_path: str, min_downloads: int = 10) -> pd.DataFrame:
    """
    Aggregate raw download records to location-level features.

    This is the core aggregation that transforms 159M records into ~71K locations
    with computed features suitable for classification.
    """
    escaped = os.path.abspath(parquet_path).replace("'", "''")

    logger.info(f"Aggregating to location level (min_downloads={min_downloads})...")

    query = f"""
    SELECT
        geo_location,
        country,
        -- Volume metrics
        COUNT(*) as total_downloads,
        COUNT(DISTINCT "user") as unique_users,
        COUNT(DISTINCT accession) as unique_projects,
        COUNT(DISTINCT filename) as unique_files,
        -- Downloads per user
        CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT "user"), 0) as downloads_per_user,
        -- Temporal features
        MIN(CAST(timestamp AS TIMESTAMP)) as first_seen,
        MAX(CAST(timestamp AS TIMESTAMP)) as last_seen,
        -- Working hours ratio (9-17 local time approximation using UTC)
        SUM(CASE WHEN EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) BETWEEN 9 AND 17
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as working_hours_ratio,
        -- Night activity (0-6, 22-23)
        SUM(CASE WHEN EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) < 6
                   OR EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) >= 22
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as night_activity_ratio,
        -- Weekend ratio
        SUM(CASE WHEN EXTRACT(DAYOFWEEK FROM CAST(timestamp AS TIMESTAMP)) IN (0, 6)
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as weekend_ratio,
        -- Hourly diversity
        COUNT(DISTINCT EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP))) as distinct_hours,
        -- File diversity
        CAST(COUNT(DISTINCT filename) AS DOUBLE) / NULLIF(COUNT(*), 0) as file_diversity_ratio,
        -- Year span
        COUNT(DISTINCT year) as years_active,
        MIN(year) as first_year,
        MAX(year) as last_year,
        -- Protocol usage
        SUM(CASE WHEN method = 'http' THEN 1 ELSE 0 END) as http_downloads,
        SUM(CASE WHEN method = 'ftp' THEN 1 ELSE 0 END) as ftp_downloads,
        SUM(CASE WHEN method = 'fasp-aspera' THEN 1 ELSE 0 END) as aspera_downloads,
        SUM(CASE WHEN method = 'gridftp-globus' THEN 1 ELSE 0 END) as globus_downloads,
        -- User diversity
        CAST(COUNT(DISTINCT "user") AS DOUBLE) / NULLIF(COUNT(DISTINCT accession), 0) as users_per_project,
        -- Projects per user
        CAST(COUNT(DISTINCT accession) AS DOUBLE) / NULLIF(COUNT(DISTINCT "user"), 0) as projects_per_user
    FROM read_parquet('{escaped}')
    WHERE geo_location IS NOT NULL
    GROUP BY geo_location, country
    HAVING COUNT(*) >= {min_downloads}
    ORDER BY total_downloads DESC
    """

    df = conn.execute(query).df()
    logger.info(f"  Aggregated to {len(df):,} locations (>={min_downloads} downloads)")

    # Add derived features
    df['downloads_per_year'] = df['total_downloads'] / df['years_active'].clip(lower=1)
    df['years_span'] = df['last_year'] - df['first_year'] + 1

    # Hourly entropy approximation
    df['hourly_entropy'] = df['distinct_hours'].apply(
        lambda x: np.log(max(x, 1)) / np.log(24) if x > 0 else 0
    )

    # Protocol ratios
    total = df['total_downloads'].clip(lower=1)
    df['http_ratio'] = df['http_downloads'] / total
    df['ftp_ratio'] = df['ftp_downloads'] / total
    df['aspera_ratio'] = df['aspera_downloads'] / total
    df['globus_ratio'] = df['globus_downloads'] / total

    return df


def save_aggregated_data(df: pd.DataFrame, output_path: str, dataset_stats: dict):
    """Save aggregated location data and metadata."""
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save as parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"  Saved aggregated data: {output_path}")
    logger.info(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

    # Also save as CSV for inspection
    csv_path = output_path.replace('.parquet', '.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved CSV: {csv_path}")

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'dataset_stats': dataset_stats,
        'aggregation': {
            'n_locations': len(df),
            'total_downloads': int(df['total_downloads'].sum()),
            'n_countries': int(df['country'].nunique()),
            'columns': list(df.columns),
        },
        'location_stats': {
            'downloads': {
                'min': int(df['total_downloads'].min()),
                'max': int(df['total_downloads'].max()),
                'median': int(df['total_downloads'].median()),
                'mean': float(df['total_downloads'].mean()),
            },
            'users': {
                'min': int(df['unique_users'].min()),
                'max': int(df['unique_users'].max()),
                'median': int(df['unique_users'].median()),
                'mean': float(df['unique_users'].mean()),
            },
        }
    }

    meta_path = os.path.join(output_dir, 'aggregation_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate 159M download records to location-level features'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input parquet file with download records'
    )
    parser.add_argument(
        '-o', '--output',
        default='output/phase3_classification/locations_aggregated.parquet',
        help='Output parquet file for aggregated locations'
    )
    parser.add_argument(
        '--min-downloads', type=int, default=10,
        help='Minimum downloads for a location to be included (default: 10)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("LOCATION AGGREGATION (Phase 3, Step 1)")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Min downloads: {args.min_downloads}")

    conn = duckdb.connect()
    configure_duckdb(conn)

    # Step 1: Dataset overview
    dataset_stats = get_dataset_overview(conn, args.input)

    # Step 2: Aggregate
    df = aggregate_to_locations(conn, args.input, args.min_downloads)

    # Step 3: Save
    save_aggregated_data(df, args.output, dataset_stats)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  {dataset_stats.get('total_records', 0):,} records → {len(df):,} locations")
    logger.info(f"  Countries: {df['country'].nunique()}")
    logger.info(f"  Total downloads: {df['total_downloads'].sum():,}")

    conn.close()


if __name__ == '__main__':
    main()
