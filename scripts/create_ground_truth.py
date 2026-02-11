#!/usr/bin/env python3
"""Create ground truth dataset for benchmarking classification methods.

Uses high-confidence heuristics to label locations as bot, hub, or organic.
These labels serve as pseudo-ground-truth for evaluating classifier agreement
and performance.

Usage:
    python scripts/create_ground_truth.py \
        --input pride_data/data_downloads_parquet.parquet \
        --output output/phase2_benchmarking/ground_truth \
        --sample-size 500000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import logging
import yaml

import pandas as pd
import numpy as np
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load config directly to avoid triggering torch import chain
config_path = project_root / 'deeplogbot' / 'config.yaml'
with open(config_path, 'r') as f:
    APP_CONFIG = yaml.safe_load(f)

# Simple logger setup
logger = logging.getLogger('deeplogbot')
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def aggregate_to_locations(conn, parquet_path: str) -> pd.DataFrame:
    """Aggregate raw download records to location-level features using DuckDB."""
    escaped_path = os.path.abspath(parquet_path).replace("'", "''")

    # Configure DuckDB
    duckdb_config = APP_CONFIG.get('duckdb', {})
    memory_limit = duckdb_config.get('memory_limit', '10GB')
    temp_dir = os.path.abspath(duckdb_config.get('temp_directory', './duckdb-tmp/'))
    max_temp = duckdb_config.get('max_temp_directory_size', '10GiB')
    os.makedirs(temp_dir, exist_ok=True)
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
    conn.execute(f"PRAGMA max_temp_directory_size='{max_temp}'")

    query = f"""
    SELECT
        geo_location,
        country,
        COUNT(*) as total_downloads,
        COUNT(DISTINCT "user") as unique_users,
        COUNT(DISTINCT accession) as unique_projects,
        MIN(CAST(timestamp AS TIMESTAMP)) as first_seen,
        MAX(CAST(timestamp AS TIMESTAMP)) as last_seen,
        -- Compute downloads_per_user
        CAST(COUNT(*) AS DOUBLE) / NULLIF(COUNT(DISTINCT "user"), 0) as downloads_per_user,
        -- Working hours ratio (9-17)
        SUM(CASE WHEN EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) BETWEEN 9 AND 17
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as working_hours_ratio,
        -- Night activity ratio (0-6, 22-23)
        SUM(CASE WHEN EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) < 6
                   OR EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP)) >= 22
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as night_activity_ratio,
        -- Weekend ratio
        SUM(CASE WHEN EXTRACT(DAYOFWEEK FROM CAST(timestamp AS TIMESTAMP)) IN (0, 6)
            THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as weekend_ratio,
        -- Hourly entropy approximation (number of distinct hours used)
        COUNT(DISTINCT EXTRACT(HOUR FROM CAST(timestamp AS TIMESTAMP))) as distinct_hours,
        -- File diversity
        COUNT(DISTINCT filename) as unique_files,
        CAST(COUNT(DISTINCT filename) AS DOUBLE) / NULLIF(COUNT(*), 0) as file_diversity_ratio
    FROM read_parquet('{escaped_path}')
    WHERE geo_location IS NOT NULL
    GROUP BY geo_location, country
    HAVING COUNT(*) >= 10
    ORDER BY total_downloads DESC
    """

    logger.info("Aggregating download records to location level...")
    df = conn.execute(query).df()
    logger.info(f"  Aggregated to {len(df):,} locations")

    return df


def create_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ground truth labels using high-confidence heuristics.

    Categories:
    - definite_bot: Very clear bot patterns (many users, very low DL/user)
    - definite_hub: Very clear hub patterns (few users, very high DL/user)
    - definite_organic: Very clear human patterns (few users, low DL/user, working hours)
    - uncertain: Ambiguous cases for analysis

    Returns DataFrame with 'ground_truth_label' and 'ground_truth_confidence' columns.
    """
    logger.info("Creating ground truth labels with high-confidence heuristics...")

    n = len(df)
    labels = pd.Series(['uncertain'] * n, index=df.index)
    confidence = pd.Series([0.0] * n, index=df.index)
    criteria = pd.Series([''] * n, index=df.index)

    users = df['unique_users']
    dpu = df['downloads_per_user']
    whr = df['working_hours_ratio']
    total_dl = df['total_downloads']
    night = df['night_activity_ratio']

    # =====================================================
    # DEFINITE BOTS (highest confidence)
    # =====================================================
    # Ground truth bots: extremely many users with minimal downloads each
    # This is the clearest bot signal - no legitimate use case has this pattern
    mask_gt_bot = (users >= 10000) & (dpu <= 10)
    labels[mask_gt_bot] = 'bot'
    confidence[mask_gt_bot] = 0.99
    criteria[mask_gt_bot] = 'ground_truth_bot: users>=10000, dpu<=10'

    # Large-scale bots: many users with low downloads
    mask_large_bot = (~mask_gt_bot) & (users >= 5000) & (dpu <= 25)
    labels[mask_large_bot] = 'bot'
    confidence[mask_large_bot] = 0.95
    criteria[mask_large_bot] = 'large_scale_bot: users>=5000, dpu<=25'

    # Bot farms: coordinated many users with low activity
    mask_farm_bot = (~mask_gt_bot) & (~mask_large_bot) & (users >= 1000) & (dpu <= 50) & (whr <= 0.3)
    labels[mask_farm_bot] = 'bot'
    confidence[mask_farm_bot] = 0.90
    criteria[mask_farm_bot] = 'bot_farm: users>=1000, dpu<=50, whr<=0.3'

    bot_mask = mask_gt_bot | mask_large_bot | mask_farm_bot
    n_bots = bot_mask.sum()

    # =====================================================
    # DEFINITE HUBS (high confidence)
    # =====================================================
    # Mirrors: very few users downloading huge amounts (institutional mirrors)
    mask_mirror = (~bot_mask) & (users <= 5) & (dpu >= 1000)
    labels[mask_mirror] = 'hub'
    confidence[mask_mirror] = 0.95
    criteria[mask_mirror] = 'mirror: users<=5, dpu>=1000'

    # Institutional hubs: few users, high downloads, some working hours
    mask_inst_hub = (~bot_mask) & (~mask_mirror) & (users <= 20) & (dpu >= 500)
    labels[mask_inst_hub] = 'hub'
    confidence[mask_inst_hub] = 0.90
    criteria[mask_inst_hub] = 'hub: users<=20, dpu>=500'

    # Research infrastructure: moderate users, very high total, regular hours
    mask_research_hub = (~bot_mask) & (~mask_mirror) & (~mask_inst_hub) & \
                        (users >= 10) & (users <= 200) & (dpu >= 200) & \
                        (total_dl >= 100000) & (whr >= 0.2)
    labels[mask_research_hub] = 'hub'
    confidence[mask_research_hub] = 0.85
    criteria[mask_research_hub] = 'research_hub: 10<=users<=200, dpu>=200, total>=100K'

    hub_mask = mask_mirror | mask_inst_hub | mask_research_hub
    n_hubs = hub_mask.sum()

    # =====================================================
    # DEFINITE ORGANIC (high confidence)
    # =====================================================
    # Individual researchers: very few users, low downloads, working hours
    mask_individual = (~bot_mask) & (~hub_mask) & \
                      (users <= 3) & (dpu <= 20) & (whr >= 0.4)
    labels[mask_individual] = 'organic'
    confidence[mask_individual] = 0.95
    criteria[mask_individual] = 'individual_user: users<=3, dpu<=20, whr>=0.4'

    # Small research groups: few users, moderate downloads, working hours
    mask_research = (~bot_mask) & (~hub_mask) & (~mask_individual) & \
                    (users >= 3) & (users <= 30) & (dpu >= 5) & (dpu <= 100) & \
                    (whr >= 0.35)
    labels[mask_research] = 'organic'
    confidence[mask_research] = 0.90
    criteria[mask_research] = 'research_group: 3<=users<=30, 5<=dpu<=100, whr>=0.35'

    # Casual users: very few users, moderate downloads
    mask_casual = (~bot_mask) & (~hub_mask) & (~mask_individual) & (~mask_research) & \
                  (users <= 5) & (dpu <= 50) & (night <= 0.3)
    labels[mask_casual] = 'organic'
    confidence[mask_casual] = 0.85
    criteria[mask_casual] = 'casual_user: users<=5, dpu<=50, night<=0.3'

    organic_mask = mask_individual | mask_research | mask_casual
    n_organic = organic_mask.sum()

    # Remaining are uncertain
    n_uncertain = n - n_bots - n_hubs - n_organic

    logger.info(f"\nGround truth label distribution:")
    logger.info(f"  Definite Bots:    {n_bots:>6,} ({100*n_bots/n:.1f}%)")
    logger.info(f"  Definite Hubs:    {n_hubs:>6,} ({100*n_hubs/n:.1f}%)")
    logger.info(f"  Definite Organic: {n_organic:>6,} ({100*n_organic/n:.1f}%)")
    logger.info(f"  Uncertain:        {n_uncertain:>6,} ({100*n_uncertain/n:.1f}%)")

    df['ground_truth_label'] = labels
    df['ground_truth_confidence'] = confidence
    df['ground_truth_criteria'] = criteria

    return df


def save_ground_truth(df: pd.DataFrame, output_dir: str):
    """Save ground truth dataset with statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full ground truth
    gt_file = os.path.join(output_dir, 'ground_truth_full.csv')
    df.to_csv(gt_file, index=False)
    logger.info(f"Full ground truth saved to: {gt_file}")

    # Save labeled-only (exclude uncertain for evaluation)
    labeled_df = df[df['ground_truth_label'] != 'uncertain']
    labeled_file = os.path.join(output_dir, 'ground_truth_labeled.csv')
    labeled_df.to_csv(labeled_file, index=False)
    logger.info(f"Labeled ground truth ({len(labeled_df):,} samples) saved to: {labeled_file}")

    # Save per-category samples
    for label in ['bot', 'hub', 'organic']:
        cat_df = df[df['ground_truth_label'] == label]
        cat_file = os.path.join(output_dir, f'ground_truth_{label}.csv')
        cat_df.to_csv(cat_file, index=False)
        logger.info(f"  {label}: {len(cat_df):,} samples")

    # Save statistics
    stats = {
        'created': datetime.now().isoformat(),
        'total_locations': len(df),
        'labeled_locations': len(labeled_df),
        'distribution': {
            label: {
                'count': int((df['ground_truth_label'] == label).sum()),
                'percentage': float(100 * (df['ground_truth_label'] == label).sum() / len(df)),
                'total_downloads': int(df.loc[df['ground_truth_label'] == label, 'total_downloads'].sum()),
            }
            for label in ['bot', 'hub', 'organic', 'uncertain']
        },
        'criteria': {
            'bot': [
                'ground_truth_bot: users>=10000, dpu<=10 (conf=0.99)',
                'large_scale_bot: users>=5000, dpu<=25 (conf=0.95)',
                'bot_farm: users>=1000, dpu<=50, whr<=0.3 (conf=0.90)',
            ],
            'hub': [
                'mirror: users<=5, dpu>=1000 (conf=0.95)',
                'hub: users<=20, dpu>=500 (conf=0.90)',
                'research_hub: 10<=users<=200, dpu>=200, total>=100K (conf=0.85)',
            ],
            'organic': [
                'individual_user: users<=3, dpu<=20, whr>=0.4 (conf=0.95)',
                'research_group: 3<=users<=30, 5<=dpu<=100, whr>=0.35 (conf=0.90)',
                'casual_user: users<=5, dpu<=50, night<=0.3 (conf=0.85)',
            ],
        }
    }

    stats_file = os.path.join(output_dir, 'ground_truth_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Create ground truth dataset for benchmarking classification methods'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input parquet file with download records'
    )
    parser.add_argument(
        '-o', '--output', default='output/phase2_benchmarking/ground_truth',
        help='Output directory for ground truth files'
    )
    parser.add_argument(
        '-s', '--sample-size', type=int, default=None,
        help='Sample size for faster processing (optional)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("GROUND TRUTH DATASET CREATION")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Connect to DuckDB and aggregate
    conn = duckdb.connect()

    # If sample_size specified, create sampled parquet first
    actual_input = args.input
    if args.sample_size:
        import tempfile
        escaped = os.path.abspath(args.input).replace("'", "''")
        total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{escaped}')").fetchone()[0]
        if args.sample_size < total:
            logger.info(f"Sampling {args.sample_size:,} from {total:,} records...")
            temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet', prefix='gt_sample_')
            os.close(temp_fd)
            frac = args.sample_size / total
            conn.execute(f"""
                COPY (SELECT * FROM read_parquet('{escaped}') USING SAMPLE {frac*100:.2f} PERCENT (bernoulli))
                TO '{temp_path}' (FORMAT PARQUET)
            """)
            actual_input = temp_path
            logger.info(f"Sampled data saved to: {temp_path}")

    # Aggregate to location level
    df = aggregate_to_locations(conn, actual_input)

    # Create ground truth labels
    df = create_ground_truth_labels(df)

    # Save
    stats = save_ground_truth(df, args.output)

    logger.info("\n" + "=" * 70)
    logger.info("GROUND TRUTH CREATION COMPLETE")
    logger.info("=" * 70)

    conn.close()


if __name__ == '__main__':
    main()
