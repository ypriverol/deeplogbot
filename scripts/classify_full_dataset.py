#!/usr/bin/env python3
"""Classify the full PRIDE dataset using the best algorithm.

Takes aggregated location data (from aggregate_locations.py) and runs
the selected classification method to produce the final classification.

Usage:
    python scripts/classify_full_dataset.py \
        --input pride_data/data_downloads_parquet.parquet \
        --output-dir output/phase3_classification \
        --method deep \
        --sample-size 20000000
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeplogbot.main import run_bot_annotator
from deeplogbot.utils import logger


def classify_dataset(
    input_parquet: str,
    output_dir: str,
    method: str = 'deep',
    sample_size: int = None,
):
    """
    Run classification on the PRIDE dataset.

    Args:
        input_parquet: Path to raw parquet file
        output_dir: Output directory for classification results
        method: Classification method ('rules', 'deep')
        sample_size: Sample size (None = full dataset)
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"FULL DATASET CLASSIFICATION (method={method})")
    logger.info("=" * 70)
    logger.info(f"Input: {input_parquet}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Method: {method}")
    if sample_size:
        logger.info(f"Sample size: {sample_size:,}")

    start_time = time.time()

    # Run classification
    result = run_bot_annotator(
        input_parquet=input_parquet,
        output_parquet=None,
        output_dir=output_dir,
        contamination=0.15,
        compute_importances=False,
        sample_size=sample_size,
        classification_method=method,
        annotate=True,
        output_strategy='reports_only',
        provider='ebi',
    )

    elapsed = time.time() - start_time

    # Load and enrich analysis CSV
    analysis_file = os.path.join(output_dir, 'location_analysis.csv')
    if os.path.exists(analysis_file):
        df = pd.read_csv(analysis_file)
        logger.info(f"\nClassification complete: {len(df):,} locations")

        # Add unified labels for convenience
        df['is_bot'] = df.get('automation_category', pd.Series()) == 'bot'
        df['is_hub'] = (
            (df.get('automation_category', pd.Series()) == 'legitimate_automation') |
            (df.get('behavior_type', pd.Series()) == 'hub')
        )
        df['is_organic'] = ~df['is_bot'] & ~df['is_hub']

        # Save enriched version
        final_path = os.path.join(output_dir, 'pride_classification_final.csv')
        df.to_csv(final_path, index=False)
        logger.info(f"Final classification saved: {final_path}")

        # Also save as parquet
        parquet_path = os.path.join(output_dir, 'pride_classification_final.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Final parquet saved: {parquet_path}")

        # Generate summary
        generate_classification_summary(df, result, output_dir, elapsed, method, sample_size)

    return result


def generate_classification_summary(
    df: pd.DataFrame,
    result: dict,
    output_dir: str,
    elapsed: float,
    method: str,
    sample_size: int,
):
    """Generate comprehensive classification summary."""
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    total_locs = len(df)
    total_downloads = df['total_downloads'].sum() if 'total_downloads' in df.columns else 0

    # Classification distribution
    bot_mask = df['is_bot']
    hub_mask = df['is_hub']
    organic_mask = df['is_organic']

    bot_locs = bot_mask.sum()
    hub_locs = hub_mask.sum()
    organic_locs = organic_mask.sum()

    bot_dl = df.loc[bot_mask, 'total_downloads'].sum() if 'total_downloads' in df.columns else 0
    hub_dl = df.loc[hub_mask, 'total_downloads'].sum() if 'total_downloads' in df.columns else 0
    organic_dl = df.loc[organic_mask, 'total_downloads'].sum() if 'total_downloads' in df.columns else 0

    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Method: {method}")
    logger.info(f"  Total locations: {total_locs:,}")
    logger.info(f"  Total downloads: {total_downloads:,}")
    logger.info(f"  Runtime: {elapsed:.1f}s")
    logger.info(f"\n  Classification:")
    logger.info(f"    Bot:     {bot_locs:>6,} locs ({100*bot_locs/total_locs:.1f}%), "
                f"{bot_dl:>12,} DL ({100*bot_dl/total_downloads:.1f}%)" if total_downloads > 0 else "")
    logger.info(f"    Hub:     {hub_locs:>6,} locs ({100*hub_locs/total_locs:.1f}%), "
                f"{hub_dl:>12,} DL ({100*hub_dl/total_downloads:.1f}%)" if total_downloads > 0 else "")
    logger.info(f"    Organic: {organic_locs:>6,} locs ({100*organic_locs/total_locs:.1f}%), "
                f"{organic_dl:>12,} DL ({100*organic_dl/total_downloads:.1f}%)" if total_downloads > 0 else "")

    # Geographic distribution
    if 'country' in df.columns:
        country_counts = df['country'].value_counts()
        n_countries = len(country_counts)
        logger.info(f"\n  Countries: {n_countries}")

        # Bot downloads by country
        bot_by_country = df[bot_mask].groupby('country')['total_downloads'].sum().sort_values(ascending=False)

    # Subcategory distribution
    if 'subcategory' in df.columns:
        subcat_dist = df['subcategory'].value_counts()
        logger.info(f"\n  Subcategories:")
        for subcat, count in subcat_dist.head(15).items():
            logger.info(f"    {subcat}: {count:,}")

    # Top bot locations
    if bot_mask.any() and 'total_downloads' in df.columns:
        top_bots = df[bot_mask].nlargest(20, 'total_downloads')
        logger.info(f"\n  Top 20 Bot Locations (by downloads):")
        for _, row in top_bots.iterrows():
            loc = row.get('geo_location', 'unknown')
            country = row.get('country', 'unknown')
            users = row.get('unique_users', 0)
            dl = row.get('total_downloads', 0)
            dpu = row.get('downloads_per_user', 0)
            logger.info(f"    {country:>20} {loc[:30]:<30} {dl:>10,} DL, {users:>6,} users, {dpu:.1f} DL/user")

    # Top hub locations
    if hub_mask.any() and 'total_downloads' in df.columns:
        top_hubs = df[hub_mask].nlargest(20, 'total_downloads')
        logger.info(f"\n  Top 20 Hub Locations (by downloads):")
        for _, row in top_hubs.iterrows():
            loc = row.get('geo_location', 'unknown')
            country = row.get('country', 'unknown')
            users = row.get('unique_users', 0)
            dl = row.get('total_downloads', 0)
            dpu = row.get('downloads_per_user', 0)
            logger.info(f"    {country:>20} {loc[:30]:<30} {dl:>10,} DL, {users:>6,} users, {dpu:.1f} DL/user")

    # Save summary JSON
    summary = {
        'created': datetime.now().isoformat(),
        'method': method,
        'sample_size': sample_size,
        'runtime_seconds': elapsed,
        'total_locations': int(total_locs),
        'total_downloads': int(total_downloads),
        'classification': {
            'bot': {
                'locations': int(bot_locs),
                'locations_pct': float(100 * bot_locs / total_locs) if total_locs > 0 else 0,
                'downloads': int(bot_dl),
                'downloads_pct': float(100 * bot_dl / total_downloads) if total_downloads > 0 else 0,
            },
            'hub': {
                'locations': int(hub_locs),
                'locations_pct': float(100 * hub_locs / total_locs) if total_locs > 0 else 0,
                'downloads': int(hub_dl),
                'downloads_pct': float(100 * hub_dl / total_downloads) if total_downloads > 0 else 0,
            },
            'organic': {
                'locations': int(organic_locs),
                'locations_pct': float(100 * organic_locs / total_locs) if total_locs > 0 else 0,
                'downloads': int(organic_dl),
                'downloads_pct': float(100 * organic_dl / total_downloads) if total_downloads > 0 else 0,
            },
        },
        'countries': int(df['country'].nunique()) if 'country' in df.columns else 0,
    }

    # Subcategory distribution
    if 'subcategory' in df.columns:
        summary['subcategories'] = df['subcategory'].value_counts().to_dict()

    # Country distribution for bots/hubs
    if 'country' in df.columns and 'total_downloads' in df.columns:
        summary['bot_by_country'] = (
            df[bot_mask].groupby('country')['total_downloads']
            .sum().sort_values(ascending=False).head(30)
            .to_dict()
        )
        summary['hub_by_country'] = (
            df[hub_mask].groupby('country')['total_downloads']
            .sum().sort_values(ascending=False).head(30)
            .to_dict()
        )
        summary['organic_by_country'] = (
            df[organic_mask].groupby('country')['total_downloads']
            .sum().sort_values(ascending=False).head(30)
            .to_dict()
        )

    summary_file = os.path.join(summary_dir, 'classification_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n  Summary saved: {summary_file}")

    # Save top bot/hub tables as CSV
    if bot_mask.any():
        cols = [c for c in ['geo_location', 'country', 'total_downloads', 'unique_users',
                            'downloads_per_user', 'behavior_type', 'subcategory',
                            'classification_confidence'] if c in df.columns]
        df[bot_mask].nlargest(100, 'total_downloads')[cols].to_csv(
            os.path.join(summary_dir, 'top_100_bot_locations.csv'), index=False
        )

    if hub_mask.any():
        cols = [c for c in ['geo_location', 'country', 'total_downloads', 'unique_users',
                            'downloads_per_user', 'behavior_type', 'subcategory',
                            'classification_confidence'] if c in df.columns]
        df[hub_mask].nlargest(100, 'total_downloads')[cols].to_csv(
            os.path.join(summary_dir, 'top_100_hub_locations.csv'), index=False
        )


def main():
    parser = argparse.ArgumentParser(
        description='Classify the full PRIDE dataset'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input parquet file with download records'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='output/phase3_classification',
        help='Output directory'
    )
    parser.add_argument(
        '-m', '--method',
        choices=['rules', 'deep'],
        default='deep',
        help='Classification method (default: deep)'
    )
    parser.add_argument(
        '-s', '--sample-size', type=int, default=None,
        help='Sample size (default: full dataset)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    classify_dataset(
        input_parquet=args.input,
        output_dir=args.output_dir,
        method=args.method,
        sample_size=args.sample_size,
    )


if __name__ == '__main__':
    main()
