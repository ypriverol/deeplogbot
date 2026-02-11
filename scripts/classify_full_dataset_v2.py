#!/usr/bin/env python3
"""Classify full PRIDE dataset using the Deep method.

Uses the pre-aggregated locations from phase3_classification and runs
the Deep classification pipeline on all ~39K locations.

Approach:
  1. Load pre-aggregated locations (already has basic features)
  2. Run Deep classification on all locations
  3. Output classified CSV with bot/hub/organic labels
  4. Generate summary statistics

Usage:
    python scripts/classify_full_dataset_v2.py
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

INPUT_PARQUET = project_root / 'pride_data' / 'data_downloads_parquet.parquet'
LOCATIONS_CSV = project_root / 'output' / 'phase3_classification' / 'locations_aggregated.csv'
SAMPLE_PARQUET = project_root / 'output' / 'phase3_classification' / 'pride_sample_20m.parquet'
OUTPUT_DIR = project_root / 'output' / 'phase5_full_classification'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    print("=" * 70)
    print("FULL DATASET CLASSIFICATION (Deep Method)")
    print("=" * 70)

    # Load pre-aggregated locations
    print(f"\nLoading pre-aggregated locations from {LOCATIONS_CSV}")
    loc_df = pd.read_csv(LOCATIONS_CSV)
    print(f"  Loaded {len(loc_df):,} locations")
    print(f"  Columns: {list(loc_df.columns)}")

    # Run Deep classification through the main pipeline
    # We use the 20M sample parquet as input (has enough data for feature extraction)
    # and pass sample_size=None to use all data in the sample
    print("\nRunning Deep classification pipeline...")

    from deeplogbot.main import run_bot_annotator

    result = run_bot_annotator(
        input_parquet=str(SAMPLE_PARQUET),
        output_parquet=None,
        output_dir=str(OUTPUT_DIR / 'deep_output'),
        contamination=0.15,
        compute_importances=False,
        sample_size=None,  # Use all records in the 20M sample
        classification_method='deep',
        output_strategy='reports_only',
        annotate=True,
        provider='ebi',
    )

    elapsed = time.time() - start_time
    print(f"\nClassification completed in {elapsed:.1f}s")

    # Load the output analysis
    analysis_file = OUTPUT_DIR / 'deep_output' / 'location_analysis.csv'
    if analysis_file.exists():
        analysis_df = pd.read_csv(analysis_file, low_memory=False)
        print(f"\nLoaded {len(analysis_df):,} classified locations")

        # Extract labels
        labels = pd.Series('organic', index=analysis_df.index)
        if 'automation_category' in analysis_df.columns:
            labels[analysis_df['automation_category'] == 'bot'] = 'bot'
            labels[analysis_df['automation_category'] == 'legitimate_automation'] = 'hub'
        if 'behavior_type' in analysis_df.columns:
            labels[analysis_df['behavior_type'] == 'hub'] = 'hub'

        analysis_df['final_label'] = labels

        # Save final classified CSV
        output_csv = OUTPUT_DIR / 'pride_classification_final.csv'
        key_cols = ['geo_location', 'country', 'city', 'total_downloads', 'unique_users',
                    'downloads_per_user', 'behavior_type', 'automation_category',
                    'subcategory', 'classification_confidence', 'final_label']
        available_cols = [c for c in key_cols if c in analysis_df.columns]
        analysis_df[available_cols].to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv}")

        # Summary statistics
        summary = {
            'total_locations': len(analysis_df),
            'bot_locations': int((labels == 'bot').sum()),
            'hub_locations': int((labels == 'hub').sum()),
            'organic_locations': int((labels == 'organic').sum()),
        }

        if 'total_downloads' in analysis_df.columns:
            summary['total_downloads'] = int(analysis_df['total_downloads'].sum())
            summary['bot_downloads'] = int(analysis_df.loc[labels == 'bot', 'total_downloads'].sum())
            summary['hub_downloads'] = int(analysis_df.loc[labels == 'hub', 'total_downloads'].sum())
            summary['organic_downloads'] = int(analysis_df.loc[labels == 'organic', 'total_downloads'].sum())
            summary['bot_dl_pct'] = summary['bot_downloads'] / summary['total_downloads'] * 100
            summary['hub_dl_pct'] = summary['hub_downloads'] / summary['total_downloads'] * 100
            summary['organic_dl_pct'] = summary['organic_downloads'] / summary['total_downloads'] * 100

        # Country breakdown
        if 'country' in analysis_df.columns:
            country_stats = []
            for country, group in analysis_df.groupby('country'):
                gl = labels[group.index]
                cs = {
                    'country': country,
                    'locations': len(group),
                    'bot_locations': int((gl == 'bot').sum()),
                    'hub_locations': int((gl == 'hub').sum()),
                    'organic_locations': int((gl == 'organic').sum()),
                }
                if 'total_downloads' in group.columns:
                    cs['total_downloads'] = int(group['total_downloads'].sum())
                    cs['bot_downloads'] = int(group.loc[gl == 'bot', 'total_downloads'].sum())
                    cs['organic_downloads'] = int(group.loc[gl == 'organic', 'total_downloads'].sum())
                country_stats.append(cs)

            country_df = pd.DataFrame(country_stats).sort_values('total_downloads', ascending=False)
            country_df.to_csv(OUTPUT_DIR / 'classification_by_country.csv', index=False)
            summary['countries'] = len(country_df)
            summary['top5_countries'] = country_df.head(5)['country'].tolist()

        print(f"\n  Summary:")
        print(f"    Total locations: {summary['total_locations']:,}")
        print(f"    Bot: {summary['bot_locations']:,} ({summary['bot_locations']/summary['total_locations']*100:.1f}%)")
        print(f"    Hub: {summary['hub_locations']:,} ({summary['hub_locations']/summary['total_locations']*100:.1f}%)")
        print(f"    Organic: {summary['organic_locations']:,} ({summary['organic_locations']/summary['total_locations']*100:.1f}%)")
        if 'total_downloads' in summary:
            print(f"    Bot DL%: {summary['bot_dl_pct']:.1f}%")
            print(f"    Hub DL%: {summary['hub_dl_pct']:.1f}%")
            print(f"    Organic DL%: {summary['organic_dl_pct']:.1f}%")

        # Save summary
        with open(OUTPUT_DIR / 'classification_summary.json', 'w') as f:
            json.dump({k: v for k, v in summary.items() if not isinstance(v, list) or k == 'top5_countries'}, f, indent=2, default=str)

    # Checkpoint
    checkpoint = {
        'phase': 'phase5_full_classification',
        'timestamp': datetime.now().isoformat(),
        'status': 'complete',
        'method': 'deep',
        'input_file': str(SAMPLE_PARQUET),
        'elapsed_seconds': elapsed,
        'output_files': [
            str(OUTPUT_DIR / 'pride_classification_final.csv'),
            str(OUTPUT_DIR / 'classification_by_country.csv'),
            str(OUTPUT_DIR / 'classification_summary.json'),
        ],
    }
    with open(OUTPUT_DIR / 'CHECKPOINT.json', 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\nCheckpoint saved: {OUTPUT_DIR / 'CHECKPOINT.json'}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
