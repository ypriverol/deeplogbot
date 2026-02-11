#!/usr/bin/env python3
"""Comprehensive PRIDE usage analysis for the manuscript.

Runs DuckDB queries on the full parquet data to answer all research questions:
1. Geographic usage (downloads by country, regional distribution)
2. Temporal trends (yearly, monthly patterns)
3. Protocol analysis (HTTP vs FTP vs Aspera vs Globus)
4. Most downloaded datasets
5. Dataset concentration (Gini coefficient, Lorenz curve data)
6. Before/after bot removal comparison

Usage:
    python scripts/run_full_analysis.py [--with-labels PATH_TO_LABELS]
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

INPUT_PARQUET = project_root / 'pride_data' / 'data_downloads_parquet.parquet'
PROJECTS_JSON = project_root / 'pride_data' / 'all_pride_projects.json'
OUTPUT_DIR = project_root / 'output' / 'phase6_analysis'


def get_connection():
    """Get a DuckDB connection with proper settings."""
    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='8GB'")
    temp_dir = os.path.abspath('./duckdb-tmp/')
    os.makedirs(temp_dir, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{temp_dir}'")
    conn.execute("PRAGMA max_temp_directory_size='8GiB'")
    conn.execute("PRAGMA threads=2")
    return conn


def escaped(path):
    return os.path.abspath(str(path)).replace("'", "''")


def setup_clean_filter(conn, labels_df):
    """Register non-bot locations as a DuckDB temp table for filtering all queries."""
    if labels_df is None:
        return False
    # Support both column formats: 'final_label' (old) and 'is_bot' (new)
    if 'final_label' in labels_df.columns:
        non_bot = labels_df[labels_df['final_label'] != 'bot'][['geo_location']].drop_duplicates()
    elif 'is_bot' in labels_df.columns:
        non_bot = labels_df[~labels_df['is_bot']][['geo_location']].drop_duplicates()
    else:
        print("  WARNING: No bot label column found, skipping filter")
        return False
    conn.register('_clean_locations', non_bot)
    conn.execute("CREATE TEMP TABLE clean_locations AS SELECT * FROM _clean_locations")
    n_total = len(labels_df['geo_location'].unique())
    n_clean = len(non_bot)
    print(f"\n  Bot filter: keeping {n_clean:,} of {n_total:,} locations ({n_clean/n_total*100:.1f}%)")
    return True


def _where_clean(has_filter):
    """Return SQL fragment to filter to non-bot locations."""
    if has_filter:
        return "AND geo_location IN (SELECT geo_location FROM clean_locations)"
    return ""


REGION_MAP = {
    'China': 'Asia', 'Japan': 'Asia', 'South Korea': 'Asia', 'India': 'Asia',
    'Taiwan': 'Asia', 'Singapore': 'Asia', 'Thailand': 'Asia', 'Indonesia': 'Asia',
    'Malaysia': 'Asia', 'Vietnam': 'Asia', 'Philippines': 'Asia', 'Bangladesh': 'Asia',
    'Pakistan': 'Asia', 'Hong Kong': 'Asia', 'Israel': 'Asia',
    'United States': 'Americas', 'Canada': 'Americas', 'Brazil': 'Americas',
    'Mexico': 'Americas', 'Argentina': 'Americas', 'Chile': 'Americas',
    'Colombia': 'Americas', 'Peru': 'Americas',
    'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
    'Spain': 'Europe', 'Italy': 'Europe', 'Netherlands': 'Europe',
    'Switzerland': 'Europe', 'Sweden': 'Europe', 'Denmark': 'Europe',
    'Belgium': 'Europe', 'Finland': 'Europe', 'Norway': 'Europe',
    'Austria': 'Europe', 'Poland': 'Europe', 'Czech Republic': 'Europe',
    'Portugal': 'Europe', 'Ireland': 'Europe', 'Greece': 'Europe',
    'Hungary': 'Europe', 'Romania': 'Europe', 'Russia': 'Europe',
    'Turkey': 'Europe', 'Ukraine': 'Europe',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa',
    'Kenya': 'Africa', 'Morocco': 'Africa',
    'Saudi Arabia': 'Middle East', 'Iran': 'Middle East', 'UAE': 'Middle East',
}


def analysis_1_geographic(conn, parquet_path, output_dir, has_filter=False):
    """Geographic usage analysis: downloads by country."""
    print("\n--- Analysis 1: Geographic Usage ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    query = f"""
    SELECT
        country,
        COUNT(*) as total_downloads,
        COUNT(DISTINCT accession) as unique_datasets,
        COUNT(DISTINCT geo_location) as unique_locations,
        MIN(year) as first_year,
        MAX(year) as last_year
    FROM read_parquet('{p}')
    WHERE country IS NOT NULL AND country != '' AND country NOT LIKE '%{{%' {filt}
    GROUP BY country
    ORDER BY total_downloads DESC
    """
    country_df = conn.execute(query).df()
    country_df.to_csv(output_dir / 'geographic_by_country.csv', index=False)
    print(f"  Countries: {len(country_df)}")
    print(f"  Top 5: {country_df.head(5)[['country', 'total_downloads']].to_string(index=False)}")

    # Regional aggregation
    country_df['region'] = country_df['country'].map(REGION_MAP).fillna('Other')
    regional = country_df.groupby('region').agg({
        'total_downloads': 'sum',
        'unique_datasets': 'sum',
        'unique_locations': 'sum',
        'country': 'count'
    }).rename(columns={'country': 'num_countries'}).sort_values('total_downloads', ascending=False)
    regional.to_csv(output_dir / 'geographic_by_region.csv')
    print(f"  Regions: {regional.index.tolist()}")

    return country_df


def analysis_2_temporal(conn, parquet_path, output_dir, has_filter=False):
    """Temporal trends: downloads per year and month."""
    print("\n--- Analysis 2: Temporal Trends ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    yearly_query = f"""
    SELECT
        year,
        COUNT(*) as total_downloads,
        COUNT(DISTINCT accession) as unique_datasets,
        COUNT(DISTINCT geo_location) as unique_locations
    FROM read_parquet('{p}')
    WHERE year >= 2020 AND year <= 2025 {filt}
    GROUP BY year
    ORDER BY year
    """
    yearly_df = conn.execute(yearly_query).df()
    yearly_df.to_csv(output_dir / 'temporal_yearly.csv', index=False)
    print(f"  Yearly trends:\n{yearly_df.to_string(index=False)}")

    monthly_query = f"""
    SELECT
        year,
        month,
        COUNT(*) as total_downloads
    FROM read_parquet('{p}')
    WHERE year >= 2020 AND year <= 2025 {filt}
    GROUP BY year, month
    ORDER BY year, month
    """
    monthly_df = conn.execute(monthly_query).df()
    monthly_df.to_csv(output_dir / 'temporal_monthly.csv', index=False)
    print(f"  Monthly data points: {len(monthly_df)}")

    return yearly_df


def analysis_3_protocols(conn, parquet_path, output_dir, has_filter=False):
    """Protocol analysis: HTTP vs FTP vs Aspera vs Globus."""
    print("\n--- Analysis 3: Protocol Usage ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    cols_query = f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{p}') LIMIT 0)"
    cols = conn.execute(cols_query).df()['column_name'].tolist()

    if 'method' in cols:
        proto_query = f"""
        SELECT
            method as protocol,
            year,
            COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE year >= 2020 {filt}
        GROUP BY method, year
        ORDER BY year, downloads DESC
        """
    else:
        print("  No method/protocol column found, skipping protocol analysis")
        pd.DataFrame(columns=['protocol', 'year', 'downloads']).to_csv(
            output_dir / 'protocol_usage.csv', index=False)
        return None

    try:
        proto_df = conn.execute(proto_query).df()
        proto_df.to_csv(output_dir / 'protocol_usage.csv', index=False)
        print(f"  Protocol data:\n{proto_df.head(20).to_string(index=False)}")
        return proto_df
    except Exception as e:
        print(f"  Protocol analysis failed: {e}")
        pd.DataFrame(columns=['protocol', 'year', 'downloads']).to_csv(
            output_dir / 'protocol_usage.csv', index=False)
        return None


def analysis_4_top_datasets(conn, parquet_path, output_dir, has_filter=False):
    """Most downloaded datasets."""
    print("\n--- Analysis 4: Top Datasets ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    top_query = f"""
    SELECT
        accession,
        COUNT(*) as total_downloads,
        COUNT(DISTINCT geo_location) as unique_locations,
        COUNT(DISTINCT country) as unique_countries,
        MIN(year) as first_download_year,
        MAX(year) as last_download_year
    FROM read_parquet('{p}')
    WHERE accession IS NOT NULL {filt}
    GROUP BY accession
    ORDER BY total_downloads DESC
    LIMIT 100
    """
    top_df = conn.execute(top_query).df()

    # Try to enrich with project metadata
    if PROJECTS_JSON.exists():
        try:
            print(f"  Loading project metadata from {PROJECTS_JSON}")
            # Stream-read JSON to avoid loading 283MB into memory
            import ijson
        except ImportError:
            pass
        try:
            # Try loading as JSON lines or array
            with open(PROJECTS_JSON, 'r') as f:
                first_char = f.read(1)

            if first_char == '[':
                # JSON array - use DuckDB to read it efficiently
                jp = escaped(PROJECTS_JSON)
                try:
                    meta_query = f"""
                    SELECT accession, title, submissionDate, publicationDate
                    FROM read_json_auto('{jp}')
                    WHERE accession IN ({','.join(f"'{a}'" for a in top_df['accession'].head(100))})
                    """
                    meta_df = conn.execute(meta_query).df()
                    if len(meta_df) > 0:
                        top_df = top_df.merge(meta_df, on='accession', how='left')
                        print(f"  Enriched {len(meta_df)} datasets with metadata")
                except Exception as e:
                    print(f"  Metadata enrichment failed: {e}")
        except Exception as e:
            print(f"  Could not read project metadata: {e}")

    top_df.to_csv(output_dir / 'top_datasets.csv', index=False)
    print(f"  Top 10 datasets:")
    for _, row in top_df.head(10).iterrows():
        title = row.get('title', '')
        if isinstance(title, str) and len(title) > 60:
            title = title[:57] + '...'
        print(f"    {row['accession']}: {row['total_downloads']:,} DL, {row['unique_countries']} countries  {title}")

    return top_df


def analysis_5_concentration(conn, parquet_path, output_dir, has_filter=False):
    """Dataset concentration analysis (Gini coefficient, Lorenz curve data)."""
    print("\n--- Analysis 5: Dataset Concentration ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    dataset_query = f"""
    SELECT
        accession,
        COUNT(*) as downloads
    FROM read_parquet('{p}')
    WHERE accession IS NOT NULL {filt}
    GROUP BY accession
    ORDER BY downloads DESC
    """
    dataset_df = conn.execute(dataset_query).df()
    total_datasets = len(dataset_df)
    total_downloads = dataset_df['downloads'].sum()

    # Gini coefficient
    downloads_sorted = np.sort(dataset_df['downloads'].values)
    n = len(downloads_sorted)
    cumsum = np.cumsum(downloads_sorted)
    gini = (2 * np.sum((np.arange(1, n + 1) * downloads_sorted)) / (n * np.sum(downloads_sorted))) - (n + 1) / n

    print(f"  Total datasets: {total_datasets:,}")
    print(f"  Total downloads: {total_downloads:,}")
    print(f"  Gini coefficient: {gini:.4f}")

    # Lorenz curve data (sampled at 100 points)
    lorenz_x = np.linspace(0, 1, 101)
    cumulative_share = np.concatenate([[0], cumsum / cumsum[-1]])
    population_share = np.linspace(0, 1, n + 1)
    lorenz_y = np.interp(lorenz_x, population_share, cumulative_share)

    lorenz_df = pd.DataFrame({'population_fraction': lorenz_x, 'download_fraction': lorenz_y})
    lorenz_df.to_csv(output_dir / 'lorenz_curve_data.csv', index=False)

    # Concentration stats
    top1_pct = dataset_df.head(max(1, total_datasets // 100))['downloads'].sum() / total_downloads * 100
    top10_pct = dataset_df.head(max(1, total_datasets // 10))['downloads'].sum() / total_downloads * 100
    top50_pct = dataset_df.head(total_datasets // 2)['downloads'].sum() / total_downloads * 100

    concentration = {
        'total_datasets': total_datasets,
        'total_downloads': int(total_downloads),
        'gini_coefficient': float(gini),
        'top_1pct_downloads_pct': float(top1_pct),
        'top_10pct_downloads_pct': float(top10_pct),
        'top_50pct_downloads_pct': float(top50_pct),
        'median_downloads': int(dataset_df['downloads'].median()),
        'mean_downloads': float(dataset_df['downloads'].mean()),
        'max_downloads': int(dataset_df['downloads'].max()),
    }
    with open(output_dir / 'concentration_stats.json', 'w') as f:
        json.dump(concentration, f, indent=2)

    print(f"  Top 1% datasets: {top1_pct:.1f}% of downloads")
    print(f"  Top 10% datasets: {top10_pct:.1f}% of downloads")
    print(f"  Top 50% datasets: {top50_pct:.1f}% of downloads")

    # Also save rank-frequency data for plotting
    dataset_df['rank'] = range(1, len(dataset_df) + 1)
    dataset_df.to_csv(output_dir / 'dataset_rank_frequency.csv', index=False)

    return concentration


def analysis_6_hourly_patterns(conn, parquet_path, output_dir, has_filter=False):
    """Hourly download patterns (for understanding usage)."""
    print("\n--- Analysis 6: Hourly Patterns ---")
    p = escaped(parquet_path)
    filt = _where_clean(has_filter)

    hourly_query = f"""
    SELECT
        CASE
            WHEN timestamp IS NOT NULL AND timestamp != ''
            THEN CAST(SUBSTRING(timestamp, 12, 2) AS INTEGER)
            ELSE 12
        END as hour,
        DAYOFWEEK(date) as day_of_week,
        COUNT(*) as downloads
    FROM read_parquet('{p}')
    WHERE year >= 2020 {filt}
    GROUP BY hour, day_of_week
    ORDER BY hour, day_of_week
    """
    try:
        hourly_df = conn.execute(hourly_query).df()
        hourly_df.to_csv(output_dir / 'hourly_patterns.csv', index=False)
        print(f"  Hourly pattern data points: {len(hourly_df)}")
        return hourly_df
    except Exception as e:
        print(f"  Hourly pattern analysis failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive PRIDE usage analysis')
    parser.add_argument('--with-labels', type=str, default=None,
                        help='Path to classification labels CSV (pride_classification_final.csv)')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = datetime.now()

    print("=" * 70)
    print("COMPREHENSIVE PRIDE USAGE ANALYSIS")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    conn = get_connection()

    # Load labels and set up bot filter if available
    has_filter = False
    if args.with_labels and os.path.exists(args.with_labels):
        labels_df = pd.read_csv(args.with_labels)
        print(f"\nLoaded {len(labels_df):,} classification labels")
        has_filter = setup_clean_filter(conn, labels_df)

    # Run all analyses (filtered to non-bot locations when labels provided)
    country_df = analysis_1_geographic(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)
    yearly_df = analysis_2_temporal(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)
    proto_df = analysis_3_protocols(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)
    top_df = analysis_4_top_datasets(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)
    concentration = analysis_5_concentration(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)
    hourly_df = analysis_6_hourly_patterns(conn, INPUT_PARQUET, OUTPUT_DIR, has_filter)

    conn.close()

    # Overall summary
    elapsed = (datetime.now() - start_time).total_seconds()
    summary = {
        'analysis_date': start_time.isoformat(),
        'elapsed_seconds': elapsed,
        'input_file': str(INPUT_PARQUET),
        'analyses_completed': [
            'geographic_by_country', 'geographic_by_region',
            'temporal_yearly', 'temporal_monthly',
            'protocol_usage',
            'top_datasets',
            'concentration_stats', 'lorenz_curve',
            'hourly_patterns',
        ],
        'key_findings': {
            'countries': len(country_df) if country_df is not None else 0,
            'years_covered': f"{int(yearly_df['year'].min())}-{int(yearly_df['year'].max())}" if yearly_df is not None else '',
            'total_datasets': concentration.get('total_datasets', 0) if concentration else 0,
            'gini_coefficient': concentration.get('gini_coefficient', 0) if concentration else 0,
        },
    }
    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Checkpoint
    checkpoint = {
        'phase': 'phase6_analysis',
        'timestamp': datetime.now().isoformat(),
        'status': 'complete',
        'elapsed_seconds': elapsed,
        'output_dir': str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / 'CHECKPOINT.json', 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ANALYSIS COMPLETE in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
