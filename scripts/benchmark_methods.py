#!/usr/bin/env python3
"""Benchmark the main classification methods in deeplogbot.

Tests three methods:
1. rules - Hierarchical rule-based classification
2. deep - Isolation Forest + behavioral features (fast mode without Transformer)
3. unsupervised - UMAP + GMM clustering

Usage:
    python scripts/benchmark_methods.py -i <input_parquet> [-n <sample_size>] [-o <output_dir>]
"""

import pandas as pd
import time
import sys
import os
import json
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')


def benchmark_method(method_name, sample_file, output_dir, **kwargs):
    """Run a single method and return results."""
    from deeplogbot import run_bot_annotator

    method_output = os.path.join(output_dir, method_name)
    os.makedirs(method_output, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Testing {method_name.upper()} method", flush=True)
    print(f"{'='*60}", flush=True)

    start = time.time()
    try:
        run_bot_annotator(
            input_parquet=sample_file,
            output_dir=method_output,
            output_strategy='reports_only',
            classification_method=method_name,
            **kwargs
        )
        elapsed = time.time() - start

        # Load results - try parquet first, then CSV
        result_file = os.path.join(method_output, 'location_analysis.parquet')
        result_file_csv = os.path.join(method_output, 'location_analysis.csv')

        if os.path.exists(result_file):
            df = pd.read_parquet(result_file)
        elif os.path.exists(result_file_csv):
            df = pd.read_csv(result_file_csv)
            n_locations = len(df)
            n_bots = (df['automation_category'] == 'bot').sum() if 'automation_category' in df.columns else 0
            n_hubs = (df['automation_category'] == 'legitimate_automation').sum() if 'automation_category' in df.columns else 0
            n_organic = n_locations - n_bots - n_hubs

            # Get subcategory breakdown
            subcats = df['subcategory'].value_counts().to_dict() if 'subcategory' in df.columns else {}

            # Calculate bot downloads
            bot_downloads = 0
            total_downloads = 0
            if 'total_downloads' in df.columns:
                total_downloads = df['total_downloads'].sum()
                if 'automation_category' in df.columns:
                    bot_downloads = df[df['automation_category'] == 'bot']['total_downloads'].sum()

            return {
                'method': method_name,
                'status': 'SUCCESS',
                'time_seconds': round(elapsed, 1),
                'locations': n_locations,
                'bots': n_bots,
                'bot_pct': round(n_bots / n_locations * 100, 1) if n_locations > 0 else 0,
                'hubs': n_hubs,
                'hub_pct': round(n_hubs / n_locations * 100, 1) if n_locations > 0 else 0,
                'organic': n_organic,
                'organic_pct': round(n_organic / n_locations * 100, 1) if n_locations > 0 else 0,
                'total_downloads': int(total_downloads),
                'bot_downloads': int(bot_downloads),
                'bot_download_pct': round(bot_downloads / total_downloads * 100, 1) if total_downloads > 0 else 0,
                'subcategories': subcats
            }
        else:
            return {
                'method': method_name,
                'status': 'FAILED',
                'error': 'No output file generated',
                'time_seconds': round(elapsed, 1)
            }
    except Exception as e:
        elapsed = time.time() - start
        import traceback
        return {
            'method': method_name,
            'status': 'FAILED',
            'error': str(e)[:500],
            'traceback': traceback.format_exc()[-500:],
            'time_seconds': round(elapsed, 1)
        }


def print_summary(results):
    """Print a summary table of all results."""
    print(f"\n{'='*80}", flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    # Header
    print(f"\n{'Method':<15} {'Status':<10} {'Time':>8} {'Locations':>12} {'Bots':>15} {'Hubs':>15} {'Bot DL%':>10}", flush=True)
    print("-" * 90, flush=True)

    for r in results:
        if r['status'] == 'SUCCESS':
            bots_str = f"{r['bots']:,} ({r['bot_pct']:.1f}%)"
            hubs_str = f"{r['hubs']:,} ({r['hub_pct']:.1f}%)"
            print(f"{r['method']:<15} {r['status']:<10} {r['time_seconds']:>6.1f}s {r['locations']:>12,} {bots_str:>15} {hubs_str:>15} {r['bot_download_pct']:>9.1f}%", flush=True)
        else:
            print(f"{r['method']:<15} {r['status']:<10} {r['time_seconds']:>6.1f}s {'N/A':>12} {'N/A':>15} {'N/A':>15} {'N/A':>10}", flush=True)
            print(f"    Error: {r.get('error', 'Unknown')[:70]}...", flush=True)

    print("-" * 90, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Benchmark deeplogbot classification methods')
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--sample-size', '-n', type=int, default=50000, help='Sample size (default: 50000)')
    parser.add_argument('--output', '-o', default='/tmp/benchmark_output', help='Output directory')
    parser.add_argument('--methods', '-m', nargs='+', default=['rules', 'deep', 'unsupervised'],
                       help='Methods to test (default: rules deep unsupervised)')
    args = parser.parse_args()

    print(f"{'='*60}", flush=True)
    print("DEEPLOGBOT CLASSIFICATION BENCHMARK", flush=True)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\nLoading data from {args.input}...", flush=True)
    df = pd.read_parquet(args.input)
    print(f"Total records: {len(df):,}", flush=True)

    # Sample data
    if len(df) > args.sample_size:
        sample_df = df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled {args.sample_size:,} records", flush=True)
    else:
        sample_df = df
        print(f"Using all {len(df):,} records", flush=True)

    # Save sample
    sample_file = os.path.join(args.output, 'sample_data.parquet')
    os.makedirs(args.output, exist_ok=True)
    sample_df.to_parquet(sample_file)
    print(f"Sample saved to {sample_file}", flush=True)

    # Run benchmarks
    results = []

    for method in args.methods:
        kwargs = {}
        # No extra kwargs needed - behavioral extraction is automatic for deep method

        result = benchmark_method(method, sample_file, args.output, **kwargs)
        results.append(result)

        # Print result immediately
        if result['status'] == 'SUCCESS':
            print(f"\n{method.upper()} completed in {result['time_seconds']:.1f}s", flush=True)
            print(f"  Locations: {result['locations']:,}", flush=True)
            print(f"  Bots: {result['bots']:,} ({result['bot_pct']:.1f}%)", flush=True)
            print(f"  Hubs: {result['hubs']:,} ({result['hub_pct']:.1f}%)", flush=True)
            print(f"  Bot downloads: {result['bot_download_pct']:.1f}% of total", flush=True)
            if result['subcategories']:
                print(f"  Top subcategories:", flush=True)
                for subcat, count in list(result['subcategories'].items())[:5]:
                    print(f"    - {subcat}: {count:,}", flush=True)
        else:
            print(f"\n{method.upper()} FAILED: {result.get('error', 'Unknown error')[:100]}", flush=True)

    # Print summary
    print_summary(results)

    # Save results to JSON
    results_file = os.path.join(args.output, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert subcategories to list for JSON serialization
        results_for_json = []
        for r in results:
            r_copy = r.copy()
            if 'subcategories' in r_copy:
                r_copy['subcategories'] = dict(r_copy['subcategories'])
            results_for_json.append(r_copy)
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'input_file': args.input,
            'sample_size': args.sample_size,
            'results': results_for_json
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}", flush=True)

    return results


if __name__ == '__main__':
    main()
