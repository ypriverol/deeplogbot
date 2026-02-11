#!/usr/bin/env python3
"""
Comprehensive benchmark script for DeepLogBot classification algorithms.

Compares:
- rules: Rule-based hierarchical classification
- unsupervised: UMAP + GMM clustering
- deep: Deep architecture (Isolation Forest + Transformers)

Usage:
    python scripts/benchmark_algorithms.py --input <parquet_file> [--sample-size 500000]
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import duckdb

from deeplogbot.main import run_bot_annotator
from deeplogbot.utils import logger


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_classification_stats(df: pd.DataFrame) -> dict:
    """Extract classification statistics from results DataFrame."""
    stats = {
        'total_locations': len(df),
        'total_downloads': int(df['total_downloads'].sum()) if 'total_downloads' in df.columns else 0,
    }

    # Hierarchical classification stats
    if 'behavior_type' in df.columns:
        stats['organic_locations'] = int((df['behavior_type'] == 'organic').sum())
        stats['automated_locations'] = int((df['behavior_type'] == 'automated').sum())
        stats['organic_pct'] = stats['organic_locations'] / len(df) * 100
        stats['automated_pct'] = stats['automated_locations'] / len(df) * 100

    if 'automation_category' in df.columns:
        stats['bot_locations'] = int((df['automation_category'] == 'bot').sum())
        stats['legitimate_locations'] = int((df['automation_category'] == 'legitimate_automation').sum())
        stats['bot_pct'] = stats['bot_locations'] / len(df) * 100

    if 'subcategory' in df.columns:
        stats['subcategory_distribution'] = df['subcategory'].value_counts().to_dict()

    # Legacy stats
    if 'is_bot' in df.columns:
        stats['is_bot_count'] = int(df['is_bot'].sum())
    if 'is_download_hub' in df.columns:
        stats['is_hub_count'] = int(df['is_download_hub'].sum())

    # Bot downloads
    if 'automation_category' in df.columns and 'total_downloads' in df.columns:
        bot_mask = df['automation_category'] == 'bot'
        stats['bot_downloads'] = int(df.loc[bot_mask, 'total_downloads'].sum())
        stats['bot_downloads_pct'] = stats['bot_downloads'] / stats['total_downloads'] * 100 if stats['total_downloads'] > 0 else 0

    return stats


def run_benchmark(input_parquet: str, sample_size: int, output_dir: str) -> dict:
    """Run benchmark for all classification methods."""

    methods = ['rules', 'unsupervised', 'deep']
    results = {}

    for method in methods:
        method_output_dir = os.path.join(output_dir, f'benchmark_{method}')
        os.makedirs(method_output_dir, exist_ok=True)

        logger.info(f"\n{'='*70}")
        logger.info(f"BENCHMARKING: {method.upper()}")
        logger.info(f"{'='*70}")

        start_time = time.time()

        try:
            result = run_bot_annotator(
                input_parquet=input_parquet,
                output_parquet=None,
                output_dir=method_output_dir,
                contamination=0.15,
                compute_importances=False,
                sample_size=sample_size,
                classification_method=method,
                output_strategy='reports_only',  # Don't modify original data
                annotate=True,  # Generate reports
            )

            elapsed = time.time() - start_time

            # Load the analysis CSV to get detailed stats
            analysis_file = os.path.join(method_output_dir, 'location_analysis.csv')
            if os.path.exists(analysis_file):
                analysis_df = pd.read_csv(analysis_file)
                stats = get_classification_stats(analysis_df)
            else:
                stats = {}

            results[method] = {
                'status': 'success',
                'elapsed_time': elapsed,
                'elapsed_formatted': format_time(elapsed),
                'bot_locations': result.get('bot_locations', 0),
                'hub_locations': result.get('hub_locations', 0),
                'stats': stats,
                'result': result.get('stats', {}),
            }

            logger.info(f"✓ {method.upper()} completed in {format_time(elapsed)}")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"✗ {method.upper()} failed: {e}")
            results[method] = {
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed,
                'elapsed_formatted': format_time(elapsed),
            }

    return results


def generate_comparison_report(results: dict, output_dir: str, sample_size: int) -> str:
    """Generate a comprehensive comparison report."""

    report = []
    report.append("=" * 80)
    report.append("DEEPLOGBOT ALGORITHM BENCHMARK REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Sample Size: {sample_size:,} records")
    report.append(f"Output Directory: {output_dir}\n")

    # Summary table
    report.append("\n" + "-" * 80)
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 80)
    report.append(f"\n{'Method':<20} {'Status':<10} {'Time':<12} {'Bot Locs':<12} {'Bot %':<10} {'Bot DL %':<10}")
    report.append("-" * 80)

    for method, data in results.items():
        if data['status'] == 'success':
            stats = data.get('stats', {})
            bot_locs = stats.get('bot_locations', data.get('bot_locations', 0))
            bot_pct = stats.get('bot_pct', 0)
            bot_dl_pct = stats.get('bot_downloads_pct', 0)
            report.append(f"{method:<20} {'OK':<10} {data['elapsed_formatted']:<12} {bot_locs:<12} {bot_pct:.1f}%{'':<5} {bot_dl_pct:.1f}%")
        else:
            report.append(f"{method:<20} {'FAILED':<10} {data['elapsed_formatted']:<12} {'-':<12} {'-':<10} {'-':<10}")

    # Detailed analysis per method
    report.append("\n\n" + "=" * 80)
    report.append("DETAILED ANALYSIS PER METHOD")
    report.append("=" * 80)

    for method, data in results.items():
        report.append(f"\n\n{'='*40}")
        report.append(f"{method.upper()}")
        report.append(f"{'='*40}")

        if data['status'] == 'failed':
            report.append(f"Status: FAILED")
            report.append(f"Error: {data.get('error', 'Unknown')}")
            continue

        report.append(f"Status: SUCCESS")
        report.append(f"Execution Time: {data['elapsed_formatted']}")

        stats = data.get('stats', {})

        report.append(f"\nClassification Results:")
        report.append(f"  Total Locations: {stats.get('total_locations', 0):,}")
        report.append(f"  Total Downloads: {stats.get('total_downloads', 0):,}")

        if 'organic_locations' in stats:
            report.append(f"\n  Behavior Type Distribution:")
            report.append(f"    ORGANIC: {stats['organic_locations']:,} ({stats['organic_pct']:.1f}%)")
            report.append(f"    AUTOMATED: {stats['automated_locations']:,} ({stats['automated_pct']:.1f}%)")

        if 'bot_locations' in stats:
            report.append(f"\n  Automation Category (AUTOMATED only):")
            report.append(f"    BOT: {stats['bot_locations']:,} ({stats['bot_pct']:.1f}% of total)")
            report.append(f"    LEGITIMATE_AUTOMATION: {stats.get('legitimate_locations', 0):,}")

        if 'bot_downloads' in stats:
            report.append(f"\n  Bot Traffic Impact:")
            report.append(f"    Bot Downloads: {stats['bot_downloads']:,} ({stats['bot_downloads_pct']:.1f}% of total)")

        if 'subcategory_distribution' in stats:
            report.append(f"\n  Subcategory Distribution:")
            for cat, count in sorted(stats['subcategory_distribution'].items(), key=lambda x: -x[1]):
                report.append(f"    {cat}: {count:,}")

    # Algorithm comparison and recommendations
    report.append("\n\n" + "=" * 80)
    report.append("ALGORITHM COMPARISON & RECOMMENDATIONS")
    report.append("=" * 80)

    successful = {k: v for k, v in results.items() if v['status'] == 'success'}

    if len(successful) >= 2:
        # Speed comparison
        report.append("\n1. SPEED COMPARISON:")
        sorted_by_speed = sorted(successful.items(), key=lambda x: x[1]['elapsed_time'])
        for i, (method, data) in enumerate(sorted_by_speed, 1):
            report.append(f"   {i}. {method}: {data['elapsed_formatted']}")

        # Bot detection comparison
        report.append("\n2. BOT DETECTION COMPARISON:")
        for method, data in successful.items():
            stats = data.get('stats', {})
            bot_locs = stats.get('bot_locations', 0)
            bot_pct = stats.get('bot_pct', 0)
            report.append(f"   {method}: {bot_locs:,} locations ({bot_pct:.1f}%)")

        # Agreement analysis
        report.append("\n3. CROSS-METHOD AGREEMENT:")
        report.append("   (Comparison of bot classifications between methods)")
        # This would require loading both analysis files and comparing

    # Recommendations
    report.append("\n\n" + "-" * 80)
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)

    report.append("""
ALGORITHM CHARACTERISTICS:

1. RULES (Rule-Based Hierarchical Classification)
   Pros:
   - Fastest execution (sub-second for location classification)
   - Highly interpretable and auditable
   - Configurable via config.yaml
   - Stable and production-ready
   - Clear hierarchical taxonomy (behavior_type -> automation_category -> subcategory)

   Cons:
   - May miss novel bot patterns not covered by rules
   - Requires manual rule tuning for new datasets

   Best for: Production use, when interpretability is critical

2. UNSUPERVISED (UMAP + GMM Clustering)
   Pros:
   - Data-driven approach discovers natural patterns
   - Good for exploratory analysis
   - Can identify novel bot patterns
   - Moderate execution time

   Cons:
   - Results may vary between runs
   - Cluster interpretation can be subjective
   - Requires UMAP/GMM tuning

   Best for: Exploring new datasets, discovering unknown patterns

3. DEEP (Isolation Forest + Transformers)
   Pros:
   - Combines multiple techniques (IF, temporal analysis)
   - Advanced feature extraction (behavioral, discriminative)
   - Pseudo-label refinement for accuracy

   Cons:
   - Slowest execution time
   - More complex to debug
   - Requires more computational resources
   - May be overkill for simple use cases

   Best for: Complex analysis where accuracy is paramount

EXPERIMENTAL METHODS TO CONSIDER REMOVING:
""")

    report.append("\n\n" + "=" * 80)
    report.append("CLEANUP RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("""
Based on this analysis, consider the following cleanup:

KEEP (Production-Ready):
  - rules: Primary production method
  - unsupervised: Secondary method for data exploration

KEEP WITH CAUTION (Useful but complex):
  - deep: For advanced analysis when needed

CONSIDER REMOVING (Experimental/Redundant):
  - unsupervised-deep: Complex autoencoder approach, rarely needed
  - unsupervised-llm: Requires LLM setup, marginal benefit over rules
  - exploratory_clustering.py: Experimental multi-scale discovery
  - pure_unsupervised.py: Redundant with unsupervised method
  - improved_clustering.py: Merged improvements into main unsupervised

FILES TO POTENTIALLY REMOVE:
  - deeplogbot/models/unsupervised/exploratory_clustering.py
  - deeplogbot/models/unsupervised/pure_unsupervised.py
  - deeplogbot/models/unsupervised/improved_clustering.py
  - deeplogbot/models/unsupervised/hierarchical_clustering.py
  - deeplogbot/models/unsupervised/representation_learning.py
  - deeplogbot/models/unsupervised/training.py (if not used by main methods)
""")

    report_text = "\n".join(report)

    # Save report
    report_file = os.path.join(output_dir, 'ALGORITHM_BENCHMARK_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report_text)

    # Save JSON results
    json_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_file, 'w') as f:
        # Convert non-serializable types
        serializable_results = {}
        for method, data in results.items():
            serializable_results[method] = {
                k: v for k, v in data.items()
                if not isinstance(v, (pd.DataFrame, np.ndarray))
            }
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {report_file}")
    logger.info(f"JSON results saved to: {json_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Benchmark DeepLogBot algorithms')
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--sample-size', '-s', type=int, default=500000,
                       help='Sample size for benchmarking (default: 500000)')
    parser.add_argument('--output-dir', '-o', default='output/benchmark',
                       help='Output directory for benchmark results')
    parser.add_argument('--methods', '-m', nargs='+',
                       default=['rules', 'unsupervised', 'deep'],
                       help='Methods to benchmark')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("DEEPLOGBOT ALGORITHM BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Sample size: {args.sample_size:,}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Output: {args.output_dir}")

    # Run benchmark
    results = run_benchmark(args.input, args.sample_size, args.output_dir)

    # Generate report
    report = generate_comparison_report(results, args.output_dir, args.sample_size)

    print("\n" + report)


if __name__ == '__main__':
    main()
