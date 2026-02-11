#!/usr/bin/env python3
"""Statistical comparison of classification methods.

Loads analysis CSVs from benchmark output directories and performs
cross-method comparison: agreement rates, discovery analysis,
consistency metrics, and per-category performance deep-dives.

Usage:
    python scripts/statistical_comparison.py \
        --benchmark-dir output/phase2_benchmarking \
        --methods rules deep \
        --ground-truth output/phase2_benchmarking/ground_truth/ground_truth_full.csv \
        --output output/phase2_benchmarking/results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeplogbot.utils import logger


def load_method_results(benchmark_dir: str, methods: list) -> dict:
    """Load location_analysis.csv for each method."""
    results = {}
    for method in methods:
        analysis_file = os.path.join(benchmark_dir, f'benchmark_{method}', 'location_analysis.csv')
        if os.path.exists(analysis_file):
            df = pd.read_csv(analysis_file)
            results[method] = df
            logger.info(f"  Loaded {method}: {len(df):,} locations")
        else:
            logger.warning(f"  Analysis file not found for {method}: {analysis_file}")
    return results


def get_unified_label(df: pd.DataFrame) -> pd.Series:
    """Map hierarchical classification to unified 3-class labels."""
    labels = pd.Series('organic', index=df.index)
    if 'automation_category' in df.columns:
        labels[df['automation_category'] == 'bot'] = 'bot'
        labels[df['automation_category'] == 'legitimate_automation'] = 'hub'
    if 'behavior_type' in df.columns:
        labels[df['behavior_type'] == 'hub'] = 'hub'
    if 'is_bot' in df.columns:
        labels[df['is_bot'].fillna(False).astype(bool)] = 'bot'
    if 'is_download_hub' in df.columns:
        hub_mask = df['is_download_hub'].fillna(False).astype(bool)
        labels[hub_mask & (labels != 'bot')] = 'hub'
    return labels


def discovery_analysis(method_results: dict) -> dict:
    """
    Analyze what unique patterns each method discovers.

    Identifies locations that only one method classifies as bot/hub
    (unique discoveries) vs consensus classifications.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DISCOVERY ANALYSIS")
    logger.info("=" * 70)

    methods = list(method_results.keys())
    if len(methods) < 2:
        return {}

    # Get labels for each method
    method_labels = {}
    for method, df in method_results.items():
        method_labels[method] = get_unified_label(df)

    # Align on common index length
    min_len = min(len(v) for v in method_labels.values())
    aligned = {m: l.values[:min_len] for m, l in method_labels.items()}

    discovery = {}
    for category in ['bot', 'hub']:
        logger.info(f"\n  Category: {category.upper()}")

        # Find locations each method classifies as this category
        method_sets = {}
        for m, labels in aligned.items():
            indices = set(np.where(labels == category)[0])
            method_sets[m] = indices
            logger.info(f"    {m}: {len(indices):,} locations")

        # Consensus: classified by ALL methods
        if method_sets:
            consensus = set.intersection(*method_sets.values())
        else:
            consensus = set()
        logger.info(f"    Consensus (all agree): {len(consensus):,}")

        # Unique discoveries: only found by one method
        for m in methods:
            others = [method_sets[o] for o in methods if o != m]
            if others:
                unique = method_sets[m] - set.union(*others)
            else:
                unique = method_sets[m]
            logger.info(f"    Unique to {m}: {len(unique):,}")

        # Majority: classified by >50% of methods
        from collections import Counter
        vote_counts = Counter()
        for idx in range(min_len):
            votes = sum(1 for m in methods if aligned[m][idx] == category)
            if votes > len(methods) / 2:
                vote_counts[idx] = votes
        logger.info(f"    Majority vote: {len(vote_counts):,}")

        discovery[category] = {
            'per_method': {m: len(s) for m, s in method_sets.items()},
            'consensus': len(consensus),
            'majority_vote': len(vote_counts),
        }
        for m in methods:
            others = [method_sets[o] for o in methods if o != m]
            unique_count = len(method_sets[m] - set.union(*others)) if others else len(method_sets[m])
            discovery[category][f'unique_to_{m}'] = unique_count

    return discovery


def consistency_analysis(method_results: dict) -> dict:
    """
    Measure classification consistency/stability for each method.

    Consistency = proportion of locations where the method gives
    the same label as the majority vote across all methods.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CONSISTENCY ANALYSIS")
    logger.info("=" * 70)

    methods = list(method_results.keys())
    if len(methods) < 2:
        return {}

    method_labels = {m: get_unified_label(df).values for m, df in method_results.items()}
    min_len = min(len(v) for v in method_labels.values())
    aligned = {m: l[:min_len] for m, l in method_labels.items()}

    # Compute majority vote
    from collections import Counter
    majority_labels = []
    for idx in range(min_len):
        votes = Counter(aligned[m][idx] for m in methods)
        majority_labels.append(votes.most_common(1)[0][0])
    majority_labels = np.array(majority_labels)

    consistency = {}
    for m in methods:
        agreement_with_majority = np.mean(aligned[m] == majority_labels)
        consistency[m] = float(agreement_with_majority)
        logger.info(f"  {m}: {agreement_with_majority:.1%} agreement with majority vote")

    return consistency


def confidence_analysis(method_results: dict) -> dict:
    """Analyze classification confidence distribution per method."""
    logger.info("\n" + "=" * 70)
    logger.info("CONFIDENCE ANALYSIS")
    logger.info("=" * 70)

    confidence_stats = {}
    for method, df in method_results.items():
        conf_col = None
        for candidate in ['classification_confidence', 'confidence', 'anomaly_score']:
            if candidate in df.columns:
                conf_col = candidate
                break

        if conf_col is None:
            logger.info(f"  {method}: No confidence column found")
            continue

        values = df[conf_col].dropna()
        stats = {
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'q25': float(values.quantile(0.25)),
            'q75': float(values.quantile(0.75)),
        }
        confidence_stats[method] = stats
        logger.info(f"  {method}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
                     f"std={stats['std']:.3f}")

    return confidence_stats


def download_impact_analysis(method_results: dict) -> dict:
    """
    Analyze download volume impact of different classifications.

    For each method, compute what percentage of total downloads are
    classified as bot, hub, and organic.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD IMPACT ANALYSIS")
    logger.info("=" * 70)

    impact = {}
    for method, df in method_results.items():
        if 'total_downloads' not in df.columns:
            continue

        labels = get_unified_label(df)
        total = df['total_downloads'].sum()

        method_impact = {}
        for cat in ['bot', 'hub', 'organic']:
            mask = labels == cat
            cat_downloads = df.loc[mask, 'total_downloads'].sum()
            cat_locations = mask.sum()
            method_impact[cat] = {
                'locations': int(cat_locations),
                'locations_pct': float(100 * cat_locations / len(df)) if len(df) > 0 else 0,
                'downloads': int(cat_downloads),
                'downloads_pct': float(100 * cat_downloads / total) if total > 0 else 0,
            }

        impact[method] = method_impact
        logger.info(f"\n  {method.upper()}:")
        for cat, stats in method_impact.items():
            logger.info(f"    {cat:>8}: {stats['locations']:>6,} locs ({stats['locations_pct']:5.1f}%), "
                         f"{stats['downloads']:>12,} DL ({stats['downloads_pct']:5.1f}%)")

    return impact


def generate_comparison_report(
    discovery: dict,
    consistency: dict,
    confidence: dict,
    impact: dict,
    output_dir: str,
) -> str:
    """Generate statistical comparison report."""
    report = []
    report.append("# Statistical Comparison Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Discovery
    if discovery:
        report.append("\n## Discovery Analysis")
        for cat, data in discovery.items():
            report.append(f"\n### {cat.upper()}")
            report.append(f"- Consensus (all methods agree): {data.get('consensus', 0):,}")
            report.append(f"- Majority vote: {data.get('majority_vote', 0):,}")
            for key, value in data.get('per_method', {}).items():
                report.append(f"- {key}: {value:,} total")
            for key, value in data.items():
                if key.startswith('unique_to_'):
                    method = key.replace('unique_to_', '')
                    report.append(f"- Unique to {method}: {value:,}")

    # Consistency
    if consistency:
        report.append("\n## Consistency (Agreement with Majority Vote)")
        report.append("")
        report.append("| Method | Consistency |")
        report.append("|--------|------------|")
        for method, score in sorted(consistency.items(), key=lambda x: -x[1]):
            report.append(f"| {method.upper()} | {score:.1%} |")

    # Download impact
    if impact:
        report.append("\n## Download Impact")
        for method, cats in impact.items():
            report.append(f"\n### {method.upper()}")
            report.append("")
            report.append("| Category | Locations | Loc % | Downloads | DL % |")
            report.append("|----------|-----------|-------|-----------|------|")
            for cat in ['bot', 'hub', 'organic']:
                d = cats.get(cat, {})
                report.append(
                    f"| {cat} | {d.get('locations', 0):,} | {d.get('locations_pct', 0):.1f}% | "
                    f"{d.get('downloads', 0):,} | {d.get('downloads_pct', 0):.1f}% |"
                )

    report_text = "\n".join(report)
    report_file = os.path.join(output_dir, 'STATISTICAL_COMPARISON.md')
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"\nReport saved: {report_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Statistical comparison of classification methods'
    )
    parser.add_argument(
        '-b', '--benchmark-dir', required=True,
        help='Benchmark output directory (contains benchmark_<method>/ subdirs)'
    )
    parser.add_argument(
        '-m', '--methods', nargs='+',
        default=['rules', 'deep'],
        help='Methods to compare'
    )
    parser.add_argument(
        '-g', '--ground-truth', type=str, default=None,
        help='Path to ground truth CSV'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output directory (default: <benchmark-dir>/results)'
    )

    args = parser.parse_args()
    output_dir = args.output or os.path.join(args.benchmark_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STATISTICAL COMPARISON OF METHODS")
    logger.info("=" * 70)

    # Load results
    method_results = load_method_results(args.benchmark_dir, args.methods)

    if len(method_results) < 2:
        logger.error("Need at least 2 method results for comparison")
        sys.exit(1)

    # Run analyses
    discovery = discovery_analysis(method_results)
    consistency = consistency_analysis(method_results)
    confidence = confidence_analysis(method_results)
    impact = download_impact_analysis(method_results)

    # Generate report
    generate_comparison_report(discovery, consistency, confidence, impact, output_dir)

    # Save consolidated JSON
    all_results = {
        'created': datetime.now().isoformat(),
        'methods': list(method_results.keys()),
        'discovery': discovery,
        'consistency': consistency,
        'confidence': confidence,
        'impact': impact,
    }

    json_file = os.path.join(output_dir, 'statistical_comparison.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved: {json_file}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
