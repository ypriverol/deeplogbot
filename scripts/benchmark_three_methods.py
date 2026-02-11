#!/usr/bin/env python3
"""Benchmark script for classification methods: rules, deep, unsupervised.

Runs all selected methods on the same data, computes agreement analysis,
performance metrics against ground truth, and statistical tests.

Usage:
    python scripts/benchmark_three_methods.py \
        --input pride_data/data_downloads_parquet.parquet \
        --output output/phase2_benchmarking \
        --methods rules deep \
        --sample-size 100000 \
        --ground-truth output/phase2_benchmarking/ground_truth/ground_truth_full.csv
"""

import os
import sys
import time
import json
import argparse
import traceback
import tracemalloc
from pathlib import Path
from datetime import datetime
from itertools import combinations

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deeplogbot.main import run_bot_annotator
from deeplogbot.utils import logger


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def get_classification_from_analysis(analysis_df: pd.DataFrame) -> pd.Series:
    """
    Extract a unified 3-class label from analysis DataFrame.

    Maps the hierarchical classification to: 'bot', 'hub', 'organic'.
    """
    labels = pd.Series('organic', index=analysis_df.index)

    if 'automation_category' in analysis_df.columns:
        labels[analysis_df['automation_category'] == 'bot'] = 'bot'
        labels[analysis_df['automation_category'] == 'legitimate_automation'] = 'hub'

    # Also check behavior_type for hub classification
    if 'behavior_type' in analysis_df.columns:
        labels[analysis_df['behavior_type'] == 'hub'] = 'hub'

    # Legacy columns fallback
    if 'is_bot' in analysis_df.columns:
        bot_mask = analysis_df['is_bot'].fillna(False).astype(bool)
        labels[bot_mask] = 'bot'
    if 'is_download_hub' in analysis_df.columns:
        hub_mask = analysis_df['is_download_hub'].fillna(False).astype(bool)
        labels[hub_mask & (labels != 'bot')] = 'hub'

    return labels


def run_single_method(
    method: str,
    input_parquet: str,
    output_dir: str,
    sample_size: int = None,
) -> dict:
    """Run a single classification method and collect results."""
    method_output_dir = os.path.join(output_dir, f'benchmark_{method}')
    os.makedirs(method_output_dir, exist_ok=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"BENCHMARKING: {method.upper()}")
    logger.info(f"{'=' * 70}")

    tracemalloc.start()
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
            output_strategy='reports_only',
            annotate=True,
            provider='ebi',
        )

        elapsed = time.time() - start_time
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Load analysis CSV for detailed stats
        analysis_file = os.path.join(method_output_dir, 'location_analysis.csv')
        analysis_df = None
        if os.path.exists(analysis_file):
            analysis_df = pd.read_csv(analysis_file)

        return {
            'status': 'success',
            'elapsed_time': elapsed,
            'elapsed_formatted': format_time(elapsed),
            'peak_memory_mb': peak_memory / (1024 * 1024),
            'bot_locations': result.get('bot_locations', 0),
            'hub_locations': result.get('hub_locations', 0),
            'stats': result.get('stats', {}),
            'analysis_df': analysis_df,
            'output_dir': method_output_dir,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        tracemalloc.stop()
        logger.error(f"  {method.upper()} failed: {e}")
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed,
            'elapsed_formatted': format_time(elapsed),
            'peak_memory_mb': 0,
            'output_dir': method_output_dir,
        }


# =========================================================================
# Agreement Analysis
# =========================================================================

def compute_cohens_kappa(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute Cohen's kappa for inter-rater agreement."""
    classes = sorted(set(labels1) | set(labels2))
    n = len(labels1)
    if n == 0:
        return 0.0

    # Observed agreement
    observed = np.sum(labels1 == labels2) / n

    # Expected agreement
    expected = 0
    for c in classes:
        p1 = np.sum(labels1 == c) / n
        p2 = np.sum(labels2 == c) / n
        expected += p1 * p2

    if expected == 1.0:
        return 1.0

    return (observed - expected) / (1 - expected)


def compute_agreement_analysis(method_results: dict, output_dir: str) -> dict:
    """
    Compute pairwise agreement between all successful methods.

    Returns agreement matrix, Cohen's kappa, and disagreement cases.
    """
    logger.info("\n" + "=" * 70)
    logger.info("AGREEMENT ANALYSIS")
    logger.info("=" * 70)

    # Collect method labels aligned on geo_location
    method_labels = {}
    method_dfs = {}
    for method, result in method_results.items():
        if result['status'] != 'success' or result.get('analysis_df') is None:
            continue
        df = result['analysis_df']
        loc_col = 'geo_location' if 'geo_location' in df.columns else df.columns[0]
        labels = get_classification_from_analysis(df)
        method_labels[method] = labels
        method_dfs[method] = df
        logger.info(f"  {method}: {len(labels):,} locations classified")

    if len(method_labels) < 2:
        logger.warning("  Need at least 2 successful methods for agreement analysis")
        return {}

    # Find common locations across all methods
    methods = list(method_labels.keys())

    # Pairwise agreement
    agreement_results = {}
    all_pairs = list(combinations(methods, 2))

    for m1, m2 in all_pairs:
        labels1 = method_labels[m1].values
        labels2 = method_labels[m2].values

        # Align lengths (should be same if same input)
        min_len = min(len(labels1), len(labels2))
        l1 = labels1[:min_len]
        l2 = labels2[:min_len]

        # Overall agreement
        overall_agreement = np.mean(l1 == l2)

        # Per-category agreement
        categories = ['bot', 'hub', 'organic']
        category_agreement = {}
        for cat in categories:
            mask1 = l1 == cat
            mask2 = l2 == cat
            # Both agree it's this category
            both_agree = np.sum(mask1 & mask2)
            # Either says it's this category
            either_says = np.sum(mask1 | mask2)
            if either_says > 0:
                category_agreement[cat] = both_agree / either_says
            else:
                category_agreement[cat] = 1.0

        # Cohen's kappa
        kappa = compute_cohens_kappa(l1, l2)

        pair_key = f"{m1}_vs_{m2}"
        agreement_results[pair_key] = {
            'overall_agreement': float(overall_agreement),
            'cohens_kappa': float(kappa),
            'category_agreement': {k: float(v) for k, v in category_agreement.items()},
            'n_samples': int(min_len),
        }

        logger.info(f"\n  {m1.upper()} vs {m2.upper()}:")
        logger.info(f"    Overall agreement: {overall_agreement:.1%}")
        logger.info(f"    Cohen's kappa:     {kappa:.3f}")
        for cat, agr in category_agreement.items():
            logger.info(f"    {cat} agreement:    {agr:.1%}")

    # Save disagreement cases
    if len(methods) >= 2:
        m1, m2 = methods[0], methods[1]
        l1 = method_labels[m1].values
        l2 = method_labels[m2].values
        min_len = min(len(l1), len(l2))
        disagree_mask = l1[:min_len] != l2[:min_len]

        if disagree_mask.any():
            df1 = method_dfs[m1].iloc[:min_len]
            disagree_df = df1[disagree_mask].copy()
            disagree_df[f'{m1}_label'] = l1[:min_len][disagree_mask]
            disagree_df[f'{m2}_label'] = l2[:min_len][disagree_mask]

            # Add other method labels
            for m in methods[2:]:
                l = method_labels[m].values[:min_len]
                disagree_df[f'{m}_label'] = l[disagree_mask]

            disagree_file = os.path.join(output_dir, 'results', 'disagreement_cases.csv')
            os.makedirs(os.path.dirname(disagree_file), exist_ok=True)
            # Save top 500 by total_downloads
            if 'total_downloads' in disagree_df.columns:
                disagree_df = disagree_df.sort_values('total_downloads', ascending=False)
            disagree_df.head(500).to_csv(disagree_file, index=False)
            logger.info(f"\n  Disagreement cases saved: {disagree_file} ({disagree_mask.sum():,} total)")

    # Save agreement matrix
    agreement_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m in methods:
        agreement_matrix.loc[m, m] = 1.0
    for pair_key, pair_data in agreement_results.items():
        m1, m2 = pair_key.split('_vs_')
        agreement_matrix.loc[m1, m2] = pair_data['overall_agreement']
        agreement_matrix.loc[m2, m1] = pair_data['overall_agreement']

    matrix_file = os.path.join(output_dir, 'results', 'agreement_matrix.csv')
    os.makedirs(os.path.dirname(matrix_file), exist_ok=True)
    agreement_matrix.to_csv(matrix_file)
    logger.info(f"  Agreement matrix saved: {matrix_file}")

    return agreement_results


# =========================================================================
# Performance Metrics (vs Ground Truth)
# =========================================================================

def compute_performance_metrics(
    method_results: dict,
    ground_truth_path: str,
    output_dir: str,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute precision, recall, F1 per category for each method vs ground truth.

    Also computes bootstrap confidence intervals.
    """
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE METRICS (vs Ground Truth)")
    logger.info("=" * 70)

    if not os.path.exists(ground_truth_path):
        logger.warning(f"  Ground truth file not found: {ground_truth_path}")
        return {}

    gt_df = pd.read_csv(ground_truth_path)
    if 'ground_truth_label' not in gt_df.columns:
        logger.warning("  Ground truth file missing 'ground_truth_label' column")
        return {}

    # Only use labeled samples (exclude 'uncertain')
    gt_labeled = gt_df[gt_df['ground_truth_label'] != 'uncertain'].copy()
    gt_labels = gt_labeled['ground_truth_label'].values
    gt_locations = gt_labeled['geo_location'].values if 'geo_location' in gt_labeled.columns else None

    logger.info(f"  Ground truth: {len(gt_labeled):,} labeled samples")
    logger.info(f"  Distribution: {pd.Series(gt_labels).value_counts().to_dict()}")

    categories = ['bot', 'hub', 'organic']
    all_metrics = {}

    for method, result in method_results.items():
        if result['status'] != 'success' or result.get('analysis_df') is None:
            continue

        df = result['analysis_df']
        pred_labels = get_classification_from_analysis(df)

        # Align predictions with ground truth by location
        if gt_locations is not None and 'geo_location' in df.columns:
            # Match by geo_location
            pred_map = dict(zip(df['geo_location'], pred_labels))
            matched_preds = np.array([pred_map.get(loc, 'organic') for loc in gt_locations])
        else:
            # Assume same order (fallback)
            min_len = min(len(pred_labels), len(gt_labels))
            matched_preds = pred_labels.values[:min_len]
            gt_labels_used = gt_labels[:min_len]

        gt_labels_used = gt_labels

        # Compute metrics per category
        method_metrics = {}
        for cat in categories:
            tp = np.sum((matched_preds == cat) & (gt_labels_used == cat))
            fp = np.sum((matched_preds == cat) & (gt_labels_used != cat))
            fn = np.sum((matched_preds != cat) & (gt_labels_used == cat))
            tn = np.sum((matched_preds != cat) & (gt_labels_used != cat))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(gt_labels_used) if len(gt_labels_used) > 0 else 0

            method_metrics[cat] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            }

        # Overall metrics (macro average)
        macro_precision = np.mean([method_metrics[c]['precision'] for c in categories])
        macro_recall = np.mean([method_metrics[c]['recall'] for c in categories])
        macro_f1 = np.mean([method_metrics[c]['f1'] for c in categories])
        overall_accuracy = np.mean(matched_preds == gt_labels_used)

        method_metrics['macro_avg'] = {
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1': float(macro_f1),
            'accuracy': float(overall_accuracy),
        }

        # Bootstrap confidence intervals for macro F1
        if n_bootstrap > 0:
            rng = np.random.RandomState(42)
            bootstrap_f1s = []
            n = len(gt_labels_used)
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot_pred = matched_preds[idx]
                boot_gt = gt_labels_used[idx]
                cat_f1s = []
                for cat in categories:
                    tp = np.sum((boot_pred == cat) & (boot_gt == cat))
                    fp = np.sum((boot_pred == cat) & (boot_gt != cat))
                    fn = np.sum((boot_pred != cat) & (boot_gt == cat))
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    cat_f1s.append(f)
                bootstrap_f1s.append(np.mean(cat_f1s))

            method_metrics['macro_f1_ci'] = {
                'lower': float(np.percentile(bootstrap_f1s, 2.5)),
                'upper': float(np.percentile(bootstrap_f1s, 97.5)),
                'mean': float(np.mean(bootstrap_f1s)),
                'std': float(np.std(bootstrap_f1s)),
            }

        all_metrics[method] = method_metrics

        logger.info(f"\n  {method.upper()}:")
        logger.info(f"    Overall accuracy: {overall_accuracy:.1%}")
        for cat in categories:
            m = method_metrics[cat]
            logger.info(f"    {cat:>8}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        logger.info(f"    {'macro':>8}: P={macro_precision:.3f}  R={macro_recall:.3f}  F1={macro_f1:.3f}")
        if 'macro_f1_ci' in method_metrics:
            ci = method_metrics['macro_f1_ci']
            logger.info(f"    F1 95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

    # Save confusion matrices
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    metrics_file = os.path.join(results_dir, 'performance_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n  Metrics saved: {metrics_file}")

    return all_metrics


# =========================================================================
# Statistical Tests
# =========================================================================

def compute_statistical_tests(method_results: dict, ground_truth_path: str, output_dir: str) -> dict:
    """
    Run McNemar's test for pairwise method comparison.

    McNemar's test compares whether two classifiers have the same error rate
    on the same dataset.
    """
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTS")
    logger.info("=" * 70)

    if not os.path.exists(ground_truth_path):
        logger.warning("  No ground truth for statistical tests")
        return {}

    gt_df = pd.read_csv(ground_truth_path)
    gt_labeled = gt_df[gt_df['ground_truth_label'] != 'uncertain']
    gt_labels = gt_labeled['ground_truth_label'].values
    gt_locations = gt_labeled['geo_location'].values if 'geo_location' in gt_labeled.columns else None

    # Get predictions aligned to ground truth
    method_preds = {}
    for method, result in method_results.items():
        if result['status'] != 'success' or result.get('analysis_df') is None:
            continue
        df = result['analysis_df']
        pred_labels = get_classification_from_analysis(df)

        if gt_locations is not None and 'geo_location' in df.columns:
            pred_map = dict(zip(df['geo_location'], pred_labels))
            matched = np.array([pred_map.get(loc, 'organic') for loc in gt_locations])
        else:
            min_len = min(len(pred_labels), len(gt_labels))
            matched = pred_labels.values[:min_len]

        method_preds[method] = matched

    if len(method_preds) < 2:
        logger.warning("  Need at least 2 methods for statistical tests")
        return {}

    # McNemar's test for each pair
    test_results = {}
    for m1, m2 in combinations(method_preds.keys(), 2):
        preds1 = method_preds[m1]
        preds2 = method_preds[m2]
        min_len = min(len(preds1), len(preds2), len(gt_labels))

        p1 = preds1[:min_len]
        p2 = preds2[:min_len]
        gt = gt_labels[:min_len]

        correct1 = (p1 == gt)
        correct2 = (p2 == gt)

        # McNemar contingency table
        # b: m1 correct, m2 wrong
        # c: m1 wrong, m2 correct
        b = np.sum(correct1 & ~correct2)
        c = np.sum(~correct1 & correct2)

        # McNemar's chi-squared statistic (with continuity correction)
        if (b + c) > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # Approximate p-value from chi-squared distribution (df=1)
            from math import erfc, sqrt
            p_value = erfc(sqrt(chi2 / 2))
        else:
            chi2 = 0.0
            p_value = 1.0

        significant = p_value < 0.05
        pair_key = f"{m1}_vs_{m2}"
        test_results[pair_key] = {
            'mcnemar_chi2': float(chi2),
            'p_value': float(p_value),
            'significant': bool(significant),
            'm1_only_correct': int(b),
            'm2_only_correct': int(c),
            'both_correct': int(np.sum(correct1 & correct2)),
            'both_wrong': int(np.sum(~correct1 & ~correct2)),
        }

        sig_str = " *" if significant else ""
        logger.info(f"\n  {m1.upper()} vs {m2.upper()}:")
        logger.info(f"    McNemar chi2={chi2:.3f}, p={p_value:.4f}{sig_str}")
        logger.info(f"    {m1} only correct: {b}, {m2} only correct: {c}")

    # Save results
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    tests_file = os.path.join(results_dir, 'statistical_tests.json')
    with open(tests_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"\n  Statistical tests saved: {tests_file}")

    return test_results


# =========================================================================
# Summary Report
# =========================================================================

def generate_benchmark_report(
    method_results: dict,
    agreement_results: dict,
    performance_metrics: dict,
    statistical_tests: dict,
    output_dir: str,
    sample_size: int = None,
) -> str:
    """Generate comprehensive benchmark report in Markdown."""
    report = []
    report.append("# DeepLogBot Algorithm Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if sample_size:
        report.append(f"Sample Size: {sample_size:,} records")
    report.append("")

    # Summary table
    report.append("## Performance Summary")
    report.append("")
    report.append("| Method | Status | Time | Memory (MB) | Bot Locs | Hub Locs | Macro F1 |")
    report.append("|--------|--------|------|-------------|----------|----------|----------|")

    for method, result in method_results.items():
        if result['status'] == 'success':
            mem = f"{result.get('peak_memory_mb', 0):.0f}"
            bot_locs = f"{result.get('bot_locations', 0):,}"
            hub_locs = f"{result.get('hub_locations', 0):,}"
            f1 = performance_metrics.get(method, {}).get('macro_avg', {}).get('f1', 'N/A')
            f1_str = f"{f1:.3f}" if isinstance(f1, float) else f1
            report.append(f"| {method.upper()} | OK | {result['elapsed_formatted']} | {mem} | {bot_locs} | {hub_locs} | {f1_str} |")
        else:
            report.append(f"| {method.upper()} | FAILED | {result.get('elapsed_formatted', 'N/A')} | - | - | - | - |")

    # Agreement analysis
    if agreement_results:
        report.append("\n## Inter-Method Agreement")
        report.append("")
        report.append("| Pair | Overall | Kappa | Bot Agr | Hub Agr | Organic Agr |")
        report.append("|------|---------|-------|---------|---------|-------------|")
        for pair, data in agreement_results.items():
            m1, m2 = pair.split('_vs_')
            cat_agr = data.get('category_agreement', {})
            report.append(
                f"| {m1.upper()} vs {m2.upper()} | "
                f"{data['overall_agreement']:.1%} | "
                f"{data['cohens_kappa']:.3f} | "
                f"{cat_agr.get('bot', 0):.1%} | "
                f"{cat_agr.get('hub', 0):.1%} | "
                f"{cat_agr.get('organic', 0):.1%} |"
            )

    # Performance metrics
    if performance_metrics:
        report.append("\n## Performance vs Ground Truth")
        report.append("")
        for method, metrics in performance_metrics.items():
            report.append(f"\n### {method.upper()}")
            report.append("")
            report.append("| Category | Precision | Recall | F1 | TP | FP | FN |")
            report.append("|----------|-----------|--------|-----|----|----|-----|")
            for cat in ['bot', 'hub', 'organic']:
                m = metrics.get(cat, {})
                report.append(
                    f"| {cat} | {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
                    f"{m.get('f1', 0):.3f} | {m.get('tp', 0)} | {m.get('fp', 0)} | {m.get('fn', 0)} |"
                )
            macro = metrics.get('macro_avg', {})
            report.append(
                f"| **macro** | **{macro.get('precision', 0):.3f}** | **{macro.get('recall', 0):.3f}** | "
                f"**{macro.get('f1', 0):.3f}** | - | - | - |"
            )
            if 'macro_f1_ci' in metrics:
                ci = metrics['macro_f1_ci']
                report.append(f"\nF1 95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")

    # Statistical tests
    if statistical_tests:
        report.append("\n## Statistical Tests (McNemar)")
        report.append("")
        report.append("| Pair | Chi2 | p-value | Significant |")
        report.append("|------|------|---------|-------------|")
        for pair, data in statistical_tests.items():
            m1, m2 = pair.split('_vs_')
            sig = "Yes" if data['significant'] else "No"
            report.append(
                f"| {m1.upper()} vs {m2.upper()} | {data['mcnemar_chi2']:.3f} | "
                f"{data['p_value']:.4f} | {sig} |"
            )

    # Recommendation
    report.append("\n## Recommendation")
    report.append("")
    if performance_metrics:
        best_method = max(
            performance_metrics.keys(),
            key=lambda m: performance_metrics[m].get('macro_avg', {}).get('f1', 0)
        )
        best_f1 = performance_metrics[best_method].get('macro_avg', {}).get('f1', 0)
        report.append(f"Best method by macro F1: **{best_method.upper()}** (F1={best_f1:.3f})")
    else:
        report.append("No ground truth available for definitive ranking.")

    report_text = "\n".join(report)

    # Save report
    report_file = os.path.join(output_dir, 'BENCHMARK_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"\nReport saved: {report_file}")

    return report_text


# =========================================================================
# Main benchmark orchestrator
# =========================================================================

def benchmark_methods(
    input_parquet: str,
    output_dir: str = 'output/phase2_benchmarking',
    methods: list = None,
    sample_size: int = None,
    ground_truth_path: str = None,
    n_bootstrap: int = 1000,
):
    """Run full benchmark pipeline."""
    if methods is None:
        methods = ['rules', 'deep']

    if not os.path.exists(input_parquet):
        logger.error(f"Input file not found: {input_parquet}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

    logger.info("=" * 80)
    logger.info("DEEPLOGBOT BENCHMARK SUITE")
    logger.info("=" * 80)
    logger.info(f"Input: {input_parquet}")
    logger.info(f"Methods: {', '.join(methods)}")
    logger.info(f"Output: {output_dir}")
    if sample_size:
        logger.info(f"Sample size: {sample_size:,}")
    if ground_truth_path:
        logger.info(f"Ground truth: {ground_truth_path}")
    logger.info("=" * 80)

    # Step 1: Run all methods
    method_results = {}
    for method in methods:
        method_results[method] = run_single_method(
            method, input_parquet, output_dir, sample_size
        )

    # Log quick summary
    successful = [m for m, r in method_results.items() if r['status'] == 'success']
    logger.info(f"\n  Successfully completed: {', '.join(successful)}")

    # Step 2: Agreement analysis
    agreement_results = compute_agreement_analysis(method_results, output_dir)

    # Step 3: Performance metrics (if ground truth available)
    performance_metrics = {}
    if ground_truth_path and os.path.exists(ground_truth_path):
        performance_metrics = compute_performance_metrics(
            method_results, ground_truth_path, output_dir, n_bootstrap
        )

    # Step 4: Statistical tests
    statistical_tests = {}
    if ground_truth_path and os.path.exists(ground_truth_path):
        statistical_tests = compute_statistical_tests(
            method_results, ground_truth_path, output_dir
        )

    # Step 5: Generate report
    report = generate_benchmark_report(
        method_results, agreement_results, performance_metrics,
        statistical_tests, output_dir, sample_size
    )

    # Save consolidated results JSON
    consolidated = {
        'created': datetime.now().isoformat(),
        'methods': methods,
        'sample_size': sample_size,
        'method_summary': {
            method: {
                'status': r['status'],
                'elapsed_time': r.get('elapsed_time', 0),
                'peak_memory_mb': r.get('peak_memory_mb', 0),
                'bot_locations': r.get('bot_locations', 0),
                'hub_locations': r.get('hub_locations', 0),
            }
            for method, r in method_results.items()
        },
        'agreement': agreement_results,
        'performance': {
            method: {k: v for k, v in metrics.items() if k != 'analysis_df'}
            for method, metrics in performance_metrics.items()
        },
        'statistical_tests': statistical_tests,
    }

    results_file = os.path.join(output_dir, 'results', 'benchmark_summary.json')
    with open(results_file, 'w') as f:
        json.dump(consolidated, f, indent=2, default=str)
    logger.info(f"Consolidated results: {results_file}")

    # Method comparison CSV
    comparison_data = []
    for method, result in method_results.items():
        if result['status'] != 'success':
            continue
        row = {
            'method': method,
            'time_seconds': result.get('elapsed_time', 0),
            'memory_mb': result.get('peak_memory_mb', 0),
            'bot_locations': result.get('bot_locations', 0),
            'hub_locations': result.get('hub_locations', 0),
        }
        if method in performance_metrics:
            pm = performance_metrics[method]
            for cat in ['bot', 'hub', 'organic']:
                row[f'{cat}_precision'] = pm.get(cat, {}).get('precision', 0)
                row[f'{cat}_recall'] = pm.get(cat, {}).get('recall', 0)
                row[f'{cat}_f1'] = pm.get(cat, {}).get('f1', 0)
            row['macro_f1'] = pm.get('macro_avg', {}).get('f1', 0)
        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(output_dir, 'results', 'method_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Method comparison: {comparison_file}")

    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)

    return method_results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark classification methods with agreement analysis and statistical tests'
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input parquet file path'
    )
    parser.add_argument(
        '-o', '--output-dir', default='output/phase2_benchmarking',
        help='Output directory (default: output/phase2_benchmarking)'
    )
    parser.add_argument(
        '-m', '--methods', nargs='+',
        choices=['rules', 'deep', 'deep_v2', 'unsupervised'],
        default=['rules', 'deep'],
        help='Methods to benchmark (default: rules deep)'
    )
    parser.add_argument(
        '-s', '--sample-size', type=int, default=None,
        help='Sample size for faster testing (optional)'
    )
    parser.add_argument(
        '-g', '--ground-truth', type=str, default=None,
        help='Path to ground truth CSV (from create_ground_truth.py)'
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=1000,
        help='Number of bootstrap iterations for confidence intervals (default: 1000)'
    )

    args = parser.parse_args()

    benchmark_methods(
        input_parquet=args.input,
        output_dir=args.output_dir,
        methods=args.methods,
        sample_size=args.sample_size,
        ground_truth_path=args.ground_truth,
        n_bootstrap=args.n_bootstrap,
    )


if __name__ == '__main__':
    main()
