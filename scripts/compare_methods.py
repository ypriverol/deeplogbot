#!/usr/bin/env python3
"""Compare classification results from different methods.

Usage:
    python scripts/compare_methods.py output/rules/location_analysis.csv output/unsupervised/location_analysis.csv
    python scripts/compare_methods.py output/  # Compare all methods in output folder
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(csv_path: str) -> pd.DataFrame:
    """Load location analysis CSV."""
    df = pd.read_csv(csv_path)
    return df


def compute_stats(df: pd.DataFrame, method_name: str) -> dict:
    """Compute classification statistics."""
    total_locations = len(df)

    # Count classifications
    if 'automation_category' in df.columns:
        n_bots = (df['automation_category'] == 'bot').sum()
        n_hubs = (df['automation_category'] == 'legitimate_automation').sum()
        n_organic = df['automation_category'].isna().sum()
    else:
        n_bots = 0
        n_hubs = 0
        n_organic = total_locations

    # Count downloads if available
    if 'total_downloads' in df.columns:
        total_downloads = df['total_downloads'].sum()

        if 'automation_category' in df.columns:
            bot_downloads = df.loc[df['automation_category'] == 'bot', 'total_downloads'].sum()
            hub_downloads = df.loc[df['automation_category'] == 'legitimate_automation', 'total_downloads'].sum()
            organic_downloads = df.loc[df['automation_category'].isna(), 'total_downloads'].sum()
        else:
            bot_downloads = 0
            hub_downloads = 0
            organic_downloads = total_downloads
    else:
        total_downloads = 0
        bot_downloads = 0
        hub_downloads = 0
        organic_downloads = 0

    return {
        'method': method_name,
        'total_locations': total_locations,
        'bot_locations': n_bots,
        'hub_locations': n_hubs,
        'organic_locations': n_organic,
        'bot_pct': 100 * n_bots / total_locations if total_locations > 0 else 0,
        'hub_pct': 100 * n_hubs / total_locations if total_locations > 0 else 0,
        'organic_pct': 100 * n_organic / total_locations if total_locations > 0 else 0,
        'total_downloads': total_downloads,
        'bot_downloads': bot_downloads,
        'hub_downloads': hub_downloads,
        'organic_downloads': organic_downloads,
        'bot_dl_pct': 100 * bot_downloads / total_downloads if total_downloads > 0 else 0,
        'hub_dl_pct': 100 * hub_downloads / total_downloads if total_downloads > 0 else 0,
        'organic_dl_pct': 100 * organic_downloads / total_downloads if total_downloads > 0 else 0,
    }


def compute_agreement(df1: pd.DataFrame, df2: pd.DataFrame, method1: str, method2: str) -> dict:
    """Compute agreement between two methods."""
    # Ensure both have the same locations
    if 'geo_location' in df1.columns and 'geo_location' in df2.columns:
        merged = df1.merge(df2, on='geo_location', suffixes=('_1', '_2'))
    else:
        # Assume same order
        merged = pd.DataFrame({
            'automation_category_1': df1['automation_category'].values,
            'automation_category_2': df2['automation_category'].values,
        })

    # Fill NaN with 'organic' for comparison
    cat1 = merged['automation_category_1'].fillna('organic')
    cat2 = merged['automation_category_2'].fillna('organic')

    # Compute agreement
    agreement = (cat1 == cat2).sum()
    total = len(merged)

    # Compute confusion matrix
    bot_bot = ((cat1 == 'bot') & (cat2 == 'bot')).sum()
    bot_hub = ((cat1 == 'bot') & (cat2 == 'legitimate_automation')).sum()
    bot_org = ((cat1 == 'bot') & (cat2 == 'organic')).sum()
    hub_bot = ((cat1 == 'legitimate_automation') & (cat2 == 'bot')).sum()
    hub_hub = ((cat1 == 'legitimate_automation') & (cat2 == 'legitimate_automation')).sum()
    hub_org = ((cat1 == 'legitimate_automation') & (cat2 == 'organic')).sum()
    org_bot = ((cat1 == 'organic') & (cat2 == 'bot')).sum()
    org_hub = ((cat1 == 'organic') & (cat2 == 'legitimate_automation')).sum()
    org_org = ((cat1 == 'organic') & (cat2 == 'organic')).sum()

    return {
        'method_1': method1,
        'method_2': method2,
        'total': total,
        'agreement': agreement,
        'agreement_pct': 100 * agreement / total if total > 0 else 0,
        'confusion': {
            f'{method1}_bot': {'bot': bot_bot, 'hub': bot_hub, 'organic': bot_org},
            f'{method1}_hub': {'bot': hub_bot, 'hub': hub_hub, 'organic': hub_org},
            f'{method1}_organic': {'bot': org_bot, 'hub': org_hub, 'organic': org_org},
        },
    }


def format_number(n: float, is_pct: bool = False) -> str:
    """Format number for display."""
    if is_pct:
        return f"{n:.1f}%"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return f"{n:.0f}"


def print_comparison(stats_list: list, agreements: list):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<25}", end="")
    for stats in stats_list:
        print(f"{stats['method']:<20}", end="")
    print()
    print("-" * (25 + 20 * len(stats_list)))

    # Location counts
    print(f"{'Total Locations':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['total_locations']):<20}", end="")
    print()

    print(f"{'Bot Locations':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['bot_locations'])} ({stats['bot_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    print(f"{'Hub Locations':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['hub_locations'])} ({stats['hub_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    print(f"{'Organic Locations':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['organic_locations'])} ({stats['organic_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    print("-" * (25 + 20 * len(stats_list)))

    # Download counts
    print(f"{'Total Downloads':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['total_downloads']):<20}", end="")
    print()

    print(f"{'Bot Downloads':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['bot_downloads'])} ({stats['bot_dl_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    print(f"{'Hub Downloads':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['hub_downloads'])} ({stats['hub_dl_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    print(f"{'Organic Downloads':<25}", end="")
    for stats in stats_list:
        print(f"{format_number(stats['organic_downloads'])} ({stats['organic_dl_pct']:.1f}%)"[:19].ljust(20), end="")
    print()

    # Agreement
    if agreements:
        print("\n" + "=" * 80)
        print("METHOD AGREEMENT")
        print("=" * 80)

        for agreement in agreements:
            print(f"\n{agreement['method_1']} vs {agreement['method_2']}:")
            print(f"  Overall Agreement: {agreement['agreement']} / {agreement['total']} ({agreement['agreement_pct']:.1f}%)")

            print(f"\n  Confusion Matrix ({agreement['method_1']} rows, {agreement['method_2']} columns):")
            print(f"  {'':>15} {'bot':>10} {'hub':>10} {'organic':>10}")
            for row_name, row_data in agreement['confusion'].items():
                short_name = row_name.split('_')[-1]
                print(f"  {short_name:>15} {row_data['bot']:>10} {row_data['hub']:>10} {row_data['organic']:>10}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_methods.py <csv1> <csv2> ...")
        print("       python compare_methods.py <output_dir>")
        sys.exit(1)

    # Check if argument is a directory
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        output_dir = Path(sys.argv[1])
        csv_files = []
        for method_dir in ['rules', 'deep', 'unsupervised']:
            csv_path = output_dir / method_dir / 'location_analysis.csv'
            if csv_path.exists():
                csv_files.append((method_dir.upper(), str(csv_path)))

        if not csv_files:
            print(f"No location_analysis.csv files found in {output_dir}")
            sys.exit(1)
    else:
        csv_files = [(f"Method{i+1}", path) for i, path in enumerate(sys.argv[1:])]

    # Load all results
    print("Loading results...")
    results = {}
    for name, path in csv_files:
        try:
            results[name] = load_results(path)
            print(f"  Loaded {name}: {len(results[name])} locations")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    if not results:
        print("No valid results to compare.")
        sys.exit(1)

    # Compute stats
    stats_list = []
    for name, df in results.items():
        stats = compute_stats(df, name)
        stats_list.append(stats)

    # Compute pairwise agreement
    agreements = []
    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            agreement = compute_agreement(results[name1], results[name2], name1, name2)
            agreements.append(agreement)

    # Print comparison
    print_comparison(stats_list, agreements)

    # Success criteria check
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    for stats in stats_list:
        method = stats['method']
        passed = True
        issues = []

        if stats['bot_locations'] == 0:
            passed = False
            issues.append("No bots detected")

        if stats['hub_locations'] == 0:
            passed = False
            issues.append("No hubs detected")

        if stats['organic_locations'] == stats['total_locations']:
            passed = False
            issues.append("All locations classified as organic")

        if passed:
            print(f"  {method}: PASS")
        else:
            print(f"  {method}: FAIL - {', '.join(issues)}")


if __name__ == '__main__':
    main()
