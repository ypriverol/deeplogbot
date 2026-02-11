#!/usr/bin/env python3
"""Benchmark rules vs deep classification on the same sample.

Uses DuckDB RESERVOIR sampling to avoid loading the full parquet into RAM.
Creates a single sampled parquet file, then runs both methods on it.

Usage:
    python scripts/benchmark_rules_vs_deep.py \
        -i pride_data/data_downloads_parquet.parquet \
        -n 10000000 \
        -o /tmp/benchmark_rules_vs_deep
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import pandas as pd

warnings.filterwarnings("ignore")


def create_sample(input_parquet: str, sample_size: int, output_dir: str) -> str:
    """Create a sampled parquet file using DuckDB BERNOULLI sampling.

    Uses percentage-based BERNOULLI sampling (streaming, no memory overhead)
    instead of RESERVOIR (which must hold N rows in memory and is very slow
    on large files).
    """
    sample_file = os.path.join(output_dir, "sample_data.parquet")
    if os.path.exists(sample_file):
        conn = duckdb.connect()
        count = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sample_file}')"
        ).fetchone()[0]
        conn.close()
        if count > 0:
            print(f"  Reusing existing sample: {sample_file} ({count:,} records)")
            return sample_file

    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='4GB'")
    conn.execute("PRAGMA threads=2")

    escaped = os.path.abspath(input_parquet).replace("'", "''")
    total = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{escaped}')"
    ).fetchone()[0]
    print(f"  Total records in source: {total:,}")

    if sample_size >= total:
        conn.close()
        print(f"  Sample size >= total, using original file")
        return input_parquet

    # Use BERNOULLI (streaming) with a slightly higher percentage to ensure
    # we get enough rows, then LIMIT to exact count
    pct = min(100.0, (sample_size / total) * 110)  # 10% oversampling margin
    escaped_out = os.path.abspath(sample_file).replace("'", "''")
    print(f"  Sampling ~{sample_size:,} records via BERNOULLI ({pct:.2f}%) + LIMIT ...")
    conn.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{escaped}')
            TABLESAMPLE BERNOULLI ({pct:.4f} PERCENT)
            LIMIT {sample_size}
        ) TO '{escaped_out}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
    """)

    actual = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{escaped_out}')"
    ).fetchone()[0]
    print(f"  Sampled {actual:,} records -> {sample_file}")

    # Show year distribution
    years = conn.execute(f"""
        SELECT year, COUNT(*) as n
        FROM read_parquet('{escaped_out}')
        GROUP BY year ORDER BY year
    """).df()
    print(f"  Year distribution:")
    for _, row in years.iterrows():
        print(f"    {int(row['year'])}: {int(row['n']):>12,}")

    conn.close()
    return sample_file


def run_method(method: str, sample_file: str, output_dir: str) -> dict:
    """Run a classification method and return metrics."""
    from deeplogbot import run_bot_annotator

    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {method.upper()}")
    print(f"{'=' * 70}\n")

    start = time.time()
    try:
        run_bot_annotator(
            input_parquet=sample_file,
            output_dir=method_dir,
            output_strategy="reports_only",
            classification_method=method,
        )
        elapsed = time.time() - start

        # Read results
        csv_path = os.path.join(method_dir, "location_analysis.csv")
        if not os.path.exists(csv_path):
            return {"method": method, "status": "FAILED", "error": "No output CSV", "time_s": elapsed}

        df = pd.read_csv(csv_path)
        n = len(df)

        # Count by automation_category
        n_bots = (df["automation_category"] == "bot").sum() if "automation_category" in df.columns else 0
        n_hubs = (df["automation_category"] == "legitimate_automation").sum() if "automation_category" in df.columns else 0
        n_organic = n - n_bots - n_hubs

        total_dl = df["total_downloads"].sum() if "total_downloads" in df.columns else 0
        bot_dl = df.loc[df["automation_category"] == "bot", "total_downloads"].sum() if "automation_category" in df.columns and "total_downloads" in df.columns else 0
        hub_dl = df.loc[df["automation_category"] == "legitimate_automation", "total_downloads"].sum() if "automation_category" in df.columns and "total_downloads" in df.columns else 0

        subcats = df["subcategory"].value_counts().to_dict() if "subcategory" in df.columns else {}

        return {
            "method": method,
            "status": "SUCCESS",
            "time_s": round(elapsed, 1),
            "locations": n,
            "bots": int(n_bots),
            "bot_pct": round(n_bots / n * 100, 1) if n else 0,
            "hubs": int(n_hubs),
            "hub_pct": round(n_hubs / n * 100, 1) if n else 0,
            "organic": int(n_organic),
            "organic_pct": round(n_organic / n * 100, 1) if n else 0,
            "total_downloads": int(total_dl),
            "bot_downloads": int(bot_dl),
            "bot_dl_pct": round(bot_dl / total_dl * 100, 1) if total_dl else 0,
            "hub_downloads": int(hub_dl),
            "hub_dl_pct": round(hub_dl / total_dl * 100, 1) if total_dl else 0,
            "subcategories": {k: int(v) for k, v in list(subcats.items())[:10]},
        }
    except Exception as e:
        import traceback
        elapsed = time.time() - start
        return {
            "method": method,
            "status": "FAILED",
            "error": str(e)[:500],
            "traceback": traceback.format_exc()[-500:],
            "time_s": round(elapsed, 1),
        }


def print_comparison(results: list):
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 90}")
    print("BENCHMARK COMPARISON: rules vs deep")
    print(f"{'=' * 90}")

    header = f"{'Metric':<30} "
    for r in results:
        header += f"{'  ' + r['method'].upper():>28}"
    print(header)
    print("-" * 90)

    def row(label, key, fmt="{:>25,}"):
        line = f"{label:<30} "
        for r in results:
            if r["status"] != "SUCCESS":
                line += f"{'FAILED':>28}"
            else:
                val = r.get(key, "N/A")
                if isinstance(val, float):
                    line += f"{val:>27.1f}%"
                elif isinstance(val, int):
                    line += f"{val:>28,}"
                else:
                    line += f"{str(val):>28}"
        print(line)

    row("Time (seconds)", "time_s")
    row("Locations", "locations")
    print("-" * 90)
    row("Bot locations", "bots")
    row("Bot locations %", "bot_pct")
    row("Hub locations", "hubs")
    row("Hub locations %", "hub_pct")
    row("Organic locations", "organic")
    row("Organic locations %", "organic_pct")
    print("-" * 90)
    row("Total downloads", "total_downloads")
    row("Bot downloads", "bot_downloads")
    row("Bot download %", "bot_dl_pct")
    row("Hub downloads", "hub_downloads")
    row("Hub download %", "hub_dl_pct")
    print("-" * 90)

    # Subcategory breakdown
    for r in results:
        if r["status"] == "SUCCESS" and r.get("subcategories"):
            print(f"\n  {r['method'].upper()} subcategories:")
            for subcat, count in sorted(r["subcategories"].items(), key=lambda x: -x[1]):
                pct = count / r["locations"] * 100 if r["locations"] else 0
                print(f"    {subcat:<30} {count:>8,}  ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rules vs deep classification")
    parser.add_argument("--input", "-i", required=True, help="Input parquet file")
    parser.add_argument("--sample-size", "-n", type=int, default=10_000_000, help="Sample size (default: 10M)")
    parser.add_argument("--output", "-o", default="/tmp/benchmark_rules_vs_deep", help="Output directory")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"DEEPLOGBOT: Rules vs Deep Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample size: {args.sample_size:,} records")
    print(f"{'=' * 70}")

    os.makedirs(args.output, exist_ok=True)

    # Step 1: Create a single sample for both methods
    print("\nStep 1: Creating sample ...")
    sample_file = create_sample(args.input, args.sample_size, args.output)

    # Step 2: Run both methods on the same sample
    results = []
    for method in ["rules", "deep"]:
        result = run_method(method, sample_file, args.output)
        results.append(result)

        if result["status"] == "SUCCESS":
            print(f"\n  {method.upper()} done in {result['time_s']:.1f}s: "
                  f"{result['bots']:,} bots ({result['bot_pct']:.1f}%), "
                  f"{result['hubs']:,} hubs ({result['hub_pct']:.1f}%), "
                  f"bot DL={result['bot_dl_pct']:.1f}%")
        else:
            print(f"\n  {method.upper()} FAILED: {result.get('error', 'unknown')[:200]}")

    # Step 3: Print comparison
    print_comparison(results)

    # Step 4: Save results
    results_file = os.path.join(args.output, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "input_file": args.input,
            "sample_size": args.sample_size,
            "sample_file": sample_file,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
