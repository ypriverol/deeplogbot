#!/usr/bin/env python3
"""Generate supplementary figures for the PRIDE manuscript.

All data-dependent figures are regenerated from bot-filtered data.

Usage:
    python scripts/generate_supp_figures.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from pathlib import Path
import duckdb

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

project_root = Path(__file__).parent.parent
ANALYSIS_DIR = project_root / 'output' / 'phase6_analysis'
BENCHMARK_DIR = project_root / 'output' / 'phase4_final_benchmark'
CLASSIFICATION_DIR = project_root / 'output' / 'phase5_full_classification'
FIGURES_DIR = project_root / 'paper' / 'figures'
PARQUET_PATH = project_root / 'pride_data' / 'data_downloads_parquet.parquet'

COLORS = {
    'bot': '#E74C3C',
    'hub': '#3498DB',
    'organic': '#2ECC71',
    'rules': '#E67E22',
    'deep': '#9B59B6',
}

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
    'Austria': 'Europe', 'Poland': 'Europe', 'Czechia': 'Europe',
    'Portugal': 'Europe', 'Ireland': 'Europe', 'Greece': 'Europe',
    'Hungary': 'Europe', 'Romania': 'Europe', 'Russia': 'Europe',
    'Turkey': 'Europe', 'Ukraine': 'Europe',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Nigeria': 'Africa',
    'Kenya': 'Africa', 'Morocco': 'Africa',
    'Saudi Arabia': 'Middle East', 'Iran': 'Middle East',
}

EUROPEAN_COUNTRIES = [
    'Germany', 'United Kingdom', 'France', 'Spain', 'Italy', 'Netherlands',
    'Switzerland', 'Sweden', 'Denmark', 'Belgium', 'Finland', 'Norway',
    'Austria', 'Poland', 'Ireland',
]


def get_filtered_connection():
    """Get DuckDB connection with non-bot location filter set up."""
    labels_path = CLASSIFICATION_DIR / 'pride_classification_final.csv'
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    if 'final_label' in labels_df.columns:
        non_bot = labels_df[labels_df['final_label'] != 'bot'][['geo_location']].drop_duplicates()
    elif 'is_bot' in labels_df.columns:
        non_bot = labels_df[~labels_df['is_bot']][['geo_location']].drop_duplicates()
    else:
        raise KeyError("No bot label column found (expected 'final_label' or 'is_bot')")

    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='4GB'")
    tmp = os.path.abspath('./duckdb-tmp/')
    os.makedirs(tmp, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{tmp}'")
    conn.execute("PRAGMA threads=2")
    conn.register('_cl', non_bot)
    conn.execute("CREATE TEMP TABLE clean_locations AS SELECT * FROM _cl")
    return conn


def P():
    return str(PARQUET_PATH).replace("'", "''")


def FILT():
    return "AND geo_location IN (SELECT geo_location FROM clean_locations)"


# ====================================================================
# Bot removal figures (before/after)
# ====================================================================

def supp_fig_bot_removal_geographic(conn, output_dir):
    """Before/after bot removal geographic comparison."""
    print("  Bot removal geographic...")
    p = P()

    raw_df = conn.execute(f"""
        SELECT country, COUNT(*) as total_downloads
        FROM read_parquet('{p}')
        WHERE country IS NOT NULL AND country != '' AND country NOT LIKE '%{{%'
        GROUP BY country ORDER BY total_downloads DESC LIMIT 15
    """).df()

    clean_df = conn.execute(f"""
        SELECT country, COUNT(*) as total_downloads
        FROM read_parquet('{p}')
        WHERE country IS NOT NULL AND country != '' AND country NOT LIKE '%{{%' {FILT()}
        GROUP BY country ORDER BY total_downloads DESC LIMIT 15
    """).df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    y1 = range(len(raw_df))
    bars1 = ax1.barh(y1, raw_df['total_downloads'] / 1e6, color='#95A5A6', edgecolor='white')
    ax1.set_yticks(y1); ax1.set_yticklabels(raw_df['country']); ax1.invert_yaxis()
    ax1.set_xlabel('Downloads (millions)'); ax1.set_title('A) Raw Downloads (Before Bot Removal)')
    for bar, val in zip(bars1, raw_df['total_downloads']):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.1f}M', va='center', fontsize=7)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    y2 = range(len(clean_df))
    bars2 = ax2.barh(y2, clean_df['total_downloads'] / 1e6, color=COLORS['organic'], edgecolor='white')
    ax2.set_yticks(y2); ax2.set_yticklabels(clean_df['country']); ax2.invert_yaxis()
    ax2.set_xlabel('Downloads (millions)'); ax2.set_title('B) After Bot Removal')
    for bar, val in zip(bars2, clean_df['total_downloads']):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.2f}M', va='center', fontsize=7)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'supp_bot_removal_geographic.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_classification_distribution(output_dir):
    """Classification distribution: pie + bar."""
    print("  Classification distribution...")
    summary_path = CLASSIFICATION_DIR / 'classification_summary.json'
    if not summary_path.exists():
        print("    SKIPPED"); return
    with open(summary_path) as f:
        stats = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    loc_vals = [stats['organic_locations'], stats['hub_locations'], stats['bot_locations']]
    loc_labels = [f"Organic\n({stats['organic_locations']:,})",
                  f"Hub\n({stats['hub_locations']:,})",
                  f"Bot\n({stats['bot_locations']:,})"]
    colors = [COLORS['organic'], COLORS['hub'], COLORS['bot']]
    wedges, texts, autotexts = ax1.pie(loc_vals, labels=loc_labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90, pctdistance=0.75)
    for t in autotexts:
        t.set_fontsize(10); t.set_fontweight('bold')
    ax1.set_title('A) Locations by Classification')

    dl_cats = ['Organic', 'Hub', 'Bot']
    dl_vals = [stats['organic_dl_pct'], stats['hub_dl_pct'], stats['bot_dl_pct']]
    bars = ax2.bar(dl_cats, dl_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Percentage of Total Downloads'); ax2.set_title('B) Download Share by Classification')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, dl_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'supp_classification_distribution.pdf', format='pdf')
    plt.close()
    print("    OK")


# ====================================================================
# Benchmark figures
# ====================================================================

def supp_fig_benchmark_details(output_dir):
    """Per-category P/R/F1 heatmaps."""
    print("  Benchmark heatmaps...")
    metrics_path = BENCHMARK_DIR / 'results' / 'performance_metrics.json'
    if not metrics_path.exists():
        print("    SKIPPED"); return
    with open(metrics_path) as f:
        metrics = json.load(f)

    methods = ['rules', 'deep']
    categories = ['bot', 'hub', 'organic']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for idx, method in enumerate(methods):
        ax = axes[idx]; m = metrics[method]
        data = np.array([[m[cat]['precision'], m[cat]['recall'], m[cat]['f1']] for cat in categories])
        im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(3)); ax.set_xticklabels(['Precision', 'Recall', 'F1'])
        ax.set_yticks(range(3)); ax.set_yticklabels([c.capitalize() for c in categories])
        ax.set_title(f'{method.capitalize()} Method')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', fontsize=10,
                       fontweight='bold', color='white' if data[i, j] < 0.4 else 'black')
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label='Score')
    plt.suptitle('Per-Category Performance Metrics by Method', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_benchmark_heatmaps.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_bootstrap_ci(output_dir):
    """Bootstrap CI for macro F1."""
    print("  Bootstrap CI...")
    metrics_path = BENCHMARK_DIR / 'results' / 'performance_metrics.json'
    if not metrics_path.exists():
        print("    SKIPPED"); return
    with open(metrics_path) as f:
        metrics = json.load(f)

    methods = ['rules', 'deep']
    method_colors = [COLORS['rules'], COLORS['deep']]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        m = metrics[method]
        f1 = m['macro_avg']['f1']
        ci_lo, ci_hi = m['macro_f1_ci']['lower'], m['macro_f1_ci']['upper']
        ax.barh(i, f1, color=method_colors[i], edgecolor='black', linewidth=0.5, height=0.6)
        ax.errorbar(f1, i, xerr=[[f1 - ci_lo], [ci_hi - f1]], fmt='none',
                   ecolor='black', capsize=8, linewidth=2)
        ax.text(ci_hi + 0.01, i, f'{f1:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]', va='center', fontsize=10)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([m.upper() for m in methods])
    ax.set_xlabel('Macro F1 Score'); ax.set_title('Macro F1 with 95% Bootstrap Confidence Intervals')
    ax.set_xlim(0, 1.0); ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_bootstrap_ci.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_method_comparison_bars(output_dir):
    """Method classification distributions."""
    print("  Method comparison...")
    csv_path = BENCHMARK_DIR / 'results' / 'method_comparison.csv'
    if not csv_path.exists():
        print("    SKIPPED"); return
    df = pd.read_csv(csv_path)
    methods = df['method'].tolist()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods)); width = 0.25
    ax1.bar(x - width, df['bot_locations_pct'], width, label='Bot', color=COLORS['bot'])
    ax1.bar(x, df['hub_locations_pct'], width, label='Hub', color=COLORS['hub'])
    ax1.bar(x + width, 100 - df['bot_locations_pct'] - df['hub_locations_pct'], width,
            label='Organic', color=COLORS['organic'])
    ax1.set_xticks(x); ax1.set_xticklabels([m.upper() for m in methods])
    ax1.set_ylabel('Percentage of Locations'); ax1.set_title('A) Classification by Locations')
    ax1.legend(frameon=False); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
    ax2.bar(x - width, df['bot_dl_pct'], width, label='Bot', color=COLORS['bot'])
    ax2.bar(x, df['hub_dl_pct'], width, label='Hub', color=COLORS['hub'])
    ax2.bar(x + width, df['organic_dl_pct'], width, label='Organic', color=COLORS['organic'])
    ax2.set_xticks(x); ax2.set_xticklabels([m.upper() for m in methods])
    ax2.set_ylabel('Percentage of Downloads'); ax2.set_title('B) Classification by Downloads')
    ax2.legend(frameon=False); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_method_comparison.pdf', format='pdf')
    plt.close()
    print("    OK")


# ====================================================================
# Extended analysis figures (all from filtered data)
# ====================================================================

def supp_fig_hourly_patterns(output_dir):
    """Hourly download heatmap."""
    print("  Hourly patterns...")
    csv_path = ANALYSIS_DIR / 'hourly_patterns.csv'
    if not csv_path.exists():
        print("    SKIPPED"); return
    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(values='downloads', index='hour', columns='day_of_week', fill_value=0)
    pivot_norm = pivot / pivot.values.max()
    fig, ax = plt.subplots(figsize=(8, 8))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    im = ax.imshow(pivot_norm.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(day_names[:len(pivot.columns)])
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    ax.set_xlabel('Day of Week'); ax.set_ylabel('Hour (UTC)')
    ax.set_title('Download Activity by Hour and Day of Week\n(After Bot Removal)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Relative Activity')
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_hourly_patterns.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_monthly_trends(output_dir):
    """Monthly download line chart."""
    print("  Monthly trends...")
    csv_path = ANALYSIS_DIR / 'temporal_monthly.csv'
    if not csv_path.exists():
        print("    SKIPPED"); return
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df['date'], df['total_downloads'] / 1e3, alpha=0.3, color='#3498DB')
    ax.plot(df['date'], df['total_downloads'] / 1e3, color='#3498DB', linewidth=1.5)
    ax.set_xlabel('Date'); ax.set_ylabel('Downloads (thousands)')
    ax.set_title('Monthly Download Volume (After Bot Removal)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_monthly_trends.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_protocol_overall(conn, output_dir):
    """Overall protocol distribution bar chart (from filtered data)."""
    print("  Protocol distribution...")
    p = P()
    df = conn.execute(f"""
        SELECT method as protocol, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE year >= 2020 {FILT()}
        GROUP BY method ORDER BY downloads DESC
    """).df()

    total = df['downloads'].sum()
    df['pct'] = df['downloads'] / total * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    proto_colors = {'http': '#2ECC71', 'ftp': '#E67E22', 'gridftp-globus': '#9B59B6', 'fasp-aspera': '#3498DB'}
    colors = [proto_colors.get(p, '#95A5A6') for p in df['protocol']]
    bars = ax.bar(df['protocol'], df['pct'], color=colors, edgecolor='black', linewidth=0.5)
    for bar, pct in zip(bars, df['pct']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Percentage of Downloads'); ax.set_title('Download Methods Distribution (After Bot Removal)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_protocol_overall.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_downloads_per_file(conn, output_dir):
    """Downloads per file histogram (from filtered data)."""
    print("  Downloads per file...")
    p = P()
    df = conn.execute(f"""
        SELECT filename, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE filename IS NOT NULL {FILT()}
        GROUP BY filename
    """).df()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(df['downloads'], bins=np.logspace(0, np.log10(df['downloads'].max()), 100),
            color='#3498DB', edgecolor='white', linewidth=0.3)
    ax.set_xscale('log'); ax.set_xlabel('Downloads per File (log)')
    ax.set_ylabel('Count'); ax.set_title('Distribution of Downloads per File (Log Scale)\n(After Bot Removal)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_downloads_per_file.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_european_trends(conn, output_dir):
    """European country trends over years (from filtered data)."""
    print("  European trends...")
    p = P()
    countries_sql = ','.join(f"'{c}'" for c in EUROPEAN_COUNTRIES)
    df = conn.execute(f"""
        SELECT country, year, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE country IN ({countries_sql}) AND year >= 2021 {FILT()}
        GROUP BY country, year ORDER BY country, year
    """).df()

    # Convert to percentages within each year
    yearly_totals = df.groupby('year')['downloads'].sum()
    df['pct'] = df.apply(lambda r: r['downloads'] / yearly_totals[r['year']] * 100, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for country in EUROPEAN_COUNTRIES:
        cdf = df[df['country'] == country]
        if len(cdf) > 0:
            ax.plot(cdf['year'], cdf['pct'], 'o-', label=country, linewidth=1.5, markersize=5)

    ax.set_xlabel('Year'); ax.set_ylabel('Download Percentage (%)')
    ax.set_title('European Countries: Download Trends by Year\n(Top 15 Countries, After Bot Removal)')
    ax.legend(loc='upper right', fontsize=7, ncol=3, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_european_trends.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_consistency_heatmap(conn, output_dir):
    """Top dataset download consistency heatmap (from filtered data)."""
    print("  Consistency heatmap...")
    p = P()
    # Get top 25 datasets
    top_ds = conn.execute(f"""
        SELECT accession, COUNT(*) as total FROM read_parquet('{p}')
        WHERE accession IS NOT NULL {FILT()}
        GROUP BY accession ORDER BY total DESC LIMIT 25
    """).df()

    accessions_sql = ','.join(f"'{a}'" for a in top_ds['accession'])
    yearly = conn.execute(f"""
        SELECT accession, year, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE accession IN ({accessions_sql}) AND year >= 2020 {FILT()}
        GROUP BY accession, year ORDER BY accession, year
    """).df()

    pivot = yearly.pivot_table(values='downloads', index='accession', columns='year', fill_value=0)
    # Reorder by total downloads
    pivot = pivot.loc[top_ds['accession']]

    fig, ax = plt.subplots(figsize=(10, 8))
    data = np.log10(pivot.values + 1)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel('Year'); ax.set_ylabel('Project Accession')
    ax.set_title('Download Consistency Heatmap: Top 25 Projects\n(log10 scale, After Bot Removal)')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Downloads (log10 scale)')
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_consistency_heatmap.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_consistency_scores(conn, output_dir):
    """Top 20 most consistently downloaded datasets (from filtered data)."""
    print("  Consistency scores...")
    p = P()
    yearly = conn.execute(f"""
        SELECT accession, year, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE accession IS NOT NULL AND year >= 2021 {FILT()}
        GROUP BY accession, year
    """).df()

    # Compute consistency score per dataset
    scores = []
    for acc, grp in yearly.groupby('accession'):
        if len(grp) < 3:
            continue
        vals = grp['downloads'].values
        cv = vals.std() / vals.mean() if vals.mean() > 0 else 1.0
        activity_ratio = len(grp) / 5.0  # 5 years (2021-2025)
        consistency = (1 - min(cv, 1.0)) * activity_ratio
        scores.append({'accession': acc, 'consistency': consistency, 'total': vals.sum()})

    scores_df = pd.DataFrame(scores).sort_values('consistency', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(scores_df))
    ax.barh(y, scores_df['consistency'], color='#3498DB', edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(scores_df['accession'], fontsize=8)
    ax.invert_yaxis()
    for i, (_, row) in enumerate(scores_df.iterrows()):
        ax.text(row['consistency'] + 0.005, i, f"{row['consistency']:.3f}", va='center', fontsize=8)
    ax.set_xlabel('Consistency Score')
    ax.set_title('Top 20 Most Consistent Projects (After Bot Removal)\nConsistency = (1 - CV) x Activity Ratio')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_consistency_scores.pdf', format='pdf')
    plt.close()
    print("    OK")


def supp_fig_bubble_chart(conn, output_dir):
    """Downloads vs unique users by country bubble chart (from filtered data)."""
    print("  Bubble chart...")
    p = P()
    df = conn.execute(f"""
        SELECT country,
               COUNT(*) as total_downloads,
               COUNT(DISTINCT user) as unique_users,
               COUNT(DISTINCT accession) as unique_datasets
        FROM read_parquet('{p}')
        WHERE country IS NOT NULL AND country != '' AND country NOT LIKE '%{{%' {FILT()}
        GROUP BY country
        HAVING total_downloads > 10000
        ORDER BY total_downloads DESC
        LIMIT 50
    """).df()

    df['dl_per_user'] = df['total_downloads'] / df['unique_users'].clip(lower=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['unique_users'], df['total_downloads'],
                         s=df['dl_per_user'].clip(upper=500) * 2,
                         c=df['dl_per_user'], cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Unique Users (log scale)'); ax.set_ylabel('Total Downloads (log scale)')
    ax.set_title('Total Downloads vs Unique Users (Top 50 Countries)\n(After Bot Removal)')
    plt.colorbar(scatter, ax=ax, label='Downloads per User', fraction=0.03, pad=0.04)

    # Label top countries
    for _, row in df.head(20).iterrows():
        ax.annotate(row['country'], (row['unique_users'], row['total_downloads']),
                   fontsize=7, ha='left', va='bottom',
                   xytext=(5, 3), textcoords='offset points')

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / 'supp_bubble_chart.pdf', format='pdf')
    plt.close()
    print("    OK")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("GENERATING SUPPLEMENTARY FIGURES (ALL FROM FILTERED DATA)")
    print("=" * 70)

    conn = get_filtered_connection()

    # Bot removal
    supp_fig_classification_distribution(FIGURES_DIR)
    supp_fig_bot_removal_geographic(conn, FIGURES_DIR)

    # Benchmark
    supp_fig_benchmark_details(FIGURES_DIR)
    supp_fig_bootstrap_ci(FIGURES_DIR)
    supp_fig_method_comparison_bars(FIGURES_DIR)

    # Extended analysis (all from filtered DuckDB queries)
    supp_fig_hourly_patterns(FIGURES_DIR)
    supp_fig_monthly_trends(FIGURES_DIR)
    supp_fig_protocol_overall(conn, FIGURES_DIR)
    supp_fig_downloads_per_file(conn, FIGURES_DIR)
    supp_fig_european_trends(conn, FIGURES_DIR)
    supp_fig_consistency_heatmap(conn, FIGURES_DIR)
    supp_fig_consistency_scores(conn, FIGURES_DIR)
    supp_fig_bubble_chart(conn, FIGURES_DIR)

    conn.close()

    print(f"\nAll supplementary figures saved to: {FIGURES_DIR}")
    for f in sorted(FIGURES_DIR.glob('supp_*.pdf')):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
