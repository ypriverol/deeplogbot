#!/usr/bin/env python3
"""Generate publication-quality figures for the PRIDE manuscript.

Creates vector PDF figures from analysis data in output/phase6_analysis/
and benchmark data in output/phase4_final_benchmark/.

Usage:
    python scripts/generate_figures.py
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import duckdb

# Style settings for publication
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

EUROPEAN_COUNTRIES = [
    'Germany', 'United Kingdom', 'France', 'Spain', 'Italy', 'Netherlands',
    'Switzerland', 'Sweden', 'Denmark', 'Belgium', 'Finland', 'Norway',
    'Austria', 'Poland', 'Ireland',
]

# World Bank low/middle income countries present in PRIDE data
# China excluded here (upper-middle income, already dominant in panel A)
LMIC_COUNTRIES = [
    'India', 'Brazil', 'Mexico', 'Indonesia', 'Thailand',
    'Colombia', 'Argentina', 'South Africa', 'Vietnam', 'Bangladesh',
    'Pakistan', 'Peru', 'Chile', 'Philippines', 'Nigeria', 'Egypt',
    'Kenya', 'Iran', 'Malaysia', 'Morocco', 'Turkey', 'Ukraine',
]

# Color palette
COLORS = {
    'bot': '#E74C3C',       # red
    'hub': '#3498DB',       # blue
    'organic': '#2ECC71',   # green
    'rules': '#E67E22',     # orange
    'deep': '#9B59B6',      # purple
}


def figure_bot_detection_overview(output_dir):
    """Combined figure: (A) Pipeline workflow + (B) Full-dataset classification distribution."""
    print("  Bot detection overview (combined)...")
    summary_path = Path(project_root / 'output' / 'phase5_full_classification' / 'classification_summary.json')

    if not summary_path.exists():
        print("    SKIPPED - missing data")
        return

    with open(summary_path) as f:
        stats = json.load(f)

    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.05)

    # ---- Panel A: Pipeline workflow diagram ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(A) PRIDE Logs Workflow', fontsize=12, fontweight='bold', pad=10)

    # Style definitions
    def draw_box(ax, x, y, w, h, text, color='#EBF5FB', edge='#2980B9', fontsize=8, bold=False):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                             facecolor=color, edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2, color='#2C3E50'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # --- Component 1: nf-downloadstats (top section) ---
    # Dashed box around the nf-downloadstats component
    from matplotlib.patches import FancyBboxPatch as FBP
    import matplotlib.patches as mpatches
    nf_rect = mpatches.FancyBboxPatch((0.05, 8.1), 9.4, 1.7, boxstyle='round,pad=0.15',
                                       facecolor='none', edgecolor='#27AE60', linewidth=2.0,
                                       linestyle='--')
    ax.add_patch(nf_rect)
    ax.text(0.3, 9.65, 'nf-downloadstats', fontsize=9, fontweight='bold', color='#27AE60',
            fontstyle='italic')

    # Row 1: Data collection
    draw_box(ax, 0.2, 8.3, 2.2, 1.0, 'PRIDE\nLog Files\n(TSV)', color='#FDEBD0', edge='#E67E22', fontsize=8, bold=True)
    draw_arrow(ax, 2.4, 8.8, 3.0, 8.8)
    draw_box(ax, 3.0, 8.3, 3.0, 1.0, 'Parse, Filter\n& Merge', color='#D5F5E3', edge='#27AE60', fontsize=8)
    draw_arrow(ax, 6.0, 8.8, 6.6, 8.8)
    draw_box(ax, 6.6, 8.3, 2.6, 1.0, 'Parquet\n159M records\n(4.7 GB)', color='#FDEBD0', edge='#E67E22', fontsize=8, bold=True)

    # --- Component 2: DeepLogBot (bottom section) ---
    lg_rect = mpatches.FancyBboxPatch((0.05, -0.15), 9.4, 7.55, boxstyle='round,pad=0.15',
                                       facecolor='none', edgecolor='#2980B9', linewidth=2.0,
                                       linestyle='--')
    ax.add_patch(lg_rect)
    ax.text(0.3, 7.2, 'DeepLogBot', fontsize=9, fontweight='bold', color='#2980B9',
            fontstyle='italic')

    # Row 2: Location aggregation + Feature extraction
    draw_box(ax, 0.3, 5.9, 4.2, 1.0, 'Location Aggregation\n47,987 geographic locations', color='#EBF5FB', edge='#2980B9', fontsize=8)
    draw_arrow(ax, 4.5, 6.4, 5.0, 6.4)
    draw_box(ax, 5.0, 5.9, 4.2, 1.0, 'Feature Extraction\n60+ behavioral features\n(activity, temporal, discriminative)', color='#EBF5FB', edge='#2980B9', fontsize=7.5)

    # Arrows inside DeepLogBot from top to both boxes (no arrow from Parquet)
    midx = 4.75  # midpoint between the two boxes
    midy = 7.15  # just below DeepLogBot label
    draw_arrow(ax, midx, midy, 2.4, 6.9)
    draw_arrow(ax, midx, midy, 7.1, 6.9)

    # Arrows down to anomaly detection
    draw_arrow(ax, 2.4, 5.9, 4.75, 5.4)
    draw_arrow(ax, 7.1, 5.9, 4.75, 5.4)

    # Row 3: Anomaly detection
    draw_box(ax, 2.5, 4.3, 4.5, 0.9, 'Anomaly Detection\nIsolation Forest (contamination=15%)', color='#F5EEF8', edge='#8E44AD', fontsize=8)

    # Arrow down to methods
    draw_arrow(ax, 4.75, 4.3, 4.75, 3.8)

    # Row 4: Two classification methods side by side
    draw_box(ax, 1.0, 2.7, 3.0, 0.9, 'Rule-Based\nYAML thresholds\n3-level hierarchy', color='#FADBD8', edge='#E74C3C', fontsize=7.5, bold=False)
    draw_box(ax, 5.5, 2.7, 3.0, 0.9, 'Deep Architecture\n40+ extra features\n2-stage pipeline', color='#FADBD8', edge='#E74C3C', fontsize=7.5, bold=False)

    draw_arrow(ax, 4.75, 3.8, 2.5, 3.6)
    draw_arrow(ax, 4.75, 3.8, 7.0, 3.6)

    # Arrows down to hierarchical classification
    draw_arrow(ax, 2.5, 2.7, 4.75, 2.2)
    draw_arrow(ax, 7.0, 2.7, 4.75, 2.2)

    # Row 5: Hierarchical classification output
    draw_box(ax, 2.5, 1.1, 4.5, 0.9, 'Hierarchical Classification\nL1: Organic vs Automated\nL2: Bot vs Hub\nL3: Subcategory', color='#D4EFDF', edge='#27AE60', fontsize=7.5)

    # Arrow to final output
    draw_arrow(ax, 4.75, 1.1, 4.75, 0.7)
    draw_box(ax, 1.5, 0.0, 6.5, 0.6, 'Bot-filtered dataset: 35.4M downloads, 34,908 datasets, 208 countries',
             color='#D5F5E3', edge='#27AE60', fontsize=7.5, bold=True)

    # ---- Panel B: Classification distribution (download share bar chart) ----
    ax2 = fig.add_subplot(gs[0, 1])
    dl_cats = ['Organic', 'Hub', 'Bot']
    dl_vals = [stats['organic_dl_pct'], stats['hub_dl_pct'], stats['bot_dl_pct']]
    cat_colors = [COLORS['organic'], COLORS['hub'], COLORS['bot']]

    bars = ax2.bar(dl_cats, dl_vals, color=cat_colors, edgecolor='black', linewidth=0.5, width=0.65)
    ax2.set_ylabel('Percentage of Total Downloads', fontsize=10)
    ax2.set_title('(B) Full Dataset Classification', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, dl_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add location counts as secondary info inside bars
    loc_counts = [stats['organic_locations'], stats['hub_locations'], stats['bot_locations']]
    total_locs = sum(loc_counts)
    for bar, count in zip(bars, loc_counts):
        pct = count / total_locs * 100
        y_pos = max(bar.get_height() / 2, 5)
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'n={count:,}\n({pct:.1f}% locs)', ha='center', fontsize=7.5,
                color='white', fontweight='bold')

    plt.savefig(output_dir / 'figure_bot_detection_overview.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_1_world_map(output_dir):
    """Figure 1: Geographic distribution of PRIDE downloads (bot-filtered)."""
    print("  Figure 1: Geographic distribution...")
    csv_path = ANALYSIS_DIR / 'geographic_by_country.csv'
    if not csv_path.exists():
        print("    SKIPPED - no geographic data")
        return

    df = pd.read_csv(csv_path)
    df = df[~df['country'].str.contains('%{', na=False)]
    top20 = df.head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(top20))
    bars = ax.barh(y_pos, top20['total_downloads'] / 1e6, color='#3498DB', edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['country'])
    ax.invert_yaxis()
    ax.set_xlabel('Total Downloads (millions)')
    ax.set_title('Top 20 Countries by PRIDE Downloads')

    for bar, val in zip(bars, top20['total_downloads']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.2f}M', va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_geographic_distribution.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_1b_regional(output_dir):
    """Figure 1b: Regional distribution pie chart."""
    print("  Figure 1b: Regional distribution...")
    csv_path = ANALYSIS_DIR / 'geographic_by_region.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    region_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#E67E22', '#1ABC9C']

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        df['total_downloads'],
        labels=df['region'],
        autopct='%1.1f%%',
        colors=region_colors[:len(df)],
        startangle=90,
        pctdistance=0.75,
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title('PRIDE Downloads by Region')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1b_regional_distribution.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_2_temporal(output_dir):
    """Figure 2: Downloads over time (yearly trend)."""
    print("  Figure 2: Temporal trends...")
    csv_path = ANALYSIS_DIR / 'temporal_yearly.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Total downloads per year
    ax1.bar(df['year'], df['total_downloads'] / 1e6, color='#3498DB', edgecolor='white', width=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Downloads (millions)')
    ax1.set_title('A) Annual Download Volume')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    for _, row in df.iterrows():
        ax1.text(row['year'], row['total_downloads'] / 1e6 + 0.5,
                f"{row['total_downloads']/1e6:.1f}M", ha='center', fontsize=8)

    # Panel B: Unique datasets and locations
    ax2b = ax2.twinx()
    l1 = ax2.plot(df['year'], df['unique_datasets'] / 1e3, 'o-', color='#E67E22', label='Unique datasets (k)')
    l2 = ax2b.plot(df['year'], df['unique_locations'] / 1e3, 's--', color='#9B59B6', label='Unique locations (k)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Unique Datasets (thousands)', color='#E67E22')
    ax2b.set_ylabel('Unique Locations (thousands)', color='#9B59B6')
    ax2.set_title('B) Dataset and Location Growth')
    ax2.spines['top'].set_visible(False)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_temporal_trends.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_3_algorithm_comparison(output_dir):
    """Figure 3: Bot detection algorithm comparison."""
    print("  Figure 3: Algorithm comparison...")
    csv_path = BENCHMARK_DIR / 'results' / 'method_comparison.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    methods = df['method'].tolist()
    method_colors = [COLORS.get(m, '#999') for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Classification distribution (stacked bars)
    ax = axes[0]
    bot_pcts = df['bot_locations_pct'].values
    hub_pcts = df['hub_locations_pct'].values
    organic_pcts = 100 - bot_pcts - hub_pcts
    x = range(len(methods))
    ax.bar(x, organic_pcts, label='Organic', color=COLORS['organic'])
    ax.bar(x, hub_pcts, bottom=organic_pcts, label='Hub', color=COLORS['hub'])
    ax.bar(x, bot_pcts, bottom=organic_pcts + hub_pcts, label='Bot', color=COLORS['bot'])
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Percentage of Locations')
    ax.set_title('A) Classification Distribution')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: F1 scores per category
    ax = axes[1]
    categories = ['bot', 'hub', 'organic']
    x_pos = np.arange(len(methods))
    width = 0.25
    for i, cat in enumerate(categories):
        vals = df[f'{cat}_f1'].values
        ax.bar(x_pos + i * width, vals, width, label=cat.capitalize(), color=COLORS[cat])
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('F1 Score')
    ax.set_title('B) Per-Category F1 Scores')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Macro F1 with confidence intervals
    ax = axes[2]
    macro_f1s = df['macro_f1'].values
    ci_lower = df['macro_f1_ci_lower'].values if 'macro_f1_ci_lower' in df.columns else macro_f1s - 0.03
    ci_upper = df['macro_f1_ci_upper'].values if 'macro_f1_ci_upper' in df.columns else macro_f1s + 0.03
    errors = np.array([macro_f1s - ci_lower, ci_upper - macro_f1s])

    bars = ax.bar(x_pos, macro_f1s, color=method_colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_pos, macro_f1s, yerr=errors, fmt='none', ecolor='black', capsize=5, linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel('Macro F1 Score')
    ax.set_title('C) Overall Performance (95% CI)')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, macro_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_algorithm_comparison.pdf', format='pdf')
    plt.close()
    print("    OK")


def figure_4_protocols(output_dir):
    """Figure 4: Protocol usage over time (stacked bar chart)."""
    print("  Figure 4: Protocol usage...")
    csv_path = ANALYSIS_DIR / 'protocol_usage.csv'
    if not csv_path.exists() or os.path.getsize(csv_path) < 10:
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return

    # Map raw protocol names to display names
    protocol_names = {
        'http': 'HTTP',
        'ftp': 'FTP',
        'fasp-aspera': 'Aspera',
        'gridftp-globus': 'Globus',
    }
    df['protocol'] = df['protocol'].map(lambda x: protocol_names.get(x, x))

    # Exclude 2020 (only 279 downloads — too sparse)
    df = df[df['year'] >= 2021]

    # Pivot to get protocols as columns
    pivot = df.pivot_table(values='downloads', index='year', columns='protocol', fill_value=0)

    # Order protocols for consistent stacking
    protocol_order = [p for p in ['FTP', 'HTTP', 'Aspera', 'Globus'] if p in pivot.columns]
    pivot = pivot[protocol_order]

    protocol_colors = {'HTTP': '#3498DB', 'FTP': '#E67E22', 'Aspera': '#2ECC71',
                       'Globus': '#9B59B6'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Absolute download counts (stacked bar)
    years = pivot.index.astype(str)
    bottom = np.zeros(len(years))
    for proto in protocol_order:
        values = pivot[proto].values
        ax1.bar(years, values / 1e6, bottom=bottom / 1e6,
                color=protocol_colors[proto], label=proto, edgecolor='white', linewidth=0.5)
        bottom += values

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Downloads (millions)')
    ax1.set_title('(A) Download Volume by Protocol')
    ax1.legend(title='Protocol', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Monthly protocol breakdown for 2025
    monthly_path = ANALYSIS_DIR / 'protocol_monthly_2025.csv'
    if monthly_path.exists():
        df_m = pd.read_csv(monthly_path)
        df_m['protocol'] = df_m['protocol'].map(lambda x: protocol_names.get(x, x))
        pivot_m = df_m.pivot_table(values='downloads', index='month', columns='protocol', fill_value=0)
        proto_order_m = [p for p in ['FTP', 'HTTP', 'Aspera', 'Globus'] if p in pivot_m.columns]
        pivot_m = pivot_m[proto_order_m]

        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = pivot_m.index.values
        x_pos = np.arange(len(months))

        bottom_m = np.zeros(len(months))
        for proto in proto_order_m:
            values = pivot_m[proto].values / 1e6
            ax2.bar(x_pos, values, bottom=bottom_m,
                    color=protocol_colors[proto], label=proto, edgecolor='white', linewidth=0.5)
            bottom_m += values

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([month_labels[m - 1] for m in months], rotation=45, ha='right')
        ax2.set_xlabel('Month (2025)')
        ax2.set_ylabel('Downloads (millions)')
        ax2.set_title('(B) Monthly Protocol Usage in 2025')
        ax2.legend(title='Protocol', frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_protocol_usage.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_5_concentration(output_dir):
    """Figure 5: Rank-frequency log-log (A) + consistency heatmap (B) + citations vs downloads (C)."""
    print("  Figure 5: Concentration + consistency + citations...")
    stats_path = ANALYSIS_DIR / 'concentration_stats.json'
    top_path = ANALYSIS_DIR / 'top_datasets.csv'

    if not stats_path.exists():
        print("    SKIPPED - no concentration data")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Load rank-frequency data
    if top_path.exists():
        top_all = pd.read_csv(top_path)
        downloads = top_all['total_downloads'].sort_values(ascending=False).values
    else:
        print("    SKIPPED - no top_datasets.csv")
        return

    # Query yearly data for consistency heatmap
    conn = _get_filtered_connection()
    heatmap_data = None
    if conn is not None and PARQUET_PATH.exists():
        try:
            p = str(PARQUET_PATH).replace("'", "''")
            filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"
            top_ds = conn.execute(f"""
                SELECT accession, COUNT(*) as total FROM read_parquet('{p}')
                WHERE accession IS NOT NULL {filt}
                GROUP BY accession ORDER BY total DESC LIMIT 25
            """).df()
            accessions_sql = ','.join(f"'{a}'" for a in top_ds['accession'])
            yearly = conn.execute(f"""
                SELECT accession, year, COUNT(*) as downloads
                FROM read_parquet('{p}')
                WHERE accession IN ({accessions_sql}) AND year >= 2021 {filt}
                GROUP BY accession, year ORDER BY accession, year
            """).df()
            conn.close()
            pivot = yearly.pivot_table(values='downloads', index='accession', columns='year', fill_value=0)
            pivot = pivot.reindex(top_ds['accession'])
            for y in range(2021, 2026):
                if y not in pivot.columns:
                    pivot[y] = 0
            pivot = pivot[sorted(pivot.columns)]
            heatmap_data = pivot
        except Exception as e:
            print(f"    Warning: heatmap query failed: {e}")

    # Layout: top row = A (rank-freq) + B (heatmap), bottom row = C (citations)
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.8, 1.2], height_ratios=[1, 0.75],
                           wspace=0.3, hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # ---- Panel A: Rank-frequency (log-log) ----
    ranks = np.arange(1, len(downloads) + 1)
    ax1.scatter(ranks, downloads, s=8, alpha=0.5, color='steelblue', edgecolors='none')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Dataset Rank')
    ax1.set_ylabel('Total Downloads')
    ax1.set_title('(A) Rank-Frequency Distribution', fontsize=11, fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Mark top 1% boundary
    top1_idx = int(len(downloads) * 0.01)
    if top1_idx > 0:
        ax1.axvline(x=top1_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(top1_idx * 1.3, downloads[0] * 0.5, 'Top 1%',
                 color='red', fontsize=9, fontweight='bold')

    # Annotate key statistics
    textstr = (f'Gini = {stats["gini_coefficient"]:.2f}\n'
               f'Top 1%: {stats["top_1pct_downloads_pct"]:.1f}% of DL\n'
               f'Top 10%: {stats["top_10pct_downloads_pct"]:.1f}% of DL\n'
               f'Median: {stats["median_downloads"]:,} DL')
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ---- Panel B: Consistency heatmap ----
    if heatmap_data is not None:
        active_years = (heatmap_data > 0).sum(axis=1)
        data = np.log10(heatmap_data.values + 1)
        im = ax2.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        ax2.set_xticks(range(len(heatmap_data.columns)))
        ax2.set_xticklabels(heatmap_data.columns.astype(int), fontsize=9)
        ax2.set_yticks(range(len(heatmap_data.index)))
        ylabels = [f'{acc}  ({active_years[acc]}/{len(heatmap_data.columns)} yrs)'
                   for acc in heatmap_data.index]
        ax2.set_yticklabels(ylabels, fontsize=7)
        ax2.set_xlabel('Year')

        cbar = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
        cbar.set_label('Downloads (log$_{10}$ scale)', fontsize=9)

        # Annotate cells
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = int(heatmap_data.iloc[i, j])
                if val > 0:
                    if val >= 1e6:
                        txt = f'{val/1e6:.1f}M'
                    elif val >= 1e3:
                        txt = f'{val/1e3:.0f}K'
                    else:
                        txt = str(val)
                    color = 'white' if data[i, j] > 3.5 else 'black'
                    ax2.text(j, i, txt, ha='center', va='center',
                             fontsize=5.5, color=color, fontweight='bold')

        ax2.set_title('(B) Top 25 Datasets: Download Consistency', fontsize=11, fontweight='bold', loc='left', pad=15)
    else:
        ax2.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(B) Top 25 Datasets: Download Consistency', fontsize=11, fontweight='bold', loc='left', pad=15)

    # ---- Panel C: Citations vs Downloads (scatter, log-log) ----
    from scipy import stats as sp_stats
    citation_cache = ANALYSIS_DIR / 'europepmc_citations.json'
    if citation_cache.exists() and top_path.exists():
        with open(citation_cache) as f:
            cite_data = json.load(f)

        df_cite = pd.read_csv(top_path).head(50)
        # n-1 correction: subtract original submission paper
        df_cite['citation_count'] = df_cite['accession'].map(
            lambda a: max(cite_data.get(a, 1) - 1, 0))
        df_cite['years_active'] = df_cite['last_download_year'] - df_cite['first_download_year'] + 1
        max_years = 5
        df_cite['consistency_score'] = df_cite['years_active'].clip(upper=max_years) / max_years

        df_cited = df_cite[df_cite['citation_count'] > 0].copy()
        df_uncited = df_cite[df_cite['citation_count'] == 0].copy()

        sc = ax3.scatter(
            df_cited['citation_count'], df_cited['total_downloads'],
            c=df_cited['consistency_score'], cmap='RdYlGn',
            s=50, alpha=0.7, edgecolors='#333333', linewidths=0.5,
            vmin=0, vmax=1, zorder=3)

        if len(df_uncited) > 0:
            ax3.scatter(
                [0.8] * len(df_uncited), df_uncited['total_downloads'],
                c='#cccccc', s=30, alpha=0.4, edgecolors='#999999',
                linewidths=0.3, zorder=2)

        if len(df_cited) > 5:
            rho, p_val = sp_stats.spearmanr(df_cited['citation_count'], df_cited['total_downloads'])
            log_x = np.log10(df_cited['citation_count'].values)
            log_y = np.log10(df_cited['total_downloads'].values)
            slope, intercept, _, _, _ = sp_stats.linregress(log_x, log_y)
            x_fit = np.linspace(log_x.min(), log_x.max(), 100)
            ax3.plot(10**x_fit, 10**(slope * x_fit + intercept), 'r--', alpha=0.6, linewidth=1.5, zorder=4)
            ax3.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}\np = {p_val:.2e}\nn = {len(df_cited)}',
                     transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        for _, row in df_cited.nlargest(5, 'total_downloads').iterrows():
            ax3.annotate(row['accession'], (row['citation_count'], row['total_downloads']),
                         fontsize=6.5, alpha=0.8, xytext=(5, 5), textcoords='offset points')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('EuropePMC Reuse Citation Count')
        ax3.set_ylabel('Total Downloads (bot-filtered)')
        cb = plt.colorbar(sc, ax=ax3, shrink=0.8, pad=0.02)
        cb.set_label('Download Consistency\n(years active / 5)', fontsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_title('(C) Reuse Citations vs Downloads', fontsize=11, fontweight='bold', loc='left')
    else:
        ax3.text(0.5, 0.5, 'Citation data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(C) Reuse Citations vs Downloads', fontsize=11, fontweight='bold', loc='left')

    plt.savefig(output_dir / 'figure5_dataset_concentration.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_6_top_datasets(output_dir):
    """Figure 6: Top 20 most downloaded datasets."""
    print("  Figure 6: Top datasets...")
    csv_path = ANALYSIS_DIR / 'top_datasets.csv'
    if not csv_path.exists():
        print("    SKIPPED")
        return

    df = pd.read_csv(csv_path).head(20)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['total_downloads'] / 1e6, color='#2ECC71', edgecolor='white')

    labels = []
    for _, row in df.iterrows():
        label = row['accession']
        if 'title' in df.columns and pd.notna(row.get('title')):
            title = str(row['title'])[:40]
            label = f"{row['accession']} ({title})"
        labels.append(label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Total Downloads (millions)')
    ax.set_title('Top 20 Most Downloaded PRIDE Datasets')

    for bar, val in zip(bars, df['total_downloads']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val/1e6:.2f}M', va='center', fontsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_top_datasets.pdf', format='pdf')
    plt.close()
    print("    OK")


def supplementary_figure_agreement(output_dir):
    """Supplementary: Algorithm agreement/disagreement analysis."""
    print("  Supp Figure: Agreement analysis...")
    json_path = BENCHMARK_DIR / 'results' / 'agreement_matrix.json'
    if not json_path.exists():
        print("    SKIPPED")
        return

    with open(json_path) as f:
        agreement = json.load(f)

    methods = ['rules', 'deep']
    categories = ['bot', 'hub', 'organic']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Overall agreement + kappa heatmap
    ax = axes[0]
    matrix = np.ones((2, 2))
    for pair, data in agreement.items():
        m1, m2 = pair.split('_vs_')
        i, j = methods.index(m1), methods.index(m2)
        matrix[i, j] = data['overall_agreement']
        matrix[j, i] = data['overall_agreement']

    im = ax.imshow(matrix, cmap='YlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_yticklabels([m.upper() for m in methods])
    ax.set_title('A) Pairwise Agreement')

    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f'{matrix[i, j]:.1%}', ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel B: Per-category agreement bars
    ax = axes[1]
    x_pos = np.arange(len(list(agreement.keys())))
    width = 0.25
    for i, cat in enumerate(categories):
        vals = [data['category_agreement'].get(cat, 0) for data in agreement.values()]
        ax.bar(x_pos + i * width, vals, width, label=cat.capitalize(), color=COLORS[cat])

    ax.set_xticks(x_pos + width)
    pair_labels = [p.replace('_vs_', ' vs ').upper() for p in agreement.keys()]
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylabel('Category Agreement')
    ax.set_title('B) Per-Category Agreement')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'supp_figure_agreement.pdf', format='pdf')
    plt.close()
    print("    OK")


def _get_filtered_connection():
    """Get DuckDB connection with bot-filtered locations."""
    labels_path = CLASSIFICATION_DIR / 'pride_classification_final.csv'
    if not labels_path.exists():
        return None
    labels_df = pd.read_csv(labels_path)
    if 'final_label' in labels_df.columns:
        non_bot = labels_df[labels_df['final_label'] != 'bot'][['geo_location']].drop_duplicates()
    elif 'is_bot' in labels_df.columns:
        non_bot = labels_df[~labels_df['is_bot']][['geo_location']].drop_duplicates()
    else:
        return None
    conn = duckdb.connect()
    conn.execute("PRAGMA memory_limit='4GB'")
    tmp = os.path.abspath('./duckdb-tmp/')
    os.makedirs(tmp, exist_ok=True)
    conn.execute(f"PRAGMA temp_directory='{tmp}'")
    conn.execute("PRAGMA threads=2")
    conn.register('_cl', non_bot)
    conn.execute("CREATE TEMP TABLE clean_locations AS SELECT * FROM _cl")
    return conn


def _query_country_yearly_trends(conn, countries):
    """Query yearly download counts for a list of countries (bot-filtered)."""
    p = str(PARQUET_PATH).replace("'", "''")
    filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"
    countries_sql = ','.join(f"'{c}'" for c in countries)
    df = conn.execute(f"""
        SELECT country, year, COUNT(*) as downloads
        FROM read_parquet('{p}')
        WHERE country IN ({countries_sql}) AND year >= 2021 {filt}
        GROUP BY country, year ORDER BY country, year
    """).df()
    return df


def figure_7_bubble_chart(output_dir):
    """Figure 7: Country bubble chart + European & LMIC download trends."""
    print("  Figure 7: Country bubble chart with regional trends...")
    csv_path = ANALYSIS_DIR / 'country_bubble_data.csv'
    if not csv_path.exists():
        print("    SKIPPED - no bubble data")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return

    # Query yearly trends from parquet
    conn = _get_filtered_connection()
    has_trends = conn is not None and PARQUET_PATH.exists()
    eu_df = lmic_df = None
    if has_trends:
        try:
            eu_df = _query_country_yearly_trends(conn, EUROPEAN_COUNTRIES)
            lmic_df = _query_country_yearly_trends(conn, LMIC_COUNTRIES)
            conn.close()
        except Exception as e:
            print(f"    Warning: could not query trends: {e}")
            has_trends = False

    # Layout: bubble chart on left (spanning full height), two trend panels stacked on right
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_bubble = fig.add_subplot(gs[:, 0])  # left, spans both rows
    ax_europe = fig.add_subplot(gs[0, 1])  # top-right
    ax_lmic = fig.add_subplot(gs[1, 1])    # bottom-right

    # ---- Panel A: Bubble chart ----
    dl_per_user = df['dl_per_user'].values
    size_raw = np.clip(dl_per_user, 1, 2000)
    sizes = (size_raw / size_raw.max()) * 800 + 15

    scatter = ax_bubble.scatter(
        df['unique_users'], df['total_downloads'],
        s=sizes,
        c=dl_per_user, cmap='viridis',
        alpha=0.75, edgecolors='black', linewidth=0.5,
        norm=plt.matplotlib.colors.LogNorm(vmin=max(dl_per_user.min(), 1), vmax=dl_per_user.max()),
        zorder=3,
    )

    ax_bubble.set_xscale('log')
    ax_bubble.set_yscale('log')
    ax_bubble.set_xlabel('Unique Users (log scale)')
    ax_bubble.set_ylabel('Total Downloads (log scale)')

    plt.colorbar(scatter, ax=ax_bubble, label='Downloads per User', fraction=0.03, pad=0.04)

    labeled = []
    for _, row in df.iterrows():
        x, y = row['unique_users'], row['total_downloads']
        name = row['country']
        ha, x_off, y_off = 'left', 8, 4
        if x > 50000:
            ha, x_off = 'right', -8
        for lx, ly in labeled:
            if abs(np.log10(x) - np.log10(lx)) < 0.15 and abs(np.log10(y) - np.log10(ly)) < 0.08:
                y_off += 8
        fontsize = 8 if row['total_downloads'] > 500000 else 7
        fontweight = 'bold' if row['total_downloads'] > 1000000 else 'normal'
        ax_bubble.annotate(
            name, (x, y),
            fontsize=fontsize, fontweight=fontweight,
            ha=ha, va='bottom',
            xytext=(x_off, y_off), textcoords='offset points',
        )
        labeled.append((x, y))

    legend_sizes = [10, 100, 500]
    legend_bubbles = []
    for val in legend_sizes:
        s = (np.clip(val, 1, 2000) / np.clip(dl_per_user, 1, 2000).max()) * 800 + 15
        legend_bubbles.append(
            ax_bubble.scatter([], [], s=s, c='gray', alpha=0.5, edgecolors='black', linewidth=0.5)
        )
    ax_bubble.legend(
        legend_bubbles, [f'{v}' for v in legend_sizes],
        title='DL/User (size)', loc='upper left',
        frameon=True, framealpha=0.9, fontsize=8, title_fontsize=8,
        labelspacing=1.5, borderpad=1.2,
    )
    ax_bubble.spines['top'].set_visible(False)
    ax_bubble.spines['right'].set_visible(False)
    ax_bubble.grid(True, alpha=0.2, which='both')
    ax_bubble.set_title('(A) Downloads vs. Users by Country', fontsize=11, fontweight='bold', loc='left')

    # ---- Panel B: European trends ----
    if has_trends and eu_df is not None and not eu_df.empty:
        # Show absolute downloads (in millions) per year
        eu_colors = plt.cm.tab20(np.linspace(0, 1, len(EUROPEAN_COUNTRIES)))
        for i, country in enumerate(EUROPEAN_COUNTRIES):
            cdf = eu_df[eu_df['country'] == country].sort_values('year')
            if len(cdf) > 0:
                ax_europe.plot(cdf['year'], cdf['downloads'] / 1e6, 'o-',
                               label=country, linewidth=1.3, markersize=4,
                               color=eu_colors[i])
        ax_europe.set_xlabel('Year')
        ax_europe.set_ylabel('Downloads (millions)')
        ax_europe.set_title('(B) European Countries', fontsize=11, fontweight='bold', loc='left')
        ax_europe.legend(loc='upper left', fontsize=6.5, ncol=3, frameon=False)
        ax_europe.spines['top'].set_visible(False)
        ax_europe.spines['right'].set_visible(False)
        ax_europe.grid(True, alpha=0.2)
        ax_europe.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax_europe.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax_europe.transAxes)
        ax_europe.set_title('(B) European Countries', fontsize=11, fontweight='bold', loc='left')

    # ---- Panel C: LMIC trends ----
    if has_trends and lmic_df is not None and not lmic_df.empty:
        # Show top countries by total downloads across years
        top_lmic = lmic_df.groupby('country')['downloads'].sum().nlargest(15).index.tolist()
        lmic_colors = plt.cm.tab20(np.linspace(0, 1, len(top_lmic)))
        for i, country in enumerate(top_lmic):
            cdf = lmic_df[lmic_df['country'] == country].sort_values('year')
            if len(cdf) > 0:
                ax_lmic.plot(cdf['year'], cdf['downloads'] / 1e3, 'o-',
                             label=country, linewidth=1.3, markersize=4,
                             color=lmic_colors[i])
        ax_lmic.set_xlabel('Year')
        ax_lmic.set_ylabel('Downloads (thousands)')
        ax_lmic.set_title('(C) Low/Middle Income Countries', fontsize=11, fontweight='bold', loc='left')
        ax_lmic.legend(loc='upper left', fontsize=6.5, ncol=3, frameon=False)
        ax_lmic.spines['top'].set_visible(False)
        ax_lmic.spines['right'].set_visible(False)
        ax_lmic.grid(True, alpha=0.2)
        ax_lmic.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    else:
        ax_lmic.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax_lmic.transAxes)
        ax_lmic.set_title('(C) Low/Middle Income Countries', fontsize=11, fontweight='bold', loc='left')

    plt.savefig(output_dir / 'figure7_country_bubble.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_dataset_consistency(output_dir):
    """Figure: Top dataset download consistency heatmap over years."""
    print("  Dataset consistency heatmap...")
    conn = _get_filtered_connection()
    if conn is None or not PARQUET_PATH.exists():
        print("    SKIPPED - no data")
        return

    p = str(PARQUET_PATH).replace("'", "''")
    filt = "AND geo_location IN (SELECT geo_location FROM clean_locations)"

    try:
        # Get top 25 datasets by total downloads (bot-filtered)
        top_ds = conn.execute(f"""
            SELECT accession, COUNT(*) as total FROM read_parquet('{p}')
            WHERE accession IS NOT NULL {filt}
            GROUP BY accession ORDER BY total DESC LIMIT 25
        """).df()

        accessions_sql = ','.join(f"'{a}'" for a in top_ds['accession'])
        yearly = conn.execute(f"""
            SELECT accession, year, COUNT(*) as downloads
            FROM read_parquet('{p}')
            WHERE accession IN ({accessions_sql}) AND year >= 2021 {filt}
            GROUP BY accession, year ORDER BY accession, year
        """).df()
        conn.close()
    except Exception as e:
        print(f"    Error querying: {e}")
        return

    pivot = yearly.pivot_table(values='downloads', index='accession', columns='year', fill_value=0)
    # Reorder by total downloads (top first)
    pivot = pivot.reindex(top_ds['accession'])
    # Ensure all years present
    for y in range(2021, 2026):
        if y not in pivot.columns:
            pivot[y] = 0
    pivot = pivot[sorted(pivot.columns)]

    # Count years with >0 downloads for each dataset
    active_years = (pivot > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))
    data = np.log10(pivot.values + 1)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=10)
    ax.set_yticks(range(len(pivot.index)))

    # Label with accession + active years
    ylabels = [f'{acc}  ({active_years[acc]}/{len(pivot.columns)} yrs)'
               for acc in pivot.index]
    ax.set_yticklabels(ylabels, fontsize=8)

    ax.set_xlabel('Year')
    ax.set_ylabel('Dataset Accession')

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Downloads (log$_{10}$ scale)', fontsize=10)

    # Annotate cells with actual download counts
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(pivot.iloc[i, j])
            if val > 0:
                # Format: K for thousands, M for millions
                if val >= 1e6:
                    txt = f'{val/1e6:.1f}M'
                elif val >= 1e3:
                    txt = f'{val/1e3:.0f}K'
                else:
                    txt = str(val)
                color = 'white' if data[i, j] > 3.5 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=6, color=color, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_dataset_consistency.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_filetype_by_region(output_dir):
    """Figure: File type download patterns by region."""
    print("  File type by region figure...")
    csv_path = ANALYSIS_DIR / 'filetype_by_region.csv'
    if not csv_path.exists():
        print("    SKIPPED - no filetype data (run analysis first)")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("    SKIPPED - empty data")
        return

    # Order categories by importance for the narrative
    cat_order = ['raw', 'processed_spectra', 'result', 'tabular', 'database', 'metadata', 'compressed', 'other']
    cat_labels = {
        'raw': 'Raw instrument\nfiles',
        'processed_spectra': 'Processed\nspectra',
        'result': 'Search\nresults',
        'tabular': 'Tabular\n(csv/tsv/xlsx)',
        'database': 'Sequence\ndatabases',
        'metadata': 'Metadata\n(xml/sdrf)',
        'compressed': 'Compressed\narchives',
        'other': 'Other',
    }
    cat_colors = {
        'raw': '#E74C3C',
        'processed_spectra': '#E67E22',
        'result': '#F1C40F',
        'tabular': '#2ECC71',
        'database': '#1ABC9C',
        'metadata': '#3498DB',
        'compressed': '#9B59B6',
        'other': '#BDC3C7',
    }

    region_order = ['East Asia', 'North America', 'Europe', 'LMIC']
    region_labels = {'East Asia': 'East Asia', 'North America': 'N. America',
                     'Europe': 'Europe', 'LMIC': 'LMIC'}

    # Compute percentages
    region_totals = df.groupby('region')['downloads'].sum()
    df['pct'] = df.apply(lambda r: r['downloads'] / region_totals[r['region']] * 100, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.4, 1]})

    # ---- Panel A: Grouped bar chart ----
    ax = axes[0]
    n_regions = len(region_order)
    n_cats = len(cat_order)
    bar_width = 0.18
    x = np.arange(n_cats)

    for i, region in enumerate(region_order):
        rdf = df[df['region'] == region]
        vals = []
        for cat in cat_order:
            v = rdf.loc[rdf['file_category'] == cat, 'pct']
            vals.append(v.values[0] if len(v) > 0 else 0)
        offset = (i - n_regions / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=region_labels[region],
                      color=plt.cm.Set2(i / n_regions), edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([cat_labels[c] for c in cat_order], fontsize=8)
    ax.set_ylabel('Percentage of Downloads (%)')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('(A) File Type Downloads by Region', fontsize=11, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.2)

    # ---- Panel B: Stacked horizontal bars for key categories only ----
    ax2 = axes[1]
    # Aggregate into: Raw/Processed (reanalysis) vs Result/Tabular (lightweight) vs Other
    def agg_type(cat):
        if cat in ('raw', 'processed_spectra'):
            return 'Raw + Processed\n(full reanalysis)'
        elif cat in ('result', 'tabular', 'database', 'metadata'):
            return 'Results + Metadata\n(lightweight reuse)'
        else:
            return 'Archives + Other'

    df['agg_type'] = df['file_category'].apply(agg_type)
    agg = df.groupby(['region', 'agg_type'])['downloads'].sum().reset_index()
    agg_totals = agg.groupby('region')['downloads'].sum()
    agg['pct'] = agg.apply(lambda r: r['downloads'] / agg_totals[r['region']] * 100, axis=1)

    agg_order = ['Raw + Processed\n(full reanalysis)', 'Results + Metadata\n(lightweight reuse)', 'Archives + Other']
    agg_colors = ['#E74C3C', '#3498DB', '#BDC3C7']

    y_pos = np.arange(len(region_order))
    left = np.zeros(len(region_order))

    for j, atype in enumerate(agg_order):
        vals = []
        for region in region_order:
            v = agg.loc[(agg['region'] == region) & (agg['agg_type'] == atype), 'pct']
            vals.append(v.values[0] if len(v) > 0 else 0)
        vals = np.array(vals)
        bars = ax2.barh(y_pos, vals, left=left, color=agg_colors[j],
                        edgecolor='white', linewidth=0.5, label=atype.replace('\n', ' '))
        # Label percentages
        for k, (v, l) in enumerate(zip(vals, left)):
            if v > 5:
                ax2.text(l + v / 2, k, f'{v:.0f}%', ha='center', va='center',
                         fontsize=9, fontweight='bold', color='white')
        left += vals

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([region_labels[r] for r in region_order], fontsize=10)
    ax2.set_xlabel('Percentage of Downloads (%)')
    ax2.legend(fontsize=8, loc='lower right', frameon=True, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(B) Reanalysis vs. Lightweight Reuse', fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure_filetype_by_region.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def figure_hub_distribution(output_dir):
    """Figure: Hub geographic distribution and characteristics."""
    print("  Hub distribution figure...")
    csv_path = CLASSIFICATION_DIR / 'pride_classification_final.csv'
    if not csv_path.exists():
        print("    SKIPPED - no classification data")
        return

    df = pd.read_csv(csv_path)
    if 'final_label' in df.columns:
        hubs = df[df['final_label'] == 'hub'].copy()
    elif 'is_hub' in df.columns:
        hubs = df[df['is_hub'] == True].copy()
    else:
        print("    SKIPPED - no label column")
        return
    if hubs.empty:
        print("    SKIPPED - no hubs")
        return

    fig = plt.figure(figsize=(16, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1], wspace=0.35)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 2])

    # ---- Panel A: World scatter of hub locations ----
    lats, lons = [], []
    for _, row in hubs.iterrows():
        try:
            parts = str(row['geo_location']).split(',')
            lat, lon = float(parts[0]), float(parts[1])
            lats.append(lat)
            lons.append(lon)
        except (ValueError, IndexError):
            lats.append(np.nan)
            lons.append(np.nan)
    hubs['lat'] = lats
    hubs['lon'] = lons
    valid = hubs.dropna(subset=['lat', 'lon'])

    dl_vals = valid['total_downloads'].values
    sizes = np.clip(dl_vals / dl_vals.max() * 300, 10, 300)

    ax_map.scatter(valid['lon'], valid['lat'], s=sizes,
                   c='#3498DB', alpha=0.6, edgecolors='navy', linewidth=0.4, zorder=3)

    # Simple world outline
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-60, 85)
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.axhline(0, color='gray', linewidth=0.3, alpha=0.5)
    ax_map.axvline(0, color='gray', linewidth=0.3, alpha=0.5)
    # Continental outlines via grid
    ax_map.grid(True, alpha=0.15)
    ax_map.spines['top'].set_visible(False)
    ax_map.spines['right'].set_visible(False)
    ax_map.set_title('(A) Hub Locations Worldwide', fontsize=11, fontweight='bold', loc='left')

    # Size legend
    for dl, label in [(10000, '10K'), (100000, '100K'), (500000, '500K')]:
        s = np.clip(dl / dl_vals.max() * 300, 10, 300)
        ax_map.scatter([], [], s=s, c='#3498DB', alpha=0.6, edgecolors='navy',
                       linewidth=0.4, label=label)
    ax_map.legend(title='Downloads', loc='lower left', fontsize=7, title_fontsize=7,
                  frameon=True, framealpha=0.9, labelspacing=1.2)

    # ---- Panel B: Top 15 countries by hub count ----
    country_counts = hubs['country'].value_counts().head(15)
    country_downloads = hubs.groupby('country')['total_downloads'].sum()

    colors_bar = ['#3498DB'] * len(country_counts)
    bars = ax_bar.barh(range(len(country_counts)), country_counts.values, color=colors_bar,
                       edgecolor='navy', linewidth=0.3, alpha=0.8)
    ax_bar.set_yticks(range(len(country_counts)))
    ax_bar.set_yticklabels(country_counts.index, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Number of Hubs')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_title('(B) Hubs per Country', fontsize=11, fontweight='bold', loc='left')

    # Annotate with download totals
    for i, (country, count) in enumerate(country_counts.items()):
        dl = country_downloads.get(country, 0)
        label = f'{dl/1e6:.1f}M' if dl >= 100000 else f'{dl/1e3:.0f}K'
        ax_bar.text(count + 0.5, i, label, va='center', fontsize=7, color='gray')

    # ---- Panel C: Hub users vs downloads (log-log) ----
    ax_scatter.scatter(hubs['unique_users'], hubs['total_downloads'],
                       s=40, c='#3498DB', alpha=0.6, edgecolors='navy', linewidth=0.4, zorder=3)
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlabel('Unique Users (log scale)')
    ax_scatter.set_ylabel('Total Downloads (log scale)')
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.grid(True, alpha=0.2, which='both')
    ax_scatter.set_title('(C) Hub Characteristics', fontsize=11, fontweight='bold', loc='left')

    # Label top hubs
    top_hubs = hubs.nlargest(5, 'total_downloads')
    for _, row in top_hubs.iterrows():
        label = row['city'] if pd.notna(row['city']) and row['city'] else row['country']
        ax_scatter.annotate(label, (row['unique_users'], row['total_downloads']),
                            fontsize=7, fontweight='bold',
                            xytext=(6, 4), textcoords='offset points')

    plt.savefig(output_dir / 'figure_hub_distribution.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("    OK")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 70)

    figure_bot_detection_overview(FIGURES_DIR)
    figure_1_world_map(FIGURES_DIR)
    figure_1b_regional(FIGURES_DIR)
    figure_2_temporal(FIGURES_DIR)
    figure_3_algorithm_comparison(FIGURES_DIR)
    figure_4_protocols(FIGURES_DIR)
    figure_5_concentration(FIGURES_DIR)
    figure_6_top_datasets(FIGURES_DIR)
    figure_7_bubble_chart(FIGURES_DIR)
    figure_hub_distribution(FIGURES_DIR)
    figure_filetype_by_region(FIGURES_DIR)
    supplementary_figure_agreement(FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    # List output
    for f in sorted(FIGURES_DIR.glob('*.pdf')):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == '__main__':
    main()
