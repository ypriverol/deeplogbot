#!/usr/bin/env python3
"""Generate figure correlating EuropePMC citations with download metrics.

Queries EuropePMC for the number of publications mentioning each PXD accession,
then correlates citation count with total downloads and download consistency
(number of active years).

Output: paper/figures/figure_citation_correlation.pdf
"""

import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path

ANALYSIS_DIR = Path('output/phase6_analysis')
OUTPUT_PATH = Path('paper/figures/figure_citation_correlation.pdf')
CACHE_PATH = Path('output/phase6_analysis/europepmc_citations.json')

# Filtered parquet for yearly breakdown
PARQUET_PATH = Path('output/phase6_analysis/filtered_downloads.parquet')
# Fallback: use the full dataset parquet with the bot-filtered connection
FULL_PARQUET = Path('pride_data/data_downloads_parquet.parquet')


def query_europepmc(accession: str, max_retries: int = 3) -> int:
    """Query EuropePMC for number of publications mentioning an accession."""
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        'query': f'"{accession}"',
        'format': 'json',
        'pageSize': 1,
        'resultType': 'core',
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return int(data.get('hitCount', 0))
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  WARNING: Failed to query {accession}: {e}")
                return 0


def get_citation_counts(accessions: list) -> dict:
    """Get citation counts, using cache when available."""
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached citation counts")

    missing = [a for a in accessions if a not in cache]
    if missing:
        print(f"  Querying EuropePMC for {len(missing)} accessions...")
        for i, acc in enumerate(missing):
            cache[acc] = query_europepmc(acc)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(missing)} done")
            time.sleep(0.35)  # Be polite to the API

        # Save updated cache
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"  Saved cache ({len(cache)} entries)")

    return {a: cache.get(a, 0) for a in accessions}


def get_yearly_downloads(accessions: list) -> dict:
    """Get yearly download counts per dataset from the analysis data."""
    import duckdb

    # Try filtered parquet first, then full dataset
    parquet = None
    for p in [PARQUET_PATH, FULL_PARQUET]:
        if p.exists():
            parquet = p
            break

    if parquet is None:
        print("  WARNING: No parquet file found for yearly breakdown")
        return {}

    conn = duckdb.connect()
    try:
        escaped = str(parquet).replace("'", "''")
        acc_list = "','".join(accessions)
        query = f"""
            SELECT accession, year, COUNT(*) as downloads
            FROM read_parquet('{escaped}')
            WHERE accession IN ('{acc_list}')
            GROUP BY accession, year
            ORDER BY accession, year
        """
        df = conn.execute(query).df()

        yearly = {}
        for acc in accessions:
            acc_data = df[df['accession'] == acc]
            if len(acc_data) > 0:
                yearly[acc] = {
                    'years_active': len(acc_data),
                    'yearly_downloads': dict(zip(acc_data['year'].astype(int), acc_data['downloads'].astype(int))),
                }
            else:
                yearly[acc] = {'years_active': 0, 'yearly_downloads': {}}
        return yearly
    except Exception as e:
        print(f"  WARNING: Yearly query failed: {e}")
        return {}
    finally:
        conn.close()


def main():
    print("Generating citation correlation figure...")

    # Load top datasets
    top_path = ANALYSIS_DIR / 'top_datasets.csv'
    if not top_path.exists():
        print(f"  ERROR: {top_path} not found")
        return

    df = pd.read_csv(top_path).head(50)
    accessions = df['accession'].tolist()
    print(f"  {len(accessions)} datasets loaded (top 50)")

    # Get citation counts from EuropePMC
    citations = get_citation_counts(accessions)
    # Subtract 1: the original submission paper always mentions its own PXD,
    # so reuse citations = total mentions - 1
    df['citation_count'] = df['accession'].map(citations).apply(lambda x: max(x - 1, 0))

    # Get yearly download data for consistency metric
    yearly = get_yearly_downloads(accessions)
    if yearly:
        df['years_active'] = df['accession'].map(lambda a: yearly.get(a, {}).get('years_active', 0))
    else:
        # Use columns from top_datasets.csv
        df['years_active'] = df['last_download_year'] - df['first_download_year'] + 1

    # Compute download consistency score: years_active / max_possible_years
    max_years = 5  # 2021-2025
    df['consistency_score'] = df['years_active'].clip(upper=max_years) / max_years

    # Filter to datasets with at least 1 citation for meaningful correlation
    df_cited = df[df['citation_count'] > 0].copy()
    df_uncited = df[df['citation_count'] == 0].copy()

    print(f"  {len(df_cited)} datasets with >= 1 citation, {len(df_uncited)} with 0 citations")
    print(f"  Citation range: {df['citation_count'].min()} - {df['citation_count'].max()}")
    print(f"  Median citations: {df['citation_count'].median():.0f}")

    # ---- Create figure: 2 panels ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Panel A: Citations vs Total Downloads (scatter, log-log) ----
    # Color by consistency
    sc = ax1.scatter(
        df_cited['citation_count'],
        df_cited['total_downloads'],
        c=df_cited['consistency_score'],
        cmap='RdYlGn',
        s=50,
        alpha=0.7,
        edgecolors='#333333',
        linewidths=0.5,
        vmin=0,
        vmax=1,
        zorder=3,
    )
    # Uncited datasets in grey
    if len(df_uncited) > 0:
        ax1.scatter(
            [0.8] * len(df_uncited),  # Offset from 0 for log scale
            df_uncited['total_downloads'],
            c='#cccccc',
            s=30,
            alpha=0.4,
            edgecolors='#999999',
            linewidths=0.3,
            zorder=2,
        )

    # Spearman correlation (on cited datasets)
    if len(df_cited) > 5:
        rho, p_val = sp_stats.spearmanr(df_cited['citation_count'], df_cited['total_downloads'])
        # Add regression line on log scale
        log_x = np.log10(df_cited['citation_count'].values)
        log_y = np.log10(df_cited['total_downloads'].values)
        slope, intercept, r_val, _, _ = sp_stats.linregress(log_x, log_y)
        x_fit = np.linspace(log_x.min(), log_x.max(), 100)
        ax1.plot(10**x_fit, 10**(slope * x_fit + intercept), 'r--', alpha=0.6, linewidth=1.5, zorder=4)

        ax1.text(0.05, 0.95, f'Spearman $\\rho$ = {rho:.3f}\np = {p_val:.2e}\nn = {len(df_cited)}',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Label top datasets
    for _, row in df_cited.nlargest(5, 'total_downloads').iterrows():
        ax1.annotate(
            row['accession'],
            (row['citation_count'], row['total_downloads']),
            fontsize=6.5, alpha=0.8,
            xytext=(5, 5), textcoords='offset points',
        )

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('EuropePMC Citation Count', fontsize=11)
    ax1.set_ylabel('Total Downloads (bot-filtered)', fontsize=11)
    ax1.set_title('(A) Citations vs Downloads', fontsize=12, fontweight='bold', loc='left')
    cb = plt.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
    cb.set_label('Download Consistency\n(years active / 5)', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, which='both')

    # ---- Panel B: Citation count vs Download consistency ----
    # Box/violin plot: binned citation groups vs consistency
    bins = [0, 1, 5, 20, 50, 1000]
    labels_bins = ['0', '1-4', '5-19', '20-49', '50+']
    df['citation_bin'] = pd.cut(df['citation_count'], bins=bins, labels=labels_bins, right=False)

    bin_data = []
    bin_labels = []
    bin_counts = []
    for label in labels_bins:
        subset = df[df['citation_bin'] == label]['consistency_score']
        if len(subset) > 0:
            bin_data.append(subset.values)
            bin_labels.append(label)
            bin_counts.append(len(subset))

    if bin_data:
        bp = ax2.boxplot(bin_data, tick_labels=bin_labels, patch_artist=True,
                         widths=0.6, showfliers=True,
                         flierprops=dict(marker='o', markersize=3, alpha=0.4))
        colors = ['#ecf0f1', '#aed6f1', '#85c1e9', '#5dade2', '#2e86c1']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_edgecolor('#2c3e50')
        for median in bp['medians']:
            median.set_color('#e74c3c')
            median.set_linewidth(2)

        # Add counts
        for i, (lbl, cnt) in enumerate(zip(bin_labels, bin_counts)):
            ax2.text(i + 1, -0.08, f'n={cnt}', ha='center', va='top', fontsize=8, color='#666666')

        # Kruskal-Wallis test
        if len(bin_data) >= 2 and all(len(b) > 0 for b in bin_data):
            h_stat, kw_p = sp_stats.kruskal(*[b for b in bin_data if len(b) > 1])
            ax2.text(0.95, 0.95, f'Kruskal-Wallis\nH = {h_stat:.1f}, p = {kw_p:.2e}',
                     transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('EuropePMC Citation Count (binned)', fontsize=11)
    ax2.set_ylabel('Download Consistency Score\n(years active / 5)', fontsize=11)
    ax2.set_title('(B) Citations vs Download Sustainability', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim(-0.15, 1.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_PATH}")

    # Print summary stats
    print(f"\n  Summary:")
    print(f"  Total datasets: {len(df)}")
    print(f"  With citations: {len(df_cited)} ({len(df_cited)/len(df)*100:.1f}%)")
    print(f"  Mean citations: {df['citation_count'].mean():.1f}")
    print(f"  Median citations: {df['citation_count'].median():.0f}")
    print(f"  Max citations: {df['citation_count'].max()} ({df.loc[df['citation_count'].idxmax(), 'accession']})")


if __name__ == '__main__':
    main()
