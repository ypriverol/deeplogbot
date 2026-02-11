#!/usr/bin/env python3
"""Fetch pridepy PyPI download statistics and generate a monthly trend plot."""

import json
import urllib.request
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import os
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figures")


def fetch_pypistats_data():
    """Fetch daily download data from pypistats.org API (last 180 days)."""
    url = "https://pypistats.org/api/packages/pridepy/overall?mirrors=true"
    req = urllib.request.Request(url, headers={"User-Agent": "pridepy-stats/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())["data"]

    monthly = defaultdict(int)
    for entry in data:
        if entry["category"] == "with_mirrors":
            month = entry["date"][:7]
            monthly[month] += entry["downloads"]
    return dict(monthly)


def get_full_2025_data():
    """Combine API data with ClickHouse/pepy.tech historical data for early 2025.

    The pypistats.org API only retains 180 days of data. For months before
    the API window, we use approximate totals from ClickPy (clickpy.clickhouse.com),
    which ingests the same Google BigQuery public PyPI dataset.
    """
    # Historical monthly data from ClickPy (Jan-Jul 2025)
    historical = {
        "2025-01": 540,
        "2025-02": 824,
        "2025-03": 595,
        "2025-04": 564,
        "2025-05": 299,
        "2025-06": 234,
        "2025-07": 533,
    }

    # Fetch precise data from pypistats API
    api_data = fetch_pypistats_data()
    print("API data (last 180 days):")
    for m in sorted(api_data):
        print(f"  {m}: {api_data[m]}")

    # Merge: use API data where available, historical otherwise
    combined = {}
    for month, count in historical.items():
        if month not in api_data:
            combined[month] = count
    combined.update(api_data)

    return combined


def generate_plot(monthly_data):
    """Generate monthly download trend bar chart."""
    # Filter to 2025 only
    months_2025 = {k: v for k, v in sorted(monthly_data.items()) if k.startswith("2025")}

    labels = list(months_2025.keys())
    values = list(months_2025.values())
    dates = [datetime.strptime(m, "%Y-%m") for m in labels]
    month_labels = [d.strftime("%b") for d in dates]

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = []
    for m in labels:
        # March 2025 = pridepy publication month
        if m == "2025-03":
            colors.append("#e74c3c")  # Red for release month
        else:
            colors.append("#3498db")  # Blue for other months

    bars = ax.bar(month_labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Mark pridepy publication
    march_idx = labels.index("2025-03")
    ax.annotate(
        "pridepy\npublished",
        xy=(march_idx, values[march_idx]),
        xytext=(march_idx + 0.5, max(values) * 0.85),
        fontsize=9,
        fontweight="bold",
        color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        ha="left",
    )

    # Cumulative line
    cumulative = np.cumsum(values)
    ax2 = ax.twinx()
    ax2.plot(
        month_labels,
        cumulative,
        color="#2ecc71",
        marker="o",
        linewidth=2,
        markersize=5,
        label="Cumulative",
    )
    ax2.set_ylabel("Cumulative downloads", fontsize=11, color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

    ax.set_xlabel("Month (2025)", fontsize=11)
    ax.set_ylabel("Monthly downloads", fontsize=11)
    ax.set_title("pridepy PyPI Downloads (2025)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)

    # Add total annotation
    total = sum(values)
    ax.text(
        0.98, 0.95,
        f"Total 2025: {total:,} downloads",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    plt.tight_layout()
    outpath = f"{OUTPUT_DIR}/supp_pridepy_downloads.pdf"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.close()

    # Also save as PNG
    fig2, ax2_copy = plt.subplots(figsize=(8, 4))
    bars2 = ax2_copy.bar(month_labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, values):
        ax2_copy.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            str(val),
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    ax2_copy.annotate(
        "pridepy\npublished",
        xy=(march_idx, values[march_idx]),
        xytext=(march_idx + 0.5, max(values) * 0.85),
        fontsize=9, fontweight="bold", color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        ha="left",
    )
    cumulative2 = np.cumsum(values)
    ax2_twin = ax2_copy.twinx()
    ax2_twin.plot(month_labels, cumulative2, color="#2ecc71", marker="o", linewidth=2, markersize=5)
    ax2_twin.set_ylabel("Cumulative downloads", fontsize=11, color="#2ecc71")
    ax2_twin.tick_params(axis="y", labelcolor="#2ecc71")
    ax2_copy.set_xlabel("Month (2025)", fontsize=11)
    ax2_copy.set_ylabel("Monthly downloads", fontsize=11)
    ax2_copy.set_title("pridepy PyPI Downloads (2025)", fontsize=13, fontweight="bold")
    ax2_copy.set_ylim(0, max(values) * 1.25)
    ax2_copy.text(
        0.98, 0.95, f"Total 2025: {total:,} downloads",
        transform=ax2_copy.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )
    plt.tight_layout()
    fig2.savefig(f"{OUTPUT_DIR}/supp_pridepy_downloads.png", dpi=300, bbox_inches="tight")
    plt.close()

    return total


def main():
    print("Fetching pridepy download statistics...\n")
    monthly_data = get_full_2025_data()

    print("\nMonthly downloads (2025):")
    total = 0
    for month in sorted(monthly_data):
        if month.startswith("2025"):
            print(f"  {month}: {monthly_data[month]:>6}")
            total += monthly_data[month]
    print(f"  {'Total':>7}: {total:>6}")

    generate_plot(monthly_data)


if __name__ == "__main__":
    main()
