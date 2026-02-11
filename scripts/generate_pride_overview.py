#!/usr/bin/env python3
"""Generate the PRIDE overview supplementary figure (supp_pride_overview.png).

Creates a clean 2x3 panel figure without duplicate panels.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_PATH = "paper/figures/supp_pride_overview.png"

# Data from PRIDE download analysis (2020-2025)
SCALE = {
    "Total Downloads": 47.35e6,
    "Unique Files": 2.26e6,
    "Unique Projects": 32.11e3,
    "Unique Users": 807.16e3,
}

REUSE = {
    "Downloads\nper Project": 1474.8,
    "Downloads\nper File": 20.9,
    "Downloads\nper User": 58.7,
}

FILE_COVERAGE = 88.0  # %
PROJECT_COVERAGE = 96.4  # %
COUNTRIES = 136


def fmt_count(v):
    if v >= 1e6:
        return f"{v/1e6:.2f} M"
    if v >= 1e3:
        return f"{v/1e3:.2f} K"
    return f"{v:.0f}"


def main():
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    fig.suptitle(
        "PRIDE Download Activity Overview (2020\u20132025)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # --- (A) Overall Scale Metrics ---
    ax = axes[0, 0]
    labels = list(SCALE.keys())
    values = list(SCALE.values())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                fmt_count(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.25)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
    ax.set_title("Overall Scale", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- (B) Reuse Intensity ---
    ax = axes[0, 1]
    labels_r = list(REUSE.keys())
    values_r = list(REUSE.values())
    bars = ax.bar(labels_r, values_r, color=["#3498db", "#e67e22", "#2ecc71"],
                  edgecolor="white", width=0.6)
    for bar, v in zip(bars, values_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values_r) * 0.02,
                f"{v:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values_r) * 1.15)
    ax.set_title("Reuse Intensity", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Average Downloads")

    # --- (C) Geographic Reach ---
    ax = axes[0, 2]
    ax.text(0.5, 0.55, f"{COUNTRIES}", ha="center", va="center",
            fontsize=52, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    ax.text(0.5, 0.25, "Countries / Territories\n(>100 downloads each)",
            ha="center", va="center", fontsize=11, color="#7f8c8d",
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Geographic Reach", fontsize=12, fontweight="bold")

    # --- (D) Project Coverage ---
    ax = axes[1, 0]
    wedges, _ = ax.pie(
        [PROJECT_COVERAGE, 100 - PROJECT_COVERAGE],
        colors=["#3498db", "#ecf0f1"],
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0, f"{PROJECT_COVERAGE}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#2c3e50")
    ax.set_title("Project Coverage\n(of public datasets)", fontsize=12, fontweight="bold")

    # --- (E) File Coverage ---
    ax = axes[1, 1]
    wedges, _ = ax.pie(
        [FILE_COVERAGE, 100 - FILE_COVERAGE],
        colors=["#2ecc71", "#ecf0f1"],
        startangle=90,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0, f"{FILE_COVERAGE}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#2c3e50")
    ax.set_title("File Coverage\n(downloaded at least once)", fontsize=12, fontweight="bold")

    # --- (F) Time period label ---
    ax = axes[1, 2]
    ax.text(0.5, 0.55, "Jan 2020 \u2013 Jan 2025", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)
    ax.text(0.5, 0.30, "5 years of download logs\n159.3 M raw records",
            ha="center", va="center", fontsize=11, color="#7f8c8d",
            transform=ax.transAxes)
    ax.axis("off")
    ax.set_title("Study Period", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
