"""Visualization module for bot detection reports.

This module provides plotting functions for:
- Classification distributions
- Temporal patterns (yearly, monthly, hourly)
- Feature distributions and comparisons
- Feature importance charts
- Geographic distributions
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from ..utils import logger

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualizations will be disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# Color palettes for consistent styling
CATEGORY_COLORS = {
    'bot': '#e74c3c',           # Red
    'download_hub': '#3498db',   # Blue
    'mirror': '#2980b9',         # Dark blue
    'ci_cd_pipeline': '#9b59b6',     # Purple
    'individual_user': '#27ae60',    # Green
    'research_group': '#f39c12',     # Orange
    'normal': '#95a5a6',             # Gray
    'other': '#bdc3c7',              # Light gray
    'organic': '#2ecc71',            # Green
    'automated': '#e67e22',          # Orange
    'legitimate_automation': '#3498db',  # Blue
}

BEHAVIOR_TYPE_COLORS = {
    'organic': '#2ecc71',
    'automated': '#e74c3c',
}

AUTOMATION_CATEGORY_COLORS = {
    'bot': '#e74c3c',
    'legitimate_automation': '#3498db',
}

# Hub subcategories for classification
HUB_SUBCATEGORIES = {'mirror', 'ci_cd_pipeline', 'course_workshop'}


def get_classification_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Get bot and hub masks from hierarchical classification columns.

    Returns:
        Tuple of (bot_mask, hub_mask) as boolean Series
    """
    # Bot mask: automation_category == 'bot'
    if 'automation_category' in df.columns:
        bot_mask = df['automation_category'] == 'bot'
    else:
        bot_mask = pd.Series(False, index=df.index)

    # Hub mask: subcategory in hub subcategories
    if 'subcategory' in df.columns:
        hub_mask = df['subcategory'].isin(HUB_SUBCATEGORIES)
    else:
        hub_mask = pd.Series(False, index=df.index)

    return bot_mask, hub_mask


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure output directory exists and return Path object."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


class VisualizationGenerator:
    """Generate visualizations for bot detection analysis."""

    def __init__(self, output_dir: str, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = ensure_output_dir(output_dir)
        self.plots_dir = ensure_output_dir(os.path.join(output_dir, 'plots'))
        self.style = style
        self.generated_plots: List[str] = []

        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(style)
            except OSError:
                plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')

    def _save_plot(self, fig: 'plt.Figure', name: str, dpi: int = 150) -> str:
        """Save a plot and return the path."""
        if not MATPLOTLIB_AVAILABLE:
            return ""

        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.generated_plots.append(str(path))
        logger.debug(f"Saved plot: {path}")
        return str(path)

    def plot_classification_distribution(self, df: pd.DataFrame,
                                         classification_method: str = 'rules') -> Optional[str]:
        """
        Plot distribution of classification categories.

        Creates a pie chart and bar chart showing category distribution.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Determine which columns to use based on method
        if classification_method == 'deep' and 'behavior_type' in df.columns:
            # Hierarchical classification
            # Left: Behavior type pie chart
            bt_counts = df['behavior_type'].value_counts()
            colors = [BEHAVIOR_TYPE_COLORS.get(bt, '#95a5a6') for bt in bt_counts.index]
            axes[0].pie(bt_counts.values, labels=bt_counts.index.str.upper(),
                       autopct='%1.1f%%', colors=colors, startangle=90)
            axes[0].set_title('Level 1: Behavior Type Distribution', fontsize=12, fontweight='bold')

            # Right: Subcategory bar chart
            if 'subcategory' in df.columns:
                subcat_counts = df['subcategory'].value_counts().head(10)
                colors = [CATEGORY_COLORS.get(sc, '#95a5a6') for sc in subcat_counts.index]
                bars = axes[1].barh(range(len(subcat_counts)), subcat_counts.values, color=colors)
                axes[1].set_yticks(range(len(subcat_counts)))
                axes[1].set_yticklabels(subcat_counts.index)
                axes[1].set_xlabel('Number of Locations')
                axes[1].set_title('Level 3: Top Subcategories', fontsize=12, fontweight='bold')
                axes[1].invert_yaxis()

                # Add count labels
                for bar, count in zip(bars, subcat_counts.values):
                    axes[1].text(bar.get_width() + max(subcat_counts.values) * 0.01,
                                bar.get_y() + bar.get_height()/2,
                                f'{count:,}', va='center', fontsize=9)
        else:
            # Use hierarchical classification
            categories = []
            counts = []

            bot_mask, hub_mask = get_classification_masks(df)
            bot_count = bot_mask.sum()
            hub_count = hub_mask.sum()

            if bot_count > 0:
                categories.append('Bot')
                counts.append(bot_count)

            if hub_count > 0:
                categories.append('Download Hub')
                counts.append(hub_count)

            normal_count = len(df) - sum(counts)
            categories.append('Normal')
            counts.append(normal_count)

            colors = [CATEGORY_COLORS.get(c.lower().replace(' ', '_'), '#95a5a6') for c in categories]

            # Pie chart
            axes[0].pie(counts, labels=categories, autopct='%1.1f%%',
                       colors=colors, startangle=90)
            axes[0].set_title('Classification Distribution', fontsize=12, fontweight='bold')

            # Bar chart
            bars = axes[1].bar(categories, counts, color=colors)
            axes[1].set_ylabel('Number of Locations')
            axes[1].set_title('Classification Counts', fontsize=12, fontweight='bold')

            for bar, count in zip(bars, counts):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                            f'{count:,}', ha='center', fontsize=10)

        plt.tight_layout()
        return self._save_plot(fig, 'classification_distribution')

    def plot_downloads_by_category(self, df: pd.DataFrame,
                                   classification_method: str = 'rules') -> Optional[str]:
        """Plot total downloads by category."""
        if not MATPLOTLIB_AVAILABLE or 'total_downloads' not in df.columns:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if classification_method == 'deep' and 'behavior_type' in df.columns:
            # Group by behavior type
            downloads_by_bt = df.groupby('behavior_type')['total_downloads'].sum()
            colors = [BEHAVIOR_TYPE_COLORS.get(bt, '#95a5a6') for bt in downloads_by_bt.index]
            bars = ax.bar(downloads_by_bt.index.str.upper(), downloads_by_bt.values, color=colors)
        else:
            categories = ['Bot', 'Download Hub', 'Normal']
            bot_mask, hub_mask = get_classification_masks(df)
            downloads = [
                df.loc[bot_mask, 'total_downloads'].sum() if bot_mask.any() else 0,
                df.loc[hub_mask, 'total_downloads'].sum() if hub_mask.any() else 0,
                0  # Calculated below
            ]
            downloads[2] = df['total_downloads'].sum() - downloads[0] - downloads[1]
            colors = [CATEGORY_COLORS['bot'], CATEGORY_COLORS['download_hub'], CATEGORY_COLORS['normal']]
            bars = ax.bar(categories, downloads, color=colors)

        ax.set_ylabel('Total Downloads')
        ax.set_title('Downloads by Classification Category', fontsize=12, fontweight='bold')

        # Format y-axis with human-readable numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label = f'{height/1e6:.1f}M' if height >= 1e6 else f'{height/1e3:.0f}K'
            ax.text(bar.get_x() + bar.get_width()/2, height, label,
                   ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return self._save_plot(fig, 'downloads_by_category')

    def plot_yearly_trends(self, df: pd.DataFrame, year_col: str = 'years_span') -> Optional[str]:
        """
        Plot yearly download trends.

        Note: This requires yearly data which may not be directly available.
        Uses available yearly features to show temporal patterns.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Years span distribution
        if 'years_span' in df.columns:
            years_span = df['years_span'].dropna()
            axes[0].hist(years_span, bins=range(1, int(years_span.max()) + 2),
                        color='#3498db', edgecolor='white', alpha=0.8)
            axes[0].set_xlabel('Years of Activity')
            axes[0].set_ylabel('Number of Locations')
            axes[0].set_title('Distribution of Location Activity Duration', fontsize=11, fontweight='bold')

        # Right: Latest year fraction distribution
        if 'fraction_latest_year' in df.columns:
            bot_mask, hub_mask = get_classification_masks(df)
            for cat, color in [('Bot', '#e74c3c'), ('Hub', '#3498db'), ('Normal', '#95a5a6')]:
                if cat == 'Bot':
                    data = df[bot_mask]['fraction_latest_year'].dropna()
                elif cat == 'Hub':
                    data = df[hub_mask]['fraction_latest_year'].dropna()
                else:
                    normal_mask = ~bot_mask & ~hub_mask
                    data = df[normal_mask]['fraction_latest_year'].dropna()

                if len(data) > 0:
                    axes[1].hist(data, bins=20, alpha=0.5, label=cat, color=color)

            axes[1].set_xlabel('Fraction of Downloads in Latest Year')
            axes[1].set_ylabel('Number of Locations')
            axes[1].set_title('Latest Year Activity Concentration', fontsize=11, fontweight='bold')
            axes[1].legend()

        plt.tight_layout()
        return self._save_plot(fig, 'yearly_trends')

    def plot_feature_distributions(self, df: pd.DataFrame,
                                   features: List[str] = None) -> Optional[str]:
        """Plot distributions of key features by category."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if features is None:
            features = ['unique_users', 'downloads_per_user', 'working_hours_ratio',
                       'night_activity_ratio', 'regularity_score', 'anomaly_score']

        # Filter to available features
        features = [f for f in features if f in df.columns]
        if not features:
            return None

        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(features):
            ax = axes[i]

            # Create category column for coloring
            bot_mask, hub_mask = get_classification_masks(df)
            if bot_mask.any() or hub_mask.any():
                df_plot = df.copy()
                df_plot['_category'] = 'Normal'
                df_plot.loc[bot_mask, '_category'] = 'Bot'
                df_plot.loc[hub_mask, '_category'] = 'Hub'

                for cat, color in [('Bot', '#e74c3c'), ('Hub', '#3498db'), ('Normal', '#95a5a6')]:
                    data = df_plot[df_plot['_category'] == cat][feature].dropna()
                    if len(data) > 0:
                        # Use log scale for skewed distributions
                        if feature in ['unique_users', 'downloads_per_user', 'total_downloads']:
                            data = np.log10(data + 1)
                            ax.set_xlabel(f'log10({feature})')
                        ax.hist(data, bins=30, alpha=0.5, label=cat, color=color)
            else:
                data = df[feature].dropna()
                if feature in ['unique_users', 'downloads_per_user', 'total_downloads']:
                    data = np.log10(data + 1)
                    ax.set_xlabel(f'log10({feature})')
                ax.hist(data, bins=30, alpha=0.7, color='#3498db')

            ax.set_ylabel('Count')
            ax.set_title(feature.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            if 'automation_category' in df.columns:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return self._save_plot(fig, 'feature_distributions')

    def plot_feature_importance(self, feature_importance: Dict[str, float],
                                title: str = 'Feature Importance',
                                top_n: int = 20) -> Optional[str]:
        """Plot feature importance bar chart."""
        if not MATPLOTLIB_AVAILABLE or not feature_importance:
            return None

        # Sort and get top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        features, importance = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in importance]
        ax.barh(range(len(features)), importance, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.invert_yaxis()

        plt.tight_layout()
        return self._save_plot(fig, 'feature_importance')

    def plot_correlation_matrix(self, df: pd.DataFrame,
                                features: List[str] = None) -> Optional[str]:
        """Plot correlation matrix for key features."""
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            return None

        if features is None:
            features = ['unique_users', 'downloads_per_user', 'working_hours_ratio',
                       'night_activity_ratio', 'hourly_entropy', 'anomaly_score',
                       'regularity_score', 'user_coordination_score']

        # Filter to available numeric features
        features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(features) < 2:
            return None

        corr_matrix = df[features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', ax=ax, square=True,
                   xticklabels=[f.replace('_', '\n') for f in features],
                   yticklabels=[f.replace('_', '\n') for f in features])
        ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return self._save_plot(fig, 'correlation_matrix')

    def plot_category_feature_comparison(self, df: pd.DataFrame,
                                         features: List[str] = None) -> Optional[str]:
        """Plot feature comparison across categories (box plots)."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if features is None:
            features = ['downloads_per_user', 'working_hours_ratio', 'regularity_score']

        features = [f for f in features if f in df.columns]
        if not features:
            return None

        # Create category column
        if 'behavior_type' in df.columns:
            category_col = 'behavior_type'
        elif 'automation_category' in df.columns:
            df = df.copy()
            bot_mask, hub_mask = get_classification_masks(df)
            df['_category'] = 'Normal'
            df.loc[bot_mask, '_category'] = 'Bot'
            df.loc[hub_mask, '_category'] = 'Hub'
            category_col = '_category'
        else:
            return None

        n_features = len(features)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = axes[i]
            categories = df[category_col].unique()
            data_by_cat = [df[df[category_col] == cat][feature].dropna() for cat in categories]

            bp = ax.boxplot(data_by_cat, labels=categories, patch_artist=True)

            colors = [CATEGORY_COLORS.get(str(cat).lower(), '#95a5a6') for cat in categories]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(feature.replace('_', ' ').title())
            ax.set_title(f'{feature.replace("_", " ").title()} by Category',
                        fontsize=10, fontweight='bold')

        plt.tight_layout()
        return self._save_plot(fig, 'category_feature_comparison')

    def plot_geographic_distribution(self, df: pd.DataFrame,
                                     top_n: int = 20) -> Optional[str]:
        """Plot geographic distribution of classifications."""
        if not MATPLOTLIB_AVAILABLE or 'country' not in df.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Top countries by location count
        country_counts = df['country'].value_counts().head(top_n)
        axes[0].barh(range(len(country_counts)), country_counts.values, color='#3498db')
        axes[0].set_yticks(range(len(country_counts)))
        axes[0].set_yticklabels(country_counts.index)
        axes[0].set_xlabel('Number of Locations')
        axes[0].set_title(f'Top {top_n} Countries by Location Count', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()

        # Right: Bot percentage by country
        if 'automation_category' in df.columns:
            df_with_bot = df.copy()
            df_with_bot['_is_bot'] = (df_with_bot['automation_category'] == 'bot').astype(int)
            country_stats = df_with_bot.groupby('country').agg({
                '_is_bot': 'sum',
                'geo_location': 'count'
            }).rename(columns={'geo_location': 'total', '_is_bot': 'bot_count'})
            country_stats['bot_pct'] = country_stats['bot_count'] / country_stats['total'] * 100
            country_stats = country_stats[country_stats['total'] >= 10]  # Min 10 locations
            top_bot_countries = country_stats.nlargest(top_n, 'bot_pct')

            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_bot_countries)))
            axes[1].barh(range(len(top_bot_countries)), top_bot_countries['bot_pct'].values, color=colors)
            axes[1].set_yticks(range(len(top_bot_countries)))
            axes[1].set_yticklabels(top_bot_countries.index)
            axes[1].set_xlabel('Bot Percentage (%)')
            axes[1].set_title(f'Top {top_n} Countries by Bot Percentage', fontsize=11, fontweight='bold')
            axes[1].invert_yaxis()

        plt.tight_layout()
        return self._save_plot(fig, 'geographic_distribution')

    def plot_temporal_patterns(self, df: pd.DataFrame) -> Optional[str]:
        """Plot temporal activity patterns."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Working hours ratio distribution
        if 'working_hours_ratio' in df.columns:
            ax = axes[0, 0]
            bot_mask, _ = get_classification_masks(df)
            if bot_mask.any():
                for cat, color in [('Bot', '#e74c3c'), ('Normal', '#27ae60')]:
                    if cat == 'Bot':
                        data = df[bot_mask]['working_hours_ratio'].dropna()
                    else:
                        data = df[~bot_mask]['working_hours_ratio'].dropna()
                    ax.hist(data, bins=20, alpha=0.6, label=cat, color=color)
                ax.legend()
            else:
                ax.hist(df['working_hours_ratio'].dropna(), bins=20, color='#3498db', alpha=0.7)
            ax.set_xlabel('Working Hours Ratio')
            ax.set_ylabel('Count')
            ax.set_title('Working Hours Activity', fontsize=11, fontweight='bold')

        # Night activity ratio distribution
        if 'night_activity_ratio' in df.columns:
            ax = axes[0, 1]
            bot_mask, _ = get_classification_masks(df)
            if bot_mask.any():
                for cat, color in [('Bot', '#e74c3c'), ('Normal', '#27ae60')]:
                    if cat == 'Bot':
                        data = df[bot_mask]['night_activity_ratio'].dropna()
                    else:
                        data = df[~bot_mask]['night_activity_ratio'].dropna()
                    ax.hist(data, bins=20, alpha=0.6, label=cat, color=color)
                ax.legend()
            else:
                ax.hist(df['night_activity_ratio'].dropna(), bins=20, color='#e74c3c', alpha=0.7)
            ax.set_xlabel('Night Activity Ratio')
            ax.set_ylabel('Count')
            ax.set_title('Night Activity Distribution', fontsize=11, fontweight='bold')

        # Regularity score distribution
        if 'regularity_score' in df.columns:
            ax = axes[1, 0]
            bot_mask, _ = get_classification_masks(df)
            if bot_mask.any():
                for cat, color in [('Bot', '#e74c3c'), ('Normal', '#27ae60')]:
                    if cat == 'Bot':
                        data = df[bot_mask]['regularity_score'].dropna()
                    else:
                        data = df[~bot_mask]['regularity_score'].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=20, alpha=0.6, label=cat, color=color)
                ax.legend()
            else:
                ax.hist(df['regularity_score'].dropna(), bins=20, color='#9b59b6', alpha=0.7)
            ax.set_xlabel('Regularity Score')
            ax.set_ylabel('Count')
            ax.set_title('Download Regularity', fontsize=11, fontweight='bold')

        # Hourly entropy distribution
        if 'hourly_entropy' in df.columns:
            ax = axes[1, 1]
            ax.hist(df['hourly_entropy'].dropna(), bins=20, color='#f39c12', alpha=0.7)
            ax.set_xlabel('Hourly Entropy')
            ax.set_ylabel('Count')
            ax.set_title('Hourly Activity Entropy', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return self._save_plot(fig, 'temporal_patterns')

    def plot_anomaly_analysis(self, df: pd.DataFrame) -> Optional[str]:
        """Plot anomaly score analysis."""
        if not MATPLOTLIB_AVAILABLE or 'anomaly_score' not in df.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Anomaly score distribution
        ax = axes[0]
        bot_mask, hub_mask = get_classification_masks(df)
        if bot_mask.any() or hub_mask.any():
            for cat, color in [('Bot', '#e74c3c'), ('Hub', '#3498db'), ('Normal', '#95a5a6')]:
                if cat == 'Bot':
                    data = df[bot_mask]['anomaly_score'].dropna()
                elif cat == 'Hub':
                    data = df[hub_mask]['anomaly_score'].dropna()
                else:
                    normal_mask = ~bot_mask & ~hub_mask
                    data = df[normal_mask]['anomaly_score'].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.6, label=cat, color=color)
            ax.legend()
        else:
            ax.hist(df['anomaly_score'].dropna(), bins=30, color='#e74c3c', alpha=0.7)

        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Count')
        ax.set_title('Anomaly Score Distribution', fontsize=11, fontweight='bold')

        # Anomaly score vs downloads per user scatter
        ax = axes[1]
        if 'downloads_per_user' in df.columns:
            sample = df.sample(min(5000, len(df)))  # Limit points for performance
            sample_bot_mask, sample_hub_mask = get_classification_masks(sample)
            colors = ['#e74c3c' if sample_bot_mask.loc[idx] else
                     '#3498db' if sample_hub_mask.loc[idx] else '#95a5a6'
                     for idx in sample.index]
            ax.scatter(sample['downloads_per_user'], sample['anomaly_score'],
                      c=colors, alpha=0.3, s=10)
            ax.set_xscale('log')
            ax.set_xlabel('Downloads per User (log)')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Score vs Downloads/User', fontsize=11, fontweight='bold')

            # Add legend
            legend_elements = [
                mpatches.Patch(facecolor='#e74c3c', label='Bot', alpha=0.6),
                mpatches.Patch(facecolor='#3498db', label='Hub', alpha=0.6),
                mpatches.Patch(facecolor='#95a5a6', label='Normal', alpha=0.6),
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return self._save_plot(fig, 'anomaly_analysis')

    def generate_all_plots(self, df: pd.DataFrame,
                          classification_method: str = 'rules',
                          feature_importance: Dict[str, float] = None) -> List[str]:
        """Generate all available plots and return list of paths."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping visualizations")
            return []

        logger.info("Generating visualizations...")

        plots = []

        # Classification distribution
        plot = self.plot_classification_distribution(df, classification_method)
        if plot:
            plots.append(plot)

        # Downloads by category
        plot = self.plot_downloads_by_category(df, classification_method)
        if plot:
            plots.append(plot)

        # Yearly trends
        plot = self.plot_yearly_trends(df)
        if plot:
            plots.append(plot)

        # Feature distributions
        plot = self.plot_feature_distributions(df)
        if plot:
            plots.append(plot)

        # Feature importance
        if feature_importance:
            plot = self.plot_feature_importance(feature_importance)
            if plot:
                plots.append(plot)

        # Correlation matrix
        plot = self.plot_correlation_matrix(df)
        if plot:
            plots.append(plot)

        # Category feature comparison
        plot = self.plot_category_feature_comparison(df)
        if plot:
            plots.append(plot)

        # Geographic distribution
        plot = self.plot_geographic_distribution(df)
        if plot:
            plots.append(plot)

        # Temporal patterns
        plot = self.plot_temporal_patterns(df)
        if plot:
            plots.append(plot)

        # Anomaly analysis
        plot = self.plot_anomaly_analysis(df)
        if plot:
            plots.append(plot)

        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
