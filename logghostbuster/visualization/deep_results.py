"""Visualization module for deep classification results.

This module provides functions to visualize classification results from the deep
architecture, including geographic clusters, feature space projections, and
category distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..utils.geography import parse_geo_location
from ..utils import logger

if not MATPLOTLIB_AVAILABLE:
    logger.warning("matplotlib/seaborn not available. Visualization functions will not work.")


def load_location_analysis(csv_path: str) -> pd.DataFrame:
    """Load location analysis CSV file."""
    logger.info(f"Loading location analysis from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Parse coordinates
    coords = df['geo_location'].apply(parse_geo_location)
    df['latitude'] = coords.apply(lambda x: x[0] if x[0] is not None else np.nan)
    df['longitude'] = coords.apply(lambda x: x[1] if x[1] is not None else np.nan)
    
    return df


def plot_geographic_clusters(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    show_unclassified: bool = True,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Plot geographic clusters colored by classification category.
    
    Args:
        df: DataFrame with location analysis results
        output_path: Optional path to save the figure
        show_unclassified: Whether to highlight unclassified locations
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Cannot create plots.")
        return
    
    logger.info("Creating geographic cluster visualization...")
    
    # Filter locations with valid coordinates
    df_coords = df[df['latitude'].notna() & df['longitude'].notna()].copy()
    
    if len(df_coords) == 0:
        logger.warning("No locations with valid coordinates found. Skipping geographic plot.")
        return
    
    # Category colors
    category_colors = {
        'bot': '#FF0000',  # Red
        'download_hub': '#FFA500',  # Orange
        'independent_user': '#00FF00',  # Green
        'normal': '#0000FF',  # Blue
        'other': '#808080'  # Gray (unclassified)
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each category
    for category in ['normal', 'independent_user', 'other', 'download_hub', 'bot']:
        if category not in df_coords['user_category'].values:
            continue
            
        cat_data = df_coords[df_coords['user_category'] == category]
        if len(cat_data) == 0:
            continue
        
        # Use size based on total_downloads (log scale)
        sizes = np.log10(cat_data['total_downloads'].clip(lower=1) + 1) * 10
        
        ax.scatter(
            cat_data['longitude'],
            cat_data['latitude'],
            c=category_colors.get(category, '#000000'),
            label=f"{category.replace('_', ' ').title()} ({len(cat_data)})",
            s=sizes,
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Geographic Distribution of Classified Locations', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Geographic plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_space(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    method: str = 'tsne',
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Plot locations in 2D feature space using dimensionality reduction.
    
    Args:
        df: DataFrame with location analysis results
        output_path: Optional path to save the figure
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        features: List of feature columns to use (default: key features)
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Cannot create plots.")
        return
    
    logger.info(f"Creating feature space visualization using {method.upper()}...")
    
    # Default features if not specified
    if features is None:
        features = [
            'downloads_per_user',
            'unique_users',
            'anomaly_score',
            'working_hours_ratio',
            'hourly_entropy',
            'yearly_entropy',
            'spike_ratio',
            'fraction_latest_year'
        ]
    
    # Filter features that exist in dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < 2:
        logger.warning(f"Not enough features available. Found: {available_features}")
        return
    
    # Prepare data
    X = df[available_features].fillna(0).values
    
    # Apply dimensionality reduction
    try:
        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            xlabel = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)'
            ylabel = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)'
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            logger.info(f"    Computing t-SNE (this may take a while for {len(X)} samples)...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
            X_reduced = reducer.fit_transform(X)
            xlabel = 't-SNE Component 1'
            ylabel = 't-SNE Component 2'
        elif method.lower() == 'umap':
            try:
                import umap
                logger.info(f"    Computing UMAP (this may take a while for {len(X)} samples)...")
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) - 1))
                X_reduced = reducer.fit_transform(X)
                xlabel = 'UMAP Component 1'
                ylabel = 'UMAP Component 2'
            except ImportError:
                logger.warning("umap-learn not installed. Install with: pip install umap-learn")
                logger.info("    Falling back to PCA...")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                xlabel = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)'
                ylabel = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)'
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'")
    except ImportError:
        logger.warning("sklearn not available. Skipping feature space plot.")
        return
    
    # Category colors
    category_colors = {
        'bot': '#FF0000',
        'download_hub': '#FFA500',
        'independent_user': '#00FF00',
        'normal': '#0000FF',
        'other': '#808080'
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each category
    for category in ['normal', 'independent_user', 'other', 'download_hub', 'bot']:
        if category not in df['user_category'].values:
            continue
        
        cat_mask = df['user_category'] == category
        cat_data = X_reduced[cat_mask]
        
        if len(cat_data) == 0:
            continue
        
        ax.scatter(
            cat_data[:, 0],
            cat_data[:, 1],
            c=category_colors.get(category, '#000000'),
            label=f"{category.replace('_', ' ').title()} ({cat_mask.sum()})",
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidths=0.3
        )
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Feature Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature space plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_category_distribution(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plot distribution of categories.
    
    Args:
        df: DataFrame with location analysis results
        output_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Cannot create plots.")
        return
    
    logger.info("Creating category distribution visualization...")
    
    category_counts = df['user_category'].value_counts()
    category_colors = {
        'bot': '#FF0000',
        'download_hub': '#FFA500',
        'independent_user': '#00FF00',
        'normal': '#0000FF',
        'other': '#808080'
    }
    
    colors = [category_colors.get(cat, '#000000') for cat in category_counts.index]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(
        range(len(category_counts)),
        category_counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.5
    )
    ax1.set_xticks(range(len(category_counts)))
    ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in category_counts.index], rotation=45, ha='right')
    ax1.set_ylabel('Number of Locations', fontsize=12)
    ax1.set_title('Location Count by Category', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Pie chart
    ax2.pie(
        category_counts.values,
        labels=[cat.replace('_', ' ').title() for cat in category_counts.index],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10}
    )
    ax2.set_title('Category Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Category distribution plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_unclassified_locations(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """Plot unclassified locations (category='other') with their features.
    
    Args:
        df: DataFrame with location analysis results
        output_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Cannot create plots.")
        return
    
    logger.info("Creating unclassified locations visualization...")
    
    unclassified = df[df['user_category'] == 'other'].copy()
    
    if len(unclassified) == 0:
        logger.info("No unclassified locations found.")
        return
    
    logger.info(f"Found {len(unclassified)} unclassified locations")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Geographic distribution
    if unclassified['latitude'].notna().any():
        ax = axes[0]
        unclassified_coords = unclassified[unclassified['latitude'].notna()]
        ax.scatter(
            unclassified_coords['longitude'],
            unclassified_coords['latitude'],
            c='red',
            alpha=0.6,
            s=np.log10(unclassified_coords['total_downloads'].clip(lower=1) + 1) * 10,
            edgecolors='black',
            linewidths=0.5
        )
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(f'Unclassified Locations ({len(unclassified_coords)})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 2. Downloads per user distribution
    ax = axes[1]
    ax.hist(unclassified['downloads_per_user'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel('Downloads per User', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Downloads per User Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Unique users distribution
    ax = axes[2]
    ax.hist(np.log10(unclassified['unique_users'].clip(lower=1) + 1), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.set_xlabel('Log10(Unique Users)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Unique Users Distribution (log scale)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Anomaly score distribution
    ax = axes[3]
    ax.hist(unclassified['anomaly_score'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Anomaly Score', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Unclassified locations plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_category_features(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """Plot key features by category.
    
    Args:
        df: DataFrame with location analysis results
        output_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib/seaborn not available. Cannot create plots.")
        return
    
    logger.info("Creating category features visualization...")
    
    features_to_plot = [
        'downloads_per_user',
        'unique_users',
        'anomaly_score',
        'working_hours_ratio'
    ]
    
    available_features = [f for f in features_to_plot if f in df.columns]
    
    n_features = len(available_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    category_colors = {
        'bot': '#FF0000',
        'download_hub': '#FFA500',
        'independent_user': '#00FF00',
        'normal': '#0000FF',
        'other': '#808080'
    }
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        
        # Create box plots for each category
        categories = df['user_category'].unique()
        data_to_plot = [df[df['user_category'] == cat][feature].dropna() for cat in categories]
        colors_to_plot = [category_colors.get(cat, '#000000') for cat in categories]
        
        bp = ax.boxplot(
            data_to_plot,
            labels=[cat.replace('_', ' ').title() for cat in categories],
            patch_artist=True,
            showfliers=False  # Hide outliers for cleaner plot
        )
        
        for patch, color in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{feature.replace("_", " ").title()} by Category', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Category features plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_visualizations(
    csv_path: str,
    output_dir: Optional[str] = None,
    prefix: str = 'deep_visualization'
) -> None:
    """Generate all visualizations for deep classification results.
    
    Args:
        csv_path: Path to location_analysis.csv file
        output_dir: Output directory for plots (default: same as CSV directory)
        prefix: Prefix for output filenames
    """
    logger.info("=" * 80)
    logger.info("Generating Deep Classification Visualizations")
    logger.info("=" * 80)
    
    # Load data
    df = load_location_analysis(csv_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = str(Path(csv_path).parent)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    logger.info(f"\nGenerating visualizations in: {output_dir}")
    
    # 1. Geographic clusters
    try:
        plot_geographic_clusters(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_geographic.png')
        )
    except Exception as e:
        logger.warning(f"Failed to create geographic plot: {e}")
    
    # 2. Feature space (t-SNE) - default, best for visualization
    try:
        plot_feature_space(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_featurespace_tsne.png'),
            method='tsne'
        )
    except Exception as e:
        logger.warning(f"Failed to create t-SNE plot: {e}")
    
    # 3. Feature space (PCA) - faster alternative
    try:
        plot_feature_space(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_featurespace_pca.png'),
            method='pca'
        )
    except Exception as e:
        logger.warning(f"Failed to create PCA plot: {e}")
    
    # 4. Feature space (UMAP) - alternative visualization method
    try:
        plot_feature_space(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_featurespace_umap.png'),
            method='umap'
        )
    except Exception as e:
        logger.warning(f"Failed to create UMAP plot: {e}")
    
    # 5. Category distribution
    try:
        plot_category_distribution(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_categories.png')
        )
    except Exception as e:
        logger.warning(f"Failed to create category distribution plot: {e}")
    
    # 5. Unclassified locations
    try:
        plot_unclassified_locations(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_unclassified.png')
        )
    except Exception as e:
        logger.warning(f"Failed to create unclassified locations plot: {e}")
    
    # 6. Category features
    try:
        plot_category_features(
            df,
            output_path=str(Path(output_dir) / f'{prefix}_features.png')
        )
    except Exception as e:
        logger.warning(f"Failed to create category features plot: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Visualization generation complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate visualizations for deep classification results'
    )
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Path to location_analysis.csv file'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default=None,
        help='Output directory for plots (default: same as input directory)'
    )
    parser.add_argument(
        '--prefix',
        '-p',
        default='deep_visualization',
        help='Prefix for output filenames'
    )
    
    args = parser.parse_args()
    
    generate_all_visualizations(
        csv_path=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

