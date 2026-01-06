# Deep Classification Visualization Module

This module provides comprehensive visualization tools for analyzing deep classification results.

## Features

- **Geographic Clusters**: Plot locations on a world map colored by classification category
- **Feature Space Visualization**: 2D projections using t-SNE (default), PCA, or UMAP to show clusters in feature space
- **Category Distribution**: Bar charts and pie charts showing distribution of categories
- **Unclassified Locations**: Detailed analysis of locations that remain unclassified (category='other')
- **Category Features**: Box plots comparing key features across categories

## Usage

### Command Line

```bash
python -m logghostbuster.visualization.deep_results \
    --input output/deep_phase2_5M/location_analysis.csv \
    --output-dir output/deep_phase2_5M \
    --prefix deep_phase2
```

### Python API

```python
from logghostbuster.visualization import generate_all_visualizations

# Generate all visualizations
generate_all_visualizations(
    csv_path='output/deep_phase2_5M/location_analysis.csv',
    output_dir='output/deep_phase2_5M',
    prefix='deep_phase2'
)

# Or use individual functions
from logghostbuster.visualization import (
    load_location_analysis,
    plot_geographic_clusters,
    plot_feature_space,
    plot_category_distribution,
    plot_unclassified_locations,
    plot_category_features
)

df = load_location_analysis('output/deep_phase2_5M/location_analysis.csv')
plot_geographic_clusters(df, output_path='geographic.png')
plot_feature_space(df, output_path='featurespace_tsne.png', method='tsne')  # Default
plot_feature_space(df, output_path='featurespace_pca.png', method='pca')    # Faster
plot_feature_space(df, output_path='featurespace_umap.png', method='umap')  # Alternative
plot_category_distribution(df, output_path='categories.png')
plot_unclassified_locations(df, output_path='unclassified.png')
plot_category_features(df, output_path='features.png')
```

## Output Files

The module generates the following visualization files:

1. `{prefix}_geographic.png` - Geographic distribution of all locations
2. `{prefix}_featurespace_tsne.png` - 2D t-SNE projection (default, best for visualization)
3. `{prefix}_featurespace_pca.png` - 2D PCA projection (faster alternative)
4. `{prefix}_featurespace_umap.png` - 2D UMAP projection (alternative visualization method)
5. `{prefix}_categories.png` - Category distribution (bar chart + pie chart)
6. `{prefix}_unclassified.png` - Analysis of unclassified locations
7. `{prefix}_features.png` - Feature comparison across categories

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (for PCA/t-SNE)
- umap-learn (for UMAP - recommended for better visualizations)

## Category Colors

- **Bot**: Red (#FF0000)
- **Download Hub**: Orange (#FFA500)
- **Independent User**: Green (#00FF00)
- **Normal**: Blue (#0000FF)
- **Other/Unclassified**: Gray (#808080)

