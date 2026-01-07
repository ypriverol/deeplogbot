# Detailed Classification Categories

This document describes the new detailed classification categories that have been added to LogGhostBuster to provide more granular insights into download patterns.

## Overview

In addition to the main classification categories (Bot, Download Hub, Independent User, Normal, Other), the system now supports four new detailed categories that help identify specific usage patterns:

1. **CI/CD Pipeline** - Automated testing and build systems
2. **Research Group** - Small teams of researchers  
3. **Bulk Downloader** - Individual heavy downloaders
4. **Course/Workshop** - Educational events

These categories are applied **after** the main classification and only reclassify locations that are marked as 'normal', 'other', or 'unclassified'. Bot, hub, and independent_user classifications are protected and will not be overridden.

## Category Definitions

### 1. CI/CD Pipeline

Automated testing/build systems that regularly download the same files.

**Characteristics:**
- Users: Few (1-10)
- DL/user: Moderate-High (50-500)
- Temporal pattern: Very regular intervals (daily/weekly)
- File diversity: Low (same files repeatedly)
- Working hours: 24/7 or specific cron times

**Detection Rules:**
- `max_users`: 10
- `min_downloads_per_user`: 50
- `max_downloads_per_user`: 500
- `max_file_diversity_ratio`: 0.3
- `min_regularity_score`: 0.8

### 2. Research Group / Lab

Small teams of researchers with moderate download activity during working hours.

**Characteristics:**
- Users: Small group (5-50)
- DL/user: Moderate (10-100)
- Temporal pattern: Working hours biased
- File diversity: Moderate-High

**Detection Rules:**
- `min_users`: 5
- `max_users`: 50
- `min_downloads_per_user`: 10
- `max_downloads_per_user`: 100
- `min_working_hours_ratio`: 0.5
- `min_file_diversity_ratio`: 0.3

### 3. Bulk Downloader

Individual users or very small groups downloading large amounts of data.

**Characteristics:**
- Users: Very few (1-5)
- DL/user: Very high (100-1000)
- Temporal pattern: Concentrated bursts
- Session pattern: Long sessions, many files

**Detection Rules:**
- `max_users`: 5
- `min_downloads_per_user`: 100
- `max_downloads_per_user`: 1000

### 4. Course / Workshop

Educational events where many users download the same tutorial materials.

**Characteristics:**
- Users: Many (50-500)
- DL/user: Low-Moderate (5-20)
- Temporal pattern: Concentrated in short time period
- File diversity: Low (same tutorial files)

**Detection Rules:**
- `min_users`: 50
- `max_users`: 500
- `min_downloads_per_user`: 5
- `max_downloads_per_user`: 20
- `max_file_diversity_ratio`: 0.3

## Configuration

The category detection rules are defined in `config.yaml` under `classification.categories`:

```yaml
classification:
  categories:
    ci_cd_pipeline:
      max_users: 10
      min_downloads_per_user: 50
      max_downloads_per_user: 500
      max_file_diversity_ratio: 0.3
      min_regularity_score: 0.8
      
    research_group:
      min_users: 5
      max_users: 50
      min_downloads_per_user: 10
      max_downloads_per_user: 100
      min_working_hours_ratio: 0.5
      min_file_diversity_ratio: 0.3
      
    bulk_downloader:
      max_users: 5
      min_downloads_per_user: 100
      max_downloads_per_user: 1000
      
    course_workshop:
      min_users: 50
      max_users: 500
      min_downloads_per_user: 5
      max_downloads_per_user: 20
      max_file_diversity_ratio: 0.3
```

## Usage

The detailed categories are automatically applied when running classification with the deep architecture method:

```python
from logghostbuster.models.classification.deep_architecture import classify_locations_deep

# Run classification (includes detailed categories)
df, cluster_df = classify_locations_deep(
    df, 
    feature_columns,
    use_transformer=True,
    # ... other parameters
)

# Access detailed categories
print(df[['geo_location', 'user_category', 'detailed_category']])
```

The `detailed_category` column will contain one of:
- Original categories: `'bot'`, `'download_hub'`, `'independent_user'` (protected)
- New categories: `'ci_cd_pipeline'`, `'research_group'`, `'bulk_downloader'`, `'course_workshop'`
- Fallback: `'normal'`, `'other'`, `'unclassified'`

## Output

When running classification, you'll see logging output like:

```
    Classifying detailed categories...
Detailed category classification:
  ci_cd_pipeline: 234 locations
  research_group: 1,456 locations
  bulk_downloader: 89 locations
  course_workshop: 67 locations
```

## Benefits

1. **Better understanding** of legitimate traffic patterns
2. **Reduced false positives** - CI/CD won't be flagged as suspicious
3. **Actionable insights** - Can contact course organizers, optimize for research groups
4. **Granular reporting** - More detailed breakdown of download patterns

## Implementation Details

The detailed category classification is implemented in three files:

1. **config.yaml** - Defines the detection rules for each category
2. **config.py** - Provides `get_category_rules()` helper function
3. **deep_architecture.py** - Contains `_classify_detailed_categories()` function

The classification runs after the final safety check in `classify_locations_deep()` and before logging results. It uses a priority system where earlier categories take precedence (CI/CD → Research → Bulk → Course).

## Customization

You can customize the detection rules by modifying `config.yaml`. For example, to make CI/CD detection more strict:

```yaml
ci_cd_pipeline:
  max_users: 5              # Reduce from 10
  min_regularity_score: 0.9  # Increase from 0.8
```

Or to adjust research group thresholds:

```yaml
research_group:
  min_users: 10             # Increase from 5
  max_users: 100            # Increase from 50
```

## Notes

- The detailed categories require behavioral features to work optimally (file_diversity_ratio, regularity_score, working_hours_ratio)
- If behavioral features are missing, the system will still classify based on available features
- Bot, hub, and independent_user classifications are always protected and never overridden
- Classification is applied in priority order to prevent overlapping categories
