# LogGhostbuster

LogGhostbuster: A hybrid Isolation Forest-based system for detecting bot behavior in log data.

## Overview

LogGhostbuster is a hybrid machine learning system that detects bot behavior and download hubs in log data. It combines unsupervised anomaly detection with rule-based classification to identify suspicious download patterns.

### Algorithm Overview

The system follows a multi-stage pipeline:

1. **Feature Extraction**: Extracts location-level behavioral features from log data:
   - User activity patterns (unique users, downloads per user, user density per hour)
   - Temporal patterns (hourly entropy, working hours ratio, yearly patterns)
   - Anomaly indicators (spike ratios, latest year concentration, new location flags)
   - Geographic patterns (country-level aggregations for coordinated detection)

2. **Anomaly Detection**: Uses **Isolation Forest** to identify anomalous locations:
   - Isolation Forest is an unsupervised algorithm that isolates anomalies by randomly selecting features and split values
   - Locations with unusual behavioral patterns (short path lengths in the isolation tree) are flagged as anomalies
   - The contamination parameter controls the expected proportion of anomalies (default: 15%)

3. **Classification**: Classifies anomalies into categories using either rule-based or ML-based methods:
   
   **Rule-based classification** (default): Uses a comprehensive set of pattern-based rules:
   - **BOT**: Detected when anomalies exhibit bot-like characteristics:
     - Low downloads per user (< 100) combined with high user counts (> 5K-30K users)
     - Sudden spikes in activity (high spike ratios and latest year concentration)
     - New locations with suspicious patterns
     - Geographic bot farms (coordinated patterns across country-level metrics)
   
   - **DOWNLOAD_HUB**: Detected when anomalies show hub-like characteristics:
     - Very high downloads per user (> 500) - mirrors/single-user hubs
     - High total downloads (> 150K) with moderate downloads per user (50-500) and regular working hours patterns - research institutions
   
   **Supervised ML-based classification** (optional): Uses a RandomForestClassifier trained on rule-based labels:
   - First generates training labels using rule-based classification
   - Trains a multi-class classifier (BOT, DOWNLOAD_HUB, NORMAL) on behavioral features
   - Can generalize better to edge cases and learn complex feature interactions
   - Provides feature importance rankings for interpretability
   - Use `--classification-method ml` to enable
   
   **Unsupervised ML-based classification** (optional): Uses KMeans clustering:
   - Clusters locations into 3 groups based on behavioral features
   - Maps clusters to bot/hub/normal based on cluster characteristics
   - No labels required - fully unsupervised approach
   - Useful when rule-based patterns may not capture all patterns
   - Use `--classification-method ml-unsupervised` to enable

4. **Geographic Grouping** (optional): Groups nearby hub locations using:
   - Haversine distance calculation (default: 10km threshold)
   - Optional LLM-based canonical naming for institutions

5. **Output Generation**: Produces annotated data and comprehensive reports with detection results.

### Detection Targets

- **Bot downloads** - Automated downloads from bot farms (characterized by user ID cycling with low downloads per user)
- **Download hubs** - Legitimate mirrors/institutions with high download volumes (characterized by high downloads per user or regular institutional patterns)

## Installation

```bash
pip install -e .
```

Or with optional LLM dependencies for location grouping:
```bash
pip install -e ".[llm]"
```

## Usage

### Command Line

```bash
logghostbuster --input data_downloads_parquet.parquet --output-dir output/bot_analysis
```

Options:
- `--input, -i`: Input parquet file (default: `original_data/data_downloads_parquet.parquet`)
- `--output, -out`: Output parquet file (default: overwrites input)
- `--output-dir, -o`: Output directory for reports (default: `output/bot_analysis`)
- `--contamination, -c`: Expected proportion of anomalies (default: 0.15)
- `--compute-importances`: Compute feature importances (optional, slower)
- `--sample-size, -s`: Randomly sample N records from all years before processing (e.g., 1000000 for 1M records)
- `--classification-method, -m`: Classification method - `rules` for rule-based (default), `ml` for supervised ML-based, or `ml-unsupervised` for unsupervised ML-based

### Known Issues

- **Deep Classification with Large Sample Sizes**: When using the `--classification-method deep`, the algorithm encounters an issue and may get stuck during the data loading and feature extraction phase (specifically within the `TimeWindowExtractor`) when a `--sample-size` of 50,000,000 (50M) records is specified. Interestingly, the process completes successfully for smaller sample sizes (e.g., 5M records) and for the entire dataset without any sampling. This suggests a specific bottleneck at the 50M sample size. This issue is currently under investigation. For large datasets, consider using no `--sample-size` or a smaller `sample-size` (e.g., 5M) until this is resolved.

### Python API

```python
from logghostbuster import run_bot_annotator

results = run_bot_annotator(
    input_parquet='data_downloads_parquet.parquet',
    output_parquet='annotated_data.parquet',
    output_dir='output/bot_analysis',
    contamination=0.15,
    compute_importances=False,
    sample_size=1000000,  # Optional: sample 1M records
    classification_method='ml'  # 'rules' (default), 'ml' (supervised), or 'ml-unsupervised'
)
```

## Package Structure

- `logghostbuster/`
  - `__init__.py` - Package initialization and exports
  - `main.py` - Main bot detection pipeline and CLI
  - `utils/` - Utility functions package
    - `__init__.py` - General utilities (logging, formatting) and exports
    - `geography.py` - Geographic utility functions (haversine distance, coordinate parsing, location grouping)
  - `features/` - Feature extraction package
    - `__init__.py` - Feature extraction exports
    - `schema.py` - Schema definitions for different log formats
    - `base.py` - Base feature extractor class
    - `extraction.py` - Main feature extraction function
    - `standard.py` - Standard extractors (yearly, time-of-day, country-level)
    - `providers/` - Provider-specific extractors
      - `ebi.py` - EBI-specific extractors (if needed)
  - `isoforest/` - Isolation Forest model package
    - `__init__.py` - Model exports
    - `models.py` - Isolation Forest training and feature importance
    - `classification.py` - Bot and download hub classification logic
  - `llm/` - LLM utilities package
    - `__init__.py` - LLM exports
    - `utils.py` - LLM utilities for canonical naming (optional)
  - `reports/` - Report generation and annotation package
    - `__init__.py` - Report exports
    - `annotation.py` - Annotation utilities for marking locations with bot/download_hub flags
    - `reporting.py` - Generic report generator

## Custom Schemas and Feature Extractors

The tool supports different log formats through schema definitions and extensible feature extractors.

### Using Custom Schemas

```python
from logghostbuster import LogSchema, run_bot_annotator

# Define a custom schema for your log format
custom_schema = LogSchema(
    location_field="ip_coordinates",
    country_field="country_code",
    user_field="user_id",
    timestamp_field="event_time",
    # ... other field mappings
)

# Use it with the pipeline
results = run_bot_annotator(
    input_parquet='your_logs.parquet',
    schema=custom_schema,
)
```

### Creating Custom Feature Extractors

```python
from logghostbuster import BaseFeatureExtractor
import pandas as pd

class MyCustomExtractor(BaseFeatureExtractor):
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        # Add your custom features here
        df['my_custom_feature'] = ...  # Your calculation
        return df

# Use custom extractors
results = run_bot_annotator(
    input_parquet='your_logs.parquet',
    custom_extractors=[MyCustomExtractor(schema)],
)
```

See `examples/custom_schema_example.py` for more detailed examples.

## Detection Methodology

1. Extract location-level features (users, downloads, patterns, time-of-day)
2. Apply Isolation Forest anomaly detection
3. Classify anomalies as:
   - **BOT**: low downloads/user + many users
   - **DOWNLOAD_HUB**: high downloads/user (mirrors) or high total downloads with regular patterns (research institutions)
4. Group nearby hub locations using geographic distance + optional LLM consolidation

## Environment Variables

- `USE_LLM_GROUPING`: Enable/disable LLM-based location grouping (default: `true`)
- `HF_MODEL`: Hugging Face model name (default: `microsoft/DialoGPT-medium`)

## Comparing Classification Methods

A comparison script is provided to evaluate all three classification methods on the same dataset:

```bash
python compare_classification_methods.py \
    --input data_downloads_parquet.parquet \
    --sample-size 3000000 \
    --output-dir output/comparison
```

This will:
- Run all three methods (rules, supervised ML, unsupervised ML) on the same sampled data
- Generate comparison reports showing differences between methods
- Create CSV files with detailed comparison metrics
- Save individual results for each method

## Output

The tool generates:
- Annotated parquet file with `bot` and `download_hub` columns
- `bot_detection_report.txt` - Comprehensive detection report
- `location_analysis.csv` - Full analysis of all locations
- `feature_importances/` - Feature importance analysis (if enabled)
