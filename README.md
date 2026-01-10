# LogGhostbuster

LogGhostbuster: A hybrid machine learning system for detecting bot behavior and download patterns in log data.

## Overview

LogGhostbuster is a comprehensive machine learning system that detects bot behavior, download hubs, and various download patterns in log data. It offers multiple classification approaches: rule-based classification, supervised/unsupervised ML classification, and a novel pattern discovery architecture that uses deep learning to discover behavioral patterns automatically.

### Algorithm Overview

The system follows a multi-stage pipeline with multiple classification approaches:

1. **Feature Extraction**: Extracts location-level behavioral features from log data:
   - User activity patterns (unique users, downloads per user, user density per hour)
   - Temporal patterns (hourly entropy, working hours ratio, yearly patterns)
   - Anomaly indicators (spike ratios, latest year concentration, new location flags)
   - Geographic patterns (country-level aggregations for coordinated detection)
   - Behavioral features (temporal regularity, file diversity, session patterns) - for pattern discovery

2. **Anomaly Detection** (for rule-based/ML methods): Uses **Isolation Forest** to identify anomalous locations:
   - Isolation Forest is an unsupervised algorithm that isolates anomalies by randomly selecting features and split values
   - Locations with unusual behavioral patterns (short path lengths in the isolation tree) are flagged as anomalies
   - The contamination parameter controls the expected proportion of anomalies (default: 15%)

3. **Classification**: Two classification methods available:

   **Rule-based classification** (`--classification-method rules`, default): Uses a comprehensive set of pattern-based rules:
   - Simple, fast classification using YAML-configurable rules
   - **BOT**: Detected when anomalies exhibit bot-like characteristics:
     - Low downloads per user (< 100) combined with high user counts (> 5K-30K users)
     - Sudden spikes in activity (high spike ratios and latest year concentration)
     - New locations with suspicious patterns
   - **DOWNLOAD_HUB**: Detected when anomalies show hub-like characteristics:
     - Very high downloads per user (> 500) - mirrors/single-user hubs
     - High total downloads with regular institutional patterns
   - Output: Hierarchical columns (`behavior_type`, `automation_category`, `subcategory`)

   **Deep architecture classification** (`--classification-method deep`): Advanced deep learning approach with hierarchical classification:
   - Combines Isolation Forest anomaly detection with Transformer-based pattern discovery
   - **Hierarchical Classification**:
     - **Level 1 (behavior_type)**: ORGANIC vs AUTOMATED
     - **Level 2 (automation_category)**: BOT vs LEGITIMATE_AUTOMATION (for automated only)
     - **Level 3 (subcategory)**: Detailed classification (mirror, ci_cd_pipeline, scraper_bot, etc.)
   - **Advanced Features**: Extracts behavioral features including burst patterns, circadian rhythms, user coordination
   - **Pattern Discovery**: Uses clustering to discover natural behavioral patterns
   - Output: Hierarchical columns (`behavior_type`, `automation_category`, `subcategory`)

4. **Geographic Grouping** (optional): Groups nearby hub locations using:
   - Haversine distance calculation (default: 10km threshold)
   - Geographic center-based canonical naming for institutions

5. **Output Generation**: Produces annotated data and comprehensive reports with detection results.

### Detection Targets

- **Bot downloads** - Automated downloads from bot farms (characterized by user ID cycling with low downloads per user)
- **Download hubs** - Legitimate mirrors/institutions with high download volumes (characterized by high downloads per user or regular institutional patterns)
- **Hierarchical Classification** (deep method) - Three-level taxonomy:
  - **ORGANIC**: Human-like download patterns (individual users, research groups)
  - **AUTOMATED > BOT**: Suspicious automation (scrapers, crawlers, coordinated bots)
  - **AUTOMATED > LEGITIMATE_AUTOMATION**: Benign automation (mirrors, CI/CD, institutional hubs)

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
- `--classification-method, -m`: Classification method:
  - `rules` - Rule-based classification (default) - fast, hierarchical classification
  - `deep` - Deep architecture with hierarchical classification (ORGANIC/AUTOMATED taxonomy)
- `--provider, -p`: Log provider for configuration and rules (default: ebi). Use `--list-providers` to see available options.
- `--list-providers`: List available log providers and exit

### Known Issues

- **Deep Classification with Large Sample Sizes**: When using the `--classification-method deep`, the algorithm encounters an issue and may get stuck during the data loading and feature extraction phase (specifically within the `TimeWindowExtractor`) when a `--sample-size` of 50,000,000 (50M) records is specified. Interestingly, the process completes successfully for smaller sample sizes (e.g., 5M records) and for the entire dataset without any sampling. This suggests a specific bottleneck at the 50M sample size. This issue is currently under investigation. For large datasets, consider using no `--sample-size` or a smaller `sample-size` (e.g., 5M) until this is resolved.

### Python API

```python
from logghostbuster import run_bot_annotator

# Using rules method (simple, fast)
results = run_bot_annotator(
    input_parquet='data_downloads_parquet.parquet',
    output_dir='output/bot_analysis',
    classification_method='rules'  # Default - hierarchical classification
)

# Using deep method (hierarchical classification)
results = run_bot_annotator(
    input_parquet='data_downloads_parquet.parquet',
    output_dir='output/bot_analysis',
    classification_method='deep',  # Hierarchical: behavior_type, automation_category, subcategory
    sample_size=1000000,  # Optional: sample 1M records
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
  - `models/` - Machine learning models package
    - `classification/` - Classification models
      - `deep_architecture.py` - Deep learning architecture for bot detection
      - `pattern_discovery.py` - Pattern discovery using contrastive learning and clustering
  - `llm/` - LLM utilities package
    - `__init__.py` - LLM exports
    - `utils.py` - LLM utilities for canonical naming (optional)
  - `reports/` - Report generation and annotation package
    - `__init__.py` - Report exports
    - `annotation.py` - Annotation utilities for marking locations with bot/download_hub flags
    - `reporting.py` - Generic report generator

## Provider System

LogGhostbuster uses a provider-based architecture to support different log sources. Each provider defines its own schema, classification thresholds, and taxonomy.

### Using Different Providers

```bash
# List available providers
logghostbuster --list-providers

# Use a specific provider (default: ebi)
logghostbuster -i data.parquet -o output/ --provider ebi
```

### Creating Custom Providers

To support a new log source, create a provider directory with a `config.yaml`:

```
logghostbuster/providers/my_provider/
├── config.yaml     # Schema, thresholds, and taxonomy
├── schema.py       # Optional: custom schema class
└── extractors.py   # Optional: custom feature extractors
```

See `docs/providers.md` for detailed documentation.

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

### Rule-based Method (`--classification-method rules`):
1. Extract location-level features (users, downloads, patterns, time-of-day)
2. Apply Isolation Forest anomaly detection
3. Classify using YAML-configurable rules:
   - **BOT**: low downloads/user + many users + anomalous behavior
   - **DOWNLOAD_HUB**: high downloads/user (mirrors) or high total downloads with regular patterns
4. Output: Hierarchical columns (`behavior_type`, `automation_category`, `subcategory`)

### Deep Method (`--classification-method deep`):
1. Extract location-level features (basic + advanced behavioral)
2. Apply Isolation Forest anomaly detection
3. Extract advanced features:
   - Burst patterns, circadian rhythms, user coordination
   - Temporal sequences, file diversity, session patterns
   - Discriminative features (malicious vs legitimate automation)
4. Hierarchical classification:
   - **Level 1**: ORGANIC (human-like) vs AUTOMATED (programmatic)
   - **Level 2**: Within AUTOMATED: BOT vs LEGITIMATE_AUTOMATION
   - **Level 3**: Subcategories (mirror, ci_cd_pipeline, scraper_bot, individual_user, etc.)
5. Pattern discovery using clustering to identify natural behavioral groups
6. Output: `behavior_type`, `automation_category`, `subcategory`

## Deep Method Details

The deep architecture method (`--classification-method deep`) implements a sophisticated hierarchical classification:

### Hierarchical Taxonomy
```
behavior_type (Level 1)
├── ORGANIC (human-like patterns)
│   ├── individual_user      - Single researchers or casual users
│   ├── research_group       - Small academic teams (5-50 users)
│   └── casual_bulk          - Individual heavy downloaders with human patterns
│
└── AUTOMATED (programmatic patterns)
    ├── automation_category: BOT (suspicious/malicious)
    │   ├── scraper_bot         - High-frequency automated scrapers
    │   ├── crawler_bot         - Systematic crawlers
    │   └── coordinated_bot     - Bot farms with coordinated activity
    │
    └── automation_category: LEGITIMATE_AUTOMATION (benign)
        ├── mirror              - Institutional mirrors (very high DL/user)
        ├── institutional_hub   - Research infrastructure hubs
        ├── ci_cd_pipeline      - Automated testing/builds
        ├── course_workshop     - Educational events
        └── data_aggregator     - Legitimate data aggregation services
```

### Deep Learning Architecture
- **Feature Extraction**: Advanced behavioral features (burst patterns, circadian rhythms, user coordination)
- **Transformer Encoder**: Processes temporal sequences to learn behavioral embeddings
- **Pattern Discovery**: Clustering to identify natural behavioral patterns
- **Feature Validation**: Ensures behavioral features meet quality thresholds

### Key Features
- **Hierarchical**: Three-level classification (behavior_type → automation_category → subcategory)
- **Configurable**: Taxonomy rules defined in YAML for easy customization

## Comparing Classification Methods

To compare the two classification methods on the same dataset:

```bash
# Run rules method
logghostbuster -i data.parquet -o output/rules --classification-method rules

# Run deep method
logghostbuster -i data.parquet -o output/deep --classification-method deep
```

Compare the outputs in `location_analysis.csv` to see differences in classification results.

## Output

The tool generates:
- Annotated parquet file with classification columns:
  - `behavior_type`: 'organic' or 'automated'
    - `automation_category`: 'bot' or 'legitimate_automation' (for automated only)
    - `subcategory`: Detailed classification (mirror, scraper_bot, individual_user, etc.)
- `bot_detection_report.txt` - Comprehensive detection report with classification breakdown
- `location_analysis.csv` - Full analysis of all locations with features and classifications
- `feature_importances/` - Feature importance analysis (if enabled with `--compute-importances`)

### Deep Method Output
When using `--classification-method deep`, the output includes:
- Hierarchical classification columns: `behavior_type`, `automation_category`, `subcategory`
- User category: `user_category` for legacy compatibility
- Behavioral features: `regularity_score`, `working_hours_ratio`, `night_activity_ratio`, etc.
- Confidence scores and validation metrics
