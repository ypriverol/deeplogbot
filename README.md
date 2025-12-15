# LogGhostbuster

LogGhostbuster: A hybrid Isolation Forest-based system for detecting bot behavior in log data.

## Overview

LogGhostbuster uses machine learning (specifically Isolation Forest) to detect:
1. **Bot downloads** - automated downloads from bot farms (user ID cycling pattern)
2. **Download hubs** - legitimate mirrors/institutions with high download volumes

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

### Python API

```python
from logghostbuster import run_bot_annotator

results = run_bot_annotator(
    input_parquet='data_downloads_parquet.parquet',
    output_parquet='annotated_data.parquet',
    output_dir='output/bot_analysis',
    contamination=0.15,
    compute_importances=False
)
```

## Package Structure

- `logghostbuster/`
  - `__init__.py` - Package initialization and exports
  - `cli.py` - Command-line interface
  - `main.py` - Main bot detection pipeline
  - `schema.py` - Schema definitions for different log formats
  - `features/` - Feature extraction package
    - `__init__.py` - Feature extraction exports
    - `base.py` - Base feature extractor class
    - `extraction.py` - Main feature extraction function
    - `standard.py` - Standard extractors (yearly, time-of-day, country-level)
    - `providers/` - Provider-specific extractors
      - `ebi.py` - EBI-specific extractors (if needed)
  - `models.py` - ML models (Isolation Forest, feature importance)
  - `classification.py` - Bot and download hub classification logic
  - `geography.py` - Geographic utilities for location grouping
  - `llm_utils.py` - LLM utilities for canonical naming (optional)
  - `annotation.py` - Annotation utilities for marking locations
  - `reporting.py` - Report generation
  - `utils.py` - Utility functions

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
- `OLLAMA_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL`: Ollama model name (default: `llama3.2`)
- `HF_MODEL`: Hugging Face model name (default: `microsoft/DialoGPT-medium`)

## Output

The tool generates:
- Annotated parquet file with `bot` and `download_hub` columns
- `bot_detection_report.txt` - Comprehensive detection report
- `location_analysis.csv` - Full analysis of all locations
- `feature_importances/` - Feature importance analysis (if enabled)
