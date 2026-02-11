# DeepLogBot

Bot detection and traffic classification for scientific data repository logs.

## Overview

DeepLogBot (CLI: `deeplogbot`) detects and classifies download patterns in scientific data repository logs, distinguishing between:

- **Organic users** — Human researchers with natural download patterns
- **Bots** — Automated scrapers, crawlers, and coordinated bot farms
- **Download hubs** — Legitimate mirrors, institutional pipelines, and data aggregators

Applied to the PRIDE Archive (159M download records), the system identified that **88% of traffic is bot-generated**. After filtering, **19.1M clean downloads** remain across 34,085 datasets and 213 countries.

### Hierarchical Classification Taxonomy

```
behavior_type (Level 1)
├── ORGANIC
│   ├── individual_user
│   └── research_group
│
└── AUTOMATED
    ├── BOT
    │   └── generic_bot
    │
    └── LEGITIMATE_AUTOMATION (Hub)
        ├── mirror
        ├── ci_cd_pipeline
        └── course_workshop
```

## Classification Methods

DeepLogBot provides **2 classification methods**:

| Method | Macro F1 | Speed | Description |
|--------|----------|-------|-------------|
| `rules` | 0.632 | Fast | YAML-configurable thresholds, no training required |
| `deep` | 0.775 | Medium | Multi-stage learned pipeline with soft priors |

*Benchmarked on a 1M-record sample with manually curated ground truth.*

### Rule-Based (`--classification-method rules`)

Hierarchical threshold classification using YAML-configurable rules. Fast, interpretable, and requires no training. Best for production use with known patterns.

### Deep Architecture (`--classification-method deep`)

Multi-stage learned pipeline:

1. **Seed Selection** — Identify high-confidence bot/organic/hub seeds from feature distributions
2. **Organic VAE** — Learn the normal-behavior manifold; score reconstruction error
3. **Deep Isolation Forest** — Non-linear anomaly detection on VAE latent space
4. **Temporal Consistency** — Modified z-score spike detection (no fixed thresholds)
5. **Fusion Meta-Learner** — Gradient-boosted combination of all anomaly signals

Additional components:
- **Soft priors** — Pre-filter signals encoded as continuous features (no hard lockout)
- **Reconciliation** — Override thresholds for cases where pipeline and pre-filter disagree
- **Hub protection** — Prevent legitimate automation from being classified as bots
- **Post-classification** — Detailed subcategory assignment

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, scipy, duckdb
- Optional: torch (for deep method)

## Usage

### Command Line

```bash
# Rule-based classification (default)
deeplogbot -i data.parquet -o output/

# Deep architecture
deeplogbot -i data.parquet -o output/ -m deep

# With sampling for large datasets
deeplogbot -i data.parquet -o output/ -m deep --sample-size 1000000
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input parquet file | Required |
| `-o, --output-dir` | Output directory | `output/bot_analysis` |
| `-m, --classification-method` | `rules` or `deep` | `rules` |
| `-c, --contamination` | Anomaly proportion | `0.15` |
| `-s, --sample-size` | Sample N records | None (use all) |
| `-p, --provider` | Log provider | `ebi` |

### Python API

```python
from deeplogbot import run_bot_annotator

# Rule-based classification
results = run_bot_annotator(
    input_parquet='data.parquet',
    output_dir='output/',
    classification_method='rules'
)

# Deep architecture
results = run_bot_annotator(
    input_parquet='data.parquet',
    output_dir='output/',
    classification_method='deep'
)

print(f"Bots detected: {results['bot_count']}")
print(f"Hubs detected: {results['hub_count']}")
```

## Project Structure

```
deeplogbot/
├── __init__.py                  # Package exports
├── main.py                      # CLI entry point and pipeline
├── config.py                    # Configuration loading
├── config.yaml                  # Main configuration file
│
├── features/                    # Feature extraction (~117 features)
│   ├── base.py                  # Base extractor class
│   ├── schema.py                # Log schema definitions
│   ├── registry.py              # Feature documentation registry
│   └── providers/
│       └── ebi/                 # EBI/PRIDE provider
│           ├── ebi.py           # Location feature extraction
│           ├── behavioral.py    # Behavioral features
│           ├── discriminative.py # Discriminative features
│           ├── timeseries.py    # Time series features
│           └── schema.py        # EBI-specific schema
│
├── models/
│   ├── isoforest/               # Isolation Forest anomaly detection
│   │   └── models.py
│   └── classification/          # Classification methods
│       ├── rules.py             # Rule-based hierarchical classifier
│       ├── deep_architecture.py # Deep pipeline orchestration
│       ├── seed_selection.py    # High-confidence seed identification
│       ├── organic_vae.py       # VAE + Deep Isolation Forest
│       ├── temporal_consistency.py # Modified z-score spike detection
│       ├── fusion.py            # Gradient-boosted meta-learner
│       ├── post_classification.py # Hub protection & subcategory assignment
│       └── feature_validation.py  # Feature usage validation
│
├── reports/                     # Output generation
│   ├── reporting.py             # Text report generation
│   ├── annotation.py            # Parquet annotation
│   ├── statistics.py            # Summary statistics
│   ├── html_report.py           # Interactive HTML reports
│   └── visualizations.py        # Charts and plots
│
├── utils/                       # Utilities
│   └── geography.py             # Geographic lookups
│
└── providers/
    └── base_taxonomy.yaml       # Classification taxonomy
```

## Configuration

Configuration is in `deeplogbot/config.yaml`:

```yaml
isolation_forest:
  contamination: 0.15
  n_estimators: 200
  random_state: 42

classification:
  rule_based:
    bots:
      require_anomaly: true
      patterns:
        - downloads_per_user: {max: 100}
          unique_users: {min: 5000}
    hubs:
      require_anomaly: true
      patterns:
        - downloads_per_user: {min: 500}

deep_reconciliation:
  override_threshold: 0.7
  strict_threshold: 0.8
```

## Classifying a Download Parquet File

Given a parquet file of download logs (one row per download event), DeepLogBot aggregates records by geographic location, extracts ~117 behavioral and discriminative features, classifies each location as bot/hub/organic, and writes a new annotated parquet with classification columns appended to every row.

### Input format

The input parquet must contain at minimum:

| Column | Description |
|--------|-------------|
| `accession` | Dataset accession (e.g., `PXD000001`) |
| `geo_location` | Geographic location string (city/region) |
| `country` | Country name or code |
| `year` | Download year |
| `date` | Download date |

### Running classification

```bash
# Classify with the deep method (recommended) — writes <input>_annotated.parquet
deeplogbot -i downloads.parquet -o output/ -m deep

# Classify with rules (faster, no torch required)
deeplogbot -i downloads.parquet -o output/ -m rules

# For large files, sample first to speed up classification
deeplogbot -i downloads.parquet -o output/ -m deep --sample-size 5000000
```

The annotated parquet is written to the output directory with an `_annotated` suffix (e.g., `output/downloads_annotated.parquet`). You can also specify an explicit output path:

```bash
deeplogbot -i downloads.parquet -o output/ -m deep --output output/classified.parquet
```

### Output strategies

| Strategy | Flag | Behavior |
|----------|------|----------|
| `new_file` (default) | `--output-strategy new_file` | Creates `<input>_annotated.parquet` in the output directory |
| `overwrite` | `--output-strategy overwrite` | Rewrites the original parquet in place |
| `reports_only` | `--reports-only` | Generates text/HTML reports without writing a parquet |

### Using the annotated parquet

```python
import duckdb

conn = duckdb.connect()
df = conn.execute("""
    SELECT accession, country, year,
           is_bot, is_hub, is_organic,
           behavior_type, automation_category, subcategory
    FROM read_parquet('output/downloads_annotated.parquet')
    LIMIT 10
""").df()

# Filter to clean (non-bot, non-hub) downloads
clean = conn.execute("""
    SELECT accession, country, COUNT(*) as downloads
    FROM read_parquet('output/downloads_annotated.parquet')
    WHERE is_bot = false AND is_hub = false
    GROUP BY accession, country
    ORDER BY downloads DESC
""").df()
```

## Output Format

The annotated output parquet contains:

| Column | Description |
|--------|-------------|
| `is_bot` | Bot classification flag |
| `is_hub` | Download hub classification flag |
| `is_organic` | Organic user classification flag |
| `behavior_type` | `organic` or `automated` |
| `automation_category` | `bot` or `legitimate_automation` |
| `subcategory` | Detailed category (e.g., `mirror`, `generic_bot`) |
| `classification_confidence` | Confidence score (0-1) |

Reports generated:
- `bot_detection_report.txt` — Summary with counts and breakdowns
- `location_analysis.csv` — Per-location features and classifications
- Interactive HTML report (if enabled)

## License

MIT
