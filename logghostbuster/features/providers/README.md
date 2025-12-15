# Provider-Specific Feature Extractors

This directory contains feature extractors specific to different log providers.

## Structure

- `ebi.py` - EBI-specific extractors (if needed)
- Additional provider modules can be added here (e.g., `aws.py`, `gcp.py`, etc.)

## Adding a New Provider

1. Create a new file for your provider (e.g., `my_provider.py`)
2. Import `BaseFeatureExtractor` from `..base`
3. Implement provider-specific extractors:

```python
from ..base import BaseFeatureExtractor
import pandas as pd

class MyProviderSpecificExtractor(BaseFeatureExtractor):
    def extract(self, df: pd.DataFrame, input_parquet_path: str, conn) -> pd.DataFrame:
        # Add provider-specific features
        df['my_provider_feature'] = ...
        return df
```

4. Use it in your pipeline:

```python
from logghostbuster import run_bot_annotator
from logghostbuster.features.providers.my_provider import MyProviderSpecificExtractor

results = run_bot_annotator(
    input_parquet='logs.parquet',
    custom_extractors=[MyProviderSpecificExtractor(schema)],
)
```

## Notes

- Standard extractors in `../standard.py` work with any log format
- Provider-specific extractors should only be used when you need features
  that are specific to a particular log provider's format or metadata
- All extractors should follow the `BaseFeatureExtractor` interface
