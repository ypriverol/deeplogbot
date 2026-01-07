# Enhanced Bot Detection - Usage Guide

This document describes the 7 new enhancements to the bot detection system in `logghostbuster/models/classification/deep_architecture.py` that enable discovery of new bot patterns beyond rule-based labels.

## Problem Addressed

The original deep learning approach was limited by circular dependency on rule-based labels:
- Neural networks trained on labels generated from hand-coded rules
- Could only replicate (not exceed) rule-based performance
- Unable to discover new bot patterns not captured by existing rules

## Solution: 7 Key Enhancements

### 1. Anomaly-Based Label Generation

**Function**: `generate_anomaly_based_labels()`

Replaces rule-based labels with unsupervised clustering on anomaly patterns.

```python
from logghostbuster.models.classification.deep_architecture import generate_anomaly_based_labels

# Use HDBSCAN to cluster anomalies
cluster_labels, cluster_metadata = generate_anomaly_based_labels(
    df=location_data,
    feature_columns=['unique_users', 'downloads_per_user', 'anomaly_score'],
    anomaly_scores=iso_forest_scores,
    min_cluster_size=50
)

# Analyze clusters
for cluster_id, metadata in cluster_metadata.items():
    print(f"Cluster {cluster_id}: {metadata['size']} locations")
    print(f"  Mean anomaly: {metadata['mean_anomaly_score']:.3f}")
    print(f"  Mean users: {metadata['mean_users']:.0f}")
```

**Benefits**:
- Discovers natural groupings in data
- No hard-coded thresholds
- Can identify novel bot patterns

---

### 2. Contrastive Learning for Pattern Discovery

**Class**: `ContrastiveTransformerEncoder`

Learns representations where similar behavioral patterns cluster together without explicit labels.

```python
from logghostbuster.models.classification.deep_architecture import (
    ContrastiveTransformerEncoder,
    augment_time_series
)
import torch

# Create model
model = ContrastiveTransformerEncoder(
    ts_input_dim=10,
    fixed_input_dim=5,
    d_model=128,
    nhead=8,
    num_layers=3,
    temperature=0.5
)

# Prepare data
ts_features = torch.randn(64, 12, 10)  # batch, sequence, features
fixed_features = torch.randn(64, 5)

# Create augmented views
ts_aug = augment_time_series(ts_features, augmentation_strength=0.1)

# Forward pass
z1 = model(ts_features, fixed_features)
z2 = model(ts_aug, fixed_features)

# Compute contrastive loss
loss = model.contrastive_loss(z1, z2)
loss.backward()
```

**Benefits**:
- Learns meaningful representations without labels
- Similar patterns cluster naturally
- Can discover subtle behavioral similarities

---

### 3. Pseudo-Label Refinement

**Function**: `iterative_pseudo_label_refinement()`

Starts with rule-based labels as noisy pseudo-labels and iteratively refines them using model confidence.

```python
from logghostbuster.models.classification.deep_architecture import (
    iterative_pseudo_label_refinement,
    TransformerClassifier
)

# Train classifier
classifier = TransformerClassifier(
    ts_input_dim=10,
    fixed_input_dim=5,
    d_model=128,
    enable_reconstruction=True
)

# Refine labels
refined_classifier, refined_labels = iterative_pseudo_label_refinement(
    classifier=classifier,
    X_ts=time_series_features,
    X_fixed=fixed_features,
    initial_labels=rule_based_labels,
    device=device,
    n_iterations=3,
    confidence_threshold=0.8,
    learning_rate=1e-5
)

# Check how many labels changed
n_changed = (initial_labels != refined_labels).sum()
print(f"Refined {n_changed} labels with high confidence")
```

**Benefits**:
- Starts with existing knowledge (rules)
- Model can override incorrect rule predictions
- Gradually improves label quality

---

### 4. Temporal Anomaly Detection

**Class**: `TemporalAnomalyDetector`

Bidirectional LSTM with attention to detect bot-specific temporal patterns.

```python
from logghostbuster.models.classification.deep_architecture import TemporalAnomalyDetector
import torch

# Create detector
detector = TemporalAnomalyDetector(
    input_dim=10,
    hidden_dim=64,
    num_layers=2
)

# Prepare temporal data (e.g., hourly activity patterns)
temporal_data = torch.randn(32, 24, 10)  # batch, 24 hours, features

# Detect temporal anomalies
anomaly_scores, attention_weights = detector(temporal_data)

# Find locations with high temporal anomaly scores
threshold = anomaly_scores.mean() + 2 * anomaly_scores.std()
suspicious = anomaly_scores > threshold
print(f"Found {suspicious.sum()} locations with suspicious temporal patterns")
```

**Detects**:
- Regular/periodic access patterns (bot scheduling)
- Sudden bursts followed by silence
- Non-human timing (3 AM spikes, 24/7 activity)

---

### 5. Ensemble-Based Discovery

**Function**: `ensemble_bot_discovery()`

Combines multiple anomaly detection methods to find bots that rules miss.

```python
from logghostbuster.models.classification.deep_architecture import ensemble_bot_discovery

# Run ensemble
predictions, agreement = ensemble_bot_discovery(
    df=location_data,
    feature_columns=['unique_users', 'downloads_per_user', 'hourly_entropy'],
    contamination=0.15
)

# Analyze results
print(f"Isolation Forest: {agreement['iso_forest']} anomalies")
print(f"LOF: {agreement['lof']} anomalies")
print(f"One-Class SVM: {agreement['one_class_svm']} anomalies")
print(f"Ensemble (2+ agree): {agreement['ensemble']} anomalies")

# Find cases where ensemble agrees but rules don't
ensemble_anomalies = predictions == -1
rule_bots = df['user_category'] == 'bot'
new_discoveries = ensemble_anomalies & ~rule_bots
print(f"Discovered {new_discoveries.sum()} new potential bots!")
```

**Benefits**:
- Multiple methods catch different anomaly types
- High confidence when 2+ methods agree
- Reduces false positives

---

### 6. Bot Signature Features

**Function**: `add_bot_signature_features()`

Adds 7 discriminative features that capture bot-specific behavioral signatures.

```python
from logghostbuster.models.classification.deep_architecture import add_bot_signature_features

# Add signature features
df_enhanced = add_bot_signature_features(location_data)

# New features available:
signature_features = [
    'access_regularity',           # Inverse of hourly entropy
    'ua_per_user',                 # User-Agent diversity per user
    'request_velocity',            # Downloads per active hour
    'ip_concentration',            # 1 - IP entropy
    'session_anomaly',             # Deviation from median session length
    'request_pattern_anomaly',     # 1 / file request entropy
    'weekend_weekday_imbalance'    # Deviation from expected 2/7 ratio
]

# Analyze signatures
for feature in signature_features:
    print(f"{feature}: mean={df_enhanced[feature].mean():.3f}")

# Find locations with bot-like signatures
bot_signature = (
    (df_enhanced['access_regularity'] > 1.0) &      # Very regular access
    (df_enhanced['request_velocity'] > 100) &        # High velocity
    (df_enhanced['weekend_weekday_imbalance'] < 0.1) # Works 24/7
)
print(f"Found {bot_signature.sum()} locations with strong bot signatures")
```

**Features Explained**:
- **access_regularity**: Bots have predictable schedules (low entropy)
- **request_velocity**: Bots download faster than humans
- **ip_concentration**: Bot farms use concentrated IP ranges
- **weekend_weekday_imbalance**: Humans rest on weekends, bots don't

---

### 7. Active Learning for Edge Cases

**Function**: `identify_uncertain_cases_for_review()`

Identifies cases where human review would help discover new bot patterns.

```python
from logghostbuster.models.classification.deep_architecture import identify_uncertain_cases_for_review

# Identify uncertain cases
uncertain_df = identify_uncertain_cases_for_review(
    df=location_data,
    classifier=trained_classifier,
    X_ts=time_series_features,
    X_fixed=fixed_features,
    device=device,
    top_k=100,
    entropy_weight=0.5,
    margin_weight=0.5
)

# Review uncertain cases
print(f"Top {len(uncertain_df)} uncertain cases:")
for idx, row in uncertain_df.head(10).iterrows():
    print(f"Location: {row.get('location', idx)}")
    print(f"  Uncertainty: {row['uncertainty_score']:.3f}")
    print(f"  Top prediction: {row['top_prediction']} (prob={row['top_probability']:.3f})")
    print(f"  Entropy: {row['entropy']:.3f}, Margin: {row['margin']:.3f}")
```

**Use Cases**:
- Prioritize human review for ambiguous cases
- Discover new attack patterns not seen before
- Continuously improve the model with feedback

---

## Complete Example: Discovering New Bots

Here's how to use all 7 enhancements together:

```python
import pandas as pd
import numpy as np
import torch
from logghostbuster.models.classification.deep_architecture import (
    add_bot_signature_features,
    ensemble_bot_discovery,
    generate_anomaly_based_labels,
    identify_uncertain_cases_for_review,
    classify_locations_deep
)

# 1. Load your location data
df = pd.read_parquet('location_features.parquet')

# 2. Add bot signature features (Enhancement 6)
df = add_bot_signature_features(df)

# 3. Run ensemble anomaly detection (Enhancement 5)
feature_cols = ['unique_users', 'downloads_per_user', 'hourly_entropy',
                'access_regularity', 'request_velocity']
predictions, agreement = ensemble_bot_discovery(df, feature_cols)

# 4. Generate anomaly-based labels (Enhancement 1)
cluster_labels, metadata = generate_anomaly_based_labels(
    df, feature_cols, agreement['ensemble_scores']
)

# 5. Use existing deep classification with all enhancements
df_classified, _ = classify_locations_deep(
    df=df,
    feature_columns=feature_cols,
    use_transformer=True,
    enable_self_supervised=True,
    enable_neural_classification=True,
    enable_bot_head=True
)

# 6. Identify uncertain cases for review (Enhancement 7)
# (requires loading the trained classifier from classify_locations_deep)

# 7. Find new bot discoveries
known_bots = df_classified['user_category'] == 'bot'
ensemble_bots = predictions == -1
new_bots = ensemble_bots & ~known_bots

print(f"Known bots (rules): {known_bots.sum()}")
print(f"Ensemble anomalies: {ensemble_bots.sum()}")
print(f"NEW bot discoveries: {new_bots.sum()}")

# Export new discoveries for review
new_bot_df = df_classified[new_bots]
new_bot_df.to_csv('new_bot_discoveries.csv')
```

## Best Practices

1. **Start with Ensemble Discovery**: Use `ensemble_bot_discovery()` to get high-confidence anomalies
2. **Add Signature Features**: Always call `add_bot_signature_features()` before classification
3. **Use Contrastive Learning**: For large datasets, use `ContrastiveTransformerEncoder` for pre-training
4. **Refine Labels Iteratively**: Use `iterative_pseudo_label_refinement()` to improve predictions
5. **Review Uncertain Cases**: Regularly review output from `identify_uncertain_cases_for_review()`
6. **Combine with Rules**: Use ensemble + rules together for best results

## Performance Considerations

- **HDBSCAN**: Requires HDBSCAN library (`pip install hdbscan`)
- **GPU**: Use GPU for faster training of Transformer models
- **Batch Size**: Adjust based on available memory (default: 256)
- **Contamination**: Tune based on expected bot percentage (default: 0.15)

## Troubleshooting

**HDBSCAN not available**: The code falls back to DBSCAN automatically
**Out of memory**: Reduce batch size or use smaller models
**Poor performance**: Try different contamination rates or feature combinations

## References

- Original deep learning approach: `classify_locations_deep()`
- Existing rule-based detection: `_generate_rule_based_labels()`
- Documentation: Module docstring in `deep_architecture.py`
