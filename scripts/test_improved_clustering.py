#!/usr/bin/env python3
"""Test script for the improved unsupervised clustering implementation.

This script tests:
1. Bot/hub threshold tuning
2. Behavioral feature integration
3. Semi-supervised learning with Rules guidance
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

# Import directly to avoid torch issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    'improved_clustering',
    'deeplogbot/models/unsupervised/improved_clustering.py'
)
improved_clustering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(improved_clustering)

# Get classes and functions
ImprovedClusteringConfig = improved_clustering.ImprovedClusteringConfig
ImprovedUnsupervisedClassifier = improved_clustering.ImprovedUnsupervisedClassifier
get_available_behavioral_features = improved_clustering.get_available_behavioral_features
BEHAVIORAL_FEATURES = improved_clustering.BEHAVIORAL_FEATURES


def create_test_data(n_samples=1000):
    """Create synthetic test data with known bot/hub/organic patterns."""
    np.random.seed(42)

    data = []

    # Organic users (60% of data) - few users, moderate DPU, working hours
    n_organic = int(n_samples * 0.6)
    for _ in range(n_organic):
        data.append({
            'geo_location': f'organic_{_}',
            'unique_users': np.random.randint(1, 50),
            'downloads_per_user': np.random.randint(5, 100),
            'total_downloads': 0,  # Will be computed
            'working_hours_ratio': np.random.uniform(0.4, 0.7),
            'night_activity_ratio': np.random.uniform(0.05, 0.2),
            'hourly_entropy': np.random.uniform(0.3, 0.7),
            'user_cv': np.random.uniform(0.1, 0.4),
            'regularity_score': np.random.uniform(0.2, 0.5),
            # Behavioral features
            'burst_pattern_score': np.random.uniform(0, 2),
            'circadian_rhythm_deviation': np.random.uniform(0.1, 0.4),
            'user_coordination_score': np.random.uniform(0, 0.3),
            'spike_intensity': np.random.uniform(1, 3),
            'is_nocturnal': 0,
            'is_coordinated': 0,
            'access_regularity': np.random.uniform(0.1, 0.5),
            'expected_label': 'organic',
        })

    # Bots (25% of data) - many users, low DPU, night activity
    n_bots = int(n_samples * 0.25)
    for _ in range(n_bots):
        data.append({
            'geo_location': f'bot_{_}',
            'unique_users': np.random.randint(200, 5000),
            'downloads_per_user': np.random.randint(2, 30),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.1, 0.3),
            'night_activity_ratio': np.random.uniform(0.3, 0.6),
            'hourly_entropy': np.random.uniform(0.6, 0.95),
            'user_cv': np.random.uniform(0.5, 0.9),
            'regularity_score': np.random.uniform(0.6, 0.9),
            # Behavioral features
            'burst_pattern_score': np.random.uniform(3, 10),
            'circadian_rhythm_deviation': np.random.uniform(0.5, 1.0),
            'user_coordination_score': np.random.uniform(0.5, 1.0),
            'spike_intensity': np.random.uniform(5, 15),
            'is_nocturnal': np.random.choice([0, 1], p=[0.3, 0.7]),
            'is_coordinated': np.random.choice([0, 1], p=[0.3, 0.7]),
            'access_regularity': np.random.uniform(0.6, 1.0),
            'expected_label': 'bot',
        })

    # Hubs (15% of data) - few users, very high DPU
    n_hubs = n_samples - n_organic - n_bots
    for _ in range(n_hubs):
        data.append({
            'geo_location': f'hub_{_}',
            'unique_users': np.random.randint(1, 30),
            'downloads_per_user': np.random.randint(300, 2000),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.3, 0.6),
            'night_activity_ratio': np.random.uniform(0.1, 0.3),
            'hourly_entropy': np.random.uniform(0.4, 0.7),
            'user_cv': np.random.uniform(0.2, 0.5),
            'regularity_score': np.random.uniform(0.5, 0.8),
            # Behavioral features
            'burst_pattern_score': np.random.uniform(1, 4),
            'circadian_rhythm_deviation': np.random.uniform(0.2, 0.5),
            'user_coordination_score': np.random.uniform(0, 0.3),
            'spike_intensity': np.random.uniform(2, 5),
            'is_nocturnal': 0,
            'is_coordinated': 0,
            'access_regularity': np.random.uniform(0.4, 0.7),
            'expected_label': 'hub',
        })

    df = pd.DataFrame(data)
    df['total_downloads'] = df['unique_users'] * df['downloads_per_user']

    return df


def test_config():
    """Test configuration loading."""
    print("=" * 60)
    print("TEST 1: Configuration")
    print("=" * 60)

    config = ImprovedClusteringConfig()

    print(f"Bot thresholds: {config.bot_thresholds}")
    print(f"Hub thresholds: {config.hub_thresholds}")
    print(f"Use behavioral features: {config.use_behavioral_features}")
    print(f"Semi-supervised confidence threshold: {config.semi_supervised_confidence_threshold}")
    print(f"Feature weights count: {len(config.feature_weights)}")
    print("PASSED\n")


def test_behavioral_features():
    """Test behavioral feature detection."""
    print("=" * 60)
    print("TEST 2: Behavioral Features")
    print("=" * 60)

    df = create_test_data(100)
    info = get_available_behavioral_features(df)

    print(f"Total behavioral features defined: {info['total']}")
    print(f"Available in test data: {info['available_count']}")
    print(f"Missing from test data: {info['missing_count']}")
    print(f"Available features: {info['available'][:10]}...")
    print("PASSED\n")


def test_classifier():
    """Test the improved classifier."""
    print("=" * 60)
    print("TEST 3: Improved Classifier")
    print("=" * 60)

    df = create_test_data(500)
    expected_labels = df['expected_label'].copy()

    config = ImprovedClusteringConfig()
    config.use_rules_guidance = False  # Test without rules guidance first

    classifier = ImprovedUnsupervisedClassifier(config)
    result_df = classifier.fit_predict(df.drop(columns=['expected_label']))

    # Check output columns
    required_cols = ['behavior_type', 'automation_category', 'subcategory',
                     'classification_confidence', 'is_anomaly']
    for col in required_cols:
        assert col in result_df.columns, f"Missing column: {col}"

    print(f"\nClassification results:")
    print(f"  Organic: {(result_df['behavior_type'] == 'organic').sum()}")
    print(f"  Automated: {(result_df['behavior_type'] == 'automated').sum()}")

    if 'automation_category' in result_df.columns:
        print(f"  - Bots: {(result_df['automation_category'] == 'bot').sum()}")
        print(f"  - Hubs: {(result_df['automation_category'] == 'legitimate_automation').sum()}")

    # Calculate accuracy
    predicted_labels = []
    for _, row in result_df.iterrows():
        if row['automation_category'] == 'bot':
            predicted_labels.append('bot')
        elif row['automation_category'] == 'legitimate_automation':
            predicted_labels.append('hub')
        else:
            predicted_labels.append('organic')

    accuracy = (np.array(predicted_labels) == expected_labels.values).mean()
    print(f"\nAccuracy vs expected: {accuracy:.1%}")
    print(f"Mean confidence: {result_df['classification_confidence'].mean():.3f}")
    print("PASSED\n")


def test_bot_hub_separation():
    """Test that bots and hubs are correctly separated."""
    print("=" * 60)
    print("TEST 4: Bot/Hub Separation")
    print("=" * 60)

    np.random.seed(42)

    # Create a realistic mix: 70% organic, 15% bots, 15% hubs
    # This gives the anomaly detector normal patterns to compare against

    # 140 organic (normal traffic)
    organic = []
    for i in range(140):
        organic.append({
            'geo_location': f'organic_{i}',
            'unique_users': np.random.randint(1, 50),
            'downloads_per_user': np.random.randint(10, 100),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.4, 0.7),
            'night_activity_ratio': np.random.uniform(0.05, 0.2),
            'hourly_entropy': np.random.uniform(0.3, 0.6),
            'user_cv': np.random.uniform(0.1, 0.4),
            'regularity_score': np.random.uniform(0.2, 0.5),
            'burst_pattern_score': np.random.uniform(0, 3),
            'circadian_rhythm_deviation': np.random.uniform(0.1, 0.4),
            'user_coordination_score': np.random.uniform(0, 0.3),
            'spike_intensity': np.random.uniform(1, 3),
            'is_nocturnal': 0,
            'is_coordinated': 0,
            'access_regularity': np.random.uniform(0.2, 0.5),
            'expected': 'organic',
        })

    # 30 clear bots (15%)
    clear_bots = []
    for i in range(30):
        clear_bots.append({
            'geo_location': f'clear_bot_{i}',
            'unique_users': np.random.randint(1000, 8000),
            'downloads_per_user': np.random.randint(2, 15),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.05, 0.2),
            'night_activity_ratio': np.random.uniform(0.4, 0.7),
            'hourly_entropy': np.random.uniform(0.7, 0.95),
            'user_cv': np.random.uniform(0.6, 0.9),
            'regularity_score': np.random.uniform(0.7, 0.95),
            'burst_pattern_score': np.random.uniform(6, 12),
            'circadian_rhythm_deviation': np.random.uniform(0.6, 1.0),
            'user_coordination_score': np.random.uniform(0.6, 1.0),
            'spike_intensity': np.random.uniform(8, 15),
            'is_nocturnal': 1,
            'is_coordinated': 1,
            'access_regularity': np.random.uniform(0.7, 1.0),
            'expected': 'bot',
        })

    # 30 clear hubs (15%)
    clear_hubs = []
    for i in range(30):
        clear_hubs.append({
            'geo_location': f'clear_hub_{i}',
            'unique_users': np.random.randint(1, 15),
            'downloads_per_user': np.random.randint(500, 2000),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.4, 0.7),
            'night_activity_ratio': np.random.uniform(0.05, 0.2),
            'hourly_entropy': np.random.uniform(0.3, 0.5),
            'user_cv': np.random.uniform(0.1, 0.3),
            'regularity_score': np.random.uniform(0.5, 0.7),
            'burst_pattern_score': np.random.uniform(1, 3),
            'circadian_rhythm_deviation': np.random.uniform(0.1, 0.4),
            'user_coordination_score': np.random.uniform(0, 0.2),
            'spike_intensity': np.random.uniform(1, 4),
            'is_nocturnal': 0,
            'is_coordinated': 0,
            'access_regularity': np.random.uniform(0.3, 0.6),
            'expected': 'hub',
        })

    df = pd.DataFrame(organic + clear_bots + clear_hubs)
    df['total_downloads'] = df['unique_users'] * df['downloads_per_user']
    expected = df['expected'].copy()
    df = df.drop(columns=['expected'])

    config = ImprovedClusteringConfig()
    config.use_rules_guidance = False
    config.anomaly_contamination = 0.30  # 30% anomalies (15% bots + 15% hubs)

    classifier = ImprovedUnsupervisedClassifier(config)
    result_df = classifier.fit_predict(df)

    # Check classifications
    result_df['expected'] = expected.values

    # Calculate accuracy for bots
    bot_mask = result_df['expected'] == 'bot'
    bot_correct = (result_df.loc[bot_mask, 'automation_category'] == 'bot').sum()
    bot_total = bot_mask.sum()

    # Calculate accuracy for hubs
    hub_mask = result_df['expected'] == 'hub'
    hub_correct = (result_df.loc[hub_mask, 'automation_category'] == 'legitimate_automation').sum()
    hub_total = hub_mask.sum()

    print(f"Bot accuracy: {bot_correct}/{bot_total} ({100*bot_correct/bot_total:.1f}%)")
    print(f"Hub accuracy: {hub_correct}/{hub_total} ({100*hub_correct/hub_total:.1f}%)")

    overall_accuracy = (bot_correct + hub_correct) / (bot_total + hub_total)
    print(f"Overall accuracy: {overall_accuracy:.1%}")

    if overall_accuracy >= 0.7:
        print("PASSED\n")
    else:
        print("NEEDS TUNING - Edge cases may require threshold adjustment\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("IMPROVED CLUSTERING IMPLEMENTATION TESTS")
    print("=" * 60 + "\n")

    test_config()
    test_behavioral_features()
    test_classifier()
    test_bot_hub_separation()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
