#!/usr/bin/env python3
"""Test script for pure unsupervised clustering with LLM interpretation."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

# Import the module
import importlib.util
spec = importlib.util.spec_from_file_location(
    'pure_unsupervised',
    'deeplogbot/models/unsupervised/pure_unsupervised.py'
)
pure_unsupervised = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pure_unsupervised)

PureUnsupervisedConfig = pure_unsupervised.PureUnsupervisedConfig
PureUnsupervisedClassifier = pure_unsupervised.PureUnsupervisedClassifier
FeatureExtractor = pure_unsupervised.FeatureExtractor
AdaptiveClusterer = pure_unsupervised.AdaptiveClusterer
ClusterProfiler = pure_unsupervised.ClusterProfiler


def create_test_data(n_samples=300):
    """Create synthetic test data with distinct patterns."""
    np.random.seed(42)

    data = []

    # Pattern 1: Organic users (50%) - few users, moderate DPU, working hours
    n_organic = int(n_samples * 0.5)
    for i in range(n_organic):
        data.append({
            'geo_location': f'organic_{i}',
            'unique_users': np.random.randint(1, 30),
            'downloads_per_user': np.random.randint(10, 80),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.45, 0.75),
            'night_activity_ratio': np.random.uniform(0.05, 0.2),
            'hourly_entropy': np.random.uniform(1.5, 2.5),
            'regularity_score': np.random.uniform(0.1, 0.4),
            'burst_pattern_score': np.random.uniform(0, 2),
            'user_coordination_score': np.random.uniform(0, 0.2),
        })

    # Pattern 2: Bots (30%) - many users, low DPU, night activity
    n_bots = int(n_samples * 0.3)
    for i in range(n_bots):
        data.append({
            'geo_location': f'bot_{i}',
            'unique_users': np.random.randint(200, 5000),
            'downloads_per_user': np.random.randint(2, 25),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.1, 0.3),
            'night_activity_ratio': np.random.uniform(0.35, 0.6),
            'hourly_entropy': np.random.uniform(2.5, 3.2),
            'regularity_score': np.random.uniform(0.6, 0.9),
            'burst_pattern_score': np.random.uniform(5, 12),
            'user_coordination_score': np.random.uniform(0.5, 0.9),
        })

    # Pattern 3: Hubs (20%) - few users, very high DPU
    n_hubs = n_samples - n_organic - n_bots
    for i in range(n_hubs):
        data.append({
            'geo_location': f'hub_{i}',
            'unique_users': np.random.randint(1, 15),
            'downloads_per_user': np.random.randint(300, 1500),
            'total_downloads': 0,
            'working_hours_ratio': np.random.uniform(0.3, 0.6),
            'night_activity_ratio': np.random.uniform(0.1, 0.25),
            'hourly_entropy': np.random.uniform(1.8, 2.5),
            'regularity_score': np.random.uniform(0.5, 0.75),
            'burst_pattern_score': np.random.uniform(1, 4),
            'user_coordination_score': np.random.uniform(0, 0.25),
        })

    df = pd.DataFrame(data)
    df['total_downloads'] = df['unique_users'] * df['downloads_per_user']

    return df


def test_feature_extractor():
    """Test feature extraction without domain bias."""
    print("=" * 60)
    print("TEST 1: Feature Extraction")
    print("=" * 60)

    df = create_test_data(100)
    extractor = FeatureExtractor()
    X, feature_names = extractor.fit_transform(df)

    print(f"Features extracted: {len(feature_names)}")
    print(f"Shape: {X.shape}")
    print(f"Feature names: {feature_names}")
    print("PASSED\n")


def test_adaptive_clustering():
    """Test adaptive clustering with BIC selection."""
    print("=" * 60)
    print("TEST 2: Adaptive Clustering")
    print("=" * 60)

    df = create_test_data(200)
    extractor = FeatureExtractor()
    X, _ = extractor.fit_transform(df)

    config = PureUnsupervisedConfig()
    config.clustering_method = 'gmm'
    config.n_clusters = None  # Let BIC decide

    clusterer = AdaptiveClusterer(config)
    labels, meta = clusterer.fit_predict(X)

    print(f"Method: {meta['method']}")
    print(f"Clusters found: {meta['n_clusters']}")
    print(f"Silhouette: {meta['silhouette']:.3f}")

    # Check cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} samples")

    assert meta['n_clusters'] >= 2, "Should find at least 2 clusters"
    print("PASSED\n")


def test_cluster_profiler():
    """Test cluster profile computation."""
    print("=" * 60)
    print("TEST 3: Cluster Profiling")
    print("=" * 60)

    df = create_test_data(200)
    extractor = FeatureExtractor()
    X, _ = extractor.fit_transform(df)

    config = PureUnsupervisedConfig()
    clusterer = AdaptiveClusterer(config)
    labels, _ = clusterer.fit_predict(X)

    profiler = ClusterProfiler()
    profiles = profiler.compute_profiles(df, labels)

    print(f"Computed {len(profiles)} cluster profiles")
    for p in profiles:
        print(f"\n  Cluster {p['cluster_id']}:")
        print(f"    Size: {p['size']} ({p['size_pct']}%)")
        if 'unique_users_median' in p:
            print(f"    Users: median={p['unique_users_median']:.0f}")
        if 'downloads_per_user_median' in p:
            print(f"    DPU: median={p['downloads_per_user_median']:.1f}")

    # Test LLM prompt formatting
    profiles_text = profiler.format_for_llm(profiles)
    print(f"\nLLM prompt preview ({len(profiles_text)} chars):")
    print(profiles_text[:500] + "...")

    print("\nPASSED\n")


def test_full_pipeline_heuristic():
    """Test full pipeline with heuristic fallback (no LLM)."""
    print("=" * 60)
    print("TEST 4: Full Pipeline (Heuristic Mode)")
    print("=" * 60)

    df = create_test_data(300)

    config = PureUnsupervisedConfig()
    config.use_llm = False  # Use heuristic fallback
    config.use_umap = True

    classifier = PureUnsupervisedClassifier(config)
    result_df = classifier.fit_predict(df)

    # Check output columns
    required_cols = ['behavior_type', 'automation_category', 'subcategory',
                     'classification_confidence', 'cluster_id']
    for col in required_cols:
        assert col in result_df.columns, f"Missing column: {col}"

    print(f"\nResults:")
    print(f"  Total samples: {len(result_df)}")

    # Check behavior type distribution
    bt_counts = result_df['behavior_type'].value_counts()
    print(f"\nBehavior types:")
    for bt, count in bt_counts.items():
        pct = 100 * count / len(result_df)
        print(f"  {bt}: {count} ({pct:.1f}%)")

    # Check subcategories
    sc_counts = result_df['subcategory'].value_counts()
    print(f"\nSubcategories:")
    for sc, count in sc_counts.items():
        pct = 100 * count / len(result_df)
        print(f"  {sc}: {count} ({pct:.1f}%)")

    print(f"\nMean confidence: {result_df['classification_confidence'].mean():.3f}")
    print("PASSED\n")


def test_cluster_separation():
    """Test that distinct patterns are separated into different clusters."""
    print("=" * 60)
    print("TEST 5: Cluster Separation Quality")
    print("=" * 60)

    df = create_test_data(300)

    # Add ground truth for evaluation
    df['ground_truth'] = df['geo_location'].apply(
        lambda x: 'organic' if x.startswith('organic')
                  else 'bot' if x.startswith('bot')
                  else 'hub'
    )

    config = PureUnsupervisedConfig()
    config.use_llm = False

    classifier = PureUnsupervisedClassifier(config)
    result_df = classifier.fit_predict(df)

    # Check how well clusters align with ground truth
    print("\nCluster composition (ground truth):")
    for cluster_id in result_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        cluster_mask = result_df['cluster_id'] == cluster_id
        cluster_gt = result_df.loc[cluster_mask, 'ground_truth'].value_counts()
        dominant = cluster_gt.index[0]
        purity = cluster_gt.iloc[0] / cluster_gt.sum()
        print(f"  Cluster {cluster_id}: {cluster_gt.to_dict()} -> dominant={dominant} (purity={purity:.1%})")

    # Calculate overall purity
    total_purity = 0
    for cluster_id in result_df['cluster_id'].unique():
        if cluster_id == -1:
            continue
        cluster_mask = result_df['cluster_id'] == cluster_id
        cluster_gt = result_df.loc[cluster_mask, 'ground_truth'].value_counts()
        total_purity += cluster_gt.iloc[0]

    overall_purity = total_purity / len(result_df)
    print(f"\nOverall cluster purity: {overall_purity:.1%}")

    if overall_purity > 0.7:
        print("PASSED - Good cluster separation\n")
    else:
        print("NEEDS IMPROVEMENT - Clusters are mixing patterns\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PURE UNSUPERVISED CLASSIFICATION TESTS")
    print("=" * 60 + "\n")

    test_feature_extractor()
    test_adaptive_clustering()
    test_cluster_profiler()
    test_full_pipeline_heuristic()
    test_cluster_separation()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
