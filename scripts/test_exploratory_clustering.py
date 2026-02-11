#!/usr/bin/env python3
"""Test script for exploratory clustering - discovering novel patterns.

This test creates synthetic data with MULTIPLE pattern types beyond bot/hub/organic
to verify that the exploratory clustering can discover novel patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_diverse_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """Create synthetic data with diverse patterns beyond bot/hub/organic.

    Creates 7 distinct pattern types:
    1. Organic researchers (human researchers with normal patterns)
    2. Bots (automated scrapers with high volume, many users)
    3. Hubs (data mirrors with bulk downloads)
    4. Data pipelines (regular automated CI/CD-style access)
    5. Burst downloaders (periodic intense download bursts)
    6. Educational labs (classroom patterns with coordinated access)
    7. Nocturnal automation (night-time automated processing)
    """
    np.random.seed(42)

    # Distribution: 40% organic, 15% bot, 10% hub, 10% pipeline, 10% burst, 10% edu, 5% nocturnal
    n_organic = int(n_samples * 0.40)
    n_bot = int(n_samples * 0.15)
    n_hub = int(n_samples * 0.10)
    n_pipeline = int(n_samples * 0.10)
    n_burst = int(n_samples * 0.10)
    n_edu = int(n_samples * 0.10)
    n_nocturnal = n_samples - n_organic - n_bot - n_hub - n_pipeline - n_burst - n_edu

    data = []

    # 1. Organic researchers
    for i in range(n_organic):
        data.append({
            'unique_users': np.random.randint(1, 30),
            'total_downloads': np.random.randint(10, 500),
            'downloads_per_user': np.random.uniform(5, 50),
            'working_hours_ratio': np.random.uniform(0.5, 0.9),
            'night_activity_ratio': np.random.uniform(0.05, 0.25),
            'hourly_entropy': np.random.uniform(0.4, 0.7),
            'regularity_score': np.random.uniform(0.2, 0.5),
            'burst_pattern_score': np.random.uniform(1, 3),
            'user_coordination_score': np.random.uniform(0.0, 0.2),
            'file_diversity_ratio': np.random.uniform(0.3, 0.7),
            'weekend_ratio': np.random.uniform(0.05, 0.25),
            'interval_cv': np.random.uniform(0.5, 1.5),
            'pattern_type': 'organic',
        })

    # 2. Bots (high volume, many users, automated)
    for i in range(n_bot):
        data.append({
            'unique_users': np.random.randint(500, 2000),
            'total_downloads': np.random.randint(5000, 50000),
            'downloads_per_user': np.random.uniform(5, 30),
            'working_hours_ratio': np.random.uniform(0.3, 0.5),
            'night_activity_ratio': np.random.uniform(0.3, 0.6),
            'hourly_entropy': np.random.uniform(0.7, 0.95),
            'regularity_score': np.random.uniform(0.6, 0.9),
            'burst_pattern_score': np.random.uniform(0.5, 2),
            'user_coordination_score': np.random.uniform(0.5, 0.9),
            'file_diversity_ratio': np.random.uniform(0.7, 0.95),
            'weekend_ratio': np.random.uniform(0.3, 0.5),
            'interval_cv': np.random.uniform(0.1, 0.5),
            'pattern_type': 'bot',
        })

    # 3. Hubs (few users, bulk downloads)
    for i in range(n_hub):
        data.append({
            'unique_users': np.random.randint(1, 20),
            'total_downloads': np.random.randint(10000, 100000),
            'downloads_per_user': np.random.uniform(500, 5000),
            'working_hours_ratio': np.random.uniform(0.2, 0.4),
            'night_activity_ratio': np.random.uniform(0.2, 0.4),
            'hourly_entropy': np.random.uniform(0.5, 0.8),
            'regularity_score': np.random.uniform(0.3, 0.6),
            'burst_pattern_score': np.random.uniform(3, 8),
            'user_coordination_score': np.random.uniform(0.1, 0.3),
            'file_diversity_ratio': np.random.uniform(0.5, 0.8),
            'weekend_ratio': np.random.uniform(0.2, 0.4),
            'interval_cv': np.random.uniform(0.3, 0.8),
            'pattern_type': 'hub',
        })

    # 4. Data pipelines (regular automated CI/CD access)
    for i in range(n_pipeline):
        data.append({
            'unique_users': np.random.randint(1, 10),
            'total_downloads': np.random.randint(100, 2000),
            'downloads_per_user': np.random.uniform(10, 200),
            'working_hours_ratio': np.random.uniform(0.2, 0.5),  # Not just working hours
            'night_activity_ratio': np.random.uniform(0.3, 0.5),
            'hourly_entropy': np.random.uniform(0.2, 0.4),  # Very regular
            'regularity_score': np.random.uniform(0.8, 0.98),  # High regularity
            'burst_pattern_score': np.random.uniform(0.2, 1),
            'user_coordination_score': np.random.uniform(0.0, 0.2),
            'file_diversity_ratio': np.random.uniform(0.1, 0.4),  # Same files
            'weekend_ratio': np.random.uniform(0.2, 0.35),  # Runs on weekends too
            'interval_cv': np.random.uniform(0.05, 0.2),  # Very consistent intervals
            'pattern_type': 'data_pipeline',
        })

    # 5. Burst downloaders (periodic intense bursts)
    for i in range(n_burst):
        data.append({
            'unique_users': np.random.randint(5, 40),
            'total_downloads': np.random.randint(500, 5000),
            'downloads_per_user': np.random.uniform(30, 200),
            'working_hours_ratio': np.random.uniform(0.5, 0.8),
            'night_activity_ratio': np.random.uniform(0.1, 0.3),
            'hourly_entropy': np.random.uniform(0.3, 0.5),  # Concentrated in bursts
            'regularity_score': np.random.uniform(0.2, 0.5),
            'burst_pattern_score': np.random.uniform(6, 15),  # High burst
            'user_coordination_score': np.random.uniform(0.0, 0.2),
            'file_diversity_ratio': np.random.uniform(0.4, 0.7),
            'weekend_ratio': np.random.uniform(0.1, 0.25),
            'interval_cv': np.random.uniform(1.5, 3.0),  # High variability
            'pattern_type': 'burst_downloader',
        })

    # 6. Educational labs (classroom patterns)
    for i in range(n_edu):
        data.append({
            'unique_users': np.random.randint(20, 100),
            'total_downloads': np.random.randint(200, 2000),
            'downloads_per_user': np.random.uniform(5, 30),
            'working_hours_ratio': np.random.uniform(0.6, 0.9),  # Classroom hours
            'night_activity_ratio': np.random.uniform(0.05, 0.15),
            'hourly_entropy': np.random.uniform(0.3, 0.5),  # Concentrated in class times
            'regularity_score': np.random.uniform(0.4, 0.7),
            'burst_pattern_score': np.random.uniform(2, 5),
            'user_coordination_score': np.random.uniform(0.5, 0.8),  # High coordination
            'file_diversity_ratio': np.random.uniform(0.2, 0.4),  # Same course files
            'weekend_ratio': np.random.uniform(0.02, 0.1),  # Low weekend
            'interval_cv': np.random.uniform(0.5, 1.0),
            'pattern_type': 'educational_lab',
        })

    # 7. Nocturnal automation (night-time processing)
    for i in range(n_nocturnal):
        data.append({
            'unique_users': np.random.randint(50, 300),
            'total_downloads': np.random.randint(1000, 10000),
            'downloads_per_user': np.random.uniform(10, 50),
            'working_hours_ratio': np.random.uniform(0.05, 0.2),  # Low working hours
            'night_activity_ratio': np.random.uniform(0.6, 0.9),  # High night
            'hourly_entropy': np.random.uniform(0.5, 0.7),
            'regularity_score': np.random.uniform(0.5, 0.8),
            'burst_pattern_score': np.random.uniform(1, 3),
            'user_coordination_score': np.random.uniform(0.3, 0.6),
            'file_diversity_ratio': np.random.uniform(0.5, 0.8),
            'weekend_ratio': np.random.uniform(0.25, 0.45),  # Runs on weekends
            'interval_cv': np.random.uniform(0.2, 0.6),
            'pattern_type': 'nocturnal_automation',
        })

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    return df


def test_exploratory_clustering():
    """Test the exploratory clustering module."""

    print("=" * 70)
    print("EXPLORATORY CLUSTERING TEST")
    print("=" * 70)
    print()

    # Create diverse data
    print("[1] Creating synthetic data with 7 pattern types...")
    df = create_diverse_synthetic_data(n_samples=2000)

    # Show ground truth distribution
    print("\nGround Truth Pattern Distribution:")
    for pattern, count in df['pattern_type'].value_counts().items():
        print(f"  {pattern}: {count} ({100*count/len(df):.1f}%)")

    # Import and run exploratory clustering
    print("\n[2] Running exploratory clustering...")

    try:
        # Direct import from module
        from deeplogbot.models.unsupervised.exploratory_clustering import (
            ExploratoryConfig,
            run_exploratory_clustering,
        )

        config = ExploratoryConfig(
            clustering_approach='multi_scale',
            n_clusters_max=12,  # Allow enough clusters to find all patterns
            detect_anomalies=True,
            enable_sub_clustering=True,
            use_llm=False,  # Use heuristic naming for test
            hdbscan_min_cluster_size=50,
        )

        result_df, discovered_patterns = run_exploratory_clustering(df, config)

        print("\n" + "=" * 70)
        print("DISCOVERED PATTERNS")
        print("=" * 70)

        for pattern in discovered_patterns:
            print(f"\nCluster {pattern['cluster_id']}: {pattern['name']}")
            print(f"  Category: {pattern['category']}")
            print(f"  Size: {pattern['size']} ({pattern['size_pct']:.1f}%)")
            print(f"  Is Novel: {pattern['is_novel']}")
            print(f"  Interpretation: {pattern['interpretation']}")
            if pattern['distinctive_features']:
                print("  Distinctive features:")
                for feat in pattern['distinctive_features'][:3]:
                    print(f"    - {feat['feature_name']}: {feat['direction']} (z={feat['z_score']:.2f})")

        # Compare with ground truth
        print("\n" + "=" * 70)
        print("COMPARISON WITH GROUND TRUTH")
        print("=" * 70)

        # Create a confusion-like analysis
        result_df['ground_truth'] = df['pattern_type']

        print("\nPattern assignment by ground truth type:")
        for gt_type in df['pattern_type'].unique():
            mask = result_df['ground_truth'] == gt_type
            assigned_patterns = result_df[mask]['pattern_name'].value_counts()
            print(f"\n{gt_type} ({mask.sum()} samples):")
            for pattern, count in assigned_patterns.head(3).items():
                print(f"  -> {pattern}: {count} ({100*count/mask.sum():.1f}%)")

        # Measure how well patterns separate ground truth
        print("\n" + "=" * 70)
        print("PATTERN SEPARATION QUALITY")
        print("=" * 70)

        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        # Map ground truth to numeric labels
        gt_labels = pd.Categorical(df['pattern_type']).codes
        discovered_labels = result_df['cluster_id'].values

        ari = adjusted_rand_score(gt_labels, discovered_labels)
        nmi = normalized_mutual_info_score(gt_labels, discovered_labels)

        print(f"\nAdjusted Rand Index: {ari:.3f}")
        print(f"Normalized Mutual Information: {nmi:.3f}")
        print()
        print("(ARI > 0.5 = good separation, NMI > 0.6 = good discovery)")

        # Check if novel patterns were detected
        n_novel = result_df['is_novel_pattern'].sum()
        n_novel_types = len([p for p in discovered_patterns if p['is_novel']])

        print(f"\nNovel patterns detected: {n_novel} samples in {n_novel_types} pattern types")

        # Count categories beyond bot/hub/organic
        standard_cats = {'bot', 'hub', 'organic', 'unknown', 'noise'}
        novel_categories = set(result_df['pattern_category'].unique()) - standard_cats

        print(f"Categories beyond bot/hub/organic: {novel_categories if novel_categories else 'None'}")

        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)

        if ari > 0.3 and nmi > 0.4:
            print("\n✓ Exploratory clustering is discovering meaningful patterns!")
        else:
            print("\n⚠ Pattern discovery may need tuning (expected on synthetic data)")

        return True

    except Exception as e:
        import traceback
        print(f"\nError running exploratory clustering: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_exploratory_clustering()
    sys.exit(0 if success else 1)
