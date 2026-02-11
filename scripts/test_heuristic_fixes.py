#!/usr/bin/env python3
"""Test script to verify the data-driven unsupervised classification.

This tests that the classification uses relative/percentile-based thresholds
computed from data rather than hardcoded values.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from deeplogbot.models.unsupervised.advanced_discovery import (
    classify_location_heuristic,
    compute_data_statistics,
    DataStatistics
)


def create_test_dataset():
    """Create a synthetic dataset with known distributions."""
    import numpy as np
    np.random.seed(42)

    # Create a dataset with realistic distributions
    n = 1000
    data = {
        'unique_users': np.concatenate([
            np.random.lognormal(1, 1, 800),  # Most locations have few users
            np.random.lognormal(6, 1, 150),  # Some have many users
            np.random.lognormal(9, 0.5, 50),  # Few have very many users
        ]),
        'downloads_per_user': np.concatenate([
            np.random.lognormal(1, 1, 850),  # Most have low dpu
            np.random.lognormal(4, 1, 100),  # Some have high dpu (hubs)
            np.random.lognormal(0, 0.5, 50),  # Some have very low dpu (bots)
        ]),
        'total_downloads': np.random.lognormal(3, 2, n),
        'working_hours_ratio': np.clip(np.random.normal(0.4, 0.15, n), 0, 1),
        'night_activity_ratio': np.clip(np.random.normal(0.25, 0.15, n), 0, 1),
        'user_coordination_score': np.abs(np.random.normal(1, 2, n)),
        'burst_pattern_score': np.abs(np.random.normal(2, 3, n)),
        'access_regularity': np.clip(np.random.normal(0.2, 0.15, n), 0, 1),
    }
    return pd.DataFrame(data)


def test_data_driven_statistics():
    """Test that statistics are computed from data."""
    df = create_test_dataset()
    stats = compute_data_statistics(df)

    print("\n=== Test: Data-Driven Statistics ===")
    print(f"Users: p25={stats.users_p25:.1f}, p50={stats.users_p50:.1f}, p75={stats.users_p75:.1f}, p95={stats.users_p95:.1f}")
    print(f"DPU: p25={stats.dpu_p25:.1f}, p50={stats.dpu_p50:.1f}, p75={stats.dpu_p75:.1f}, p95={stats.dpu_p95:.1f}")
    print(f"Downloads: p25={stats.downloads_p25:.1f}, p50={stats.downloads_p50:.1f}, p95={stats.downloads_p95:.1f}")

    # Verify statistics are computed from data (not default values)
    assert stats.users_p50 != 5.0 or stats.users_p95 != 500.0, "Statistics should be computed from data"
    print("PASSED: Statistics computed from data")


def test_relative_bot_detection():
    """Test that bot detection uses relative thresholds."""
    df = create_test_dataset()
    stats = compute_data_statistics(df)

    print("\n=== Test: Relative Bot Detection ===")

    # Create a location that's in the top 5% users with bottom 25% dpu
    # This should be detected as bot regardless of absolute values
    row = pd.Series({
        'unique_users': stats.users_p95 * 1.5,  # Above 95th percentile
        'downloads_per_user': stats.dpu_p25 * 0.5,  # Below 25th percentile
        'total_downloads': stats.downloads_p75,  # Significant volume
        'working_hours_ratio': 0.3,
        'night_activity_ratio': 0.4,
        'access_regularity': 0.1,
        'user_coordination_score': stats.coord_mean,
        'hourly_entropy': 0.5,
        'burst_pattern_score': stats.burst_mean,
        'bot_composite_score': 0,
        'is_anomaly': False,
    })

    result = classify_location_heuristic(row, stats=stats)
    print(f"Input: users={row['unique_users']:.0f} (>p95), dpu={row['downloads_per_user']:.1f} (<p25)")
    print(f"Result: {result['behavior_type']}/{result['subcategory']}")
    print(f"Scores: {result['scores']}")

    assert result['automation_category'] == 'bot', f"Expected bot, got {result['automation_category']}"
    print("PASSED: High-users/low-dpu pattern detected as bot")


def test_relative_hub_detection():
    """Test that hub detection uses relative thresholds."""
    df = create_test_dataset()
    stats = compute_data_statistics(df)

    print("\n=== Test: Relative Hub Detection ===")

    # Create a location with top 5% downloads_per_user
    row = pd.Series({
        'unique_users': stats.users_p50,  # Median users
        'downloads_per_user': stats.dpu_p95 * 1.5,  # Above 95th percentile
        'total_downloads': stats.downloads_p90,
        'working_hours_ratio': 0.5,
        'night_activity_ratio': 0.2,
        'access_regularity': 0.3,
        'user_coordination_score': stats.coord_mean,
        'hourly_entropy': 0.6,
        'burst_pattern_score': stats.burst_mean,
        'bot_composite_score': 0,
        'is_anomaly': False,
    })

    result = classify_location_heuristic(row, stats=stats)
    print(f"Input: dpu={row['downloads_per_user']:.0f} (>p95)")
    print(f"Result: {result['behavior_type']}/{result['subcategory']}")
    print(f"Scores: {result['scores']}")

    assert result['automation_category'] == 'legitimate_automation', \
        f"Expected legitimate_automation, got {result['automation_category']}"
    print("PASSED: High-dpu pattern detected as hub")


def test_relative_organic_detection():
    """Test that organic detection uses relative thresholds."""
    df = create_test_dataset()
    stats = compute_data_statistics(df)

    print("\n=== Test: Relative Organic Detection ===")

    # Create a "typical" location near median in all features
    row = pd.Series({
        'unique_users': stats.users_p50,  # Median
        'downloads_per_user': stats.dpu_p50,  # Median
        'total_downloads': stats.downloads_p50,  # Median
        'working_hours_ratio': stats.whr_mean + stats.whr_std * 0.5,  # Slightly above average
        'night_activity_ratio': stats.night_mean - stats.night_std * 0.5,  # Slightly below average
        'access_regularity': stats.reg_p50,  # Median
        'user_coordination_score': stats.coord_p50 if hasattr(stats, 'coord_p50') else stats.coord_mean,
        'hourly_entropy': 0.6,
        'burst_pattern_score': stats.burst_mean,
        'bot_composite_score': 0,
        'is_anomaly': False,
    })

    result = classify_location_heuristic(row, stats=stats)
    print(f"Input: near-median in all features")
    print(f"Result: {result['behavior_type']}/{result['subcategory']}")
    print(f"Scores: {result['scores']}")

    assert result['behavior_type'] == 'organic', f"Expected organic, got {result['behavior_type']}"
    print("PASSED: Typical/median pattern detected as organic")


def test_small_user_relative():
    """Test that small users (relative to dataset) are presumed organic."""
    df = create_test_dataset()
    stats = compute_data_statistics(df)

    print("\n=== Test: Small User (Relative) Detection ===")

    # Create a location below median in users and downloads
    row = pd.Series({
        'unique_users': stats.users_p25 * 0.5,  # Well below 25th percentile
        'downloads_per_user': stats.dpu_p50,
        'total_downloads': stats.downloads_p25 * 0.5,  # Well below 25th percentile
        'working_hours_ratio': 0.1,  # Low working hours (could be different timezone)
        'night_activity_ratio': 0.6,  # High night activity
        'access_regularity': 0.5,
        'user_coordination_score': stats.coord_mean,
        'hourly_entropy': 0.5,
        'burst_pattern_score': stats.burst_mean,
        'bot_composite_score': 0,
        'is_anomaly': False,
    })

    result = classify_location_heuristic(row, stats=stats)
    print(f"Input: users={row['unique_users']:.1f} (<p25), downloads={row['total_downloads']:.1f} (<p25)")
    print(f"Result: {result['behavior_type']}/{result['subcategory']}")
    print(f"Scores: {result['scores']}")

    # Small users should be presumed organic even with unusual timing
    assert result['behavior_type'] == 'organic', f"Expected organic for small user, got {result['behavior_type']}"
    print("PASSED: Small user (relative) presumed organic")


def test_no_hardcoded_values():
    """Test that classification adapts to different data distributions."""
    print("\n=== Test: Adaptation to Different Distributions ===")

    # Dataset 1: Small-scale (e.g., niche service)
    df1 = pd.DataFrame({
        'unique_users': [1, 2, 3, 5, 10, 15, 20, 50],
        'downloads_per_user': [5, 10, 15, 20, 25, 30, 50, 100],
        'total_downloads': [10, 20, 50, 100, 200, 500, 1000, 5000],
        'working_hours_ratio': [0.4] * 8,
        'night_activity_ratio': [0.2] * 8,
        'user_coordination_score': [1] * 8,
        'burst_pattern_score': [2] * 8,
    })
    stats1 = compute_data_statistics(df1)

    # Dataset 2: Large-scale (e.g., major platform)
    df2 = pd.DataFrame({
        'unique_users': [100, 500, 1000, 5000, 10000, 50000, 100000, 500000],
        'downloads_per_user': [5, 10, 15, 20, 25, 30, 50, 100],
        'total_downloads': [1000, 5000, 20000, 100000, 500000, 1000000, 5000000, 50000000],
        'working_hours_ratio': [0.4] * 8,
        'night_activity_ratio': [0.2] * 8,
        'user_coordination_score': [1] * 8,
        'burst_pattern_score': [2] * 8,
    })
    stats2 = compute_data_statistics(df2)

    print(f"Small-scale dataset: users_p50={stats1.users_p50:.0f}, users_p95={stats1.users_p95:.0f}")
    print(f"Large-scale dataset: users_p50={stats2.users_p50:.0f}, users_p95={stats2.users_p95:.0f}")

    # The statistics should be different
    assert stats1.users_p50 != stats2.users_p50, "Statistics should differ between datasets"
    assert stats1.users_p95 != stats2.users_p95, "Statistics should differ between datasets"

    print("PASSED: Algorithm adapts to different data distributions")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Data-Driven Unsupervised Classification")
    print("=" * 70)

    tests = [
        test_data_driven_statistics,
        test_relative_bot_detection,
        test_relative_hub_detection,
        test_relative_organic_detection,
        test_small_user_relative,
        test_no_hardcoded_values,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
