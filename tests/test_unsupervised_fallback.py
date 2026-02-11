"""Tests for unsupervised classification fallback mechanisms.

Validates that the heuristic and ensemble classification functions
work correctly when LLM pattern discovery is unavailable.
"""

import pytest
import pandas as pd
import numpy as np

from deeplogbot.models.unsupervised.advanced_discovery import (
    classify_location_heuristic,
    classify_cluster_profile,
    classify_with_ensemble,
    parse_llm_json_response,
)


class TestHeuristicClassification:
    """Tests for classify_location_heuristic function."""

    def test_ground_truth_bot(self):
        """Ground truth bot: many users, very low downloads per user."""
        row = pd.Series({
            'unique_users': 15000,
            'downloads_per_user': 5,
            'total_downloads': 75000,
            'working_hours_ratio': 0.15,
            'night_activity_ratio': 0.6,
            'access_regularity': 0.8,
            'user_coordination_score': 0.7,
            'is_anomaly': 1,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'bot'
        assert result['behavior_type'] == 'automated'
        assert result['confidence'] >= 0.7
        assert 'bot' in result['subcategory'].lower()

    def test_large_scale_bot(self):
        """Large scale bot: high users, low downloads per user."""
        row = pd.Series({
            'unique_users': 6000,
            'downloads_per_user': 15,
            'total_downloads': 90000,
            'working_hours_ratio': 0.2,
            'night_activity_ratio': 0.5,
            'access_regularity': 0.6,
            'user_coordination_score': 0.5,
            'is_anomaly': 1,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'bot'
        assert result['behavior_type'] == 'automated'

    def test_bot_farm_pattern(self):
        """Bot farm: coordinated activity, many users."""
        row = pd.Series({
            'unique_users': 1500,
            'downloads_per_user': 30,
            'total_downloads': 45000,
            'working_hours_ratio': 0.25,
            'night_activity_ratio': 0.45,
            'access_regularity': 0.7,
            'user_coordination_score': 0.8,
            'burst_pattern_score': 0.6,
            'is_anomaly': 1,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'bot'

    def test_mirror_hub(self):
        """Mirror/institutional hub: very high downloads per user."""
        row = pd.Series({
            'unique_users': 5,
            'downloads_per_user': 1500,
            'total_downloads': 7500,
            'working_hours_ratio': 0.4,
            'night_activity_ratio': 0.1,
            'access_regularity': 0.3,
            'is_anomaly': 0,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'legitimate_automation'
        assert result['behavior_type'] == 'automated'
        assert 'mirror' in result['subcategory'].lower() or 'hub' in result['subcategory'].lower()

    def test_institutional_hub(self):
        """Institutional hub: high downloads per user, few users."""
        row = pd.Series({
            'unique_users': 20,
            'downloads_per_user': 600,
            'total_downloads': 12000,
            'working_hours_ratio': 0.5,
            'night_activity_ratio': 0.1,
            'is_anomaly': 0,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'legitimate_automation'

    def test_research_hub(self):
        """Research hub: moderate downloads per user, few users."""
        row = pd.Series({
            'unique_users': 50,
            'downloads_per_user': 250,
            'total_downloads': 12500,
            'working_hours_ratio': 0.6,
            'night_activity_ratio': 0.05,
            'is_anomaly': 0,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] == 'legitimate_automation'

    def test_organic_research_group(self):
        """Organic research group: few users, moderate activity, working hours."""
        row = pd.Series({
            'unique_users': 15,
            'downloads_per_user': 50,
            'total_downloads': 750,
            'working_hours_ratio': 0.6,
            'night_activity_ratio': 0.05,
            'hourly_entropy': 0.6,
            'access_regularity': 0.2,
            'is_anomaly': 0,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] is None
        assert result['behavior_type'] == 'organic'

    def test_individual_user(self):
        """Individual user: very few users, low activity."""
        row = pd.Series({
            'unique_users': 2,
            'downloads_per_user': 15,
            'total_downloads': 30,
            'working_hours_ratio': 0.7,
            'night_activity_ratio': 0.0,
            'is_anomaly': 0,
        })
        result = classify_location_heuristic(row)

        assert result['automation_category'] is None
        assert result['behavior_type'] == 'organic'

    def test_handles_missing_features(self):
        """Should handle missing features gracefully."""
        row = pd.Series({
            'unique_users': 100,
            'downloads_per_user': 50,
        })
        result = classify_location_heuristic(row)

        assert 'behavior_type' in result
        assert 'automation_category' in result
        assert 'confidence' in result

    def test_handles_nan_values(self):
        """Should handle NaN values gracefully."""
        row = pd.Series({
            'unique_users': 1000,
            'downloads_per_user': np.nan,
            'working_hours_ratio': np.nan,
            'is_anomaly': np.nan,
        })
        result = classify_location_heuristic(row)

        assert 'behavior_type' in result


class TestClusterProfileClassification:
    """Tests for classify_cluster_profile function."""

    def test_bot_cluster_profile(self):
        """Cluster with bot-like distinctive features."""
        profile = {
            'size': 5000,
            'percentage': 7.0,
            'distinctive_features': [
                {'feature': 'unique_users', 'direction': 'high', 'z_score': 3.5, 'value': 8000},
                {'feature': 'downloads_per_user', 'direction': 'low', 'z_score': -2.5, 'value': 15},
                {'feature': 'user_coordination_score', 'direction': 'high', 'z_score': 2.0, 'value': 0.7},
                {'feature': 'night_activity_ratio', 'direction': 'high', 'z_score': 1.8, 'value': 0.5},
            ],
            'features': {
                'unique_users': {'mean': 8000},
                'downloads_per_user': {'mean': 15},
            },
            'temporal_patterns': {
                'hourly': {'peak': 3, 'entropy': 0.8},
            },
        }
        result = classify_cluster_profile(profile)

        assert result['automation_category'] == 'bot'
        assert result['behavior_type'] == 'automated'

    def test_hub_cluster_profile(self):
        """Cluster with hub-like distinctive features."""
        profile = {
            'size': 100,
            'percentage': 0.1,
            'distinctive_features': [
                {'feature': 'downloads_per_user', 'direction': 'high', 'z_score': 4.0, 'value': 1000},
                {'feature': 'total_downloads', 'direction': 'high', 'z_score': 3.0, 'value': 500000},
                {'feature': 'unique_users', 'direction': 'low', 'z_score': -2.0, 'value': 10},
            ],
            'features': {
                'unique_users': {'mean': 10},
                'downloads_per_user': {'mean': 1000},
            },
            'temporal_patterns': {
                'hourly': {'peak': 14, 'entropy': 0.6},
            },
        }
        result = classify_cluster_profile(profile)

        assert result['automation_category'] == 'legitimate_automation'

    def test_organic_cluster_profile(self):
        """Cluster with organic user patterns."""
        profile = {
            'size': 10000,
            'percentage': 14.0,
            'distinctive_features': [
                {'feature': 'working_hours_ratio', 'direction': 'high', 'z_score': 2.0, 'value': 0.6},
                {'feature': 'hourly_entropy', 'direction': 'neutral', 'z_score': 0.5, 'value': 0.7},
            ],
            'features': {
                'unique_users': {'mean': 30},
                'downloads_per_user': {'mean': 50},
            },
            'temporal_patterns': {
                'hourly': {'peak': 14, 'entropy': 0.7},
            },
        }
        result = classify_cluster_profile(profile)

        assert result['automation_category'] is None
        assert result['behavior_type'] == 'organic'


class TestEnsembleClassification:
    """Tests for classify_with_ensemble function."""

    def test_ensemble_agrees_on_bot(self):
        """All sources agree it's a bot."""
        row = pd.Series({
            'unique_users': 10000,
            'downloads_per_user': 5,
            'working_hours_ratio': 0.1,
            'night_activity_ratio': 0.7,
            'access_regularity': 0.9,
            'user_coordination_score': 0.8,
            'is_anomaly': 1,
        })
        cluster_class = {
            'behavior_type': 'automated',
            'automation_category': 'bot',
            'confidence': 0.85,
        }
        pattern_class = {
            'behavior_type': 'automated',
            'automation_category': 'bot',
            'confidence': 0.9,
        }

        result = classify_with_ensemble(row, cluster_class, pattern_class)

        assert result['automation_category'] == 'bot'
        assert result['confidence'] >= 0.7

    def test_ensemble_with_only_heuristic(self):
        """Only heuristic classification available."""
        row = pd.Series({
            'unique_users': 5000,
            'downloads_per_user': 20,
            'working_hours_ratio': 0.2,
            'night_activity_ratio': 0.5,
            'is_anomaly': 1,
        })

        result = classify_with_ensemble(row, None, None)

        assert 'behavior_type' in result
        assert 'automation_category' in result

    def test_ensemble_disagreement(self):
        """Sources disagree - should use weighted voting."""
        row = pd.Series({
            'unique_users': 100,
            'downloads_per_user': 100,
            'working_hours_ratio': 0.4,
            'night_activity_ratio': 0.2,
            'is_anomaly': 0,
        })
        cluster_class = {
            'behavior_type': 'automated',
            'automation_category': 'bot',
            'confidence': 0.6,
        }
        pattern_class = {
            'behavior_type': 'organic',
            'automation_category': None,
            'confidence': 0.7,
        }

        result = classify_with_ensemble(row, cluster_class, pattern_class)

        # Result should reflect the weighted voting
        assert 'behavior_type' in result
        assert 'votes' in result


class TestLLMJsonParsing:
    """Tests for parse_llm_json_response function."""

    def test_parse_markdown_json_block(self):
        """Parse JSON in markdown code block."""
        response = '''Here's my analysis:
```json
{"is_novel": true, "confidence": 0.85, "behavior_type": "automated"}
```
That's my answer.'''

        result = parse_llm_json_response(response)

        assert result is not None
        assert result['is_novel'] is True
        assert result['confidence'] == 0.85

    def test_parse_plain_json(self):
        """Parse plain JSON object."""
        response = '{"is_novel": false, "pattern_name": "test", "confidence": 0.7}'

        result = parse_llm_json_response(response)

        assert result is not None
        assert result['is_novel'] is False
        assert result['pattern_name'] == 'test'

    def test_parse_nested_json(self):
        """Parse JSON with nested objects."""
        response = '''{"is_novel": true, "detection_rules": [{"feature": "users", "operator": ">=", "value": 100}], "confidence": 0.8}'''

        result = parse_llm_json_response(response)

        assert result is not None
        assert result['is_novel'] is True
        assert 'detection_rules' in result

    def test_parse_json_with_text(self):
        """Parse JSON embedded in explanatory text."""
        response = '''Based on my analysis, this is a bot pattern.

{"is_novel": true, "behavior_type": "automated", "confidence": 0.75}

The high user count and low downloads per user are indicative.'''

        result = parse_llm_json_response(response)

        assert result is not None
        assert result['behavior_type'] == 'automated'

    def test_parse_empty_response(self):
        """Handle empty response."""
        result = parse_llm_json_response('')
        assert result is None

        result = parse_llm_json_response(None)
        assert result is None

    def test_parse_invalid_json(self):
        """Handle invalid JSON gracefully."""
        response = 'This is not JSON at all.'
        result = parse_llm_json_response(response)
        # Should return None or extracted key-values
        # Not raise an exception


class TestIntegration:
    """Integration tests for the complete classification pipeline."""

    def test_full_pipeline_with_dataframe(self):
        """Test classification on a DataFrame of locations."""
        # Create test DataFrame
        df = pd.DataFrame({
            'unique_users': [10000, 5, 1000, 50],
            'downloads_per_user': [5, 1000, 30, 50],
            'total_downloads': [50000, 5000, 30000, 2500],
            'working_hours_ratio': [0.1, 0.5, 0.2, 0.6],
            'night_activity_ratio': [0.6, 0.1, 0.4, 0.1],
            'access_regularity': [0.8, 0.2, 0.6, 0.3],
            'user_coordination_score': [0.7, 0.1, 0.5, 0.2],
            'is_anomaly': [1, 0, 1, 0],
        })

        # Classify each row
        results = df.apply(classify_location_heuristic, axis=1)

        # Check that all rows were classified
        assert len(results) == 4

        # Check expected classifications
        assert results.iloc[0]['automation_category'] == 'bot'  # Ground truth bot
        assert results.iloc[1]['automation_category'] == 'legitimate_automation'  # Mirror
        assert results.iloc[2]['automation_category'] == 'bot'  # Bot farm
        assert results.iloc[3]['automation_category'] is None  # Organic


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
