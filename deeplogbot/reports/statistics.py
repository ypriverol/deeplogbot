"""Statistics computation module for bot detection reports.

This module computes comprehensive statistics including:
- Classification statistics
- Feature statistics (mean, std, percentiles)
- Temporal statistics
- Geographic statistics
- Feature importance analysis
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

from ..utils import logger


class StatisticsCalculator:
    """Calculate comprehensive statistics for bot detection analysis."""

    def __init__(self, df: pd.DataFrame, classification_method: str = 'rules'):
        """
        Initialize statistics calculator.

        Args:
            df: Analysis DataFrame with features and classifications
            classification_method: 'rules' or 'deep'
        """
        self.df = df
        self.classification_method = classification_method
        self.stats: Dict[str, Any] = {}

    def compute_all(self) -> Dict[str, Any]:
        """Compute all statistics and return as dictionary."""
        self.stats = {
            'classification': self._compute_classification_stats(),
            'features': self._compute_feature_stats(),
            'temporal': self._compute_temporal_stats(),
            'geographic': self._compute_geographic_stats(),
            'downloads': self._compute_download_stats(),
        }

        if self.classification_method == 'deep':
            self.stats['hierarchical'] = self._compute_hierarchical_stats()

        return self.stats

    def _compute_classification_stats(self) -> Dict[str, Any]:
        """Compute classification-related statistics."""
        stats = {
            'total_locations': len(self.df),
            'anomalous_locations': int(self.df['is_anomaly'].sum()) if 'is_anomaly' in self.df.columns else 0,
        }

        # Bot statistics (using hierarchical classification)
        if 'automation_category' in self.df.columns:
            bot_mask = self.df['automation_category'] == 'bot'
            stats['bot_locations'] = int(bot_mask.sum())
            stats['bot_percentage'] = round(bot_mask.mean() * 100, 2)
            if 'total_downloads' in self.df.columns:
                stats['bot_downloads'] = int(self.df.loc[bot_mask, 'total_downloads'].sum())
                stats['bot_downloads_percentage'] = round(
                    stats['bot_downloads'] / self.df['total_downloads'].sum() * 100, 2
                ) if self.df['total_downloads'].sum() > 0 else 0

        # Hub statistics (using hierarchical classification)
        hub_subcategories = {'mirror', 'institutional_hub', 'data_aggregator'}
        if 'subcategory' in self.df.columns:
            hub_mask = self.df['subcategory'].isin(hub_subcategories)
            stats['hub_locations'] = int(hub_mask.sum())
            stats['hub_percentage'] = round(hub_mask.mean() * 100, 2)
            if 'total_downloads' in self.df.columns:
                stats['hub_downloads'] = int(self.df.loc[hub_mask, 'total_downloads'].sum())
                stats['hub_downloads_percentage'] = round(
                    stats['hub_downloads'] / self.df['total_downloads'].sum() * 100, 2
                ) if self.df['total_downloads'].sum() > 0 else 0

        # User category breakdown (for deep method)
        if 'user_category' in self.df.columns:
            category_counts = self.df['user_category'].value_counts().to_dict()
            stats['category_counts'] = category_counts
            stats['category_percentages'] = {
                k: round(v / len(self.df) * 100, 2) for k, v in category_counts.items()
            }

        return stats

    def _compute_hierarchical_stats(self) -> Dict[str, Any]:
        """Compute hierarchical classification statistics (deep method)."""
        stats = {}

        if 'behavior_type' not in self.df.columns:
            return stats

        total = len(self.df)

        # Level 1: Behavior type
        bt_counts = self.df['behavior_type'].value_counts().to_dict()
        stats['behavior_type'] = {
            'counts': bt_counts,
            'percentages': {k: round(v / total * 100, 2) for k, v in bt_counts.items()},
        }

        if 'total_downloads' in self.df.columns:
            stats['behavior_type']['downloads'] = {
                bt: int(self.df[self.df['behavior_type'] == bt]['total_downloads'].sum())
                for bt in bt_counts.keys()
            }

        # Level 2: Automation category
        if 'automation_category' in self.df.columns:
            automated_df = self.df[self.df['behavior_type'] == 'automated']
            if len(automated_df) > 0:
                ac_counts = automated_df['automation_category'].value_counts().to_dict()
                stats['automation_category'] = {
                    'counts': ac_counts,
                    'percentages': {k: round(v / len(automated_df) * 100, 2) for k, v in ac_counts.items()},
                    'percentages_of_total': {k: round(v / total * 100, 2) for k, v in ac_counts.items()},
                }

                if 'total_downloads' in self.df.columns:
                    stats['automation_category']['downloads'] = {
                        ac: int(automated_df[automated_df['automation_category'] == ac]['total_downloads'].sum())
                        for ac in ac_counts.keys()
                    }

        # Level 3: Subcategories
        if 'subcategory' in self.df.columns:
            subcat_counts = self.df['subcategory'].value_counts().to_dict()
            stats['subcategory'] = {
                'counts': subcat_counts,
                'percentages': {k: round(v / total * 100, 2) for k, v in subcat_counts.items()},
            }

            # Group by parent
            stats['subcategory']['by_parent'] = {}
            for parent in ['organic', 'bot', 'legitimate_automation']:
                if parent in ['organic']:
                    parent_df = self.df[self.df['behavior_type'] == parent]
                else:
                    parent_df = self.df[self.df['automation_category'] == parent]

                if len(parent_df) > 0 and 'subcategory' in parent_df.columns:
                    stats['subcategory']['by_parent'][parent] = parent_df['subcategory'].value_counts().to_dict()

        return stats

    def _compute_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for each feature."""
        feature_stats = {}

        # List of features to analyze
        features = [
            'unique_users', 'downloads_per_user', 'total_downloads',
            'working_hours_ratio', 'night_activity_ratio', 'hourly_entropy',
            'regularity_score', 'anomaly_score', 'user_coordination_score',
            'interval_cv', 'file_diversity_ratio', 'burst_pattern_score',
        ]

        for feature in features:
            if feature not in self.df.columns:
                continue

            data = self.df[feature].dropna()
            if len(data) == 0:
                continue

            feature_stats[feature] = {
                'count': len(data),
                'mean': round(float(data.mean()), 4),
                'std': round(float(data.std()), 4),
                'min': round(float(data.min()), 4),
                'max': round(float(data.max()), 4),
                'median': round(float(data.median()), 4),
                'q25': round(float(data.quantile(0.25)), 4),
                'q75': round(float(data.quantile(0.75)), 4),
                'q95': round(float(data.quantile(0.95)), 4),
            }

            # Add per-category stats if classification is available
            if 'automation_category' in self.df.columns:
                bot_mask = self.df['automation_category'] == 'bot'
                bot_data = self.df[bot_mask][feature].dropna()
                normal_data = self.df[~bot_mask][feature].dropna()

                if len(bot_data) > 0:
                    feature_stats[feature]['bot_mean'] = round(float(bot_data.mean()), 4)
                    feature_stats[feature]['bot_std'] = round(float(bot_data.std()), 4)

                if len(normal_data) > 0:
                    feature_stats[feature]['normal_mean'] = round(float(normal_data.mean()), 4)
                    feature_stats[feature]['normal_std'] = round(float(normal_data.std()), 4)

        return feature_stats

    def _compute_temporal_stats(self) -> Dict[str, Any]:
        """Compute temporal-related statistics."""
        stats = {}

        # Years span statistics
        if 'years_span' in self.df.columns:
            years_data = self.df['years_span'].dropna()
            stats['years_span'] = {
                'mean': round(float(years_data.mean()), 2),
                'median': round(float(years_data.median()), 2),
                'max': int(years_data.max()),
                'single_year_locations': int((years_data == 1).sum()),
                'multi_year_locations': int((years_data > 1).sum()),
            }

        # Latest year concentration
        if 'fraction_latest_year' in self.df.columns:
            latest_data = self.df['fraction_latest_year'].dropna()
            stats['latest_year_fraction'] = {
                'mean': round(float(latest_data.mean()), 3),
                'median': round(float(latest_data.median()), 3),
                'high_concentration': int((latest_data > 0.8).sum()),
                'very_high_concentration': int((latest_data > 0.95).sum()),
            }

        # New locations
        if 'is_new_location' in self.df.columns:
            stats['new_locations'] = {
                'count': int(self.df['is_new_location'].sum()),
                'percentage': round(self.df['is_new_location'].mean() * 100, 2),
            }

        # Working hours patterns
        if 'working_hours_ratio' in self.df.columns:
            wh_data = self.df['working_hours_ratio'].dropna()
            stats['working_hours'] = {
                'mean': round(float(wh_data.mean()), 3),
                'median': round(float(wh_data.median()), 3),
                'daytime_dominant': int((wh_data > 0.6).sum()),
                'nighttime_dominant': int((wh_data < 0.3).sum()),
            }

        return stats

    def _compute_geographic_stats(self) -> Dict[str, Any]:
        """Compute geographic statistics."""
        stats = {}

        if 'country' not in self.df.columns:
            return stats

        # Country distribution
        country_counts = self.df['country'].value_counts()
        stats['countries'] = {
            'total': len(country_counts),
            'top_10': country_counts.head(10).to_dict(),
        }

        # Downloads by country
        if 'total_downloads' in self.df.columns and 'country' in self.df.columns:
            country_downloads = self.df.groupby('country')['total_downloads'].sum().sort_values(ascending=False)
            stats['downloads_by_country'] = {
                'top_10': country_downloads.head(10).to_dict(),
            }

        # Bots by country (using hierarchical classification)
        if 'automation_category' in self.df.columns and 'country' in self.df.columns:
            bot_mask = self.df['automation_category'] == 'bot'
            country_bot_counts = self.df[bot_mask]['country'].value_counts()
            stats['bots_by_country'] = {
                'top_10': country_bot_counts.head(10).to_dict(),
            }

            # Bot percentage by country (min 10 locations)
            if 'geo_location' in self.df.columns:
                df_with_bot = self.df.copy()
                df_with_bot['_is_bot'] = (df_with_bot['automation_category'] == 'bot').astype(int)
                country_stats = df_with_bot.groupby('country').agg({
                    '_is_bot': 'sum',
                    'geo_location': 'count'
                }).rename(columns={'geo_location': 'total', '_is_bot': 'bot_count'})
                country_stats['bot_pct'] = country_stats['bot_count'] / country_stats['total'] * 100
                country_stats = country_stats[country_stats['total'] >= 10]
                stats['bot_percentage_by_country'] = {
                    'top_10': country_stats.nlargest(10, 'bot_pct')['bot_pct'].round(2).to_dict(),
                }

        return stats

    def _compute_download_stats(self) -> Dict[str, Any]:
        """Compute download-related statistics."""
        stats = {}

        if 'total_downloads' not in self.df.columns:
            return stats

        total = self.df['total_downloads'].sum()
        stats['total_downloads'] = int(total)

        # Downloads per user statistics
        if 'downloads_per_user' in self.df.columns:
            dpu = self.df['downloads_per_user'].dropna()
            stats['downloads_per_user'] = {
                'mean': round(float(dpu.mean()), 2),
                'median': round(float(dpu.median()), 2),
                'max': round(float(dpu.max()), 2),
                'very_high': int((dpu > 500).sum()),
                'very_low': int((dpu < 10).sum()),
            }

        # Unique users statistics
        if 'unique_users' in self.df.columns:
            users = self.df['unique_users'].dropna()
            stats['unique_users'] = {
                'total': int(users.sum()),
                'mean_per_location': round(float(users.mean()), 2),
                'median_per_location': round(float(users.median()), 2),
                'max_per_location': int(users.max()),
                'single_user_locations': int((users == 1).sum()),
                'high_user_locations': int((users > 1000).sum()),
            }

        return stats

    def compute_feature_importance(self, target_col: str = 'automation_category',
                                   features: List[str] = None) -> Dict[str, float]:
        """
        Compute feature importance using correlation and statistical tests.

        This provides a quick estimate of feature importance without training a model.
        """
        if target_col not in self.df.columns:
            return {}

        if features is None:
            features = [
                'unique_users', 'downloads_per_user', 'working_hours_ratio',
                'night_activity_ratio', 'hourly_entropy', 'regularity_score',
                'anomaly_score', 'user_coordination_score', 'fraction_latest_year',
                'spike_ratio', 'interval_cv', 'burst_pattern_score',
            ]

        importance = {}
        # Convert categorical target to binary (bot vs not bot)
        if target_col == 'automation_category':
            target = (self.df[target_col] == 'bot').astype(float)
        else:
            target = self.df[target_col].astype(float)

        for feature in features:
            if feature not in self.df.columns:
                continue

            data = self.df[feature].dropna()
            if len(data) < 10:
                continue

            # Compute point-biserial correlation (correlation with binary target)
            try:
                aligned_target = target.loc[data.index]
                corr = data.corr(aligned_target)
                if not np.isnan(corr):
                    importance[feature] = round(float(corr), 4)
            except Exception:
                pass

        # Sort by absolute importance
        importance = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

        return importance

    def compute_discriminative_features(self, category1: str = 'bot',
                                        category2: str = 'normal') -> Dict[str, Dict[str, float]]:
        """
        Compute which features best discriminate between two categories.

        Returns mean difference and effect size for each feature.
        """
        discriminative = {}

        # Define category masks
        if category1 == 'bot' and 'automation_category' in self.df.columns:
            mask1 = self.df['automation_category'] == 'bot'
        elif category1 in self.df.get('behavior_type', pd.Series()).values:
            mask1 = self.df['behavior_type'] == category1
        else:
            return {}

        if category2 == 'normal':
            mask2 = ~mask1
        elif category2 in self.df.get('behavior_type', pd.Series()).values:
            mask2 = self.df['behavior_type'] == category2
        else:
            mask2 = ~mask1

        features = [
            'unique_users', 'downloads_per_user', 'working_hours_ratio',
            'night_activity_ratio', 'regularity_score', 'anomaly_score',
        ]

        for feature in features:
            if feature not in self.df.columns:
                continue

            data1 = self.df.loc[mask1, feature].dropna()
            data2 = self.df.loc[mask2, feature].dropna()

            if len(data1) < 5 or len(data2) < 5:
                continue

            mean1, mean2 = data1.mean(), data2.mean()
            std_pooled = np.sqrt((data1.std()**2 + data2.std()**2) / 2)

            if std_pooled > 0:
                effect_size = (mean1 - mean2) / std_pooled
            else:
                effect_size = 0

            discriminative[feature] = {
                f'{category1}_mean': round(float(mean1), 4),
                f'{category2}_mean': round(float(mean2), 4),
                'difference': round(float(mean1 - mean2), 4),
                'effect_size': round(float(effect_size), 4),
            }

        return discriminative

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of key statistics."""
        if not self.stats:
            self.compute_all()

        summary = {
            'total_locations': self.stats['classification']['total_locations'],
            'bot_locations': self.stats['classification'].get('bot_locations', 0),
            'hub_locations': self.stats['classification'].get('hub_locations', 0),
            'bot_percentage': self.stats['classification'].get('bot_percentage', 0),
            'total_downloads': self.stats['downloads'].get('total_downloads', 0),
        }

        if self.classification_method == 'deep' and 'hierarchical' in self.stats:
            hier = self.stats['hierarchical']
            if 'behavior_type' in hier:
                summary['organic_locations'] = hier['behavior_type']['counts'].get('organic', 0)
                summary['automated_locations'] = hier['behavior_type']['counts'].get('automated', 0)

        return summary
