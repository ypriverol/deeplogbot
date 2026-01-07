"""Report generation for bot detection results."""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..utils import logger, format_number, group_nearby_locations_with_llm
from ..features.schema import LogSchema


class ReportGenerator:
    """
    Generic report generator for bot detection results.
    
    Can be customized for different log formats and feature sets.
    """
    
    def __init__(self, schema: Optional[LogSchema] = None, feature_descriptions: Optional[Dict[str, str]] = None):
        """
        Initialize report generator.
        
        Args:
            schema: LogSchema defining field mappings
            feature_descriptions: Optional dict mapping feature names to descriptions
        """
        self.schema = schema
        self.feature_descriptions = feature_descriptions or self._default_feature_descriptions()
    
    def _default_feature_descriptions(self) -> Dict[str, str]:
        """Default feature descriptions (generic)."""
        return {
            'unique_users': 'Number of distinct user IDs',
            'downloads_per_user': 'Total downloads / unique users',
            'avg_users_per_hour': 'Average user density per hour',
            'max_users_per_hour': 'Peak user density',
            'user_cv': 'Coefficient of variation (pattern regularity)',
            'users_per_active_hour': 'User concentration',
            'projects_per_user': 'Download diversity',
            'hourly_download_std': 'Standard deviation of downloads across hours',
            'peak_hour_concentration': 'Fraction of downloads in busiest hour',
            'working_hours_ratio': 'Fraction of downloads during working hours',
            'hourly_entropy': 'Entropy of hourly distribution (uniformity measure)',
            'night_activity_ratio': 'Fraction of downloads during night hours',
            'yearly_entropy': 'Entropy of yearly distribution (sustained vs bursty)',
            'peak_year_concentration': 'Fraction of downloads in busiest year',
            'years_span': 'Number of years with activity',
            'downloads_per_year': 'Average downloads per year',
            'year_over_year_cv': 'Coefficient of variation across years (consistency)',
            'fraction_latest_year': 'Fraction of downloads in latest year (suspicious if high)',
            'is_new_location': 'Binary flag if location first appeared in latest year',
            'spike_ratio': 'Latest year downloads vs average of previous years',
            'years_before_latest': 'Number of years with activity before latest year',
        }
    
    def get_feature_description(self, feature_name: str) -> str:
        """Get description for a feature, with fallback."""
        return self.feature_descriptions.get(
            feature_name, 
            f'{feature_name}: Feature extracted from log data'
        )
    
    def _get_available_features(self, df: pd.DataFrame, standard_features: List[str]) -> List[str]:
        """Get list of features that are actually present in the dataframe."""
        available = [f for f in standard_features if f in df.columns]
        return available
    
    def _write_feature_list(self, f, available_features: List[str]):
        """Write feature descriptions to report."""
        f.write("Features used:\n")
        for feature in available_features:
            desc = self.get_feature_description(feature)
            f.write(f"  - {feature}: {desc}\n")
        f.write("\n")
    
    def _write_classification_rules(self, f, classification_method='rules'):
        """Write classification method description to report."""
        if classification_method.lower() == 'ml':
            f.write("Classification method: ML-based (RandomForestClassifier)\n")
            f.write("  The classifier was trained on rule-based labels and uses the following features:\n")
            f.write("  - Behavioral patterns (user counts, downloads per user, temporal patterns)\n")
            f.write("  - Anomaly scores from Isolation Forest\n")
            f.write("  - Temporal features (yearly patterns, time-of-day patterns)\n")
            f.write("  - Geographic features (country-level patterns)\n")
            f.write("  Classes: BOT (0), DOWNLOAD_HUB (1), NORMAL (2)\n\n")
        elif classification_method.lower() == 'deep':
            f.write("Classification method: Deep Architecture (Isolation Forest + Transformers)\n")
            f.write("  This method combines:\n")
            f.write("    - Isolation Forest for initial anomaly detection.\n")
            f.write("    - Transformer for sequence-based feature encoding (time-series + fixed features).\n")
            f.write("    - Rule-based classification using Transformer embeddings and original features.\n")
            f.write("  Categories: BOT, DOWNLOAD_HUB, INDEPENDENT_USER, NORMAL, OTHER\n\n")
        else:
            f.write("Classification method: Rule-based\n")
            f.write("Classification rules:\n")
            f.write("  BOT:\n")
            f.write("    Detected based on anomaly scores and user/download ratios:\n")
            f.write("    - Anomalous locations with many users and low downloads per user\n")
            f.write("    - Anomalous locations with very high user counts and moderate downloads per user\n")
            f.write("    (Note: bot rules intentionally avoid year-specific spike heuristics to remain stable over time.)\n")
            f.write("  DOWNLOAD_HUB:\n")
            f.write("    Detected based on:\n")
            f.write("    - Very high downloads per user (mirrors/single-user hubs)\n")
            f.write("    - Large total downloads with moderate downloads per user\n")
            f.write("    - Regular working hours patterns (research institutions)\n\n")
    
    def _write_summary_stats(self, f, df: pd.DataFrame, bot_locs: pd.DataFrame, 
                            hub_locs: pd.DataFrame, independent_user_locs: pd.DataFrame, 
                            other_locs: pd.DataFrame, stats: Dict[str, Any]):
        """Write summary statistics."""
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total locations analyzed: {len(df):,}\n")
        f.write(f"Anomalous locations: {df['is_anomaly'].sum():,}\n")
        f.write(f"Bot locations: {len(bot_locs):,}\n")
        f.write(f"Download hub locations: {len(hub_locs):,}\n")
        f.write(f"Independent user locations: {len(independent_user_locs):,}\n")
        f.write(f"Other/Unclassified locations: {len(other_locs):,}\n")        
        f.write(f"Total downloads: {format_number(stats['total'])}\n")
        f.write(f"Bot downloads: {format_number(stats['bots'])} ({stats['bots']/stats['total']*100:.2f}%)\n")
        f.write(f"Hub downloads: {format_number(stats['hubs'])} ({stats['hubs']/stats['total']*100:.2f}%)\n")
        f.write(f"Normal downloads: {format_number(stats['normal'])} ({stats['normal']/stats['total']*100:.2f}%)\n")
        independent_users = stats.get('independent_users', 0)
        if independent_users > 0:
            f.write(f"Independent user downloads: {format_number(independent_users)} ({independent_users/stats['total']*100:.2f}%)\n")
        other_val = stats.get('other', stats.get('other_downloads', 0))
        if other_val > 0:
            f.write(f"Other/Unclassified downloads: {format_number(other_val)} ({other_val/stats['total']*100:.2f}%)\n")    

    def _write_cluster_details(self, f, cluster_df: pd.DataFrame):
        """Write detailed information about clusters (if available)."""
        # Note: Deep architecture no longer uses DBSCAN clustering
        # This method is kept for compatibility but will be empty for deep method
        if cluster_df is not None and not cluster_df.empty:
            f.write("\n" + "=" * 80 + "\n")
            f.write("CLUSTER DETAILS\n")
            f.write("=" * 80 + "\n")
            f.write("Note: Deep architecture uses Transformer embeddings for direct classification,\n")
            f.write("not clustering. Cluster information is not available.\n\n")

    def _write_transformer_explanation(self, f, classification_method: str):
        """Write a section explaining the Transformer architecture."""
        if classification_method.lower() == 'deep':
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRANSFORMER ARCHITECTURE EXPLANATION (Deep Method)\n")
            f.write("=" * 80 + "\n")
            f.write("The Deep Classification method leverages a Transformer Encoder to process \n")
            f.write("time-series features for each geo-location. This architecture is particularly \n")
            f.write("beneficial because it can capture temporal dependencies and patterns in \n")
            f.write("the sequence of downloads over time. By considering not just static features \n")
            f.write("but also the *order* and *evolution* of downloads, the Transformer can build \n")
            f.write("richer representations of each location's behavior. This allows for: \n")
            f.write("  - Detecting subtle shifts in download patterns that might indicate bot activity.\n")
            f.write("  - Distinguishing between legitimate, evolving user behavior and static, \n")
            f.write("    anomalous patterns.\n")
            f.write("  - Providing rich embeddings that capture temporal patterns for direct classification.\n")
            f.write("    The Transformer embeddings are combined with fixed features and used with\n")
            f.write("    rule-based thresholds for classification, eliminating the need for clustering.\n")
            f.write("The Transformer uses various aggregated time-windowed features (e.g., downloads, \n")
            f.write("unique users, downloads per user per month/week) as its input sequence.\n\n")

    def _write_city_level_aggregation(self, f, df: pd.DataFrame, city_field: str = 'city'):
        """Write city-level aggregation section."""
        f.write("=" * 80 + "\n")
        f.write("CITY-LEVEL AGGREGATION (Research Hubs)\n")
        f.write("=" * 80 + "\n")
        f.write("Note: Same city may have multiple geo_locations due to GPS precision.\n")
        f.write("This view aggregates all locations within a city.\n\n")
        
        # Only aggregate by city if the field exists
        if city_field not in df.columns:
            f.write("City-level aggregation not available (city field not in data).\n\n")
            return
        
        groupby_cols = ['country']
        if city_field in df.columns:
            groupby_cols.append(city_field)
        
        city_agg = df.groupby(groupby_cols).agg({
            'unique_users': 'sum',
            'total_downloads': 'sum',
            'geo_location': 'count'
        }).reset_index()
        city_agg.columns = groupby_cols + ['total_users', 'total_downloads', 'num_locations']
        city_agg['downloads_per_user'] = city_agg['total_downloads'] / city_agg['total_users']
        
        # Filter out invalid entries
        city_agg = city_agg[
            (city_agg['total_downloads'] > 100000) &
            (city_agg['country'].notna()) &
            (~city_agg['country'].astype(str).str.contains('%{', na=False)) &
            (city_agg['country'].astype(str) != 'N/A') &
            (city_agg['country'].astype(str) != 'Unknown')
        ]
        city_agg = city_agg.sort_values('downloads_per_user', ascending=False)
        
        f.write(f"{'Country':<18} {'City':<20} {'Locs':>5} {'Users':>10} {'Downloads':>12} {'DL/User':>10}\n")
        f.write("-" * 80 + "\n")
        for _, row in city_agg.head(50).iterrows():
            city = str(row.get(city_field, 'N/A'))[:18] if city_field in row and pd.notna(row[city_field]) else 'N/A'
            f.write(f"{row['country']:<18} {city:<20} {int(row['num_locations']):>5} "
                   f"{int(row['total_users']):>10,} {int(row['total_downloads']):>12,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
    
    def _write_bot_locations(self, f, bot_locs: pd.DataFrame, city_field: str = 'city'):
        """Write bot locations section."""
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"BOT LOCATIONS ({len(bot_locs)})\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Country':<18} {'City':<22} {'Users':>10} {'DL/User':>10}\n")
        f.write("-" * 65 + "\n")
        
        # Filter out invalid bot locations
        valid_bot_locs = bot_locs[
            (bot_locs['country'].notna()) &
            (~bot_locs['country'].astype(str).str.contains('%{', na=False)) &
            (bot_locs['country'].astype(str) != 'N/A') &
            (bot_locs['country'].astype(str) != 'Unknown')
        ]
        
        for _, row in valid_bot_locs.sort_values('unique_users', ascending=False).iterrows():
            city = str(row.get(city_field, ''))[:20] if city_field in row and pd.notna(row[city_field]) and str(row[city_field]) != 'N/A' else ''
            country = str(row['country'])[:16]
            f.write(f"{country:<18} {city:<22} {int(row['unique_users']):>10,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
    
    def _write_hub_locations(self, f, hub_locs: pd.DataFrame, city_field: str = 'city'):
        """Write hub locations section."""
        # Group nearby hub locations (with option to skip LLM for speed)
        logger.info("Grouping nearby hub locations for consolidated view...")
        use_llm_grouping = os.getenv('USE_LLM_GROUPING', 'true').lower() == 'true'
        if not use_llm_grouping:
            logger.info("  LLM grouping disabled (set USE_LLM_GROUPING=true to enable)")
        location_groups = group_nearby_locations_with_llm(
            hub_locs.copy(), 
            max_distance_km=10, 
            use_llm=use_llm_grouping
        )
        
        # Create consolidated hub view
        hub_locs_grouped = hub_locs.copy()
        hub_locs_grouped['group_id'] = hub_locs_grouped['geo_location'].map(location_groups)
        
        # Aggregate grouped locations
        agg_dict = {
            'geo_location': 'count',
            'unique_users': 'sum',
            'total_downloads': 'sum',
        }
        if city_field in hub_locs_grouped.columns:
            agg_dict[city_field] = lambda x: ', '.join([str(c) for c in x.dropna().unique()[:3]])
        
        consolidated = hub_locs_grouped.groupby(['country', 'group_id']).agg(agg_dict).reset_index()
        
        # Handle column names based on what's available
        cols = ['country', 'group_id', 'num_locations', 'total_users', 'total_downloads']
        if city_field in hub_locs_grouped.columns:
            cols.append('cities')
        consolidated.columns = cols
        consolidated['downloads_per_user'] = consolidated['total_downloads'] / consolidated['total_users']
        
        # Filter out invalid entries from consolidated
        consolidated = consolidated[
            (consolidated['country'].notna()) &
            (~consolidated['country'].astype(str).str.contains('%{', na=False)) &
            (consolidated['country'].astype(str) != 'N/A') &
            (consolidated['country'].astype(str) != 'Unknown')
        ]
        
        # Get canonical location details for display
        group_to_canonical = {}
        for geo_loc, group_id in location_groups.items():
            if group_id not in group_to_canonical:
                canonical_loc = hub_locs[hub_locs['geo_location'] == group_id]
                if len(canonical_loc) > 0:
                    group_to_canonical[group_id] = {
                        city_field: canonical_loc.iloc[0].get(city_field, '') if city_field in canonical_loc.columns else '',
                        'geo_location': group_id
                    }
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"DOWNLOAD HUB LOCATIONS ({len(hub_locs)} individual, {len(consolidated)} consolidated)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Country':<18} {'Location(s)':<30} {'Locs':>5} {'Users':>10} {'DL/User':>10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in consolidated.sort_values('downloads_per_user', ascending=False).iterrows():
            canonical = group_to_canonical.get(row['group_id'], {})
            display_city = canonical.get(city_field, '')
            if pd.isna(display_city) or display_city == '' or str(display_city) == 'N/A':
                # Try to get from cities list
                if 'cities' in row and row['cities'] and str(row['cities']) != 'N/A':
                    display_city = str(row['cities']).split(',')[0].strip()
                else:
                    display_city = ''
            
            # Show grouped cities if multiple
            if row['num_locations'] > 1 and 'cities' in row and row['cities'] and str(row['cities']) != 'N/A':
                city_display = f"{display_city} ({row['cities']})"[:28]
            else:
                city_display = str(display_city)[:28] if display_city else ''
            
            country = str(row['country'])[:16]
            f.write(f"{country:<18} {city_display:<30} {int(row['num_locations']):>5} "
                   f"{int(row['total_users']):>10,} {row['downloads_per_user']:>10.1f}\n")
        
        # Also show individual locations in a separate section for reference
        f.write("\n" + "-" * 80 + "\n")
        f.write("INDIVIDUAL DOWNLOAD HUB LOCATIONS (for reference)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Country':<18} {'City':<22} {'Users':>10} {'DL/User':>10}\n")
        f.write("-" * 65 + "\n")
        
        # Filter out invalid hub locations
        valid_hub_locs = hub_locs[
            (hub_locs['country'].notna()) &
            (~hub_locs['country'].astype(str).str.contains('%{', na=False)) &
            (hub_locs['country'].astype(str) != 'N/A') &
            (hub_locs['country'].astype(str) != 'Unknown')
        ]
        
        for _, row in valid_hub_locs.sort_values('downloads_per_user', ascending=False).iterrows():
            city = str(row.get(city_field, ''))[:20] if city_field in row and pd.notna(row[city_field]) and str(row[city_field]) != 'N/A' else ''
            country = str(row['country'])[:16]
            f.write(f"{country:<18} {city:<22} {int(row['unique_users']):>10,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
    
    def _write_detailed_category_breakdown(self, f, df: pd.DataFrame):
        """Write detailed category breakdown section."""
        if 'detailed_category' not in df.columns:
            return
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED CATEGORY BREAKDOWN\n")
        f.write("=" * 80 + "\n")
        f.write("Granular classification of location usage patterns:\n\n")
        
        for category in df['detailed_category'].unique():
            if pd.isna(category):
                continue
            cat_df = df[df['detailed_category'] == category]
            if len(cat_df) == 0:
                continue
            
            f.write(f"\n{category.upper()}: {len(cat_df):,} locations\n")
            f.write(f"  Total downloads: {cat_df['total_downloads'].sum():,.0f}\n")
            f.write(f"  Avg users: {cat_df['unique_users'].mean():.1f}\n")
            f.write(f"  Avg DL/user: {cat_df['downloads_per_user'].mean():.1f}\n")
            
            # Add behavioral features if available
            if 'working_hours_ratio' in cat_df.columns:
                f.write(f"  Avg working hours ratio: {cat_df['working_hours_ratio'].mean():.2f}\n")
            if 'file_diversity_ratio' in cat_df.columns:
                f.write(f"  Avg file diversity: {cat_df['file_diversity_ratio'].mean():.2f}\n")
            if 'regularity_score' in cat_df.columns:
                f.write(f"  Avg regularity score: {cat_df['regularity_score'].mean():.2f}\n")
    
    
    def generate(self, df: pd.DataFrame, bot_locs: pd.DataFrame, hub_locs: pd.DataFrame, 
                independent_user_locs: pd.DataFrame, other_locs: pd.DataFrame, 
                stats: Dict[str, Any], output_dir: str, 
                cluster_df: Optional[pd.DataFrame] = None, # New parameter
                available_features: Optional[List[str]] = None,
                classification_method: str = 'rules') -> str:
        """
        Generate comprehensive report.
        
        Args:
            df: Full analysis dataframe
            bot_locs: Bot locations dataframe
            hub_locs: Hub locations dataframe
            stats: Statistics dictionary
            output_dir: Output directory
            available_features: List of feature names actually used (auto-detected if None)
            
        Returns:
            Path to generated report file
        """
        report_file = os.path.join(output_dir, 'bot_detection_report.txt')
        
        # Auto-detect available features if not provided
        if available_features is None:
            standard_features = list(self.feature_descriptions.keys())
            available_features = self._get_available_features(df, standard_features)
        
        # The city column is always named 'city' in the analysis DataFrame
        # (aliased in extract_location_features), regardless of the original schema field name
        city_field = 'city'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BOT AND DOWNLOAD HUB DETECTION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.schema:
                f.write(f"Log Format: {self.schema.__class__.__name__}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("METHODOLOGY\n")
            f.write("-" * 60 + "\n")
            f.write("Algorithm: Isolation Forest (unsupervised anomaly detection)\n\n")
            self._write_feature_list(f, available_features)
            self._write_classification_rules(f, classification_method)
            self._write_transformer_explanation(f, classification_method) # New call here
            
            self._write_summary_stats(f, df, bot_locs, hub_locs, independent_user_locs, other_locs, stats)
            
            if classification_method.lower() == 'deep':
                if cluster_df is not None and not cluster_df.empty:
                    self._write_cluster_details(f, cluster_df)
            
            self._write_detailed_category_breakdown(f, df)

            self._write_city_level_aggregation(f, df, city_field)
            self._write_bot_locations(f, bot_locs, city_field)
            self._write_hub_locations(f, hub_locs, city_field)
        
        logger.info(f"Report saved to: {report_file}")
        return report_file


def generate_report(df: pd.DataFrame, bot_locs: pd.DataFrame, hub_locs: pd.DataFrame, 
                   independent_user_locs: pd.DataFrame, other_locs: pd.DataFrame, 
                   stats: Dict[str, Any], output_dir: str, 
                   cluster_df: Optional[pd.DataFrame] = None, # New parameter
                   schema: Optional[LogSchema] = None,
                   available_features: Optional[List[str]] = None,
                   classification_method: str = 'rules') -> str:
    """
    Convenience function for generating reports (backward compatibility).
    
    This creates a ReportGenerator and calls generate().
    """
    generator = ReportGenerator(schema=schema)
    return generator.generate(df, bot_locs, hub_locs, independent_user_locs, other_locs, stats, output_dir, cluster_df, available_features, classification_method)
