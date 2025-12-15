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
    
    def _write_classification_rules(self, f):
        """Write classification rules to report (generic version)."""
        f.write("Classification rules:\n")
        f.write("  BOT (multiple patterns, aggressive thresholds):\n")
        f.write("    Detected based on anomaly scores and behavioral patterns:\n")
        f.write("    - Low downloads per user combined with high user counts\n")
        f.write("    - Suspicious temporal patterns (recent spikes, new locations)\n")
        f.write("    - Irregular activity patterns\n")
        f.write("  DOWNLOAD_HUB:\n")
        f.write("    Detected based on:\n")
        f.write("    - Very high downloads per user (mirrors/single-user hubs)\n")
        f.write("    - Large total downloads with moderate downloads per user\n")
        f.write("    - Regular working hours patterns (research institutions)\n\n")
    
    def _write_summary_stats(self, f, df: pd.DataFrame, bot_locs: pd.DataFrame, 
                            hub_locs: pd.DataFrame, stats: Dict[str, Any]):
        """Write summary statistics."""
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total locations analyzed: {len(df):,}\n")
        f.write(f"Anomalous locations: {df['is_anomaly'].sum():,}\n")
        f.write(f"Bot locations: {len(bot_locs):,}\n")
        f.write(f"Download hub locations: {len(hub_locs):,}\n\n")
        
        f.write(f"Total downloads: {format_number(stats['total'])}\n")
        f.write(f"Bot downloads: {format_number(stats['bots'])} ({stats['bots']/stats['total']*100:.2f}%)\n")
        f.write(f"Hub downloads: {format_number(stats['hubs'])} ({stats['hubs']/stats['total']*100:.2f}%)\n")
        f.write(f"Normal downloads: {format_number(stats['normal'])} ({stats['normal']/stats['total']*100:.2f}%)\n\n")
    
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
    
    def generate(self, df: pd.DataFrame, bot_locs: pd.DataFrame, hub_locs: pd.DataFrame, 
                stats: Dict[str, Any], output_dir: str, available_features: Optional[List[str]] = None) -> str:
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
        
        # Get city field name from schema or use default
        city_field = self.schema.city_field if self.schema and self.schema.city_field else 'city'
        
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
            self._write_classification_rules(f)
            
            self._write_summary_stats(f, df, bot_locs, hub_locs, stats)
            self._write_city_level_aggregation(f, df, city_field)
            self._write_bot_locations(f, bot_locs, city_field)
            self._write_hub_locations(f, hub_locs, city_field)
        
        logger.info(f"Report saved to: {report_file}")
        return report_file


def generate_report(df: pd.DataFrame, bot_locs: pd.DataFrame, hub_locs: pd.DataFrame, 
                   stats: Dict[str, Any], output_dir: str, 
                   schema: Optional[LogSchema] = None,
                   available_features: Optional[List[str]] = None) -> str:
    """
    Convenience function for generating reports (backward compatibility).
    
    This creates a ReportGenerator and calls generate().
    """
    generator = ReportGenerator(schema=schema)
    return generator.generate(df, bot_locs, hub_locs, stats, output_dir, available_features)
