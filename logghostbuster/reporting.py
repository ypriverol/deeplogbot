"""Report generation for bot detection results."""

import os
import pandas as pd
from datetime import datetime

from .utils import logger, format_number
from .geography import group_nearby_locations_with_llm


def generate_report(df, bot_locs, hub_locs, stats, output_dir):
    """Generate comprehensive report."""
    
    report_file = os.path.join(output_dir, 'bot_detection_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BOT AND DOWNLOAD HUB DETECTION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 60 + "\n")
        f.write("Algorithm: Isolation Forest (unsupervised anomaly detection)\n\n")
        f.write("Features used:\n")
        f.write("  - unique_users: Number of distinct user IDs\n")
        f.write("  - downloads_per_user: Total downloads / unique users\n")
        f.write("  - avg_users_per_hour: Average user density per hour\n")
        f.write("  - max_users_per_hour: Peak user density\n")
        f.write("  - user_cv: Coefficient of variation (pattern regularity)\n")
        f.write("  - users_per_active_hour: User concentration\n")
        f.write("  - projects_per_user: Download diversity\n")
        f.write("  - hourly_download_std: Standard deviation of downloads across hours\n")
        f.write("  - peak_hour_concentration: Fraction of downloads in busiest hour\n")
        f.write("  - working_hours_ratio: Fraction of downloads during 9 AM - 5 PM (UTC)\n")
        f.write("  - hourly_entropy: Entropy of hourly distribution (uniformity measure)\n")
        f.write("  - night_activity_ratio: Fraction of downloads during 10 PM - 6 AM (UTC)\n")
        f.write("  - yearly_entropy: Entropy of yearly distribution (sustained vs bursty)\n")
        f.write("  - peak_year_concentration: Fraction of downloads in busiest year\n")
        f.write("  - years_span: Number of years with activity\n")
        f.write("  - downloads_per_year: Average downloads per year\n")
        f.write("  - year_over_year_cv: Coefficient of variation across years (consistency)\n")
        f.write("  - fraction_latest_year: Fraction of downloads in latest year (suspicious if high)\n")
        f.write("  - is_new_location: Binary flag if location first appeared in latest year\n")
        f.write("  - spike_ratio: Latest year downloads vs average of previous years\n")
        f.write("  - years_before_latest: Number of years with activity before latest year\n\n")
        
        f.write("Classification rules:\n")
        f.write("  BOT (multiple patterns, aggressive thresholds):\n")
        f.write("    (1) anomalous + downloads/user < 12 + users > 7,000 (strict baseline)\n")
        f.write("    (2) anomalous + fraction_latest_year > 0.5 + spike_ratio > 3x + downloads/user < 20 + users > 3,000\n")
        f.write("    (3) anomalous + is_new_location = 1 + downloads/user < 15 + users > 3,000\n")
        f.write("    (4) anomalous + spike_ratio > 5x + downloads/user < 15 + users > 5,000 + years_before > 0\n")
        f.write("    (5) anomalous + spike_ratio > 3x + fraction_latest_year > 0.5 + downloads/user < 12 + users > 5,000\n")
        f.write("    (6) anomalous + spike_ratio > 1.5x + fraction_latest_year > 0.5 + downloads/user < 20 + users > 2,000\n")
        f.write("    (7) anomalous + fraction_latest_year > 0.7 + downloads/user < 30 + users > 1,000\n")
        f.write("    (8) anomalous + spike_ratio > 3x + fraction_latest_year > 0.7 + downloads/user < 35 + users > 300\n")
        f.write("    (9) anomalous + is_new_location = 1 + downloads/user < 35 + users > 500 + total > 3K\n")
        f.write("    (10) anomalous + fraction_latest_year > 0.85 + downloads/user < 35 + users > 300 + spike_ratio > 1.5x\n")
        f.write("    (11) anomalous + spike_ratio > 8x + fraction_latest_year > 0.6 + downloads/user < 40 + users > 500\n")
        f.write("    (12) anomalous + fraction_latest_year > 0.5 + downloads/user < 10 + users > 1,500 + spike_ratio > 1.5x\n")
        f.write("  EXTREME PATTERNS (bypass anomaly requirement):\n")
        f.write("    (13) fraction_latest_year > 0.9 + spike_ratio > 5x + downloads/user < 35 + users > 300\n")
        f.write("    (14) is_new_location = 1 + downloads/user < 30 + users > 1,000 + total > 5K\n")
        f.write("    (15) spike_ratio > 15x + fraction_latest_year > 0.8 + downloads/user < 35 + users > 500\n")
        f.write("    (16) fraction_latest_year > 0.85 + spike_ratio > 3x + downloads/user < 40 + users > 500\n")
        f.write("    (17) is_new_location = 1 + users > 2,000 + total > 20K + downloads/user < 40\n")
        f.write("  DOWNLOAD_HUB:\n")
        f.write("    (1) anomalous + downloads/user > 500 (mirrors/single-user hubs)\n")
        f.write("    (2) anomalous + total downloads > 150K + downloads/user 50-500\n")
        f.write("        + users > 1,000 + working_hours_ratio > 0.25 (research institutions)\n\n")
        
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
        
        # City-level aggregation for better hub detection
        f.write("=" * 80 + "\n")
        f.write("CITY-LEVEL AGGREGATION (Research Hubs)\n")
        f.write("=" * 80 + "\n")
        f.write("Note: Same city may have multiple geo_locations due to GPS precision.\n")
        f.write("This view aggregates all locations within a city.\n\n")
        
        city_agg = df.groupby(['country', 'city']).agg({
            'unique_users': 'sum',
            'total_downloads': 'sum',
            'geo_location': 'count'
        }).reset_index()
        city_agg.columns = ['country', 'city', 'total_users', 'total_downloads', 'num_locations']
        city_agg['downloads_per_user'] = city_agg['total_downloads'] / city_agg['total_users']
        
        # Filter out invalid entries
        city_agg = city_agg[
            (city_agg['total_downloads'] > 100000) &
            (city_agg['country'].notna()) &
            (~city_agg['country'].astype(str).str.contains('%{', na=False)) &  # Remove template strings
            (city_agg['country'].astype(str) != 'N/A') &
            (city_agg['country'].astype(str) != 'Unknown')
        ]
        city_agg = city_agg.sort_values('downloads_per_user', ascending=False)
        
        f.write(f"{'Country':<18} {'City':<20} {'Locs':>5} {'Users':>10} {'Downloads':>12} {'DL/User':>10}\n")
        f.write("-" * 80 + "\n")
        for _, row in city_agg.head(50).iterrows():
            city = str(row['city'])[:18] if pd.notna(row['city']) else 'N/A'
            f.write(f"{row['country']:<18} {city:<20} {int(row['num_locations']):>5} "
                   f"{int(row['total_users']):>10,} {int(row['total_downloads']):>12,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
        
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
            city = str(row['city'])[:20] if pd.notna(row['city']) and str(row['city']) != 'N/A' else ''
            country = str(row['country'])[:16]
            f.write(f"{country:<18} {city:<22} {int(row['unique_users']):>10,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
        
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
        consolidated = hub_locs_grouped.groupby(['country', 'group_id']).agg({
            'geo_location': 'count',
            'unique_users': 'sum',
            'total_downloads': 'sum',
            'city': lambda x: ', '.join([str(c) for c in x.dropna().unique()[:3]])  # Show up to 3 city names
        }).reset_index()
        consolidated.columns = ['country', 'group_id', 'num_locations', 'total_users', 'total_downloads', 'cities']
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
                        'city': canonical_loc.iloc[0].get('city', ''),
                        'geo_location': group_id
                    }
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"DOWNLOAD HUB LOCATIONS ({len(hub_locs)} individual, {len(consolidated)} consolidated)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Country':<18} {'Location(s)':<30} {'Locs':>5} {'Users':>10} {'DL/User':>10}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in consolidated.sort_values('downloads_per_user', ascending=False).iterrows():
            canonical = group_to_canonical.get(row['group_id'], {})
            display_city = canonical.get('city', '')
            if pd.isna(display_city) or display_city == '' or str(display_city) == 'N/A':
                # Try to get from cities list
                if row['cities'] and row['cities'] != 'N/A':
                    display_city = row['cities'].split(',')[0].strip()
                else:
                    display_city = ''
            
            # Show grouped cities if multiple
            if row['num_locations'] > 1 and row['cities'] and row['cities'] != 'N/A':
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
            city = str(row['city'])[:20] if pd.notna(row['city']) and str(row['city']) != 'N/A' else ''
            country = str(row['country'])[:16]
            f.write(f"{country:<18} {city:<22} {int(row['unique_users']):>10,} "
                   f"{row['downloads_per_user']:>10.1f}\n")
    
    logger.info(f"Report saved to: {report_file}")
    return report_file

