"""Classification logic for bot and download hub detection."""

import pandas as pd

from .utils import logger


def classify_locations(df):
    """
    Classify locations into categories:
    - bot: anomalous + low downloads/user + many users
    - download_hub: 
        (1) anomalous + high downloads/user (>500) - mirrors/single-user hubs
        (2) anomalous + high total downloads (>150K) + moderate downloads/user (>50) 
            + regular patterns (working hours > 0.25) - research institutions
    - normal: not anomalous
    """
    # Bot pattern: many users cycling with few downloads each
    # Much stricter thresholds + aggressive spike detection
    # Some extreme patterns bypass anomaly detection requirement
    # 
    # IMPORTANT: We classify anomalies FIRST, then add extreme patterns
    # This ensures we catch all anomalous bot-like behavior
    bot_mask_anomalous = (
        df['is_anomaly'] &
        (
            # Pattern 1: High user count with low downloads per user (very strict)
            (
                (df['downloads_per_user'] < 12) &  # Strict threshold
                (df['unique_users'] > 7000)
            ) |
            # Pattern 1b: Very high user count with moderate downloads per user (e.g., Singapore)
            (
                (df['unique_users'] > 25000) &  # Very high user count
                (df['downloads_per_user'] < 100) &  # Moderate DL/user (not too high)
                (df['downloads_per_user'] > 10)  # But not too low (to avoid duplicates)
            ) |
            # Pattern 1c: High user count (>15K) with moderate-low DL/user (<80)
            (
                (df['unique_users'] > 15000) &
                (df['downloads_per_user'] < 80) &
                (df['downloads_per_user'] > 8)
            ) |
            # Pattern 2: Sudden surge in latest year (more aggressive thresholds)
            (
                (df['fraction_latest_year'] > 0.4) &  # >40% of activity in latest year
                (df['downloads_per_user'] < 25) &  # Low downloads per user
                (df['unique_users'] > 2000) &  # Moderate user count
                (df['spike_ratio'] > 2)  # At least 2x spike
            ) |
            # Pattern 3: New locations with suspicious patterns (lower threshold)
            (
                (df['is_new_location'] == 1) &
                (df['downloads_per_user'] < 15) &
                (df['unique_users'] > 3000)  # Lower threshold: was 5000
            ) |
            # Pattern 4: Massive spike (5x+ increase - lower threshold)
            (
                (df['spike_ratio'] > 5) &  # Lower threshold: was 10
                (df['downloads_per_user'] < 15) &
                (df['unique_users'] > 5000) &
                (df['years_before_latest'] > 0)
            ) |
            # Pattern 5: Moderate spike with multiple suspicious signals
            (
                (df['spike_ratio'] > 3) &  # Lower threshold: was 5
                (df['fraction_latest_year'] > 0.5) &  # Lower threshold: was 0.6
                (df['downloads_per_user'] < 12) &
                (df['unique_users'] > 5000)  # Lower threshold: was 7000
            ) |
            # Pattern 6: Lower user threshold but high spike + latest year concentration
            (
                (df['spike_ratio'] > 1.5) &  # Even lower: 1.5x spike
                (df['fraction_latest_year'] > 0.5) &  # >50% in latest year
                (df['downloads_per_user'] < 20) &
                (df['unique_users'] > 2000) &
                (df['years_before_latest'] >= 1)  # Existing location with surge
            ) |
            # Pattern 7: Very high latest year concentration with suspicious user patterns
            (
                (df['fraction_latest_year'] > 0.7) &  # >70% in latest year
                (df['downloads_per_user'] < 30) &
                (df['unique_users'] > 1000)  # Lower threshold
            ) |
            # Pattern 8: High spike ratio with moderate concentration (catches more locations)
            (
                (df['spike_ratio'] > 3) &  # 3x+ spike (was 5x)
                (df['fraction_latest_year'] > 0.7) &  # >70% in latest year (was 0.8)
                (df['downloads_per_user'] < 35) &  # Low downloads per user
                (df['unique_users'] > 300)  # Even lower threshold - catch smaller bot farms
            ) |
            # Pattern 9: New locations with moderate activity (very aggressive)
            (
                (df['is_new_location'] == 1) &
                (df['downloads_per_user'] < 35) &  # Higher threshold
                (df['unique_users'] > 500) &  # Lower user threshold
                (df['total_downloads'] > 3000)  # Lower total threshold
            ) |
            # Pattern 10: Extreme latest year concentration (>85%)
            (
                (df['fraction_latest_year'] > 0.85) &  # >85% in latest year (was 0.9)
                (df['downloads_per_user'] < 35) &
                (df['unique_users'] > 300) &
                (df['spike_ratio'] > 1.5)  # Lower spike threshold
            ) |
            # Pattern 11: Any location with very high spike and moderate latest year concentration
            (
                (df['spike_ratio'] > 8) &  # 8x+ spike
                (df['fraction_latest_year'] > 0.6) &
                (df['downloads_per_user'] < 40) &
                (df['unique_users'] > 500)
            ) |
            # Pattern 12: Moderate latest year concentration but very low downloads per user
            (
                (df['fraction_latest_year'] > 0.5) &
                (df['downloads_per_user'] < 10) &  # Very low
                (df['unique_users'] > 1500) &
                (df['spike_ratio'] > 1.5)
            ) |
            # Pattern 13: High latest year concentration (>60%) with moderate spike and low DL/user
            (
                (df['fraction_latest_year'] > 0.6) &
                (df['spike_ratio'] > 1.2) &  # Even lower spike threshold
                (df['downloads_per_user'] < 30) &
                (df['unique_users'] > 1000)
            ) |
            # Pattern 14: Locations with >50% latest year and significant volume (>50K downloads)
            (
                (df['fraction_latest_year'] > 0.5) &
                (df['total_downloads'] > 50000) &
                (df['downloads_per_user'] < 35) &
                (df['unique_users'] > 500)
            ) |
            # Pattern 15: Moderate spike (>2x) with high latest year concentration (>55%)
            (
                (df['spike_ratio'] > 2.0) &
                (df['fraction_latest_year'] > 0.55) &
                (df['downloads_per_user'] < 40) &
                (df['unique_users'] > 800)
            ) |
            # Pattern 16: High absolute volume in latest year (>100K) with suspicious patterns
            (
                (df['fraction_latest_year'] > 0.55) &
                (df['total_downloads'] > 100000) &
                (df['downloads_per_user'] < 45) &
                (df['unique_users'] > 300)
            ) |
            # Pattern 17: Very high latest year absolute volume (>200K downloads) with low DL/user
            (
                (df['fraction_latest_year'] > 0.5) &
                (df['latest_year_downloads'] > 200000) &
                (df['downloads_per_user'] < 50) &
                (df['unique_users'] > 500)
            ) |
            # Pattern 18: Moderate latest year concentration (>45%) but huge absolute volume in latest year
            (
                (df['fraction_latest_year'] > 0.45) &
                (df['latest_year_downloads'] > 500000) &
                (df['downloads_per_user'] < 55) &
                (df['unique_users'] > 400)
            ) |
            # Pattern 19: High absolute latest year downloads (>100K) with moderate spike
            (
                (df['latest_year_downloads'] > 100000) &
                (df['spike_ratio'] > 1.3) &
                (df['fraction_latest_year'] > 0.4) &
                (df['downloads_per_user'] < 50) &
                (df['unique_users'] > 300)
            )
        )
    )
    
    # Determine hub patterns first (needed for extreme bot detection)
    hub_mask_mirrors = (
        df['is_anomaly'] &
        (df['downloads_per_user'] > 500)
    )
    
    # Download hub pattern 2: research institutions (high volume, moderate per-user, regular patterns)
    # This catches legitimate institutions like universities that do bulk reanalysis
    # Check if working_hours_ratio exists (it should after feature extraction)
    if 'working_hours_ratio' in df.columns:
        working_hours_check = df['working_hours_ratio'] > 0.25
    else:
        working_hours_check = pd.Series([True] * len(df), index=df.index)
    
    hub_mask_research = (
        df['is_anomaly'] &
        (~bot_mask_anomalous) &  # Not already classified as bot
        (df['total_downloads'] > 150000) &  # High total volume
        (df['downloads_per_user'] > 50) &  # Moderate per-user ratio (not too low, not mirror-level)
        (df['downloads_per_user'] < 500) &  # Below mirror threshold
        (df['unique_users'] > 1000) &  # Substantial user base
        working_hours_check  # Regular working-hours pattern
    )
    
    # Combine both hub patterns
    hub_mask = hub_mask_mirrors | hub_mask_research
    
    # Extreme patterns that bypass anomaly requirement (very aggressive)
    # These are clearly bots even if not flagged as anomalous by ML
    bot_mask_extreme = (
        ~hub_mask &  # Not a hub (using hub_mask instead of df['is_download_hub'])
        (
            # Extreme pattern 1: >90% latest year with moderate spike and low DL/user
            (
                (df['fraction_latest_year'] > 0.9) &  # >90% (was 0.95)
                (df['spike_ratio'] > 5) &  # 5x+ spike (was 10x)
                (df['downloads_per_user'] < 35) &
                (df['unique_users'] > 300)  # Lower threshold
            ) |
            # Extreme pattern 2: New location with high activity (lower thresholds)
            (
                (df['is_new_location'] == 1) &
                (df['downloads_per_user'] < 30) &
                (df['unique_users'] > 1000) &  # Lower threshold
                (df['total_downloads'] > 5000)  # Lower threshold
            ) |
            # Extreme pattern 3: Very high spike (>15x) with latest year concentration
            (
                (df['spike_ratio'] > 15) &  # 15x+ spike (was 20x)
                (df['fraction_latest_year'] > 0.8) &  # >80% (was 0.85)
                (df['downloads_per_user'] < 35) &
                (df['unique_users'] > 500)  # Lower threshold
            ) |
            # Extreme pattern 4: Moderate spike but very high latest year concentration
            (
                (df['fraction_latest_year'] > 0.85) &
                (df['spike_ratio'] > 3) &
                (df['downloads_per_user'] < 40) &
                (df['unique_users'] > 500)
            ) |
            # Extreme pattern 5: New location with substantial volume
            (
                (df['is_new_location'] == 1) &
                (df['unique_users'] > 2000) &
                (df['total_downloads'] > 20000) &
                (df['downloads_per_user'] < 40)
            ) |
            # Extreme pattern 6: High latest year concentration (>65%) with moderate volume
            (
                (df['fraction_latest_year'] > 0.65) &
                (df['total_downloads'] > 30000) &
                (df['downloads_per_user'] < 45) &
                (df['unique_users'] > 300)
            ) |
            # Extreme pattern 7: Moderate spike (>1.5x) but high absolute latest year downloads
            (
                (df['spike_ratio'] > 1.5) &
                (df['fraction_latest_year'] > 0.6) &
                (df['total_downloads'] > 80000) &
                (df['downloads_per_user'] < 50) &
                (df['unique_users'] > 400)
            ) |
            # Extreme pattern 8: >55% latest year with significant spike (>2x)
            (
                (df['fraction_latest_year'] > 0.55) &
                (df['spike_ratio'] > 2.0) &
                (df['downloads_per_user'] < 45) &
                (df['unique_users'] > 600)
            ) |
            # Extreme pattern 9: Very high absolute latest year volume (>300K) with moderate concentration
            (
                (df['fraction_latest_year'] > 0.4) &
                (df['latest_year_downloads'] > 300000) &
                (df['downloads_per_user'] < 60) &
                (df['unique_users'] > 300)
            ) |
            # Extreme pattern 10: Massive absolute latest year volume (>1M) with suspicious patterns
            (
                (df['latest_year_downloads'] > 1000000) &
                (df['fraction_latest_year'] > 0.35) &
                (df['downloads_per_user'] < 70) &
                (df['unique_users'] > 500)
            ) |
            # Extreme pattern 11: High latest year absolute downloads (>150K) with spike
            (
                (df['latest_year_downloads'] > 150000) &
                (df['spike_ratio'] > 1.5) &
                (df['fraction_latest_year'] > 0.4) &
                (df['downloads_per_user'] < 55) &
                (df['unique_users'] > 400)
            ) |
            # Extreme pattern 12: Country with massive coordinated bot network
            (
                (df['country_suspicious_location_ratio'] > 0.4) &  # >40% suspicious locations
                (df['country_latest_year_dl'] > 2000000) &  # >2M latest year downloads
                (df['country_fraction_latest'] > 0.6) &  # >60% latest year activity
                (df['fraction_latest_year'] > 0.35) &
                (df['downloads_per_user'] < 60) &
                (df['unique_users'] > 200)
            ) |
            # Extreme pattern 13: Geographic bot farm - country with many new bot-like locations
            (
                (df['country_new_locations'] > 10) &  # >10 new locations in country
                (df['country_new_location_ratio'] > 0.25) &  # >25% are new
                (df['country_fraction_latest'] > 0.65) &  # >65% latest year
                (df['is_new_location'] == 1) &
                (df['downloads_per_user'] < 55) &
                (df['total_downloads'] > 10000)
            ) |
            # Extreme pattern 14: Country-level rate anomaly combined with location patterns
            (
                (df['country_low_dl_user_locations'] > 8) &  # >8 locations with low DL/user
                (df['country_fraction_latest'] > 0.55) &
                (df['downloads_per_user'] < 40) &  # This location also low DL/user
                (df['fraction_latest_year'] > 0.4) &
                (df['unique_users'] > 250)
            ) |
            # Extreme pattern 15: Multi-factor country + location pattern
            (
                (df['country_latest_year_dl'] > 1500000) &
                (df['country_high_spike_locations'] > 8) &
                (df['country_fraction_latest'] > 0.55) &
                (df['spike_ratio'] > 1.3) &
                (df['fraction_latest_year'] > 0.4) &
                (df['downloads_per_user'] < 50) &
                (df['unique_users'] > 300)
            ) |
            # Extreme pattern 16: Very high user count with moderate DL/user (bypasses anomaly check)
            (
                (df['unique_users'] > 30000) &  # Very high user count (like Singapore)
                (df['downloads_per_user'] < 100) &  # Moderate DL/user
                (df['downloads_per_user'] > 20)  # Not extremely low
            ) |
            # Extreme pattern 17: High user volume with suspicious country patterns
            (
                (df['unique_users'] > 25000) &
                (df['downloads_per_user'] < 120) &
                (df['country_fraction_latest'] > 0.4) &  # Country has latest year activity
                (df['total_downloads'] > 1000000)  # Significant total volume
            ) |
            # Extreme pattern 18: High latest year concentration with moderate-high user count
            (
                (df['fraction_latest_year'] > 0.85) &
                (df['unique_users'] > 5000) &
                (df['downloads_per_user'] < 100) &
                (df['downloads_per_user'] > 20)
            ) |
            # Extreme pattern 19: Very high spike with moderate user count
            (
                (df['spike_ratio'] > 20) &
                (df['unique_users'] > 3000) &
                (df['fraction_latest_year'] > 0.4) &
                (df['downloads_per_user'] < 100)
            ) |
            # Extreme pattern 20: Catch-all for anomalies that look bot-like but don't match other patterns
            # This catches anomalies with high user count OR high latest year concentration
            (
                (
                    (df['unique_users'] > 8000) & (df['downloads_per_user'] < 100) |
                    (df['fraction_latest_year'] > 0.8) & (df['unique_users'] > 2000) & (df['downloads_per_user'] < 80)
                ) &
                ~hub_mask  # Not already a hub
            )
        )
    )
    
    # CRITICAL FIX: Catch anomalies that weren't classified by patterns
    # Many anomalies are bot-like but don't match specific pattern thresholds
    # This catch-all ensures we don't lose bot-like anomalies
    anomalies_not_classified = df['is_anomaly'] & ~bot_mask_anomalous & ~hub_mask
    
    # Classify unclassified anomalies as bots if they have bot-like characteristics
    # More aggressive catch-all to catch missed bot patterns
    catch_all_bot_mask = anomalies_not_classified & (
        # High user count with moderate DL/user (catches cases like Los Angeles: 11K users, 65 DL/user)
        ((df['unique_users'] > 5000) & (df['downloads_per_user'] < 100) & (df['downloads_per_user'] > 15)) |
        # High latest year concentration with moderate user count (lowered threshold)
        ((df['fraction_latest_year'] > 0.7) & (df['unique_users'] > 1500) & (df['downloads_per_user'] < 120)) |
        # Very high spike ratio (catches sudden bot surges) - lowered threshold
        ((df['spike_ratio'] > 3) & (df['unique_users'] > 800) & (df['downloads_per_user'] < 100)) |
        # Moderate user count with suspicious patterns (lowered thresholds)
        ((df['unique_users'] > 2000) & (df['downloads_per_user'] < 60) & (df['fraction_latest_year'] > 0.3)) |
        # High spike with moderate user count
        ((df['spike_ratio'] > 5) & (df['unique_users'] > 1000) & (df['downloads_per_user'] < 100)) |
        # Moderate-high user count (2-5K) with low-moderate DL/user
        ((df['unique_users'] > 2000) & (df['unique_users'] < 6000) & (df['downloads_per_user'] < 50) & (df['downloads_per_user'] > 10)) |
        # High latest year (>60%) with any significant user base
        ((df['fraction_latest_year'] > 0.6) & (df['unique_users'] > 1000) & (df['downloads_per_user'] < 100))
    )
    
    # Also check if some unclassified anomalies should be hubs (high DL/user but not caught)
    catch_all_hub_mask = anomalies_not_classified & (
        (df['downloads_per_user'] > 150) &  # High DL/user
        (df['total_downloads'] > 50000)  # Significant volume
    )
    
    # Update hub_mask to include catch-all hubs
    hub_mask = hub_mask | catch_all_hub_mask
    
    # Combine all bot masks (anomalous patterns + extreme patterns + catch-all for missed anomalies)
    bot_mask = bot_mask_anomalous | bot_mask_extreme | catch_all_bot_mask
    
    df['is_bot'] = bot_mask
    df['is_download_hub'] = hub_mask
    
    return df

