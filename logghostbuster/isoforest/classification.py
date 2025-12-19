"""Classification logic for bot and download hub detection."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ..utils import logger


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


def classify_locations_ml(df, feature_columns, use_rule_based_labels=True, test_size=0.2, random_state=42):
    """
    Classify locations using ML-based approach (RandomForestClassifier).
    
    This function trains a multi-class classifier to distinguish between:
    - bot (class 0)
    - download_hub (class 1) 
    - normal (class 2)
    
    Args:
        df: DataFrame with features and anomaly scores
        feature_columns: List of feature column names to use for classification
        use_rule_based_labels: If True, uses rule-based classification to generate training labels.
                               If False, expects 'is_bot' and 'is_download_hub' columns already present.
        test_size: Fraction of data to use for testing (for evaluation metrics)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with 'is_bot' and 'is_download_hub' columns added
    """
    logger.info("Training ML-based classifier...")
    
    # Prepare features
    X = df[feature_columns].fillna(0).values
    
    # Generate labels using rule-based classification if needed
    if use_rule_based_labels:
        logger.info("  Generating training labels using rule-based classification...")
        df_labeled = classify_locations(df.copy())
        # Create multi-class labels: 0=bot, 1=hub, 2=normal
        y = np.where(df_labeled['is_bot'], 0,
                    np.where(df_labeled['is_download_hub'], 1, 2))
    else:
        # Use existing labels
        if 'is_bot' not in df.columns or 'is_download_hub' not in df.columns:
            raise ValueError("If use_rule_based_labels=False, 'is_bot' and 'is_download_hub' columns must exist")
        y = np.where(df['is_bot'], 0,
                    np.where(df['is_download_hub'], 1, 2))
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['bot', 'hub', 'normal']
    logger.info("  Class distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"    {class_names[cls]}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for evaluation (optional, but useful for metrics)
    # Check if stratified split is possible (each class needs at least 2 members)
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = all(count >= 2 for count in counts)
    min_samples_per_class = min(counts)
    
    # Skip test split if dataset is too small or stratification not possible
    if test_size > 0 and can_stratify and len(y) > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        if test_size > 0:
            if not can_stratify:
                logger.info(f"  Skipping test split: some classes have <2 members (min: {min_samples_per_class})")
            else:
                logger.info(f"  Skipping test split: dataset too small ({len(y)} samples)")
        X_train, y_train = X_scaled, y
        X_test, y_test = None, None
    
    # Train RandomForestClassifier
    logger.info("  Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on test set if available
    if X_test is not None:
        y_pred_test = clf.predict(X_test)
        logger.info("\n  Test Set Performance:")
        logger.info(f"  {classification_report(y_test, y_pred_test, target_names=class_names)}")
        
        logger.info("  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_test)
        logger.info(f"  {cm}")
    
    # Predict on all data
    logger.info("  Predicting on all locations...")
    y_pred = clf.predict(X_scaled)
    
    # Log feature importances
    feature_importances = pd.Series(clf.feature_importances_, index=feature_columns).sort_values(ascending=False)
    logger.info("\n  Top 10 Most Important Features:")
    for feat, importance in feature_importances.head(10).items():
        logger.info(f"    {feat}: {importance:.4f}")
    
    # Add predictions to dataframe
    df['is_bot'] = (y_pred == 0)
    df['is_download_hub'] = (y_pred == 1)
    
    # Log final distribution
    n_bots = df['is_bot'].sum()
    n_hubs = df['is_download_hub'].sum()
    n_normal = len(df) - n_bots - n_hubs
    logger.info(f"\n  Final Classification:")
    logger.info(f"    Bot locations: {n_bots:,} ({n_bots/len(df)*100:.1f}%)")
    logger.info(f"    Hub locations: {n_hubs:,} ({n_hubs/len(df)*100:.1f}%)")
    logger.info(f"    Normal locations: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    
    return df


def classify_locations_ml_unsupervised(df, feature_columns, n_clusters=5, random_state=42):
    """
    Classify locations using unsupervised ML approach (KMeans clustering).
    
    This function uses KMeans clustering with improved feature selection and mapping logic.
    Creates multiple categories: bots, hubs, independent users, normal users, and other patterns.
    
    Categories:
    - Bot: Coordinated bot activity (many users, low DL/user, high anomaly)
    - Download Hub: High downloads per user (mirrors, institutional)
    - Independent User: Single or few users with low DL/user (legitimate individual users)
    - Normal: Regular user patterns
    - Other: Unclassified patterns that don't fit clearly
    
    Args:
        df: DataFrame with features and anomaly scores
        feature_columns: List of feature column names to use for classification
        n_clusters: Number of clusters (default: 5 for better categorization)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with 'is_bot', 'is_download_hub', 'is_independent_user', 'user_category' columns added
    """
    logger.info("Training unsupervised ML-based classifier (KMeans with improved mapping)...")
    
    # Select most discriminative features for clustering
    # Focus on features that best distinguish bots vs hubs vs normal
    key_features = [
        'unique_users',
        'downloads_per_user', 
        'anomaly_score',  # Use anomaly score as a feature
        'fraction_latest_year',
        'spike_ratio',
        'hourly_entropy',
        'working_hours_ratio',
        'users_per_active_hour'
    ]
    
    # Use only features that exist in the dataframe
    available_key_features = [f for f in key_features if f in df.columns]
    if 'anomaly_score' not in df.columns:
        logger.warning("  anomaly_score not found, using 0 as default")
        df['anomaly_score'] = 0
    
    logger.info(f"  Using {len(available_key_features)} key features for clustering: {available_key_features}")
    
    # Prepare features
    X = df[available_key_features].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    logger.info(f"  Clustering into {n_clusters} groups using KMeans...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,  # More initializations for better results
        max_iter=500
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Analyze cluster characteristics to map to categories
    logger.info("  Analyzing cluster characteristics...")
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Calculate cluster centroids/means for key features
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'avg_unique_users': cluster_data['unique_users'].mean(),
            'avg_downloads_per_user': cluster_data['downloads_per_user'].mean(),
            'avg_total_downloads': cluster_data['total_downloads'].mean(),
            'avg_anomaly_score': cluster_data['anomaly_score'].mean(),
            'avg_fraction_latest_year': cluster_data.get('fraction_latest_year', pd.Series([0] * len(cluster_data))).mean(),
            'avg_spike_ratio': cluster_data.get('spike_ratio', pd.Series([0] * len(cluster_data))).mean(),
            'median_downloads_per_user': cluster_data['downloads_per_user'].median(),
            'median_unique_users': cluster_data['unique_users'].median(),
            'p75_downloads_per_user': cluster_data['downloads_per_user'].quantile(0.75),
            'p25_downloads_per_user': cluster_data['downloads_per_user'].quantile(0.25),
        }
        cluster_stats.append(stats)
        logger.info(f"    Cluster {cluster_id}: {stats['size']} locations")
        logger.info(f"      Avg users: {stats['avg_unique_users']:.0f}, "
                   f"Median users: {stats['median_unique_users']:.0f}, "
                   f"Avg DL/user: {stats['avg_downloads_per_user']:.1f}, "
                   f"Median DL/user: {stats['median_downloads_per_user']:.1f}, "
                   f"Avg anomaly: {stats['avg_anomaly_score']:.3f}")
    
    cluster_df = pd.DataFrame(cluster_stats)
    
    # Multi-category mapping strategy:
    # 1. Hub: Very high downloads_per_user (median > 100 or avg > 500)
    # 2. Bot: High users + low DL/user + high anomaly + high spike (coordinated activity)
    # 3. Independent User: Low users (1-5) + low DL/user (1-3) + low anomaly (legitimate individual)
    # 4. Normal: Moderate users, moderate DL/user, low anomaly (regular users)
    # 5. Other: Everything else that doesn't fit clearly
    
    # Initialize category columns
    df['user_category'] = 'other'
    df['is_bot'] = False
    df['is_download_hub'] = False
    df['is_independent_user'] = False
    
    # 1. Identify Hub cluster (highest downloads_per_user)
    cluster_df['hub_score'] = (
        cluster_df['median_downloads_per_user'] / (cluster_df['median_downloads_per_user'].max() + 1e-10) * 0.6 +
        cluster_df['avg_downloads_per_user'] / (cluster_df['avg_downloads_per_user'].max() + 1e-10) * 0.4
    )
    hub_cluster = cluster_df.loc[cluster_df['hub_score'].idxmax(), 'cluster']
    hub_median_dl = cluster_df.loc[cluster_df['hub_score'].idxmax(), 'median_downloads_per_user']
    hub_avg_dl = cluster_df.loc[cluster_df['hub_score'].idxmax(), 'avg_downloads_per_user']
    logger.info(f"  Identified Cluster {hub_cluster} as DOWNLOAD_HUB (median DL/user: {hub_median_dl:.1f}, avg DL/user: {hub_avg_dl:.1f})")
    
    # 2. Identify Independent User clusters (low users, low DL/user, low anomaly)
    remaining_clusters = cluster_df[cluster_df['cluster'] != hub_cluster].copy()
    if len(remaining_clusters) > 0:
        # Independent user score: low users + low DL/user + low anomaly
        remaining_clusters['independent_score'] = (
            (1 - remaining_clusters['median_unique_users'] / (remaining_clusters['median_unique_users'].max() + 1e-10)) * 0.4 +
            (1 - remaining_clusters['median_downloads_per_user'] / (remaining_clusters['median_downloads_per_user'].max() + 1e-10)) * 0.4 +
            (1 - (remaining_clusters['avg_anomaly_score'] - remaining_clusters['avg_anomaly_score'].min()) / 
             (remaining_clusters['avg_anomaly_score'].max() - remaining_clusters['avg_anomaly_score'].min() + 1e-10)) * 0.2
        )
        
        # Independent users: median users <= 5, median DL/user <= 3, low anomaly
        independent_candidates = remaining_clusters[
            (remaining_clusters['median_unique_users'] <= 5) &
            (remaining_clusters['median_downloads_per_user'] <= 3) &
            (remaining_clusters['avg_anomaly_score'] < 0.1)
        ]
        
        if len(independent_candidates) > 0:
            independent_cluster = independent_candidates.loc[independent_candidates['independent_score'].idxmax(), 'cluster']
            independent_stats = independent_candidates.loc[independent_candidates['independent_score'].idxmax()]
            logger.info(f"  Identified Cluster {independent_cluster} as INDEPENDENT_USER")
            logger.info(f"    Median users: {independent_stats['median_unique_users']:.0f}, "
                       f"Median DL/user: {independent_stats['median_downloads_per_user']:.1f}, "
                       f"Avg anomaly: {independent_stats['avg_anomaly_score']:.3f}")
        else:
            independent_cluster = None
            logger.info("  No cluster identified as INDEPENDENT_USER")
        
        # 3. Identify Bot cluster from remaining (high users, low DL/user, high anomaly, high spike)
        bot_candidates = remaining_clusters[remaining_clusters['cluster'] != independent_cluster] if independent_cluster is not None else remaining_clusters
        
        if len(bot_candidates) > 0:
            max_users = bot_candidates['avg_unique_users'].max()
            max_dl_user = bot_candidates['avg_downloads_per_user'].max()
            max_anomaly = bot_candidates['avg_anomaly_score'].max() + 1e-10
            max_spike = bot_candidates['avg_spike_ratio'].max() + 1e-10
            
            # Bot score: high users + low DL/user + high anomaly + high spike
            bot_candidates['bot_score'] = (
                (bot_candidates['avg_unique_users'] / max_users) * 0.3 +
                (1 - bot_candidates['avg_downloads_per_user'] / max_dl_user) * 0.3 +
                (bot_candidates['avg_anomaly_score'] / max_anomaly) * 0.25 +
                (bot_candidates['avg_spike_ratio'] / max_spike) * 0.15
            )
            
            bot_cluster = bot_candidates.loc[bot_candidates['bot_score'].idxmax(), 'cluster']
            bot_stats = bot_candidates.loc[bot_candidates['bot_score'].idxmax()]
            logger.info(f"  Identified Cluster {bot_cluster} as BOT")
            logger.info(f"    Avg users: {bot_stats['avg_unique_users']:.0f}, "
                       f"Avg DL/user: {bot_stats['avg_downloads_per_user']:.1f}, "
                       f"Avg anomaly: {bot_stats['avg_anomaly_score']:.3f}, "
                       f"Avg spike: {bot_stats['avg_spike_ratio']:.2f}")
        else:
            bot_cluster = None
            logger.info("  No cluster identified as BOT")
    else:
        bot_cluster = None
        independent_cluster = None
        logger.info("  Only one cluster found, cannot identify bot or independent user clusters")
    
    # Assign category labels (using cluster_labels directly)
    df.loc[cluster_labels == hub_cluster, 'user_category'] = 'hub'
    df.loc[cluster_labels == hub_cluster, 'is_download_hub'] = True
    
    if bot_cluster is not None:
        df.loc[cluster_labels == bot_cluster, 'user_category'] = 'bot'
        df.loc[cluster_labels == bot_cluster, 'is_bot'] = True
    
    if independent_cluster is not None:
        df.loc[cluster_labels == independent_cluster, 'user_category'] = 'independent_user'
        df.loc[cluster_labels == independent_cluster, 'is_independent_user'] = True
    
    # Mark remaining clusters as 'normal' or 'other' based on characteristics
    unclassified_mask = df['user_category'] == 'other'
    if unclassified_mask.sum() > 0:
        # If unclassified clusters have moderate characteristics, mark as 'normal'
        for cluster_id in df_clustered[unclassified_mask]['cluster'].unique():
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            if (cluster_data['unique_users'].median() <= 50 and 
                cluster_data['downloads_per_user'].median() <= 20 and
                cluster_data['anomaly_score'].mean() < 0.05):
                df.loc[cluster_labels == cluster_id, 'user_category'] = 'normal'
            else:
                df.loc[cluster_labels == cluster_id, 'user_category'] = 'other'
    
    # Log final distribution
    n_bots = df['is_bot'].sum()
    n_hubs = df['is_download_hub'].sum()
    n_independent = df['is_independent_user'].sum()
    n_normal = (df['user_category'] == 'normal').sum()
    n_other = (df['user_category'] == 'other').sum()
    
    logger.info(f"\n  Final Classification:")
    logger.info(f"    Bot locations: {n_bots:,} ({n_bots/len(df)*100:.1f}%)")
    logger.info(f"    Hub locations: {n_hubs:,} ({n_hubs/len(df)*100:.1f}%)")
    logger.info(f"    Independent User locations: {n_independent:,} ({n_independent/len(df)*100:.1f}%)")
    logger.info(f"    Normal locations: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    logger.info(f"    Other/Unclassified locations: {n_other:,} ({n_other/len(df)*100:.1f}%)")
    
    return df

