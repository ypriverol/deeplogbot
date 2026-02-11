"""Feature usage validation for behavioral-first bot detection.

This module validates that behavioral features are driving predictions (target: 70%+)
rather than rule-based features. This ensures the model maintains its behavioral-first
approach and catches regressions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ...utils import logger


def validate_feature_usage(
    df: pd.DataFrame, 
    feature_columns: List[str],
    predictions: np.ndarray
) -> Dict:
    """
    Validate that behavioral features are driving predictions.
    
    Ensures that the model is actually using behavioral features (target: 70%+)
    rather than rule-based features.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        predictions: Prediction labels or scores (binary or continuous)
        
    Returns:
        Dictionary with validation metrics
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        logger.warning("  sklearn not available, skipping feature validation")
        return {'passes_threshold': False, 'error': 'sklearn not available'}
    
    logger.info("Validating feature usage...")
    
    # Categorize features
    behavioral_features = [
        'hourly_entropy', 'working_hours_ratio', 'request_velocity',
        'burst_pattern_score', 'circadian_rhythm_deviation',
        'user_coordination_score', 'regularity_score', 'file_diversity_ratio',
        'access_regularity', 'spike_intensity', 'user_peak_ratio',
        'night_ratio', 'work_ratio', 'hourly_cv', 'hourly_cv_burst',
        'night_ratio_advanced', 'work_ratio_advanced', 'evening_ratio', 'morning_ratio',
        'temporal_irregularity', 'weekend_weekday_imbalance', 'users_per_active_hour',
        'year_over_year_cv', 'years_span', 'years_before_latest', 'yearly_entropy',
        'peak_hour_concentration', 'peak_year_concentration', 'spike_ratio',
        'is_mechanical', 'is_weekday_biased', 'is_bursty', 'is_bursty_advanced',
        'is_nocturnal', 'is_coordinated'
    ]
    
    rule_based_features = [
        'unique_users', 'downloads_per_user', 'total_downloads', 
        'max_users_per_hour', 'downloads_per_year', 'projects_per_user'
    ]
    
    # Filter to features that actually exist in the dataframe
    available_features = [f for f in feature_columns if f in df.columns]
    behavioral_in_data = [f for f in available_features if f in behavioral_features]
    rule_based_in_data = [f for f in available_features if f in rule_based_features]
    
    if len(behavioral_in_data) == 0 and len(rule_based_in_data) == 0:
        logger.warning("  Cannot validate: no categorized features found")
        return {'passes_threshold': False, 'error': 'no categorized features'}
    
    logger.info(f"  Behavioral features found: {len(behavioral_in_data)}")
    logger.info(f"  Rule-based features found: {len(rule_based_in_data)}")
    
    # Prepare data for RandomForest
    X = df[available_features].fillna(0).values
    
    # Handle predictions - convert to binary integer array
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    
    # Convert predictions to binary integer array
    if predictions.dtype == object or predictions.dtype.name == 'object':
        # String/object type (e.g., 'bot', 'normal')
        unique_vals = np.unique(predictions)
        if 'bot' in unique_vals or True in unique_vals:
            y = np.array([1 if (val == 'bot' or val is True or val == 'True') else 0 for val in predictions])
        else:
            # Convert to binary: use first value as negative class
            y = (predictions != unique_vals[0]).astype(int)
    elif predictions.dtype == bool:
        # Boolean type
        y = predictions.astype(int)
    elif predictions.dtype == float:
        if (predictions > 1).any():
            # Likely continuous scores, convert to binary using threshold
            y = (predictions > np.median(predictions)).astype(int)
        else:
            # Likely probabilities, convert to binary
            y = (predictions > 0.5).astype(int)
    else:
        # Already integer or numeric
        try:
            y = predictions.astype(int)
        except (ValueError, TypeError):
            # Fallback: convert to binary
            unique_vals = np.unique(predictions)
            y = (predictions != unique_vals[0]).astype(int)
    
    if len(np.unique(y)) < 2:
        logger.warning("  Cannot validate: predictions have only one class")
        return {'passes_threshold': False, 'error': 'single class predictions'}
    
    # Train RandomForest to measure feature importance
    try:
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        rf.fit(X, y)
    except Exception as e:
        logger.warning(f"  RandomForest training failed: {e}")
        return {'passes_threshold': False, 'error': str(e)}
    
    # Calculate importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Categorize weights
    behavioral_weight = feature_importance_df[
        feature_importance_df['feature'].isin(behavioral_features)
    ]['importance'].sum()
    
    rule_based_weight = feature_importance_df[
        feature_importance_df['feature'].isin(rule_based_features)
    ]['importance'].sum()
    
    total_weight = behavioral_weight + rule_based_weight
    behavioral_pct = (behavioral_weight / total_weight * 100) if total_weight > 0 else 0
    
    # Validation results
    validation = {
        'behavioral_weight': float(behavioral_weight),
        'rule_based_weight': float(rule_based_weight),
        'behavioral_percentage': float(behavioral_pct),
        'passes_70_percent_threshold': behavioral_pct >= 70,
        'total_features_analyzed': len(available_features),
        'behavioral_features_count': len(behavioral_in_data),
        'rule_based_features_count': len(rule_based_in_data),
        'top_10_features': feature_importance_df.head(10).to_dict('records'),
        'top_5_behavioral': feature_importance_df[
            feature_importance_df['feature'].isin(behavioral_features)
        ].head(5).to_dict('records'),
        'top_5_rule_based': feature_importance_df[
            feature_importance_df['feature'].isin(rule_based_features)
        ].head(5).to_dict('records'),
    }
    
    # Log results
    logger.info(f"\n{'='*70}")
    logger.info("FEATURE USAGE VALIDATION")
    logger.info(f"{'='*70}")
    logger.info(f"Behavioral features: {behavioral_weight:.3f} ({behavioral_pct:.1f}%)")
    logger.info(f"Rule-based features: {rule_based_weight:.3f} ({100-behavioral_pct:.1f}%)")
    logger.info("Target: ≥70% behavioral")
    logger.info(f"Status: {'✓ PASS' if validation['passes_70_percent_threshold'] else '✗ FAIL'}")
    
    logger.info("\nTop 10 Features:")
    for i, feat in enumerate(validation['top_10_features'], 1):
        feat_type = "🎯 Behavioral" if feat['feature'] in behavioral_features else "📊 Rule-based"
        logger.info(f"  {i}. {feat['feature']:<30} {feat['importance']:.4f} {feat_type}")
    
    if validation['top_5_behavioral']:
        logger.info("\nTop 5 Behavioral Features:")
        for i, feat in enumerate(validation['top_5_behavioral'], 1):
            logger.info(f"  {i}. {feat['feature']:<30} {feat['importance']:.4f}")
    
    if validation['top_5_rule_based']:
        logger.info("\nTop 5 Rule-based Features:")
        for i, feat in enumerate(validation['top_5_rule_based'], 1):
            logger.info(f"  {i}. {feat['feature']:<30} {feat['importance']:.4f}")
    
    return validation
