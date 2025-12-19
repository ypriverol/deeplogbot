"""Supervised ML-based classification for bot and download hub detection."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ...utils import logger
from .rules import classify_locations


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

