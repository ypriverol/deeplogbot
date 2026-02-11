"""ML models for bot detection."""

import os
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from ...utils import logger


def train_isolation_forest(df, feature_columns, contamination=0.05):
    """
    Train Isolation Forest for anomaly detection.
    
    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are isolated quickly (short path length).
    """
    logger.info(f"Training Isolation Forest (contamination={contamination})...")
    
    X = df[feature_columns].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    predictions = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)
    
    return predictions, scores, model, scaler


def compute_feature_importances(analysis_df, feature_columns, labels, output_dir):
    """
    Compute feature importances using a surrogate RandomForestClassifier and permutation importance.
    This is optional and intended for interpretability/debugging.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Computing feature importances (surrogate RF + permutation)...")
    logger.info("=" * 70)
    
    X = analysis_df[feature_columns].fillna(0)
    y = labels.astype(int)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        max_depth=None
    )
    rf.fit(X, y)
    
    # Gini importances
    gini_importances = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=False)

    # Permutation importances (roc_auc scorer)
    # Try parallel execution first, fall back to sequential if it fails
    n_jobs_val = min(os.cpu_count() or 1, 4)  # Limit to 4 to avoid resource issues
    try:
        perm = permutation_importance(
            rf, X, y, n_repeats=5, random_state=42, n_jobs=n_jobs_val, scoring='roc_auc'
        )
    except Exception as e:
        logger.warning(f"Parallel permutation importance failed, trying sequential: {e}")
        perm = permutation_importance(
            rf, X, y, n_repeats=5, random_state=42, n_jobs=1, scoring='roc_auc'
        )
    perm_importances = pd.Series(perm.importances_mean, index=feature_columns).sort_values(ascending=False)
    
    os.makedirs(output_dir, exist_ok=True)
    gini_path = os.path.join(output_dir, 'feature_importances_gini.csv')
    perm_path = os.path.join(output_dir, 'feature_importances_permutation.csv')
    gini_importances.to_csv(gini_path, header=['importance'])
    perm_importances.to_csv(perm_path, header=['importance_mean'])
    
    logger.info("Top 10 (Gini):")
    for feat, val in gini_importances.head(10).items():
        logger.info(f"  {feat}: {val:.4f}")
    
    logger.info("Top 10 (Permutation, mean delta AUC):")
    for feat, val in perm_importances.head(10).items():
        logger.info(f"  {feat}: {val:.4f}")
    
    return {'gini': gini_path, 'permutation': perm_path}

