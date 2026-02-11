"""Fusion meta-learner for combining anomaly signals.

Replaces hand-coded override/reconciliation logic with a learned
combination of anomaly scores from multiple sources (VAE, IF, temporal,
behavioral features). Uses gradient boosting with confidence-weighted
training and Platt-calibrated probabilities.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Labels for 3-class classification
LABEL_ORGANIC = 0
LABEL_BOT = 1
LABEL_HUB = 2
LABEL_NAMES = {0: 'organic', 1: 'bot', 2: 'hub'}


def prepare_fusion_features(df: pd.DataFrame,
                            vae_scores: np.ndarray = None,
                            vae_latent: np.ndarray = None,
                            dif_scores: np.ndarray = None,
                            temporal_scores: np.ndarray = None,
                            behavioral_cols: list = None) -> np.ndarray:
    """Assemble feature matrix for the meta-learner.

    Args:
        df: DataFrame with location features
        vae_scores: VAE reconstruction error (n_samples,)
        vae_latent: VAE latent features (n_samples, latent_dim)
        dif_scores: Deep Isolation Forest anomaly scores (n_samples,)
        temporal_scores: Temporal anomaly scores (n_samples,)
        behavioral_cols: List of behavioral feature column names to include

    Returns:
        (n_samples, n_features) feature matrix
    """
    parts = []

    # Core behavioral features
    if behavioral_cols:
        available = [c for c in behavioral_cols if c in df.columns]
        if available:
            feat_df = df[available].fillna(0).replace([np.inf, -np.inf], 0)
            parts.append(feat_df.values)

    # VAE anomaly score
    if vae_scores is not None:
        parts.append(vae_scores.reshape(-1, 1))

    # VAE latent representation
    if vae_latent is not None:
        parts.append(vae_latent)

    # Deep IF anomaly score
    if dif_scores is not None:
        parts.append(dif_scores.reshape(-1, 1))

    # Temporal anomaly score
    if temporal_scores is not None:
        parts.append(temporal_scores.reshape(-1, 1))

    if not parts:
        raise ValueError("No features provided for fusion")

    return np.hstack(parts)


def train_meta_learner(X_train: np.ndarray, y_train: np.ndarray,
                       weights: np.ndarray = None) -> tuple:
    """Train confidence-weighted gradient boosting meta-learner.

    Args:
        X_train: Training features from seed sets
        y_train: Labels (0=organic, 1=bot, 2=hub)
        weights: Per-sample confidence weights

    Returns:
        (calibrated_model, scaler) tuple
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Handle NaN/inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    base_model.fit(X_scaled, y_train, sample_weight=weights)

    # Calibrate probabilities (Platt scaling)
    try:
        calibrated = CalibratedClassifierCV(base_model, cv='prefit', method='sigmoid')
        calibrated.fit(X_scaled, y_train, sample_weight=weights)
        logger.info("  Meta-learner trained with Platt calibration")
        return calibrated, scaler
    except Exception as e:
        logger.warning(f"  Calibration failed ({e}), using uncalibrated model")
        return base_model, scaler


def predict_with_confidence(model, scaler: StandardScaler,
                            X: np.ndarray) -> tuple:
    """Predict labels with calibrated confidence scores.

    Args:
        model: Trained (calibrated) meta-learner
        scaler: Fitted StandardScaler
        X: Feature matrix (n_samples, n_features)

    Returns:
        (labels, confidences, probabilities) tuple:
          labels: (n_samples,) int labels (0/1/2)
          confidences: (n_samples,) confidence of the prediction (0-1)
          probabilities: (n_samples, 3) class probabilities
    """
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    probas = model.predict_proba(X_scaled)

    # Ensure all 3 classes are represented in output
    if probas.shape[1] < 3:
        # Pad with zeros for missing classes
        full_probas = np.zeros((len(X_scaled), 3))
        classes = model.classes_ if hasattr(model, 'classes_') else np.arange(probas.shape[1])
        for i, c in enumerate(classes):
            if c < 3:
                full_probas[:, c] = probas[:, i]
        probas = full_probas

    labels = np.argmax(probas, axis=1)
    confidences = np.max(probas, axis=1)

    return labels, confidences, probas


def get_feature_importances(model, feature_names: list = None) -> pd.DataFrame:
    """Extract feature importances from the meta-learner.

    Args:
        model: Trained meta-learner (or CalibratedClassifierCV wrapper)
        feature_names: Optional list of feature names

    Returns:
        DataFrame with feature importance rankings
    """
    # Unwrap CalibratedClassifierCV if needed
    base = model
    if hasattr(model, 'estimator'):
        base = model.estimator
    elif hasattr(model, 'calibrated_classifiers_'):
        base = model.calibrated_classifiers_[0].estimator

    if not hasattr(base, 'feature_importances_'):
        logger.warning("Model does not support feature importances")
        return pd.DataFrame()

    importances = base.feature_importances_
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]

    imp_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances,
    }).sort_values('importance', ascending=False)

    return imp_df
