"""Learned deep classification for bot / hub / organic detection.

Replaces the previous rule-heavy approach (70+ threshold constants, ~4300 lines)
with a learned pipeline:

  1. **Seed Selection** – identifies high-confidence organic / bot / hub locations
     using simple behavioural heuristics (training data only, not final decisions).
  2. **Organic VAE** – learns the manifold of normal download behaviour; high
     reconstruction error ⇒ anomalous.
  3. **Deep Isolation Forest** – non-linear anomaly detection via neural
     projections (DeepOD) with sklearn fallback.
  4. **Temporal Consistency** – modified z-score spike detection without fixed
     thresholds.
  5. **Fusion Meta-learner** – gradient-boosted classifier combining all anomaly
     signals with Platt-calibrated probabilities.

The only hand-tuned rules remaining are hub-protection (strong structural signal).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from ...utils import logger
from .post_classification import (
    apply_hub_protection,
    log_prediction_summary,
    log_hierarchical_summary,
)

from .seed_selection import select_organic_seed, select_bot_seed, select_hub_seed
from .organic_vae import (
    train_organic_vae,
    compute_vae_anomaly_scores,
    train_deep_isolation_forest,
)
from .temporal_consistency import compute_temporal_anomaly
from .fusion import (
    LABEL_ORGANIC, LABEL_BOT, LABEL_HUB,
    prepare_fusion_features,
    train_meta_learner,
    predict_with_confidence,
    get_feature_importances,
)


# ---------------------------------------------------------------------------
# Core behavioural features used as fusion inputs
# ---------------------------------------------------------------------------

BEHAVIORAL_FEATURE_COLS = [
    'unique_users', 'downloads_per_user', 'total_downloads',
    'working_hours_ratio', 'night_activity_ratio', 'hourly_entropy',
    'burst_pattern_score', 'user_coordination_score',
    'spike_ratio', 'fraction_latest_year', 'year_over_year_cv',
    'years_span', 'protocol_legitimacy_score',
    'aspera_ratio', 'globus_ratio',
    'regularity_score', 'file_diversity_ratio',
    'bot_composite_score', 'user_scarcity_score',
    'download_concentration', 'temporal_irregularity',
    'request_velocity', 'access_regularity',
    'weekend_weekday_imbalance',
    'user_entropy', 'user_gini_coefficient',
    'single_download_user_ratio', 'power_user_ratio',
    'session_duration_cv', 'inter_session_regularity',
    'momentum_score', 'recent_activity_ratio',
    'unique_projects',
]


# ===================================================================
# Main entry point
# ===================================================================

def classify_locations_deep(
    df: pd.DataFrame,
    feature_columns: List[str],
    compute_feature_importance: bool = False,
    feature_importance_output_dir: Optional[str] = None,
    input_parquet: Optional[str] = None,
    conn=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Classify locations using the learned deep pipeline.

    Pipeline: Seed → VAE → Deep IF → Temporal → Fusion → Hub Protection.

    Args:
        df: DataFrame with location features.
        feature_columns: Available feature column names.
        compute_feature_importance: Log and optionally save feature importances.
        feature_importance_output_dir: Directory to save importance CSVs.
        input_parquet: Path to raw parquet (for temporal spike detection).
        conn: DuckDB connection (for temporal spike detection).

    Returns:
        (classified_df, empty_cluster_df) matching the legacy contract.
    """
    logger.info("=" * 70)
    logger.info("LEARNED DEEP CLASSIFICATION")
    logger.info("  Pipeline: Seed → VAE → Deep IF → Temporal → Fusion")
    logger.info("=" * 70)

    n_locations = len(df)
    logger.info(f"  Locations: {n_locations:,}")

    # ------------------------------------------------------------------
    # 0. Initialize output columns
    # ------------------------------------------------------------------
    df['user_category'] = 'normal'
    df['classification_confidence'] = 0.0
    df['needs_review'] = False
    df['behavior_type'] = 'organic'
    df['automation_category'] = None
    df['is_bot_neural'] = False
    df['is_protected_hub'] = False

    # Ensure total_downloads exists
    if 'total_downloads' not in df.columns or df['total_downloads'].eq(0).all():
        if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
            df['total_downloads'] = df['unique_users'] * df['downloads_per_user']

    # ------------------------------------------------------------------
    # 1. Seed selection
    # ------------------------------------------------------------------
    logger.info("\n  Phase 1: Seed selection ...")
    organic_seed = select_organic_seed(df)
    bot_seed = select_bot_seed(df)
    hub_seed = select_hub_seed(df)

    n_organic_seed = len(organic_seed)
    n_bot_seed = len(bot_seed)
    n_hub_seed = len(hub_seed)
    logger.info(f"    Seeds: {n_organic_seed} organic, {n_bot_seed} bot, {n_hub_seed} hub")

    if n_organic_seed < 50:
        logger.warning("    Very few organic seeds – VAE training may be unreliable")
    if n_bot_seed < 10:
        logger.warning("    Very few bot seeds – meta-learner may under-detect bots")

    # ------------------------------------------------------------------
    # 2. Prepare feature matrix for all locations
    # ------------------------------------------------------------------
    available_behavioral = [c for c in BEHAVIORAL_FEATURE_COLS if c in df.columns]
    # Also include any feature_columns not already covered
    extra = [c for c in feature_columns
             if c in df.columns and c not in available_behavioral
             and c != 'time_series_features_present']
    all_feature_cols = list(dict.fromkeys(available_behavioral + extra))

    logger.info(f"    Using {len(all_feature_cols)} behavioural features")

    X_all = df[all_feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    # ------------------------------------------------------------------
    # 3. Organic VAE
    # ------------------------------------------------------------------
    logger.info("\n  Phase 2: Training Organic VAE ...")
    organic_idx = organic_seed.index
    X_organic = df.loc[organic_idx, all_feature_cols].fillna(0).replace(
        [np.inf, -np.inf], 0).values
    organic_weights = organic_seed['seed_confidence'].values

    vae_scores = np.zeros(n_locations)
    vae_latent = np.zeros((n_locations, 16))

    if len(X_organic) >= 30:
        try:
            vae_model, vae_scaler = train_organic_vae(
                X_organic, weights=organic_weights,
                latent_dim=16, epochs=100, batch_size=256,
            )
            vae_scores, vae_latent = compute_vae_anomaly_scores(
                vae_model, vae_scaler, X_all,
            )
            logger.info(f"    VAE anomaly scores: mean={vae_scores.mean():.4f}, "
                        f"std={vae_scores.std():.4f}, max={vae_scores.max():.4f}")
        except Exception as e:
            logger.warning(f"    VAE training failed ({e}), using zeros")
    else:
        logger.warning("    Not enough organic seeds for VAE, skipping")

    # ------------------------------------------------------------------
    # 4. Deep Isolation Forest
    # ------------------------------------------------------------------
    logger.info("\n  Phase 3: Deep Isolation Forest (all locations) ...")
    try:
        dif_scores, _ = train_deep_isolation_forest(X_all)
        logger.info(f"    DIF anomaly scores: mean={dif_scores.mean():.4f}, "
                    f"std={dif_scores.std():.4f}")
    except Exception as e:
        logger.warning(f"    Deep IF failed ({e}), using zeros")
        dif_scores = np.zeros(n_locations)

    # ------------------------------------------------------------------
    # 5. Temporal consistency
    # ------------------------------------------------------------------
    logger.info("\n  Phase 4: Temporal consistency ...")
    df = compute_temporal_anomaly(df, conn=conn, input_parquet=input_parquet)
    temporal_scores = df['temporal_anomaly_score'].fillna(0).values

    # ------------------------------------------------------------------
    # 6. Fusion meta-learner
    # ------------------------------------------------------------------
    logger.info("\n  Phase 5: Training fusion meta-learner ...")

    # Assemble training data from seeds
    seed_indices = []
    seed_labels = []
    seed_weights = []

    for idx in organic_seed.index:
        seed_indices.append(df.index.get_loc(idx))
        seed_labels.append(LABEL_ORGANIC)
        seed_weights.append(organic_seed.loc[idx, 'seed_confidence'])

    for idx in bot_seed.index:
        seed_indices.append(df.index.get_loc(idx))
        seed_labels.append(LABEL_BOT)
        seed_weights.append(bot_seed.loc[idx, 'seed_confidence'])

    for idx in hub_seed.index:
        seed_indices.append(df.index.get_loc(idx))
        seed_labels.append(LABEL_HUB)
        seed_weights.append(hub_seed.loc[idx, 'seed_confidence'])

    seed_indices = np.array(seed_indices)
    seed_labels = np.array(seed_labels)
    seed_weights = np.array(seed_weights)

    logger.info(f"    Training set: {len(seed_indices)} seeds "
                f"({(seed_labels == LABEL_ORGANIC).sum()} organic, "
                f"{(seed_labels == LABEL_BOT).sum()} bot, "
                f"{(seed_labels == LABEL_HUB).sum()} hub)")

    # Build fusion features for ALL locations
    X_fusion_all = prepare_fusion_features(
        df, vae_scores=vae_scores, vae_latent=vae_latent,
        dif_scores=dif_scores, temporal_scores=temporal_scores,
        behavioral_cols=all_feature_cols,
    )

    # Extract training subset
    X_train = X_fusion_all[seed_indices]
    y_train = seed_labels

    # Train
    meta_model, meta_scaler = train_meta_learner(
        X_train, y_train, weights=seed_weights,
    )

    # Predict on ALL locations
    labels, confidences, probas = predict_with_confidence(
        meta_model, meta_scaler, X_fusion_all,
    )

    # Log feature importances
    if compute_feature_importance:
        # Build feature names: behavioral cols + vae_score + vae_latent_0..15 + dif_score + temporal_score
        feat_names = list(all_feature_cols)
        feat_names.append('vae_anomaly_score')
        feat_names.extend([f'vae_latent_{i}' for i in range(vae_latent.shape[1])])
        feat_names.append('dif_anomaly_score')
        feat_names.append('temporal_anomaly_score')
        imp_df = get_feature_importances(meta_model, feat_names)
        if not imp_df.empty:
            logger.info("\n  Top 15 feature importances:")
            for _, row in imp_df.head(15).iterrows():
                logger.info(f"    {row['feature']:40s} {row['importance']:.4f}")
            if feature_importance_output_dir:
                import os
                os.makedirs(feature_importance_output_dir, exist_ok=True)
                imp_df.to_csv(os.path.join(feature_importance_output_dir,
                                           'fusion_importances.csv'), index=False)

    # ------------------------------------------------------------------
    # 7. Map fusion labels → hierarchical classification
    # ------------------------------------------------------------------
    logger.info("\n  Phase 6: Mapping predictions to hierarchical labels ...")

    df['classification_confidence'] = confidences
    df['prob_organic'] = probas[:, LABEL_ORGANIC]
    df['prob_bot'] = probas[:, LABEL_BOT]
    df['prob_hub'] = probas[:, LABEL_HUB]

    # Organic
    organic_mask = labels == LABEL_ORGANIC
    df.loc[df.index[organic_mask], 'user_category'] = 'normal'
    df.loc[df.index[organic_mask], 'behavior_type'] = 'organic'
    df.loc[df.index[organic_mask], 'automation_category'] = None

    # Refine organic → independent_user for small locations
    if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
        indep_mask = (
            organic_mask &
            (df['unique_users'].values <= 10) &
            (df['downloads_per_user'].values <= 5)
        )
        df.loc[df.index[indep_mask], 'user_category'] = 'independent_user'

    # Bot
    bot_mask = labels == LABEL_BOT
    df.loc[df.index[bot_mask], 'user_category'] = 'bot'
    df.loc[df.index[bot_mask], 'behavior_type'] = 'automated'
    df.loc[df.index[bot_mask], 'automation_category'] = 'bot'
    df.loc[df.index[bot_mask], 'is_bot_neural'] = True

    # Hub
    hub_mask = labels == LABEL_HUB
    df.loc[df.index[hub_mask], 'user_category'] = 'download_hub'
    df.loc[df.index[hub_mask], 'behavior_type'] = 'automated'
    df.loc[df.index[hub_mask], 'automation_category'] = 'legitimate_automation'

    # Flag low-confidence predictions for review
    df.loc[confidences < 0.5, 'needs_review'] = True

    log_prediction_summary(df, labels, confidences)

    # ------------------------------------------------------------------
    # 8. Hub protection (structural override)
    # ------------------------------------------------------------------
    logger.info("\n  Phase 7: Hub protection ...")
    df = apply_hub_protection(df)

    # ------------------------------------------------------------------
    # 9. Derive boolean columns & finalize
    # ------------------------------------------------------------------
    logger.info("\n  Phase 8: Deriving final labels ...")
    df['is_bot'] = df['automation_category'] == 'bot'
    df['is_hub'] = df['automation_category'] == 'legitimate_automation'
    df['is_organic'] = df['behavior_type'] == 'organic'
    df['is_download_hub'] = df['is_hub']  # Legacy alias
    log_hierarchical_summary(df)

    cluster_df = pd.DataFrame()
    return df, cluster_df
