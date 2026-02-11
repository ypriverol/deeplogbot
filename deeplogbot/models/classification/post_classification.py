"""Post-classification steps shared across classification methods.

Includes hub protection (structural override) and logging summaries.
These are rule-based refinements applied *after* the learned pipeline
produces initial labels.
"""

import numpy as np
import pandas as pd

from ...utils import logger
from ...config import get_hub_protection_rules

from .fusion import LABEL_NAMES


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _has_required_columns(df: pd.DataFrame, *columns: str) -> bool:
    """Check if DataFrame has all required columns."""
    return all(col in df.columns for col in columns)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_prediction_summary(df: pd.DataFrame, labels: np.ndarray,
                           confidences: np.ndarray) -> None:
    """Log fusion prediction summary."""
    n = len(df)
    for lbl, name in LABEL_NAMES.items():
        mask = labels == lbl
        count = mask.sum()
        mean_conf = confidences[mask].mean() if count > 0 else 0
        logger.info(f"    {name.upper():8s}: {count:,} ({count/n*100:.1f}%), "
                    f"mean confidence {mean_conf:.3f}")

    low_conf = (confidences < 0.5).sum()
    logger.info(f"    Low confidence (<0.5): {low_conf:,} ({low_conf/n*100:.1f}%)")


def log_hierarchical_summary(df: pd.DataFrame) -> None:
    """Log hierarchical classification summary."""
    total = len(df)
    if total == 0:
        return

    logger.info("\n  " + "=" * 60)
    logger.info("  HIERARCHICAL CLASSIFICATION SUMMARY")
    logger.info("  " + "=" * 60)

    logger.info("\n  Level 1 – Behaviour Type:")
    for bt in ['organic', 'automated']:
        count = (df['behavior_type'] == bt).sum()
        pct = count / total * 100
        logger.info(f"    {bt.upper()}: {count:,} ({pct:.1f}%)")

    automated_count = (df['behavior_type'] == 'automated').sum()
    if automated_count > 0:
        logger.info("\n  Level 2 – Automation Category (within AUTOMATED):")
        for ac in ['bot', 'legitimate_automation']:
            count = (df['automation_category'] == ac).sum()
            pct = count / automated_count * 100
            logger.info(f"    {ac.upper()}: {count:,} ({pct:.1f}% of automated)")

    # Final category counts
    if 'is_bot' in df.columns:
        bot_count = df['is_bot'].sum() if 'is_bot' in df.columns else 0
        hub_count = df['is_hub'].sum() if 'is_hub' in df.columns else 0
        organic_count = df['is_organic'].sum() if 'is_organic' in df.columns else total - bot_count - hub_count
        logger.info(f"\n  Final: {bot_count:,} bot, {hub_count:,} hub, {organic_count:,} organic")


# ---------------------------------------------------------------------------
# Hub protection
# ---------------------------------------------------------------------------

def apply_hub_protection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict hub protection rules.

    Definite hub patterns should NEVER be classified as bots.
    Uses structural signals (few users, very high DL/user, legitimate protocols)
    that are reliable regardless of the learned model's output.
    """
    hub_rules = get_hub_protection_rules()

    if 'is_protected_hub' not in df.columns:
        df['is_protected_hub'] = False

    definite_hub_mask = pd.Series(False, index=df.index)

    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        high_dl_rule = hub_rules.get('high_dl_per_user', {})
        few_users_rule = hub_rules.get('few_users_high_dl', {})
        single_user_rule = hub_rules.get('single_user', {})
        very_few_rule = hub_rules.get('very_few_users', {})
        behavioral_rules = hub_rules.get('behavioral_exclusion', {})

        # Behavioural exclusion: don't protect if clearly bot-like
        behavioral_exclusion = pd.Series(False, index=df.index)
        if _has_required_columns(df, 'working_hours_ratio', 'night_activity_ratio'):
            behavioral_exclusion = (
                (df['working_hours_ratio'] < behavioral_rules.get('max_working_hours_ratio', 0.1)) &
                (df['night_activity_ratio'] > behavioral_rules.get('min_night_activity_ratio', 0.7))
            )

        # Scraper exclusion
        scraper_exclusion = pd.Series(False, index=df.index)
        if 'unique_projects' in df.columns:
            scraper_exclusion = df['unique_projects'] > 15000

        # Protocol-based hub detection
        protocol_hub = pd.Series(False, index=df.index)
        if _has_required_columns(df, 'aspera_ratio', 'globus_ratio'):
            protocol_hub = (df['aspera_ratio'] > 0.3) | (df['globus_ratio'] > 0.1)

        definite_hub_mask = (
            ((df['downloads_per_user'] > high_dl_rule.get('min_downloads_per_user', 500)) &
             (df['unique_users'] <= high_dl_rule.get('max_users', 200))) |
            ((df['unique_users'] <= few_users_rule.get('max_users', 100)) &
             (df['downloads_per_user'] > few_users_rule.get('min_downloads_per_user', 100))) |
            ((df['unique_users'] <= single_user_rule.get('max_users', 1)) &
             (df['downloads_per_user'] > single_user_rule.get('min_downloads_per_user', 50))) |
            ((df['unique_users'] <= very_few_rule.get('max_users', 10)) &
             (df['downloads_per_user'] > very_few_rule.get('min_downloads_per_user', 200))) |
            protocol_hub
        ) & ~behavioral_exclusion & ~scraper_exclusion

    df.loc[definite_hub_mask, 'is_protected_hub'] = True
    df.loc[definite_hub_mask, 'is_bot_neural'] = False
    df.loc[definite_hub_mask, 'behavior_type'] = 'automated'
    df.loc[definite_hub_mask, 'automation_category'] = 'legitimate_automation'
    if 'user_category' in df.columns:
        df.loc[definite_hub_mask & (df['user_category'] == 'bot'), 'user_category'] = 'download_hub'

    n_protected = definite_hub_mask.sum()
    if n_protected > 0:
        logger.info(f"    Hub protection: {n_protected:,} locations protected")

    return df


