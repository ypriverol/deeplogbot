"""Seed selection for learned bot detection.

Selects high-confidence organic and bot locations from behavioral features
to train the anomaly detection models. Uses a 3-tier system for organic
locations and behavioral heuristics for bot locations.

No hard-coded thresholds are exported -- all thresholds are internal to
seed selection and are only used to identify training examples, not to
classify production data.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def select_organic_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence organic locations for VAE training.

    Uses a 3-tier system:
      Tier A: Individual researchers (very high confidence)
      Tier B: Active researchers (high confidence)
      Tier C: Research groups (moderate confidence)

    Each tier gets a confidence weight used during VAE training.

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    # ------------------------------------------------------------------
    # Tier A: Individual researchers -- very high confidence
    # ------------------------------------------------------------------
    tier_a = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_a &= df['unique_users'] <= 10
    if has('downloads_per_user'):
        tier_a &= df['downloads_per_user'] <= 5
    if has('total_downloads'):
        tier_a &= df['total_downloads'] < 200
    if has('working_hours_ratio'):
        tier_a &= df['working_hours_ratio'] > 0.3
    if has('night_activity_ratio'):
        tier_a &= df['night_activity_ratio'] < 0.5
    if has('years_span'):
        tier_a &= df['years_span'] >= 2

    # ------------------------------------------------------------------
    # Tier B: Active researchers -- high confidence
    # ------------------------------------------------------------------
    tier_b = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_b &= df['unique_users'] <= 50
    if has('downloads_per_user'):
        tier_b &= df['downloads_per_user'] <= 10
    if has('total_downloads'):
        tier_b &= df['total_downloads'] < 1000
    if has('working_hours_ratio'):
        tier_b &= df['working_hours_ratio'] > 0.25
    if has('hourly_entropy'):
        tier_b &= df['hourly_entropy'] > 1.5
    if has('burst_pattern_score'):
        tier_b &= df['burst_pattern_score'] < 0.5
    # Exclude tier A (already captured)
    tier_b &= ~tier_a

    # ------------------------------------------------------------------
    # Tier C: Research groups -- moderate confidence
    # ------------------------------------------------------------------
    tier_c = pd.Series(True, index=df.index)
    if has('unique_users'):
        tier_c &= df['unique_users'] <= 200
    if has('downloads_per_user'):
        tier_c &= df['downloads_per_user'] <= 20
    if has('user_coordination_score'):
        tier_c &= df['user_coordination_score'] < 0.3
    if has('protocol_legitimacy_score'):
        tier_c &= df['protocol_legitimacy_score'] > 0.3
    # Exclude tiers A and B
    tier_c &= ~tier_a & ~tier_b

    # Combine with confidence weights
    seed_mask = tier_a | tier_b | tier_c
    seed_df = df.loc[seed_mask].copy()

    seed_df['seed_confidence'] = 0.0
    seed_df.loc[tier_a[seed_mask].values, 'seed_confidence'] = 1.0
    seed_df.loc[tier_b[seed_mask].values, 'seed_confidence'] = 0.7
    seed_df.loc[tier_c[seed_mask].values, 'seed_confidence'] = 0.4

    logger.info(f"Organic seed: {tier_a.sum()} Tier A, {tier_b.sum()} Tier B, "
                f"{tier_c.sum()} Tier C = {len(seed_df)} total")
    return seed_df


def select_bot_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence bot locations for meta-learner training.

    Uses strong behavioral signals that are very unlikely to be organic.

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    # Strong bot signal: many users with low DL/user (distributed bot farm)
    bot_farm = pd.Series(False, index=df.index)
    if has('unique_users', 'downloads_per_user'):
        bot_farm = (df['unique_users'] > 5000) & (df['downloads_per_user'] < 50)

    # Strong bot signal: extreme nocturnal + no working hours
    nocturnal = pd.Series(False, index=df.index)
    if has('night_activity_ratio', 'working_hours_ratio'):
        nocturnal = (df['night_activity_ratio'] > 0.8) & (df['working_hours_ratio'] < 0.1)

    # Strong bot signal: massive user count (coordinated)
    coordinated = pd.Series(False, index=df.index)
    if has('unique_users', 'downloads_per_user'):
        coordinated = (df['unique_users'] > 10000) & (df['downloads_per_user'] < 20)

    # Strong bot signal: scraper (accesses huge % of all datasets)
    scraper = pd.Series(False, index=df.index)
    if has('unique_projects'):
        scraper = df['unique_projects'] > 15000

    bot_mask = bot_farm | nocturnal | coordinated | scraper
    bot_df = df.loc[bot_mask].copy()

    # Assign confidence based on signal strength
    bot_df['seed_confidence'] = 0.7  # base confidence
    if has('unique_users'):
        bot_df.loc[bot_df['unique_users'] > 10000, 'seed_confidence'] = 0.9
    bot_df.loc[bot_farm[bot_mask].values & nocturnal[bot_mask].values, 'seed_confidence'] = 0.95

    logger.info(f"Bot seed: {bot_farm.sum()} bot farm, {nocturnal.sum()} nocturnal, "
                f"{coordinated.sum()} coordinated, {scraper.sum()} scraper = {len(bot_df)} total")
    return bot_df


def select_hub_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Select high-confidence hub locations.

    Hubs: few users, very high DL/user, legitimate protocols.

    Args:
        df: DataFrame with location features

    Returns:
        DataFrame subset with added 'seed_confidence' column (0-1)
    """
    def has(*cols):
        return all(c in df.columns for c in cols)

    hub_mask = pd.Series(False, index=df.index)
    if has('downloads_per_user', 'unique_users'):
        hub_mask = (df['downloads_per_user'] > 500) & (df['unique_users'] < 100)

    # Exclude nocturnal behavior (likely bot, not hub)
    if has('working_hours_ratio', 'night_activity_ratio'):
        bot_behavior = (df['working_hours_ratio'] < 0.1) & (df['night_activity_ratio'] > 0.7)
        hub_mask &= ~bot_behavior

    hub_df = df.loc[hub_mask].copy()
    hub_df['seed_confidence'] = 0.8

    # Boost confidence for protocol-verified hubs
    if has('aspera_ratio', 'globus_ratio'):
        protocol_verified = (hub_df['aspera_ratio'] > 0.3) | (hub_df['globus_ratio'] > 0.1)
        hub_df.loc[protocol_verified, 'seed_confidence'] = 0.95

    logger.info(f"Hub seed: {len(hub_df)} locations")
    return hub_df
