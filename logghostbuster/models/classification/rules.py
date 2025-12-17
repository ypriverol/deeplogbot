"""Rule-based classification for bot and download hub detection.

This module now reads its thresholds from a YAML configuration file
(`rules.yaml`) so that users can adjust classes and rules without
changing code.
"""

import os

import pandas as pd

from ...utils import logger
from ...config import CONFIG_FILE_PATH, load_config


# Load rules configuration from YAML
RULES_FILE_PATH = os.path.join(os.path.dirname(CONFIG_FILE_PATH), "rules.yaml")
try:
    RULES_CONFIG = load_config(RULES_FILE_PATH)
except FileNotFoundError:
    logger.warning(f"Rules configuration file not found: {RULES_FILE_PATH}")
    RULES_CONFIG = {}


def _build_mask_from_section(df: pd.DataFrame, section_name: str) -> pd.Series:
    """Build a boolean mask from a rules.yaml section (bots, hubs, etc.)."""
    section = RULES_CONFIG.get(section_name, {})
    patterns = section.get("patterns", [])
    require_anomaly = section.get("require_anomaly", False)

    if require_anomaly:
        base = df.get("is_anomaly", pd.Series(False, index=df.index)).astype(bool)
    else:
        base = pd.Series(True, index=df.index)

    overall = pd.Series(False, index=df.index)

    for pattern in patterns:
        mask = base.copy()
        for field, spec in pattern.items():
            if field in ("id", "description"):
                continue

            if field not in df.columns:
                # If the feature is missing, this pattern cannot apply
                mask &= False
                continue

            col = df[field]
            if isinstance(spec, dict):
                min_val = spec.get("min", None)
                max_val = spec.get("max", None)
                if min_val is not None:
                    mask &= col >= min_val
                if max_val is not None:
                    mask &= col <= max_val
            else:
                # Simple equality rule if a scalar is given
                mask &= col == spec

        overall |= mask

    return overall


def classify_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify locations into categories based on YAML-defined rules.

    YAML sections:
      - bots: define bot rules (anomalous + many users with low/moderate DL/user)
      - hubs: define download hub rules
      - independent_users: define independent user rules (few users, low DL/user)
      - other: anomalous locations not matching any pattern
    """
    if not RULES_CONFIG:
        logger.warning("RULES_CONFIG is empty; no bot/hub rules will be applied.")
        df["is_bot"] = False
        df["is_download_hub"] = False
        return df

    # Ensure expected columns exist
    if "total_downloads" not in df.columns:
        df["total_downloads"] = df["unique_users"] * df["downloads_per_user"]

    # Initialize flags
    df["is_bot"] = False
    df["is_download_hub"] = False

    # BOT rules
    bot_mask = _build_mask_from_section(df, "bots")

    # HUB rules
    hub_mask = _build_mask_from_section(df, "hubs")

    df.loc[bot_mask, "is_bot"] = True
    df.loc[hub_mask, "is_download_hub"] = True

    return df


