"""Rule-based classification for bot and download hub detection.

This module implements a hierarchical classification system:
  - Level 1 (behavior_type): ORGANIC vs AUTOMATED
  - Level 2 (automation_category): BOT vs LEGITIMATE_AUTOMATION (for AUTOMATED only)
  - Level 3 (subcategory): Detailed classification (e.g., mirror, ci_cd_pipeline, etc.)

Configuration is read from `config.yaml` with the hierarchical taxonomy structure.

Users can customize the taxonomy by editing the configuration file.
"""

import os

import pandas as pd

from ...utils import logger
from ...config import CONFIG_FILE_PATH, load_config


# Load rules configuration from YAML
# First try to load from config.yaml (merged location)
# Fall back to rules.yaml for backward compatibility
from ...config import (
    APP_CONFIG,
    get_behavior_type_rules,
    get_automation_category_rules,
    get_subcategory_rules,
    get_taxonomy_info,
)

# Try to get rule_based section from config.yaml
RULES_CONFIG = APP_CONFIG.get('classification', {}).get('rule_based', {})

# If not found in config.yaml, try loading from rules.yaml (backward compatibility)
if not RULES_CONFIG:
    RULES_FILE_PATH = os.path.join(os.path.dirname(CONFIG_FILE_PATH), "rules.yaml")
    try:
        RULES_CONFIG = load_config(RULES_FILE_PATH)
        logger.info(f"Loaded rules from legacy rules.yaml file. Consider migrating to config.yaml")
    except FileNotFoundError:
        logger.warning(f"Rules configuration not found in config.yaml or rules.yaml")
        RULES_CONFIG = {}


def _build_mask_from_section(df: pd.DataFrame, section_name: str) -> pd.Series:
    """Build a boolean mask from a rule-based section (bots, hubs, etc.).
    
    Rules are loaded from config.yaml (classification.rule_based) or rules.yaml.
    """
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
    Legacy classification function for backward compatibility.
    
    This function uses the hierarchical classification system and derives
    legacy is_bot/is_download_hub columns. For new code, use
    classify_locations_hierarchical() directly.
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with is_bot and is_download_hub columns (legacy format)
    """
    # Use hierarchical classification
    df = classify_locations_hierarchical(df)
    
    # Derive legacy columns for backward compatibility
    df = derive_legacy_columns(df)
    
    return df


# =====================================================================
# Hierarchical Classification Functions
# =====================================================================

def _match_pattern(df: pd.DataFrame, pattern: dict) -> pd.Series:
    """
    Match a single pattern against the DataFrame.

    Args:
        df: DataFrame with features
        pattern: Pattern dictionary with field constraints (min/max values)

    Returns:
        Boolean Series indicating which rows match the pattern
    """
    mask = pd.Series(True, index=df.index)

    for field, spec in pattern.items():
        # Skip metadata fields
        if field in ("id", "description", "parent"):
            continue

        if field not in df.columns:
            # If the feature is missing, this pattern cannot fully apply
            # Set mask to False for rows where we can't evaluate
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

    return mask


def _match_any_pattern(df: pd.DataFrame, patterns: list) -> pd.Series:
    """
    Match any pattern from a list of patterns (OR logic).

    Args:
        df: DataFrame with features
        patterns: List of pattern dictionaries

    Returns:
        Boolean Series indicating which rows match ANY pattern
    """
    if not patterns:
        return pd.Series(False, index=df.index)

    overall = pd.Series(False, index=df.index)
    for pattern in patterns:
        pattern_mask = _match_pattern(df, pattern)
        overall |= pattern_mask

    return overall


def classify_behavior_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Level 1 Classification: Classify locations as ORGANIC or AUTOMATED.

    Adds 'behavior_type' column with values: 'organic', 'automated', or 'unknown'.

    Classification logic:
    1. First, check if location matches AUTOMATED patterns
    2. Then, check if location matches ORGANIC patterns
    3. If neither, classify based on heuristics (default: organic for low users)

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with 'behavior_type' column added
    """
    behavior_rules = get_behavior_type_rules()

    # Initialize as unknown
    df['behavior_type'] = 'unknown'

    # Get patterns for each behavior type
    organic_patterns = behavior_rules.get('organic', {}).get('patterns', [])
    automated_patterns = behavior_rules.get('automated', {}).get('patterns', [])

    # Match automated patterns first (more specific)
    automated_mask = _match_any_pattern(df, automated_patterns)

    # Match organic patterns
    organic_mask = _match_any_pattern(df, organic_patterns)

    # Apply classifications
    # Automated takes precedence if both match (e.g., high volume overrides working hours)
    df.loc[organic_mask & ~automated_mask, 'behavior_type'] = 'organic'
    df.loc[automated_mask, 'behavior_type'] = 'automated'

    # Default classification for unmatched locations
    # Small user counts with moderate activity default to organic
    unknown_mask = df['behavior_type'] == 'unknown'
    if unknown_mask.any():
        # Default heuristic: few users = organic, many users = automated
        default_organic = unknown_mask & (df['unique_users'] <= 100)
        default_automated = unknown_mask & (df['unique_users'] > 100)
        df.loc[default_organic, 'behavior_type'] = 'organic'
        df.loc[default_automated, 'behavior_type'] = 'automated'

    return df


def classify_automation_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Level 2 Classification: Classify AUTOMATED locations as BOT or LEGITIMATE_AUTOMATION.

    Only applies to locations where behavior_type == 'automated'.
    Adds 'automation_category' column with values: 'bot', 'legitimate_automation', or None.

    Classification logic:
    1. Check if location matches BOT patterns
    2. Check if location matches LEGITIMATE_AUTOMATION patterns
    3. If both match, use priority rules (legitimate_automation if clearly institutional)

    Args:
        df: DataFrame with 'behavior_type' column

    Returns:
        DataFrame with 'automation_category' column added
    """
    automation_rules = get_automation_category_rules()

    # Initialize as None (not applicable for organic)
    df['automation_category'] = None

    # Only classify automated locations
    automated_mask = df['behavior_type'] == 'automated'
    if not automated_mask.any():
        return df

    automated_df = df[automated_mask]

    # Get patterns for each category
    bot_patterns = automation_rules.get('bot', {}).get('patterns', [])
    legitimate_patterns = automation_rules.get('legitimate_automation', {}).get('patterns', [])

    # Match patterns
    bot_mask = _match_any_pattern(automated_df, bot_patterns)
    legitimate_mask = _match_any_pattern(automated_df, legitimate_patterns)

    # Apply classifications
    # Legitimate automation takes precedence if both match (protect legitimate hubs)
    df.loc[automated_df[bot_mask & ~legitimate_mask].index, 'automation_category'] = 'bot'
    df.loc[automated_df[legitimate_mask].index, 'automation_category'] = 'legitimate_automation'

    # For automated locations that don't match any pattern, use heuristics
    unclassified = automated_mask & df['automation_category'].isna()
    if unclassified.any():
        # Heuristic: high DL/user = legitimate, many users with low DL = bot
        heuristic_bot = unclassified & (df['unique_users'] > 500) & (df['downloads_per_user'] < 100)
        heuristic_legitimate = unclassified & (df['downloads_per_user'] >= 100)
        df.loc[heuristic_bot, 'automation_category'] = 'bot'
        df.loc[heuristic_legitimate, 'automation_category'] = 'legitimate_automation'

    # Any remaining automated locations default to 'bot' (safer default)
    remaining_unclassified = automated_mask & df['automation_category'].isna()
    df.loc[remaining_unclassified, 'automation_category'] = 'bot'

    return df


def classify_subcategory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Level 3 Classification: Assign detailed subcategories.

    Uses the 'parent' field in subcategory rules to only apply subcategories
    to locations with matching behavior_type or automation_category.

    Adds 'subcategory' column with specific category names.

    Args:
        df: DataFrame with 'behavior_type' and 'automation_category' columns

    Returns:
        DataFrame with 'subcategory' column added
    """
    subcategory_rules = get_subcategory_rules()

    # Initialize as unclassified
    df['subcategory'] = 'unclassified'

    for subcat_name, subcat_config in subcategory_rules.items():
        parent = subcat_config.get('parent', None)

        if parent is None:
            continue

        # Determine which locations are eligible for this subcategory
        if parent == 'organic':
            eligible_mask = df['behavior_type'] == 'organic'
        elif parent == 'bot':
            eligible_mask = df['automation_category'] == 'bot'
        elif parent == 'legitimate_automation':
            eligible_mask = df['automation_category'] == 'legitimate_automation'
        else:
            continue

        if not eligible_mask.any():
            continue

        # Build pattern from subcategory config (excluding metadata)
        pattern = {
            k: v for k, v in subcat_config.items()
            if k not in ('parent', 'description')
        }

        # Only apply to eligible AND unclassified locations
        unclassified_mask = df['subcategory'] == 'unclassified'
        candidates = eligible_mask & unclassified_mask

        if not candidates.any():
            continue

        # Match the pattern
        pattern_mask = _match_pattern(df[candidates], pattern)
        matched_indices = df[candidates][pattern_mask].index

        df.loc[matched_indices, 'subcategory'] = subcat_name

    # Set default subcategories for unclassified locations
    # Organic unclassified -> 'individual_user' as default
    organic_unclassified = (df['behavior_type'] == 'organic') & (df['subcategory'] == 'unclassified')
    df.loc[organic_unclassified, 'subcategory'] = 'individual_user'

    # Bot unclassified -> 'generic_bot'
    bot_unclassified = (df['automation_category'] == 'bot') & (df['subcategory'] == 'unclassified')
    df.loc[bot_unclassified, 'subcategory'] = 'generic_bot'

    # Legitimate automation unclassified -> 'generic_automation'
    legit_unclassified = (df['automation_category'] == 'legitimate_automation') & (df['subcategory'] == 'unclassified')
    df.loc[legit_unclassified, 'subcategory'] = 'generic_automation'

    return df




def classify_locations_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full hierarchical classification on locations.

    This is the main entry point for the hierarchical classification system.
    It applies all three levels of classification:
      1. behavior_type: organic vs automated
      2. automation_category: bot vs legitimate_automation (for automated only)
      3. subcategory: detailed classification

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with all classification columns added:
          - behavior_type: 'organic' | 'automated'
          - automation_category: 'bot' | 'legitimate_automation' | None
          - subcategory: detailed category name
    """
    # Ensure expected columns exist
    if "total_downloads" not in df.columns:
        df["total_downloads"] = df["unique_users"] * df["downloads_per_user"]

    # Log taxonomy info
    taxonomy = get_taxonomy_info()
    logger.info(f"Using taxonomy: {taxonomy.get('name', 'default')} v{taxonomy.get('version', '1.0')}")

    # Level 1: Behavior Type
    logger.info("Classifying behavior type (Level 1: organic vs automated)...")
    df = classify_behavior_type(df)

    # Level 2: Automation Category
    logger.info("Classifying automation category (Level 2: bot vs legitimate_automation)...")
    df = classify_automation_category(df)

    # Level 3: Subcategory
    logger.info("Classifying subcategory (Level 3: detailed classification)...")
    df = classify_subcategory(df)

    # Log classification summary
    _log_classification_summary(df)

    return df


def _log_classification_summary(df: pd.DataFrame) -> None:
    """Log a summary of the hierarchical classification results."""
    total = len(df)

    logger.info("\nHierarchical Classification Summary:")
    logger.info("=" * 50)

    # Level 1 summary
    logger.info("\nLevel 1 - Behavior Type:")
    for bt in ['organic', 'automated']:
        count = (df['behavior_type'] == bt).sum()
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {bt.upper()}: {count:,} locations ({pct:.1f}%)")

    # Level 2 summary (for automated only)
    automated_count = (df['behavior_type'] == 'automated').sum()
    if automated_count > 0:
        logger.info("\nLevel 2 - Automation Category (within AUTOMATED):")
        for ac in ['bot', 'legitimate_automation']:
            count = (df['automation_category'] == ac).sum()
            pct = count / automated_count * 100 if automated_count > 0 else 0
            logger.info(f"  {ac.upper()}: {count:,} locations ({pct:.1f}% of automated)")

    # Level 3 summary
    logger.info("\nLevel 3 - Subcategories:")
    subcategory_counts = df['subcategory'].value_counts()
    for subcat, count in subcategory_counts.items():
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  {subcat}: {count:,} locations ({pct:.1f}%)")

