"""Configuration management for LogGhostbuster.

This module handles:
- Global application configuration (from config.yaml)
- Provider-specific configuration (from providers/<name>/config.yaml)
- Feature column definitions
- Classification rule accessors

Usage:
    from logghostbuster.config import get_provider_config, set_active_provider

    # Use default provider (EBI)
    config = get_provider_config()

    # Switch to a different provider
    set_active_provider('custom')
    config = get_provider_config()
"""

import yaml
import os
from typing import Dict, Any, Optional

# Path to the YAML configuration file
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')


def load_config(config_path: str = CONFIG_FILE_PATH) -> dict:
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Load global configuration on module import
APP_CONFIG = load_config()

# Active provider (lazy-loaded to avoid circular imports)
_active_provider_name: Optional[str] = None
_provider_config_cache: Dict[str, Any] = {}

# Feature columns used in ML models
FEATURE_COLUMNS = [
    'unique_users',
    'downloads_per_user',
    'avg_users_per_hour',
    'max_users_per_hour',
    'user_cv',
    'users_per_active_hour',
    'projects_per_user',
    'hourly_download_std',
    'peak_hour_concentration',
    'working_hours_ratio',
    'hourly_entropy',
    'night_activity_ratio',
    'yearly_entropy',
    'peak_year_concentration',
    'years_span',
    'downloads_per_year',
    'year_over_year_cv',
    'fraction_latest_year',
    'is_new_location',
    'spike_ratio',
    'years_before_latest',
    # Advanced behavioral features (added for deep classification)
    # Created by extract_advanced_behavioral_features
    'burst_pattern_score',
    'circadian_rhythm_deviation',
    'user_coordination_score',
    'hourly_cv_burst',
    'spike_intensity',
    'user_peak_ratio',
    'night_ratio_advanced',
    'work_ratio_advanced',
    'evening_ratio',
    'morning_ratio',
    'user_coordination_std',
    'avg_concurrent_users',
    'max_concurrent_users',
    'is_bursty_advanced',
    'is_nocturnal',
    'is_coordinated',
    # Features from add_bot_interaction_features (created before Isolation Forest)
    'dl_user_per_log_users',
    'user_scarcity_score',
    'download_concentration',
    'temporal_irregularity',
    'bot_composite_score',
    'anomaly_dl_interaction',
    # Features from add_bot_signature_features (created after Isolation Forest when anomaly_score is available)
    'request_velocity',
    'access_regularity',
    'ua_per_user',
    'ip_concentration',
    'session_anomaly',
    'request_pattern_anomaly',
    'weekend_weekday_imbalance',
    'is_high_velocity',
    # Discriminative features (for Stage 2: Malicious vs Legitimate Automation)
    'file_exploration_score',
    'file_mirroring_score',
    'file_entropy',
    'bot_farm_score',
    'user_authenticity_score',
    'user_homogeneity_score',
    'geographic_stability',
    'version_concentration',
    'targets_latest_only',
    'unique_versions',
    'lifespan_days',
    'activity_density',
    'persistence_score',
    'malicious_bot_score',
    'legitimate_automation_score',
    'bot_vs_legitimate_score',
    'is_likely_malicious',
    'is_likely_legitimate_automation',
    'time_series_features_present' # Placeholder feature for deep model to know time series exist
]

# You can add other configurable parameters here as well, e.g.:
# DEFAULT_CONTAMINATION = 0.15
# DEFAULT_EPS = 0.5
# DEFAULT_MIN_SAMPLES = 5
# DEFAULT_TIME_WINDOW = 'month'
# DEFAULT_SEQUENCE_LENGTH = 12


# =====================================================================
# Provider Management Functions
# =====================================================================

def get_active_provider_name() -> str:
    """Get the name of the currently active provider."""
    global _active_provider_name
    if _active_provider_name is None:
        _active_provider_name = "ebi"  # Default provider
    return _active_provider_name


def set_active_provider(name: str) -> None:
    """Set the active provider for configuration lookups.

    Args:
        name: Provider name (e.g., 'ebi', 'custom')
    """
    global _active_provider_name, _provider_config_cache
    _active_provider_name = name
    # Clear cache when switching providers
    _provider_config_cache.clear()


def get_provider_config() -> Dict[str, Any]:
    """Get the configuration for the active provider.

    Returns:
        Provider configuration dictionary with taxonomy and rules.
    """
    global _provider_config_cache

    provider_name = get_active_provider_name()

    if provider_name not in _provider_config_cache:
        # Lazy import to avoid circular dependencies
        try:
            from .providers import get_provider
            provider = get_provider(provider_name)
            _provider_config_cache[provider_name] = provider.get_config()
        except ImportError:
            # Fallback to legacy config if providers module not available
            _provider_config_cache[provider_name] = APP_CONFIG

    return _provider_config_cache[provider_name]


def get_provider_taxonomy() -> Dict[str, Any]:
    """Get the merged taxonomy for the active provider.

    Returns the base taxonomy merged with provider-specific overrides.
    """
    try:
        from .providers import get_provider
        provider = get_provider(get_active_provider_name())
        return provider.get_taxonomy()
    except ImportError:
        # Fallback to classification section of APP_CONFIG
        return APP_CONFIG.get('classification', {})


def list_available_providers() -> list:
    """List all available providers."""
    try:
        from .providers import list_providers
        return list_providers()
    except ImportError:
        return ['ebi']  # Default


# =====================================================================
# Classification Configuration Helper Functions
# =====================================================================

def get_classification_config() -> dict:
    """Get the classification configuration section."""
    return APP_CONFIG.get('classification', {})


def get_hub_protection_rules() -> dict:
    """Get hub protection rules from config."""
    return get_classification_config().get('hub_protection', {
        'high_dl_per_user': {'min_downloads_per_user': 500},
        'few_users_high_dl': {'max_users': 100, 'min_downloads_per_user': 100},
        'single_user': {'max_users': 1, 'min_downloads_per_user': 50},
        'very_few_users': {'max_users': 10, 'min_downloads_per_user': 200},
    })


def get_bot_detection_rules() -> dict:
    """Get bot detection rules from config."""
    return get_classification_config().get('bot_detection', {
        'ground_truth': {'min_users': 10000, 'max_downloads_per_user': 10},
        'large_scale': {'min_users': 5000, 'max_downloads_per_user': 100},
        'many_users_low_dl': {'min_users': 1000, 'max_downloads_per_user': 20},
        'very_many_users_moderate_dl': {
            'min_users': 5000,
            'min_downloads_per_user': 20,
            'max_downloads_per_user': 100
        },
        'moderate_users_suspicious': {
            'min_users': 500,
            'max_users': 5000,
            'min_downloads_per_user': 10,
            'max_downloads_per_user': 50
        },
        'bot_head_override': {'min_users': 100, 'max_downloads_per_user': 100},
    })


def get_download_hub_thresholds() -> dict:
    """Get download hub thresholds from config."""
    return get_classification_config().get('download_hub', {
        'definite': {'min_downloads_per_user': 1000},
        'standard': {'min_downloads_per_user': 500, 'max_users': 100},
    })


def get_independent_user_thresholds() -> dict:
    """Get independent user thresholds from config."""
    return get_classification_config().get('independent_user', {
        'max_users': 5,
        'max_downloads_per_user': 3,
    })


def get_bot_score_weights() -> dict:
    """Get bot score weights from config."""
    return get_classification_config().get('bot_score_weights', {
        'many_users_low_dl': 0.7,
        'very_many_users_moderate_dl': 0.6,
        'moderate_users_suspicious': 0.4,
        'high_anomaly': 0.2,
        'very_high_anomaly': 0.15,
        'non_working_hours': 0.25,
        'low_entropy': 0.15,
        'rule_based_bot': 0.5,
    })


def get_bot_thresholds() -> dict:
    """Get bot detection thresholds from config."""
    return get_classification_config().get('bot_thresholds', {
        'high_anomaly_score': 0.2,
        'very_high_anomaly_score': 0.25,
        'low_working_hours_ratio': 0.3,
        'min_total_downloads': 1000,
        'low_entropy_quantile': 0.2,
    })


def get_category_rules() -> dict:
    """Get category detection rules from config."""
    return get_classification_config().get('categories', {
        'ci_cd_pipeline': {
            'max_users': 10,
            'min_downloads_per_user': 50,
            'max_downloads_per_user': 500,
            'max_file_diversity_ratio': 0.3,
            'min_regularity_score': 0.8,
        },
        'research_group': {
            'min_users': 5,
            'max_users': 50,
            'min_downloads_per_user': 10,
            'max_downloads_per_user': 100,
            'min_working_hours_ratio': 0.5,
            'min_file_diversity_ratio': 0.3,
        },
        'bulk_downloader': {
            'max_users': 5,
            'min_downloads_per_user': 100,
            'max_downloads_per_user': 1000,
        },
        'course_workshop': {
            'min_users': 50,
            'max_users': 500,
            'min_downloads_per_user': 5,
            'max_downloads_per_user': 20,
            'max_file_diversity_ratio': 0.3,
        },
    })


def get_stratified_prefiltering_thresholds() -> dict:
    """Get stratified pre-filtering thresholds from config."""
    return get_classification_config().get('stratified_prefiltering', {
        'obvious_bots': {
            'min_users': 2000,
            'many_users_low_dl': {
                'min_users': 500,
                'max_downloads_per_user': 100,
            },
            'large_scale_low_dl': {
                'min_users': 100,
                'max_downloads_per_user': 200,
            },
        },
        'obvious_legitimate': {
            'max_users': 5,
            'max_downloads_per_user': 3,
            'max_total_downloads': 50,
            'max_anomaly_score': 0.15,
        },
    })


# =====================================================================
# Hierarchical Classification Configuration Functions
# =====================================================================
# These functions support both legacy (APP_CONFIG) and provider-based config.
# When a provider is active, rules are loaded from the provider's taxonomy.

def get_taxonomy_info(use_provider: bool = True) -> dict:
    """Get taxonomy metadata (name, version, description).

    Args:
        use_provider: If True, use active provider's taxonomy. If False, use legacy config.
    """
    if use_provider:
        try:
            taxonomy = get_provider_taxonomy()
            return taxonomy.get('taxonomy', {
                'name': get_active_provider_name(),
                'version': '1.0',
                'description': f'Taxonomy for {get_active_provider_name()} provider'
            })
        except Exception:
            pass

    return get_classification_config().get('taxonomy', {
        'name': 'default',
        'version': '1.0',
        'description': 'Default classification taxonomy'
    })


def get_behavior_type_rules(use_provider: bool = True) -> dict:
    """
    Get Level 1 behavior type classification rules.

    Returns rules for classifying locations as 'organic' or 'automated'.
    Each behavior type has patterns that, if ANY match, classify the location.

    Args:
        use_provider: If True, use active provider's rules. If False, use legacy config.
    """
    default_rules = {
        'organic': {
            'description': 'Human-like download patterns',
            'patterns': [
                {
                    'id': 'default_organic',
                    'working_hours_ratio': {'min': 0.4},
                    'regularity_score': {'max': 0.6}
                }
            ]
        },
        'automated': {
            'description': 'Programmatic download patterns',
            'patterns': [
                {
                    'id': 'default_automated',
                    'regularity_score': {'min': 0.7}
                }
            ]
        }
    }

    if use_provider:
        try:
            taxonomy = get_provider_taxonomy()
            return taxonomy.get('behavior_type', default_rules)
        except Exception:
            pass

    return get_classification_config().get('behavior_type', default_rules)


def get_automation_category_rules(use_provider: bool = True) -> dict:
    """
    Get Level 2 automation category classification rules.

    Returns rules for classifying AUTOMATED locations as 'bot' or 'legitimate_automation'.
    Only applied when behavior_type == 'automated'.

    Args:
        use_provider: If True, use active provider's rules. If False, use legacy config.
    """
    default_rules = {
        'bot': {
            'description': 'Suspicious or malicious automated activity',
            'patterns': [
                {
                    'id': 'default_bot',
                    'unique_users': {'min': 1000},
                    'downloads_per_user': {'max': 50}
                }
            ]
        },
        'legitimate_automation': {
            'description': 'Benign automated systems',
            'patterns': [
                {
                    'id': 'default_legitimate',
                    'downloads_per_user': {'min': 500}
                }
            ]
        }
    }

    if use_provider:
        try:
            taxonomy = get_provider_taxonomy()
            return taxonomy.get('automation_category', default_rules)
        except Exception:
            pass

    return get_classification_config().get('automation_category', default_rules)


def get_subcategory_rules(use_provider: bool = True) -> dict:
    """
    Get Level 3 subcategory classification rules.

    Returns all subcategory rules with their parent category relationships.
    Each subcategory has a 'parent' field indicating which behavior_type or
    automation_category it belongs to.

    Args:
        use_provider: If True, use active provider's rules. If False, use legacy config.
    """
    default_rules = {
        'individual_user': {
            'parent': 'organic',
            'description': 'Single researchers or casual users',
            'unique_users': {'max': 5},
            'downloads_per_user': {'max': 30}
        },
        'research_group': {
            'parent': 'organic',
            'description': 'Small academic research teams',
            'unique_users': {'min': 5, 'max': 50},
            'downloads_per_user': {'min': 10, 'max': 150}
        },
        'mirror': {
            'parent': 'legitimate_automation',
            'description': 'Institutional mirrors',
            'downloads_per_user': {'min': 500}
        },
        'scraper_bot': {
            'parent': 'bot',
            'description': 'High-frequency automated scrapers',
            'unique_users': {'min': 5000},
            'downloads_per_user': {'max': 25}
        }
    }

    if use_provider:
        try:
            taxonomy = get_provider_taxonomy()
            return taxonomy.get('subcategories', default_rules)
        except Exception:
            pass

    return get_classification_config().get('subcategories', default_rules)


def get_organic_subcategory_rules() -> dict:
    """Get subcategory rules for ORGANIC behavior type."""
    all_subcategories = get_subcategory_rules()
    return {
        name: rules for name, rules in all_subcategories.items()
        if rules.get('parent') == 'organic'
    }


def get_bot_subcategory_rules() -> dict:
    """Get subcategory rules for BOT automation category."""
    all_subcategories = get_subcategory_rules()
    return {
        name: rules for name, rules in all_subcategories.items()
        if rules.get('parent') == 'bot'
    }


def get_legitimate_automation_subcategory_rules() -> dict:
    """Get subcategory rules for LEGITIMATE_AUTOMATION category."""
    all_subcategories = get_subcategory_rules()
    return {
        name: rules for name, rules in all_subcategories.items()
        if rules.get('parent') == 'legitimate_automation'
    }


def get_subcategories_by_parent(parent: str) -> dict:
    """
    Get subcategory rules filtered by parent category.

    Args:
        parent: Parent category name ('organic', 'bot', 'legitimate_automation')

    Returns:
        Dict of subcategory rules where parent matches
    """
    all_subcategories = get_subcategory_rules()
    return {
        name: rules for name, rules in all_subcategories.items()
        if rules.get('parent') == parent
    }


def get_hierarchical_classification_config() -> dict:
    """
    Get the complete hierarchical classification configuration.

    Returns a dictionary with all classification levels:
    - taxonomy: Metadata about the taxonomy
    - behavior_type: Level 1 rules (organic vs automated)
    - automation_category: Level 2 rules (bot vs legitimate_automation)
    - subcategories: Level 3 rules (detailed categories)
    """
    return {
        'taxonomy': get_taxonomy_info(),
        'behavior_type': get_behavior_type_rules(),
        'automation_category': get_automation_category_rules(),
        'subcategories': get_subcategory_rules(),
    }
