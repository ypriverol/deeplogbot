import yaml
import os

# Path to the YAML configuration file
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config(config_path: str = CONFIG_FILE_PATH) -> dict:
    """Loads the configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration on module import
APP_CONFIG = load_config()

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
    'time_series_features_present' # Placeholder feature for deep model to know time series exist
]

# You can add other configurable parameters here as well, e.g.:
# DEFAULT_CONTAMINATION = 0.15
# DEFAULT_EPS = 0.5
# DEFAULT_MIN_SAMPLES = 5
# DEFAULT_TIME_WINDOW = 'month'
# DEFAULT_SEQUENCE_LENGTH = 12


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
