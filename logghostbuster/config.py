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
