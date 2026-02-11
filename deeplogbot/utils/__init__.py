"""Utility functions package."""

import logging
import sys
import warnings

warnings.filterwarnings('ignore')

# Configure logging with immediate flushing
# Use DEBUG level by default, or override with environment variable LOG_LEVEL
import os
log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
log_level = getattr(logging, log_level, logging.DEBUG)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,  # Force reconfiguration
    stream=sys.stdout  # Explicitly write to stdout for immediate visibility
)

logger = logging.getLogger(__name__)

# Although stream=sys.stdout and python -u should handle flushing,
# aggressive flushing can be added here if needed, but it's often redundant.
# for handler in logger.handlers:
#     handler.flush()
# if hasattr(sys.stdout, 'flush'):
#     sys.stdout.flush()
# if hasattr(sys.stderr, 'flush'):
#     sys.stderr.flush()


def format_number(num):
    """Format number with K/M suffix."""
    if num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    return str(int(num))


# Geographic utilities
from .geography import haversine_distance, parse_geo_location, group_nearby_locations

__all__ = [
    "logger",
    "format_number",
    "haversine_distance",
    "parse_geo_location",
    "group_nearby_locations",
]
