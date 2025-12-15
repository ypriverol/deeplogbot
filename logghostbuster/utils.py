"""Utility functions for LogGhostbuster."""

import logging
import sys
import warnings

warnings.filterwarnings('ignore')

# Configure logging with immediate flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration
)

logger = logging.getLogger(__name__)

# Ensure logs are flushed immediately
for handler in logger.handlers:
    handler.flush()

# Also flush stdout/stderr
if hasattr(sys.stdout, 'flush'):
    sys.stdout.flush()
if hasattr(sys.stderr, 'flush'):
    sys.stderr.flush()


def format_number(num):
    """Format number with K/M suffix."""
    if num >= 1e6:
        return f'{num/1e6:.1f}M'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    return str(int(num))

