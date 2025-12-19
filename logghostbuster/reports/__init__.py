"""Report generation and annotation package for bot detection results."""

from .reporting import generate_report, ReportGenerator
from .annotation import annotate_downloads

__all__ = [
    "generate_report",
    "ReportGenerator",
    "annotate_downloads",
]
