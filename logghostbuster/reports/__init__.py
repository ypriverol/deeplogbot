"""Report generation and annotation package for bot detection results.

This package provides:
- Text reports with detailed analysis
- HTML reports with interactive visualizations
- Comprehensive statistics computation
- Visualization generation (plots and charts)
- Parquet annotation with classification columns
"""

from .reporting import generate_report, ReportGenerator
from .annotation import annotate_downloads
from .statistics import StatisticsCalculator
from .visualizations import VisualizationGenerator
from .html_report import HTMLReportGenerator

__all__ = [
    # Main report generation
    "generate_report",
    "ReportGenerator",
    # Annotation
    "annotate_downloads",
    # Statistics
    "StatisticsCalculator",
    # Visualizations
    "VisualizationGenerator",
    # HTML reports
    "HTMLReportGenerator",
]
