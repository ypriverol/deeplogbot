"""Command-line interface for LogGhostbuster."""

import argparse

from .utils import logger
from .main import run_bot_annotator


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Annotate downloads with bot and download_hub flags using ML detection'
    )
    parser.add_argument('--input', '-i', 
                       default='original_data/data_downloads_parquet.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', '-out',
                       default=None,
                       help='Output parquet file (default: overwrites input)')
    parser.add_argument('--output-dir', '-o',
                       default='output/bot_analysis',
                       help='Output directory for reports')
    parser.add_argument('--contamination', '-c', type=float, default=0.15,
                       help='Expected proportion of anomalies (default: 0.15)')
    parser.add_argument('--compute-importances', action='store_true',
                       help='Compute feature importances (optional, slower)')
    
    args = parser.parse_args()
    
    try:
        run_bot_annotator(
            args.input,
            args.output,
            args.output_dir,
            args.contamination,
            compute_importances=args.compute_importances
        )
        
        logger.info("\nDone!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

