"""Main bot detection pipeline and CLI."""

import os
import argparse
import pandas as pd
import duckdb

from .utils import logger, format_number
from .features import extract_location_features
from .isoforest import train_isolation_forest, compute_feature_importances, classify_locations
from .reports import annotate_downloads
from .reports import generate_report
from .features.schema import LogSchema, EBI_SCHEMA
from .features.base import BaseFeatureExtractor
from typing import Optional, List


def run_bot_annotator(
    input_parquet='original_data/data_downloads_parquet.parquet',
    output_parquet=None,
    output_dir='output/bot_analysis',
    contamination=0.15,
    compute_importances=False,
    schema: Optional[LogSchema] = None,
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None
):
    """
    Main function to detect bots and download hubs, and annotate the parquet file.
    
    Args:
        input_parquet: Path to input parquet file
        output_parquet: Path to output parquet file (default: overwrites input)
        output_dir: Directory for output files
        contamination: Expected proportion of anomalies (default: 0.15)
        compute_importances: Whether to compute feature importances (optional, slower)
        schema: LogSchema defining field mappings (defaults to EBI_SCHEMA for backward compatibility)
        custom_extractors: Optional list of custom feature extractors to apply
    
    Returns:
        Dictionary with detection results and statistics
    """
    logger.info("=" * 70)
    logger.info("Bot and Download Hub Annotator")
    logger.info("=" * 70)
    logger.info(f"Input: {input_parquet}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Contamination: {contamination}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    conn = duckdb.connect()
    conn.execute("SET threads=1")  # Single thread for stability
    conn.execute("SET memory_limit='2GB'")
    conn.execute("SET preserve_insertion_order=false")
    
    try:
        # Step 1: Extract features
        logger.info("\n" + "=" * 70)
        logger.info("Step 1: Extracting location features")
        logger.info("=" * 70)
        location_df = extract_location_features(
            conn, 
            input_parquet,
            schema=schema if schema is not None else EBI_SCHEMA,
            custom_extractors=custom_extractors
        )
        
        # Define ML features (including time-of-day and yearly patterns)
        feature_columns = [
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
            'years_before_latest'
        ]
        
        # Filter valid rows
        valid_mask = location_df[feature_columns].notna().all(axis=1)
        analysis_df = location_df[valid_mask].copy()
        logger.info(f"Analyzing {len(analysis_df):,} locations")
        
        # Step 2: Train Isolation Forest
        logger.info("\n" + "=" * 70)
        logger.info("Step 2: Training Isolation Forest")
        logger.info("=" * 70)
        predictions, scores, _, _ = train_isolation_forest(
            analysis_df, feature_columns, contamination=contamination
        )
        
        analysis_df['is_anomaly'] = predictions == -1
        analysis_df['anomaly_score'] = -scores
        
        n_anomalies = analysis_df['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies:,} anomalous locations")
        
        # Optional: compute feature importances for interpretability
        if compute_importances:
            imp_dir = os.path.join(output_dir, 'feature_importances')
            compute_feature_importances(
                analysis_df,
                feature_columns,
                analysis_df['is_anomaly'],
                imp_dir
            )
        
        # Step 3: Classify locations
        logger.info("\n" + "=" * 70)
        logger.info("Step 3: Classifying locations")
        logger.info("=" * 70)
        analysis_df = classify_locations(analysis_df)
        
        bot_locs = analysis_df[analysis_df['is_bot']].copy()
        hub_locs = analysis_df[analysis_df['is_download_hub']].copy()
        
        logger.info(f"Bot locations: {len(bot_locs):,}")
        logger.info(f"Download hub locations: {len(hub_locs):,}")
        
        # Show top bot locations
        logger.info("\nTop 10 Bot Locations:")
        for _, row in bot_locs.sort_values('unique_users', ascending=False).head(10).iterrows():
            city = str(row['city'])[:20] if pd.notna(row['city']) else 'N/A'
            logger.info(f"  {row['country']:<15} {city:<20} {int(row['unique_users']):>10,} users, "
                       f"{row['downloads_per_user']:.1f} DL/user")
        
        # Show top hub locations
        logger.info("\nTop 10 Download Hub Locations:")
        for _, row in hub_locs.sort_values('downloads_per_user', ascending=False).head(10).iterrows():
            city = str(row['city'])[:20] if pd.notna(row['city']) else 'N/A'
            logger.info(f"  {row['country']:<15} {city:<20} {int(row['unique_users']):>10,} users, "
                       f"{row['downloads_per_user']:.1f} DL/user")
        
        # Step 4: Annotate downloads
        logger.info("\n" + "=" * 70)
        logger.info("Step 4: Annotating downloads")
        logger.info("=" * 70)
        
        # Output to specified file or same file (overwrite)
        if output_parquet is None:
            output_parquet = input_parquet
        
        annotate_downloads(conn, input_parquet, output_parquet, 
                          bot_locs, hub_locs, output_dir)
        
        # Step 5: Calculate statistics
        logger.info("\n" + "=" * 70)
        logger.info("Step 5: Calculating statistics")
        logger.info("=" * 70)
        
        escaped_output = os.path.abspath(output_parquet).replace("'", "''")
        stats_result = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN bot THEN 1 ELSE 0 END) as bots,
                SUM(CASE WHEN download_hub THEN 1 ELSE 0 END) as hubs,
                SUM(CASE WHEN NOT bot AND NOT download_hub THEN 1 ELSE 0 END) as normal
            FROM read_parquet('{escaped_output}')
        """).df().iloc[0]
        
        stats = {
            'total': int(stats_result['total']),
            'bots': int(stats_result['bots']),
            'hubs': int(stats_result['hubs']),
            'normal': int(stats_result['normal'])
        }
        
        logger.info(f"\nTotal downloads: {format_number(stats['total'])}")
        logger.info(f"Bot downloads: {format_number(stats['bots'])} ({stats['bots']/stats['total']*100:.2f}%)")
        logger.info(f"Hub downloads: {format_number(stats['hubs'])} ({stats['hubs']/stats['total']*100:.2f}%)")
        logger.info(f"Normal downloads: {format_number(stats['normal'])} ({stats['normal']/stats['total']*100:.2f}%)")
        
        # Step 6: Save analysis and generate report
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: Generating reports")
        logger.info("=" * 70)
        
        # Save full analysis
        analysis_file = os.path.join(output_dir, 'location_analysis.csv')
        analysis_df.to_csv(analysis_file, index=False)
        logger.info(f"Location analysis saved to: {analysis_file}")
        
        # Generate report
        generate_report(analysis_df, bot_locs, hub_locs, stats, output_dir, 
                       schema=schema if schema is not None else EBI_SCHEMA,
                       available_features=feature_columns)
        
        logger.info("\n" + "=" * 70)
        logger.info("Bot Annotation Complete!")
        logger.info("=" * 70)
        logger.info(f"\nOutput files:")
        logger.info(f"  - {output_parquet} (annotated with 'bot' and 'download_hub' columns)")
        logger.info(f"  - {output_dir}/bot_detection_report.txt")
        logger.info(f"  - {output_dir}/location_analysis.csv")
        
        return {
            'bot_locations': len(bot_locs),
            'hub_locations': len(hub_locs),
            'stats': stats,
            'output_parquet': output_parquet
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
    finally:
        conn.close()


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

