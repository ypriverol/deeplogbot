"""Main bot detection pipeline and CLI."""

import os
import argparse
import pandas as pd
import duckdb
import tempfile

from .utils import logger, format_number
from .features.providers.ebi import extract_location_features
from .config import FEATURE_COLUMNS, APP_CONFIG
from .models import (
    train_isolation_forest,
    compute_feature_importances,
    classify_locations_hierarchical,
    classify_locations_deep,
)

from .reports import annotate_downloads
from .reports import generate_report
from .features.providers.ebi import LogSchema, EBI_SCHEMA
from .features.base import BaseFeatureExtractor
from typing import Optional, List


def sample_parquet_records(conn, input_parquet: str, sample_size: int, schema: Optional[LogSchema] = None) -> str:
    """
    Randomly sample N records from a parquet file across all years.
    
    Args:
        conn: DuckDB connection
        input_parquet: Path to input parquet file
        sample_size: Number of records to sample
        schema: LogSchema for field mappings (optional)
    
    Returns:
        Path to temporary parquet file with sampled records
    """
    if schema is None:
        schema = EBI_SCHEMA
    
    logger.info(f"Sampling {sample_size:,} records randomly from all years...")
    
    escaped_path = os.path.abspath(input_parquet).replace("'", "''")

    # Get DuckDB configuration
    duckdb_config = APP_CONFIG.get('duckdb', {})
    memory_limit = duckdb_config.get('memory_limit', '16GB')
    max_temp_directory_size = duckdb_config.get('max_temp_directory_size', '20GiB')
    temp_directory_config = duckdb_config.get('temp_directory', './duckdb-tmp/')

    # Resolve absolute path for temp directory
    temp_directory_abs = os.path.abspath(temp_directory_config)

    # Create temp directory if it doesn't exist
    os.makedirs(temp_directory_abs, exist_ok=True)

    # Set DuckDB memory limits and temp directory
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA max_temp_directory_size='{max_temp_directory_size}'")
    conn.execute(f"PRAGMA temp_directory='{temp_directory_abs}'")

    # First, get total count
    count_result = conn.execute(f"SELECT COUNT(*) as total FROM read_parquet('{escaped_path}')").df()
    total_records = count_result.iloc[0]['total']
    
    logger.info(f"Total records in file: {total_records:,}")
    
    if sample_size >= total_records:
        logger.info(f"Sample size ({sample_size:,}) >= total records ({total_records:,}), using original file")
        return input_parquet
    
    # Create temporary file for sampled data
    temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet', prefix='deeplogbot_sample_')
    os.close(temp_fd)
    escaped_temp = os.path.abspath(temp_path).replace("'", "''")
    
    # Use TABLESAMPLE RESERVOIR for efficient exact-count sampling (single pass, memory-efficient)
    # RESERVOIR sampling is designed for exact row counts and is more efficient than ORDER BY RANDOM()
    sample_query = f"""
    COPY (
        SELECT * FROM read_parquet('{escaped_path}')
        TABLESAMPLE RESERVOIR ({sample_size} ROWS)
    ) TO '{escaped_temp}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
    """
    
    logger.info(f"Sampling {sample_size:,} records using efficient RESERVOIR sampling method...")
    conn.execute(sample_query)
    
    # Verify sample size
    sample_count = conn.execute(f"SELECT COUNT(*) as total FROM read_parquet('{escaped_temp}')").df()
    actual_sample = sample_count.iloc[0]['total']
    logger.info(f"Sampled {actual_sample:,} records to temporary file: {temp_path}")
    
    return temp_path


def run_bot_annotator(
    input_parquet='original_data/data_downloads_parquet.parquet',
    output_parquet=None,
    output_dir='output/bot_analysis',
    contamination=0.15,
    compute_importances=False,
    schema: Optional[LogSchema] = None,
    custom_extractors: Optional[List[BaseFeatureExtractor]] = None,
    sample_size: Optional[int] = None,
    classification_method: str = 'rules',
    min_location_downloads: Optional[int] = None,
    annotate: bool = True,
    output_strategy: str = 'new_file',
    provider: str = 'ebi',
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
        sample_size: Optional number of records to randomly sample from all years (default: None, uses all data)
        classification_method: Classification method to use - 'rules' (default) or 'deep'
        min_location_downloads: Minimum downloads required for a location to be included (default: None, uses schema default of 1)
        annotate: Whether to write an annotated parquet file with bot/download_hub flags
                  (default: True). NOTE: When sampling is enabled (sample_size is not None),
                  annotation is automatically disabled to avoid overwriting the full dataset
                  with a sampled subset.
        output_strategy: How to handle output file (default: 'new_file'):
            - 'new_file': Create a new file with '_annotated' suffix (safest, recommended)
            - 'reports_only': Don't write to parquet, only generate reports
            - 'overwrite': Rewrite the original file (may fail if file is locked)
        provider: Log provider for configuration and rules (default: 'ebi').
            Providers define schema mappings, classification thresholds, and taxonomy.
            Available providers can be listed with list_available_providers().

    Returns:
        Dictionary with detection results and statistics
    """
    # Set active provider for configuration lookups
    from .config import set_active_provider
    set_active_provider(provider)
    logger.info("=" * 70)
    logger.info("Bot and Download Hub Annotator")
    logger.info("=" * 70)
    logger.info(f"Input: {input_parquet}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Contamination: {contamination}")
    logger.info(f"Classification method: {classification_method}")
    if sample_size:
        logger.info(f"Sample size: {sample_size:,} records")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up schema with optional min_location_downloads override
    if schema is None:
        schema = EBI_SCHEMA
    
    # Override min_location_downloads if provided
    if min_location_downloads is not None:
        schema = LogSchema(
            location_field=schema.location_field,
            country_field=schema.country_field,
            city_field=schema.city_field,
            user_field=schema.user_field,
            project_field=schema.project_field,
            timestamp_field=schema.timestamp_field,
            year_field=schema.year_field,
            min_location_downloads=min_location_downloads,
            min_year=schema.min_year,
            working_hours_start=schema.working_hours_start,
            working_hours_end=schema.working_hours_end,
            night_hours_start=schema.night_hours_start,
            night_hours_end=schema.night_hours_end,
        )
        logger.info(f"Minimum location downloads threshold: {min_location_downloads}")
    else:
        logger.info(f"Minimum location downloads threshold: {schema.min_location_downloads} (from schema)")
    
    conn = duckdb.connect()
    # Configure memory limits to prevent OOM issues
    conn.execute("SET memory_limit='4GB'")
    conn.execute("SET max_memory='4GB'")
    conn.execute("SET threads=2")  # Limit parallelism to reduce memory pressure
    conn.execute("SET threads=1")  # Single thread for stability
    conn.execute("SET preserve_insertion_order=false")
    
    # Apply DuckDB configuration from config.yaml
    duckdb_config = APP_CONFIG.get('duckdb', {})
    memory_limit = duckdb_config.get('memory_limit', '16GB')
    temp_directory_config = duckdb_config.get('temp_directory', './duckdb-tmp/')
    temp_directory_abs = os.path.abspath(temp_directory_config)
    os.makedirs(temp_directory_abs, exist_ok=True)
    
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    # Reduce temp directory size to prevent disk space issues
    conn.execute("PRAGMA max_temp_directory_size='5GiB'")
    conn.execute(f"PRAGMA temp_directory='{temp_directory_abs}'")
    # Disable temp file spilling if possible to reduce disk usage
    conn.execute("SET enable_object_cache=true")
    conn.execute("SET enable_progress_bar=false")
    
    # Handle sampling if requested
    sampled_file = None
    actual_input_parquet = input_parquet
    
    try:
        if sample_size:
            sampled_file = sample_parquet_records(conn, input_parquet, sample_size, schema)
            actual_input_parquet = sampled_file
        
        # Step 1: Extract features
        logger.info("\n" + "=" * 70)
        logger.info("Step 1: Extracting location features")
        logger.info("=" * 70)
        # Use EBI-specific extraction (default behavior)
        # For other providers, they should implement their own extraction functions
        # and pass them via custom_extractors, or use a different schema with custom extractors
        location_df = extract_location_features(
            conn, 
            actual_input_parquet,
            schema=schema if schema is not None else EBI_SCHEMA,
            custom_extractors=custom_extractors
        )
        
        # Filter valid rows - only check features that exist at this point
        # Advanced behavioral features are added later, so exclude them from validation here
        basic_features = [f for f in FEATURE_COLUMNS[:-1] if f in location_df.columns]  # Exclude the placeholder for NA check
        if len(basic_features) > 0:
            valid_mask = location_df[basic_features].notna().all(axis=1)
        else:
            valid_mask = pd.Series([True] * len(location_df), index=location_df.index)
        analysis_df = location_df[valid_mask].copy()
        analysis_df['time_series_features_present'] = analysis_df['time_series_features'].apply(lambda x: 1 if x is not None and len(x) > 0 else 0)

        logger.info(f"Analyzing {len(analysis_df):,} locations")
        
        # Extract advanced behavioral features for deep method
        if classification_method.lower() == 'deep':
            logger.info("\nExtracting advanced behavioral features (burst patterns, circadian rhythms, user coordination)...")
            from deeplogbot.features.providers.ebi import (
                extract_advanced_behavioral_features,
                add_bot_signature_features,
                add_bot_interaction_features
            )
            try:
                analysis_df = extract_advanced_behavioral_features(
                    analysis_df,
                    actual_input_parquet,
                    conn,
                    schema=schema if schema is not None else EBI_SCHEMA
                )
                logger.info("✓ Advanced behavioral features extracted successfully")
            except Exception as e:
                logger.warning(f"Advanced behavioral feature extraction failed: {e}")
                logger.info("Continuing with basic features only...")
            
            # Extract discriminative features (NEW - Two-Stage Classification)
            logger.info("\nExtracting discriminative features (malicious vs legitimate automation)...")
            from deeplogbot.features.providers.ebi import extract_discriminative_features
            try:
                analysis_df = extract_discriminative_features(
                    analysis_df,
                    actual_input_parquet,
                    conn,
                    schema=schema if schema is not None else EBI_SCHEMA
                )
                logger.info("✓ Discriminative features extracted successfully")
            except Exception as e:
                logger.warning(f"Discriminative feature extraction failed: {e}")
                logger.info("Continuing without discriminative features...")
        
        # Step 2: Train Isolation Forest
        logger.info("\n" + "=" * 70)
        logger.info("Step 2: Training Isolation Forest")
        logger.info("=" * 70)
        # Filter feature columns to only those that actually exist in the dataframe
        available_features = [f for f in FEATURE_COLUMNS if f != 'time_series_features_present' and f in analysis_df.columns]
        logger.info(f"Using {len(available_features)} features for Isolation Forest training")
        if len(available_features) < len(FEATURE_COLUMNS) - 1:
            missing = [f for f in FEATURE_COLUMNS if f != 'time_series_features_present' and f not in analysis_df.columns]
            logger.info(f"Note: {len(missing)} features not available: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        predictions, scores, _, _ = train_isolation_forest(
            analysis_df, available_features, contamination=contamination
        )
        
        analysis_df['is_anomaly'] = predictions == -1
        analysis_df['anomaly_score'] = -scores
        
        n_anomalies = analysis_df['is_anomaly'].sum()
        logger.info(f"Detected {n_anomalies:,} anomalous locations")
        
        # Add bot interaction and signature features now that anomaly_score is available
        if classification_method.lower() == 'deep':
            try:
                from deeplogbot.features.providers.ebi import (
                    add_bot_interaction_features,
                    add_bot_signature_features
                )
                # Add interaction features (needs anomaly_score which now exists)
                analysis_df = add_bot_interaction_features(analysis_df)
                logger.info("✓ Bot interaction features added successfully")
                
                # Add signature features (also needs anomaly_score and other features)
                analysis_df = add_bot_signature_features(analysis_df)
                logger.info("✓ Bot signature features added successfully")
            except Exception as e:
                logger.warning(f"Bot feature addition failed: {e}")
                logger.info("Continuing without additional features...")
        
        # Optional: compute feature importances for interpretability
        # Note: For 'deep' method, feature importance is computed within classify_locations_deep
        # So we only compute isolation forest importances for 'rules' method
        if compute_importances and classification_method.lower() != 'deep':
            imp_dir = os.path.join(output_dir, 'feature_importances')
            try:
                compute_feature_importances(
                    analysis_df,
                    [f for f in FEATURE_COLUMNS if f != 'time_series_features_present'],
                    analysis_df['is_anomaly'],
                    imp_dir
                )
            except Exception as e:
                logger.warning(f"Feature importance computation failed: {e}")
                logger.info("Continuing without isolation forest feature importances...")
        
        # Step 3: Classify locations
        logger.info("\n" + "=" * 70)
        logger.info("Step 3: Classifying locations")
        logger.info("=" * 70)
        
        cluster_df = None # Initialize cluster_df

        if classification_method.lower() == 'deep':
            logger.info("Using learned deep classification (Seed -> VAE -> IF -> Temporal -> Fusion)...")
            # Set up feature importance output directory if requested
            feature_importance_dir = None
            if compute_importances:
                feature_importance_dir = os.path.join(output_dir, 'feature_importances_deep')
                logger.info(f"Feature importance analysis will be saved to: {feature_importance_dir}")

            # Filter feature columns to only those that actually exist in the dataframe
            available_features_for_deep = [f for f in FEATURE_COLUMNS if f != 'time_series_features_present' and f in analysis_df.columns]
            logger.info(f"Using {len(available_features_for_deep)} features for deep classification")
            if len(available_features_for_deep) < len(FEATURE_COLUMNS) - 1:
                missing_deep = [f for f in FEATURE_COLUMNS if f != 'time_series_features_present' and f not in analysis_df.columns]
                logger.info(f"Note: {len(missing_deep)} features not available for deep classification: {missing_deep[:5]}{'...' if len(missing_deep) > 5 else ''}")

            # Learned pipeline: Seed → VAE → Deep IF → Temporal → Fusion
            analysis_df, cluster_df = classify_locations_deep(analysis_df,
                                                  available_features_for_deep,
                                                  compute_feature_importance=compute_importances,
                                                  feature_importance_output_dir=feature_importance_dir,
                                                  input_parquet=actual_input_parquet,
                                                  conn=conn)
        elif classification_method.lower() == 'rules':
            logger.info("Using rule-based classification with hierarchical taxonomy...")
            # Apply hierarchical classification
            analysis_df = classify_locations_hierarchical(analysis_df)
        else:
            raise ValueError(f"Unknown classification method: {classification_method}. Must be 'rules' or 'deep'")
        
        # Feature Usage Validation (NEW - High Priority Feature)
        if classification_method.lower() == 'deep':
            logger.info("\n" + "=" * 70)
            logger.info("Step 4: Feature Usage Validation")
            logger.info("=" * 70)
            try:
                from deeplogbot.models.classification.feature_validation import validate_feature_usage
                import json
                
                # Get predictions for validation
                if 'automation_category' in analysis_df.columns:
                    predictions = (analysis_df['automation_category'] == 'bot').astype(int).values
                elif 'bot_likelihood_score' in analysis_df.columns:
                    predictions = analysis_df['bot_likelihood_score'].values
                else:
                    predictions = (analysis_df['user_category'] == 'bot').astype(int).values
                
                # Get all feature columns
                all_feature_cols = [col for col in analysis_df.columns
                                   if col not in ['geo_location', 'country', 'city', 'user_category',
                                                 'behavior_type', 'automation_category', 'subcategory',
                                                 'classification_confidence', 'needs_review',
                                                 'time_series_features', 'is_anomaly', 'anomaly_score']]
                
                validation_results = validate_feature_usage(
                    analysis_df,
                    all_feature_cols,
                    predictions
                )
                
                # Save validation results
                validation_file = os.path.join(output_dir, 'feature_validation.json')
                with open(validation_file, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
                logger.info(f"\nValidation results saved to: {validation_file}")
                
                if not validation_results.get('passes_70_percent_threshold', False):
                    logger.warning("⚠️  WARNING: Behavioral features are below 70% threshold!")
                    logger.warning(f"   Current: {validation_results.get('behavioral_percentage', 0):.1f}%")
                    logger.warning("   Consider adjusting feature weights or extracting more behavioral features")
                else:
                    logger.info(f"✓ Validation PASSED: {validation_results.get('behavioral_percentage', 0):.1f}% behavioral features")
                    
            except Exception as e:
                logger.warning(f"Feature validation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Filter locations by hierarchical classification
        # Check if hierarchical columns exist (should exist for rules method, may exist for deep)
        if 'automation_category' in analysis_df.columns:
            bot_locs = analysis_df[analysis_df['automation_category'] == 'bot'].copy()
        elif 'is_bot' in analysis_df.columns:
            # Fallback to legacy column if hierarchical not available
            bot_locs = analysis_df[analysis_df['is_bot']].copy()
        else:
            bot_locs = pd.DataFrame()
        
        hub_subcategories = {'mirror', 'institutional_hub', 'data_aggregator'}
        if 'subcategory' in analysis_df.columns:
            hub_locs = analysis_df[analysis_df['subcategory'].isin(hub_subcategories)].copy()
        elif 'is_download_hub' in analysis_df.columns:
            # Fallback to legacy column if hierarchical not available
            hub_locs = analysis_df[analysis_df['is_download_hub']].copy()
        else:
            hub_locs = pd.DataFrame()
        
        # For deep architecture method, also show independent users and categories
        independent_locs = pd.DataFrame()  # Initialize empty
        other_locs = pd.DataFrame() # Initialize empty

        if classification_method.lower() == 'deep':
            if 'user_category' in analysis_df.columns:
                independent_locs = analysis_df[analysis_df['user_category'] == 'independent_user'].copy()
                other_locs = analysis_df[analysis_df['user_category'] == 'other'].copy()

            # Log detailed category counts
            category_counts = analysis_df['user_category'].value_counts()
            logger.info("\nLocation Categories:")
            for cat, count in category_counts.items():
                logger.info(f"  {cat}: {count:,} locations ({count/len(analysis_df)*100:.1f}%)")
        
        logger.info(f"\nBot locations: {len(bot_locs):,}")
        logger.info(f"Download hub locations: {len(hub_locs):,}")
        logger.info(f"Independent user locations: {len(independent_locs):,}")
        logger.info(f"Other/Unclassified locations: {len(other_locs):,}")
        
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
        
        # Show independent users if available
        if classification_method.lower() == 'deep' and len(independent_locs) > 0:
            logger.info("\nTop 10 Independent User Locations:")
            for _, row in independent_locs.sort_values('total_downloads', ascending=False).head(10).iterrows():
                city = str(row['city'])[:20] if pd.notna(row['city']) else 'N/A'
                logger.info(f"  {row['country']:<15} {city:<20} {int(row['unique_users']):>10,} users, "
                           f"{row['downloads_per_user']:.1f} DL/user, {int(row['total_downloads']):>6,} total DL")
        
        # Decide whether we should run annotation/reporting in this run.
        # IMPORTANT: When sampling is enabled, we must NOT annotate the original
        # parquet with the sampled subset, so we automatically disable annotation.
        # However, reports_only strategy can still run (it won't write parquet files).
        # Always run if reports_only is requested (even if annotate=False), as it only generates reports.
        effective_annotate = annotate and (sample_size is None)
        should_run_annotation = effective_annotate or output_strategy == 'reports_only'
        
        if annotate and sample_size is not None:
            if output_strategy == 'reports_only':
                logger.info(
                    "Sampling is enabled, but reports_only strategy will still generate reports "
                    "(no parquet files will be written)."
                )
            else:
                logger.info(
                    "Sampling is enabled (sample_size != None); skipping annotation to avoid "
                    "overwriting the full input parquet with a sampled subset. "
                    "To annotate the full dataset, rerun without --sample-size."
                )

        # Step 4: Annotate downloads or generate reports (optional)
        logger.info("\n" + "=" * 70)
        if output_strategy == 'reports_only':
            logger.info("Step 4: Generating reports (no parquet annotation)")
        else:
            logger.info("Step 4: Annotating downloads")
        logger.info("=" * 70)
        
        annotated_file = None
        if should_run_annotation:
            # Handle output strategy
            # For overwrite strategy, default to input file if no output specified
            # For new_file strategy, output_parquet can be None (auto-generate) or specified by user
            if output_strategy == 'overwrite' and output_parquet is None:
                output_parquet = input_parquet
            
            # Pass location_df for hierarchical columns (both rules and deep methods now support hierarchical)
            # Check if hierarchical columns exist
            has_hierarchical = all(col in analysis_df.columns for col in ['behavior_type', 'automation_category', 'subcategory'])
            location_df_for_annotation = analysis_df if has_hierarchical else None
            annotated_file = annotate_downloads(conn, actual_input_parquet, output_parquet,
                                                bot_locs, hub_locs, output_dir,
                                                output_strategy=output_strategy,
                                                location_df=location_df_for_annotation)
        else:
            logger.info("Annotation skipped (annotate=False and output_strategy != 'reports_only').")
        
        # Step 5: Calculate statistics
        logger.info("\n" + "=" * 70)
        logger.info("Step 5: Calculating statistics")
        logger.info("=" * 70)

        # Prefer statistics derived from the algorithm output (analysis_df)
        # This ensures bot/hub downloads are computed directly from the
        # classified locations and their total_downloads, rather than
        # recomputing from the raw parquet.
        if 'total_downloads' in analysis_df.columns:
            total_downloads = int(analysis_df['total_downloads'].sum())
            
            # Use hierarchical columns if available, fallback to legacy
            if 'automation_category' in analysis_df.columns:
                bot_mask = analysis_df['automation_category'] == 'bot'
            elif 'is_bot' in analysis_df.columns:
                bot_mask = analysis_df['is_bot']
            else:
                bot_mask = pd.Series(False, index=analysis_df.index)
            
            hub_subcategories = {'mirror', 'institutional_hub', 'data_aggregator'}
            if 'subcategory' in analysis_df.columns:
                hub_mask = analysis_df['subcategory'].isin(hub_subcategories)
            elif 'is_download_hub' in analysis_df.columns:
                hub_mask = analysis_df['is_download_hub']
            else:
                hub_mask = pd.Series(False, index=analysis_df.index)
            
            bot_downloads = int(analysis_df.loc[bot_mask, 'total_downloads'].sum())
            hub_downloads = int(analysis_df.loc[hub_mask, 'total_downloads'].sum())

            stats: dict = {
                'total': total_downloads,
                'bots': bot_downloads,
                'hubs': hub_downloads,
            }

            if classification_method.lower() == 'deep' and 'user_category' in analysis_df.columns:
                # Deep method: explicit categories
                stats['independent_users'] = int(
                    analysis_df.loc[analysis_df['user_category'] == 'independent_user', 'total_downloads'].sum()
                )
                stats['other_downloads'] = int(
                    analysis_df.loc[analysis_df['user_category'] == 'other', 'total_downloads'].sum()
                )
                stats['normal'] = int(
                    analysis_df.loc[analysis_df['user_category'] == 'normal', 'total_downloads'].sum()
                )
            else:
                # Rules / ML: normal = everything that is not bot or hub
                normal_downloads = total_downloads - bot_downloads - hub_downloads
                stats['normal'] = normal_downloads
        else:
            # Fallback: use annotated parquet if total_downloads is not available
            escaped_output = os.path.abspath(output_parquet).replace("'", "''")
            stats_query = f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN bot THEN 1 ELSE 0 END) as bots,
                    SUM(CASE WHEN download_hub THEN 1 ELSE 0 END) as hubs,
                    SUM(CASE WHEN NOT bot AND NOT download_hub THEN 1 ELSE 0 END) as normal
                FROM read_parquet('{escaped_output}')
            """
            stats_result = conn.execute(stats_query).df().iloc[0]
            stats = {
                'total': int(stats_result['total']),
                'bots': int(stats_result['bots']),
                'hubs': int(stats_result['hubs']),
                'normal': int(stats_result['normal']),
            }

        logger.info(f"\nTotal downloads: {format_number(stats['total'])}")
        logger.info(f"Bot downloads: {format_number(stats['bots'])} ({stats['bots']/stats['total']*100:.2f}%)")
        logger.info(f"Hub downloads: {format_number(stats['hubs'])} ({stats['hubs']/stats['total']*100:.2f}%)")
        if 'independent_users' in stats:
            logger.info(
                f"Independent user downloads: {format_number(stats['independent_users'])} "
                f"({stats['independent_users']/stats['total']*100:.2f}%)"
            )
        if 'other_downloads' in stats:
            logger.info(
                f"Other/Unclassified downloads: {format_number(stats['other_downloads'])} "
                f"({stats['other_downloads']/stats['total']*100:.2f}%)"
            )
        logger.info(f"Normal downloads: {format_number(stats['normal'])} ({stats['normal']/stats['total']*100:.2f}%)")

        # Log hierarchical classification statistics (only for deep method)
        if classification_method.lower() == 'deep' and 'behavior_type' in analysis_df.columns:
            logger.info("\nHierarchical Classification Statistics (Deep Method):")
            total = len(analysis_df)
            organic_count = (analysis_df['behavior_type'] == 'organic').sum()
            automated_count = (analysis_df['behavior_type'] == 'automated').sum()
            logger.info(f"  ORGANIC locations: {organic_count:,} ({organic_count/total*100:.1f}%)")
            logger.info(f"  AUTOMATED locations: {automated_count:,} ({automated_count/total*100:.1f}%)")

            if 'automation_category' in analysis_df.columns and automated_count > 0:
                bot_count = (analysis_df['automation_category'] == 'bot').sum()
                legit_count = (analysis_df['automation_category'] == 'legitimate_automation').sum()
                logger.info(f"    - BOT: {bot_count:,} ({bot_count/automated_count*100:.1f}% of automated)")
                logger.info(f"    - LEGITIMATE_AUTOMATION: {legit_count:,} ({legit_count/automated_count*100:.1f}% of automated)")

        # Step 6: Save analysis and generate report
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: Generating reports")
        logger.info("=" * 70)
        
        # Save full analysis
        analysis_file = os.path.join(output_dir, 'location_analysis.csv')
        analysis_df.to_csv(analysis_file, index=False)
        logger.info(f"Location analysis saved to: {analysis_file}")
        
        # Generate report
        generate_report(analysis_df, bot_locs, hub_locs, independent_locs, other_locs, stats, output_dir, 
                        cluster_df=cluster_df, # Pass cluster_df here
                        schema=schema if schema is not None else EBI_SCHEMA,
                       available_features=FEATURE_COLUMNS,
                       classification_method=classification_method)
        
        logger.info("\n" + "=" * 70)
        logger.info("Bot Annotation Complete!")
        logger.info("=" * 70)
        logger.info("\nOutput files:")
        if annotated_file:
            logger.info(f"  - {annotated_file} (annotated with 'bot' and 'download_hub' columns)")
        elif effective_annotate and output_strategy == 'reports_only':
            logger.info("  - No parquet file written (reports_only strategy)")
        logger.info(f"  - {output_dir}/bot_detection_report.txt")
        logger.info(f"  - {output_dir}/location_analysis.csv")
        logger.info(f"  - {output_dir}/report.html (interactive HTML report)")
        logger.info(f"  - {output_dir}/plots/ (visualization charts)")
        
        return {
            'bot_locations': len(bot_locs),
            'hub_locations': len(hub_locs),
            'stats': stats,
            'output_parquet': annotated_file if annotated_file else None
        }
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
    finally:
        # Clean up temporary sampled file if created
        if sampled_file and sampled_file != input_parquet and os.path.exists(sampled_file):
            logger.info(f"Cleaning up temporary sampled file: {sampled_file}")
            os.remove(sampled_file)
        conn.close()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Annotate downloads with bot and download_hub flags using ML detection'
    )
    logger.debug("Starting main CLI function...")
    parser.add_argument('--input', '-i', 
                       default='original_data/data_downloads_parquet.parquet',
                       help='Input parquet file')
    parser.add_argument('--output', '-out',
                       default=None,
                       help='Output parquet file. Behavior depends on --output-strategy: '
                            'new_file: creates this file (or auto-generates with _annotated suffix if not specified), '
                            'reports_only: ignored, '
                            'overwrite: overwrites this file (or input file if not specified)')
    parser.add_argument('--output-dir', '-o',
                       default='output/bot_analysis',
                       help='Output directory for reports')
    parser.add_argument('--contamination', '-c', type=float, default=0.15,
                       help='Expected proportion of anomalies (default: 0.15)')
    parser.add_argument('--compute-importances', action='store_true',
                       help='Compute feature importances (optional, slower)')
    parser.add_argument('--sample-size', '-s', type=int, default=None,
                       help='Randomly sample N records from all years before processing (e.g., 1000000 for 1M records)')
    parser.add_argument('--classification-method', '-m', type=str, default='rules',
                       choices=['rules', 'deep'],
                       help='Classification method: "rules" for rule-based (default), "deep" for deep architecture')
    parser.add_argument('--min-location-downloads', type=int, default=None,
                       help='Minimum downloads required for a location to be included (default: 1, set higher to filter noise)')
    parser.add_argument('--output-strategy', type=str, default='new_file',
                       choices=['new_file', 'reports_only', 'overwrite'],
                       help='Output file strategy: "new_file" creates annotated file with _annotated suffix (default), "reports_only" only generates reports, "overwrite" rewrites original file')
    parser.add_argument('--reports-only', action='store_true',
                       help='Shortcut flag: Only generate reports, skip parquet annotation (equivalent to --output-strategy reports_only)')
    parser.add_argument('--provider', '-p', type=str, default='ebi',
                       help='Log provider for configuration and rules (default: ebi). '
                            'Use --list-providers to see available providers.')
    parser.add_argument('--list-providers', action='store_true',
                       help='List available providers and exit')

    args = parser.parse_args()

    # Handle --list-providers
    if args.list_providers:
        from .config import list_available_providers
        providers = list_available_providers()
        print("Available providers:")
        for p in providers:
            print(f"  - {p}")
        return
    
    # If --reports-only is set, override output_strategy
    if args.reports_only:
        args.output_strategy = 'reports_only'

    # Set the active provider for configuration
    from .config import set_active_provider
    set_active_provider(args.provider)
    logger.info(f"Using provider: {args.provider}")

    try:
        run_bot_annotator(
            args.input,
            args.output,
            args.output_dir,
            args.contamination,
            compute_importances=args.compute_importances,
            sample_size=args.sample_size,
            classification_method=args.classification_method,
            min_location_downloads=args.min_location_downloads,
            output_strategy=args.output_strategy,
            provider=args.provider,
        )
        
        logger.info("\nDone!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

