"""Annotation utilities for marking bot and download hub locations.

This module provides functions to annotate download logs with classification
information using the hierarchical taxonomy:
  - behavior_type: 'organic' | 'automated'
  - automation_category: 'bot' | 'legitimate_automation' | None
  - subcategory: detailed classification
"""

import os
from typing import Literal, Optional

import pandas as pd

from ..utils import logger


def annotate_downloads(conn, input_parquet, output_parquet,
                       bot_locations, hub_locations, output_dir,
                       output_strategy: Literal['new_file', 'reports_only', 'overwrite'] = 'new_file',
                       location_df: Optional[pd.DataFrame] = None):
    """
    Annotate the parquet file with classification columns.

    Adds hierarchical classification columns:
      - behavior_type: 'organic' | 'automated'
      - automation_category: 'bot' | 'legitimate_automation' | None
      - subcategory: detailed classification

    Args:
        conn: DuckDB connection
        input_parquet: Path to input parquet file
        output_parquet: Path to output parquet file (may be modified based on strategy)
        bot_locations: DataFrame with bot locations
        hub_locations: DataFrame with hub locations
        output_dir: Directory for output files
        output_strategy: How to handle output file:
            - 'new_file': Create a new file with '_annotated' suffix (default)
            - 'reports_only': Don't write to parquet, only generate reports
            - 'overwrite': Rewrite the original file (may fail if file is locked)
        location_df: Optional DataFrame with full hierarchical classification.
            If provided, uses hierarchical columns. Otherwise, falls back to
            bot_locations/hub_locations only.

    Returns:
        Path to output file (None if reports_only)
    """
    logger.info("Annotating downloads...")
    logger.info(f"  Output strategy: {output_strategy}")

    # Handle reports_only strategy
    if output_strategy == 'reports_only':
        logger.info("  Skipping parquet annotation (reports_only strategy)")
        logger.info("  Reports will be generated in output directory")
        return None

    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    escaped_input = os.path.abspath(input_parquet).replace("'", "''")

    # Determine output file based on strategy
    if output_strategy == 'new_file':
        if output_parquet is None:
            input_basename = os.path.basename(input_parquet)
            input_name, input_ext = os.path.splitext(input_basename)
            new_filename = f"{input_name}_annotated{input_ext}"
            output_parquet = os.path.join(output_dir, new_filename)
            logger.info(f"  Creating new annotated file (auto-generated): {output_parquet}")
        else:
            output_parquet_dir = os.path.dirname(output_parquet)
            if output_parquet_dir and not os.path.exists(output_parquet_dir):
                os.makedirs(output_parquet_dir, exist_ok=True)
            logger.info(f"  Creating new annotated file (user-specified): {output_parquet}")
    elif output_strategy == 'overwrite':
        if output_parquet is None:
            output_parquet = input_parquet
        logger.info(f"  Overwriting original file: {output_parquet}")
        logger.warning("  Note: This may fail if the file is locked by another process")

    # Ensure output directory exists
    output_parquet_dir = os.path.dirname(os.path.abspath(output_parquet))
    if output_parquet_dir and not os.path.exists(output_parquet_dir):
        os.makedirs(output_parquet_dir, exist_ok=True)
        logger.info(f"  Created output directory: {output_parquet_dir}")

    escaped_output = os.path.abspath(output_parquet).replace("'", "''")

    # Check if we have hierarchical classification data
    has_hierarchical = (
        location_df is not None and
        'behavior_type' in location_df.columns and
        'automation_category' in location_df.columns and
        'subcategory' in location_df.columns
    )

    if has_hierarchical:
        logger.info("  Using hierarchical classification (behavior_type, automation_category, subcategory)")
        annotation_query = _build_hierarchical_annotation_query(
            escaped_input, location_df, temp_dir
        )
    else:
        logger.info("  Using legacy annotation (bot, download_hub only)")
        annotation_query = _build_legacy_annotation_query(
            escaped_input, bot_locations, hub_locations, temp_dir
        )

    write_query = f"""
    COPY (
        {annotation_query}
    ) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION 'snappy')
    """

    try:
        logger.info(f"Writing annotated parquet to: {output_parquet}")
        conn.execute(write_query)
        logger.info("Annotation complete!")
        return escaped_output
    except Exception as e:
        if "lock" in str(e).lower() or "concurrency" in str(e).lower():
            logger.error(f"File locking error: {e}")
            logger.error("The file may be open in another process or DuckDB connection.")
            logger.error("Consider using output_strategy='new_file' or 'reports_only'")
            raise
        else:
            raise


def _build_legacy_annotation_query(escaped_input: str, bot_locations: pd.DataFrame,
                                    hub_locations: pd.DataFrame, temp_dir: str) -> str:
    """Build annotation query using bot/hub locations only (when full DataFrame not available)."""
    # Save bot locations
    bot_file = os.path.join(temp_dir, 'bot_locations.parquet')
    bot_locations[['geo_location']].to_parquet(bot_file, index=False)

    # Save hub locations
    hub_file = os.path.join(temp_dir, 'hub_locations.parquet')
    hub_locations[['geo_location']].to_parquet(hub_file, index=False)

    escaped_bots = os.path.abspath(bot_file).replace("'", "''")
    escaped_hubs = os.path.abspath(hub_file).replace("'", "''")

    return f"""
    WITH bot_locs AS (
        SELECT DISTINCT geo_location FROM read_parquet('{escaped_bots}')
    ),
    hub_locs AS (
        SELECT DISTINCT geo_location FROM read_parquet('{escaped_hubs}')
    )
    SELECT
        d.*,
        CASE WHEN bl.geo_location IS NOT NULL OR hl.geo_location IS NOT NULL THEN 'automated' ELSE 'organic' END as behavior_type,
        CASE
            WHEN bl.geo_location IS NOT NULL THEN 'bot'
            WHEN hl.geo_location IS NOT NULL THEN 'legitimate_automation'
            ELSE NULL
        END as automation_category,
        CASE
            WHEN bl.geo_location IS NOT NULL THEN 'generic_bot'
            WHEN hl.geo_location IS NOT NULL THEN 'mirror'
            ELSE 'individual_user'
        END as subcategory
    FROM read_parquet('{escaped_input}') d
    LEFT JOIN bot_locs bl ON d.geo_location = bl.geo_location
    LEFT JOIN hub_locs hl ON d.geo_location = hl.geo_location
    """


def _build_hierarchical_annotation_query(escaped_input: str, location_df: pd.DataFrame,
                                          temp_dir: str) -> str:
    """Build annotation query using full hierarchical classification."""
    # Select only the columns we need for annotation
    annotation_cols = ['geo_location', 'behavior_type', 'automation_category', 'subcategory']
    available_cols = [col for col in annotation_cols if col in location_df.columns]

    # Create classification DataFrame
    classification_df = location_df[available_cols].copy()

    # Save to temporary parquet
    classification_file = os.path.join(temp_dir, 'location_classification.parquet')
    classification_df.to_parquet(classification_file, index=False)
    escaped_classification = os.path.abspath(classification_file).replace("'", "''")

    return f"""
    WITH classification AS (
        SELECT
            geo_location,
            behavior_type,
            automation_category,
            subcategory
        FROM read_parquet('{escaped_classification}')
    )
    SELECT
        d.*,
        COALESCE(c.behavior_type, 'organic') as behavior_type,
        c.automation_category as automation_category,
        COALESCE(c.subcategory, 'individual_user') as subcategory
    FROM read_parquet('{escaped_input}') d
    LEFT JOIN classification c ON d.geo_location = c.geo_location
    """

