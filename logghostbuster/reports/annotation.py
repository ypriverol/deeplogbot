"""Annotation utilities for marking bot and download hub locations."""

import os
from typing import Literal

# pandas is used implicitly via DataFrame.to_parquet() method
# from bot_locations and hub_locations DataFrames

from ..utils import logger


def annotate_downloads(conn, input_parquet, output_parquet, 
                       bot_locations, hub_locations, output_dir,
                       output_strategy: Literal['new_file', 'reports_only', 'overwrite'] = 'new_file'):
    """
    Annotate the parquet file with bot and download_hub columns.
    
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
    
    # Save bot locations
    bot_file = os.path.join(temp_dir, 'bot_locations.parquet')
    bot_locations[['geo_location']].to_parquet(bot_file, index=False)
    
    # Save hub locations
    hub_file = os.path.join(temp_dir, 'hub_locations.parquet')
    hub_locations[['geo_location']].to_parquet(hub_file, index=False)
    
    escaped_input = os.path.abspath(input_parquet).replace("'", "''")
    escaped_bots = os.path.abspath(bot_file).replace("'", "''")
    escaped_hubs = os.path.abspath(hub_file).replace("'", "''")
    
    # Determine output file based on strategy
    if output_strategy == 'new_file':
        # Create new file with _annotated suffix, or use provided output_parquet if specified
        if output_parquet is None:
            # Auto-generate filename with _annotated suffix
            input_dir = os.path.dirname(input_parquet)
            input_basename = os.path.basename(input_parquet)
            input_name, input_ext = os.path.splitext(input_basename)
            new_filename = f"{input_name}_annotated{input_ext}"
            output_parquet = os.path.join(input_dir, new_filename)
            logger.info(f"  Creating new annotated file (auto-generated): {output_parquet}")
        else:
            # Use provided output filename
            logger.info(f"  Creating new annotated file (user-specified): {output_parquet}")
    elif output_strategy == 'overwrite':
        # Use the provided output_parquet (or input_parquet if None)
        if output_parquet is None:
            output_parquet = input_parquet
        logger.info(f"  Overwriting original file: {output_parquet}")
        logger.warning("  Note: This may fail if the file is locked by another process")
    
    escaped_output = os.path.abspath(output_parquet).replace("'", "''")
    
    # Build annotation query
    annotation_query = f"""
    WITH bot_locs AS (
        SELECT DISTINCT geo_location FROM read_parquet('{escaped_bots}')
    ),
    hub_locs AS (
        SELECT DISTINCT geo_location FROM read_parquet('{escaped_hubs}')
    )
    SELECT 
        d.*,
        CASE WHEN bl.geo_location IS NOT NULL THEN TRUE ELSE FALSE END as bot,
        CASE WHEN hl.geo_location IS NOT NULL THEN TRUE ELSE FALSE END as download_hub
    FROM read_parquet('{escaped_input}') d
    LEFT JOIN bot_locs bl ON d.geo_location = bl.geo_location
    LEFT JOIN hub_locs hl ON d.geo_location = hl.geo_location
    """
    
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

