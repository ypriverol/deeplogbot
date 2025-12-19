"""Annotation utilities for marking bot and download hub locations."""

import os
import pandas as pd

from ..utils import logger


def annotate_downloads(conn, input_parquet, output_parquet, 
                       bot_locations, hub_locations, output_dir):
    """
    Annotate the parquet file with bot and download_hub columns.
    """
    logger.info("Annotating downloads...")
    
    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save bot locations
    bot_file = os.path.join(temp_dir, 'bot_locations.parquet')
    bot_locations[['geo_location']].to_parquet(bot_file, index=False)
    
    # Save hub locations
    hub_file = os.path.join(temp_dir, 'hub_locations.parquet')
    hub_locations[['geo_location']].to_parquet(hub_file, index=False)
    
    escaped_input = os.path.abspath(input_parquet).replace("'", "''")
    escaped_output = os.path.abspath(output_parquet).replace("'", "''")
    escaped_bots = os.path.abspath(bot_file).replace("'", "''")
    escaped_hubs = os.path.abspath(hub_file).replace("'", "''")
    
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
    
    logger.info(f"Writing annotated parquet to: {output_parquet}")
    conn.execute(write_query)
    logger.info("Annotation complete!")
    
    return escaped_output

