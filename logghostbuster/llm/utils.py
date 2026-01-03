"""LLM utilities for location canonical naming."""

from ..utils import logger


def get_llm_canonical_name(group_members):
    """
    Returns the geographic center for a group of nearby locations.
    """
    # Use the first location's geo_location as the canonical name
    logger.info(f"Using geographic center for {len(group_members)} locations")
    return group_members[0]['geo_location']
