"""Geographic utility functions."""

from math import radians, cos, sin, asin, sqrt
from typing import Tuple, Optional, Dict
import pandas as pd


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (in km).
    
    Uses the Haversine formula to compute the distance between two points
    on the surface of a sphere given their latitudes and longitudes.
    
    Args:
        lat1: Latitude of first point (in degrees)
        lon1: Longitude of first point (in degrees)
        lat2: Latitude of second point (in degrees)
        lon2: Longitude of second point (in degrees)
        
    Returns:
        Distance between the two points in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def parse_geo_location(geo_loc_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse 'lat,lon' string to (lat, lon) tuple.
    
    Args:
        geo_loc_str: String in format "latitude,longitude" (e.g., "51.1234,-0.5678")
        
    Returns:
        Tuple of (latitude, longitude) as floats, or (None, None) if parsing fails
    """
    try:
        parts = geo_loc_str.split(',')
        return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, IndexError, AttributeError):
        return None, None


def group_nearby_locations_with_llm(hub_locations: pd.DataFrame, max_distance_km: int = 10, use_llm: bool = True) -> Dict[str, str]:
    """
    Group nearby hub locations using geographic distance and optionally LLM.
    
    Returns a mapping: original_geo_location -> group_id (canonical location)
    
    Args:
        hub_locations: DataFrame with hub locations including 'geo_location', 'country', 'city', etc.
        max_distance_km: Maximum distance in km for grouping locations
        use_llm: Whether to use LLM for canonical naming
        
    Returns:
        Dictionary mapping geo_location to canonical location
    """
    from ..utils import logger
    from ..llm import get_llm_canonical_name
    
    logger.info(f"Grouping {len(hub_locations)} hub locations (max_distance={max_distance_km}km)...")
    logger.info("  Step 1: Geographic distance-based grouping...")
    
    # Parse coordinates
    locations_with_coords = []
    for _, row in hub_locations.iterrows():
        lat, lon = parse_geo_location(row['geo_location'])
        if lat is not None and lon is not None:
            locations_with_coords.append({
                'geo_location': row['geo_location'],
                'country': row['country'],
                'city': row.get('city', '') if pd.notna(row.get('city')) else '',
                'lat': lat,
                'lon': lon,
                'unique_users': row['unique_users'],
                'total_downloads': row['total_downloads'],
                'downloads_per_user': row['downloads_per_user']
            })
    
    if len(locations_with_coords) == 0:
        return {loc['geo_location']: loc['geo_location'] for loc in locations_with_coords}
    
    # Calculate distance matrix and group nearby locations
    groups = {}
    group_id = 0
    processed = set()
    
    for i, loc1 in enumerate(locations_with_coords):
        if loc1['geo_location'] in processed:
            continue
        
        # Start a new group
        group_members = [loc1]
        processed.add(loc1['geo_location'])
        
        # Find nearby locations
        for _, loc2 in enumerate(locations_with_coords[i+1:], start=i+1):
            if loc2['geo_location'] in processed:
                continue
            
            distance = haversine_distance(
                loc1['lat'], loc1['lon'],
                loc2['lat'], loc2['lon']
            )
            
            # Same country and within distance threshold
            if (loc1['country'] == loc2['country'] and 
                distance <= max_distance_km):
                group_members.append(loc2)
                processed.add(loc2['geo_location'])
        
        # Assign group ID
        group_geo_locs = [loc['geo_location'] for loc in group_members]
        canonical_location = group_members[0]['geo_location']  # Use first as default
        
        # If using LLM and group has multiple members, get canonical name
        if use_llm and len(group_members) > 1:
            logger.info(f"    Calling LLM for group {group_id + 1} ({len(group_members)} locations)...")
            canonical_location = get_llm_canonical_name(group_members)
        
        for geo_loc in group_geo_locs:
            groups[geo_loc] = canonical_location
        
        group_id += 1
    
    # Assign single locations to themselves
    for loc in locations_with_coords:
        if loc['geo_location'] not in groups:
            groups[loc['geo_location']] = loc['geo_location']
    
    n_groups = len(set(groups.values()))
    logger.info(f"Grouped into {n_groups} consolidated locations (from {len(locations_with_coords)} original)")
    
    return groups
