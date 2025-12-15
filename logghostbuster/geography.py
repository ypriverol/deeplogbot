"""Geographic utilities for location grouping."""

import pandas as pd
from math import radians, cos, sin, asin, sqrt

from .utils import logger
from .llm_utils import get_llm_canonical_name


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth (in km).
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


def parse_geo_location(geo_loc_str):
    """Parse 'lat,lon' string to (lat, lon) tuple."""
    try:
        parts = geo_loc_str.split(',')
        return float(parts[0].strip()), float(parts[1].strip())
    except:
        return None, None


def group_nearby_locations_with_llm(hub_locations, max_distance_km=10, use_llm=True):
    """
    Group nearby hub locations using geographic distance and optionally LLM.
    
    Returns a mapping: original_geo_location -> group_id (canonical location)
    """
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

