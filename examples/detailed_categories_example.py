#!/usr/bin/env python3
"""
Example demonstrating the detailed classification categories feature.

This script shows how the new categories (ci_cd_pipeline, research_group,
bulk_downloader, course_workshop) work with sample data.
"""

import pandas as pd

# Sample data representing different download patterns
sample_data = {
    'geo_location': [
        'CI/CD-Server-US',
        'University-Lab-UK', 
        'Individual-Power-User-DE',
        'Workshop-Event-CA',
        'Known-Bot-Farm-CN',
        'Download-Hub-Mirror-US',
        'Normal-User-FR',
    ],
    'unique_users': [
        8,      # CI/CD: Few users
        25,     # Research: Small group
        3,      # Bulk: Very few users
        150,    # Course: Many users
        50000,  # Bot: Huge number of users
        10,     # Hub: Few users
        50,     # Normal: Moderate users
    ],
    'downloads_per_user': [
        200,    # CI/CD: High per user
        45,     # Research: Moderate
        750,    # Bulk: Very high
        12,     # Course: Low per user
        8,      # Bot: Low per user
        1200,   # Hub: Extreme per user
        15,     # Normal: Low-moderate
    ],
    'file_diversity_ratio': [
        0.15,   # CI/CD: Low diversity (same files)
        0.65,   # Research: High diversity
        0.45,   # Bulk: Moderate diversity
        0.25,   # Course: Low diversity (tutorial files)
        0.80,   # Bot: High diversity (exploring)
        0.30,   # Hub: Low-moderate diversity
        0.50,   # Normal: Moderate diversity
    ],
    'regularity_score': [
        0.95,   # CI/CD: Very regular (automated)
        0.45,   # Research: Irregular (human)
        0.30,   # Bulk: Irregular bursts
        0.50,   # Course: Semi-regular
        0.70,   # Bot: Somewhat regular
        0.85,   # Hub: Regular syncing
        0.40,   # Normal: Irregular
    ],
    'working_hours_ratio': [
        0.25,   # CI/CD: 24/7 operation
        0.75,   # Research: Working hours
        0.35,   # Bulk: Various times
        0.60,   # Course: Event hours
        0.15,   # Bot: 24/7
        0.30,   # Hub: Automated sync
        0.65,   # Normal: Working hours
    ],
    'user_category': [
        'normal',           # Will be reclassified
        'normal',           # Will be reclassified
        'normal',           # Will be reclassified  
        'normal',           # Will be reclassified
        'bot',              # Protected
        'download_hub',     # Protected
        'normal',           # Stays normal
    ],
}

def simulate_detailed_classification(df, rules):
    """
    Simulate the detailed classification logic.
    This is a simplified version of _classify_detailed_categories().
    """
    # Initialize detailed_category from user_category
    df['detailed_category'] = df['user_category']
    
    # Only reclassify normal, other, or unclassified
    reclassify_mask = df['detailed_category'].isin(['normal', 'other', 'unclassified'])
    
    print("=" * 80)
    print("DETAILED CATEGORY CLASSIFICATION")
    print("=" * 80)
    print()
    
    # 1. CI/CD Pipeline Detection
    ci_cd_rules = rules['ci_cd_pipeline']
    ci_cd_mask = reclassify_mask & (
        (df['unique_users'] <= ci_cd_rules['max_users']) &
        (df['downloads_per_user'] >= ci_cd_rules['min_downloads_per_user']) &
        (df['downloads_per_user'] <= ci_cd_rules['max_downloads_per_user']) &
        (df['file_diversity_ratio'] <= ci_cd_rules['max_file_diversity_ratio']) &
        (df['regularity_score'] >= ci_cd_rules['min_regularity_score'])
    )
    df.loc[ci_cd_mask, 'detailed_category'] = 'ci_cd_pipeline'
    print(f"CI/CD Pipeline: {ci_cd_mask.sum()} location(s)")
    if ci_cd_mask.any():
        print(f"  → {', '.join(df[ci_cd_mask]['geo_location'].tolist())}")
    
    # 2. Research Group Detection
    research_rules = rules['research_group']
    research_mask = reclassify_mask & ~ci_cd_mask & (
        (df['unique_users'] >= research_rules['min_users']) &
        (df['unique_users'] <= research_rules['max_users']) &
        (df['downloads_per_user'] >= research_rules['min_downloads_per_user']) &
        (df['downloads_per_user'] <= research_rules['max_downloads_per_user']) &
        (df['working_hours_ratio'] >= research_rules['min_working_hours_ratio']) &
        (df['file_diversity_ratio'] >= research_rules['min_file_diversity_ratio'])
    )
    df.loc[research_mask, 'detailed_category'] = 'research_group'
    print(f"Research Group: {research_mask.sum()} location(s)")
    if research_mask.any():
        print(f"  → {', '.join(df[research_mask]['geo_location'].tolist())}")
    
    # 3. Bulk Downloader Detection
    bulk_rules = rules['bulk_downloader']
    bulk_mask = reclassify_mask & ~ci_cd_mask & ~research_mask & (
        (df['unique_users'] <= bulk_rules['max_users']) &
        (df['downloads_per_user'] >= bulk_rules['min_downloads_per_user']) &
        (df['downloads_per_user'] <= bulk_rules['max_downloads_per_user'])
    )
    df.loc[bulk_mask, 'detailed_category'] = 'bulk_downloader'
    print(f"Bulk Downloader: {bulk_mask.sum()} location(s)")
    if bulk_mask.any():
        print(f"  → {', '.join(df[bulk_mask]['geo_location'].tolist())}")
    
    # 4. Course/Workshop Detection
    course_rules = rules['course_workshop']
    course_mask = reclassify_mask & ~ci_cd_mask & ~research_mask & ~bulk_mask & (
        (df['unique_users'] >= course_rules['min_users']) &
        (df['unique_users'] <= course_rules['max_users']) &
        (df['downloads_per_user'] >= course_rules['min_downloads_per_user']) &
        (df['downloads_per_user'] <= course_rules['max_downloads_per_user']) &
        (df['file_diversity_ratio'] <= course_rules['max_file_diversity_ratio'])
    )
    df.loc[course_mask, 'detailed_category'] = 'course_workshop'
    print(f"Course/Workshop: {course_mask.sum()} location(s)")
    if course_mask.any():
        print(f"  → {', '.join(df[course_mask]['geo_location'].tolist())}")
    
    print()
    return df

def main():
    """Main function to demonstrate the feature."""
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION CATEGORIES - EXAMPLE")
    print("=" * 80)
    print()
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Show input data
    print("Input Data:")
    print("-" * 80)
    print(df[['geo_location', 'unique_users', 'downloads_per_user', 'user_category']].to_string(index=False))
    print()
    
    # Load rules (hardcoded for example)
    rules = {
        'ci_cd_pipeline': {
            'max_users': 10,
            'min_downloads_per_user': 50,
            'max_downloads_per_user': 500,
            'max_file_diversity_ratio': 0.3,
            'min_regularity_score': 0.8,
        },
        'research_group': {
            'min_users': 5,
            'max_users': 50,
            'min_downloads_per_user': 10,
            'max_downloads_per_user': 100,
            'min_working_hours_ratio': 0.5,
            'min_file_diversity_ratio': 0.3,
        },
        'bulk_downloader': {
            'max_users': 5,
            'min_downloads_per_user': 100,
            'max_downloads_per_user': 1000,
        },
        'course_workshop': {
            'min_users': 50,
            'max_users': 500,
            'min_downloads_per_user': 5,
            'max_downloads_per_user': 20,
            'max_file_diversity_ratio': 0.3,
        },
    }
    
    # Run classification
    df = simulate_detailed_classification(df, rules)
    
    # Show results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(df[['geo_location', 'user_category', 'detailed_category']].to_string(index=False))
    print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Protected Categories (not reclassified):")
    protected = df[df['user_category'].isin(['bot', 'download_hub', 'independent_user'])]
    for _, row in protected.iterrows():
        print(f"  ✓ {row['geo_location']:<30} → {row['user_category']}")
    
    print()
    print("Reclassified Categories:")
    reclassified = df[~df['user_category'].isin(['bot', 'download_hub', 'independent_user']) & 
                     (df['detailed_category'] != df['user_category'])]
    for _, row in reclassified.iterrows():
        print(f"  ✓ {row['geo_location']:<30} → {row['detailed_category']}")
    
    print()
    print("Unchanged:")
    unchanged = df[(df['user_category'] == df['detailed_category']) & 
                  ~df['user_category'].isin(['bot', 'download_hub', 'independent_user'])]
    for _, row in unchanged.iterrows():
        print(f"  ✓ {row['geo_location']:<30} → {row['detailed_category']}")
    
    print()
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print()

if __name__ == '__main__':
    main()
