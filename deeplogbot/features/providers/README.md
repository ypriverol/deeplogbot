# EBI Provider - Feature Extraction

This module contains EBI-specific feature extractors and extraction logic for bot and download hub detection.

## Overview

The EBI provider extracts behavioral features from log data to identify:
- **Bots**: Locations with many users, low downloads per user, high hourly user density, and irregular time patterns
- **Download Hubs**: Locations with few users, high downloads per user, systematic patterns, and regular time patterns
- **Normal Users**: Locations with typical user behavior patterns

## Feature Categories

Features are extracted at the **location level**, where a location is defined as a unique combination of:
- `geo_location`: Geographic coordinates (latitude, longitude)
- `country`: Country name

By default, all locations are included in the analysis (minimum 1 download). You can increase the `min_location_downloads` threshold to filter out low-activity locations if needed (e.g., to reduce noise or improve performance on very large datasets).

---

## Feature List

### Basic Location Features (7 features)

These features capture fundamental statistics about user activity at each location.

- **`unique_users`**: Total number of distinct users who downloaded from this location.
- **`downloads_per_user`**: Average number of downloads per unique user at this location (total_downloads / unique_users).
- **`avg_users_per_hour`**: Average number of distinct users active per hour at this location.
- **`max_users_per_hour`**: Maximum number of distinct users active in any single hour at this location.
- **`user_cv`**: Coefficient of variation of users per hour (normalized by global mean), indicating consistency of user activity patterns.
- **`users_per_active_hour`**: Average number of users per active hour (unique_users / active_hours), measuring user density.
- **`projects_per_user`**: Average number of distinct projects accessed per user at this location.

---

### Time-of-Day Features (5 features)

These features capture temporal patterns of activity throughout the day.

- **`hourly_download_std`**: Standard deviation of downloads across 24 hours, measuring how evenly distributed activity is throughout the day.
- **`peak_hour_concentration`**: Fraction of downloads occurring during the peak hour, indicating concentration of activity.
- **`working_hours_ratio`**: Fraction of downloads occurring during working hours (typically 9 AM - 5 PM local time), indicating institutional vs. individual user patterns.
- **`hourly_entropy`**: Shannon entropy of the hourly download distribution, measuring randomness/regularity of time patterns (higher = more random, lower = more regular).
- **`night_activity_ratio`**: Fraction of downloads occurring during night hours (typically 11 PM - 6 AM local time), indicating automated or non-human patterns.

---

### Yearly Pattern Features (10 features)

These features capture long-term temporal patterns and trends across years.

- **`yearly_entropy`**: Shannon entropy of downloads across years, measuring how evenly distributed activity is across different years.
- **`peak_year_concentration`**: Fraction of downloads occurring in the peak year, indicating concentration of activity in a single year.
- **`years_span`**: Number of distinct years with activity at this location.
- **`downloads_per_year`**: Average number of downloads per year (total_downloads / years_span).
- **`year_over_year_cv`**: Coefficient of variation of downloads across years, measuring consistency of activity over time.
- **`fraction_latest_year`**: Fraction of total downloads occurring in the most recent year, indicating recent surge in activity.
- **`is_new_location`**: Binary indicator (0 or 1) whether this location first appeared in the latest year.
- **`spike_ratio`**: Ratio of downloads in latest year to average downloads in previous years, measuring sudden increase in activity.
- **`years_before_latest`**: Number of years with activity before the latest year.
- **`latest_year_downloads`**: Total number of downloads in the most recent year.

---

### Country-Level Features (11 features)

These features aggregate statistics at the country level to detect coordinated bot activity across multiple locations.

- **`locations_per_country`**: Total number of locations in the same country.
- **`country_latest_year_dl`**: Total downloads from all locations in this country in the latest year.
- **`country_total_dl`**: Total downloads from all locations in this country across all years.
- **`country_avg_fraction_latest`**: Average fraction of latest year downloads across all locations in this country.
- **`country_new_locations`**: Number of new locations (first appeared in latest year) in this country.
- **`country_high_spike_locations`**: Number of locations in this country with spike_ratio > 1.5.
- **`country_low_dl_user_locations`**: Number of locations in this country with downloads_per_user < 30.
- **`country_total_users`**: Total number of unique users across all locations in this country.
- **`country_fraction_latest`**: Fraction of country's total downloads occurring in the latest year.
- **`country_new_location_ratio`**: Fraction of locations in this country that are new (first appeared in latest year).
- **`country_suspicious_location_ratio`**: Fraction of locations in this country showing suspicious patterns (high spike or low downloads per user).

---

### Anomaly Detection Feature (1 feature)

- **`anomaly_score`**: Isolation Forest anomaly score for this location (higher = more anomalous), computed using all location-level features.

---

## Feature Extractors

The EBI provider includes three main feature extractors:

### 1. YearlyPatternExtractor
Extracts temporal patterns across years, including entropy, concentration, spike detection, and new location identification.

### 2. TimeOfDayExtractor
Extracts hourly activity patterns, including working hours ratio, night activity, and entropy of time distribution.

### 3. CountryLevelExtractor
Aggregates location-level statistics to country level for detecting coordinated bot activity.

---

## Usage

```python
from deeplogbot.features.providers.ebi import extract_location_features_ebi
import duckdb

conn = duckdb.connect()
df = extract_location_features_ebi(conn, 'path/to/data.parquet')
```

---

## Feature Usage by Classification Method

### Rules Method
- Uses all features implicitly through YAML-configurable rule conditions
- Also uses `anomaly_score` from Isolation Forest

### Deep Method
- Uses all features plus ~40 additional behavioral and discriminative features
- Multi-stage pipeline: seed selection, Organic VAE, Deep Isolation Forest, temporal consistency, fusion

---

## Notes

- All features are computed at the **location level** (geo_location + country)
- By default, all locations are included (minimum 1 download); use `--min-location-downloads` to filter
- Features are designed to distinguish bots (many users, low DL/user) from hubs (few users, high DL/user)
- Country-level features help detect coordinated bot activity across multiple locations in the same country

