"""Deep classification combining Isolation Forest and Transformers.

This module implements a multi-stage architecture:
1. Isolation Forest: Initial anomaly detection
2. Transformers: Sequence-based feature encoding for direct classification

The Transformer processes time-series features and combines them with fixed features
to directly classify locations into categories (BOT, DOWNLOAD_HUB, NORMAL, INDEPENDENT_USER, OTHER).
This approach is similar to the paper's architecture without clustering.

ENHANCEMENTS FOR DISCOVERING NEW BOT PATTERNS:
===============================================

The original approach was limited by circular dependency on rule-based labels.
This module now includes 7 key enhancements to discover bots beyond existing rules:

1. **Anomaly-Based Label Generation** (generate_anomaly_based_labels):
   - Uses HDBSCAN clustering on Isolation Forest anomalies
   - Discovers natural groupings based on behavioral patterns
   - Eliminates dependency on hard-coded thresholds

2. **Contrastive Learning** (ContrastiveTransformerEncoder):
   - Learns representations where similar patterns cluster together
   - Uses NT-Xent contrastive loss with data augmentation
   - Discovers patterns WITHOUT explicit labels

3. **Pseudo-Label Refinement** (iterative_pseudo_label_refinement):
   - Starts with rule-based labels as noisy pseudo-labels
   - Iteratively refines predictions using model confidence
   - Allows model to override low-confidence rule predictions

4. **Temporal Anomaly Detection** (TemporalAnomalyDetector):
   - Bidirectional LSTM with attention for temporal patterns
   - Detects bot-specific signatures:
     * Regular/periodic access patterns
     * Sudden bursts followed by silence
     * Non-human access timing (3 AM spikes)

5. **Ensemble-Based Discovery** (ensemble_bot_discovery):
   - Combines Isolation Forest, LOF, and One-Class SVM
   - Focuses on cases where 2+ methods agree
   - Discovers anomalies that rules miss

6. **Bot Signature Features** (add_bot_signature_features):
   - Adds 7 discriminative features:
     * access_regularity: Std dev of hourly distribution
     * ua_per_user: User-Agent diversity per user
     * request_velocity: Downloads per active hour
     * ip_concentration: 1 - IP entropy
     * session_anomaly: Deviation from median session length
     * request_pattern_anomaly: 1 / file request entropy
     * weekend_weekday_imbalance: Deviation from expected 2/7 ratio

7. **Active Learning** (identify_uncertain_cases_for_review):
   - Uses entropy + margin-based uncertainty
   - Identifies edge cases for human review
   - Helps discover new attack patterns

These enhancements enable the model to discover NEW bots that don't match
existing rule patterns, with better generalization to evolving behaviors.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

from ...utils import logger
from ..isoforest.models import train_isolation_forest

# Try to import HDBSCAN, fall back to DBSCAN if not available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("HDBSCAN not available, falling back to DBSCAN for clustering")


# =====================================================================
# Constants for Bot Detection
# =====================================================================

# Bot label signal weights
BOT_SIGNAL_WEIGHTS = {
    'FEW_USERS_EXTREME_DL': 0.8,
    'VERY_FEW_USERS_HIGH_DL': 0.6,
    'MODERATE_USERS_HIGH_DL': 0.5,
    'RULE_BOT': 0.5,
    'RULE_HUB': 0.4,
    'EXTREME_DL_ZSCORE': 0.4,
    'HIGH_ANOMALY': 0.3,
    'NON_WORKING_HIGH_ACTIVITY': 0.2,
    'VERY_HIGH_ANOMALY': 0.2,
    'VERY_EXTREME_DL_ZSCORE': 0.2,
    'LOW_ENTROPY': 0.15,
}

# Bot detection thresholds
BOT_THRESHOLDS = {
    'FEW_USERS': 100,
    'VERY_FEW_USERS': 50,
    'EXTREME_DL_PER_USER': 50,
    'HIGH_DL_PER_USER': 30,
    'MODERATE_DL_PER_USER': 20,
    'HIGH_ANOMALY_SCORE': 0.2,
    'VERY_HIGH_ANOMALY_SCORE': 0.25,
    'LOW_WORKING_HOURS_RATIO': 0.3,
    'MIN_TOTAL_DOWNLOADS': 1000,
    'EXTREME_ZSCORE': 3.0,
    'VERY_EXTREME_ZSCORE': 4.0,
    'ADAPTIVE_THRESHOLD_PERCENTILE': 85,
    'FIXED_THRESHOLD': 0.5,
    'LOW_ENTROPY_QUANTILE': 0.2,
}

# Override thresholds (lowered for better bot detection)
OVERRIDE_THRESHOLDS = {
    'OVERRIDE1_DL_PER_USER': 30,  # Was 50
    'OVERRIDE1_MAX_USERS': 200,   # Was 100
    'OVERRIDE2_DL_PER_USER': 20,  # Was 30
    'OVERRIDE2_MAX_USERS': 100,   # Was 50
}

# Stratified processing thresholds
STRATIFICATION_THRESHOLDS = {
    'OBVIOUS_BOT_USERS': 2000,           # >2000 users = obvious bot
    'OBVIOUS_BOT_SINGLE_USER_DL': 1000,  # Single user with >1000 DL = bot
    'OBVIOUS_BOT_DL_PER_USER': 100,      # >100 DL/user with >500 users = bot
    'OBVIOUS_BOT_MODERATE_DL': 200,      # >200 DL/user with >100 users = bot
    'LEGITIMATE_MAX_USERS': 5,           # <=5 users
    'LEGITIMATE_MAX_DL_PER_USER': 3,     # <=3 DL/user
    'LEGITIMATE_MAX_TOTAL_DL': 50,       # <50 total downloads
    'LEGITIMATE_MAX_ANOMALY': 0.15,      # Low anomaly score
}

# Scale-aware anomaly detection thresholds
SCALE_THRESHOLDS = {
    'SMALL_MAX_USERS': 100,              # Small locations: 1-100 users
    'MEDIUM_MAX_USERS': 2000,            # Medium locations: 100-2000 users
    'SMALL_CONTAMINATION': 0.10,         # 10% bots in small locations
    'MEDIUM_CONTAMINATION': 0.20,        # 20% bots in medium locations
}

# Bot likelihood scoring weights
BOT_LIKELIHOOD_WEIGHTS = {
    'USER_COUNT_LOG': 0.10,              # log(users) / 10
    'DL_PER_USER': 0.04,                 # DL/user / 25, capped at 2
    'HOURLY_ENTROPY': 0.20,              # entropy / 5
    'NIGHT_ACTIVITY': 3.0,               # night ratio * 3
    'BURST_COEFFICIENT': 0.10,           # burst / 10, capped at 3
    'LOW_WORKING_HOURS': 2.0,            # Penalty for low working hours
    'HIGH_DL_CONCENTRATION': 2.0,        # High DL concentration penalty
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH_CONFIDENCE': 0.7,              # Trust predictions above this
    'LOW_CONFIDENCE': 0.4,               # Flag for review below this
}

# Feature weights for composite score
COMPOSITE_SCORE_WEIGHTS = {
    'DL_USER_PER_LOG_USERS': 0.3,
    'USER_SCARCITY': 0.25,
    'DOWNLOAD_CONCENTRATION': 0.25,
    'ANOMALY_SCORE': 0.2,
}

# Focal loss parameters
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0

# Attention and model parameters
ATTENTION_RESIDUAL_WEIGHT = 0.5
ANOMALY_SCORE_OFFSET = 0.5
LOG_USERS_OFFSET = 2
EPSILON = 1e-10


# =====================================================================
# Phase 5 Improvements: Smart pseudo-labels and enhanced architecture
# =====================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in bot detection."""
    def __init__(self, alpha: float = FOCAL_LOSS_ALPHA, gamma: float = FOCAL_LOSS_GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EnhancedBotHead(nn.Module):
    """Enhanced bot detection head with attention mechanism."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Self-attention for feature relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        # Feature interaction layers
        self.interaction_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        
        # Context layer
        self.context_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 2)  # Binary: bot or not
        )
        
    def forward(self, x):
        # Add sequence dimension for attention
        x_seq = x.unsqueeze(1)
        
        # Self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = attn_out.squeeze(1)
        
        # Residual connection
        combined = x + ATTENTION_RESIDUAL_WEIGHT * attn_out
        
        # Feature interaction with gating
        interaction_out = self.interaction_layer(combined)
        gate_out = self.gate(combined)
        gated_features = interaction_out * gate_out
        
        # Context features
        context_features = self.context_layer(x)
        
        # Combine features
        final_features = torch.cat([gated_features, context_features], dim=1)
        
        # Final classification
        return self.classifier(final_features)


def _has_required_columns(df: pd.DataFrame, *columns: str) -> bool:
    """Check if DataFrame has all required columns."""
    return all(col in df.columns for col in columns)


# =============================================================================
# Ground Truth Rules - Simple, Clear, Defensible
# =============================================================================

@dataclass
class GroundTruthThresholds:
    """Clear thresholds for ground truth classification."""
    # BOT: Many coordinated fake users with minimal activity
    bot_min_users: int = 10_000
    bot_max_dl_per_user: float = 10.0
    
    # HUB: Institutional mirrors with heavy automated sync
    hub_min_dl_per_user: float = 1_000.0
    
    # INDIVIDUAL: Few users with minimal activity
    individual_max_users: int = 5
    individual_max_dl_per_user: float = 3.0


class GroundTruthCategory(Enum):
    """Ground truth categories with clear definitions."""
    BOT = "bot"                    # Coordinated fake users
    HUB = "download_hub"           # Institutional mirrors
    INDIVIDUAL = "individual"      # Single researchers
    UNCERTAIN = "uncertain"        # Needs pattern discovery


def apply_ground_truth_rules(
    df: pd.DataFrame, 
    thresholds: Optional[GroundTruthThresholds] = None
) -> pd.DataFrame:
    """
    Apply simple, clear ground truth rules.
    
    Only 3 clear rules - everything else goes to pattern discovery.
    
    Args:
        df: DataFrame with features
        thresholds: Ground truth thresholds
        
    Returns:
        DataFrame with ground_truth_category column
    """
    if thresholds is None:
        thresholds = GroundTruthThresholds()
    
    # Initialize as uncertain
    df['ground_truth_category'] = GroundTruthCategory.UNCERTAIN.value
    df['ground_truth_confidence'] = 0.0
    
    # Rule 1: BOT - Many users, low DL/user
    bot_mask = (
        (df['unique_users'] > thresholds.bot_min_users) & 
        (df['downloads_per_user'] < thresholds.bot_max_dl_per_user)
    )
    df.loc[bot_mask, 'ground_truth_category'] = GroundTruthCategory.BOT.value
    df.loc[bot_mask, 'ground_truth_confidence'] = 1.0
    
    # Rule 2: HUB - Very high DL/user (regardless of user count)
    hub_mask = df['downloads_per_user'] > thresholds.hub_min_dl_per_user
    df.loc[hub_mask, 'ground_truth_category'] = GroundTruthCategory.HUB.value
    df.loc[hub_mask, 'ground_truth_confidence'] = 1.0
    
    # Rule 3: INDIVIDUAL - Few users, low DL/user
    individual_mask = (
        (df['unique_users'] <= thresholds.individual_max_users) & 
        (df['downloads_per_user'] <= thresholds.individual_max_dl_per_user)
    )
    df.loc[individual_mask, 'ground_truth_category'] = GroundTruthCategory.INDIVIDUAL.value
    df.loc[individual_mask, 'ground_truth_confidence'] = 1.0
    
    # Log statistics
    n_bot = (df['ground_truth_category'] == GroundTruthCategory.BOT.value).sum()
    n_hub = (df['ground_truth_category'] == GroundTruthCategory.HUB.value).sum()
    n_individual = (df['ground_truth_category'] == GroundTruthCategory.INDIVIDUAL.value).sum()
    n_uncertain = (df['ground_truth_category'] == GroundTruthCategory.UNCERTAIN.value).sum()
    
    logger.info("Ground Truth Classification:")
    logger.info(f"  BOT (>10K users, <10 DL/user): {n_bot:,} ({n_bot/len(df)*100:.1f}%)")
    logger.info(f"  HUB (>1K DL/user): {n_hub:,} ({n_hub/len(df)*100:.1f}%)")
    logger.info(f"  INDIVIDUAL (≤5 users, ≤3 DL/user): {n_individual:,} ({n_individual/len(df)*100:.1f}%)")
    logger.info(f"  UNCERTAIN (needs pattern discovery): {n_uncertain:,} ({n_uncertain/len(df)*100:.1f}%)")
    
    return df


# =============================================================================
# Pattern Discovery Features - Behavioral Signatures
# =============================================================================

def extract_behavioral_features(df: pd.DataFrame, input_parquet: str, conn) -> pd.DataFrame:
    """
    Extract features designed for pattern discovery.
    
    These features capture behavioral signatures that distinguish patterns like:
    - Pipeline/CI: Regular intervals, same files
    - Research group: Working hours, diverse files
    - Automated sync: Daily/weekly patterns
    
    Args:
        df: DataFrame with basic features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        
    Returns:
        DataFrame with behavioral features added
    """
    from ...features.schema import EBI_SCHEMA
    schema = EBI_SCHEMA
    
    escaped_path = input_parquet.replace("'", "''")
    
    # Feature 1: Temporal regularity (how mechanical are the download times?)
    logger.info("  Extracting temporal regularity features...")
    
    # Simplified query to reduce memory usage - sample instead of using all intervals
    temporal_query = f"""
    WITH sampled_downloads AS (
        SELECT 
            {schema.location_field} as geo_location,
            {schema.timestamp_field} as download_time,
            ROW_NUMBER() OVER (PARTITION BY {schema.location_field} ORDER BY RANDOM()) as rn
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    limited_downloads AS (
        -- Sample up to 50 downloads per location to reduce memory/disk usage
        SELECT geo_location, download_time 
        FROM sampled_downloads 
        WHERE rn <= 50
    ),
    download_intervals AS (
        SELECT 
            geo_location,
            EPOCH(CAST(download_time AS TIMESTAMP) - 
                  LAG(CAST(download_time AS TIMESTAMP)) 
                  OVER (PARTITION BY geo_location ORDER BY download_time)) as interval_seconds
        FROM limited_downloads
    )
    SELECT 
        geo_location,
        AVG(interval_seconds) as mean_interval,
        STDDEV(interval_seconds) as std_interval,
        -- Coefficient of variation: low = mechanical, high = random
        CASE 
            WHEN AVG(interval_seconds) > 0 
            THEN STDDEV(interval_seconds) / AVG(interval_seconds) 
            ELSE 1.0 
        END as interval_cv,
        -- Regularity score: inverse of CV (high = mechanical)
        CASE 
            WHEN STDDEV(interval_seconds) > 0 
            THEN AVG(interval_seconds) / (STDDEV(interval_seconds) + 1) 
            ELSE 0 
        END as regularity_score
    FROM download_intervals
    WHERE interval_seconds IS NOT NULL AND interval_seconds > 0
    GROUP BY geo_location
    """
    
    try:
        temporal_df = conn.execute(temporal_query).df()
        df = df.merge(temporal_df, on='geo_location', how='left')
        df['regularity_score'] = df['regularity_score'].fillna(0)
        df['interval_cv'] = df['interval_cv'].fillna(1)
    except Exception as e:
        logger.warning(f"  Temporal features extraction failed: {e}")
        df['regularity_score'] = 0
        df['interval_cv'] = 1
    
    # Feature 2: Day-of-week patterns (weekday vs weekend)
    logger.info("  Extracting day-of-week patterns...")
    
    dow_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        -- Weekend ratio: 0 = all weekday, 1 = all weekend
        AVG(CASE 
            WHEN DAYOFWEEK(CAST({schema.timestamp_field} AS TIMESTAMP)) IN (0, 6) 
            THEN 1.0 ELSE 0.0 
        END) as weekend_ratio,
        -- Weekday concentration: how concentrated on specific days?
        COUNT(DISTINCT DAYOFWEEK(CAST({schema.timestamp_field} AS TIMESTAMP))) as unique_days_of_week
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    AND {schema.timestamp_field} IS NOT NULL
    GROUP BY {schema.location_field}
    """
    
    try:
        dow_df = conn.execute(dow_query).df()
        df = df.merge(dow_df, on='geo_location', how='left')
        df['weekend_ratio'] = df['weekend_ratio'].fillna(0.3)
        df['unique_days_of_week'] = df['unique_days_of_week'].fillna(5)
    except Exception as e:
        logger.warning(f"  Day-of-week features extraction failed: {e}")
        df['weekend_ratio'] = 0.3
        df['unique_days_of_week'] = 5
    
    # Feature 3: File diversity (same files vs diverse files)
    logger.info("  Extracting file diversity features...")
    
    file_query = f"""
    SELECT 
        {schema.location_field} as geo_location,
        COUNT(DISTINCT filename) as unique_files,
        COUNT(*) as total_downloads,
        -- File concentration: 1/unique_files (high = few files, low = many files)
        1.0 / NULLIF(COUNT(DISTINCT filename), 0) as file_concentration,
        -- File diversity ratio: unique_files / total_downloads
        CAST(COUNT(DISTINCT filename) AS DOUBLE) / COUNT(*) as file_diversity_ratio
    FROM read_parquet('{escaped_path}')
    WHERE {schema.location_field} IS NOT NULL
    GROUP BY {schema.location_field}
    """
    
    try:
        file_df = conn.execute(file_query).df()
        df = df.merge(
            file_df[['geo_location', 'unique_files', 'file_concentration', 'file_diversity_ratio']], 
            on='geo_location', 
            how='left'
        )
        df['file_diversity_ratio'] = df['file_diversity_ratio'].fillna(1)
        df['file_concentration'] = df['file_concentration'].fillna(1)
    except Exception as e:
        logger.warning(f"  File diversity features extraction failed: {e}")
        df['file_diversity_ratio'] = 1
        df['file_concentration'] = 1
    
    # Feature 4: Session patterns (do downloads come in bursts/sessions?)
    logger.info("  Extracting session pattern features...")
    
    # Define a session as downloads within 30 minutes of each other
    session_query = f"""
    WITH numbered_downloads AS (
        SELECT 
            {schema.location_field} as geo_location,
            CAST({schema.timestamp_field} AS TIMESTAMP) as ts,
            ROW_NUMBER() OVER (PARTITION BY {schema.location_field} ORDER BY {schema.timestamp_field}) as rn
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
    ),
    session_breaks AS (
        SELECT 
            a.geo_location,
            CASE 
                WHEN EPOCH(a.ts - b.ts) > 1800 THEN 1  -- 30 min gap = new session
                ELSE 0 
            END as is_new_session
        FROM numbered_downloads a
        LEFT JOIN numbered_downloads b ON a.geo_location = b.geo_location AND a.rn = b.rn + 1
    )
    SELECT 
        geo_location,
        SUM(is_new_session) + 1 as num_sessions,
        COUNT(*) as total_in_sessions
    FROM session_breaks
    GROUP BY geo_location
    """
    
    try:
        session_df = conn.execute(session_query).df()
        session_df['downloads_per_session'] = session_df['total_in_sessions'] / session_df['num_sessions'].replace(0, 1)
        df = df.merge(
            session_df[['geo_location', 'num_sessions', 'downloads_per_session']], 
            on='geo_location', 
            how='left'
        )
        df['num_sessions'] = df['num_sessions'].fillna(1)
        df['downloads_per_session'] = df['downloads_per_session'].fillna(df['total_downloads'])
    except Exception as e:
        logger.warning(f"  Session pattern features extraction failed: {e}")
        df['num_sessions'] = 1
        df['downloads_per_session'] = df['total_downloads']
    
    # Derived behavioral features
    df['is_mechanical'] = (df['regularity_score'] > df['regularity_score'].quantile(0.75)).astype(float)
    df['is_weekday_biased'] = (df['weekend_ratio'] < 0.15).astype(float)
    df['is_single_file'] = (df.get('file_diversity_ratio', 0) < 0.1).astype(float)
    df['is_bursty'] = (df['downloads_per_session'] > df['downloads_per_session'].quantile(0.75)).astype(float)
    
    logger.info("  Behavioral features extracted successfully")
    
    return df


# =============================================================================
# Temporal Sequence Extraction
# =============================================================================

def extract_temporal_sequences(
    df: pd.DataFrame,
    input_parquet: str,
    conn,
    max_sequence_length: int = 100,
    sequence_features: List[str] = None
) -> pd.DataFrame:
    """
    Extract temporal sequences from raw download data.
    
    Creates sequences of temporal features for each location:
    - Time intervals between downloads
    - Hour of day for each download
    - Day of week for each download
    - Download volume (counts per time window)
    
    Args:
        df: DataFrame with location features
        input_parquet: Path to input parquet file
        conn: DuckDB connection
        max_sequence_length: Maximum sequence length (will pad/truncate)
        sequence_features: List of feature names to extract (default: all)
        
    Returns:
        DataFrame with 'temporal_sequence' column containing sequences
    """
    logger.info("Extracting temporal sequences from download data...")
    
    from ...features.schema import EBI_SCHEMA
    schema = EBI_SCHEMA
    escaped_path = input_parquet.replace("'", "''")
    
    # Get list of locations to process
    locations = df['geo_location'].unique().tolist()
    n_locations = len(locations)
    
    # Create sequences for each location
    sequences = {}
    
    # Process in smaller batches to avoid memory/disk issues
    batch_size = 500
    for batch_start in range(0, n_locations, batch_size):
        batch_end = min(batch_start + batch_size, n_locations)
        batch_locations = locations[batch_start:batch_end]
        
        if batch_start % 5000 == 0:
            logger.info(f"  Processing locations {batch_start:,}/{n_locations:,}")
        
        # Create location filter
        location_filter = "', '".join([loc.replace("'", "''") for loc in batch_locations])
        
        # Extract temporal sequences
        sequence_query = f"""
        WITH location_downloads AS (
            SELECT 
                {schema.location_field} as geo_location,
                CAST({schema.timestamp_field} AS TIMESTAMP) as download_time,
                ROW_NUMBER() OVER (
                    PARTITION BY {schema.location_field} 
                    ORDER BY CAST({schema.timestamp_field} AS TIMESTAMP)
                ) as download_order
            FROM read_parquet('{escaped_path}')
            WHERE {schema.location_field} IN ('{location_filter}')
            AND {schema.timestamp_field} IS NOT NULL
        ),
        limited_downloads AS (
            -- Limit to first 200 downloads per location to reduce memory/disk usage
            SELECT geo_location, download_time, download_order
            FROM location_downloads
            WHERE download_order <= 200
        ),
        sequences AS (
            SELECT 
                geo_location,
                -- Time interval since previous download (in hours)
                EPOCH(download_time - LAG(download_time) OVER (
                    PARTITION BY geo_location 
                    ORDER BY download_time
                )) / 3600.0 as interval_hours,
                -- Hour of day (0-23)
                EXTRACT(HOUR FROM download_time) as hour_of_day,
                -- Day of week (0=Sunday, 6=Saturday)
                EXTRACT(DOW FROM download_time) as day_of_week,
                -- Day of month (1-31)
                EXTRACT(DAY FROM download_time) as day_of_month,
                -- Week of year (1-52)
                EXTRACT(WEEK FROM download_time) as week_of_year,
                -- Normalized timestamp (days since first download)
                EPOCH(download_time - MIN(download_time) OVER (PARTITION BY geo_location)) / 86400.0 as days_since_start,
                download_order
            FROM limited_downloads
        )
        SELECT 
            geo_location,
            LIST(interval_hours ORDER BY download_order) as intervals,
            LIST(hour_of_day ORDER BY download_order) as hours,
            LIST(day_of_week ORDER BY download_order) as days_of_week,
            LIST(day_of_month ORDER BY download_order) as days_of_month,
            LIST(week_of_year ORDER BY download_order) as weeks,
            LIST(days_since_start ORDER BY download_order) as normalized_times,
            COUNT(*) as sequence_length
        FROM sequences
        WHERE interval_hours IS NOT NULL
        GROUP BY geo_location
        """
        
        try:
            sequence_df = conn.execute(sequence_query).df()
            
            for _, row in sequence_df.iterrows():
                loc = row['geo_location']
                seq_len = int(row['sequence_length']) if pd.notna(row['sequence_length']) else 0
                
                # Create feature vectors for each time step
                # Convert to lists and handle None/NaN values
                def safe_list(value):
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        return []
                    if isinstance(value, (list, np.ndarray)):
                        return list(value)
                    return []
                
                intervals = safe_list(row.get('intervals'))
                hours = safe_list(row.get('hours'))
                days_of_week = safe_list(row.get('days_of_week'))
                days_of_month = safe_list(row.get('days_of_month'))
                weeks = safe_list(row.get('weeks'))
                normalized_times = safe_list(row.get('normalized_times'))
                
                # Build sequence: each time step has [interval_hours, hour_of_day, day_of_week, day_of_month, week_of_year, normalized_time]
                sequence = []
                for i in range(min(seq_len, max_sequence_length)):
                    if i < len(intervals) and len(intervals) > 0:
                        # Normalize features - handle None/NaN values
                        interval_val = intervals[i]
                        if interval_val is None or (isinstance(interval_val, float) and np.isnan(interval_val)):
                            interval = 0.0
                        else:
                            interval = float(interval_val)
                        # Safely extract other features
                        hour = float(hours[i]) if i < len(hours) and hours[i] is not None and not (isinstance(hours[i], float) and np.isnan(hours[i])) else 12.0
                        dow = float(days_of_week[i]) if i < len(days_of_week) and days_of_week[i] is not None and not (isinstance(days_of_week[i], float) and np.isnan(days_of_week[i])) else 3.0
                        dom = float(days_of_month[i]) if i < len(days_of_month) and days_of_month[i] is not None and not (isinstance(days_of_month[i], float) and np.isnan(days_of_month[i])) else 15.0
                        week = float(weeks[i]) if i < len(weeks) and weeks[i] is not None and not (isinstance(weeks[i], float) and np.isnan(weeks[i])) else 26.0
                        norm_time = float(normalized_times[i]) if i < len(normalized_times) and normalized_times[i] is not None and not (isinstance(normalized_times[i], float) and np.isnan(normalized_times[i])) else 0.0
                        
                        # Normalize interval (log scale for better distribution)
                        interval_norm = np.log1p(interval) / 10.0  # log(1+x)/10 to keep in reasonable range
                        
                        # Normalize hour to [0, 1]
                        hour_norm = hour / 23.0
                        
                        # Normalize day of week to [0, 1]
                        dow_norm = dow / 6.0
                        
                        # Normalize day of month to [0, 1]
                        dom_norm = (dom - 1) / 30.0
                        
                        # Normalize week to [0, 1]
                        week_norm = (week - 1) / 51.0
                        
                        # Normalize time (already in days)
                        time_norm = min(norm_time / 365.0, 1.0)  # Cap at 1 year
                        
                        sequence.append([
                            interval_norm,
                            hour_norm,
                            dow_norm,
                            dom_norm,
                            week_norm,
                            time_norm
                        ])
                    else:
                        # Padding
                        sequence.append([0.0] * 6)
                
                # Pad or truncate to max_sequence_length
                if len(sequence) < max_sequence_length:
                    padding = [[0.0] * 6] * (max_sequence_length - len(sequence))
                    sequence = padding + sequence
                else:
                    sequence = sequence[:max_sequence_length]
                
                sequences[loc] = sequence
                
        except Exception as e:
            logger.warning(f"  Error extracting sequences for batch {batch_start}-{batch_end}: {e}")
            # Fill with empty sequences
            for loc in batch_locations:
                if loc not in sequences:
                    sequences[loc] = [[0.0] * 6] * max_sequence_length
    
    # Add sequences to dataframe
    df['temporal_sequence'] = df['geo_location'].map(sequences)
    
    # Fill missing sequences with zeros
    missing_mask = df['temporal_sequence'].isna()
    if missing_mask.any():
        empty_sequence = [[0.0] * 6] * max_sequence_length
        df.loc[missing_mask, 'temporal_sequence'] = df.loc[missing_mask, 'geo_location'].apply(
            lambda x: empty_sequence
        )
    
    logger.info(f"  Extracted temporal sequences for {len(sequences):,} locations")
    logger.info(f"  Sequence length: {max_sequence_length}, Features per timestep: 6")
    
    return df


# =====================================================================
# Phase 6: Stratified Processing, Scale-Aware Detection, Bot Likelihood
# =====================================================================

def stratified_prefilter(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Multi-tier pre-filtering to separate obvious cases from uncertain ones.
    
    This significantly reduces computational load and improves accuracy by:
    1. Skipping deep learning for obvious bots (>2000 users)
    2. Skipping deep learning for obvious legitimate users (<5 users, <3 DL/user)
    3. Focusing deep learning on genuinely ambiguous patterns
    
    Args:
        df: DataFrame with location features
        
    Returns:
        Tuple of (obvious_bots, obvious_legitimate, uncertain) boolean Series
    """
    thresholds = STRATIFICATION_THRESHOLDS
    
    # Tier 1: Obvious bots (high confidence)
    obvious_bots = pd.Series(False, index=df.index)
    
    if _has_required_columns(df, 'unique_users', 'total_downloads', 'downloads_per_user'):
        obvious_bots = (
            # Primary threshold: >2000 users per location
            (df['unique_users'] > thresholds['OBVIOUS_BOT_USERS']) |
            
            # Single user hammering (1 user with >1000 downloads)
            ((df['unique_users'] == 1) & (df['total_downloads'] > thresholds['OBVIOUS_BOT_SINGLE_USER_DL'])) |
            
            # Coordinated high-volume activity (>100 DL/user with >500 users)
            ((df['downloads_per_user'] > thresholds['OBVIOUS_BOT_DL_PER_USER']) & 
             (df['unique_users'] > 500)) |
            
            # Extreme download concentration (>200 DL/user with >100 users)
            ((df['downloads_per_user'] > thresholds['OBVIOUS_BOT_MODERATE_DL']) & 
             (df['unique_users'] > 100))
        )
    
    # Tier 2: Obvious legitimate (high confidence)
    obvious_legitimate = pd.Series(False, index=df.index)
    
    if _has_required_columns(df, 'unique_users', 'downloads_per_user', 'total_downloads'):
        base_legitimate = (
            (df['unique_users'] <= thresholds['LEGITIMATE_MAX_USERS']) & 
            (df['downloads_per_user'] <= thresholds['LEGITIMATE_MAX_DL_PER_USER']) &
            (df['total_downloads'] < thresholds['LEGITIMATE_MAX_TOTAL_DL'])
        )
        
        # Add anomaly score check if available
        if 'anomaly_score' in df.columns:
            obvious_legitimate = base_legitimate & (df['anomaly_score'] < thresholds['LEGITIMATE_MAX_ANOMALY'])
        else:
            obvious_legitimate = base_legitimate
    
    # Tier 3: Uncertain - needs deep analysis
    uncertain = ~(obvious_bots | obvious_legitimate)
    
    # Log statistics
    n_obvious_bots = obvious_bots.sum()
    n_obvious_legitimate = obvious_legitimate.sum()
    n_uncertain = uncertain.sum()
    total = len(df)
    
    logger.info(f"  Stratified pre-filtering results:")
    logger.info(f"    Tier 1 (Obvious bots): {n_obvious_bots:,} ({n_obvious_bots/total*100:.1f}%)")
    logger.info(f"    Tier 2 (Obvious legitimate): {n_obvious_legitimate:,} ({n_obvious_legitimate/total*100:.1f}%)")
    logger.info(f"    Tier 3 (Uncertain - deep analysis): {n_uncertain:,} ({n_uncertain/total*100:.1f}%)")
    
    return obvious_bots, obvious_legitimate, uncertain


def compute_bot_likelihood_score(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a composite bot likelihood score for each location.
    
    Higher scores indicate higher probability of being a bot.
    Typical range: 0-15+ (bots: 8-12, legitimate: 1-3)
    
    Args:
        df: DataFrame with location features
        
    Returns:
        numpy array of bot likelihood scores
    """
    weights = BOT_LIKELIHOOD_WEIGHTS
    score = np.zeros(len(df))
    
    # Signal 1: User count (logarithmic scale)
    # More users = higher likelihood of being a bot farm
    if 'unique_users' in df.columns:
        score += np.log1p(df['unique_users'].values) * weights['USER_COUNT_LOG']
    
    # Signal 2: Downloads per user
    # High DL/user = bot-like behavior
    if 'downloads_per_user' in df.columns:
        score += np.clip(df['downloads_per_user'].values * weights['DL_PER_USER'], 0, 2)
    
    # Signal 3: Temporal irregularity (hourly entropy)
    # High entropy = irregular patterns = bot-like
    if 'hourly_entropy' in df.columns:
        score += df['hourly_entropy'].values * weights['HOURLY_ENTROPY']
    
    # Signal 4: Night activity ratio
    # Bots work 24/7, humans sleep
    if 'night_ratio' in df.columns:
        score += df['night_ratio'].values * weights['NIGHT_ACTIVITY']
    elif 'working_hours_ratio' in df.columns:
        # Invert: low working hours = high night activity
        score += (1 - df['working_hours_ratio'].values) * weights['LOW_WORKING_HOURS']
    
    # Signal 5: Burst coefficient (spike ratio)
    # Sudden spikes = bot behavior
    if 'spike_ratio' in df.columns:
        score += np.clip(df['spike_ratio'].values * weights['BURST_COEFFICIENT'], 0, 3)
    
    # Signal 6: Download concentration
    # High concentration with moderate users = suspicious
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        concentration = df['downloads_per_user'].values / (np.log(df['unique_users'].values + 2))
        high_concentration = concentration > np.percentile(concentration, 90)
        score[high_concentration] += weights['HIGH_DL_CONCENTRATION']
    
    # Signal 7: Anomaly score boost
    if 'anomaly_score' in df.columns:
        score += df['anomaly_score'].values * 3  # Scale anomaly to similar range
    
    return score


def scale_aware_anomaly_detection(df: pd.DataFrame, feature_columns: List[str], 
                                   contamination: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run separate anomaly detection for different user count scales.
    
    Different scales have different bot patterns:
    - Small locations (1-100 users): Focus on DL/user ratio, temporal patterns
    - Medium locations (100-2000 users): Focus on coordination, user diversity
    
    Args:
        df: DataFrame with location features
        feature_columns: List of feature column names
        contamination: Base contamination rate
        
    Returns:
        Tuple of (predictions, anomaly_scores)
    """
    from sklearn.ensemble import IsolationForest
    
    thresholds = SCALE_THRESHOLDS
    predictions = np.zeros(len(df))
    anomaly_scores = np.zeros(len(df))
    
    # Prepare feature matrix
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Small locations (1-100 users): Different bot patterns
    small_mask = (df['unique_users'] >= 1) & (df['unique_users'] <= thresholds['SMALL_MAX_USERS'])
    if small_mask.any():
        logger.info(f"    Scale-aware: Processing {small_mask.sum():,} small locations (1-{thresholds['SMALL_MAX_USERS']} users)")
        
        iso_forest_small = IsolationForest(
            contamination=thresholds['SMALL_CONTAMINATION'],
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        small_indices = np.where(small_mask)[0]
        X_small = X_scaled[small_indices]
        
        predictions[small_indices] = iso_forest_small.fit_predict(X_small)
        anomaly_scores[small_indices] = -iso_forest_small.score_samples(X_small)
    
    # Medium locations (100-2000 users): Your "uncertain" zone
    medium_mask = ((df['unique_users'] > thresholds['SMALL_MAX_USERS']) & 
                   (df['unique_users'] <= thresholds['MEDIUM_MAX_USERS']))
    if medium_mask.any():
        logger.info(f"    Scale-aware: Processing {medium_mask.sum():,} medium locations ({thresholds['SMALL_MAX_USERS']}-{thresholds['MEDIUM_MAX_USERS']} users)")
        
        iso_forest_medium = IsolationForest(
            contamination=thresholds['MEDIUM_CONTAMINATION'],
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        medium_indices = np.where(medium_mask)[0]
        X_medium = X_scaled[medium_indices]
        
        predictions[medium_indices] = iso_forest_medium.fit_predict(X_medium)
        anomaly_scores[medium_indices] = -iso_forest_medium.score_samples(X_medium)
    
    # Large locations (>2000 users): Already pre-filtered, but process if any remain
    large_mask = df['unique_users'] > thresholds['MEDIUM_MAX_USERS']
    if large_mask.any():
        logger.info(f"    Scale-aware: Processing {large_mask.sum():,} large locations (>{thresholds['MEDIUM_MAX_USERS']} users)")
        
        iso_forest_large = IsolationForest(
            contamination=0.30,  # Higher contamination for large locations
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        large_indices = np.where(large_mask)[0]
        X_large = X_scaled[large_indices]
        
        predictions[large_indices] = iso_forest_large.fit_predict(X_large)
        anomaly_scores[large_indices] = -iso_forest_large.score_samples(X_large)
    
    # Normalize anomaly scores to 0-1 range
    if anomaly_scores.max() > anomaly_scores.min():
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    return predictions, anomaly_scores


def confidence_based_classification(df: pd.DataFrame, logits: torch.Tensor, 
                                     bot_logits: Optional[torch.Tensor] = None,
                                     category_map: Optional[dict] = None) -> pd.DataFrame:
    """
    Apply confidence-based classification with fallback strategies.
    
    Only trusts high-confidence predictions. Low-confidence cases
    are flagged for review or use bot head as fallback.
    
    Args:
        df: DataFrame with location features
        logits: Classification logits from the model
        bot_logits: Optional bot head logits
        category_map: Mapping from class indices to category names
        
    Returns:
        DataFrame with classification results and confidence levels
    """
    thresholds = CONFIDENCE_THRESHOLDS
    
    if category_map is None:
        category_map = {0: 'bot', 1: 'download_hub', 2: 'independent_user', 3: 'normal', 4: 'other'}
    
    # Get prediction probabilities
    probs = F.softmax(logits, dim=1)
    max_probs, predicted = torch.max(probs, dim=1)
    
    max_probs_np = max_probs.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    
    # Initialize columns
    df['user_category'] = 'other'
    df['classification_confidence'] = max_probs_np
    df['needs_review'] = False
    
    # High confidence: Trust the prediction
    high_confidence_mask = max_probs_np >= thresholds['HIGH_CONFIDENCE']
    if high_confidence_mask.any():
        high_conf_indices = np.where(high_confidence_mask)[0]
        df.iloc[high_conf_indices, df.columns.get_loc('user_category')] = [
            category_map[pred] for pred in predicted_np[high_conf_indices]
        ]
    
    # Medium confidence: Use prediction but flag for potential review
    medium_confidence_mask = (max_probs_np >= thresholds['LOW_CONFIDENCE']) & \
                             (max_probs_np < thresholds['HIGH_CONFIDENCE'])
    if medium_confidence_mask.any():
        medium_conf_indices = np.where(medium_confidence_mask)[0]
        df.iloc[medium_conf_indices, df.columns.get_loc('user_category')] = [
            category_map[pred] for pred in predicted_np[medium_conf_indices]
        ]
    
    # Low confidence: Use bot head as fallback or flag for review
    low_confidence_mask = max_probs_np < thresholds['LOW_CONFIDENCE']
    if low_confidence_mask.any():
        low_conf_indices = np.where(low_confidence_mask)[0]
        df.iloc[low_conf_indices, df.columns.get_loc('needs_review')] = True
        
        if bot_logits is not None:
            # Use bot head as fallback
            bot_probs = F.softmax(bot_logits, dim=1)
            _, bot_predicted = torch.max(bot_probs, dim=1)
            bot_predicted_np = bot_predicted.cpu().numpy()
            
            for idx in low_conf_indices:
                if bot_predicted_np[idx] == 1:  # Bot head says bot
                    df.iloc[idx, df.columns.get_loc('user_category')] = 'bot'
                else:
                    df.iloc[idx, df.columns.get_loc('user_category')] = 'other'
        else:
            # No bot head, mark as other
            df.iloc[low_conf_indices, df.columns.get_loc('user_category')] = 'other'
    
    # Log statistics
    n_high = high_confidence_mask.sum()
    n_medium = medium_confidence_mask.sum()
    n_low = low_confidence_mask.sum()
    total = len(df)
    
    logger.info(f"  Confidence-based classification:")
    logger.info(f"    High confidence (>={thresholds['HIGH_CONFIDENCE']}): {n_high:,} ({n_high/total*100:.1f}%)")
    logger.info(f"    Medium confidence: {n_medium:,} ({n_medium/total*100:.1f}%)")
    logger.info(f"    Low confidence (<{thresholds['LOW_CONFIDENCE']}): {n_low:,} ({n_low/total*100:.1f}%)")
    
    return df


def generate_smart_bot_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Generate intelligent bot labels based on multiple signals.
    
    CRITICAL: Bots and Download Hubs have OPPOSITE patterns!
    - BOTS: Many users (>1000), low-moderate DL/user (<50), coordinated behavior
    - HUBS: Few users (<100), VERY high DL/user (>500), legitimate mirrors
    
    This function should ONLY give high scores to BOT patterns, NOT hub patterns.
    
    Args:
        df: DataFrame with location features
        
    Returns:
        Tuple of (hard_labels, soft_labels) as numpy arrays
    """
    # Initialize bot score (0-1)
    bot_score = np.zeros(len(df))
    
    # First, identify download hubs - these should NEVER get bot labels
    is_download_hub = pd.Series(False, index=df.index)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        # Hub pattern: Very high DL/user with few users
        is_download_hub = (
            ((df['downloads_per_user'] > 500) & (df['unique_users'] < 100)) |
            (df['downloads_per_user'] > 1000)  # Extreme DL/user is always a hub
        )
    
    # =========================================================================
    # BOT SIGNALS: Focus on MANY users + low-moderate DL/user
    # =========================================================================
    
    # Signal 1: Many users with low DL/user (classic bot farm pattern)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        # Bot pattern: Many users, low DL/user
        many_users_low_dl = (
            (df['unique_users'] > 1000) &  # Many users
            (df['downloads_per_user'] < 20) &  # Low DL/user
            ~is_download_hub  # NOT a hub
        )
        bot_score[many_users_low_dl] += 0.7
        
        # Bot pattern: Very many users, moderate DL/user
        very_many_users_moderate_dl = (
            (df['unique_users'] > 5000) &  # Very many users
            (df['downloads_per_user'] >= 20) &
            (df['downloads_per_user'] < 100) &  # Moderate DL/user
            ~is_download_hub
        )
        bot_score[very_many_users_moderate_dl] += 0.6
        
        # Bot pattern: Moderate users with suspicious DL/user ratio
        moderate_users_suspicious = (
            (df['unique_users'] > 500) &
            (df['unique_users'] <= 5000) &
            (df['downloads_per_user'] > 10) &
            (df['downloads_per_user'] < 50) &
            ~is_download_hub
        )
        bot_score[moderate_users_suspicious] += 0.4
    
    # Signal 2: High anomaly score (only for non-hubs)
    if 'anomaly_score' in df.columns:
        high_anomaly_non_hub = (df['anomaly_score'] > BOT_THRESHOLDS['HIGH_ANOMALY_SCORE']) & ~is_download_hub
        very_high_anomaly_non_hub = (df['anomaly_score'] > BOT_THRESHOLDS['VERY_HIGH_ANOMALY_SCORE']) & ~is_download_hub
        bot_score[high_anomaly_non_hub] += 0.2
        bot_score[very_high_anomaly_non_hub] += 0.15
    
    # Signal 3: Low working hours ratio with high activity (bots work 24/7)
    if _has_required_columns(df, 'working_hours_ratio', 'total_downloads', 'unique_users'):
        non_working_high_activity = (
            (df['working_hours_ratio'] < BOT_THRESHOLDS['LOW_WORKING_HOURS_RATIO']) &
            (df['total_downloads'] > BOT_THRESHOLDS['MIN_TOTAL_DOWNLOADS']) &
            (df['unique_users'] > 100) &  # Bots have many users
            ~is_download_hub
        )
        bot_score[non_working_high_activity] += 0.25
    
    # Signal 4: Low hourly entropy (coordinated access patterns)
    if 'hourly_entropy' in df.columns and 'unique_users' in df.columns:
        low_entropy_many_users = (
            (df['hourly_entropy'] < df['hourly_entropy'].quantile(BOT_THRESHOLDS['LOW_ENTROPY_QUANTILE'])) &
            (df['unique_users'] > 100) &  # Bots have many users
            ~is_download_hub
        )
        bot_score[low_entropy_many_users] += 0.15
    
    # Signal 5: Original rule-based bot classification (only for bots, NOT hubs)
    if 'user_category' in df.columns:
        rule_bot = df['user_category'] == 'bot'
        bot_score[rule_bot & ~is_download_hub] += 0.5
        # NOTE: We do NOT add score for hubs anymore!
    
    # =========================================================================
    # NEGATIVE SIGNALS: Reduce score for hub-like patterns
    # =========================================================================
    # Locations that look like hubs should have REDUCED bot scores
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        hub_like = (
            (df['downloads_per_user'] > 100) & 
            (df['unique_users'] < 50)
        )
        bot_score[hub_like] *= 0.1  # Reduce score by 90%
    
    # Normalize to [0, 1]
    bot_score = np.clip(bot_score, 0, 1)
    
    # Force hubs to have zero bot score
    bot_score[is_download_hub.values] = 0.0
    
    # Create both soft and hard labels
    soft_labels = bot_score
    
    # Adaptive threshold (stricter)
    percentile_threshold = np.percentile(bot_score[bot_score > 0], 75) if (bot_score > 0).any() else 0.5
    fixed_threshold = 0.4  # Stricter threshold
    threshold = max(percentile_threshold, fixed_threshold)  # Use MAX not MIN for stricter filtering
    
    hard_labels = (bot_score >= threshold).astype(float)
    
    logger.info(f"    Smart bot label statistics:")
    logger.info(f"      - Download hubs excluded: {is_download_hub.sum()}")
    logger.info(f"      - Locations with bot score > 0: {(bot_score > 0).sum()}")
    logger.info(f"      - Locations with bot score > 0.5: {(bot_score > 0.5).sum()}")
    logger.info(f"      - Threshold used: {threshold:.3f}")
    logger.info(f"      - Hard bot labels: {hard_labels.sum()}")
    
    return hard_labels, soft_labels


def add_bot_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features for better bot detection.
    
    Args:
        df: DataFrame with location features
        
    Returns:
        DataFrame with additional interaction features
    """
    # Core bot pattern: High DL/user with few users
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        df['dl_user_per_log_users'] = df['downloads_per_user'] / np.log(df['unique_users'] + LOG_USERS_OFFSET)
        
        df['user_scarcity_score'] = np.where(
            df['downloads_per_user'] > BOT_THRESHOLDS['MODERATE_DL_PER_USER'],
            np.exp(-df['unique_users'] / BOT_THRESHOLDS['FEW_USERS']),
            0
        )
        
        df['download_concentration'] = df['downloads_per_user'] * (1 / (df['unique_users'] + 1))
    
    # Anomaly-weighted features
    if 'anomaly_score' in df.columns and 'downloads_per_user' in df.columns:
        df['anomaly_dl_interaction'] = (df['anomaly_score'] + ANOMALY_SCORE_OFFSET).clip(0, 1) * df['downloads_per_user']
    
    # Temporal features
    if 'hourly_entropy' in df.columns and 'downloads_per_user' in df.columns:
        df['temporal_irregularity'] = (1 / (df['hourly_entropy'] + 0.1)) * np.log(df['downloads_per_user'] + 1)
    
    # Composite bot score
    score_components = []
    weights = []
    
    if 'dl_user_per_log_users' in df.columns:
        max_val = df['dl_user_per_log_users'].quantile(0.95)
        if max_val > EPSILON:
            score_components.append(np.clip(df['dl_user_per_log_users'] / max_val, 0, 1))
            weights.append(COMPOSITE_SCORE_WEIGHTS['DL_USER_PER_LOG_USERS'])
    
    if 'user_scarcity_score' in df.columns:
        score_components.append(df['user_scarcity_score'])
        weights.append(COMPOSITE_SCORE_WEIGHTS['USER_SCARCITY'])
    
    if 'download_concentration' in df.columns:
        max_val = df['download_concentration'].quantile(0.95)
        if max_val > EPSILON:
            score_components.append(np.clip(df['download_concentration'] / max_val, 0, 1))
            weights.append(COMPOSITE_SCORE_WEIGHTS['DOWNLOAD_CONCENTRATION'])
    
    if 'anomaly_score' in df.columns:
        score_components.append((df['anomaly_score'] + ANOMALY_SCORE_OFFSET).clip(0, 1))
        weights.append(COMPOSITE_SCORE_WEIGHTS['ANOMALY_SCORE'])
    
    if score_components:
        weights = np.array(weights) / np.sum(weights)
        df['bot_composite_score'] = sum(w * s for w, s in zip(weights, score_components))
    
    return df


def apply_bot_detection_override(df: pd.DataFrame) -> pd.DataFrame:
    """Apply post-processing overrides for obvious bot patterns.
    
    CRITICAL: Bots and Download Hubs have OPPOSITE patterns!
    - BOTS: Many users (>1000), low-moderate DL/user (<50), coordinated behavior
    - HUBS: Few users (<100), VERY high DL/user (>500), legitimate mirrors
    
    This function should ONLY override to 'bot' for BOT patterns.
    """
    
    override_count = 0
    hub_override_count = 0
    
    # First, protect download hubs - these should NEVER be classified as bots
    # Ensure is_protected_hub is boolean type (not float with NaN)
    if 'is_protected_hub' not in df.columns:
        df['is_protected_hub'] = False
    else:
        # Convert to boolean, treating NaN as False
        df['is_protected_hub'] = df['is_protected_hub'].fillna(False).astype(bool)
    
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        # Hub pattern: Very high DL/user with few users (mirrors, institutional)
        obvious_hubs = (
            ((df['downloads_per_user'] > 500) & (df['unique_users'] < 100)) |
            (df['downloads_per_user'] > 1000)
        )
        
        if obvious_hubs.any():
            logger.info(f"    Protecting {obvious_hubs.sum()} download hub patterns from bot override")
            df.loc[obvious_hubs, 'is_protected_hub'] = True
            df.loc[obvious_hubs, 'user_category'] = 'download_hub'
            df.loc[obvious_hubs, 'is_bot_neural'] = False
            hub_override_count = obvious_hubs.sum()
    
    # BOT Override 1: Many users with moderate DL/user (distributed bot farm)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        obvious_bots = (
            (df['unique_users'] > 5000) &  # Many users (bot farms)
            (df['downloads_per_user'] > 10) &
            (df['downloads_per_user'] < 100) &  # Moderate DL/user
            (df['is_protected_hub'] == False)  # NOT a protected hub
        )
        
        if 'is_bot_neural' in df.columns:
            obvious_bots = obvious_bots & (df['is_bot_neural'] == False)
        
        if obvious_bots.any():
            logger.info(f"    Applying bot override for {obvious_bots.sum()} obvious bot patterns "
                       f"(>5000 users, 10-100 DL/user)")
            df.loc[obvious_bots, 'is_bot_neural'] = True
            df.loc[obvious_bots, 'user_category'] = 'bot'
            override_count += obvious_bots.sum()
    
    # BOT Override 2: Very many users with low DL/user (coordinated access)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        coordinated_bots = (
            (df['unique_users'] > 10000) &  # Very many users
            (df['downloads_per_user'] < 20) &  # Low DL/user (coordinated)
            (df['is_protected_hub'] == False)  # NOT a protected hub
        )
        
        if 'is_bot_neural' in df.columns:
            coordinated_bots = coordinated_bots & (df['is_bot_neural'] == False)
        
        if coordinated_bots.any():
            logger.info(f"    Applying bot override for {coordinated_bots.sum()} coordinated bot patterns "
                      f"(>10000 users, <20 DL/user)")
            df.loc[coordinated_bots, 'is_bot_neural'] = True
            df.loc[coordinated_bots, 'user_category'] = 'bot'
            override_count += coordinated_bots.sum()
    
    logger.info(f"    Total bot overrides applied: {override_count}")
    logger.info(f"    Total hub protections applied: {hub_override_count}")
    
    return df


class TransformerClassifier(nn.Module):
    """Transformer-based classifier that combines time-series and fixed features."""
    
    def __init__(self, ts_input_dim: int, fixed_input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512,
                 num_classes: int = 5, enable_reconstruction: bool = False, enable_bot_head: bool = False):
        """
        Args:
            ts_input_dim: Dimension of time-series features per window
            fixed_input_dim: Dimension of fixed (non-time-series) features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            num_classes: Number of output classes (bot, hub, normal, independent_user, other)
            enable_reconstruction: Whether to enable reconstruction head for self-supervised learning
        """
        super(TransformerClassifier, self).__init__()
        
        # Transformer encoder for time-series features
        self.ts_input_projection = nn.Linear(ts_input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection for fixed features
        self.fixed_projection = nn.Linear(fixed_input_dim, d_model)
        
        # Reconstruction head for self-supervised learning (masked time-step prediction)
        self.enable_reconstruction = enable_reconstruction
        if enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_feedforward, ts_input_dim)
            )
        
        # Combine time-series and fixed features
        combined_dim = d_model + d_model  # Transformer output + fixed features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        # Optional: Binary classification head for 'is_bot' prediction
        self.enable_bot_head = enable_bot_head
        if enable_bot_head:
            # Enhanced bot head with attention mechanism
            self.bot_head = EnhancedBotHead(combined_dim, hidden_dim=dim_feedforward // 2)
    
    def forward(self, ts_features: torch.Tensor, fixed_features: torch.Tensor, 
                mask_indices: Optional[torch.Tensor] = None, return_reconstruction: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass combining time-series and fixed features.
        
        Args:
            ts_features: Time-series features [batch_size, seq_len, ts_input_dim]
            fixed_features: Fixed features [batch_size, fixed_input_dim]
            mask_indices: Optional mask for self-supervised learning [batch_size, seq_len] (True = masked)
            return_reconstruction: Whether to return reconstruction for masked steps
        
        Returns:
            Tuple of (classification_logits [batch_size, num_classes], bot_logits [batch_size, 2] or None, reconstruction [batch_size, seq_len, ts_input_dim] or None)
        """
        # Encode time-series features
        ts_proj = self.ts_input_projection(ts_features)
        ts_encoded = self.transformer(ts_proj)  # [batch_size, seq_len, d_model]
        # Global average pooling over sequence length
        ts_pooled = ts_encoded.mean(dim=1)  # [batch_size, d_model]
        
        # Project fixed features
        fixed_proj = self.fixed_projection(fixed_features)  # [batch_size, d_model]
        
        # Concatenate and classify
        combined = torch.cat([ts_pooled, fixed_proj], dim=1)  # [batch_size, 2*d_model]
        logits = self.classifier(combined)  # [batch_size, num_classes]
        
        if return_reconstruction and self.enable_reconstruction and mask_indices is not None:
            # Reconstruct masked time steps
            reconstruction = self.reconstruction_head(ts_encoded)  # [batch_size, seq_len, ts_input_dim]
        else:
            reconstruction = None

        if self.enable_bot_head:
            bot_logits = self.bot_head(combined) # [batch_size, 2]
        else:
            bot_logits = None
        
        return logits, bot_logits, reconstruction


class TimeSeriesDataset(Dataset):
    """Dataset for self-supervised pre-training of Transformer."""
    
    def __init__(self, ts_features: np.ndarray, fixed_features: np.ndarray, 
                 mask_prob: float = 0.15):
        """
        Args:
            ts_features: Time-series features [num_samples, seq_len, num_features]
            fixed_features: Fixed features [num_samples, num_features]
            mask_prob: Probability of masking each time step
        """
        self.ts_features = torch.FloatTensor(ts_features)
        self.fixed_features = torch.FloatTensor(fixed_features)
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.ts_features)
    
    def __getitem__(self, idx):
        ts = self.ts_features[idx]
        fixed = self.fixed_features[idx]
        
        # Create random mask for this sample
        seq_len = ts.shape[0]
        mask = torch.rand(seq_len) < self.mask_prob
        
        return ts, fixed, mask


def train_self_supervised(
    classifier: TransformerClassifier,
    X_ts: torch.Tensor,
    X_fixed: torch.Tensor,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    mask_prob: float = 0.15,
    validation_split: float = 0.1
) -> TransformerClassifier:
    """
    Self-supervised pre-training using masked time-step prediction.
    
    Args:
        classifier: TransformerClassifier model
        X_ts: Time-series features [num_samples, seq_len, num_features]
        X_fixed: Fixed features [num_samples, num_features]
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        mask_prob: Probability of masking each time step
        validation_split: Fraction of data to use for validation
    
    Returns:
        Trained classifier
    """
    logger.info("    Starting self-supervised pre-training...")
    
    # Split data
    n_samples = len(X_ts)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    X_ts_train = X_ts[train_indices]
    X_fixed_train = X_fixed[train_indices]
    X_ts_val = X_ts[val_indices]
    X_fixed_val = X_fixed[val_indices]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        X_ts_train.cpu().numpy(),
        X_fixed_train.cpu().numpy(),
        mask_prob=mask_prob
    )
    val_dataset = TimeSeriesDataset(
        X_ts_val.cpu().numpy(),
        X_fixed_val.cpu().numpy(),
        mask_prob=mask_prob
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    classifier.train()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss = 0.0
        for ts_batch, fixed_batch, mask_batch in train_loader:
            ts_batch = ts_batch.to(device)
            fixed_batch = fixed_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with reconstruction
            _, _, reconstruction = classifier(
                ts_batch, fixed_batch, 
                mask_indices=mask_batch, 
                return_reconstruction=True
            )
            
            # Compute loss only on masked positions
            mask_expanded = mask_batch.unsqueeze(-1).expand_as(ts_batch)
            masked_reconstruction = reconstruction[mask_expanded].view(-1, ts_batch.shape[-1])
            masked_target = ts_batch[mask_expanded].view(-1, ts_batch.shape[-1])
            
            if masked_target.numel() > 0:
                loss = criterion(masked_reconstruction, masked_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ts_batch, fixed_batch, mask_batch in val_loader:
                ts_batch = ts_batch.to(device)
                fixed_batch = fixed_batch.to(device)
                mask_batch = mask_batch.to(device)
                
                _, _, reconstruction = classifier(
                    ts_batch, fixed_batch,
                    mask_indices=mask_batch,
                    return_reconstruction=True
                )
                
                mask_expanded = mask_batch.unsqueeze(-1).expand_as(ts_batch)
                masked_reconstruction = reconstruction[mask_expanded].view(-1, ts_batch.shape[-1])
                masked_target = ts_batch[mask_expanded].view(-1, ts_batch.shape[-1])
                
                if masked_target.numel() > 0:
                    loss = criterion(masked_reconstruction, masked_target)
                    val_loss += loss.item()
        
        classifier.train()
        
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"      Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"      Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"    Self-supervised pre-training completed. Best validation loss: {best_val_loss:.6f}")
    return classifier


def train_supervised_classifier(
    classifier: TransformerClassifier,
    X_ts: torch.Tensor,
    X_fixed: torch.Tensor,
    y_labels: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
                 validation_split: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None,
                 y_is_bot: Optional[torch.Tensor] = None,
                 lambda_bot_loss: float = 0.5
) -> TransformerClassifier:
    """
    Supervised fine-tuning of the classifier using rule-based labels.
    
    Args:
        classifier: Pre-trained TransformerClassifier
        X_ts: Time-series features [num_samples, seq_len, num_features]
        X_fixed: Fixed features [num_samples, num_features]
        y_labels: Class labels [num_samples] (0=BOT, 1=DOWNLOAD_HUB, 2=INDEPENDENT_USER, 3=NORMAL, 4=OTHER)
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Fraction of data to use for validation
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Trained classifier
    """
    logger.info("    Starting supervised fine-tuning...")
    
    # Split data
    n_samples = len(X_ts)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    X_ts_train = X_ts[train_indices]
    X_fixed_train = X_fixed[train_indices]
    y_train = y_labels[train_indices]
    y_is_bot_train = y_is_bot[train_indices] if y_is_bot is not None else None
    X_ts_val = X_ts[val_indices]
    X_fixed_val = X_fixed[val_indices]
    y_val = y_labels[val_indices]
    y_is_bot_val = y_is_bot[val_indices] if y_is_bot is not None else None
    
    # Create datasets
    if y_is_bot_train is not None:
        train_dataset = torch.utils.data.TensorDataset(X_ts_train, X_fixed_train, y_train, y_is_bot_train)
        val_dataset = torch.utils.data.TensorDataset(X_ts_val, X_fixed_val, y_val, y_is_bot_val)
    else:
        train_dataset = torch.utils.data.TensorDataset(X_ts_train, X_fixed_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_ts_val, X_fixed_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function with class weights
    if class_weights is None:
        criterion_classification = nn.CrossEntropyLoss()
    else:
        criterion_classification = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Focal loss for bot head to handle class imbalance
    if y_is_bot is not None:
        criterion_bot = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)  # Focal loss for better bot detection
    
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    best_val_bot_f1 = 0.0 # Track F1 for bot head
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_bot_correct = 0
        train_bot_total = 0
        
        for batch_data in train_loader:
            if y_is_bot_train is not None:
                ts_batch, fixed_batch, y_batch, y_is_bot_batch = batch_data
                y_is_bot_batch = y_is_bot_batch.to(device)
            else:
                ts_batch, fixed_batch, y_batch = batch_data
                y_is_bot_batch = None
            
            ts_batch = ts_batch.to(device)
            fixed_batch = fixed_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            classification_logits, bot_logits, _ = classifier(ts_batch, fixed_batch)
            
            loss_classification = criterion_classification(classification_logits, y_batch)
            total_loss = loss_classification

            if y_is_bot_batch is not None and bot_logits is not None:
                loss_bot = criterion_bot(bot_logits, y_is_bot_batch)
                total_loss += lambda_bot_loss * loss_bot
                
                _, predicted_bot = torch.max(bot_logits.data, 1)
                train_bot_total += y_is_bot_batch.size(0)
                train_bot_correct += (predicted_bot == y_is_bot_batch).sum().item()

            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            _, predicted = torch.max(classification_logits.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_bot_correct = 0
        val_bot_total = 0
        val_bot_predictions = []
        val_bot_targets = []

        with torch.no_grad():
            for batch_data_val in val_loader:
                if y_is_bot_val is not None:
                    ts_batch, fixed_batch, y_batch, y_is_bot_batch = batch_data_val
                    y_is_bot_batch = y_is_bot_batch.to(device)
                else:
                    ts_batch, fixed_batch, y_batch = batch_data_val
                    y_is_bot_batch = None

                ts_batch = ts_batch.to(device)
                fixed_batch = fixed_batch.to(device)
                y_batch = y_batch.to(device)
                
                classification_logits, bot_logits, _ = classifier(ts_batch, fixed_batch)
                
                loss_classification = criterion_classification(classification_logits, y_batch)
                total_loss = loss_classification

                if y_is_bot_batch is not None and bot_logits is not None:
                    loss_bot = criterion_bot(bot_logits, y_is_bot_batch)
                    total_loss += lambda_bot_loss * loss_bot

                    _, predicted_bot = torch.max(bot_logits.data, 1)
                    val_bot_total += y_is_bot_batch.size(0)
                    val_bot_correct += (predicted_bot == y_is_bot_batch).sum().item()
                    val_bot_predictions.extend(predicted_bot.cpu().numpy())
                    val_bot_targets.extend(y_is_bot_batch.cpu().numpy())

                val_loss += total_loss.item()
                
                _, predicted = torch.max(classification_logits.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        train_bot_acc = train_bot_correct / train_bot_total if train_bot_total > 0 else 0.0
        val_bot_acc = val_bot_correct / val_bot_total if val_bot_total > 0 else 0.0

        # Calculate F1-score for bot head
        val_bot_f1 = 0.0
        if y_is_bot_val is not None and len(val_bot_targets) > 0:
            from sklearn.metrics import f1_score
            val_bot_f1 = f1_score(val_bot_targets, val_bot_predictions, average='binary', pos_label=1)

        if (epoch + 1) % 5 == 0:
            logger.info(f"      Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Train Acc={train_acc:.4f} (Bot Acc={train_bot_acc:.4f}), Val Loss={avg_val_loss:.6f}, Val Acc={val_acc:.4f} (Bot Acc={val_bot_acc:.4f}, Bot F1={val_bot_f1:.4f})")
        
        # Early stopping based on overall validation accuracy OR bot F1
        if val_acc > best_val_acc or val_bot_f1 > best_val_bot_f1: # Prioritize bot F1 if it's better
            best_val_acc = max(val_acc, best_val_acc)
            best_val_bot_f1 = max(val_bot_f1, best_val_bot_f1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"      Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"    Supervised fine-tuning completed. Best validation accuracy: {best_val_acc:.4f}, Best Bot F1: {best_val_bot_f1:.4f}")
    return classifier


# =====================================================================
# Enhancement 1: Anomaly-Based Label Generation
# =====================================================================

def generate_anomaly_based_labels(df: pd.DataFrame, feature_columns: List[str], 
                                   anomaly_scores: np.ndarray,
                                   min_cluster_size: int = 50) -> Tuple[np.ndarray, Dict]:
    """
    Generate labels using HDBSCAN clustering on Isolation Forest anomalies.
    
    This replaces rule-based labels with unsupervised discovery of natural groupings
    based on behavioral patterns rather than hard-coded thresholds.
    
    Args:
        df: DataFrame with location features
        feature_columns: List of feature column names
        anomaly_scores: Anomaly scores from Isolation Forest
        min_cluster_size: Minimum cluster size for HDBSCAN
        
    Returns:
        Tuple of (cluster_labels, cluster_metadata)
    """
    logger.info("    Generating anomaly-based labels with clustering...")
    
    # Prepare features
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Combine features with anomaly scores
    X_with_anomaly = np.column_stack([X_scaled, anomaly_scores.reshape(-1, 1)])
    
    # Use HDBSCAN if available, otherwise fall back to DBSCAN
    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=10,
            metric='euclidean',
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(X_with_anomaly)
        cluster_probs = clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None
    else:
        # Fall back to DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X_with_anomaly)
        cluster_probs = None
    
    # Analyze clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])
    n_noise = np.sum(cluster_labels == -1)
    
    logger.info(f"      Found {n_clusters} clusters, {n_noise} noise points")
    
    # Compute cluster metadata
    cluster_metadata = {}
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        mask = cluster_labels == cluster_id
        cluster_metadata[cluster_id] = {
            'size': mask.sum(),
            'mean_anomaly_score': anomaly_scores[mask].mean(),
            'mean_users': df.loc[mask, 'unique_users'].mean() if 'unique_users' in df.columns else 0,
            'mean_dl_per_user': df.loc[mask, 'downloads_per_user'].mean() if 'downloads_per_user' in df.columns else 0,
        }
        logger.info(f"      Cluster {cluster_id}: size={cluster_metadata[cluster_id]['size']}, "
                   f"anomaly={cluster_metadata[cluster_id]['mean_anomaly_score']:.3f}, "
                   f"users={cluster_metadata[cluster_id]['mean_users']:.1f}, "
                   f"dl/user={cluster_metadata[cluster_id]['mean_dl_per_user']:.1f}")
    
    return cluster_labels, cluster_metadata


# =====================================================================
# Enhancement 2: Contrastive Learning for Pattern Discovery
# =====================================================================

class ContrastiveTransformerEncoder(nn.Module):
    """
    Transformer encoder that learns representations through contrastive learning.
    
    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss to pull
    similar behavioral patterns together without explicit labels.
    """
    
    def __init__(self, ts_input_dim: int, fixed_input_dim: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512,
                 projection_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        
        # Shared encoder
        self.ts_input_projection = nn.Linear(ts_input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fixed_projection = nn.Linear(fixed_input_dim, d_model)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model * 2, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, projection_dim)
        )
        
        self.temperature = temperature
    
    def forward(self, ts_features: torch.Tensor, fixed_features: torch.Tensor) -> torch.Tensor:
        """Encode features and project to contrastive space."""
        # Encode time-series
        ts_proj = self.ts_input_projection(ts_features)
        ts_encoded = self.transformer(ts_proj)
        ts_pooled = ts_encoded.mean(dim=1)
        
        # Project fixed features
        fixed_proj = self.fixed_projection(fixed_features)
        
        # Combine and project
        combined = torch.cat([ts_pooled, fixed_proj], dim=1)
        z = self.projection_head(combined)
        
        # L2 normalize for cosine similarity
        z = F.normalize(z, dim=1)
        return z
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent contrastive loss between two augmented views.
        
        Args:
            z1, z2: Normalized embeddings [batch_size, projection_dim]
        
        Returns:
            Contrastive loss scalar
        """
        batch_size = z1.shape[0]
        
        # Compute similarity matrix
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create positive pair mask
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Positive pairs are at positions (i, i+batch_size) and (i+batch_size, i)
        positive_pairs = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        
        # Compute loss
        log_prob = F.log_softmax(sim_matrix, dim=1)
        loss = -log_prob[torch.arange(2 * batch_size, device=z.device), positive_pairs].mean()
        
        return loss


def augment_time_series(ts: torch.Tensor, augmentation_strength: float = 0.1) -> torch.Tensor:
    """
    Create augmented view of time series data for contrastive learning.
    
    Applies random transformations: noise, scaling, masking.
    
    Args:
        ts: Time series tensor [batch_size, seq_len, features]
        augmentation_strength: Strength of augmentation
        
    Returns:
        Augmented time series
    """
    ts_aug = ts.clone()
    
    # Add random noise
    noise = torch.randn_like(ts) * augmentation_strength
    ts_aug = ts_aug + noise
    
    # Random scaling per feature
    scale = 1.0 + torch.randn(ts.shape[0], 1, ts.shape[2], device=ts.device) * augmentation_strength
    ts_aug = ts_aug * scale
    
    # Random masking (set some timesteps to zero)
    mask_prob = augmentation_strength
    mask = torch.rand(ts.shape[0], ts.shape[1], 1, device=ts.device) > mask_prob
    ts_aug = ts_aug * mask
    
    return ts_aug


# =====================================================================
# Enhancement 3: Pseudo-Label Refinement
# =====================================================================

def iterative_pseudo_label_refinement(
    classifier: TransformerClassifier,
    X_ts: torch.Tensor,
    X_fixed: torch.Tensor,
    initial_labels: np.ndarray,
    device: torch.device,
    n_iterations: int = 3,
    confidence_threshold: float = 0.8,
    learning_rate: float = 1e-5,
    batch_size: int = 256
) -> Tuple[TransformerClassifier, np.ndarray]:
    """
    Iteratively refine pseudo-labels using model confidence.
    
    Starts with rule-based labels as noisy pseudo-labels, then allows
    the model to override low-confidence predictions in subsequent iterations.
    
    Args:
        classifier: Pre-trained classifier
        X_ts: Time-series features
        X_fixed: Fixed features
        initial_labels: Initial rule-based labels
        device: Training device
        n_iterations: Number of refinement iterations
        confidence_threshold: Threshold for high-confidence predictions
        learning_rate: Learning rate for refinement
        batch_size: Batch size
        
    Returns:
        Tuple of (refined_classifier, refined_labels)
    """
    logger.info(f"    Starting pseudo-label refinement ({n_iterations} iterations)...")
    
    current_labels = initial_labels.copy()
    
    for iteration in range(n_iterations):
        logger.info(f"      Iteration {iteration + 1}/{n_iterations}")
        
        # Train on current labels
        y_labels = torch.LongTensor(current_labels).to(device)
        dataset = torch.utils.data.TensorDataset(X_ts, X_fixed, y_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Brief training
        classifier.train()
        for _ in range(5):  # 5 epochs per iteration
            for ts_batch, fixed_batch, y_batch in loader:
                ts_batch = ts_batch.to(device)
                fixed_batch = fixed_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                logits, _, _ = classifier(ts_batch, fixed_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        
        # Get predictions and confidence
        classifier.eval()
        with torch.no_grad():
            logits, _, _ = classifier(X_ts, X_fixed)
            probs = F.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probs, dim=1)
            
            max_probs = max_probs.cpu().numpy()
            predictions = predictions.cpu().numpy()
        
        # Update labels for high-confidence predictions
        high_confidence_mask = max_probs > confidence_threshold
        n_updated = np.sum(current_labels[high_confidence_mask] != predictions[high_confidence_mask])
        current_labels[high_confidence_mask] = predictions[high_confidence_mask]
        
        logger.info(f"        Updated {n_updated} labels with high confidence (>{confidence_threshold})")
        logger.info(f"        Label distribution: {np.bincount(current_labels)}")
    
    logger.info("    Pseudo-label refinement completed")
    return classifier, current_labels


# =====================================================================
# Enhancement 4: Temporal Anomaly Detection
# =====================================================================

class TemporalAnomalyDetector(nn.Module):
    """
    Bidirectional LSTM with attention for detecting temporal bot patterns.
    
    Detects patterns like:
    - Regular/periodic access (bot scheduling)
    - Sudden bursts followed by silence
    - Non-human timing (3 AM spikes)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Anomaly score head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Time series [batch_size, seq_len, input_dim]
        
        Returns:
            Tuple of (anomaly_scores [batch_size], attention_weights)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pool over sequence
        pooled = attn_out.mean(dim=1)  # [batch_size, hidden_dim*2]
        
        # Anomaly score
        anomaly_scores = self.anomaly_head(pooled).squeeze(-1)  # [batch_size]
        
        return anomaly_scores, attn_weights


# =====================================================================
# Enhancement 5: Ensemble-Based Discovery
# =====================================================================

def ensemble_bot_discovery(df: pd.DataFrame, feature_columns: List[str],
                           contamination: float = 0.15) -> Tuple[np.ndarray, Dict]:
    """
    Combine multiple anomaly detection methods to discover bots.
    
    Uses Isolation Forest, LOF, and One-Class SVM. Focuses on cases where
    2+ methods agree but existing rules don't catch them.
    
    Args:
        df: DataFrame with location features
        feature_columns: List of feature column names
        contamination: Expected proportion of anomalies
        
    Returns:
        Tuple of (ensemble_predictions, method_agreement_counts)
    """
    logger.info("    Running ensemble anomaly detection...")
    
    # Prepare features
    X = df[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Method 1: Isolation Forest
    logger.info("      Running Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    iso_predictions = iso_forest.fit_predict(X_scaled)
    iso_scores = -iso_forest.score_samples(X_scaled)
    
    # Method 2: Local Outlier Factor
    logger.info("      Running Local Outlier Factor...")
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        n_jobs=-1
    )
    lof_predictions = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    
    # Method 3: One-Class SVM
    logger.info("      Running One-Class SVM...")
    ocsvm = OneClassSVM(
        nu=contamination,
        kernel='rbf',
        gamma='auto'
    )
    ocsvm_predictions = ocsvm.fit_predict(X_scaled)
    ocsvm_scores = -ocsvm.score_samples(X_scaled)
    
    # Normalize scores to [0, 1] - compute min/max once for efficiency
    iso_min, iso_max = iso_scores.min(), iso_scores.max()
    lof_min, lof_max = lof_scores.min(), lof_scores.max()
    ocsvm_min, ocsvm_max = ocsvm_scores.min(), ocsvm_scores.max()
    
    iso_scores = (iso_scores - iso_min) / (iso_max - iso_min + 1e-10)
    lof_scores = (lof_scores - lof_min) / (lof_max - lof_min + 1e-10)
    ocsvm_scores = (ocsvm_scores - ocsvm_min) / (ocsvm_max - ocsvm_min + 1e-10)
    
    # Count agreements (anomaly = -1, normal = 1)
    predictions = np.column_stack([iso_predictions, lof_predictions, ocsvm_predictions])
    anomaly_votes = np.sum(predictions == -1, axis=1)
    
    # Ensemble: 2 or more methods agree on anomaly
    ensemble_predictions = np.where(anomaly_votes >= 2, -1, 1)
    
    # Average scores for anomalies
    ensemble_scores = (iso_scores + lof_scores + ocsvm_scores) / 3
    
    # Statistics
    n_iso = np.sum(iso_predictions == -1)
    n_lof = np.sum(lof_predictions == -1)
    n_ocsvm = np.sum(ocsvm_predictions == -1)
    n_ensemble = np.sum(ensemble_predictions == -1)
    n_all_agree = np.sum(anomaly_votes == 3)
    n_two_agree = np.sum(anomaly_votes == 2)
    
    logger.info(f"      Isolation Forest: {n_iso} anomalies")
    logger.info(f"      LOF: {n_lof} anomalies")
    logger.info(f"      One-Class SVM: {n_ocsvm} anomalies")
    logger.info(f"      Ensemble (2+ agree): {n_ensemble} anomalies")
    logger.info(f"        All 3 methods agree: {n_all_agree}")
    logger.info(f"        2 methods agree: {n_two_agree}")
    
    method_agreement = {
        'iso_forest': n_iso,
        'lof': n_lof,
        'one_class_svm': n_ocsvm,
        'ensemble': n_ensemble,
        'all_agree': n_all_agree,
        'two_agree': n_two_agree,
        'anomaly_votes': anomaly_votes,
        'ensemble_scores': ensemble_scores
    }
    
    return ensemble_predictions, method_agreement


# =====================================================================
# Enhancement 6: Bot Signature Features
# =====================================================================

def add_bot_signature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add discriminative bot signature features.
    
    Features added:
    - access_regularity: Inverse of hourly entropy (low entropy = regular = bot-like)
    - ua_per_user: User-Agent diversity per user
    - request_velocity: Downloads per active hour
    - ip_concentration: 1 - IP entropy
    - session_anomaly: Deviation from median session length
    - request_pattern_anomaly: 1 / file request entropy
    - weekend_weekday_imbalance: Deviation from expected 2/7 ratio
    
    Args:
        df: DataFrame with location features
        
    Returns:
        DataFrame with additional signature features
    """
    logger.info("    Adding bot signature features...")
    
    # 1. Access regularity (temporal pattern consistency)
    if 'hourly_entropy' in df.columns:
        # High entropy = irregular, low entropy = regular (bot-like)
        df['access_regularity'] = 1.0 / (df['hourly_entropy'] + 0.1)
    else:
        df['access_regularity'] = 0.0
    
    # 2. User-Agent diversity (if available in raw data, use proxy)
    # Proxy: locations with many users but low diversity are suspicious
    if 'unique_users' in df.columns and 'total_downloads' in df.columns:
        df['ua_per_user'] = 1.0 / (df['unique_users'] / (df['total_downloads'] + 1) + 0.01)
    else:
        df['ua_per_user'] = 1.0
    
    # 3. Request velocity (downloads per active time)
    if 'total_downloads' in df.columns:
        # Estimate active hours from entropy and working hours ratio
        if 'working_hours_ratio' in df.columns:
            active_hours = df['working_hours_ratio'] * 24 * 7  # Hours per week
            df['request_velocity'] = df['total_downloads'] / (active_hours + 1)
        else:
            df['request_velocity'] = df['total_downloads'] / 168  # Assume full week
    else:
        df['request_velocity'] = 0.0
    
    # 4. IP concentration (1 - entropy)
    # Proxy: high users with low diversity suggests IP cycling
    if 'unique_users' in df.columns and 'downloads_per_user' in df.columns:
        # Estimate IP entropy from user distribution
        user_entropy = np.log1p(df['unique_users']) / np.log1p(df['unique_users'].max() + 1)
        df['ip_concentration'] = 1.0 - user_entropy
    else:
        df['ip_concentration'] = 0.0
    
    # 5. Session anomaly (deviation from median)
    if 'downloads_per_user' in df.columns:
        median_dl_per_user = df['downloads_per_user'].median()
        df['session_anomaly'] = np.abs(df['downloads_per_user'] - median_dl_per_user) / (median_dl_per_user + 1)
    else:
        df['session_anomaly'] = 0.0
    
    # 6. Request pattern anomaly (file diversity)
    # Bots often request same files repeatedly
    # Proxy: low DL/user with high users suggests coordinated same-file requests
    if 'downloads_per_user' in df.columns and 'unique_users' in df.columns:
        # Low DL/user = likely requesting same files
        file_entropy_proxy = df['downloads_per_user'] / (np.log1p(df['unique_users']) + 1)
        df['request_pattern_anomaly'] = 1.0 / (file_entropy_proxy + 0.1)
    else:
        df['request_pattern_anomaly'] = 0.0
    
    # 7. Weekend/weekday imbalance
    # Bots work 24/7, humans have patterns
    if 'working_hours_ratio' in df.columns:
        # Expected ratio: ~5/7 weekdays for humans
        # Bots: closer to uniform or inverted
        expected_weekday_ratio = 5.0 / 7.0
        df['weekend_weekday_imbalance'] = np.abs(df['working_hours_ratio'] - expected_weekday_ratio)
    else:
        df['weekend_weekday_imbalance'] = 0.0
    
    n_new_features = 7
    logger.info(f"      Added {n_new_features} bot signature features")
    
    return df


# =====================================================================
# Enhancement 7: Active Learning for Edge Cases
# =====================================================================

def identify_uncertain_cases_for_review(
    df: pd.DataFrame,
    classifier: TransformerClassifier,
    X_ts: torch.Tensor,
    X_fixed: torch.Tensor,
    device: torch.device,
    top_k: int = 100,
    entropy_weight: float = 0.5,
    margin_weight: float = 0.5
) -> pd.DataFrame:
    """
    Identify uncertain cases using entropy and margin-based uncertainty.
    
    These are cases where human review would help discover new bot patterns.
    
    Args:
        df: DataFrame with location features
        classifier: Trained classifier
        X_ts: Time-series features
        X_fixed: Fixed features
        device: Device
        top_k: Number of top uncertain cases to return
        entropy_weight: Weight for entropy-based uncertainty
        margin_weight: Weight for margin-based uncertainty
        
    Returns:
        DataFrame with top uncertain cases and their uncertainty scores
    """
    logger.info(f"    Identifying top {top_k} uncertain cases for review...")
    
    classifier.eval()
    with torch.no_grad():
        logits, _, _ = classifier(X_ts.to(device), X_fixed.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    
    # 1. Entropy-based uncertainty
    # High entropy = model is uncertain
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    entropy_normalized = entropy / np.log(probs.shape[1])  # Normalize to [0, 1]
    
    # 2. Margin-based uncertainty
    # Small margin between top 2 predictions = model is uncertain
    # Use partition for efficiency: O(n) instead of O(n log n)
    top_2_probs = np.partition(probs, -2, axis=1)[:, -2:]
    margin = top_2_probs[:, 1] - top_2_probs[:, 0]
    margin_uncertainty = 1.0 - margin  # Convert to uncertainty (low margin = high uncertainty)
    
    # 3. Combined uncertainty score
    uncertainty_score = entropy_weight * entropy_normalized + margin_weight * margin_uncertainty
    
    # Get top uncertain cases
    top_indices = np.argsort(uncertainty_score)[-top_k:][::-1]
    
    # Create result dataframe
    uncertain_df = df.iloc[top_indices].copy()
    uncertain_df['uncertainty_score'] = uncertainty_score[top_indices]
    uncertain_df['entropy'] = entropy[top_indices]
    uncertain_df['margin'] = margin[top_indices]
    uncertain_df['top_prediction'] = np.argmax(probs[top_indices], axis=1)
    uncertain_df['top_probability'] = np.max(probs[top_indices], axis=1)
    
    logger.info(f"      Found {top_k} uncertain cases with uncertainty scores:")
    logger.info(f"        Mean uncertainty: {uncertainty_score[top_indices].mean():.3f}")
    logger.info(f"        Mean entropy: {entropy[top_indices].mean():.3f}")
    logger.info(f"        Mean margin: {margin[top_indices].mean():.3f}")
    
    return uncertain_df


def classify_locations_deep(df: pd.DataFrame, feature_columns: List[str],
                              use_transformer: bool = True, random_state: int = 42,
                              contamination: float = 0.15, sequence_length: int = 12,
                              enable_self_supervised: bool = True,
                              pretrain_epochs: int = 20, pretrain_batch_size: int = 256,
                              pretrain_learning_rate: float = 1e-4,
                              enable_neural_classification: bool = True,
                              finetune_epochs: int = 30, finetune_batch_size: int = 256,
                              finetune_learning_rate: float = 1e-4,
                              enable_bot_head: bool = True,
                              lambda_bot_loss: float = 1.5,
                              enable_stratified_processing: bool = True,
                              enable_scale_aware_anomaly: bool = True,
                              enable_bot_likelihood: bool = True,
                              enable_confidence_classification: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify locations using deep architecture: Isolation Forest + Transformers.
    
    This method combines:
    1. Stratified pre-filtering (separate obvious bots/legitimate from uncertain)
    2. Scale-aware Isolation Forest for anomaly detection
    3. Bot likelihood scoring for enhanced pattern detection
    4. Transformers for sequence-based feature encoding
    5. Confidence-based classification with fallback strategies
    
    The Transformer processes time-series features to create rich embeddings,
    which are combined with fixed features for rule-based classification.
    This approach removes the need for clustering (DBSCAN).
    
    Categories generated:
    - BOT: Coordinated bot activity
    - DOWNLOAD_HUB: High downloads per user (mirrors, institutional)
    - INDEPENDENT_USER: Single or few users with low DL/user
    - NORMAL: Regular user patterns
    - OTHER: Unclassified patterns
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names (for Isolation Forest and fixed features)
        use_transformer: Whether to use Transformer encoding (default: True)
        random_state: Random seed
        contamination: Contamination rate for Isolation Forest
        sequence_length: Number of time windows in the time-series features
        enable_self_supervised: Whether to enable self-supervised pre-training (default: True)
        pretrain_epochs: Number of epochs for self-supervised pre-training (default: 20)
        pretrain_batch_size: Batch size for pre-training (default: 256)
        pretrain_learning_rate: Learning rate for pre-training (default: 1e-4)
        enable_neural_classification: Whether to use neural classification instead of rules (default: True)
        finetune_epochs: Number of epochs for supervised fine-tuning (default: 30)
        finetune_batch_size: Batch size for fine-tuning (default: 256)
        finetune_learning_rate: Learning rate for fine-tuning (default: 1e-4)
        lambda_bot_loss: Weight for bot detection loss (default: 1.5, increased for better bot detection)
        enable_stratified_processing: Whether to use stratified pre-filtering (default: True)
        enable_scale_aware_anomaly: Whether to use scale-aware anomaly detection (default: True)
        enable_bot_likelihood: Whether to compute bot likelihood scores (default: True)
        enable_confidence_classification: Whether to use confidence-based classification (default: True)
    
    Returns:
        Tuple of (DataFrame with classification columns added, empty cluster_df for compatibility)
    """
    logger.info("Training deep architecture classifier (Isolation Forest + Transformers)...")
    
    # Initialize classification columns
    df['user_category'] = 'normal'
    df['is_bot'] = False
    df['is_download_hub'] = False
    df['classification_confidence'] = 0.0
    df['needs_review'] = False
    
    # =========================================================================
    # Phase 6 Enhancement: Stratified Pre-filtering
    # =========================================================================
    obvious_bots_mask = pd.Series(False, index=df.index)
    obvious_legitimate_mask = pd.Series(False, index=df.index)
    uncertain_mask = pd.Series(True, index=df.index)
    
    if enable_stratified_processing:
        logger.info("  Step 0: Stratified pre-filtering (Phase 6 enhancement)...")
        obvious_bots_mask, obvious_legitimate_mask, uncertain_mask = stratified_prefilter(df)
        
        # Classify obvious cases immediately
        df.loc[obvious_bots_mask, 'user_category'] = 'bot'
        df.loc[obvious_bots_mask, 'is_bot'] = True
        df.loc[obvious_bots_mask, 'classification_confidence'] = 0.95
        
        df.loc[obvious_legitimate_mask, 'user_category'] = 'independent_user'
        df.loc[obvious_legitimate_mask, 'classification_confidence'] = 0.90
        
        # Only process uncertain cases with deep learning
        if not uncertain_mask.any():
            logger.info("  All locations classified by pre-filtering. Skipping deep learning.")
            cluster_df = pd.DataFrame()
            df['is_download_hub'] = df['user_category'] == 'download_hub'
            df['is_independent_user'] = df['user_category'] == 'independent_user'
            df['is_normal_user'] = df['user_category'] == 'normal'
            return df, cluster_df
        
        logger.info(f"  Processing {uncertain_mask.sum():,} uncertain locations with deep learning...")
    
    # Get subset for deep learning (uncertain cases only if stratified processing enabled)
    if enable_stratified_processing:
        df_uncertain = df[uncertain_mask].copy()
        uncertain_indices = df[uncertain_mask].index
    else:
        df_uncertain = df.copy()
        uncertain_indices = df.index
    
    # =========================================================================
    # Step 1: Anomaly Detection (Scale-aware or Standard)
    # =========================================================================
    if enable_scale_aware_anomaly and enable_stratified_processing:
        logger.info("  Step 1/3: Running scale-aware anomaly detection (Phase 6 enhancement)...")
        predictions, scores = scale_aware_anomaly_detection(
            df_uncertain, feature_columns, contamination=contamination
        )
        df_uncertain['is_anomaly'] = predictions == -1
        df_uncertain['anomaly_score'] = scores
    else:
        logger.info("  Step 1/3: Running standard Isolation Forest for anomaly detection...")
        predictions, scores, _, _ = train_isolation_forest(
            df_uncertain, feature_columns, contamination=contamination
        )
        df_uncertain['is_anomaly'] = predictions == -1
        df_uncertain['anomaly_score'] = -scores
    
    logger.info(f"    Detected {df_uncertain['is_anomaly'].sum():,} anomalous locations")
    
    # Copy anomaly results back to main dataframe
    df.loc[uncertain_indices, 'is_anomaly'] = df_uncertain['is_anomaly'].values
    df.loc[uncertain_indices, 'anomaly_score'] = df_uncertain['anomaly_score'].values
    
    # =========================================================================
    # Phase 6 Enhancement: Bot Likelihood Scoring
    # =========================================================================
    if enable_bot_likelihood:
        logger.info("  Computing bot likelihood scores (Phase 6 enhancement)...")
        bot_likelihood = compute_bot_likelihood_score(df_uncertain)
        df_uncertain['bot_likelihood_score'] = bot_likelihood
        df.loc[uncertain_indices, 'bot_likelihood_score'] = bot_likelihood
        
        # Use bot likelihood for high-confidence classifications
        high_likelihood_threshold = 8.0
        low_likelihood_threshold = 2.0
        
        high_likelihood_bots = bot_likelihood > high_likelihood_threshold
        low_likelihood = bot_likelihood < low_likelihood_threshold
        
        logger.info(f"    High bot likelihood (>{high_likelihood_threshold}): {high_likelihood_bots.sum():,}")
        logger.info(f"    Low bot likelihood (<{low_likelihood_threshold}): {low_likelihood.sum():,}")
        
        # Mark high-likelihood as bots with high confidence
        if high_likelihood_bots.any():
            high_likelihood_indices = uncertain_indices[high_likelihood_bots]
            df.loc[high_likelihood_indices, 'user_category'] = 'bot'
            df.loc[high_likelihood_indices, 'is_bot'] = True
            df.loc[high_likelihood_indices, 'classification_confidence'] = 0.85
    
    # Add interaction features for better bot detection
    logger.info("  Adding bot interaction features...")
    df = add_bot_interaction_features(df)
    
    # Add new features to feature columns if they exist
    new_features = ['dl_user_per_log_users', 'user_scarcity_score', 
                   'download_concentration', 'bot_composite_score',
                   'anomaly_dl_interaction', 'temporal_irregularity']
    for feat in new_features:
        if feat in df.columns and feat not in feature_columns:
            feature_columns.append(feat)
    
    # Prepare features for Transformer
    # Use time_series_features if available, otherwise fallback to flat features
    if 'time_series_features' in df.columns and df['time_series_features'].apply(lambda x: isinstance(x, list) and len(x) > 0).any():
        logger.info("  Using time-series features for Transformer.")
        # Convert list of lists to 3D numpy array: [num_locations, sequence_length, num_features_per_window]
        # Pad shorter sequences with zeros to match `sequence_length`
        max_seq_len = df['time_series_features'].apply(len).max()
        if max_seq_len < sequence_length:
            logger.warning(f"  Max sequence length found ({max_seq_len}) is less than requested ({sequence_length}). Padding with zeros.")
        
        # Determine num_features_per_window from the first valid entry
        valid_ts = df['time_series_features'].dropna()
        if len(valid_ts) == 0:
            logger.warning("  No valid time-series features found. Falling back to flat features.")
            X_ts = df[feature_columns].fillna(0).values.reshape(-1, 1, len(feature_columns))
            sequence_length = 1
            num_features_per_window = len(feature_columns)
        else:
            first_valid_ts = valid_ts.iloc[0]
            num_features_per_window = len(first_valid_ts[0]) if len(first_valid_ts) > 0 else 0

            if num_features_per_window == 0:
                logger.warning("  No features found in time_series_features. Falling back to flat features.")
                X_ts = df[feature_columns].fillna(0).values.reshape(-1, 1, len(feature_columns))
                sequence_length = 1
                num_features_per_window = len(feature_columns)
            else:
                X_ts_list = []
                for ts_list in df['time_series_features']:
                    if isinstance(ts_list, list):
                        # Pad or truncate to desired sequence_length
                        if len(ts_list) < sequence_length:
                            padded_ts = [[0.0] * num_features_per_window] * (sequence_length - len(ts_list)) + ts_list
                        elif len(ts_list) > sequence_length:
                            padded_ts = ts_list[-sequence_length:]
                        else:
                            padded_ts = ts_list
                        X_ts_list.append(padded_ts)
                    else:
                        X_ts_list.append([[0.0] * num_features_per_window] * sequence_length)
                X_ts = np.array(X_ts_list)
    else:
        logger.info("  Time-series features not available or empty, falling back to flat features.")
        X_ts = df[feature_columns].fillna(0).values.reshape(-1, 1, len(feature_columns)) # Reshape flat features as sequence length 1
        sequence_length = 1 # Override sequence_length if falling back to flat features
        num_features_per_window = len(feature_columns)

    # Step 2: Transformer-based feature encoding (no clustering)
    neural_predictions = None
    bot_predictions = None
    if use_transformer:
        logger.info("  Step 2/2: Encoding features with Transformer...")
        try:
            # Scale features
            scaler = StandardScaler()
            # Reshape for scaling: [num_locations * sequence_length, num_features_per_window]
            X_scaled_flat = scaler.fit_transform(X_ts.reshape(-1, num_features_per_window))
            X_scaled = X_scaled_flat.reshape(-1, sequence_length, num_features_per_window)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Prepare fixed features (non-time-series features)
            fixed_feature_cols = [col for col in feature_columns if col != 'time_series_features_present']
            X_fixed = df[fixed_feature_cols].fillna(0).values
            
            # Scale fixed features
            fixed_scaler = StandardScaler()
            X_fixed_scaled = fixed_scaler.fit_transform(X_fixed)
            X_fixed_tensor = torch.FloatTensor(X_fixed_scaled)
            
            # Initialize Transformer classifier
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            classifier = TransformerClassifier(
                ts_input_dim=num_features_per_window,
                fixed_input_dim=len(fixed_feature_cols),
                d_model=128,
                nhead=8,
                num_layers=3,
                enable_reconstruction=enable_self_supervised,
                enable_bot_head=enable_bot_head
            ).to(device)
            
            X_tensor = X_tensor.to(device)
            X_fixed_tensor = X_fixed_tensor.to(device)
            
            # Self-supervised pre-training (if enabled)
            if enable_self_supervised:
                logger.info("    Pre-training Transformer with self-supervised learning...")
                np.random.seed(random_state)
                torch.manual_seed(random_state)
                classifier = train_self_supervised(
                    classifier,
                    X_tensor,
                    X_fixed_tensor,
                    device,
                    epochs=pretrain_epochs,
                    batch_size=pretrain_batch_size,
                    learning_rate=pretrain_learning_rate
                )
            
            # Generate rule-based labels for training (pseudo-labels)
            logger.info("    Generating rule-based labels for classification training...")
            rule_labels = _generate_rule_based_labels(df)

            # Generate SMART bot labels for multi-task learning
            logger.info("    Generating smart 'is_bot' labels for multi-task training...")
            # Use smart labels that capture nuanced bot patterns
            hard_bot_labels, _ = generate_smart_bot_labels(df)
            is_bot_labels = hard_bot_labels  # Use hard labels for training

            
            # Supervised fine-tuning (if enabled)
            if enable_neural_classification:
                logger.info("    Fine-tuning classifier with supervised learning...")
                np.random.seed(random_state)
                torch.manual_seed(random_state)
                
                # Convert labels to tensor
                y_labels = torch.LongTensor(rule_labels).to(device)
                y_is_bot_labels = torch.LongTensor(is_bot_labels).to(device)
                
                # Calculate class weights for imbalanced data
                unique_labels, counts = np.unique(rule_labels, return_counts=True)
                total_samples = len(rule_labels)
                class_weights = torch.FloatTensor([
                    total_samples / (len(unique_labels) * count) if count > 0 else 0
                    for count in counts
                ]).to(device)
                
                classifier = train_supervised_classifier(
                    classifier,
                    X_tensor,
                    X_fixed_tensor,
                    y_labels,
                    device,
                    epochs=finetune_epochs,
                    batch_size=finetune_batch_size,
                    learning_rate=finetune_learning_rate,
                    class_weights=class_weights,
                    y_is_bot=y_is_bot_labels,
                    lambda_bot_loss=lambda_bot_loss
                )
            
            # Forward pass to get predictions
            classifier.eval()
            with torch.no_grad():
                if enable_neural_classification:
                    # Get neural predictions
                    logits, bot_logits, _ = classifier(X_tensor, X_fixed_tensor)
                    _, predicted_classes = torch.max(logits, 1)
                    neural_predictions = predicted_classes.cpu().numpy()
                    if enable_bot_head:
                        _, predicted_bot_classes = torch.max(bot_logits, 1)
                        bot_predictions = predicted_bot_classes.cpu().numpy()
                    else:
                        bot_predictions = None
                else:
                    # Extract embeddings for rule-based classification
                    # Note: Embeddings computed here could be used for future enhancements
                    # (e.g., clustering, similarity search, visualization)
                    _ = classifier.ts_input_projection(X_tensor)
                    _ = classifier.fixed_projection(X_fixed_tensor)
                    neural_predictions = None
                    bot_predictions = None  # No bot predictions if not neural classification
            
            logger.info(f"    Transformer processing completed")
            
        except Exception as e:
            logger.warning(f"    Transformer encoding failed ({e}), using original features for classification")
            use_transformer = False
            neural_predictions = None
            bot_predictions = None
    
    # Create empty cluster_df for compatibility (no clustering anymore)
    cluster_df = pd.DataFrame()
    
    # Ensure we have the required columns with defaults if missing
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = df['anomaly_score'] > 0
    if 'total_downloads' not in df.columns:
        df['total_downloads'] = df['unique_users'] * df['downloads_per_user']
    if 'fraction_latest_year' not in df.columns:
        df['fraction_latest_year'] = 0.0
    if 'spike_ratio' not in df.columns:
        df['spike_ratio'] = 1.0
    if 'is_new_location' not in df.columns:
        df['is_new_location'] = 0
    if 'years_before_latest' not in df.columns:
        df['years_before_latest'] = 0
    if 'working_hours_ratio' not in df.columns:
        df['working_hours_ratio'] = 0.0
    
    # =========================================================================
    # Classification with Proper Category Separation (Phase 6 Fix)
    # =========================================================================
    # 
    # KEY INSIGHT: Bots and Download Hubs have OPPOSITE patterns:
    #   - BOTS: Many users (>1000), low-moderate DL/user (<50), coordinated 
    #   - HUBS: Few users (<100), VERY high DL/user (>500), legitimate mirrors
    #
    # The bot head should NEVER override download hub classifications!
    # =========================================================================
    
    category_map = {0: 'bot', 1: 'download_hub', 2: 'independent_user', 3: 'normal', 4: 'other'}
    
    # =========================================================================
    # Step 1: First, identify CLEAR download hubs BEFORE any neural classification
    # These should NEVER be classified as bots
    # =========================================================================
    logger.info("    Step 1: Identifying download hubs (protected from bot override)...")
    
    download_hub_mask = pd.Series(False, index=df.index)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users', 'total_downloads'):
        # Pattern 1: Very high DL/user with few users (mirrors, institutional)
        hub_pattern_mirror = (
            (df['downloads_per_user'] > 500) & 
            (df['unique_users'] < 100)
        )
        
        # Pattern 2: High total downloads with moderate DL/user and regular working hours
        hub_pattern_institution = (
            (df['total_downloads'] > 100000) & 
            (df['downloads_per_user'] > 50) & 
            (df['downloads_per_user'] < 500) &
            (df['unique_users'] < 500)
        )
        if 'working_hours_ratio' in df.columns:
            hub_pattern_institution = hub_pattern_institution & (df['working_hours_ratio'] > 0.25)
        
        # Pattern 3: Extreme DL/user (automated sync, data mirrors)
        hub_pattern_extreme = (
            (df['downloads_per_user'] > 1000)
        )
        
        download_hub_mask = hub_pattern_mirror | hub_pattern_institution | hub_pattern_extreme
        
        n_hubs_detected = download_hub_mask.sum()
        logger.info(f"      Detected {n_hubs_detected:,} download hub locations (protected)")
        
        # Pre-classify hubs - these will NOT be overridden by bot head
        df.loc[download_hub_mask, 'user_category'] = 'download_hub'
        df.loc[download_hub_mask, 'is_protected_hub'] = True
    
    # =========================================================================
    # Step 2: Apply neural classification to non-hub locations
    # =========================================================================
    if enable_neural_classification and neural_predictions is not None:
        logger.info("    Step 2: Applying neural classification...")
        
        # Only apply neural predictions to non-hub locations
        non_hub_mask = ~download_hub_mask
        
        if enable_confidence_classification and use_transformer:
            logger.info("    Applying confidence-based classification (Phase 6 enhancement)...")
            try:
                classifier.eval()
                with torch.no_grad():
                    logits, bot_logits_conf, _ = classifier(X_tensor, X_fixed_tensor)
                
                # Apply confidence-based classification only to non-hub locations
                df_temp = confidence_based_classification(
                    df.copy(), logits, bot_logits_conf, category_map
                )
                # Copy results only for non-hub locations
                df.loc[non_hub_mask, 'user_category'] = df_temp.loc[non_hub_mask, 'user_category']
                if 'classification_confidence' in df_temp.columns:
                    df.loc[non_hub_mask, 'classification_confidence'] = df_temp.loc[non_hub_mask, 'classification_confidence']
                if 'needs_review' in df_temp.columns:
                    df.loc[non_hub_mask, 'needs_review'] = df_temp.loc[non_hub_mask, 'needs_review']
            except Exception as e:
                logger.warning(f"    Confidence-based classification failed ({e}), using standard predictions")
                for i, pred in enumerate(neural_predictions):
                    if non_hub_mask.iloc[i]:
                        df.iloc[i, df.columns.get_loc('user_category')] = category_map[pred]
        else:
            # Standard neural predictions (only for non-hubs)
            for i, pred in enumerate(neural_predictions):
                if non_hub_mask.iloc[i]:
                    df.iloc[i, df.columns.get_loc('user_category')] = category_map[pred]
        
        # =========================================================================
        # Step 3: Bot head override - BUT ONLY for non-hub locations
        # =========================================================================
        if enable_bot_head and bot_predictions is not None:
            df['is_bot_neural'] = bot_predictions == 1
            
            # Bot head override ONLY for non-hub locations with BOT patterns
            # Key criteria for bots: Many users, low-moderate DL/user
            bot_head_bots = (
                (bot_predictions == 1) & 
                ~download_hub_mask &  # NOT a download hub
                (df['unique_users'] > 100) &  # Bots have many users
                (df['downloads_per_user'] < 100)  # Bots have low-moderate DL/user
            )
            
            if bot_head_bots.any():
                logger.info(f"    Bot head override (filtered): {bot_head_bots.sum():,} locations marked as bots")
                df.loc[bot_head_bots, 'user_category'] = 'bot'
            
        # Apply post-processing overrides for obvious bot patterns
        logger.info("    Applying post-processing bot detection overrides...")
        df = apply_bot_detection_override(df)
        
        # Anomaly-guided classification for edge cases (excluding hubs)
        if 'anomaly_score' in df.columns:
            logger.info("    Applying anomaly-guided classification...")
            high_anomaly = df['anomaly_score'] > 0.25
            moderate_users = (df['unique_users'] > 5000) & (df['unique_users'] < 15000)
            moderate_dl = (df['downloads_per_user'] > 15) & (df['downloads_per_user'] < 50)
            
            suspicious_bot_pattern = (
                high_anomaly & 
                moderate_users & 
                moderate_dl & 
                ~download_hub_mask &  # NOT a download hub
                (df['user_category'] != 'bot')
            )
            
            if suspicious_bot_pattern.any():
                logger.info(f"    Anomaly-guided override: reclassifying {suspicious_bot_pattern.sum()} "
                          f"suspicious locations as bots")
                df.loc[suspicious_bot_pattern, 'user_category'] = 'bot'
                if 'is_bot_neural' in df.columns:
                    df.loc[suspicious_bot_pattern, 'is_bot_neural'] = True
    else:
        logger.info("    Using rule-based classification...")
        df = _apply_rule_based_classification(df)
    
    # =========================================================================
    # Merge stratified pre-filter results back
    # =========================================================================
    if enable_stratified_processing:
        # Ensure pre-filtered classifications are preserved
        df.loc[obvious_bots_mask, 'user_category'] = 'bot'
        df.loc[obvious_legitimate_mask, 'user_category'] = 'independent_user'
    
    # Set boolean flags based on category
    df['is_bot'] = df['user_category'] == 'bot'
    df['is_download_hub'] = df['user_category'] == 'download_hub'
    df['is_independent_user'] = df['user_category'] == 'independent_user'
    df['is_normal_user'] = df['user_category'] == 'normal'

    # Log results
    n_bots = df['is_bot'].sum()
    n_hubs = df['is_download_hub'].sum()
    n_independent = df['is_independent_user'].sum()
    n_normal = df['is_normal_user'].sum()
    n_other = (df['user_category'] == 'other').sum()
    n_needs_review = df['needs_review'].sum() if 'needs_review' in df.columns else 0
    
    method_name = "Neural + Phase 6" if enable_neural_classification else "Rule-based"
    logger.info(f"\n  Final Classification (Transformer + {method_name}):")
    logger.info(f"    Bot locations: {n_bots:,} ({n_bots/len(df)*100:.1f}%)")
    logger.info(f"    Hub locations: {n_hubs:,} ({n_hubs/len(df)*100:.1f}%)")
    logger.info(f"    Independent User locations: {n_independent:,} ({n_independent/len(df)*100:.1f}%)")
    logger.info(f"    Normal locations: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    logger.info(f"    Other/Unclassified locations: {n_other:,} ({n_other/len(df)*100:.1f}%)")
    if n_needs_review > 0:
        logger.info(f"    Locations needing review: {n_needs_review:,} ({n_needs_review/len(df)*100:.1f}%)")
    
    if enable_stratified_processing:
        logger.info(f"\n  Stratified Processing Summary:")
        logger.info(f"    Pre-filtered as bots: {obvious_bots_mask.sum():,}")
        logger.info(f"    Pre-filtered as legitimate: {obvious_legitimate_mask.sum():,}")
        logger.info(f"    Processed by deep learning: {uncertain_mask.sum():,}")
    
    return df, cluster_df


# =============================================================================
# BehavioralEncoder with LSTM for Pattern Discovery
# =============================================================================

class BehavioralEncoder(nn.Module):
    """
    Encoder that learns behavioral embeddings through contrastive learning.
    
    Uses LSTM for temporal sequence modeling and Transformer for attention.
    The encoder learns to place similar behavioral patterns close together
    and different patterns far apart in the embedding space.
    """
    
    def __init__(
        self, 
        ts_input_dim: int,
        fixed_input_dim: int, 
        embedding_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        use_lstm: bool = True
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        
        if use_lstm:
            # LSTM encoder for temporal sequences
            # Bidirectional LSTM to capture both forward and backward patterns
            self.lstm = nn.LSTM(
                input_size=ts_input_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if lstm_num_layers > 1 else 0
            )
            # LSTM output is [batch, seq_len, hidden_dim * 2] (bidirectional)
            lstm_output_dim = lstm_hidden_dim * 2
            
            # Attention mechanism to weight important time steps
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True
            )
            
            # Project LSTM output to d_model
            self.ts_projection = nn.Sequential(
                nn.Linear(lstm_output_dim, d_model),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # Fallback: Transformer encoder (original approach)
            self.ts_projection = nn.Linear(ts_input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fixed features encoder
        self.fixed_encoder = nn.Sequential(
            nn.Linear(fixed_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
        # Projection to embedding space
        self.projection_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, embedding_dim)
        )
        
        # Normalize embeddings
        self.normalize = nn.functional.normalize
    
    def forward(
        self, 
        ts_features: torch.Tensor, 
        fixed_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode location features into behavioral embedding.
        
        Args:
            ts_features: Time-series features [batch, seq_len, ts_dim]
            fixed_features: Fixed features [batch, fixed_dim]
            
        Returns:
            Normalized embeddings [batch, embedding_dim]
        """
        if self.use_lstm:
            # LSTM encoding
            lstm_out, (hidden, _) = self.lstm(ts_features)
            # lstm_out: [batch, seq_len, hidden_dim * 2]
            # hidden: [num_layers * num_directions, batch, hidden_dim]
            # For bidirectional: hidden[0]=forward_layer0, hidden[1]=backward_layer0, etc.
            
            # Concatenate forward and backward hidden states from last layer
            num_layers = self.lstm.num_layers
            if num_layers == 1:
                # Single layer: hidden[0]=forward, hidden[1]=backward
                forward_hidden = hidden[0]  # [batch, hidden_dim]
                backward_hidden = hidden[1]  # [batch, hidden_dim]
            else:
                # Multiple layers: last forward is at index (num_layers-1)*2, last backward at (num_layers-1)*2+1
                forward_hidden = hidden[(num_layers - 1) * 2]  # [batch, hidden_dim]
                backward_hidden = hidden[(num_layers - 1) * 2 + 1]  # [batch, hidden_dim]
            
            # Concatenate to get full hidden state
            combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=-1)  # [batch, hidden_dim * 2]
            
            # Apply attention to weight important time steps
            # Use the combined hidden state as query
            query = combined_hidden.unsqueeze(1)  # [batch, 1, hidden_dim * 2]
            attended_out, _ = self.attention(query, lstm_out, lstm_out)
            # attended_out: [batch, 1, hidden_dim * 2]
            
            # Project to d_model
            ts_encoded = self.ts_projection(attended_out.squeeze(1))  # [batch, d_model]
        else:
            # Transformer encoding (fallback)
            ts_proj = self.ts_projection(ts_features)
            ts_encoded_seq = self.transformer(ts_proj)
            ts_encoded = ts_encoded_seq.mean(dim=1)  # Global average pooling
        
        # Encode fixed features
        fixed_encoded = self.fixed_encoder(fixed_features)
        
        # Combine and project
        combined = torch.cat([ts_encoded, fixed_encoded], dim=-1)
        embedding = self.projection_head(combined)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


# =============================================================================
# Pattern Analysis Functions
# =============================================================================

def analyze_patterns(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_columns: List[str]
) -> Dict[int, Dict]:
    """
    Analyze and describe discovered patterns.
    
    For each cluster, compute statistics and generate interpretable descriptions.
    
    Args:
        df: DataFrame with features
        cluster_labels: Cluster assignments
        feature_columns: Features to analyze
        
    Returns:
        Dictionary with pattern analysis for each cluster
    """
    logger.info("Analyzing discovered patterns...")
    
    df = df.copy()
    df['pattern_cluster'] = cluster_labels
    
    patterns = {}
    
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = df['pattern_cluster'] == cluster_id
        cluster_df = df[cluster_mask]
        
        if len(cluster_df) == 0:
            continue
        
        # Compute statistics
        stats = {
            'count': len(cluster_df),
            'mean_users': cluster_df['unique_users'].mean(),
            'mean_dl_per_user': cluster_df['downloads_per_user'].mean(),
            'mean_total_downloads': cluster_df['total_downloads'].mean(),
        }
        
        # Add behavioral features if available
        if 'regularity_score' in cluster_df.columns:
            stats['mean_regularity'] = cluster_df['regularity_score'].mean()
        if 'weekend_ratio' in cluster_df.columns:
            stats['mean_weekend_ratio'] = cluster_df['weekend_ratio'].mean()
        if 'file_diversity_ratio' in cluster_df.columns:
            stats['mean_file_diversity'] = cluster_df['file_diversity_ratio'].mean()
        if 'working_hours_ratio' in cluster_df.columns:
            stats['mean_working_hours'] = cluster_df['working_hours_ratio'].mean()
        
        # Generate description based on characteristics
        description = _generate_pattern_description(stats, cluster_id)
        stats['description'] = description
        stats['suggested_name'] = _suggest_pattern_name(stats)
        
        patterns[cluster_id] = stats
        
        logger.info(f"  Pattern {cluster_id}: {stats['suggested_name']}")
        logger.info(f"    Count: {stats['count']:,}, Avg users: {stats['mean_users']:.1f}, "
                   f"Avg DL/user: {stats['mean_dl_per_user']:.1f}")
    
    return patterns


def _generate_pattern_description(stats: Dict, cluster_id: int) -> str:
    """Generate human-readable description of a pattern."""
    parts = []
    
    # User scale
    mean_users = stats.get('mean_users', 0)
    if mean_users > 1000:
        parts.append("large-scale")
    elif mean_users > 100:
        parts.append("medium-scale")
    elif mean_users > 10:
        parts.append("small-group")
    else:
        parts.append("individual")
    
    # Download intensity
    mean_dl = stats.get('mean_dl_per_user', 0)
    if mean_dl > 100:
        parts.append("high-volume")
    elif mean_dl > 20:
        parts.append("moderate-volume")
    else:
        parts.append("low-volume")
    
    # Regularity
    regularity = stats.get('mean_regularity', 0)
    if regularity > 1.0:
        parts.append("mechanical/automated")
    elif regularity > 0.5:
        parts.append("semi-regular")
    else:
        parts.append("irregular/organic")
    
    # Working hours
    working_hours = stats.get('mean_working_hours', 0.5)
    if working_hours > 0.6:
        parts.append("business-hours")
    elif working_hours < 0.3:
        parts.append("off-hours")
    
    return ", ".join(parts)


def _suggest_pattern_name(stats: Dict) -> str:
    """Suggest a name for the pattern based on characteristics."""
    mean_users = stats.get('mean_users', 0)
    mean_dl = stats.get('mean_dl_per_user', 0)
    regularity = stats.get('mean_regularity', 0)
    working_hours = stats.get('mean_working_hours', 0.5)
    file_diversity = stats.get('mean_file_diversity', 0.5)
    
    # Pipeline/CI pattern: regular, weekday, low file diversity
    if regularity > 0.8 and working_hours > 0.5 and file_diversity < 0.3:
        return "Pipeline/CI"
    
    # Research group: working hours, diverse files, moderate users
    if working_hours > 0.5 and file_diversity > 0.5 and 10 < mean_users < 500:
        return "Research_Group"
    
    # Automated sync: very regular, any time, consistent
    if regularity > 1.0:
        return "Automated_Sync"
    
    # Bulk download: high DL/user, few sessions
    if mean_dl > 50 and mean_users < 50:
        return "Bulk_Download"
    
    # Coordinated activity: many users, similar behavior
    if mean_users > 500 and mean_dl < 20:
        return "Coordinated_Activity"
    
    # Organic/normal: irregular, diverse, working hours
    if regularity < 0.3 and file_diversity > 0.3:
        return "Organic_Normal"
    
    return "Mixed_Pattern"


# =============================================================================
# Coordinated Activity Refinement
# =============================================================================

def refine_coordinated_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refine Coordinated_Activity pattern into sub-categories using file patterns.
    
    Based on the insight that:
    - CI/CD and pipelines download same files repeatedly (low file diversity)
    - Bot-like behavior explores many files (high file diversity)
    - Legitimate coordination has moderate file diversity
    
    Args:
        df: DataFrame with discovered_pattern column
        
    Returns:
        DataFrame with refined Coordinated_Activity sub-categories
    """
    logger.info("\n" + "=" * 60)
    logger.info("REFINING COORDINATED_ACTIVITY CLASSIFICATION")
    logger.info("=" * 60)
    
    coordinated_mask = df['discovered_pattern'] == 'Coordinated_Activity'
    n_coordinated = coordinated_mask.sum()
    
    if n_coordinated == 0:
        logger.info("No Coordinated_Activity locations to refine.")
        return df
    
    logger.info(f"Refining {n_coordinated:,} Coordinated_Activity locations...")
    
    # Check if we have file diversity feature
    if 'file_diversity_ratio' not in df.columns:
        logger.warning("  file_diversity_ratio not available, skipping refinement")
        return df
    
    coordinated_df = df[coordinated_mask].copy()
    
    # Category 1: CI/CD_Pipeline - Low file diversity (<0.3)
    # These download same files repeatedly (testing, building)
    cicd_mask = coordinated_df['file_diversity_ratio'] < 0.3
    n_cicd = cicd_mask.sum()
    
    if n_cicd > 0:
        cicd_indices = coordinated_df[cicd_mask].index
        df.loc[cicd_indices, 'discovered_pattern'] = 'CI_CD_Pipeline'
        logger.info(f"  CI/CD_Pipeline (low file diversity): {n_cicd:,} locations")
    
    # Category 2: Bot_Like_Coordination - High file diversity + bot patterns
    # Criteria:
    #   - High file diversity (>0.6) OR
    #   - (High users >2K AND Low DL/user <10) OR
    #   - (Moderate users 1K-5K AND Very low DL/user <5)
    bot_like_mask = (
        (coordinated_df['file_diversity_ratio'] > 0.6) |
        ((coordinated_df['unique_users'] > 2000) & (coordinated_df['downloads_per_user'] < 10)) |
        ((coordinated_df['unique_users'] >= 1000) & (coordinated_df['unique_users'] <= 5000) &
         (coordinated_df['downloads_per_user'] < 5))
    ) & ~cicd_mask  # Exclude CI/CD locations
    
    n_bot_like = bot_like_mask.sum()
    
    if n_bot_like > 0:
        bot_like_indices = coordinated_df[bot_like_mask].index
        df.loc[bot_like_indices, 'discovered_pattern'] = 'Bot_Like_Coordination'
        logger.info(f"  Bot_Like_Coordination: {n_bot_like:,} locations")
        logger.info(f"    - High file diversity: {(coordinated_df[bot_like_mask]['file_diversity_ratio'] > 0.6).sum():,}")
        logger.info(f"    - High users + low DL/user: {((coordinated_df[bot_like_mask]['unique_users'] > 2000) & (coordinated_df[bot_like_mask]['downloads_per_user'] < 10)).sum():,}")
        logger.info(f"    - Moderate users + very low DL/user: {((coordinated_df[bot_like_mask]['unique_users'] >= 1000) & (coordinated_df[bot_like_mask]['unique_users'] <= 5000) & (coordinated_df[bot_like_mask]['downloads_per_user'] < 5)).sum():,}")
    
    # Category 3: Legitimate_Coordination - Remaining locations
    # These have moderate file diversity and reasonable user patterns
    legitimate_mask = ~cicd_mask & ~bot_like_mask
    n_legitimate = legitimate_mask.sum()
    
    if n_legitimate > 0:
        legitimate_indices = coordinated_df[legitimate_mask].index
        df.loc[legitimate_indices, 'discovered_pattern'] = 'Legitimate_Coordination'
        logger.info(f"  Legitimate_Coordination: {n_legitimate:,} locations")
    
    # Summary
    logger.info("\nRefined Coordinated_Activity breakdown:")
    refined_coordinated = df[df['discovered_pattern'].isin([
        'CI_CD_Pipeline', 'Bot_Like_Coordination', 'Legitimate_Coordination'
    ])]
    
    for pattern in ['CI_CD_Pipeline', 'Bot_Like_Coordination', 'Legitimate_Coordination']:
        pattern_df = df[df['discovered_pattern'] == pattern]
        if len(pattern_df) > 0:
            pct = len(pattern_df) / n_coordinated * 100
            dl_pct = pattern_df['total_downloads'].sum() / coordinated_df['total_downloads'].sum() * 100
            logger.info(f"  {pattern:25s}: {len(pattern_df):6,} ({pct:5.1f}%) | {pattern_df['total_downloads'].sum():15,.0f} downloads ({dl_pct:5.1f}%)")
    
    return df


def _generate_rule_based_labels(df: pd.DataFrame) -> np.ndarray:
    """Generate rule-based labels for training (0=BOT, 1=DOWNLOAD_HUB, 2=INDEPENDENT_USER, 3=NORMAL, 4=OTHER)."""
    labels = np.full(len(df), 3)  # Default to NORMAL
    
    # Ensure required columns exist
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = df.get('anomaly_score', pd.Series([0] * len(df))) > 0
    if 'total_downloads' not in df.columns:
        df['total_downloads'] = df.get('unique_users', pd.Series([0] * len(df))) * df.get('downloads_per_user', pd.Series([0] * len(df)))
    if 'working_hours_ratio' not in df.columns:
        df['working_hours_ratio'] = 0.0
    
    # 1. Bot classification
    bot_mask = (
        df['is_anomaly'] &
        (
            ((df['downloads_per_user'] < 12) & (df['unique_users'] > 7000)) |
            ((df['unique_users'] > 25000) & (df['downloads_per_user'] < 100) & (df['downloads_per_user'] > 10)) |
            ((df['unique_users'] > 15000) & (df['downloads_per_user'] < 80) & (df['downloads_per_user'] > 8))
        )
    )
    labels[bot_mask] = 0  # BOT
    
    # 2. Download Hub classification
    hub_mask = (
        df['is_anomaly'] &
        (
            (df['downloads_per_user'] > 500) |
            ((df['total_downloads'] > 150000) & (df['downloads_per_user'] > 50) & (df['working_hours_ratio'] > 0.25))
        )
    )
    labels[hub_mask] = 1  # DOWNLOAD_HUB
    
    # 3. Independent User classification
    independent_mask = (
        (~df['is_anomaly'] | (df.get('anomaly_score', pd.Series([0] * len(df))) < 0.1)) &
        (df['unique_users'] <= 5) &
        (df['downloads_per_user'] <= 3)
    )
    labels[independent_mask] = 2  # INDEPENDENT_USER
    
    # 4. Other (anomalous but doesn't match other patterns)
    other_mask = df['is_anomaly'] & (labels == 3)  # Still NORMAL but anomalous
    labels[other_mask] = 4  # OTHER
    
    return labels


def _apply_rule_based_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based classification (fallback method)."""
    df['user_category'] = 'normal'
    
    # 1. Bot classification
    bot_mask = (
        df['is_anomaly'] &
        (
            ((df['downloads_per_user'] < 12) & (df['unique_users'] > 7000)) |
            ((df['unique_users'] > 25000) & (df['downloads_per_user'] < 100) & (df['downloads_per_user'] > 10)) |
            ((df['unique_users'] > 15000) & (df['downloads_per_user'] < 80) & (df['downloads_per_user'] > 8))
        )
    )
    df.loc[bot_mask, 'user_category'] = 'bot'
    
    # 2. Download Hub classification
    hub_mask = (
        df['is_anomaly'] &
        (
            (df['downloads_per_user'] > 500) |
            ((df['total_downloads'] > 150000) & (df['downloads_per_user'] > 50) & (df['working_hours_ratio'] > 0.25))
        )
    )
    df.loc[hub_mask, 'user_category'] = 'download_hub'
    
    # 3. Independent User classification
    independent_mask = (
        (~df['is_anomaly'] | (df.get('anomaly_score', pd.Series([0] * len(df))) < 0.1)) &
        (df['unique_users'] <= 5) &
        (df['downloads_per_user'] <= 3)
    )
    df.loc[independent_mask, 'user_category'] = 'independent_user'
    
    # 4. Other
    other_mask = df['is_anomaly'] & (df['user_category'] == 'normal')
    df.loc[other_mask, 'user_category'] = 'other'
    
    return df