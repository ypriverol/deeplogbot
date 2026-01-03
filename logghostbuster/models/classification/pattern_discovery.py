"""Pattern Discovery Architecture for Download Behavior Classification.

This module implements a new approach to classification:
1. Clear ground truth rules for obvious cases (BOT, HUB, INDIVIDUAL)
2. Deep learning for pattern discovery on uncertain cases
3. Contrastive learning to discover behavioral clusters
4. Interpretable pattern descriptions

The goal is NOT to force data into predefined categories, but to let the
model discover natural behavioral patterns like:
- Pipeline/CI patterns (same files, regular intervals)
- Research group patterns (diverse files, working hours)
- Automated sync patterns (daily/weekly regularity)
- Organic/normal patterns (human-like irregularity)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from ...utils import logger


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
    
    temporal_query = f"""
    WITH download_intervals AS (
        SELECT 
            {schema.location_field} as geo_location,
            EPOCH(CAST({schema.timestamp_field} AS TIMESTAMP) - 
                  LAG(CAST({schema.timestamp_field} AS TIMESTAMP)) 
                  OVER (PARTITION BY {schema.location_field} ORDER BY {schema.timestamp_field})) as interval_seconds
        FROM read_parquet('{escaped_path}')
        WHERE {schema.location_field} IS NOT NULL
        AND {schema.timestamp_field} IS NOT NULL
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
# Contrastive Learning for Pattern Discovery
# =============================================================================

class BehavioralEncoder(nn.Module):
    """
    Encoder that learns behavioral embeddings through contrastive learning.
    
    The encoder learns to place similar behavioral patterns close together
    and different patterns far apart in the embedding space.
    """
    
    def __init__(
        self, 
        ts_input_dim: int,
        fixed_input_dim: int, 
        embedding_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        
        # Time-series encoder (Transformer)
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
        # Encode time-series
        ts_proj = self.ts_projection(ts_features)
        ts_encoded = self.transformer(ts_proj)
        ts_pooled = ts_encoded.mean(dim=1)  # Global average pooling
        
        # Encode fixed features
        fixed_encoded = self.fixed_encoder(fixed_features)
        
        # Combine and project
        combined = torch.cat([ts_pooled, fixed_encoded], dim=-1)
        embedding = self.projection_head(combined)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Used for self-supervised contrastive learning where we want
    similar patterns to be close and different patterns to be far.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two views.
        
        Args:
            z1: Embeddings from view 1 [batch, embedding_dim]
            z2: Embeddings from view 2 [batch, embedding_dim]
            
        Returns:
            Scalar loss
        """
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        z = torch.cat([z1, z2], dim=0)  # [2*batch, embedding_dim]
        similarity = torch.mm(z, z.t()) / self.temperature  # [2*batch, 2*batch]
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])  # [2*batch]
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity = similarity.masked_fill(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


def augment_features(
    ts_features: torch.Tensor, 
    fixed_features: torch.Tensor,
    noise_scale: float = 0.1,
    mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create augmented views of features for contrastive learning.
    
    Augmentations:
    - Add Gaussian noise
    - Randomly mask time steps
    - Randomly shuffle time order (slight)
    
    Args:
        ts_features: Time-series features
        fixed_features: Fixed features
        noise_scale: Scale of Gaussian noise
        mask_prob: Probability of masking each time step
        
    Returns:
        Augmented (ts_features, fixed_features)
    """
    # Add noise to fixed features
    fixed_aug = fixed_features + torch.randn_like(fixed_features) * noise_scale
    
    # Augment time-series: mask + noise
    ts_aug = ts_features.clone()
    
    # Random masking
    mask = torch.rand(ts_features.shape[:-1], device=ts_features.device) < mask_prob
    ts_aug[mask] = 0
    
    # Add noise
    ts_aug = ts_aug + torch.randn_like(ts_aug) * noise_scale
    
    return ts_aug, fixed_aug


# =============================================================================
# Pattern Discovery Pipeline
# =============================================================================

def train_pattern_encoder(
    df: pd.DataFrame,
    ts_features: np.ndarray,
    fixed_feature_columns: List[str],
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    embedding_dim: int = 64
) -> Tuple[BehavioralEncoder, np.ndarray]:
    """
    Train the behavioral encoder using contrastive learning.
    
    No labels needed - the model learns to group similar patterns.
    
    Args:
        df: DataFrame with features
        ts_features: Time-series features [n_samples, seq_len, ts_dim]
        fixed_feature_columns: List of fixed feature column names
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        embedding_dim: Dimension of learned embeddings
        
    Returns:
        Trained encoder and embeddings for all locations
    """
    logger.info("Training behavioral encoder with contrastive learning...")
    
    # Prepare data
    fixed_features = df[fixed_feature_columns].fillna(0).values
    
    # Scale features
    ts_scaler = StandardScaler()
    ts_flat = ts_features.reshape(-1, ts_features.shape[-1])
    ts_scaled = ts_scaler.fit_transform(ts_flat).reshape(ts_features.shape)
    
    fixed_scaler = StandardScaler()
    fixed_scaled = fixed_scaler.fit_transform(fixed_features)
    
    # Convert to tensors
    ts_tensor = torch.FloatTensor(ts_scaled)
    fixed_tensor = torch.FloatTensor(fixed_scaled)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = BehavioralEncoder(
        ts_input_dim=ts_features.shape[-1],
        fixed_input_dim=len(fixed_feature_columns),
        embedding_dim=embedding_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(temperature=0.07)
    
    # Training loop
    encoder.train()
    n_samples = len(df)
    
    for epoch in range(epochs):
        # Shuffle data
        perm = torch.randperm(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            
            ts_batch = ts_tensor[idx].to(device)
            fixed_batch = fixed_tensor[idx].to(device)
            
            # Create two augmented views
            ts_aug1, fixed_aug1 = augment_features(ts_batch, fixed_batch)
            ts_aug2, fixed_aug2 = augment_features(ts_batch, fixed_batch)
            
            # Get embeddings for both views
            z1 = encoder(ts_aug1, fixed_aug1)
            z2 = encoder(ts_aug2, fixed_aug2)
            
            # Compute contrastive loss
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            logger.info(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Get final embeddings
    encoder.eval()
    with torch.no_grad():
        all_embeddings = encoder(
            ts_tensor.to(device), 
            fixed_tensor.to(device)
        ).cpu().numpy()
    
    logger.info(f"  Training complete. Embedding shape: {all_embeddings.shape}")
    
    return encoder, all_embeddings


def discover_patterns(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Discover behavioral patterns through clustering in embedding space.
    
    Uses HDBSCAN which can find clusters of varying density and
    automatically determines the number of clusters.
    
    Args:
        embeddings: Learned embeddings [n_samples, embedding_dim]
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points
        
    Returns:
        Cluster labels and pattern descriptions
    """
    logger.info("Discovering behavioral patterns through clustering...")
    
    # Cluster embeddings
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Count clusters
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    
    logger.info(f"  Discovered {n_clusters} distinct behavioral patterns")
    logger.info(f"  Noise points (no clear pattern): {n_noise:,}")
    
    # Log cluster sizes
    for label in sorted(unique_labels):
        if label != -1:
            count = (cluster_labels == label).sum()
            logger.info(f"    Pattern {label}: {count:,} locations")
    
    # Create placeholder descriptions (will be filled by analyze_patterns)
    pattern_descriptions = {i: f"Pattern_{i}" for i in unique_labels if i != -1}
    pattern_descriptions[-1] = "No_Clear_Pattern"
    
    return cluster_labels, pattern_descriptions


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
# Main Classification Pipeline
# =============================================================================

def classify_with_pattern_discovery(
    df: pd.DataFrame,
    feature_columns: List[str],
    input_parquet: Optional[str] = None,
    conn=None,
    enable_behavioral_features: bool = True,
    embedding_dim: int = 64,
    contrastive_epochs: int = 50,
    min_cluster_size: int = 50
) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
    """
    Main classification pipeline using pattern discovery.
    
    Pipeline:
    1. Apply ground truth rules (BOT, HUB, INDIVIDUAL)
    2. Extract behavioral features for uncertain cases
    3. Train contrastive encoder to learn behavioral embeddings
    4. Discover patterns through clustering
    5. Analyze and describe patterns
    
    Args:
        df: DataFrame with basic features
        feature_columns: List of feature columns for encoding
        input_parquet: Path to parquet file for additional feature extraction
        conn: DuckDB connection
        enable_behavioral_features: Whether to extract behavioral features
        embedding_dim: Dimension of learned embeddings
        contrastive_epochs: Epochs for contrastive learning
        min_cluster_size: Minimum cluster size for HDBSCAN
        
    Returns:
        DataFrame with classifications and pattern dictionary
    """
    logger.info("=" * 60)
    logger.info("PATTERN DISCOVERY CLASSIFICATION")
    logger.info("=" * 60)
    
    # Step 1: Apply ground truth rules
    logger.info("\nStep 1: Applying ground truth rules...")
    df = apply_ground_truth_rules(df)
    
    # Get uncertain locations for pattern discovery
    uncertain_mask = df['ground_truth_category'] == GroundTruthCategory.UNCERTAIN.value
    n_uncertain = uncertain_mask.sum()
    
    if n_uncertain == 0:
        logger.info("All locations classified by ground truth rules.")
        df['discovered_pattern'] = df['ground_truth_category']
        return df, {}
    
    logger.info(f"\nStep 2: Pattern discovery for {n_uncertain:,} uncertain locations...")
    
    # Step 2: Extract behavioral features (if enabled and possible)
    if enable_behavioral_features and input_parquet is not None and conn is not None:
        logger.info("  Extracting behavioral features...")
        df = extract_behavioral_features(df, input_parquet, conn)
    
    # Step 3: Prepare features for encoding
    # Use time-series features if available
    if 'time_series_features' in df.columns:
        ts_list = df['time_series_features'].tolist()
        # Handle None values
        max_seq_len = max(len(ts) if isinstance(ts, list) else 0 for ts in ts_list)
        ts_dim = len(ts_list[0][0]) if ts_list and isinstance(ts_list[0], list) and len(ts_list[0]) > 0 else 12
        
        ts_features = []
        for ts in ts_list:
            if isinstance(ts, list) and len(ts) > 0:
                if len(ts) < max_seq_len:
                    # Pad
                    padding = [[0.0] * ts_dim] * (max_seq_len - len(ts))
                    ts_features.append(padding + ts)
                else:
                    ts_features.append(ts[-max_seq_len:])
            else:
                ts_features.append([[0.0] * ts_dim] * max_seq_len)
        
        ts_features = np.array(ts_features)
    else:
        # Fallback to fixed features as sequence length 1
        ts_features = df[feature_columns].fillna(0).values.reshape(-1, 1, len(feature_columns))
    
    # Behavioral feature columns
    behavioral_columns = [
        'regularity_score', 'interval_cv', 'weekend_ratio', 'unique_days_of_week',
        'file_diversity_ratio', 'file_concentration', 'num_sessions', 'downloads_per_session',
        'is_mechanical', 'is_weekday_biased', 'is_single_file', 'is_bursty'
    ]
    
    # Use available columns
    fixed_columns = feature_columns + [c for c in behavioral_columns if c in df.columns]
    fixed_columns = list(set(fixed_columns))  # Remove duplicates
    
    # Step 4: Train contrastive encoder (only on uncertain locations)
    df_uncertain = df[uncertain_mask].copy()
    ts_uncertain = ts_features[uncertain_mask]
    
    encoder, embeddings = train_pattern_encoder(
        df_uncertain,
        ts_uncertain,
        fixed_columns,
        epochs=contrastive_epochs,
        embedding_dim=embedding_dim
    )
    
    # Step 5: Discover patterns through clustering
    cluster_labels, pattern_names = discover_patterns(
        embeddings,
        min_cluster_size=min_cluster_size
    )
    
    # Step 6: Analyze patterns
    patterns = analyze_patterns(df_uncertain, cluster_labels, fixed_columns)
    
    # Step 7: Assign discovered patterns back to dataframe
    df['discovered_pattern'] = df['ground_truth_category']
    df.loc[uncertain_mask, 'discovered_pattern'] = [
        patterns.get(label, {}).get('suggested_name', 'Unknown')
        for label in cluster_labels
    ]
    df.loc[uncertain_mask, 'pattern_cluster'] = cluster_labels
    
    # Store embeddings for visualization
    df['embedding'] = None
    embedding_list = embeddings.tolist()
    uncertain_indices = df.index[uncertain_mask].tolist()
    for i, idx in enumerate(uncertain_indices):
        df.at[idx, 'embedding'] = embedding_list[i]
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for category in df['discovered_pattern'].unique():
        count = (df['discovered_pattern'] == category).sum()
        pct = count / len(df) * 100
        logger.info(f"  {category}: {count:,} ({pct:.1f}%)")
    
    return df, patterns


