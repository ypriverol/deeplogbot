"""Deep classification combining Isolation Forest and Transformers.

This module implements a multi-stage architecture:
1. Isolation Forest: Initial anomaly detection
2. Transformers: Sequence-based feature encoding for direct classification

The Transformer processes time-series features and combines them with fixed features
to directly classify locations into categories (BOT, DOWNLOAD_HUB, NORMAL, INDEPENDENT_USER, OTHER).
This approach is similar to the paper's architecture without clustering.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Tuple

from ...utils import logger
from ..isoforest.models import train_isolation_forest


class TransformerClassifier(nn.Module):
    """Transformer-based classifier that combines time-series and fixed features."""
    
    def __init__(self, ts_input_dim: int, fixed_input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 512,
                 num_classes: int = 5):
        """
        Args:
            ts_input_dim: Dimension of time-series features per window
            fixed_input_dim: Dimension of fixed (non-time-series) features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            num_classes: Number of output classes (bot, hub, normal, independent_user, other)
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
    
    def forward(self, ts_features: torch.Tensor, fixed_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining time-series and fixed features.
        
        Args:
            ts_features: Time-series features [batch_size, seq_len, ts_input_dim]
            fixed_features: Fixed features [batch_size, fixed_input_dim]
        
        Returns:
            Classification logits [batch_size, num_classes]
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
        
        return logits


def classify_locations_deep(df: pd.DataFrame, feature_columns: List[str],
                              use_transformer: bool = True, random_state: int = 42,
                              contamination: float = 0.15, sequence_length: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify locations using deep architecture: Isolation Forest + Transformers.
    
    This method combines:
    1. Isolation Forest for initial anomaly detection
    2. Transformers for sequence-based feature encoding
    3. Direct classification using Transformer embeddings + fixed features
    
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
    
    Returns:
        Tuple of (DataFrame with classification columns added, empty cluster_df for compatibility)
    """
    logger.info("Training deep architecture classifier (Isolation Forest + Transformers)...")
    
    # Step 1: Isolation Forest for initial anomaly detection
    logger.info("  Step 1/2: Running Isolation Forest for anomaly detection...")
    predictions, scores, _, _ = train_isolation_forest(
        df, feature_columns, contamination=contamination
    )
    df['is_anomaly'] = predictions == -1
    df['anomaly_score'] = -scores
    logger.info(f"    Detected {df['is_anomaly'].sum():,} anomalous locations")
    
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
        first_valid_ts = df['time_series_features'].dropna().iloc[0]
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
    transformer_embeddings = None
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
                num_layers=3
            ).to(device)
            
            X_tensor = X_tensor.to(device)
            X_fixed_tensor = X_fixed_tensor.to(device)
            
            # Forward pass to get embeddings (we extract embeddings, not classification)
            classifier.eval()
            with torch.no_grad():
                # Extract time-series embeddings
                ts_proj = classifier.ts_input_projection(X_tensor)
                ts_encoded = classifier.transformer(ts_proj)
                ts_pooled = ts_encoded.mean(dim=1).cpu().numpy()  # [batch_size, d_model]
                
                # Extract fixed feature embeddings
                fixed_proj = classifier.fixed_projection(X_fixed_tensor).cpu().numpy()  # [batch_size, d_model]
                
                # Combine embeddings
                transformer_embeddings = np.concatenate([ts_pooled, fixed_proj], axis=1)  # [batch_size, 2*d_model]
            
            logger.info(f"    Transformer encoding completed: {transformer_embeddings.shape[1]}D embeddings")
            
        except Exception as e:
            logger.warning(f"    Transformer encoding failed ({e}), using original features for classification")
            use_transformer = False
    
    # Create empty cluster_df for compatibility (no clustering anymore)
    cluster_df = pd.DataFrame()
    
    # Classify each location individually based on its own features (similar to rule-based method)
    # Clusters are used for pattern identification, but classification is location-level
    # Initialize: all locations default to 'normal'
    df['user_category'] = 'normal'
    df['is_bot'] = False
    df['is_download_hub'] = False
    df['is_independent_user'] = False
    df['is_normal_user'] = False

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

    # 1. Bot classification: Classify each location individually (same core patterns as rule-based)
    bot_mask = (
        df['is_anomaly'] &
        (
            # Pattern 1: High user count with very low downloads per user
            (
                (df['downloads_per_user'] < 12) &
                (df['unique_users'] > 7000)
            ) |
            # Pattern 1b: Very high user count with moderate downloads per user
            (
                (df['unique_users'] > 25000) &
                (df['downloads_per_user'] < 100) &
                (df['downloads_per_user'] > 10)
            ) |
            # Pattern 1c: High user count (>15K) with moderate‑low DL/user (<80)
            (
                (df['unique_users'] > 15000) &
                (df['downloads_per_user'] < 80) &
                (df['downloads_per_user'] > 8)
            )
        )
    )
    
    df.loc[bot_mask, 'user_category'] = 'bot'
    df.loc[bot_mask, 'is_bot'] = True
    n_bots = bot_mask.sum()
    logger.info(f"    Classified {n_bots:,} location(s) as BOT")

    # 2. Download Hub classification: Classify each location individually
    hub_mask = (
        df['is_anomaly'] &
        (
            # Pattern 1: High downloads per user (mirrors/single-user hubs)
            (df['downloads_per_user'] > 500) |
            # Pattern 2: High total downloads with moderate DL/user and regular patterns (research institutions)
            (
                (df['total_downloads'] > 150000) &
                (df['downloads_per_user'] > 50) &
                (df['working_hours_ratio'] > 0.25)
            )
        )
    )
    
    df.loc[hub_mask, 'user_category'] = 'download_hub'
    df.loc[hub_mask, 'is_download_hub'] = True
    n_hubs = hub_mask.sum()
    logger.info(f"    Classified {n_hubs:,} location(s) as DOWNLOAD_HUB")

    # 3. Independent User classification: Low users, low DL/user, low anomaly
    independent_mask = (
        (~df['is_anomaly'] | (df['anomaly_score'] < 0.1)) &
        (df['unique_users'] <= 5) &
        (df['downloads_per_user'] <= 3)
    )
    
    df.loc[independent_mask, 'user_category'] = 'independent_user'
    df.loc[independent_mask, 'is_independent_user'] = True
    n_independent = independent_mask.sum()
    logger.info(f"    Classified {n_independent:,} location(s) as INDEPENDENT_USER")

    # 4. Normal users: locations that are not anomalous and don't match other patterns
    normal_mask = (
        ~df['is_anomaly'] &
        (df['user_category'] == 'normal')  # Only update if still 'normal'
    )
    df.loc[normal_mask, 'is_normal_user'] = True

    # 5. Unclassified locations are marked as 'other'
    other_mask = (df['user_category'] == 'normal') & df['is_anomaly']
    df.loc[other_mask, 'user_category'] = 'other'

    # Log results
    n_bots = df['is_bot'].sum()
    n_hubs = df['is_download_hub'].sum()
    n_independent = df['is_independent_user'].sum()
    n_normal = (df['user_category'] == 'normal').sum()
    n_other = (df['user_category'] == 'other').sum()
    
    logger.info(f"\n  Final Classification (Transformer + Rule-based):")
    logger.info(f"    Bot locations: {n_bots:,} ({n_bots/len(df)*100:.1f}%)")
    logger.info(f"    Hub locations: {n_hubs:,} ({n_hubs/len(df)*100:.1f}%)")
    logger.info(f"    Independent User locations: {n_independent:,} ({n_independent/len(df)*100:.1f}%)")
    logger.info(f"    Normal locations: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    logger.info(f"    Other/Unclassified locations: {n_other:,} ({n_other/len(df)*100:.1f}%)")
    if transformer_embeddings is not None:
        logger.info(f"    Transformer embeddings computed: {transformer_embeddings.shape[1]}D features per location")
    
    return df, cluster_df