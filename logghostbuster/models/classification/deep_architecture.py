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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple

from ...utils import logger
from ..isoforest.models import train_isolation_forest


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

# Override thresholds
OVERRIDE_THRESHOLDS = {
    'OVERRIDE1_DL_PER_USER': 50,
    'OVERRIDE1_MAX_USERS': 100,
    'OVERRIDE2_DL_PER_USER': 30,
    'OVERRIDE2_MAX_USERS': 50,
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


def generate_smart_bot_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Generate intelligent bot labels based on multiple signals.
    
    Args:
        df: DataFrame with location features
        
    Returns:
        Tuple of (hard_labels, soft_labels) as numpy arrays
    """
    # Initialize bot score (0-1)
    bot_score = np.zeros(len(df))
    
    # Signal 1: Few users + extreme DL/user (HIGHEST PRIORITY)
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        few_users_extreme_dl = (
            (df['downloads_per_user'] > BOT_THRESHOLDS['EXTREME_DL_PER_USER']) & 
            (df['unique_users'] < BOT_THRESHOLDS['FEW_USERS'])
        )
        bot_score[few_users_extreme_dl] += BOT_SIGNAL_WEIGHTS['FEW_USERS_EXTREME_DL']
        
        very_few_users_high_dl = (
            (df['downloads_per_user'] > BOT_THRESHOLDS['HIGH_DL_PER_USER']) &
            (df['downloads_per_user'] <= BOT_THRESHOLDS['EXTREME_DL_PER_USER']) &
            (df['unique_users'] < BOT_THRESHOLDS['VERY_FEW_USERS'])
        )
        bot_score[very_few_users_high_dl] += BOT_SIGNAL_WEIGHTS['VERY_FEW_USERS_HIGH_DL']
    
    # Signal 2: Moderate users + high DL/user
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        moderate_users_high_dl = (
            (df['downloads_per_user'] > BOT_THRESHOLDS['HIGH_DL_PER_USER']) & 
            (df['unique_users'] >= BOT_THRESHOLDS['FEW_USERS']) &
            (df['unique_users'] < 500)
        )
        bot_score[moderate_users_high_dl] += BOT_SIGNAL_WEIGHTS['MODERATE_USERS_HIGH_DL']
    
    # Signal 3: High anomaly score
    if 'anomaly_score' in df.columns:
        high_anomaly = df['anomaly_score'] > BOT_THRESHOLDS['HIGH_ANOMALY_SCORE']
        very_high_anomaly = df['anomaly_score'] > BOT_THRESHOLDS['VERY_HIGH_ANOMALY_SCORE']
        bot_score[high_anomaly] += BOT_SIGNAL_WEIGHTS['HIGH_ANOMALY']
        bot_score[very_high_anomaly] += BOT_SIGNAL_WEIGHTS['VERY_HIGH_ANOMALY']
    
    # Signal 4: Original rule-based classification
    if 'user_category' in df.columns:
        rule_bot = df['user_category'] == 'bot'
        rule_hub = df['user_category'] == 'download_hub'
        bot_score[rule_bot] += BOT_SIGNAL_WEIGHTS['RULE_BOT']
        bot_score[rule_hub] += BOT_SIGNAL_WEIGHTS['RULE_HUB']
    
    # Signal 5: Extreme DL/user (statistical outlier)
    if 'downloads_per_user' in df.columns:
        dl_user_values = df['downloads_per_user'].values
        dl_user_median = np.median(dl_user_values)
        dl_user_std = np.std(dl_user_values)
        
        if dl_user_std > EPSILON:
            dl_user_zscore = (dl_user_values - dl_user_median) / dl_user_std
            extreme_dl = dl_user_zscore > BOT_THRESHOLDS['EXTREME_ZSCORE']
            very_extreme_dl = dl_user_zscore > BOT_THRESHOLDS['VERY_EXTREME_ZSCORE']
            bot_score[extreme_dl] += BOT_SIGNAL_WEIGHTS['EXTREME_DL_ZSCORE']
            bot_score[very_extreme_dl] += BOT_SIGNAL_WEIGHTS['VERY_EXTREME_DL_ZSCORE']
    
    # Signal 6: Low working hours ratio with high activity
    if _has_required_columns(df, 'working_hours_ratio', 'total_downloads'):
        non_working_high_activity = (
            (df['working_hours_ratio'] < BOT_THRESHOLDS['LOW_WORKING_HOURS_RATIO']) &
            (df['total_downloads'] > BOT_THRESHOLDS['MIN_TOTAL_DOWNLOADS'])
        )
        bot_score[non_working_high_activity] += BOT_SIGNAL_WEIGHTS['NON_WORKING_HIGH_ACTIVITY']
    
    # Signal 7: Low hourly entropy
    if 'hourly_entropy' in df.columns:
        low_entropy = df['hourly_entropy'] < df['hourly_entropy'].quantile(BOT_THRESHOLDS['LOW_ENTROPY_QUANTILE'])
        bot_score[low_entropy] += BOT_SIGNAL_WEIGHTS['LOW_ENTROPY']
    
    # Normalize to [0, 1]
    bot_score = np.clip(bot_score, 0, 1)
    
    # Create both soft and hard labels
    soft_labels = bot_score
    
    # Adaptive threshold
    percentile_threshold = np.percentile(bot_score, BOT_THRESHOLDS['ADAPTIVE_THRESHOLD_PERCENTILE'])
    fixed_threshold = BOT_THRESHOLDS['FIXED_THRESHOLD']
    threshold = min(percentile_threshold, fixed_threshold)
    
    hard_labels = (bot_score >= threshold).astype(float)
    
    logger.info(f"    Smart bot label statistics:")
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
    """Apply post-processing overrides for obvious bot patterns."""
    
    override_count = 0
    
    # Override 1: Few users with extreme DL/user
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        obvious_bots = (
            (df['downloads_per_user'] > OVERRIDE_THRESHOLDS['OVERRIDE1_DL_PER_USER']) & 
            (df['unique_users'] < OVERRIDE_THRESHOLDS['OVERRIDE1_MAX_USERS'])
        )
        
        if 'is_bot_neural' in df.columns:
            obvious_bots = obvious_bots & (df['is_bot_neural'] == False)
        
        if obvious_bots.any():
            logger.info(f"    Applying bot override for {obvious_bots.sum()} obvious patterns "
                       f"(>{OVERRIDE_THRESHOLDS['OVERRIDE1_DL_PER_USER']} DL/user, "
                       f"<{OVERRIDE_THRESHOLDS['OVERRIDE1_MAX_USERS']} users)")
            df.loc[obvious_bots, 'is_bot_neural'] = True
            if 'user_category' in df.columns:
                df.loc[obvious_bots, 'user_category'] = 'bot'
            override_count += obvious_bots.sum()
    
    # Override 2: Very few users with high DL/user
    if _has_required_columns(df, 'downloads_per_user', 'unique_users'):
        very_obvious_bots = (
            (df['downloads_per_user'] > OVERRIDE_THRESHOLDS['OVERRIDE2_DL_PER_USER']) &
            (df['unique_users'] < OVERRIDE_THRESHOLDS['OVERRIDE2_MAX_USERS'])
        )
        
        if 'is_bot_neural' in df.columns:
            very_obvious_bots = very_obvious_bots & (df['is_bot_neural'] == False)
        
        if very_obvious_bots.any():
            logger.info(f"    Applying bot override for {very_obvious_bots.sum()} very obvious patterns "
                      f"(>{OVERRIDE_THRESHOLDS['OVERRIDE2_DL_PER_USER']} DL/user, "
                      f"<{OVERRIDE_THRESHOLDS['OVERRIDE2_MAX_USERS']} users)")
            df.loc[very_obvious_bots, 'is_bot_neural'] = True
            if 'user_category' in df.columns:
                df.loc[very_obvious_bots, 'user_category'] = 'bot'
            override_count += very_obvious_bots.sum()
    
    logger.info(f"    Total bot overrides applied: {override_count}")
    
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
                              lambda_bot_loss: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        enable_self_supervised: Whether to enable self-supervised pre-training (default: True)
        pretrain_epochs: Number of epochs for self-supervised pre-training (default: 20)
        pretrain_batch_size: Batch size for pre-training (default: 256)
        pretrain_learning_rate: Learning rate for pre-training (default: 1e-4)
        enable_neural_classification: Whether to use neural classification instead of rules (default: True)
        finetune_epochs: Number of epochs for supervised fine-tuning (default: 30)
        finetune_batch_size: Batch size for fine-tuning (default: 256)
        finetune_learning_rate: Learning rate for fine-tuning (default: 1e-4)
    
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
                    ts_proj = classifier.ts_input_projection(X_tensor)
                    ts_encoded = classifier.transformer(ts_proj)
                    ts_pooled = ts_encoded.mean(dim=1).cpu().numpy()
                    fixed_proj = classifier.fixed_projection(X_fixed_tensor).cpu().numpy()
                    # Embeddings available if needed for future use
                    # transformer_embeddings = np.concatenate([ts_pooled, fixed_proj], axis=1)
                    neural_predictions = None
                    bot_predictions = None # No bot predictions if not neural classification
            
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
    
    # Classify using neural predictions or rule-based
    if enable_neural_classification and neural_predictions is not None:
        logger.info("    Using neural classification predictions...")
        # Map neural predictions to categories
        category_map = {0: 'bot', 1: 'download_hub', 2: 'independent_user', 3: 'normal', 4: 'other'}
        df['user_category'] = [category_map[pred] for pred in neural_predictions]
        if enable_bot_head and bot_predictions is not None:
            df['is_bot_neural'] = bot_predictions == 1 # Add a new column for the explicit bot prediction
            
        # Apply post-processing overrides for obvious bot patterns
        logger.info("    Applying post-processing bot detection overrides...")
        df = apply_bot_detection_override(df)
    else:
        logger.info("    Using rule-based classification...")
        # Fallback to rule-based classification
        df = _apply_rule_based_classification(df)
    
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
    
    method_name = "Neural" if enable_neural_classification else "Rule-based"
    logger.info(f"\n  Final Classification (Transformer + {method_name}):")
    logger.info(f"    Bot locations: {n_bots:,} ({n_bots/len(df)*100:.1f}%)")
    logger.info(f"    Hub locations: {n_hubs:,} ({n_hubs/len(df)*100:.1f}%)")
    logger.info(f"    Independent User locations: {n_independent:,} ({n_independent/len(df)*100:.1f}%)")
    logger.info(f"    Normal locations: {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    logger.info(f"    Other/Unclassified locations: {n_other:,} ({n_other/len(df)*100:.1f}%)")
    
    return df, cluster_df


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