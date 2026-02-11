"""Organic Variational Autoencoder for learned anomaly detection.

Trains a VAE on high-confidence organic locations to learn the manifold
of normal download behavior. Reconstruction error serves as a continuous
anomaly score -- no hand-coded thresholds needed.

Also provides a wrapper around Deep Isolation Forest (DeepOD) for
non-linear anomaly detection that captures feature interactions.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OrganicVAE(nn.Module):
    """VAE trained exclusively on organic download behavior."""

    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reconstruction_error(self, x):
        """Per-sample reconstruction error (anomaly score)."""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            return ((x - recon) ** 2).mean(dim=1)

    def latent_features(self, x):
        """Extract latent representation for downstream use."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


def _vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_organic_vae(features: np.ndarray, weights: np.ndarray = None,
                      latent_dim: int = 16, epochs: int = 100,
                      batch_size: int = 256, lr: float = 1e-3) -> tuple:
    """Train a VAE on organic location features.

    Args:
        features: (n_samples, n_features) array of organic features
        weights: Optional per-sample confidence weights
        latent_dim: Dimensionality of latent space
        epochs: Training epochs
        batch_size: Mini-batch size
        lr: Learning rate

    Returns:
        (model, scaler) tuple -- trained VAE and fitted StandardScaler
    """
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    X_tensor = torch.FloatTensor(X)

    if weights is not None:
        W = torch.FloatTensor(weights)
    else:
        W = torch.ones(len(X))

    input_dim = X.shape[1]
    model = OrganicVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Simple training loop (no GPU needed for 48K locations)
    model.train()
    n = len(X_tensor)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_tensor[idx]
            batch_w = W[idx]

            recon, mu, logvar = model(batch_x)

            # Weighted reconstruction loss
            recon_err = ((batch_x - recon) ** 2).mean(dim=1)
            weighted_recon = (recon_err * batch_w).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Beta-annealing: start with low KL weight, increase
            beta = min(1.0, epoch / 20.0)
            loss = weighted_recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 20 == 0:
            logger.info(f"  VAE epoch {epoch}: loss={avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"  VAE early stopping at epoch {epoch}")
                break

    model.eval()
    logger.info(f"  VAE trained: {input_dim} features → {latent_dim}d latent, "
                f"final loss={best_loss:.4f}")
    return model, scaler


def compute_vae_anomaly_scores(model: OrganicVAE, scaler: StandardScaler,
                               features: np.ndarray) -> tuple:
    """Compute VAE anomaly scores for all locations.

    Args:
        model: Trained OrganicVAE
        scaler: Fitted StandardScaler from training
        features: (n_samples, n_features) array

    Returns:
        (anomaly_scores, latent_features) tuple:
          anomaly_scores: (n_samples,) reconstruction error per location
          latent_features: (n_samples, latent_dim) latent representations
    """
    X = scaler.transform(features)
    X_tensor = torch.FloatTensor(X)

    model.eval()
    with torch.no_grad():
        scores = model.reconstruction_error(X_tensor)
        latent = model.latent_features(X_tensor)

    # Use tolist() to avoid numpy/pytorch compatibility issues
    return np.array(scores.tolist()), np.array(latent.tolist())


def train_deep_isolation_forest(features: np.ndarray,
                                n_ensemble: int = 50,
                                n_estimators: int = 100) -> tuple:
    """Train Deep Isolation Forest for non-linear anomaly detection.

    Falls back to standard IsolationForest if DeepOD is unavailable.

    Args:
        features: (n_samples, n_features) array
        n_ensemble: Number of neural representation ensembles
        n_estimators: Trees per ensemble

    Returns:
        (anomaly_scores, model) tuple
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    try:
        from deepod.models.tabular import DeepIsolationForest
        logger.info("  Using Deep Isolation Forest (DeepOD)")
        model = DeepIsolationForest(
            n_ensemble=n_ensemble,
            n_estimators=n_estimators,
            random_state=42,
            device='cpu',
        )
        model.fit(X)
        scores = model.decision_function(X)
        # Normalize to 0-1 (higher = more anomalous)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores, (model, scaler)

    except ImportError:
        logger.warning("  DeepOD not available, falling back to standard Isolation Forest")
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(
            contamination=0.15,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)
        scores = -model.decision_function(X)  # Flip: higher = more anomalous
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores, (model, scaler)
