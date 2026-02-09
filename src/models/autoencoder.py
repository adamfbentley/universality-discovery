"""
Convolutional Autoencoder for Surface Growth Dynamics
======================================================

Learns unsupervised representations of height fields h(x,t).
No hand-crafted features - discovers what matters for universality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class ConvEncoder(nn.Module):
    """
    Convolutional encoder: height field -> latent vector.
    
    Input: (batch, 1, width, time_steps) - 2D "image" of surface evolution
    Output: (batch, latent_dim) - compressed representation
    """
    
    def __init__(self, width: int = 128, time_steps: int = 200, latent_dim: int = 32):
        super().__init__()
        self.width = width
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate flattened size after convolutions
        self._conv_out_size = self._get_conv_output_size()
        
        # Fully connected to latent space
        self.fc = nn.Linear(self._conv_out_size, latent_dim)
    
    def _get_conv_output_size(self) -> int:
        """Calculate size of flattened conv output."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.width, self.time_steps)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """
    Convolutional decoder: latent vector -> reconstructed height field.
    """
    
    def __init__(self, width: int = 128, time_steps: int = 200, latent_dim: int = 32):
        super().__init__()
        self.width = width
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        
        # Calculate intermediate sizes
        self.init_width = width // 16
        self.init_time = time_steps // 16
        
        # Fully connected from latent to initial conv size
        self.fc = nn.Linear(latent_dim, 256 * self.init_width * self.init_time)
        
        # Transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.init_width, self.init_time)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = self.deconv4(x)  # No activation on output
        
        # Ensure output matches expected dimensions
        x = F.interpolate(x, size=(self.width, self.time_steps), mode='bilinear', align_corners=False)
        return x


class SurfaceAutoencoder(nn.Module):
    """
    Complete autoencoder for surface height fields.
    
    Usage:
        model = SurfaceAutoencoder(width=128, time_steps=200, latent_dim=32)
        
        # Training
        z = model.encode(surfaces)
        reconstructed = model.decode(z)
        loss = model.reconstruction_loss(surfaces, reconstructed)
        
        # Anomaly detection
        anomaly_score = model.anomaly_score(test_surface)
    """
    
    def __init__(self, width: int = 128, time_steps: int = 200, latent_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(width, time_steps, latent_dim)
        self.decoder = ConvDecoder(width, time_steps, latent_dim)
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode surface to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to surface."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def reconstruction_loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        """Mean squared error reconstruction loss."""
        return F.mse_loss(x_reconstructed, x)
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score as reconstruction error.
        Higher score = more anomalous (different from training distribution).
        """
        with torch.no_grad():
            x_reconstructed, _ = self.forward(x)
            # Per-sample MSE
            mse = ((x - x_reconstructed) ** 2).mean(dim=(1, 2, 3))
            return mse


class VariationalEncoder(nn.Module):
    """
    Variational encoder: outputs mean and log-variance for latent distribution.
    """
    
    def __init__(self, width: int = 128, time_steps: int = 200, latent_dim: int = 32):
        super().__init__()
        self.base_encoder = ConvEncoder(width, time_steps, latent_dim * 2)
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base_encoder(x)
        mu = h[:, :self.latent_dim]
        log_var = h[:, self.latent_dim:]
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class SurfaceVAE(nn.Module):
    """
    Variational Autoencoder for surface height fields.
    
    Adds KL divergence regularization for smoother latent space.
    Better for interpolation and sampling.
    """
    
    def __init__(self, width: int = 128, time_steps: int = 200, latent_dim: int = 32, beta: float = 1.0):
        super().__init__()
        self.encoder = VariationalEncoder(width, time_steps, latent_dim)
        self.decoder = ConvDecoder(width, time_steps, latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta  # β-VAE weight for disentanglement
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent sample to surface."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass with reparameterization."""
        mu, log_var = self.encode(x)
        z = self.encoder.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, mu, log_var
    
    def loss(self, x: torch.Tensor, x_reconstructed: torch.Tensor, 
             mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss = reconstruction + β * KL divergence
        
        Returns total loss, reconstruction loss, KL loss
        """
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score combining reconstruction error and KL divergence.
        """
        with torch.no_grad():
            x_reconstructed, z, mu, log_var = self.forward(x)
            recon_error = ((x - x_reconstructed) ** 2).mean(dim=(1, 2, 3))
            kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean(dim=1)
            return recon_error + self.beta * kl_div


def prepare_surface_batch(surfaces: np.ndarray) -> torch.Tensor:
    """
    Convert numpy surface array to torch tensor for model input.
    
    Args:
        surfaces: Array of shape (n_samples, width, time_steps) or (width, time_steps)
    
    Returns:
        Tensor of shape (n_samples, 1, width, time_steps)
    """
    if surfaces.ndim == 2:
        surfaces = surfaces[np.newaxis, ...]
    
    # Add channel dimension
    surfaces = surfaces[:, np.newaxis, :, :]
    
    # Normalize per-sample
    surfaces = (surfaces - surfaces.mean(axis=(2, 3), keepdims=True)) / (surfaces.std(axis=(2, 3), keepdims=True) + 1e-8)
    
    return torch.from_numpy(surfaces).float()
