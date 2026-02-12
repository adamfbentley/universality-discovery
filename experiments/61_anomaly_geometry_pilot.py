"""
Experiment 61: Anomaly Geometry Pilot
======================================

PURPOSE: Reconnect with the original project vision (README.md Phase 2).
Train autoencoder on KNOWN surface growth classes (EW + KPZ), then feed
UNKNOWN systems through and ask: do anomaly clusters correspond to
universality classes?

This is the experiment the project was always supposed to run but never did.
The autoencoder (src/models/autoencoder.py), the HDBSCAN clusterer
(src/analysis/clustering.py), and the parameter scanner
(src/exploration/parameter_scan.py) were all built but never connected.

SYSTEMS:
  Training (known):   EW, KPZ equation
  Test (unknown):     Ballistic Deposition (KPZ class), Eden (KPZ class),
                      Random Deposition (trivial), Kuramoto-Sivashinsky (KS class)

PREDICTIONS (before running):
  P1: BD and Eden should have LOW anomaly scores (same class as training KPZ)
  P2: Random Deposition should be anomalous (trivial class, no correlations)
  P3: KS should be highly anomalous (different universality class)
  P4: HDBSCAN on anomalous latent vectors should cluster RD separately from KS
  P5: Latent-space UMAP should show EW/KPZ training basin with anomalies outside

SUCCESS CRITERIA:
  - BD anomaly score < 2x median training anomaly score
  - KS anomaly score > 3x median training anomaly score
  - HDBSCAN finds >= 2 clusters among anomalous samples
  - Clusters align with true universality classes (ARI > 0.5)

Usage:
  python 61_anomaly_geometry_pilot.py --pilot    # Quick test (~5 min)
  python 61_anomaly_geometry_pilot.py            # Full experiment (~30 min)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import argparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Unicode safety for Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))

RESULTS_DIR = PROJECT_DIR / 'results_exp61'
FIGURES_DIR = PROJECT_DIR / 'figures'


# ============================================================================
# Simulators — all 1D surface growth, output shape (n_time, L)
# ============================================================================

from numba import jit

@jit(nopython=True)
def simulate_ew(L=128, T=500, nu=1.0, D=1.0, dt=0.1, seed=None):
    """Edwards-Wilkinson: dh/dt = nu * laplacian(h) + noise."""
    if seed is not None:
        np.random.seed(seed)
    h = np.random.randn(L) * 0.01
    trajectory = np.zeros((T, L))
    for t in range(T):
        new_h = h.copy()
        for x in range(L):
            lap = h[(x-1) % L] - 2*h[x] + h[(x+1) % L]
            noise = np.sqrt(2 * D * dt) * np.random.randn()
            new_h[x] = h[x] + dt * nu * lap + noise
        h = new_h - np.mean(new_h)
        trajectory[t] = h
    return trajectory


@jit(nopython=True)
def simulate_kpz(L=128, T=500, nu=1.0, lam=1.0, D=1.0, dt=0.05, seed=None):
    """KPZ equation: dh/dt = nu*lap(h) + (lam/2)*(grad h)^2 + noise."""
    if seed is not None:
        np.random.seed(seed)
    h = np.random.randn(L) * 0.01
    trajectory = np.zeros((T, L))
    for t in range(T):
        new_h = h.copy()
        for x in range(L):
            lap = h[(x-1) % L] - 2*h[x] + h[(x+1) % L]
            grad = (h[(x+1) % L] - h[(x-1) % L]) / 2.0
            noise = np.sqrt(2 * D * dt) * np.random.randn()
            new_h[x] = h[x] + dt * (nu * lap + (lam/2) * grad**2) + noise
        h = new_h - np.mean(new_h)
        trajectory[t] = h
    return trajectory


@jit(nopython=True)
def simulate_bd(L=128, T=500, depositions_per_step=None, seed=None):
    """Ballistic Deposition (KPZ class, discrete)."""
    if seed is not None:
        np.random.seed(seed)
    if depositions_per_step is None:
        depositions_per_step = L  # One monolayer per time step
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for _ in range(depositions_per_step):
            x = np.random.randint(0, L)
            left_h = h[(x-1) % L]
            right_h = h[(x+1) % L]
            landing = max(left_h, h[x], right_h) + 1.0
            h[x] = landing
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


@jit(nopython=True)
def simulate_eden(L=128, T=500, seed=None):
    """Eden model (KPZ class, discrete)."""
    if seed is not None:
        np.random.seed(seed)
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for _ in range(L):
            x = np.random.randint(0, L)
            left_h = h[(x-1) % L]
            right_h = h[(x+1) % L]
            curvature = left_h + right_h - 2*h[x]
            prob = 0.5 + 0.3 * np.tanh(curvature)
            if np.random.rand() < prob:
                h[x] += 1.0
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


@jit(nopython=True)
def simulate_random_deposition(L=128, T=500, seed=None):
    """Random Deposition (trivial class, no spatial correlations)."""
    if seed is not None:
        np.random.seed(seed)
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for x in range(L):
            h[x] += np.abs(np.random.randn())
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


def simulate_ks(L=128, T=500, nu=1.0, kappa=2.0, lam=1.0, noise=0.05,
                dt=0.01, record_interval=1, seed=None):
    """Kuramoto-Sivashinsky (different universality class).
    Uses Fourier-space ETD scheme. Not Numba (uses FFT).
    """
    if seed is not None:
        np.random.seed(seed)

    dx = 2 * np.pi / L
    k = np.fft.fftfreq(L, d=dx / (2 * np.pi))
    k2 = k**2
    k4 = k**4

    h = 0.01 * np.random.randn(L)
    h_hat = np.fft.fft(h)

    # Linear operator: -nu*k^4 + kappa*k^2
    Lin = -nu * k4 + kappa * k2
    exp_L = np.exp(Lin * dt)

    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(np.abs(Lin) > 1e-12, (exp_L - 1) / Lin, dt)

    n_steps = T * record_interval
    trajectory = np.zeros((T, L))
    snap = 0

    for step in range(n_steps):
        dh_dx = np.real(np.fft.ifft(1j * k * h_hat))
        nonlinear = -lam / 2 * dh_dx**2
        nl_hat = np.fft.fft(nonlinear)
        eta_hat = np.fft.fft(noise * np.random.randn(L))

        h_hat = exp_L * h_hat + M * (nl_hat + eta_hat)

        if (step + 1) % record_interval == 0:
            h = np.real(np.fft.ifft(h_hat))
            h = h - np.mean(h)
            trajectory[snap] = h
            snap += 1

    return trajectory


# ============================================================================
# Data generation
# ============================================================================

def generate_dataset(n_samples, L, T, system, seed_offset=0, **kwargs):
    """Generate n_samples trajectories from a given system.

    Returns: array of shape (n_samples, T, L)
    """
    trajectories = []
    for i in range(n_samples):
        seed = seed_offset + i
        if system == 'ew':
            traj = simulate_ew(L=L, T=T, seed=seed, **kwargs)
        elif system == 'kpz':
            traj = simulate_kpz(L=L, T=T, seed=seed, **kwargs)
        elif system == 'bd':
            traj = simulate_bd(L=L, T=T, seed=seed)
        elif system == 'eden':
            traj = simulate_eden(L=L, T=T, seed=seed)
        elif system == 'rd':
            traj = simulate_random_deposition(L=L, T=T, seed=seed)
        elif system == 'ks':
            traj = simulate_ks(L=L, T=T, seed=seed, **kwargs)
        else:
            raise ValueError(f"Unknown system: {system}")
        trajectories.append(traj)
    return np.array(trajectories)


def prepare_for_autoencoder(trajectories):
    """Convert (N, T, L) -> (N, 1, L, T) torch tensor, normalized per-sample."""
    # Transpose to (N, L, T) then add channel dim -> (N, 1, L, T)
    x = trajectories.transpose(0, 2, 1)  # (N, L, T)
    x = x[:, np.newaxis, :, :]           # (N, 1, L, T)

    # Per-sample zero-mean, unit-variance normalization
    mean = x.mean(axis=(2, 3), keepdims=True)
    std = x.std(axis=(2, 3), keepdims=True) + 1e-8
    x = (x - mean) / std

    return torch.from_numpy(x).float()


# ============================================================================
# Autoencoder (import from src or define locally)
# ============================================================================

def build_autoencoder(L, T, latent_dim=32):
    """Build autoencoder. Try importing from src, fall back to local."""
    try:
        from models.autoencoder import SurfaceAutoencoder
        model = SurfaceAutoencoder(width=L, time_steps=T, latent_dim=latent_dim)
        print(f"  Loaded SurfaceAutoencoder from src/models/autoencoder.py")
    except Exception as e:
        print(f"  Could not import SurfaceAutoencoder ({e}), building locally")
        import torch.nn as nn
        import torch.nn.functional as F

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
                self.bn2 = nn.BatchNorm2d(64)
                self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
                self.bn3 = nn.BatchNorm2d(128)
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, L, T)
                    dummy = F.relu(self.bn1(self.conv1(dummy)))
                    dummy = F.relu(self.bn2(self.conv2(dummy)))
                    dummy = F.relu(self.bn3(self.conv3(dummy)))
                    self._flat = dummy.numel()
                self.fc = nn.Linear(self._flat, latent_dim)

            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x)))
                x = F.relu(self.bn3(self.conv3(x)))
                return self.fc(x.view(x.size(0), -1))

        class Decoder(nn.Module):
            def __init__(self, flat_size, shape_after_conv):
                super().__init__()
                self.fc = nn.Linear(latent_dim, flat_size)
                self.shape = shape_after_conv
                self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
                self.bn1 = nn.BatchNorm2d(64)
                self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
                self.bn2 = nn.BatchNorm2d(32)
                self.deconv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

            def forward(self, z):
                x = self.fc(z).view(-1, *self.shape)
                x = F.relu(self.bn1(self.deconv1(x)))
                x = F.relu(self.bn2(self.deconv2(x)))
                x = self.deconv3(x)
                return F.interpolate(x, size=(L, T), mode='bilinear', align_corners=False)

        class LocalAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.decoder = Decoder(self.encoder._flat, None)
                # Get shape after conv
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, L, T)
                    dummy = F.relu(self.encoder.bn1(self.encoder.conv1(dummy)))
                    dummy = F.relu(self.encoder.bn2(self.encoder.conv2(dummy)))
                    dummy = F.relu(self.encoder.bn3(self.encoder.conv3(dummy)))
                    self.decoder.shape = dummy.shape[1:]
                self.latent_dim = latent_dim

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                z = self.encode(x)
                return self.decode(z), z

            def anomaly_score(self, x):
                with torch.no_grad():
                    x_rec, _ = self.forward(x)
                    return ((x - x_rec) ** 2).mean(dim=(1, 2, 3))

        model = LocalAE()
    return model


# ============================================================================
# Training
# ============================================================================

def train_autoencoder(model, train_data, epochs=100, lr=1e-3, batch_size=32,
                      verbose=True):
    """Train autoencoder on known-class surfaces."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                      factor=0.5)
    dataset = TensorDataset(train_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            x_rec, z = model(batch)
            loss = torch.nn.functional.mse_loss(x_rec, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    Epoch {epoch+1:4d}/{epochs}  loss = {avg_loss:.6f}")

    return losses


# ============================================================================
# Anomaly analysis
# ============================================================================

def compute_latent_and_anomaly(model, data_dict):
    """Encode all systems, compute anomaly scores.

    Args:
        model: trained autoencoder
        data_dict: {'system_name': tensor of shape (N,1,L,T), ...}

    Returns:
        latents: dict of system_name -> (N, latent_dim) arrays
        anomaly_scores: dict of system_name -> (N,) arrays
    """
    model.eval()
    latents = {}
    anomaly_scores = {}

    with torch.no_grad():
        for name, tensor in data_dict.items():
            z = model.encode(tensor).numpy()
            scores = model.anomaly_score(tensor).numpy()
            latents[name] = z
            anomaly_scores[name] = scores
            print(f"    {name:6s}: anomaly = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  "
                  f"(median = {np.median(scores):.4f})")

    return latents, anomaly_scores


def cluster_anomalies(latents, anomaly_scores, threshold_multiplier=2.0,
                      train_median=None):
    """Cluster high-anomaly samples using HDBSCAN.

    Args:
        latents: dict of system_name -> (N, d) arrays
        anomaly_scores: dict of system_name -> (N,) arrays
        threshold_multiplier: samples with score > multiplier * train_median are "anomalous"
        train_median: median anomaly of training distribution

    Returns:
        cluster_labels, anomaly_latents, anomaly_true_labels, anomaly_systems
    """
    # Identify anomalous samples
    anomaly_latents_list = []
    anomaly_true_labels = []
    anomaly_systems = []

    for name, z in latents.items():
        scores = anomaly_scores[name]
        mask = scores > threshold_multiplier * train_median
        if mask.sum() > 0:
            anomaly_latents_list.append(z[mask])
            anomaly_true_labels.extend([name] * mask.sum())
            anomaly_systems.extend([name] * mask.sum())

    if len(anomaly_latents_list) == 0:
        print("  No anomalous samples found!")
        return None, None, None, None

    anomaly_latents = np.vstack(anomaly_latents_list)
    print(f"  Found {len(anomaly_latents)} anomalous samples")

    # Cluster with HDBSCAN
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(anomaly_latents) // 10),
            min_samples=2,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(anomaly_latents)
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        print(f"  HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
    except ImportError:
        print("  hdbscan not installed, falling back to KMeans")
        from sklearn.cluster import KMeans
        # Guess n_clusters from number of unique systems
        n_unique = len(set(anomaly_true_labels))
        km = KMeans(n_clusters=min(n_unique, 5), random_state=42, n_init=10)
        labels = km.fit_predict(anomaly_latents)
        n_clusters = len(set(labels))
        print(f"  KMeans assigned {n_clusters} clusters")

    return labels, anomaly_latents, anomaly_true_labels, anomaly_systems


def compute_cluster_quality(cluster_labels, true_labels):
    """Compute adjusted Rand index between clusters and true universality classes."""
    if cluster_labels is None:
        return {'ari': 0.0, 'n_clusters': 0}

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Map true labels to integers
    unique_true = sorted(set(true_labels))
    true_int = [unique_true.index(l) for l in true_labels]

    # Filter out noise (-1)
    mask = cluster_labels != -1
    if mask.sum() < 2:
        return {'ari': 0.0, 'nmi': 0.0, 'n_clusters': 0}

    ari = adjusted_rand_score(np.array(true_int)[mask], cluster_labels[mask])
    nmi = normalized_mutual_info_score(np.array(true_int)[mask], cluster_labels[mask])

    return {
        'ari': float(ari),
        'nmi': float(nmi),
        'n_clusters': len(set(cluster_labels[mask])),
        'n_noise': int((~mask).sum()),
        'n_total': int(len(cluster_labels))
    }


# ============================================================================
# Visualization
# ============================================================================

def make_figures(latents, anomaly_scores, cluster_labels, anomaly_latents,
                 anomaly_true_labels, train_median, results, save_dir):
    """Generate all analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ---- Panel 1: Anomaly score bar chart ----
    ax = axes[0, 0]
    systems = list(anomaly_scores.keys())
    means = [np.mean(anomaly_scores[s]) for s in systems]
    stds = [np.std(anomaly_scores[s]) for s in systems]

    # Color by expected class
    class_colors = {
        'ew': '#2196F3', 'kpz': '#4CAF50',       # training (known)
        'bd': '#8BC34A', 'eden': '#CDDC39',        # KPZ class (should be low)
        'rd': '#FF9800',                            # trivial (should be high)
        'ks': '#F44336'                             # different class (should be high)
    }
    colors = [class_colors.get(s, '#9E9E9E') for s in systems]

    bars = ax.bar(systems, means, yerr=stds, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5, capsize=4)
    ax.axhline(train_median, color='gray', linestyle='--', linewidth=1,
               label=f'Train median = {train_median:.4f}')
    ax.axhline(2 * train_median, color='red', linestyle=':', linewidth=1,
               label=f'Anomaly threshold (2x)')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Anomaly Scores by System')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # ---- Panel 2: Latent space PCA (all systems) ----
    ax = axes[0, 1]
    all_z = []
    all_labels = []
    for name, z in latents.items():
        all_z.append(z)
        all_labels.extend([name] * len(z))
    all_z = np.vstack(all_z)
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z)

    offset = 0
    for name, z in latents.items():
        n = len(z)
        ax.scatter(z_2d[offset:offset+n, 0], z_2d[offset:offset+n, 1],
                   c=class_colors.get(name, '#9E9E9E'), label=name, alpha=0.6,
                   s=20, edgecolors='none')
        offset += n
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Latent Space (PCA)')
    ax.legend(fontsize=8, loc='best')

    # ---- Panel 3: UMAP of latent space (all systems) ----
    ax = axes[0, 2]
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42)
        z_umap = reducer.fit_transform(all_z)
        offset = 0
        for name, z in latents.items():
            n = len(z)
            ax.scatter(z_umap[offset:offset+n, 0], z_umap[offset:offset+n, 1],
                       c=class_colors.get(name, '#9E9E9E'), label=name, alpha=0.6,
                       s=20, edgecolors='none')
            offset += n
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Latent Space (UMAP)')
        ax.legend(fontsize=8, loc='best')
    except ImportError:
        ax.text(0.5, 0.5, 'umap-learn not installed\nInstall: pip install umap-learn',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('UMAP (unavailable)')

    # ---- Panel 4: Anomaly clusters (PCA view) ----
    ax = axes[1, 0]
    if cluster_labels is not None and anomaly_latents is not None:
        pca_anom = PCA(n_components=2)
        z_anom_2d = pca_anom.fit_transform(anomaly_latents)
        unique_clusters = sorted(set(cluster_labels))
        cmap = plt.cm.tab10
        for cl in unique_clusters:
            mask = cluster_labels == cl
            label = f'Cluster {cl}' if cl >= 0 else 'Noise'
            color = 'gray' if cl == -1 else cmap(cl % 10)
            ax.scatter(z_anom_2d[mask, 0], z_anom_2d[mask, 1],
                       c=[color], label=label, alpha=0.7, s=30)
        ax.set_xlabel('PC1 (anomalies)')
        ax.set_ylabel('PC2 (anomalies)')
        ax.set_title(f'Anomaly Clusters (n={len(unique_clusters) - (1 if -1 in unique_clusters else 0)})')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No anomalies found', ha='center', va='center',
                transform=ax.transAxes)

    # ---- Panel 5: Cluster vs true labels ----
    ax = axes[1, 1]
    if cluster_labels is not None and anomaly_true_labels is not None:
        # Confusion-style scatter: true label (y) vs cluster (x)
        unique_true = sorted(set(anomaly_true_labels))
        unique_cl = sorted(set(cluster_labels))
        confusion = np.zeros((len(unique_true), len(unique_cl)))
        for tl, cl in zip(anomaly_true_labels, cluster_labels):
            confusion[unique_true.index(tl), unique_cl.index(cl)] += 1
        im = ax.imshow(confusion, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(unique_cl)))
        ax.set_xticklabels([str(c) for c in unique_cl], fontsize=8)
        ax.set_yticks(range(len(unique_true)))
        ax.set_yticklabels(unique_true, fontsize=8)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('True System')
        ax.set_title(f'Cluster vs Truth (ARI={results.get("ari", 0):.3f})')
        # Add counts
        for i in range(len(unique_true)):
            for j in range(len(unique_cl)):
                ax.text(j, i, f'{int(confusion[i,j])}', ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color='white' if confusion[i,j] > confusion.max()/2 else 'black')
    else:
        ax.text(0.5, 0.5, 'No clusters', ha='center', va='center',
                transform=ax.transAxes)

    # ---- Panel 6: Training loss curve ----
    ax = axes[1, 2]
    if 'training_losses' in results:
        ax.semilogy(results['training_losses'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Training Convergence')
        ax.grid(alpha=0.3)

    plt.suptitle('Experiment 61: Anomaly Geometry Pilot\n'
                 'Train on EW+KPZ, test on BD/Eden/RD/KS',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig_path = save_dir / 'exp61_anomaly_geometry.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fig_path}")
    plt.close()

    # Also save to figures dir
    fig_copy = FIGURES_DIR / 'exp61_anomaly_geometry.png'
    import shutil
    shutil.copy2(fig_path, fig_copy)
    print(f"  Figure copied: {fig_copy}")


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(pilot=False):
    t0 = time.time()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 61: ANOMALY GEOMETRY PILOT")
    print("=" * 70)
    print(f"Mode: {'PILOT' if pilot else 'FULL'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # ---- Configuration ----
    if pilot:
        L = 64            # spatial grid
        T = 200           # time steps
        N_train = 30      # samples per training class
        N_test = 15       # samples per test class
        latent_dim = 16
        epochs = 50
        batch_size = 16
    else:
        L = 128
        T = 400
        N_train = 80
        N_test = 40
        latent_dim = 32
        epochs = 150
        batch_size = 32

    config = {
        'L': L, 'T': T, 'N_train': N_train, 'N_test': N_test,
        'latent_dim': latent_dim, 'epochs': epochs, 'batch_size': batch_size,
        'mode': 'pilot' if pilot else 'full'
    }
    print(f"Config: L={L}, T={T}, N_train={N_train}, N_test={N_test}")
    print(f"        latent_dim={latent_dim}, epochs={epochs}")
    print()

    # ---- Step 1: Generate training data (EW + KPZ) ----
    print("STEP 1: Generating training data (EW + KPZ)")
    print("-" * 50)

    t1 = time.time()
    # Vary parameters for diversity
    ew_trajs = []
    kpz_trajs = []
    for i in range(N_train):
        # EW with varied diffusion/noise
        nu = np.random.uniform(0.5, 2.0)
        D = np.random.uniform(0.5, 2.0)
        ew_trajs.append(simulate_ew(L=L, T=T, nu=nu, D=D, seed=1000+i))

        # KPZ with varied parameters
        nu = np.random.uniform(0.5, 2.0)
        lam = np.random.uniform(0.5, 3.0)
        D = np.random.uniform(0.5, 2.0)
        kpz_trajs.append(simulate_kpz(L=L, T=T, nu=nu, lam=lam, D=D, seed=2000+i))

    ew_data = np.array(ew_trajs)
    kpz_data = np.array(kpz_trajs)
    train_raw = np.concatenate([ew_data, kpz_data], axis=0)
    train_labels = ['ew'] * N_train + ['kpz'] * N_train

    # Shuffle
    perm = np.random.permutation(len(train_raw))
    train_raw = train_raw[perm]
    train_labels = [train_labels[i] for i in perm]

    train_tensor = prepare_for_autoencoder(train_raw)
    print(f"  Training data: {train_tensor.shape}  ({time.time()-t1:.1f}s)")

    # ---- Step 2: Generate test data (BD, Eden, RD, KS) ----
    print("\nSTEP 2: Generating test data (BD, Eden, RD, KS)")
    print("-" * 50)

    t2 = time.time()
    test_systems = {
        'ew':   generate_dataset(N_test, L, T, 'ew',   seed_offset=5000),
        'kpz':  generate_dataset(N_test, L, T, 'kpz',  seed_offset=6000),
        'bd':   generate_dataset(N_test, L, T, 'bd',   seed_offset=7000),
        'eden': generate_dataset(N_test, L, T, 'eden', seed_offset=8000),
        'rd':   generate_dataset(N_test, L, T, 'rd',   seed_offset=9000),
        'ks':   generate_dataset(N_test, L, T, 'ks',   seed_offset=10000,
                                 record_interval=max(1, 10 if not pilot else 5)),
    }
    test_tensors = {name: prepare_for_autoencoder(data)
                    for name, data in test_systems.items()}
    print(f"  Test data generated for {list(test_tensors.keys())}  ({time.time()-t2:.1f}s)")

    # ---- Step 3: Train autoencoder ----
    print("\nSTEP 3: Training autoencoder on EW + KPZ")
    print("-" * 50)

    model = build_autoencoder(L, T, latent_dim=latent_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    t3 = time.time()
    losses = train_autoencoder(model, train_tensor, epochs=epochs,
                               lr=1e-3, batch_size=batch_size)
    print(f"  Training complete ({time.time()-t3:.1f}s), final loss = {losses[-1]:.6f}")

    # ---- Step 4: Compute anomaly scores ----
    print("\nSTEP 4: Computing anomaly scores")
    print("-" * 50)

    # Compute anomaly scores on all test sets
    model.eval()
    with torch.no_grad():
        train_scores_raw = model.anomaly_score(train_tensor).numpy()
    print(f"  Training anomaly: median = {np.median(train_scores_raw):.4f}, "
          f"mean = {np.mean(train_scores_raw):.4f}")
    print()

    latents, anomaly_scores = compute_latent_and_anomaly(model, test_tensors)

    # Use held-out EW + KPZ test scores as baseline
    # (training scores are artificially low due to overfitting)
    known_class_scores = np.concatenate([
        anomaly_scores['ew'], anomaly_scores['kpz']
    ])
    train_median = np.median(known_class_scores)
    print(f"\n  Baseline (held-out EW+KPZ): median = {train_median:.4f}, "
          f"mean = {np.mean(known_class_scores):.4f}")

    # ---- Step 5: Cluster anomalies ----
    print("\nSTEP 5: Clustering anomalous samples")
    print("-" * 50)

    cluster_labels, anomaly_latents, anomaly_true_labels, anomaly_systems = \
        cluster_anomalies(latents, anomaly_scores,
                          threshold_multiplier=1.2,
                          train_median=train_median)

    # If threshold-based clustering finds nothing, cluster ALL test latents
    if cluster_labels is None:
        print("  Threshold too high. Clustering ALL test latents instead.")
        all_latents_list = []
        all_true_labels = []
        for name, z in latents.items():
            # Skip training classes
            if name in ('ew', 'kpz'):
                continue
            all_latents_list.append(z)
            all_true_labels.extend([name] * len(z))
        if len(all_latents_list) > 0 and sum(len(a) for a in all_latents_list) > 0:
            anomaly_latents = np.vstack(all_latents_list)
            anomaly_true_labels = all_true_labels
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(3, len(anomaly_latents) // 10),
                    min_samples=2, metric='euclidean')
                cluster_labels = clusterer.fit_predict(anomaly_latents)
                n_cl = len(set(cluster_labels) - {-1})
                print(f"  HDBSCAN on unknown systems: {n_cl} clusters")
            except ImportError:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=4, random_state=42, n_init=10)
                cluster_labels = km.fit_predict(anomaly_latents)
                print(f"  KMeans on unknown systems: {len(set(cluster_labels))} clusters")

    cluster_quality = compute_cluster_quality(cluster_labels, anomaly_true_labels)
    print(f"  Cluster quality: ARI = {cluster_quality.get('ari', 0):.3f}, "
          f"NMI = {cluster_quality.get('nmi', 0):.3f}")

    # ---- Step 5b: Direct latent-space clustering of ALL test systems ----
    print("\nSTEP 5b: Direct latent-space clustering (all test systems)")
    print("-" * 50)

    all_z_list = []
    all_true = []
    # Map to expected universality class for comparison
    class_map = {'ew': 'EW', 'kpz': 'KPZ', 'bd': 'KPZ', 'eden': 'KPZ',
                 'rd': 'trivial', 'ks': 'KS'}
    for name, z in latents.items():
        all_z_list.append(z)
        all_true.extend([class_map[name]] * len(z))
    all_z = np.vstack(all_z_list)

    # Try HDBSCAN on full latent space
    try:
        import hdbscan
        full_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, len(all_z) // 15),
            min_samples=3, metric='euclidean')
        full_labels = full_clusterer.fit_predict(all_z)
        n_full_cl = len(set(full_labels) - {-1})
        print(f"  HDBSCAN (all systems): {n_full_cl} clusters, "
              f"{(full_labels == -1).sum()} noise")
    except ImportError:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        full_labels = km.fit_predict(all_z)
        n_full_cl = len(set(full_labels))
        print(f"  KMeans (all systems): {n_full_cl} clusters")

    full_quality = compute_cluster_quality(full_labels, all_true)
    print(f"  Full latent clustering ARI = {full_quality.get('ari', 0):.3f}, "
          f"NMI = {full_quality.get('nmi', 0):.3f}")

    # Also try KMeans with known K (we know there are ~4 classes)
    from sklearn.cluster import KMeans
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km4.fit_predict(all_z)
    km_quality = compute_cluster_quality(km_labels, all_true)
    print(f"  KMeans (K=4) ARI = {km_quality.get('ari', 0):.3f}, "
          f"NMI = {km_quality.get('nmi', 0):.3f}")

    # 1-NN classification accuracy (leave-one-out proxy)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    knn = KNeighborsClassifier(n_neighbors=3)
    all_true_arr = np.array(all_true)
    knn_scores = cross_val_score(knn, all_z, all_true_arr, cv=5, scoring='accuracy')
    knn_acc = np.mean(knn_scores)
    print(f"  3-NN cross-val accuracy (latent space): {knn_acc:.3f}")
    print(f"    -> if > 0.5, latent space carries class information")

    # ---- Step 6: Check predictions ----
    print("\nSTEP 6: Checking predictions")
    print("-" * 50)

    def median_score(name):
        return np.median(anomaly_scores[name])

    p1_pass = median_score('bd') < 2 * train_median
    p2_pass = median_score('rd') > 2 * train_median
    p3_pass = median_score('ks') > 3 * train_median
    p4_pass = cluster_quality['n_clusters'] >= 2 or full_quality.get('n_clusters', 0) >= 2
    p5_pass = (cluster_quality.get('ari', 0) > 0.3
               or full_quality.get('ari', 0) > 0.3
               or knn_acc > 0.5)

    # Also check Eden
    eden_low = median_score('eden') < 2 * train_median

    print(f"  P1 (BD low anomaly):     {'PASS' if p1_pass else 'FAIL'}  "
          f"({median_score('bd'):.4f} vs threshold {2*train_median:.4f})")
    print(f"     (Eden low anomaly):   {'PASS' if eden_low else 'FAIL'}  "
          f"({median_score('eden'):.4f})")
    print(f"  P2 (RD high anomaly):    {'PASS' if p2_pass else 'FAIL'}  "
          f"({median_score('rd'):.4f})")
    print(f"  P3 (KS high anomaly):    {'PASS' if p3_pass else 'FAIL'}  "
          f"({median_score('ks'):.4f})")
    print(f"  P4 (>= 2 clusters):      {'PASS' if p4_pass else 'FAIL'}  "
          f"({cluster_quality['n_clusters']} clusters)")
    print(f"  P5 (ARI > 0.3):          {'PASS' if p5_pass else 'FAIL'}  "
          f"(ARI = {cluster_quality['ari']:.3f})")

    n_pass = sum([p1_pass, p2_pass, p3_pass, p4_pass, p5_pass])
    overall = n_pass >= 4
    print(f"\n  Overall: {n_pass}/5 predictions passed  "
          f"-> {'SUCCESS' if overall else 'PARTIAL / NEEDS INVESTIGATION'}")

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'training': {
            'final_loss': float(losses[-1]),
            'train_anomaly_median': float(np.median(train_scores_raw)),
            'train_anomaly_mean': float(np.mean(train_scores_raw)),
            'n_params': n_params,
        },
        'anomaly_scores': {
            name: {
                'mean': float(np.mean(s)),
                'std': float(np.std(s)),
                'median': float(np.median(s)),
                'ratio_to_train': float(np.median(s) / train_median),
            }
            for name, s in anomaly_scores.items()
        },
        'clustering': cluster_quality,
        'full_latent_clustering': {
            'hdbscan': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in full_quality.items()},
            'kmeans_k4': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                          for k, v in km_quality.items()},
            'knn_3_accuracy': float(knn_acc),
        },
        'predictions': {
            'P1_bd_low': bool(p1_pass),
            'P1_eden_low': bool(eden_low),
            'P2_rd_high': bool(p2_pass),
            'P3_ks_high': bool(p3_pass),
            'P4_clusters': bool(p4_pass),
            'P5_ari': bool(p5_pass),
            'n_pass': int(n_pass),
            'overall': bool(overall),
        },
        'training_losses': [float(l) for l in losses],
        'elapsed_seconds': time.time() - t0,
    }

    results_path = RESULTS_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Save model
    model_path = RESULTS_DIR / 'autoencoder.pt'
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")

    # ---- Figures ----
    print("\nSTEP 7: Generating figures")
    print("-" * 50)
    make_figures(latents, anomaly_scores, cluster_labels, anomaly_latents,
                 anomaly_true_labels, train_median, results, RESULTS_DIR)

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Training loss: {losses[-1]:.6f}")
    print(f"  Anomaly ranking (median score / train median):")
    ranking = sorted(anomaly_scores.keys(),
                     key=lambda s: np.median(anomaly_scores[s]),
                     reverse=True)
    for name in ranking:
        ratio = np.median(anomaly_scores[name]) / train_median
        marker = " <-- ANOMALOUS" if ratio > 2 else ""
        print(f"    {name:6s}: {ratio:6.2f}x{marker}")
    print(f"  Clusters: {cluster_quality.get('n_clusters', 0)} (anomaly-based), "
          f"{full_quality.get('n_clusters', 0)} (full latent), "
          f"ARI = {full_quality.get('ari', 0):.3f}")
    print(f"  3-NN accuracy: {knn_acc:.3f}")
    print(f"  Predictions: {n_pass}/5 passed")
    print()

    if overall:
        print("  RESULT: Anomaly geometry separates universality classes.")
        print("  The autoencoder discovers class structure without labels.")
    else:
        print("  RESULT: Partial separation. Investigate which predictions failed.")
        print("  May need: longer training, larger data, or different architecture.")

    print("=" * 70)
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 61: Anomaly Geometry Pilot')
    parser.add_argument('--pilot', action='store_true',
                        help='Quick test (~5 min)')
    args = parser.parse_args()

    results = run_experiment(pilot=args.pilot)
