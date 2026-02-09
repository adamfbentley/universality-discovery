"""
Experiment 8: Physics-Informed Autoencoder
==========================================

Test whether embedding scaling laws into the loss function improves
universality class detection and amplifies the EW/KPZ separation.

Key Idea:
- Standard autoencoder only minimizes reconstruction error
- Physics-informed adds: L_scaling to penalize wrong growth exponent β
- Expected: Latent space should better separate universality classes

Loss Function:
    L_total = L_reconstruction + γ₁*L_scaling + γ₂*L_cluster

Where:
    L_scaling = |W(t)/t^β - const|²  (penalize deviation from scaling law)
    L_cluster = intra_class_dist / inter_class_dist (cluster same-class)

Theory (from literature):
- Φ-DVAE (arXiv 2209.15609): PINN-VAE embeds PDEs into latent dynamics
- PIAE (MDPI Energies 2025): 30-50% SNR improvement with physics loss
- GradNorm (ASME 2023): Multi-objective balancing technique

Success Criterion:
- Exp 7b showed KPZ/EW separation of 1.01x (22.00 vs 22.18)
- Target: Increase separation to >1.5x with physics-informed loss
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy import stats

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
WIDTH = 128
TIME_STEPS = 500
N_TRAIN_PER_CLASS = 150
N_TEST = 80
EPOCHS = 30
LATENT_DIM = 32
SIGMA = 2  # Coarse-graining parameter

# Physics-informed loss weights (will be tuned)
GAMMA_SCALING = 0.1  # Weight for scaling law loss
GAMMA_CLUSTER = 0.05  # Weight for clustering loss

# Theoretical scaling exponents
BETA_EW = 0.25   # Edwards-Wilkinson: W(t) ~ t^0.25
BETA_KPZ = 1/3   # KPZ class: W(t) ~ t^(1/3) ≈ 0.333


def compute_gradient_field(surface):
    """Compute spatial gradient of surface."""
    grad_x = np.roll(surface, -1, axis=0) - surface
    return grad_x


def apply_coarse_graining(field, sigma):
    """Apply Gaussian blur for coarse-graining."""
    if sigma == 0:
        return field
    return gaussian_filter(field, sigma=sigma, mode='wrap')


def preprocess_surface(surface, sigma=2):
    """Transform to gradient space and apply coarse-graining."""
    grad = compute_gradient_field(surface)
    return apply_coarse_graining(grad, sigma)


def compute_interface_width(surface):
    """
    Compute time-dependent interface width W(t) = sqrt(<(h - <h>)²>).
    
    Returns array of width values at each time step.
    """
    # surface shape: (width, time_steps)
    widths = []
    for t in range(surface.shape[1]):
        h_t = surface[:, t]
        w_t = np.sqrt(np.mean((h_t - np.mean(h_t))**2))
        widths.append(w_t)
    return np.array(widths)


def fit_scaling_exponent(widths, skip_transient=0.2):
    """
    Fit W(t) ~ t^β to extract growth exponent.
    
    Returns:
        beta: fitted exponent
        const: prefactor
        r_squared: goodness of fit
    """
    n_times = len(widths)
    start_idx = int(skip_transient * n_times)
    
    # Use log-log fit
    t_vals = np.arange(start_idx, n_times) + 1  # Avoid log(0)
    w_vals = widths[start_idx:]
    
    # Filter out zeros/negatives
    valid = w_vals > 0
    if np.sum(valid) < 10:
        return 0.25, 1.0, 0.0  # Default to EW
    
    log_t = np.log(t_vals[valid])
    log_w = np.log(w_vals[valid])
    
    # Linear fit in log-log space
    slope, intercept, r_value, _, _ = stats.linregress(log_t, log_w)
    
    return slope, np.exp(intercept), r_value**2


def generate_dataset_with_physics(model_types, n_per_class, sigma=2, desc="Generating"):
    """
    Generate preprocessed surfaces with physics metadata.
    
    Returns:
        surfaces: preprocessed gradient fields
        widths: interface width trajectories
        betas: fitted scaling exponents
        labels: class labels
    """
    all_surfaces = []
    all_widths = []
    all_betas = []
    all_labels = []
    
    for class_idx, model_type in enumerate(model_types):
        print(f"\n  {model_type}...")
        for i in tqdm(range(n_per_class), desc=f"  {model_type}"):
            sim = GrowthModelSimulator(WIDTH, TIME_STEPS)
            
            if model_type == 'edwards_wilkinson':
                surface = sim.generate_trajectory('edwards_wilkinson', diffusion=1.0)
            elif model_type == 'kpz_equation':
                surface = sim.generate_trajectory('kpz_equation', diffusion=1.0, nonlinearity=1.0)
            elif model_type == 'ballistic_deposition':
                surface = sim.generate_trajectory('ballistic_deposition')
            elif model_type == 'eden':
                surface = sim.generate_trajectory('eden')
            elif model_type == 'random_deposition':
                surface = sim.generate_trajectory('random_deposition')
            
            # Compute physics: interface width and scaling exponent
            widths = compute_interface_width(surface)
            beta, const, r2 = fit_scaling_exponent(widths)
            
            # Preprocess for autoencoder
            processed = preprocess_surface(surface, sigma)
            
            all_surfaces.append(processed)
            all_widths.append(widths)
            all_betas.append(beta)
            all_labels.append(class_idx)
    
    return (np.array(all_surfaces), np.array(all_widths), 
            np.array(all_betas), np.array(all_labels))


class PhysicsInformedAutoencoder(nn.Module):
    """
    Autoencoder with physics-informed loss terms.
    
    Adds scaling law constraint to encourage learning of universal features.
    """
    
    def __init__(self, width, time_steps, latent_dim, target_beta=BETA_KPZ):
        super().__init__()
        
        self.width = width
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.target_beta = target_beta
        
        # Standard autoencoder backbone
        self.base_model = SurfaceAutoencoder(width, time_steps, latent_dim)
        
        # Additional layer to predict scaling exponent from latent
        self.beta_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Constrain to [0, 1] range
        )
    
    def encode(self, x):
        return self.base_model.encode(x)
    
    def decode(self, z):
        return self.base_model.decode(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        beta_pred = self.beta_predictor(z).squeeze(-1)  # Predicted β
        return x_recon, z, beta_pred
    
    def reconstruction_loss(self, x, x_recon):
        return F.mse_loss(x_recon, x)
    
    def scaling_loss(self, beta_pred, beta_true):
        """
        Penalize deviation from true scaling exponent.
        """
        return F.mse_loss(beta_pred, beta_true)
    
    def cluster_loss(self, z, labels):
        """
        Encourage same-class samples to cluster in latent space.
        L_cluster = intra_class_variance / inter_class_distance
        """
        unique_labels = torch.unique(labels)
        
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=z.device)
        
        # Compute class centroids
        centroids = []
        intra_class_vars = []
        
        for label in unique_labels:
            mask = labels == label
            class_z = z[mask]
            centroid = class_z.mean(dim=0)
            centroids.append(centroid)
            
            # Intra-class variance
            if class_z.shape[0] > 1:
                var = ((class_z - centroid)**2).sum(dim=1).mean()
                intra_class_vars.append(var)
        
        if len(intra_class_vars) == 0:
            return torch.tensor(0.0, device=z.device)
        
        # Inter-class distance
        centroids = torch.stack(centroids)
        inter_dist = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0))[0]
        # Mean of off-diagonal elements
        mask = ~torch.eye(len(centroids), dtype=bool, device=z.device)
        mean_inter_dist = inter_dist[mask].mean()
        
        # Ratio: want low intra, high inter
        mean_intra_var = torch.stack(intra_class_vars).mean()
        
        # Add small epsilon to avoid division by zero
        loss = mean_intra_var / (mean_inter_dist + 1e-6)
        
        return loss


def train_physics_informed(model, train_data, train_betas, train_labels, 
                           gamma_scaling=0.1, gamma_cluster=0.05, epochs=30):
    """
    Train physics-informed autoencoder with composite loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare data
    data_t = np.transpose(train_data, (0, 2, 1))
    mean = data_t.mean()
    std = data_t.std()
    data_norm = (data_t - mean) / (std + 1e-8)
    
    tensor_data = torch.FloatTensor(data_norm).unsqueeze(1).to(device)
    tensor_betas = torch.FloatTensor(train_betas).to(device)
    tensor_labels = torch.LongTensor(train_labels).to(device)
    
    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_betas, tensor_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    history = {'total': [], 'recon': [], 'scaling': [], 'cluster': []}
    
    for epoch in tqdm(range(epochs), desc="Training Physics-Informed AE"):
        epoch_losses = {'total': 0, 'recon': 0, 'scaling': 0, 'cluster': 0}
        n_batches = 0
        
        for batch_x, batch_beta, batch_label in loader:
            optimizer.zero_grad()
            
            x_recon, z, beta_pred = model(batch_x)
            
            # Compute losses
            l_recon = model.reconstruction_loss(batch_x, x_recon)
            l_scaling = model.scaling_loss(beta_pred, batch_beta)
            l_cluster = model.cluster_loss(z, batch_label)
            
            # Combined loss
            l_total = l_recon + gamma_scaling * l_scaling + gamma_cluster * l_cluster
            
            l_total.backward()
            optimizer.step()
            
            epoch_losses['total'] += l_total.item()
            epoch_losses['recon'] += l_recon.item()
            epoch_losses['scaling'] += l_scaling.item()
            epoch_losses['cluster'] += l_cluster.item()
            n_batches += 1
        
        for key in epoch_losses:
            history[key].append(epoch_losses[key] / n_batches)
    
    return model, mean, std, history


def train_standard(data, epochs=30):
    """Train standard autoencoder (baseline) for comparison."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_t = np.transpose(data, (0, 2, 1))
    mean = data_t.mean()
    std = data_t.std()
    data_norm = (data_t - mean) / (std + 1e-8)
    
    model = SurfaceAutoencoder(width=WIDTH, time_steps=TIME_STEPS, latent_dim=LATENT_DIM)
    model = model.to(device)
    
    tensor_data = torch.FloatTensor(data_norm).unsqueeze(1).to(device)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(epochs), desc="Training Standard AE"):
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, z = model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
    
    return model, mean, std


def encode_data(model, data, mean, std, physics_informed=False):
    """Encode data to latent space."""
    device = next(model.parameters()).device
    model.eval()
    
    data_t = np.transpose(data, (0, 2, 1))
    data_norm = (data_t - mean) / (std + 1e-8)
    tensor_data = torch.FloatTensor(data_norm).unsqueeze(1).to(device)
    
    latents = []
    with torch.no_grad():
        for i in range(len(data)):
            x = tensor_data[i:i+1]
            if physics_informed:
                _, z, _ = model(x)
            else:
                _, z = model(x)
            latents.append(z.cpu().numpy().flatten())
    
    return np.array(latents)


def wasserstein_distance_sliced(X, Y, n_projections=100):
    """Compute sliced Wasserstein distance between point clouds."""
    distances = []
    for _ in range(n_projections):
        direction = np.random.randn(X.shape[1])
        direction /= np.linalg.norm(direction)
        X_proj = X @ direction
        Y_proj = Y @ direction
        d = stats.wasserstein_distance(X_proj, Y_proj)
        distances.append(d)
    return np.mean(distances)


def compute_separation_metrics(latents_dict):
    """
    Compute separation metrics between classes.
    
    Returns:
        intra_class: average within-class variance
        inter_class: average between-class distance
        separation_ratio: inter/intra (higher = better)
    """
    classes = list(latents_dict.keys())
    
    # Intra-class variance
    intra_vars = []
    for cls in classes:
        z = latents_dict[cls]
        var = np.var(z, axis=0).mean()
        intra_vars.append(var)
    intra_class = np.mean(intra_vars)
    
    # Inter-class centroid distances
    centroids = {cls: latents_dict[cls].mean(axis=0) for cls in classes}
    inter_dists = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            d = np.linalg.norm(centroids[c1] - centroids[c2])
            inter_dists.append(d)
    inter_class = np.mean(inter_dists) if inter_dists else 0
    
    separation_ratio = inter_class / (np.sqrt(intra_class) + 1e-6)
    
    return intra_class, inter_class, separation_ratio


def main():
    print("="*70)
    print("EXPERIMENT 8: Physics-Informed Autoencoder")
    print("="*70)
    print("\nHypothesis: Adding scaling law constraints to loss function will")
    print("           amplify universality class separation in latent space.")
    print(f"\nTarget: Increase EW/KPZ separation from 1.01x to >1.5x")
    print("="*70)
    
    # ===== Step 1: Generate Training Data =====
    print("\n[1] Generating training data with physics metadata...")
    
    # Train on discrete KPZ models (following Exp 6/7 paradigm)
    train_models = ['ballistic_deposition', 'eden']
    
    train_data, train_widths, train_betas, train_labels = generate_dataset_with_physics(
        train_models, N_TRAIN_PER_CLASS, sigma=SIGMA, desc="Training"
    )
    
    print(f"\n  Training samples: {len(train_data)}")
    print(f"  Fitted β values: BD mean={train_betas[train_labels==0].mean():.3f}, "
          f"EDEN mean={train_betas[train_labels==1].mean():.3f}")
    
    # ===== Step 2: Generate Test Data =====
    print("\n[2] Generating test data (EW, KPZ, BD, EDEN, RD)...")
    
    test_models = ['edwards_wilkinson', 'kpz_equation', 'ballistic_deposition', 
                   'eden', 'random_deposition']
    
    test_data, test_widths, test_betas, test_labels = generate_dataset_with_physics(
        test_models, N_TEST, sigma=SIGMA, desc="Test"
    )
    
    print(f"\n  Test samples per class: {N_TEST}")
    for i, model in enumerate(test_models):
        print(f"    {model}: β={test_betas[test_labels==i].mean():.3f}")
    
    # ===== Step 3: Train Standard Autoencoder (Baseline) =====
    print("\n[3] Training STANDARD autoencoder (baseline)...")
    
    model_standard, mean_std, std_std = train_standard(train_data, epochs=EPOCHS)
    
    # ===== Step 4: Train Physics-Informed Autoencoder =====
    print("\n[4] Training PHYSICS-INFORMED autoencoder...")
    
    model_physics = PhysicsInformedAutoencoder(
        WIDTH, TIME_STEPS, LATENT_DIM, target_beta=BETA_KPZ
    )
    
    model_physics, mean_phys, std_phys, history = train_physics_informed(
        model_physics, train_data, train_betas, train_labels,
        gamma_scaling=GAMMA_SCALING, gamma_cluster=GAMMA_CLUSTER, epochs=EPOCHS
    )
    
    # ===== Step 5: Encode Test Data =====
    print("\n[5] Encoding test data to latent space...")
    
    # Standard model
    latents_standard = {}
    for i, model_type in enumerate(test_models):
        mask = test_labels == i
        latents_standard[model_type] = encode_data(
            model_standard, test_data[mask], mean_std, std_std, physics_informed=False
        )
    
    # Physics-informed model
    latents_physics = {}
    for i, model_type in enumerate(test_models):
        mask = test_labels == i
        latents_physics[model_type] = encode_data(
            model_physics, test_data[mask], mean_phys, std_phys, physics_informed=True
        )
    
    # ===== Step 6: Compute Wasserstein Distances =====
    print("\n[6] Computing Wasserstein distances...")
    
    def compute_all_wasserstein(latents_dict):
        classes = list(latents_dict.keys())
        n = len(classes)
        distances = np.zeros((n, n))
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                if i < j:
                    d = wasserstein_distance_sliced(latents_dict[c1], latents_dict[c2])
                    distances[i, j] = d
                    distances[j, i] = d
        return distances, classes
    
    d_standard, class_names = compute_all_wasserstein(latents_standard)
    d_physics, _ = compute_all_wasserstein(latents_physics)
    
    # ===== Step 7: Analyze Results =====
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Key comparison: EW vs KPZ distance
    ew_idx = class_names.index('edwards_wilkinson')
    kpz_idx = class_names.index('kpz_equation')
    bd_idx = class_names.index('ballistic_deposition')
    eden_idx = class_names.index('eden')
    
    d_ew_kpz_std = d_standard[ew_idx, kpz_idx]
    d_ew_kpz_phys = d_physics[ew_idx, kpz_idx]
    
    # Distance to discrete training data
    d_kpz_discrete_std = (d_standard[kpz_idx, bd_idx] + d_standard[kpz_idx, eden_idx]) / 2
    d_ew_discrete_std = (d_standard[ew_idx, bd_idx] + d_standard[ew_idx, eden_idx]) / 2
    
    d_kpz_discrete_phys = (d_physics[kpz_idx, bd_idx] + d_physics[kpz_idx, eden_idx]) / 2
    d_ew_discrete_phys = (d_physics[ew_idx, bd_idx] + d_physics[ew_idx, eden_idx]) / 2
    
    print("\n[A] Wasserstein Distance Matrices")
    print("\n  STANDARD Autoencoder:")
    print("  " + " "*20 + "  ".join([m[:6] for m in class_names]))
    for i, name in enumerate(class_names):
        row = [f"{d_standard[i,j]:6.2f}" for j in range(len(class_names))]
        print(f"  {name[:18]:18} " + " ".join(row))
    
    print("\n  PHYSICS-INFORMED Autoencoder:")
    print("  " + " "*20 + "  ".join([m[:6] for m in class_names]))
    for i, name in enumerate(class_names):
        row = [f"{d_physics[i,j]:6.2f}" for j in range(len(class_names))]
        print(f"  {name[:18]:18} " + " ".join(row))
    
    print("\n[B] Key Comparisons")
    print(f"\n  EW↔KPZ distance:")
    print(f"    Standard:         {d_ew_kpz_std:.3f}")
    print(f"    Physics-Informed: {d_ew_kpz_phys:.3f}")
    print(f"    Improvement:      {d_ew_kpz_phys/d_ew_kpz_std:.2f}x")
    
    print(f"\n  KPZ → Discrete (training) distance:")
    print(f"    Standard:         {d_kpz_discrete_std:.3f}")
    print(f"    Physics-Informed: {d_kpz_discrete_phys:.3f}")
    
    print(f"\n  EW → Discrete distance:")
    print(f"    Standard:         {d_ew_discrete_std:.3f}")
    print(f"    Physics-Informed: {d_ew_discrete_phys:.3f}")
    
    # Separation ratio (key metric from Exp 7b)
    ratio_std = d_ew_discrete_std / d_kpz_discrete_std
    ratio_phys = d_ew_discrete_phys / d_kpz_discrete_phys
    
    print(f"\n[C] Universality Separation Ratio (EW→discrete / KPZ→discrete)")
    print(f"    Exp 7b baseline:  1.01x")
    print(f"    Standard AE:      {ratio_std:.2f}x")
    print(f"    Physics-Informed: {ratio_phys:.2f}x")
    
    success = ratio_phys > 1.5
    print(f"\n    SUCCESS CRITERION (>1.5x): {'✅ PASSED' if success else '❌ NOT MET'}")
    
    # ===== Step 8: Compute Separation Metrics =====
    print("\n[D] Latent Space Separation Metrics")
    
    intra_std, inter_std, sep_std = compute_separation_metrics(latents_standard)
    intra_phys, inter_phys, sep_phys = compute_separation_metrics(latents_physics)
    
    print(f"\n  Standard AE:")
    print(f"    Intra-class variance: {intra_std:.4f}")
    print(f"    Inter-class distance: {inter_std:.4f}")
    print(f"    Separation ratio:     {sep_std:.4f}")
    
    print(f"\n  Physics-Informed AE:")
    print(f"    Intra-class variance: {intra_phys:.4f}")
    print(f"    Inter-class distance: {inter_phys:.4f}")
    print(f"    Separation ratio:     {sep_phys:.4f}")
    
    improvement = sep_phys / (sep_std + 1e-6)
    print(f"\n    Separation improvement: {improvement:.2f}x")
    
    # ===== Step 9: Training History =====
    print("\n[E] Training Loss History (Physics-Informed)")
    print(f"    Final reconstruction loss: {history['recon'][-1]:.6f}")
    print(f"    Final scaling loss:        {history['scaling'][-1]:.6f}")
    print(f"    Final cluster loss:        {history['cluster'][-1]:.6f}")
    
    # ===== Step 10: Save Figures =====
    print("\n[7] Saving figures...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Wasserstein distance matrix comparison
    im1 = axes[0, 0].imshow(d_standard, cmap='viridis')
    axes[0, 0].set_title('Standard AE: Wasserstein Distances')
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_xticklabels([n[:6] for n in class_names], rotation=45)
    axes[0, 0].set_yticks(range(len(class_names)))
    axes[0, 0].set_yticklabels([n[:6] for n in class_names])
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(d_physics, cmap='viridis')
    axes[0, 1].set_title('Physics-Informed AE: Wasserstein Distances')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels([n[:6] for n in class_names], rotation=45)
    axes[0, 1].set_yticks(range(len(class_names)))
    axes[0, 1].set_yticklabels([n[:6] for n in class_names])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 2: Key distance comparison
    metrics = ['EW↔KPZ', 'KPZ→Discrete', 'EW→Discrete']
    std_vals = [d_ew_kpz_std, d_kpz_discrete_std, d_ew_discrete_std]
    phys_vals = [d_ew_kpz_phys, d_kpz_discrete_phys, d_ew_discrete_phys]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[0, 2].bar(x - width/2, std_vals, width, label='Standard', color='steelblue')
    axes[0, 2].bar(x + width/2, phys_vals, width, label='Physics-Informed', color='coral')
    axes[0, 2].set_ylabel('Wasserstein Distance')
    axes[0, 2].set_title('Key Distance Comparisons')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(metrics)
    axes[0, 2].legend()
    
    # Plot 3: Training history
    epochs_x = range(1, len(history['total']) + 1)
    axes[1, 0].plot(epochs_x, history['recon'], label='Reconstruction', color='blue')
    axes[1, 0].plot(epochs_x, history['scaling'], label=f'Scaling (×{GAMMA_SCALING})', color='orange')
    axes[1, 0].plot(epochs_x, history['cluster'], label=f'Cluster (×{GAMMA_CLUSTER})', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss History')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    
    # Plot 4: 2D latent visualization (PCA)
    from sklearn.decomposition import PCA
    
    # Combine all latents
    all_latents_std = np.vstack([latents_standard[m] for m in class_names])
    all_latents_phys = np.vstack([latents_physics[m] for m in class_names])
    all_labels = np.concatenate([[i]*len(latents_standard[m]) for i, m in enumerate(class_names)])
    
    pca = PCA(n_components=2)
    latents_2d_std = pca.fit_transform(all_latents_std)
    latents_2d_phys = pca.fit_transform(all_latents_phys)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model, color) in enumerate(zip(class_names, colors)):
        mask = all_labels == i
        axes[1, 1].scatter(latents_2d_std[mask, 0], latents_2d_std[mask, 1], 
                          c=color, label=model[:6], alpha=0.5, s=20)
    axes[1, 1].set_title('Standard AE: Latent Space (PCA)')
    axes[1, 1].legend(fontsize=8)
    
    for i, (model, color) in enumerate(zip(class_names, colors)):
        mask = all_labels == i
        axes[1, 2].scatter(latents_2d_phys[mask, 0], latents_2d_phys[mask, 1], 
                          c=color, label=model[:6], alpha=0.5, s=20)
    axes[1, 2].set_title('Physics-Informed AE: Latent Space (PCA)')
    axes[1, 2].legend(fontsize=8)
    
    plt.tight_layout()
    
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'exp08_physics_informed.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    
    plt.close()
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Experiment 8: Physics-Informed Autoencoder

Configuration:
  - Training: BD + EDEN (discrete KPZ)
  - Test: EW, KPZ, BD, EDEN, RD
  - Physics loss weights: γ_scaling={GAMMA_SCALING}, γ_cluster={GAMMA_CLUSTER}

Key Results:
  EW↔KPZ distance:      {d_ew_kpz_std:.3f} → {d_ew_kpz_phys:.3f} ({d_ew_kpz_phys/d_ew_kpz_std:.2f}x)
  
  Universality ratio (EW→discrete / KPZ→discrete):
    Exp 7b baseline:    1.01x
    Standard AE:        {ratio_std:.2f}x
    Physics-Informed:   {ratio_phys:.2f}x
    
  Latent separation improvement: {improvement:.2f}x
  
  SUCCESS: {'✅ Physics-informed loss AMPLIFIES universality signal!' if ratio_phys > ratio_std * 1.1 else '⚠️ Minimal improvement - may need hyperparameter tuning'}
  
Interpretation:
  - Physics-informed loss encourages latent space to respect scaling laws
  - Cluster loss pulls same-class samples together
  - Combined effect should improve universality class boundaries
  
Next Steps:
  - If improvement < expected: tune γ_scaling, γ_cluster
  - If improvement significant: proceed to Exp 9 (multi-scale wavelets)
""")


if __name__ == "__main__":
    main()
