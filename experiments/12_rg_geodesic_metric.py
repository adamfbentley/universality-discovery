"""
Experiment 12: RG-Geodesic Metric Learning

Theoretical Motivation:
    Cotler & Rezchikov (2022, arXiv:2202.11737) propose that RG flow can be 
    viewed as optimal transport minimizing a dissipation functional. The key 
    insight: coarse-graining (RG transformations) should follow geodesics in 
    the space of probability distributions.

Key Idea:
    Instead of just using Wasserstein distance as a metric (Exp 7b), we make 
    it part of the LEARNING PROCESS. The autoencoder loss includes a term that 
    penalizes deviation from geodesic RG flow:
    
    L_total = L_recon + γ_RG * L_geodesic
    
    where L_geodesic measures how far the learned metric deviates from the 
    "dissipation-minimizing" path during coarse-graining.

Connection to Literature:
    - Cotler-Rezchikov: RG flow minimizes relative entropy production
    - Ferrari-Spohn (2011): KPZ fluctuations have Tracy-Widom structure
    - Barabási-Stanley (1995): Universality emerges under coarse-graining
    
Implementation:
    1. Generate surfaces at multiple blur scales σ ∈ {0, 1, 2, 4}
    2. Encoder maps each scale to latent space
    3. Compute Wasserstein distance between adjacent scales in latent space
    4. Regularization: penalize non-geodesic RG trajectories
    5. Hypothesis: This forces the metric to respect universality class structure

Expected Outcome:
    If RG flow truly separates universality classes, then a metric that 
    "knows about" RG geodesics should naturally place EW far from KPZ-class 
    while grouping KPZ/BD/EDEN together.

Date: January 18, 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

from simulation.physics_simulation import GrowthModelSimulator

# ============================================================================
# Multi-Scale Surface Dataset with RG Flow
# ============================================================================

class MultiScaleSurfaceDataset(Dataset):
    """Dataset that includes surfaces at multiple coarse-graining scales."""
    
    def __init__(self, surfaces, labels, blur_scales=[0, 1, 2, 4]):
        """
        Args:
            surfaces: List of (L, T) height field arrays
            labels: List of model type labels
            blur_scales: List of Gaussian blur σ values for coarse-graining
        """
        self.surfaces = surfaces
        self.labels = labels
        self.blur_scales = blur_scales
        
    def __len__(self):
        return len(self.surfaces)
    
    def apply_coarse_graining(self, surface, sigma):
        """Apply Gaussian blur (RG transformation) to gradient field."""
        from scipy.ndimage import gaussian_filter1d
        
        # Compute gradient field
        gradients = np.diff(surface, axis=0)
        
        if sigma > 0:
            # Apply blur to each time slice
            blurred = np.array([gaussian_filter1d(gradients[:, t], sigma=sigma) 
                               for t in range(gradients.shape[1])]).T
            return blurred
        return gradients
    
    def __getitem__(self, idx):
        """Return surface at all coarse-graining scales."""
        surface = self.surfaces[idx]
        label = self.labels[idx]
        
        # Generate multi-scale representations
        scales = []
        for sigma in self.blur_scales:
            blurred = self.apply_coarse_graining(surface, sigma)
            
            # Normalize
            blurred = (blurred - blurred.mean()) / (blurred.std() + 1e-8)
            
            # Take late-time snapshot
            snapshot = blurred[:, -1]
            scales.append(torch.FloatTensor(snapshot))
        
        return scales, label

# ============================================================================
# RG-Geodesic Autoencoder Architecture
# ============================================================================

class RGGeodesicAutoencoder(nn.Module):
    """
    Autoencoder with embedded knowledge of RG flow geodesics.
    
    Architecture:
        - Shared encoder for all scales
        - Separate decoders per scale
        - Latent space designed to preserve RG flow geometry
    """
    
    def __init__(self, input_dim=499, latent_dim=32, hidden_dims=[256, 128]):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Shared encoder (maps any scale to latent space)
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Scale-specific decoders
        self.decoders = nn.ModuleList([
            self._build_decoder(latent_dim, hidden_dims[::-1], input_dim)
            for _ in range(4)  # One per blur scale
        ])
        
    def _build_decoder(self, latent_dim, hidden_dims, output_dim):
        """Build decoder network."""
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Map surface to latent space."""
        return self.encoder(x)
    
    def decode(self, z, scale_idx):
        """Reconstruct surface at specified scale."""
        return self.decoders[scale_idx](z)
    
    def forward(self, scales):
        """
        Forward pass through all scales.
        
        Args:
            scales: List of [batch_size, input_dim] tensors, one per scale
            
        Returns:
            latents: List of latent representations
            reconstructions: List of reconstructed surfaces
        """
        latents = []
        reconstructions = []
        
        for scale_idx, x in enumerate(scales):
            z = self.encode(x)
            x_recon = self.decode(z, scale_idx)
            
            latents.append(z)
            reconstructions.append(x_recon)
        
        return latents, reconstructions

# ============================================================================
# RG-Geodesic Loss Function
# ============================================================================

def compute_sliced_wasserstein(z1, z2, num_projections=100):
    """
    Compute Sliced Wasserstein distance between two batches in latent space.
    
    This is the core metric from Cotler-Rezchikov: measures how "far" two 
    distributions are in the optimal transport sense.
    """
    device = z1.device
    d = z1.shape[1]
    
    distances = []
    for _ in range(num_projections):
        # Random projection direction
        theta = torch.randn(d, device=device)
        theta = theta / torch.norm(theta)
        
        # Project data
        proj1 = torch.matmul(z1, theta)
        proj2 = torch.matmul(z2, theta)
        
        # Sort projections
        proj1_sorted, _ = torch.sort(proj1)
        proj2_sorted, _ = torch.sort(proj2)
        
        # 1D Wasserstein distance
        dist = torch.mean(torch.abs(proj1_sorted - proj2_sorted))
        distances.append(dist)
    
    return torch.mean(torch.stack(distances))

def rg_geodesic_loss(latents, gamma_geodesic=0.5, gamma_smoothness=0.1):
    """
    Geodesic regularization loss inspired by Cotler-Rezchikov.
    
    Key idea: Under RG flow (coarse-graining), the trajectory in latent space 
    should follow a geodesic minimizing dissipation. This loss penalizes 
    deviations from:
    
    1. Triangle inequality: d(scale_0, scale_2) ≤ d(scale_0, scale_1) + d(scale_1, scale_2)
    2. Smoothness: Adjacent scales should have similar latent codes
    
    Theory (Cotler-Rezchikov Sec 2.3):
        RG flow minimizes relative entropy production:
        dS/dt = ∫ (δF/δρ) ∂ρ/∂t dx ≥ 0
        
        The geodesic condition means: shortest path in Wasserstein space.
    
    Args:
        latents: List of [batch_size, latent_dim] tensors at different scales
        gamma_geodesic: Weight for triangle inequality penalty
        gamma_smoothness: Weight for smoothness penalty
        
    Returns:
        loss: Scalar tensor
    """
    n_scales = len(latents)
    
    # Geodesic penalty: triangle inequality violation
    geodesic_penalty = 0
    for i in range(n_scales - 2):
        d_direct = compute_sliced_wasserstein(latents[i], latents[i+2])
        d_via = (compute_sliced_wasserstein(latents[i], latents[i+1]) + 
                 compute_sliced_wasserstein(latents[i+1], latents[i+2]))
        
        # Penalize if direct distance > detour (non-geodesic)
        violation = torch.relu(d_direct - d_via)
        geodesic_penalty += violation
    
    # Smoothness penalty: adjacent scales should be close
    smoothness_penalty = 0
    for i in range(n_scales - 1):
        smoothness_penalty += torch.mean(torch.norm(latents[i] - latents[i+1], dim=1))
    
    return gamma_geodesic * geodesic_penalty + gamma_smoothness * smoothness_penalty

# ============================================================================
# Training Loop with RG-Aware Loss
# ============================================================================

def train_rg_geodesic_autoencoder(train_loader, latent_dim=32, num_epochs=50,
                                   gamma_geodesic=0.5, gamma_smoothness=0.1):
    """
    Train autoencoder with RG-geodesic regularization.
    
    Loss function:
        L_total = L_reconstruction + L_rg_geodesic
        
    where L_rg_geodesic encodes the Cotler-Rezchikov constraint that 
    coarse-graining should follow geodesics in Wasserstein space.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = RGGeodesicAutoencoder(input_dim=499, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'total_loss': [], 'recon_loss': [], 'rg_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_total = 0
        epoch_recon = 0
        epoch_rg = 0
        
        for scales, labels in train_loader:
            # Move to device
            scales = [s.to(device) for s in scales]
            
            # Forward pass
            latents, reconstructions = model(scales)
            
            # Reconstruction loss (per scale)
            recon_loss = 0
            for scale_idx, (x, x_recon) in enumerate(zip(scales, reconstructions)):
                recon_loss += nn.MSELoss()(x_recon, x)
            recon_loss /= len(scales)
            
            # RG-geodesic loss
            rg_loss = rg_geodesic_loss(latents, gamma_geodesic, gamma_smoothness)
            
            # Total loss
            total_loss = recon_loss + rg_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_rg += rg_loss.item()
        
        # Average losses
        n_batches = len(train_loader)
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_rg = epoch_rg / n_batches
        
        history['total_loss'].append(avg_total)
        history['recon_loss'].append(avg_recon)
        history['rg_loss'].append(avg_rg)
        
        scheduler.step(avg_total)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Total: {avg_total:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"RG: {avg_rg:.4f}")
    
    return model, history

# ============================================================================
# Evaluation: Wasserstein Distances in Learned Metric
# ============================================================================

def compute_pairwise_wasserstein(model, test_data, model_names, device):
    """
    Compute pairwise Wasserstein distances in the learned RG-geodesic metric.
    
    Hypothesis: If the metric learned to respect RG flow, then models in the 
    same universality class should have smaller distances (they flow to the 
    same fixed point).
    """
    model.eval()
    
    # Encode all samples to latent space (use σ=2 scale as in Exp 6)
    latent_representations = {name: [] for name in model_names}
    
    with torch.no_grad():
        for scales, labels in test_data:
            scales = [s.to(device) for s in scales]
            
            # Use scale index 2 (σ=2, optimal from Exp 5)
            z = model.encode(scales[2])
            
            for i, label in enumerate(labels):
                model_name = model_names[label]
                latent_representations[model_name].append(z[i].cpu().numpy())
    
    # Stack into arrays
    for name in model_names:
        latent_representations[name] = np.array(latent_representations[name])
    
    # Compute pairwise Wasserstein distances
    print("\nComputing pairwise Wasserstein distances in RG-geodesic metric...")
    
    from scipy.stats import wasserstein_distance
    
    distance_matrix = np.zeros((len(model_names), len(model_names)))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i <= j:
                z1 = latent_representations[name1]
                z2 = latent_representations[name2]
                
                # Sliced Wasserstein distance
                distances = []
                for _ in range(100):
                    theta = np.random.randn(z1.shape[1])
                    theta /= np.linalg.norm(theta)
                    
                    proj1 = z1 @ theta
                    proj2 = z2 @ theta
                    
                    dist = wasserstein_distance(proj1, proj2)
                    distances.append(dist)
                
                avg_dist = np.mean(distances)
                distance_matrix[i, j] = avg_dist
                distance_matrix[j, i] = avg_dist
    
    return distance_matrix

# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("="*80)
    print("EXPERIMENT 12: RG-GEODESIC METRIC LEARNING")
    print("="*80)
    print("\nTheoretical Framework:")
    print("  Cotler & Rezchikov (2022): RG flow as optimal transport")
    print("  Key insight: Coarse-graining minimizes dissipation")
    print("  Hypothesis: Geodesic metric respects universality classes")
    print("="*80)
    
    # Simulation parameters
    L = 128
    T = 500
    n_train_per_class = 75  # Train on BD + EDEN (150 total)
    n_test_per_class = 80
    
    # Initialize simulator
    simulator = GrowthModelSimulator(width=L, height=T)
    
    model_names = ['edwards_wilkinson', 'kpz_equation', 'ballistic_deposition', 'eden', 'random_deposition']
    
    # Generate training data (discrete KPZ class only)
    print("\n[1/5] Generating training data...")
    train_surfaces = []
    train_labels = []
    
    for model_idx, name in enumerate(['ballistic_deposition', 'eden']):
        print(f"  Training: {name} ({n_train_per_class} samples)")
        for _ in range(n_train_per_class):
            surface = simulator.generate_trajectory(name)
            train_surfaces.append(surface)
            train_labels.append(model_idx)
    
    print(f"  Total training samples: {len(train_surfaces)}")
    
    # Generate test data (all models)
    print("\n[2/5] Generating test data...")
    test_surfaces = []
    test_labels = []
    
    for model_idx, name in enumerate(model_names):
        print(f"  Testing: {name} ({n_test_per_class} samples)")
        for _ in range(n_test_per_class):
            surface = simulator.generate_trajectory(name)
            test_surfaces.append(surface)
            test_labels.append(model_idx)
    
    print(f"  Total test samples: {len(test_surfaces)}")
    
    # Create datasets with multi-scale RG flow
    blur_scales = [0, 1, 2, 4]  # σ values for Gaussian coarse-graining
    
    train_dataset = MultiScaleSurfaceDataset(train_surfaces, train_labels, blur_scales)
    test_dataset = MultiScaleSurfaceDataset(test_surfaces, test_labels, blur_scales)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train RG-geodesic autoencoder
    print("\n[3/5] Training RG-geodesic autoencoder...")
    print("  Loss components:")
    print("    L_recon: Standard reconstruction error")
    print("    L_geodesic: Triangle inequality penalty (Cotler-Rezchikov)")
    print("    L_smoothness: Adjacent scale similarity")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_rg_geodesic_autoencoder(
        train_loader, 
        latent_dim=32,
        num_epochs=50,
        gamma_geodesic=0.5,
        gamma_smoothness=0.1
    )
    
    # Compute Wasserstein distances in learned metric
    print("\n[4/5] Evaluating RG-geodesic metric...")
    distance_matrix = compute_pairwise_wasserstein(model, test_loader, model_names, device)
    
    # Analyze results
    print("\n[5/5] Analysis: Universality Class Separation")
    print("="*80)
    
    # Print distance matrix
    print("\nWasserstein Distance Matrix (RG-Geodesic Metric):")
    print("-" * 80)
    print(f"{'Model':<20}", end="")
    for name in model_names:
        print(f"{name[:5]:>8}", end="")
    print()
    print("-" * 80)
    
    for i, name1 in enumerate(model_names):
        print(f"{name1:<20}", end="")
        for j, name2 in enumerate(model_names):
            print(f"{distance_matrix[i, j]:>8.2f}", end="")
        print()
    print("-" * 80)
    
    # Key metrics
    ew_kpz = distance_matrix[0, 1]
    kpz_bd = distance_matrix[1, 2]
    kpz_eden = distance_matrix[1, 3]
    bd_eden = distance_matrix[2, 3]
    
    # KPZ-class spread (how far apart are models in same class?)
    kpz_class_distances = [kpz_bd, kpz_eden, bd_eden]
    kpz_class_spread = np.mean(kpz_class_distances)
    
    # EW separation from KPZ-class
    ew_to_kpz_class = [distance_matrix[0, 1], distance_matrix[0, 2], distance_matrix[0, 3]]
    ew_separation = np.mean(ew_to_kpz_class)
    
    # Universality ratio (want > 1.5x)
    universality_ratio = ew_separation / kpz_class_spread
    
    print("\n" + "="*80)
    print("KEY METRICS (Universality Class Detection)")
    print("="*80)
    print(f"EW ↔ KPZ distance:        {ew_kpz:.3f}")
    print(f"KPZ-class spread (avg):   {kpz_class_spread:.3f}")
    print(f"  KPZ ↔ BD:               {kpz_bd:.3f}")
    print(f"  KPZ ↔ EDEN:             {kpz_eden:.3f}")
    print(f"  BD ↔ EDEN:              {bd_eden:.3f}")
    print(f"\nEW separation (avg):      {ew_separation:.3f}")
    print(f"\n{'='*80}")
    print(f"UNIVERSALITY RATIO: {universality_ratio:.2f}x")
    print(f"SUCCESS CRITERION (>1.5x): {'✓ MET' if universality_ratio > 1.5 else '✗ NOT MET'}")
    print(f"{'='*80}")
    
    # Comparison to previous experiments
    print("\nCOMPARISON TO PREVIOUS EXPERIMENTS:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'Ratio':<10} {'Status':<10}")
    print("-" * 80)
    print(f"{'Exp 7b (Gradient AE)':<30} {'1.01x':<10} {'✗':<10}")
    print(f"{'Exp 8 (Physics-Informed)':<30} {'1.02x':<10} {'✗':<10}")
    print(f"{'Exp 9 (Wavelet on W(t))':<30} {'0.55x':<10} {'✗':<10}")
    print(f"{'Exp 10 (Skewness/Kurtosis)':<30} {'0.79x':<10} {'✗':<10}")
    print(f"{'Exp 11 (Gradient-Wavelet)':<30} {'0.64x':<10} {'✗':<10}")
    print(f"{'Exp 12 (RG-Geodesic)':<30} {f'{universality_ratio:.2f}x':<10} {'✓' if universality_ratio > 1.5 else '✗':<10}")
    print("-" * 80)
    
    # Visualization
    print("\n[Visualization] Generating figure...")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Distance matrix heatmap
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(distance_matrix, cmap='viridis', aspect='auto')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_yticks(range(len(model_names)))
    ax1.set_xticklabels([n.replace('_', ' ').title() for n in model_names], rotation=45, ha='right')
    ax1.set_yticklabels([n.replace('_', ' ').title() for n in model_names])
    ax1.set_title('Wasserstein Distance Matrix\n(RG-Geodesic Metric)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Distance')
    
    # Highlight universality classes
    from matplotlib.patches import Rectangle
    # KPZ class box (KPZ, BD, EDEN)
    rect = Rectangle((0.5, 0.5), 3, 3, linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    ax1.text(2, 3.7, 'KPZ Class', color='red', fontsize=10, ha='center', fontweight='bold')
    
    # 2. Training history
    ax2 = plt.subplot(2, 3, 2)
    epochs = range(1, len(history['total_loss']) + 1)
    ax2.plot(epochs, history['recon_loss'], label='Reconstruction Loss', linewidth=2)
    ax2.plot(epochs, history['rg_loss'], label='RG-Geodesic Loss', linewidth=2)
    ax2.plot(epochs, history['total_loss'], label='Total Loss', linewidth=2, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training History\n(RG-Aware Learning)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Universality ratio comparison
    ax3 = plt.subplot(2, 3, 3)
    exp_names = ['Exp 7b', 'Exp 8', 'Exp 9', 'Exp 10', 'Exp 11', 'Exp 12']
    ratios = [1.01, 1.02, 0.55, 0.79, 0.64, universality_ratio]
    colors = ['red' if r < 1.5 else 'green' for r in ratios]
    bars = ax3.bar(exp_names, ratios, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=1.5, color='blue', linestyle='--', linewidth=2, label='Target (1.5x)')
    ax3.set_ylabel('EW Separation / KPZ-Class Spread')
    ax3.set_title('Universality Ratio Across Experiments', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Rotate labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Within-class vs between-class distances
    ax4 = plt.subplot(2, 3, 4)
    
    within_kpz = [kpz_bd, kpz_eden, bd_eden]
    between_class = [ew_kpz, distance_matrix[0, 2], distance_matrix[0, 3],  # EW to KPZ-class
                     distance_matrix[4, 1], distance_matrix[4, 2], distance_matrix[4, 3]]  # RD to KPZ-class
    
    bp = ax4.boxplot([within_kpz, between_class], 
                      labels=['Within KPZ-Class', 'Between Classes'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_ylabel('Wasserstein Distance')
    ax4.set_title('Universality Class Structure\n(Lower = More Similar)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. RG flow trajectories (conceptual)
    ax5 = plt.subplot(2, 3, 5)
    
    # Illustrate the concept: RG flow should follow geodesics
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Fixed point (KPZ class)
    ax5.plot(0, 0, 'ro', markersize=15, label='KPZ Fixed Point')
    
    # RG trajectories for different models
    # KPZ, BD, EDEN converge to same point
    for offset, label in zip([0.3, 0.5, 0.7], ['KPZ', 'BD', 'EDEN']):
        t = np.linspace(offset, 0, 50)
        ax5.plot(t * np.cos(offset), t * np.sin(offset), linewidth=2, alpha=0.7, label=label)
    
    # EW converges to different fixed point
    t = np.linspace(0.6, 0.1, 50)
    ax5.plot(t * np.cos(np.pi), t * np.sin(np.pi), linewidth=2, alpha=0.7, label='EW', linestyle='--')
    ax5.plot(-0.1, 0, 'bs', markersize=12, label='EW Fixed Point')
    
    ax5.set_xlabel('Latent Dimension 1')
    ax5.set_ylabel('Latent Dimension 2')
    ax5.set_title('Conceptual RG Flow\n(Geodesics in Learned Metric)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_aspect('equal')
    ax5.grid(alpha=0.3)
    
    # 6. Distance distribution
    ax6 = plt.subplot(2, 3, 6)
    
    # Flatten upper triangle of distance matrix (excluding diagonal)
    all_distances = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            all_distances.append(distance_matrix[i, j])
    
    ax6.hist(all_distances, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.axvline(x=ew_kpz, color='red', linestyle='--', linewidth=2, label=f'EW↔KPZ: {ew_kpz:.2f}')
    ax6.axvline(x=kpz_class_spread, color='green', linestyle='--', linewidth=2, 
                label=f'KPZ-class spread: {kpz_class_spread:.2f}')
    ax6.set_xlabel('Wasserstein Distance')
    ax6.set_ylabel('Count')
    ax6.set_title('Distribution of Pairwise Distances', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'exp12_rg_geodesic_metric.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    if universality_ratio > 1.5:
        print("  ✓ RG-geodesic metric successfully separates universality classes!")
        print("  ✓ Cotler-Rezchikov framework validated experimentally")
        print("  ✓ Optimal transport + RG flow = correct universality structure")
    else:
        print("  ✗ RG-geodesic metric did not achieve target separation")
        print("  → Possible causes:")
        print("    - Finite-size effects still dominate")
        print("    - Need stronger RG regularization")
        print("    - Coarse-graining scales not optimal")
    
    print("\nNext Steps:")
    print("  1. Analyze sensitivity to γ_geodesic and γ_smoothness hyperparameters")
    print("  2. Test with more RG scales (σ ∈ {0, 0.5, 1, 2, 4, 8})")
    print("  3. Incorporate explicit Tracy-Widom constraints (Theorem 5)")
    print("  4. Extend to height-height correlation functions")

if __name__ == "__main__":
    main()
