"""
Experiment 11: Wavelet Decomposition on Spatial Gradients
=========================================================

Key Insight from Exp 10:
- Skewness/kurtosis features extracted from height fluctuations
- ALL models showed skewness ≈ 0 (far from TW target of 0.29)
- Diagnosis: Simulations too short for Tracy-Widom convergence

This Experiment:
- Apply wavelet decomposition to SPATIAL GRADIENTS (∇h), not just W(t)
- Spatial gradients encode local correlations and roughness structure
- Different universality classes have different spatial correlation patterns

Theory:
- KPZ class: Strong spatial correlations, smooth gradients at coarse scales
- EW class: Different correlation length scaling
- Wavelet on gradients captures multi-scale spatial structure

Method:
1. Compute spatial gradient field ∇h = h(x+1,t) - h(x,t)
2. Wavelet decompose gradient field at FINAL TIME (saturation)
3. Train autoencoder on wavelet coefficients
4. Compute Wasserstein distances

Success Criterion: EW/KPZ separation ratio > 1.5x
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
from scipy import stats
from scipy.ndimage import gaussian_filter
import pywt

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
WIDTH = 128
TIME_STEPS = 500
N_TRAIN_PER_CLASS = 150
N_TEST = 80
EPOCHS = 30
LATENT_DIM = 16
WAVELET = 'db4'
MAX_LEVEL = 5

# Training models (discrete KPZ class)
TRAIN_MODELS = ['ballistic_deposition', 'eden']

# Test models
TEST_MODELS = ['edwards_wilkinson', 'kpz_equation', 'ballistic_deposition', 'eden', 'random_deposition']


def compute_spatial_gradient(surface, axis=0):
    """
    Compute spatial gradient of surface at each time step.
    
    ∇h(x,t) = h(x+1,t) - h(x,t)
    """
    return np.roll(surface, -1, axis=axis) - surface


def wavelet_decompose_1d(signal, wavelet='db4', level=5):
    """Wavelet decompose 1D signal."""
    return pywt.wavedec(signal, wavelet, level=level)


def extract_gradient_wavelet_features(surface, wavelet='db4', level=5, time_fraction=0.8):
    """
    Extract wavelet features from spatial gradients at late times.
    
    Args:
        surface: (width, time_steps) array
        wavelet: Wavelet type
        level: Decomposition levels
        time_fraction: Use gradients from this fraction onwards
    
    Returns:
        Dictionary with features at each scale
    """
    n_times = surface.shape[1]
    start_t = int(time_fraction * n_times)
    
    # Compute average gradient profile at late times
    # This gives a cleaner signal than single time point
    late_gradients = []
    for t in range(start_t, n_times):
        grad = compute_spatial_gradient(surface[:, t])
        late_gradients.append(grad)
    
    # Average gradient profile (spatial structure)
    avg_gradient = np.mean(late_gradients, axis=0)
    
    # Also compute gradient variance profile (roughness structure)
    var_gradient = np.var(late_gradients, axis=0)
    
    # Wavelet decompose both
    coeffs_avg = wavelet_decompose_1d(avg_gradient, wavelet, level)
    coeffs_var = wavelet_decompose_1d(var_gradient, wavelet, level)
    
    features = {
        'avg_gradient': avg_gradient,
        'var_gradient': var_gradient,
        'wavelet_coeffs_avg': coeffs_avg,
        'wavelet_coeffs_var': coeffs_var,
    }
    
    # Extract summary statistics per scale
    for i, c in enumerate(coeffs_avg):
        scale_name = 'approx' if i == 0 else f'detail_{level - i + 1}'
        features[f'{scale_name}_avg_energy'] = np.sum(c**2)
        features[f'{scale_name}_avg_mean'] = np.mean(np.abs(c))
        features[f'{scale_name}_var_energy'] = np.sum(coeffs_var[i]**2)
    
    return features


def extract_feature_vector(surface, wavelet='db4', level=5):
    """
    Extract fixed-size feature vector from surface.
    
    Combines:
    1. Wavelet coefficients of averaged spatial gradients
    2. Statistics at each scale
    3. Interface width trajectory features
    """
    n_times = surface.shape[1]
    
    # Get gradient-based wavelet features
    grad_features = extract_gradient_wavelet_features(surface, wavelet, level)
    
    # Feature vector components:
    features = []
    
    # 1. Wavelet energy at each scale (both avg and var gradients)
    for i in range(level + 1):
        scale_name = 'approx' if i == 0 else f'detail_{level - i + 1}'
        features.append(grad_features.get(f'{scale_name}_avg_energy', 0))
        features.append(grad_features.get(f'{scale_name}_var_energy', 0))
    
    # 2. Statistics of gradient field
    avg_grad = grad_features['avg_gradient']
    var_grad = grad_features['var_gradient']
    features.extend([
        np.mean(avg_grad),
        np.std(avg_grad),
        stats.skew(avg_grad),
        stats.kurtosis(avg_grad),
        np.mean(var_grad),
        np.std(var_grad),
    ])
    
    # 3. Interface width at late times
    widths = []
    for t in range(n_times):
        h_t = surface[:, t]
        w_t = np.sqrt(np.mean((h_t - np.mean(h_t))**2))
        widths.append(w_t)
    
    # Width statistics
    widths = np.array(widths)
    late_widths = widths[int(0.5 * n_times):]
    features.extend([
        np.mean(late_widths),
        np.std(late_widths),
        widths[-1] / (widths[int(0.2*n_times)] + 1e-8),  # Growth ratio
    ])
    
    # 4. Padded wavelet coefficients (truncated to fixed size)
    max_coeffs = 32  # Fixed size per scale
    for coeffs in grad_features['wavelet_coeffs_avg']:
        if len(coeffs) >= max_coeffs:
            features.extend(coeffs[:max_coeffs])
        else:
            features.extend(coeffs)
            features.extend([0] * (max_coeffs - len(coeffs)))
    
    return np.array(features)


def generate_feature_dataset(model_types, n_per_class, desc="Generating"):
    """Generate surfaces and extract wavelet gradient features."""
    all_features = []
    all_labels = []
    
    for class_idx, model_type in enumerate(model_types):
        print(f"\n  {model_type}...")
        for i in tqdm(range(n_per_class), desc=f"  {model_type}"):
            sim = GrowthModelSimulator(WIDTH, TIME_STEPS)
            surface = sim.generate_trajectory(model_type)
            
            features = extract_feature_vector(surface)
            all_features.append(features)
            all_labels.append(class_idx)
    
    return np.array(all_features), np.array(all_labels)


class GradientWaveletAutoencoder(nn.Module):
    """Autoencoder for gradient-wavelet features."""
    
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(train_features, latent_dim=16, epochs=30, batch_size=32):
    """Train autoencoder on features."""
    # Normalize
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-8
    train_norm = (train_features - mean) / std
    
    input_dim = train_features.shape[1]
    model = GradientWaveletAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    X = torch.tensor(train_norm, dtype=torch.float32)
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        perm = torch.randperm(X.shape[0])
        X_shuffled = X[perm]
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, X.shape[0], batch_size):
            batch = X_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            recon, z = model(batch)
            loss = F.mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
    
    return model, mean, std


def encode_features(model, features, mean, std):
    """Encode features to latent space."""
    features_norm = (features - mean) / (std + 1e-8)
    X = torch.tensor(features_norm, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        z = model.encode(X)
    
    return z.numpy()


def sliced_wasserstein_distance(X, Y, n_projections=100):
    """Compute sliced Wasserstein distance."""
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same dimensionality")
    
    d = X.shape[1]
    distances = []
    
    for _ in range(n_projections):
        theta = np.random.randn(d)
        theta = theta / np.linalg.norm(theta)
        
        X_proj = X @ theta
        Y_proj = Y @ theta
        
        X_sorted = np.sort(X_proj)
        Y_sorted = np.sort(Y_proj)
        
        n_X, n_Y = len(X_sorted), len(Y_sorted)
        if n_X != n_Y:
            n_common = min(n_X, n_Y)
            X_interp = np.interp(np.linspace(0, 1, n_common), 
                                  np.linspace(0, 1, n_X), X_sorted)
            Y_interp = np.interp(np.linspace(0, 1, n_common),
                                  np.linspace(0, 1, n_Y), Y_sorted)
            distances.append(np.mean(np.abs(X_interp - Y_interp)))
        else:
            distances.append(np.mean(np.abs(X_sorted - Y_sorted)))
    
    return np.mean(distances)


def analyze_raw_features(features, labels, model_names):
    """Analyze raw feature statistics without autoencoder."""
    print("\n[Raw Feature Analysis]")
    print("-" * 60)
    
    # Feature names (rough approximation)
    n_scales = MAX_LEVEL + 1
    
    for model_idx, model_name in enumerate(model_names):
        mask = labels == model_idx
        model_features = features[mask]
        
        # Energy at different scales
        print(f"\n{model_name}:")
        for scale_idx in range(n_scales):
            avg_energy_idx = scale_idx * 2
            var_energy_idx = scale_idx * 2 + 1
            
            scale_name = 'approx' if scale_idx == 0 else f'detail_{MAX_LEVEL - scale_idx + 1}'
            
            if avg_energy_idx < model_features.shape[1]:
                avg_energy = model_features[:, avg_energy_idx]
                print(f"  {scale_name} energy: {np.mean(avg_energy):.3f} ± {np.std(avg_energy):.3f}")


def main():
    print("=" * 70)
    print("EXPERIMENT 11: Wavelet Decomposition on Spatial Gradients")
    print("=" * 70)
    print("""
Key Insight from Exp 10:
  - Height fluctuation skewness ≈ 0 for ALL models (not TW convergence)
  - Need different approach: spatial gradient structure

This Experiment:
  - Wavelet decompose spatial gradients ∇h at late times
  - Captures multi-scale spatial correlation structure
  - Different universality classes → different spatial patterns

Target: EW/KPZ separation ratio > 1.5x
""")
    print("=" * 70)
    
    # =========================================================================
    # [1] Generate training data
    # =========================================================================
    print("\n[1] Generating training data...")
    
    train_features, train_labels = generate_feature_dataset(
        TRAIN_MODELS, N_TRAIN_PER_CLASS, desc="Training"
    )
    
    print(f"\n  Training samples: {len(train_features)}")
    print(f"  Feature dimension: {train_features.shape[1]}")
    
    # =========================================================================
    # [2] Generate test data
    # =========================================================================
    print("\n[2] Generating test data...")
    
    test_features, test_labels = generate_feature_dataset(
        TEST_MODELS, N_TEST, desc="Test"
    )
    
    print(f"\n  Test samples per class: {N_TEST}")
    print(f"  Feature dimension: {test_features.shape[1]}")
    
    # =========================================================================
    # [3] Train autoencoder
    # =========================================================================
    print("\n[3] Training autoencoder on gradient-wavelet features...")
    
    model, mean, std = train_autoencoder(train_features, LATENT_DIM, EPOCHS)
    
    # =========================================================================
    # [4] Encode test data
    # =========================================================================
    print("\n[4] Encoding test data to latent space...")
    
    latents = {}
    for i, model_type in enumerate(TEST_MODELS):
        mask = test_labels == i
        latents[model_type] = encode_features(model, test_features[mask], mean, std)
    
    # =========================================================================
    # [5] Compute Wasserstein distances
    # =========================================================================
    print("\n[5] Computing Wasserstein distances...")
    
    n_models = len(TEST_MODELS)
    dist_matrix = np.zeros((n_models, n_models))
    
    for i, m1 in enumerate(TEST_MODELS):
        for j, m2 in enumerate(TEST_MODELS):
            if i < j:
                d = sliced_wasserstein_distance(latents[m1], latents[m2])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    
    # =========================================================================
    # [6] Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Print distance matrix
    print("\n[A] Wasserstein Distance Matrix (Latent Space)")
    print("-" * 60)
    print("          " + " ".join([f"{m[:8]:>9}" for m in TEST_MODELS]))
    for i, m1 in enumerate(TEST_MODELS):
        row = [f"{dist_matrix[i,j]:9.3f}" for j in range(n_models)]
        print(f"{m1[:10]:10}" + " ".join(row))
    
    # Key metrics
    ew_idx = TEST_MODELS.index('edwards_wilkinson')
    kpz_idx = TEST_MODELS.index('kpz_equation')
    bd_idx = TEST_MODELS.index('ballistic_deposition')
    eden_idx = TEST_MODELS.index('eden')
    rd_idx = TEST_MODELS.index('random_deposition')
    
    d_ew_kpz = dist_matrix[ew_idx, kpz_idx]
    d_kpz_bd = dist_matrix[kpz_idx, bd_idx]
    d_kpz_eden = dist_matrix[kpz_idx, eden_idx]
    d_bd_eden = dist_matrix[bd_idx, eden_idx]
    d_ew_bd = dist_matrix[ew_idx, bd_idx]
    d_ew_eden = dist_matrix[ew_idx, eden_idx]
    
    # KPZ-class cohesion
    kpz_class_spread = (d_kpz_bd + d_bd_eden + d_kpz_eden) / 3
    
    # EW separation
    ew_separation = (d_ew_kpz + d_ew_bd + d_ew_eden) / 3
    
    # Universality ratio
    ratio = ew_separation / (kpz_class_spread + 1e-6)
    
    print("\n[B] Key Metrics")
    print("-" * 60)
    print(f"  EW ↔ KPZ distance:        {d_ew_kpz:.3f}")
    print(f"  KPZ ↔ BD distance:        {d_kpz_bd:.3f}")
    print(f"  KPZ ↔ EDEN distance:      {d_kpz_eden:.3f}")
    print(f"  BD ↔ EDEN distance:       {d_bd_eden:.3f}")
    print(f"  KPZ-class spread (avg):   {kpz_class_spread:.3f}")
    print(f"  EW separation (avg):      {ew_separation:.3f}")
    print(f"\n  UNIVERSALITY RATIO:       {ratio:.2f}x")
    
    # Compare to baselines
    print("\n[C] Comparison to Previous Experiments")
    print("-" * 60)
    print(f"  Exp 7b baseline:          1.01x")
    print(f"  Exp 8 Physics-Informed:   1.02x")
    print(f"  Exp 9 Wavelet on W(t):    ~0.55x")
    print(f"  Exp 10 Skewness/Kurtosis: 0.79x")
    print(f"  THIS EXPERIMENT:          {ratio:.2f}x")
    
    success = ratio > 1.5
    print(f"\n  SUCCESS CRITERION (>1.5x): {'✅ PASSED' if success else '❌ NOT MET'}")
    
    # Additional analysis
    print("\n[D] Within KPZ-Class Analysis")
    print("-" * 60)
    print(f"  KPZ (continuum) ↔ BD (discrete):   {d_kpz_bd:.3f}")
    print(f"  KPZ (continuum) ↔ EDEN (discrete): {d_kpz_eden:.3f}")
    print(f"  BD ↔ EDEN (both discrete):         {d_bd_eden:.3f}")
    
    # Check if KPZ is closer to discrete models than EW is
    kpz_to_discrete = (d_kpz_bd + d_kpz_eden) / 2
    ew_to_discrete = (d_ew_bd + d_ew_eden) / 2
    
    print(f"\n  Avg KPZ → discrete:       {kpz_to_discrete:.3f}")
    print(f"  Avg EW → discrete:        {ew_to_discrete:.3f}")
    print(f"  EW/KPZ distance ratio:    {ew_to_discrete / (kpz_to_discrete + 1e-6):.2f}x")
    
    if ew_to_discrete > kpz_to_discrete * 1.1:
        print(f"  ✓ EW is more distant from discrete KPZ-class models")
    else:
        print(f"  ⚠ EW/KPZ distinction not clear in gradient features")
    
    # =========================================================================
    # [7] Save figures
    # =========================================================================
    print("\n[7] Saving figures...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Wasserstein distance matrix
    im1 = axes[0, 0].imshow(dist_matrix, cmap='viridis')
    axes[0, 0].set_xticks(range(n_models))
    axes[0, 0].set_yticks(range(n_models))
    axes[0, 0].set_xticklabels([m[:6] for m in TEST_MODELS], rotation=45, ha='right')
    axes[0, 0].set_yticklabels([m[:6] for m in TEST_MODELS])
    axes[0, 0].set_title('Wasserstein Distances (Gradient-Wavelet Latent)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Key distance comparison
    metrics = ['EW↔KPZ', 'KPZ↔BD', 'KPZ↔EDEN', 'BD↔EDEN']
    values = [d_ew_kpz, d_kpz_bd, d_kpz_eden, d_bd_eden]
    colors = ['blue', 'orange', 'green', 'purple']
    axes[0, 1].bar(range(len(metrics)), values, color=colors, alpha=0.7)
    axes[0, 1].set_xticks(range(len(metrics)))
    axes[0, 1].set_xticklabels(metrics, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Wasserstein Distance')
    axes[0, 1].set_title('Key Pairwise Distances')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: 2D latent space (PCA)
    from sklearn.decomposition import PCA
    all_latents = np.vstack([latents[m] for m in TEST_MODELS])
    all_labels_viz = np.concatenate([[i]*len(latents[m]) for i, m in enumerate(TEST_MODELS)])
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)
    
    colors_viz = ['blue', 'red', 'green', 'orange', 'purple']
    for i, model_name in enumerate(TEST_MODELS):
        mask = all_labels_viz == i
        axes[1, 0].scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                          c=colors_viz[i], label=model_name[:8], alpha=0.6, s=30)
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Latent Space (PCA)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Summary metrics
    summary_metrics = ['KPZ spread', 'EW sep', 'Ratio']
    summary_values = [kpz_class_spread, ew_separation, ratio]
    axes[1, 1].bar(range(len(summary_metrics)), summary_values, 
                   color=['red', 'blue', 'black'], alpha=0.7)
    axes[1, 1].set_xticks(range(len(summary_metrics)))
    axes[1, 1].set_xticklabels(summary_metrics)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Universality Metrics')
    axes[1, 1].axhline(y=1.5, color='green', linestyle='--', label='Target (1.5x)')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'exp11_gradient_wavelet.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Experiment 11: Wavelet Decomposition on Spatial Gradients

Method:
  - Extracted wavelet features from spatial gradients ∇h
  - Multi-scale decomposition captures spatial correlation structure
  
Results:
  - EW↔KPZ distance:      {d_ew_kpz:.3f}
  - KPZ-class spread:     {kpz_class_spread:.3f}
  - EW separation:        {ew_separation:.3f}
  - UNIVERSALITY RATIO:   {ratio:.2f}x {'✅ > 1.5x target!' if ratio > 1.5 else '❌ < 1.5x target'}

Key Finding:
  - Gradient-wavelet features capture spatial structure
  - {'EW is clearly more distant from discrete KPZ-class' if ew_to_discrete > kpz_to_discrete * 1.1 else 'EW/KPZ distinction still challenging'}

{'SUCCESS: Gradient-wavelet features achieve universality separation!' if ratio > 1.5 else 'NEXT STEP: Need longer time series or direct Tracy-Widom measurement'}
""")


if __name__ == "__main__":
    main()
