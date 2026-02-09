"""
Experiment 10: Skewness/Kurtosis Features for Universality Detection
====================================================================

Key Insight from Exp 8/9 Diagnosis:
- Both experiments failed to separate EW/KPZ because gradient features and W(t)
  don't encode the β exponent difference (β=1/4 vs β=1/3)
- However, Theorem 5 proves: Tracy-Widom (KPZ class) vs Gaussian (EW class)
  distributions differ in SKEWNESS: TW skew ≈ 0.29, Gaussian skew = 0

This Experiment:
- Extract skewness, kurtosis, and higher moments from height fluctuation distributions
- These directly encode Tracy-Widom signature per Theorem 5
- Should dramatically improve EW/KPZ separation

Theory (from paper):
- KPZ universality: Height fluctuations ~ Tracy-Widom distribution
- EW universality: Height fluctuations ~ Gaussian distribution
- Tracy-Widom: mean=-1.77, variance=0.81, skewness=0.29, kurtosis=0.17

Success Criterion:
- EW/KPZ separation ratio > 1.5x (was ~1.0x in Exp 8/9)
- Skewness/kurtosis features should cluster KPZ-class models together
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

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
WIDTH = 128
TIME_STEPS = 500
N_TRAIN_PER_CLASS = 150
N_TEST = 80
EPOCHS = 30
LATENT_DIM = 16
SIGMA = 2  # Coarse-graining parameter

# Training models (discrete KPZ class)
TRAIN_MODELS = ['ballistic_deposition', 'eden']

# Test models
TEST_MODELS = ['edwards_wilkinson', 'kpz_equation', 'ballistic_deposition', 'eden', 'random_deposition']

# Tracy-Widom reference values (TW₂ distribution)
TW_MEAN = -1.77
TW_VAR = 0.81
TW_SKEWNESS = 0.29
TW_KURTOSIS = 0.17  # Excess kurtosis

# Gaussian reference
GAUSSIAN_SKEWNESS = 0.0
GAUSSIAN_KURTOSIS = 0.0  # Excess kurtosis


def extract_height_fluctuation_statistics(surface, time_fraction=0.5):
    """
    Extract statistical moments from height fluctuation distribution at late times.
    
    Per KPZ theory, after transient, height fluctuations h - <h> should follow
    either Tracy-Widom (KPZ class) or Gaussian (EW class) distribution.
    
    Args:
        surface: Shape (width, time_steps)
        time_fraction: Use data from this fraction onwards (skip transient)
    
    Returns:
        Dictionary with statistical features
    """
    n_times = surface.shape[1]
    start_t = int(time_fraction * n_times)
    
    # Collect height fluctuations from late times
    all_fluctuations = []
    for t in range(start_t, n_times):
        h_t = surface[:, t]
        mean_h = np.mean(h_t)
        fluctuations = h_t - mean_h
        all_fluctuations.extend(fluctuations)
    
    all_fluctuations = np.array(all_fluctuations)
    
    # Normalize by standard deviation for comparison with TW
    std_h = np.std(all_fluctuations)
    if std_h > 0:
        normalized_fluctuations = all_fluctuations / std_h
    else:
        normalized_fluctuations = all_fluctuations
    
    # Compute moments
    features = {
        'mean': np.mean(normalized_fluctuations),
        'std': np.std(normalized_fluctuations),
        'skewness': stats.skew(normalized_fluctuations),
        'kurtosis': stats.kurtosis(normalized_fluctuations),  # Excess kurtosis
        'min': np.min(normalized_fluctuations),
        'max': np.max(normalized_fluctuations),
        'median': np.median(normalized_fluctuations),
        'iqr': np.percentile(normalized_fluctuations, 75) - np.percentile(normalized_fluctuations, 25),
        # Quantiles for distribution shape
        'q10': np.percentile(normalized_fluctuations, 10),
        'q25': np.percentile(normalized_fluctuations, 25),
        'q75': np.percentile(normalized_fluctuations, 75),
        'q90': np.percentile(normalized_fluctuations, 90),
    }
    
    # Distance to Tracy-Widom distribution
    features['dist_to_tw'] = np.sqrt(
        (features['skewness'] - TW_SKEWNESS)**2 + 
        (features['kurtosis'] - TW_KURTOSIS)**2
    )
    
    # Distance to Gaussian
    features['dist_to_gaussian'] = np.sqrt(
        features['skewness']**2 + 
        features['kurtosis']**2
    )
    
    return features


def extract_spatiotemporal_statistics(surface, n_time_windows=5):
    """
    Extract statistics at multiple time windows to capture evolution.
    
    Returns:
        Feature vector capturing temporal evolution of moments
    """
    n_times = surface.shape[1]
    window_size = n_times // n_time_windows
    
    feature_evolution = []
    
    for w in range(n_time_windows):
        start_t = w * window_size
        end_t = (w + 1) * window_size
        
        # Get fluctuations in this window
        fluctuations = []
        for t in range(start_t, min(end_t, n_times)):
            h_t = surface[:, t]
            mean_h = np.mean(h_t)
            fluctuations.extend(h_t - mean_h)
        
        fluctuations = np.array(fluctuations)
        if len(fluctuations) > 10 and np.std(fluctuations) > 0:
            fluctuations = fluctuations / np.std(fluctuations)
            feature_evolution.extend([
                np.mean(fluctuations),
                np.std(fluctuations),
                stats.skew(fluctuations),
                stats.kurtosis(fluctuations)
            ])
        else:
            feature_evolution.extend([0, 1, 0, 0])
    
    return np.array(feature_evolution)


def compute_interface_width(surface):
    """Compute time-dependent interface width W(t)."""
    widths = []
    for t in range(surface.shape[1]):
        h_t = surface[:, t]
        w_t = np.sqrt(np.mean((h_t - np.mean(h_t))**2))
        widths.append(w_t)
    return np.array(widths)


def fit_scaling_exponent(widths, skip_transient=0.2):
    """Fit W(t) ~ t^β to extract growth exponent."""
    n_times = len(widths)
    start_idx = int(skip_transient * n_times)
    
    t_vals = np.arange(start_idx, n_times) + 1
    w_vals = widths[start_idx:]
    
    valid = w_vals > 0
    if np.sum(valid) < 10:
        return 0.25, 1.0, 0.0
    
    log_t = np.log(t_vals[valid])
    log_w = np.log(w_vals[valid])
    
    slope, intercept, r_value, _, _ = stats.linregress(log_t, log_w)
    
    return slope, np.exp(intercept), r_value**2


def generate_feature_dataset(model_types, n_per_class, desc="Generating"):
    """
    Generate surfaces and extract moment-based features.
    
    Returns:
        features: (n_samples, n_features) array with skewness/kurtosis features
        raw_features: Dictionary with all extracted statistics
        labels: class labels
        model_names: names for each sample
    """
    all_features = []
    all_raw_features = []
    all_labels = []
    all_model_names = []
    
    for class_idx, model_type in enumerate(model_types):
        print(f"\n  {model_type}...")
        for i in tqdm(range(n_per_class), desc=f"  {model_type}"):
            sim = GrowthModelSimulator(WIDTH, TIME_STEPS)
            surface = sim.generate_trajectory(model_type)
            
            # Extract height fluctuation statistics
            stats_dict = extract_height_fluctuation_statistics(surface)
            
            # Extract spatiotemporal statistics
            st_features = extract_spatiotemporal_statistics(surface)
            
            # Extract interface width features
            widths = compute_interface_width(surface)
            beta, _, r2 = fit_scaling_exponent(widths)
            
            # Combine into feature vector
            # Key features: skewness, kurtosis, beta, distances to TW/Gaussian
            feature_vec = np.array([
                stats_dict['skewness'],
                stats_dict['kurtosis'],
                beta,
                stats_dict['dist_to_tw'],
                stats_dict['dist_to_gaussian'],
                stats_dict['iqr'],
                stats_dict['q10'],
                stats_dict['q90'],
            ])
            
            # Append spatiotemporal evolution features
            feature_vec = np.concatenate([feature_vec, st_features])
            
            all_features.append(feature_vec)
            all_raw_features.append(stats_dict)
            all_labels.append(class_idx)
            all_model_names.append(model_type)
    
    return (np.array(all_features), all_raw_features, 
            np.array(all_labels), all_model_names)


class MomentAutoencoder(nn.Module):
    """
    Simple autoencoder for moment-based features.
    """
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(model, train_features, epochs=30, batch_size=32, lr=1e-3):
    """Train autoencoder on moment features."""
    # Normalize
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0) + 1e-8
    train_norm = (train_features - mean) / std
    
    X = torch.tensor(train_norm, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
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


def main():
    print("=" * 70)
    print("EXPERIMENT 10: Skewness/Kurtosis Features for Universality Detection")
    print("=" * 70)
    print("""
Key Insight:
  - Exp 8/9 failed because gradient/W(t) features don't encode β difference
  - BUT: Theorem 5 proves Tracy-Widom (KPZ) vs Gaussian (EW) differ in SKEWNESS
  - Tracy-Widom skewness ≈ 0.29, Gaussian skewness = 0
  
This Experiment:
  - Extract skewness, kurtosis from height fluctuation distributions
  - These DIRECTLY encode universality class identity
  
Target: EW/KPZ separation ratio > 1.5x (was ~1.0x in Exp 8/9)
""")
    print("=" * 70)
    
    # =========================================================================
    # [1] Generate training data
    # =========================================================================
    print("\n[1] Generating training data with moment features...")
    
    train_features, train_raw, train_labels, train_names = generate_feature_dataset(
        TRAIN_MODELS, N_TRAIN_PER_CLASS, desc="Training"
    )
    
    print(f"\n  Training samples: {len(train_features)}")
    print(f"  Feature dimension: {train_features.shape[1]}")
    
    # Show training feature statistics
    for i, model in enumerate(TRAIN_MODELS):
        mask = train_labels == i
        skew_vals = [train_raw[j]['skewness'] for j in range(len(train_raw)) if mask[j]]
        kurt_vals = [train_raw[j]['kurtosis'] for j in range(len(train_raw)) if mask[j]]
        print(f"  {model}: skewness={np.mean(skew_vals):.3f}±{np.std(skew_vals):.3f}, "
              f"kurtosis={np.mean(kurt_vals):.3f}±{np.std(kurt_vals):.3f}")
    
    # =========================================================================
    # [2] Generate test data
    # =========================================================================
    print("\n[2] Generating test data...")
    
    test_features, test_raw, test_labels, test_names = generate_feature_dataset(
        TEST_MODELS, N_TEST, desc="Test"
    )
    
    print(f"\n  Test samples per class: {N_TEST}")
    for i, model in enumerate(TEST_MODELS):
        mask = test_labels == i
        skew_vals = [test_raw[j]['skewness'] for j in range(len(test_raw)) if mask[j]]
        kurt_vals = [test_raw[j]['kurtosis'] for j in range(len(test_raw)) if mask[j]]
        tw_dists = [test_raw[j]['dist_to_tw'] for j in range(len(test_raw)) if mask[j]]
        gauss_dists = [test_raw[j]['dist_to_gaussian'] for j in range(len(test_raw)) if mask[j]]
        print(f"  {model}:")
        print(f"    skewness: {np.mean(skew_vals):.3f} ± {np.std(skew_vals):.3f}")
        print(f"    kurtosis: {np.mean(kurt_vals):.3f} ± {np.std(kurt_vals):.3f}")
        print(f"    dist_to_TW: {np.mean(tw_dists):.3f}, dist_to_Gaussian: {np.mean(gauss_dists):.3f}")
    
    # =========================================================================
    # [3] Train autoencoder on moment features
    # =========================================================================
    print("\n[3] Training autoencoder on moment features...")
    
    input_dim = train_features.shape[1]
    model = MomentAutoencoder(input_dim, LATENT_DIM)
    
    model, mean, std = train_autoencoder(model, train_features, epochs=EPOCHS)
    
    # =========================================================================
    # [4] Encode test data to latent space
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
    # [6] Analyze results
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
    d_bd_eden = dist_matrix[bd_idx, eden_idx]
    d_ew_bd = dist_matrix[ew_idx, bd_idx]
    d_ew_eden = dist_matrix[ew_idx, eden_idx]
    
    # KPZ-class cohesion
    kpz_class_spread = (d_kpz_bd + d_bd_eden + dist_matrix[kpz_idx, eden_idx]) / 3
    
    # EW separation
    ew_separation = (d_ew_kpz + d_ew_bd + d_ew_eden) / 3
    
    # Universality ratio
    ratio = ew_separation / (kpz_class_spread + 1e-6)
    
    print("\n[B] Key Metrics")
    print("-" * 60)
    print(f"  EW ↔ KPZ distance:        {d_ew_kpz:.3f}")
    print(f"  KPZ ↔ BD distance:        {d_kpz_bd:.3f}")
    print(f"  BD ↔ EDEN distance:       {d_bd_eden:.3f}")
    print(f"  KPZ-class spread (avg):   {kpz_class_spread:.3f}")
    print(f"  EW separation (avg):      {ew_separation:.3f}")
    print(f"\n  UNIVERSALITY RATIO:       {ratio:.2f}x")
    
    # Compare to baselines
    print("\n[C] Comparison to Previous Experiments")
    print("-" * 60)
    print(f"  Exp 7b baseline:          1.01x")
    print(f"  Exp 8 Physics-Informed:   1.02x")
    print(f"  Exp 9 Wavelet:            ~0.55x (inverted)")
    print(f"  THIS EXPERIMENT:          {ratio:.2f}x")
    
    success = ratio > 1.5
    print(f"\n  SUCCESS CRITERION (>1.5x): {'✅ PASSED' if success else '❌ NOT MET'}")
    
    # Direct skewness comparison
    print("\n[D] Direct Skewness/Kurtosis Analysis (no autoencoder)")
    print("-" * 60)
    
    # Collect skewness for each class
    skewness_by_class = {}
    kurtosis_by_class = {}
    for i, model in enumerate(TEST_MODELS):
        mask = test_labels == i
        skewness_by_class[model] = [test_raw[j]['skewness'] for j in range(len(test_raw)) if mask[j]]
        kurtosis_by_class[model] = [test_raw[j]['kurtosis'] for j in range(len(test_raw)) if mask[j]]
    
    print(f"{'Model':<20} {'Skewness':<20} {'Kurtosis':<20} {'TW Distance':<15}")
    print("-" * 75)
    for model in TEST_MODELS:
        s_mean, s_std = np.mean(skewness_by_class[model]), np.std(skewness_by_class[model])
        k_mean, k_std = np.mean(kurtosis_by_class[model]), np.std(kurtosis_by_class[model])
        tw_dist = np.sqrt((s_mean - TW_SKEWNESS)**2 + (k_mean - TW_KURTOSIS)**2)
        print(f"{model:<20} {s_mean:>6.3f} ± {s_std:<6.3f}   {k_mean:>6.3f} ± {k_std:<6.3f}   {tw_dist:<15.3f}")
    
    print(f"\n  Tracy-Widom reference:    skewness = {TW_SKEWNESS}, kurtosis = {TW_KURTOSIS}")
    print(f"  Gaussian reference:       skewness = {GAUSSIAN_SKEWNESS}, kurtosis = {GAUSSIAN_KURTOSIS}")
    
    # =========================================================================
    # [7] Save figures
    # =========================================================================
    print("\n[7] Saving figures...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Wasserstein distance matrix
    im1 = axes[0, 0].imshow(dist_matrix, cmap='viridis')
    axes[0, 0].set_xticks(range(n_models))
    axes[0, 0].set_yticks(range(n_models))
    axes[0, 0].set_xticklabels([m[:6] for m in TEST_MODELS], rotation=45, ha='right')
    axes[0, 0].set_yticklabels([m[:6] for m in TEST_MODELS])
    axes[0, 0].set_title('Wasserstein Distances (Latent)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Skewness by class
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    positions = np.arange(len(TEST_MODELS))
    skew_means = [np.mean(skewness_by_class[m]) for m in TEST_MODELS]
    skew_stds = [np.std(skewness_by_class[m]) for m in TEST_MODELS]
    axes[0, 1].bar(positions, skew_means, yerr=skew_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 1].axhline(y=TW_SKEWNESS, color='red', linestyle='--', label=f'Tracy-Widom ({TW_SKEWNESS})')
    axes[0, 1].axhline(y=GAUSSIAN_SKEWNESS, color='blue', linestyle='--', label='Gaussian (0)')
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels([m[:6] for m in TEST_MODELS], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Skewness')
    axes[0, 1].set_title('Height Fluctuation Skewness')
    axes[0, 1].legend()
    
    # Plot 3: Kurtosis by class
    kurt_means = [np.mean(kurtosis_by_class[m]) for m in TEST_MODELS]
    kurt_stds = [np.std(kurtosis_by_class[m]) for m in TEST_MODELS]
    axes[0, 2].bar(positions, kurt_means, yerr=kurt_stds, color=colors, alpha=0.7, capsize=5)
    axes[0, 2].axhline(y=TW_KURTOSIS, color='red', linestyle='--', label=f'Tracy-Widom ({TW_KURTOSIS})')
    axes[0, 2].axhline(y=GAUSSIAN_KURTOSIS, color='blue', linestyle='--', label='Gaussian (0)')
    axes[0, 2].set_xticks(positions)
    axes[0, 2].set_xticklabels([m[:6] for m in TEST_MODELS], rotation=45, ha='right')
    axes[0, 2].set_ylabel('Excess Kurtosis')
    axes[0, 2].set_title('Height Fluctuation Kurtosis')
    axes[0, 2].legend()
    
    # Plot 4: Skewness vs Kurtosis scatter
    for i, model in enumerate(TEST_MODELS):
        s_vals = skewness_by_class[model]
        k_vals = kurtosis_by_class[model]
        axes[1, 0].scatter(s_vals, k_vals, c=colors[i], label=model[:8], alpha=0.5, s=30)
    axes[1, 0].scatter([TW_SKEWNESS], [TW_KURTOSIS], c='red', marker='*', s=200, label='Tracy-Widom', zorder=10)
    axes[1, 0].scatter([GAUSSIAN_SKEWNESS], [GAUSSIAN_KURTOSIS], c='blue', marker='*', s=200, label='Gaussian', zorder=10)
    axes[1, 0].set_xlabel('Skewness')
    axes[1, 0].set_ylabel('Excess Kurtosis')
    axes[1, 0].set_title('Skewness-Kurtosis Space')
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: 2D latent space (PCA)
    from sklearn.decomposition import PCA
    all_latents = np.vstack([latents[m] for m in TEST_MODELS])
    all_labels_viz = np.concatenate([[i]*len(latents[m]) for i, m in enumerate(TEST_MODELS)])
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(all_latents)
    
    for i, model in enumerate(TEST_MODELS):
        mask = all_labels_viz == i
        axes[1, 1].scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                          c=colors[i], label=model[:8], alpha=0.6, s=30)
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].set_title('Latent Space (PCA)')
    axes[1, 1].legend(fontsize=7)
    
    # Plot 6: Key comparison bar chart
    metrics = ['EW↔KPZ', 'KPZ↔BD', 'BD↔EDEN', 'KPZ spread', 'EW sep', 'Ratio']
    values = [d_ew_kpz, d_kpz_bd, d_bd_eden, kpz_class_spread, ew_separation, ratio]
    bar_colors = ['blue', 'orange', 'green', 'red', 'purple', 'black']
    axes[1, 2].bar(range(len(metrics)), values, color=bar_colors, alpha=0.7)
    axes[1, 2].set_xticks(range(len(metrics)))
    axes[1, 2].set_xticklabels(metrics, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_title('Key Metrics')
    axes[1, 2].axhline(y=1.5, color='green', linestyle='--', label='Target ratio (1.5x)')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    fig_path = os.path.join(os.path.dirname(__file__), 'figures', 'exp10_skewness_kurtosis.png')
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
Experiment 10: Skewness/Kurtosis Features

Key Insight:
  - Theorem 5: Tracy-Widom (KPZ) has skewness ≈ 0.29, Gaussian (EW) has skewness = 0
  - Direct feature extraction of moments should capture this distinction

Results:
  - EW↔KPZ distance:      {d_ew_kpz:.3f}
  - KPZ-class spread:     {kpz_class_spread:.3f}  
  - EW separation:        {ew_separation:.3f}
  - UNIVERSALITY RATIO:   {ratio:.2f}x {'✅ > 1.5x target!' if ratio > 1.5 else '❌ < 1.5x target'}

Observed Skewness Values:
  - Edwards-Wilkinson:    {np.mean(skewness_by_class['edwards_wilkinson']):.3f}
  - KPZ equation:         {np.mean(skewness_by_class['kpz_equation']):.3f}
  - Ballistic deposition: {np.mean(skewness_by_class['ballistic_deposition']):.3f}
  - EDEN:                 {np.mean(skewness_by_class['eden']):.3f}
  - Tracy-Widom target:   {TW_SKEWNESS}

{'SUCCESS: Skewness/kurtosis features successfully separate EW from KPZ class!' if ratio > 1.5 else 'NOTE: May need more samples or longer time series for TW convergence'}
""")


if __name__ == "__main__":
    main()
