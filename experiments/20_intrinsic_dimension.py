"""
EXPERIMENT 20: Intrinsic Dimension of Solution Manifolds
=========================================================

CORE CLAIM TO TEST:
"Universality classes are characterized by membership in LOW-DIMENSIONAL 
attractors in feature space."

We claim manifolds are "low-dimensional" but haven't measured this.
This experiment directly tests that claim.

METHODS:
1. PCA - Principal Component Analysis (linear dimension)
2. MLE - Maximum Likelihood Estimator (Levina-Bickel 2004)
3. TwoNN - Two Nearest Neighbors estimator (Facco et al. 2017)

PREDICTIONS:
- d_int(EW manifold) ~ O(1-10), NOT O(system size)
- d_int(KPZ manifold) ~ O(1-10)
- d_int(combined EW+KPZ) may be higher (union of two manifolds)
- Dimension should be stable across system sizes L

IF CONFIRMED: Direct evidence for "low-dimensional attractor" claim
IF FALSIFIED: Manifolds are high-dimensional, claim needs revision
"""

import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_dir)

from simulation.physics_simulation import GrowthModelSimulator
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# INTRINSIC DIMENSION ESTIMATORS
# ============================================================================

def pca_dimension(X, variance_threshold=0.95):
    """
    Linear dimension via PCA.
    Returns number of components explaining `variance_threshold` of variance.
    
    This is a LOWER BOUND on intrinsic dimension (assumes linear manifold).
    """
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    d_pca = np.argmax(cumvar >= variance_threshold) + 1
    return d_pca, cumvar, pca.explained_variance_ratio_


def mle_dimension(X, k=5):
    """
    Maximum Likelihood Estimator for intrinsic dimension.
    Levina & Bickel (2004): "Maximum Likelihood Estimation of Intrinsic Dimension"
    
    Based on the observation that for points on a d-dimensional manifold,
    the ratio of distances to k-th and (k-1)-th neighbors follows a 
    specific distribution that depends on d.
    
    Returns: estimated dimension, standard error
    """
    n_samples = X.shape[0]
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Remove self-distance (first column)
    distances = distances[:, 1:]
    
    # MLE estimator
    # d_hat = 1 / (1/n * sum_i (1/(k-1) * sum_j log(r_k / r_j)))
    
    estimates = []
    for i in range(n_samples):
        r_k = distances[i, k-1]  # distance to k-th neighbor
        if r_k > 0:
            log_ratios = np.log(r_k / distances[i, :k-1])
            log_ratios = log_ratios[np.isfinite(log_ratios)]
            if len(log_ratios) > 0:
                d_local = (k - 1) / np.sum(log_ratios)
                if d_local > 0 and d_local < 100:  # sanity check
                    estimates.append(d_local)
    
    if len(estimates) == 0:
        return np.nan, np.nan
    
    d_mle = np.mean(estimates)
    d_std = np.std(estimates) / np.sqrt(len(estimates))
    
    return d_mle, d_std


def twonn_dimension(X):
    """
    Two Nearest Neighbors estimator for intrinsic dimension.
    Facco et al. (2017): "Estimating the intrinsic dimension of datasets"
    
    Uses ratio μ = r2/r1 where r1, r2 are distances to 1st and 2nd neighbors.
    For d-dimensional uniform data: P(μ) = d * μ^(d-1), giving E[log(μ)] = 1/d.
    
    Robust and requires no hyperparameter tuning.
    """
    n_samples = X.shape[0]
    
    # Find 2 nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # r1, r2 are distances to 1st and 2nd neighbors (skip self at index 0)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filter out zero distances
    valid = (r1 > 0) & (r2 > 0)
    r1, r2 = r1[valid], r2[valid]
    
    if len(r1) < 10:
        return np.nan, np.nan
    
    # mu = r2 / r1
    mu = r2 / r1
    
    # Empirical CDF of mu
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    
    # Linear fit: log(1 - F(μ)) vs log(μ) has slope -d
    # Use empirical CDF: F(μ_i) = i/n
    # So: log(1 - i/n) = -d * log(μ_i) + const
    
    # Exclude extremes for stability
    idx_start = int(0.1 * n)
    idx_end = int(0.9 * n)
    
    log_mu = np.log(mu_sorted[idx_start:idx_end])
    log_survival = np.log(1 - np.arange(idx_start, idx_end) / n)
    
    # Linear regression
    valid_fit = np.isfinite(log_mu) & np.isfinite(log_survival)
    if np.sum(valid_fit) < 10:
        return np.nan, np.nan
    
    slope, intercept = np.polyfit(log_mu[valid_fit], log_survival[valid_fit], 1)
    d_twonn = -slope
    
    # Bootstrap for error estimate
    n_boot = 100
    d_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(len(mu), len(mu), replace=True)
        mu_boot = np.sort(mu[idx])
        log_mu_b = np.log(mu_boot[idx_start:idx_end])
        log_surv_b = np.log(1 - np.arange(idx_start, idx_end) / n)
        valid_b = np.isfinite(log_mu_b) & np.isfinite(log_surv_b)
        if np.sum(valid_b) > 5:
            slope_b, _ = np.polyfit(log_mu_b[valid_b], log_surv_b[valid_b], 1)
            d_boots.append(-slope_b)
    
    d_std = np.std(d_boots) if len(d_boots) > 0 else np.nan
    
    return d_twonn, d_std


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_gradient_features(surfaces):
    """
    Extract gradient-based feature vector from each surface.
    
    Features per surface:
    - Gradient variance
    - Gradient skewness  
    - Gradient kurtosis
    - Laplacian variance
    - Gradient-Laplacian covariance
    - Height variance (interface width squared)
    
    Returns: (n_surfaces, n_features) array
    """
    n_surfaces = len(surfaces)
    features = []
    
    for h in surfaces:
        L = len(h)
        
        # Compute gradients (periodic BC)
        grad = np.roll(h, -1) - np.roll(h, 1)
        grad = grad / 2  # central difference
        
        # Compute Laplacian
        lap = np.roll(h, -1) + np.roll(h, 1) - 2*h
        
        # Gradient statistics
        grad_var = np.var(grad)
        grad_mean = np.mean(grad)
        grad_std = np.std(grad)
        
        if grad_std > 1e-10:
            grad_skew = np.mean((grad - grad_mean)**3) / grad_std**3
            grad_kurt = np.mean((grad - grad_mean)**4) / grad_std**4 - 3
        else:
            grad_skew = 0
            grad_kurt = 0
        
        # Laplacian statistics
        lap_var = np.var(lap)
        
        # Cross-statistics
        grad_lap_cov = np.mean((grad - np.mean(grad)) * (lap - np.mean(lap)))
        
        # Height statistics
        h_centered = h - np.mean(h)
        h_var = np.var(h_centered)
        
        features.append([
            grad_var,
            grad_skew,
            grad_kurt,
            lap_var,
            grad_lap_cov,
            h_var
        ])
    
    return np.array(features)


def extract_full_gradient_distribution(surfaces, n_bins=50):
    """
    Extract histogram of gradient distribution as feature vector.
    This captures the full shape, not just moments.
    """
    n_surfaces = len(surfaces)
    features = []
    
    # First pass: find global gradient range
    all_grads = []
    for h in surfaces[:min(100, n_surfaces)]:  # Sample for range
        grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
        all_grads.extend(grad)
    
    grad_min, grad_max = np.percentile(all_grads, [1, 99])
    bins = np.linspace(grad_min, grad_max, n_bins + 1)
    
    # Second pass: compute histograms
    for h in surfaces:
        grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
        hist, _ = np.histogram(grad, bins=bins, density=True)
        features.append(hist)
    
    return np.array(features)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_surfaces(model_type, n_surfaces=500, L=128, T=1000):
    """Generate surfaces from specified model."""
    
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    surfaces = []
    
    for i in range(n_surfaces):
        if i % 100 == 0:
            print(f"    Generating {model_type}: {i}/{n_surfaces}")
        
        # Reset random state for variety
        np.random.seed(42 + i)
        
        # Run simulation - get final surface (last time step)
        trajectory = simulator.generate_trajectory(model_type)
        surfaces.append(trajectory[-1].copy())  # Take final surface
    
    return surfaces


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run intrinsic dimension estimation experiment."""
    
    print("=" * 70)
    print("EXPERIMENT 20: Intrinsic Dimension of Solution Manifolds")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parameters
    n_surfaces = 500  # Number of surfaces per class
    L = 128           # System size
    T = 1000          # Time steps
    
    results = {}
    
    # ========================================================================
    # PART 1: Generate surfaces
    # ========================================================================
    print("PART 1: Generating surfaces")
    print("-" * 40)
    
    print("  Generating EW surfaces...")
    ew_surfaces = generate_surfaces('edwards_wilkinson', n_surfaces, L, T)
    
    print("  Generating KPZ surfaces...")
    kpz_surfaces = generate_surfaces('kpz_equation', n_surfaces, L, T)
    
    print("  Generating BD surfaces...")
    bd_surfaces = generate_surfaces('ballistic_deposition', n_surfaces, L, T)
    
    # ========================================================================
    # PART 2: Extract features
    # ========================================================================
    print("\nPART 2: Extracting features")
    print("-" * 40)
    
    # Method A: Moment-based features (6D)
    print("  Extracting moment features...")
    ew_moments = extract_gradient_features(ew_surfaces)
    kpz_moments = extract_gradient_features(kpz_surfaces)
    bd_moments = extract_gradient_features(bd_surfaces)
    
    # Standardize
    all_moments = np.vstack([ew_moments, kpz_moments, bd_moments])
    mean_m = np.mean(all_moments, axis=0)
    std_m = np.std(all_moments, axis=0)
    std_m[std_m < 1e-10] = 1
    
    ew_moments_std = (ew_moments - mean_m) / std_m
    kpz_moments_std = (kpz_moments - mean_m) / std_m
    bd_moments_std = (bd_moments - mean_m) / std_m
    combined_moments_std = np.vstack([ew_moments_std, kpz_moments_std])
    
    # Method B: Histogram features (50D)
    print("  Extracting histogram features...")
    all_surfaces = ew_surfaces + kpz_surfaces + bd_surfaces
    all_hist = extract_full_gradient_distribution(all_surfaces, n_bins=50)
    
    ew_hist = all_hist[:n_surfaces]
    kpz_hist = all_hist[n_surfaces:2*n_surfaces]
    bd_hist = all_hist[2*n_surfaces:]
    combined_hist = np.vstack([ew_hist, kpz_hist])
    
    print(f"\n  Moment feature dimension: {ew_moments.shape[1]}")
    print(f"  Histogram feature dimension: {ew_hist.shape[1]}")
    
    # ========================================================================
    # PART 3: Estimate intrinsic dimensions
    # ========================================================================
    print("\nPART 3: Intrinsic Dimension Estimation")
    print("-" * 40)
    
    feature_sets = {
        'EW (moments)': ew_moments_std,
        'KPZ (moments)': kpz_moments_std,
        'BD (moments)': bd_moments_std,
        'EW+KPZ (moments)': combined_moments_std,
        'EW (histogram)': ew_hist,
        'KPZ (histogram)': kpz_hist,
        'EW+KPZ (histogram)': combined_hist,
    }
    
    print("\n  {:25s} {:>8s} {:>12s} {:>12s}".format(
        "Dataset", "PCA(95%)", "MLE", "TwoNN"))
    print("  " + "-" * 60)
    
    for name, X in feature_sets.items():
        # PCA dimension
        d_pca, cumvar, _ = pca_dimension(X, variance_threshold=0.95)
        
        # MLE dimension (use k=10 for more stability)
        d_mle, d_mle_std = mle_dimension(X, k=10)
        
        # TwoNN dimension
        d_twonn, d_twonn_std = twonn_dimension(X)
        
        results[name] = {
            'pca': d_pca,
            'mle': d_mle,
            'mle_std': d_mle_std,
            'twonn': d_twonn,
            'twonn_std': d_twonn_std,
            'ambient_dim': X.shape[1]
        }
        
        print(f"  {name:25s} {d_pca:8d} {d_mle:8.2f}±{d_mle_std:.2f} {d_twonn:8.2f}±{d_twonn_std:.2f}")
    
    # ========================================================================
    # PART 4: Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check if dimensions are "low"
    print("\n1. Are manifolds low-dimensional?")
    print("-" * 40)
    
    for name in ['EW (moments)', 'KPZ (moments)', 'EW (histogram)', 'KPZ (histogram)']:
        r = results[name]
        ambient = r['ambient_dim']
        d_est = r['twonn']  # Use TwoNN as primary estimator
        ratio = d_est / ambient if ambient > 0 else np.nan
        
        status = "✓ LOW-DIM" if ratio < 0.3 else "? MODERATE" if ratio < 0.7 else "✗ HIGH-DIM"
        print(f"  {name:25s}: d={d_est:.1f}, ambient={ambient}, ratio={ratio:.2f} {status}")
    
    # Compare EW vs KPZ dimensions
    print("\n2. EW vs KPZ manifold dimensions")
    print("-" * 40)
    
    d_ew_twonn = results['EW (moments)']['twonn']
    d_kpz_twonn = results['KPZ (moments)']['twonn']
    d_combined = results['EW+KPZ (moments)']['twonn']
    
    print(f"  d(EW):     {d_ew_twonn:.2f}")
    print(f"  d(KPZ):    {d_kpz_twonn:.2f}")
    print(f"  d(EW+KPZ): {d_combined:.2f}")
    
    if d_combined > max(d_ew_twonn, d_kpz_twonn) + 0.5:
        print("  → Combined dimension HIGHER: suggests distinct manifolds ✓")
    else:
        print("  → Combined dimension similar: manifolds may overlap")
    
    # Stability check
    print("\n3. Estimator agreement")
    print("-" * 40)
    
    for name in ['EW (moments)', 'KPZ (moments)']:
        r = results[name]
        print(f"  {name}: PCA={r['pca']}, MLE={r['mle']:.1f}, TwoNN={r['twonn']:.1f}")
    
    # ========================================================================
    # PART 5: Conclusions
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    # Determine if claim is supported
    avg_dim = np.mean([results['EW (moments)']['twonn'], results['KPZ (moments)']['twonn']])
    
    print(f"\nAverage intrinsic dimension (TwoNN): {avg_dim:.2f}")
    print(f"Ambient dimension (moments): 6")
    print(f"Ambient dimension (histogram): 50")
    
    if avg_dim < 5:
        print("\n✓ CLAIM SUPPORTED: Manifolds are low-dimensional (d < 5)")
        print("  This validates 'low-dimensional attractor' in framework statement.")
    elif avg_dim < 10:
        print("\n? PARTIALLY SUPPORTED: Manifolds have moderate dimension (5 ≤ d < 10)")
        print("  'Low-dimensional' claim needs qualification.")
    else:
        print("\n✗ CLAIM NOT SUPPORTED: Manifolds are high-dimensional (d ≥ 10)")
        print("  Framework statement needs revision.")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
