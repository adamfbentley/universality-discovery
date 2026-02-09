"""
EXPERIMENT 47: Information-Geometric Distances Across Scales
==============================================================

MOTIVATION (Assessment 2, Section III.D)
-----------------------------------------
PCA geometry (Euclidean distances in feature space) is not the natural geometry
for distributions. Information geometry provides theoretically grounded metrics:

- **Fisher-Rao metric**: Riemannian metric on probability distributions
- **KL divergence**: Distinguishability between distributions
- **Bhattacharyya distance**: Overlapping coefficient-based metric

**KEY HYPOTHESIS**: RG-relevant directions maintain or increase distinguishability
across scales; irrelevant directions become less distinguishable.

**TEST**: Compute information-geometric distances at multiple coarse-graining scales.
If features encode RG structure, distances should reveal fixed-point structure.

EXPERIMENTAL DESIGN
-------------------
1. Generate surface snapshots: EW vs KPZ, droplet IC (best separation)
2. Compute gradient moments Φ(h) at each scale b = 1, 2, 4, 8
3. For each scale, estimate p(Φ | class) via Gaussian approximation
4. Compute information-geometric distances:
   - KL divergence: D_KL(p_EW || p_KPZ)
   - Symmetrized KL: (D_KL(p||q) + D_KL(q||p))/2
   - Bhattacharyya distance: -log(BC) where BC = ∫√(p·q)
5. Track distance evolution across scales

PREDICTIONS
-----------
If gradient moments encode RG-relevant structure:
- Distances should increase or plateau at large scales (classes diverge)
- Consistent with universality: EW ≠ KPZ preserved under coarse-graining

If features are RG-irrelevant:
- Distances should decrease (become indistinguishable)

COMPARISON TO EXP 45b
---------------------
Exp 45b: Learn features to be RG-covariant (neural network, supervised)
Exp 47: Test if hand-crafted features reveal RG structure via information geometry

Both probe "does feature space respect RG symmetries?"

IMPLEMENTATION NOTES
--------------------
- Use multivariate Gaussian approximation: p(Φ|class) ~ N(μ, Σ)
- KL divergence for Gaussians: analytic formula
- Bhattacharyya distance for Gaussians: analytic formula
- Generate sufficient samples for stable covariance estimation (N~1000)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import multivariate_normal
from scipy.spatial.distance import jensenshannon
from numba import jit
import pickle
from datetime import datetime

# ============================================================================
# SIMULATION FUNCTIONS (Copied from Exp 27)
# ============================================================================

def simulate_ew_droplet(L=256, T=500, nu=1.0, noise_strength=1.0, dt=0.1):
    """Edwards-Wilkinson with droplet IC."""
    h = np.zeros(L)
    h[L//2] = 10.0  # Droplet IC
    
    trajectory = []
    for t in range(T):
        if t % 10 == 0:
            trajectory.append(h.copy())
        
        laplacian = np.roll(h, -1) + np.roll(h, 1) - 2*h
        noise = np.random.randn(L) * noise_strength * np.sqrt(dt)
        h += nu * laplacian * dt + noise
    
    return np.array(trajectory)

def simulate_kpz_droplet(L=256, T=500, nu=1.0, lam=1.0, noise_strength=1.0, dt=0.1):
    """KPZ with droplet IC."""
    h = np.zeros(L)
    h[L//2] = 10.0  # Droplet IC
    
    trajectory = []
    for t in range(T):
        if t % 10 == 0:
            trajectory.append(h.copy())
        
        laplacian = np.roll(h, -1) + np.roll(h, 1) - 2*h
        grad = (np.roll(h, -1) - np.roll(h, 1)) / 2.0
        nonlinear = lam * grad**2 / 2.0
        noise = np.random.randn(L) * noise_strength * np.sqrt(dt)
        h += nu * laplacian * dt + nonlinear * dt + noise
    
    return np.array(trajectory)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_gradient_moments(trajectory, order=6):
    """
    Extract gradient moments from trajectory.
    Returns: (N_snapshots, order) array
    """
    features = []
    for h in trajectory:
        grad = np.gradient(h)
        moments = np.array([np.mean(np.abs(grad)**k) for k in range(1, order+1)])
        features.append(moments)
    return np.array(features)

def coarse_grain(trajectory, factor=2):
    """
    Coarse-grain trajectory by averaging blocks.
    factor: Coarse-graining scale (2, 4, 8, etc.)
    """
    coarsened = []
    for h in trajectory:
        L = len(h)
        new_L = L // factor
        h_coarse = np.array([h[i*factor:(i+1)*factor].mean() for i in range(new_L)])
        coarsened.append(h_coarse)
    return np.array(coarsened)

# ============================================================================
# INFORMATION-GEOMETRIC DISTANCES
# ============================================================================

def kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2):
    """
    KL divergence D_KL(N1 || N2) for multivariate Gaussians.
    D_KL = 0.5 * [log(|Σ2|/|Σ1|) + tr(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - d]
    """
    d = len(mu1)
    
    # Add small regularization for numerical stability
    Sigma1_reg = Sigma1 + 1e-6 * np.eye(d)
    Sigma2_reg = Sigma2 + 1e-6 * np.eye(d)
    
    try:
        Sigma2_inv = np.linalg.inv(Sigma2_reg)
        
        log_det_term = np.log(np.linalg.det(Sigma2_reg) / np.linalg.det(Sigma1_reg))
        trace_term = np.trace(Sigma2_inv @ Sigma1_reg)
        
        diff = mu2 - mu1
        mahalanobis_term = diff @ Sigma2_inv @ diff
        
        kl = 0.5 * (log_det_term + trace_term + mahalanobis_term - d)
        return max(0, kl)  # Ensure non-negative (numerical errors can give small negatives)
    except np.linalg.LinAlgError:
        return np.nan

def symmetrized_kl(mu1, Sigma1, mu2, Sigma2):
    """Symmetrized KL divergence: (D_KL(p||q) + D_KL(q||p))/2"""
    kl1 = kl_divergence_gaussian(mu1, Sigma1, mu2, Sigma2)
    kl2 = kl_divergence_gaussian(mu2, Sigma2, mu1, Sigma1)
    return (kl1 + kl2) / 2.0

def bhattacharyya_distance_gaussian(mu1, Sigma1, mu2, Sigma2):
    """
    Bhattacharyya distance for multivariate Gaussians.
    D_B = (1/8)(μ1-μ2)^T Σ^{-1} (μ1-μ2) + (1/2)log(|Σ|/√(|Σ1||Σ2|))
    where Σ = (Σ1+Σ2)/2
    """
    d = len(mu1)
    
    # Add regularization
    Sigma1_reg = Sigma1 + 1e-6 * np.eye(d)
    Sigma2_reg = Sigma2 + 1e-6 * np.eye(d)
    Sigma_avg = (Sigma1_reg + Sigma2_reg) / 2.0
    
    try:
        Sigma_inv = np.linalg.inv(Sigma_avg)
        
        diff = mu1 - mu2
        mahalanobis = diff @ Sigma_inv @ diff
        
        det_term = np.log(
            np.linalg.det(Sigma_avg) / np.sqrt(np.linalg.det(Sigma1_reg) * np.linalg.det(Sigma2_reg))
        )
        
        db = (1/8) * mahalanobis + (1/2) * det_term
        return max(0, db)
    except np.linalg.LinAlgError:
        return np.nan

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 47: Information-Geometric Distances Across Scales")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Parameters
    L = 256
    T = 500
    n_trajectories = 30  # Generate 30 trajectories per class
    scales = [1, 2, 4, 8]  # Coarse-graining scales
    feature_dim = 6  # Gradient moments up to 6th order
    
    print("PARAMETERS")
    print("=" * 70)
    print(f"System size: L = {L}")
    print(f"Simulation time: T = {T}")
    print(f"Trajectories per class: {n_trajectories}")
    print(f"Coarse-graining scales: {scales}")
    print(f"Feature dimension: {feature_dim}")
    print()
    
    # ========================================================================
    # GENERATE DATA
    # ========================================================================
    
    print("=" * 70)
    print("GENERATING DATA (DROPLET IC)")
    print("=" * 70)
    print()
    
    print("Generating EW trajectories...")
    ew_trajectories = []
    for i in range(n_trajectories):
        traj = simulate_ew_droplet(L=L, T=T)
        ew_trajectories.append(traj)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n_trajectories} complete")
    
    print("\nGenerating KPZ trajectories...")
    kpz_trajectories = []
    for i in range(n_trajectories):
        traj = simulate_kpz_droplet(L=L, T=T)
        kpz_trajectories.append(traj)
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{n_trajectories} complete")
    
    print()
    
    # ========================================================================
    # COMPUTE DISTANCES AT EACH SCALE
    # ========================================================================
    
    results = {
        'scales': scales,
        'kl_divergence': [],
        'symmetrized_kl': [],
        'bhattacharyya': [],
        'ew_means': [],
        'ew_covs': [],
        'kpz_means': [],
        'kpz_covs': [],
    }
    
    print("=" * 70)
    print("COMPUTING INFORMATION-GEOMETRIC DISTANCES")
    print("=" * 70)
    print()
    
    for scale in scales:
        print(f"Scale b = {scale}:")
        
        # Coarse-grain trajectories
        if scale == 1:
            ew_coarse = ew_trajectories
            kpz_coarse = kpz_trajectories
        else:
            ew_coarse = [coarse_grain(traj, scale) for traj in ew_trajectories]
            kpz_coarse = [coarse_grain(traj, scale) for traj in kpz_trajectories]
        
        # Extract features
        ew_features = []
        for traj in ew_coarse:
            feats = extract_gradient_moments(traj, order=feature_dim)
            ew_features.append(feats)
        ew_features = np.vstack(ew_features)  # (N_total, feature_dim)
        
        kpz_features = []
        for traj in kpz_coarse:
            feats = extract_gradient_moments(traj, order=feature_dim)
            kpz_features.append(feats)
        kpz_features = np.vstack(kpz_features)
        
        print(f"  EW features: {ew_features.shape}")
        print(f"  KPZ features: {kpz_features.shape}")
        
        # Estimate Gaussian parameters
        mu_ew = np.mean(ew_features, axis=0)
        Sigma_ew = np.cov(ew_features.T)
        
        mu_kpz = np.mean(kpz_features, axis=0)
        Sigma_kpz = np.cov(kpz_features.T)
        
        # Compute distances
        kl = kl_divergence_gaussian(mu_ew, Sigma_ew, mu_kpz, Sigma_kpz)
        sym_kl = symmetrized_kl(mu_ew, Sigma_ew, mu_kpz, Sigma_kpz)
        bhatt = bhattacharyya_distance_gaussian(mu_ew, Sigma_ew, mu_kpz, Sigma_kpz)
        
        print(f"  D_KL(EW || KPZ) = {kl:.4f}")
        print(f"  Symmetrized KL = {sym_kl:.4f}")
        print(f"  Bhattacharyya distance = {bhatt:.4f}")
        print()
        
        # Store results
        results['kl_divergence'].append(kl)
        results['symmetrized_kl'].append(sym_kl)
        results['bhattacharyya'].append(bhatt)
        results['ew_means'].append(mu_ew)
        results['ew_covs'].append(Sigma_ew)
        results['kpz_means'].append(mu_kpz)
        results['kpz_covs'].append(Sigma_kpz)
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    print("=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: KL Divergence
    axes[0].plot(scales, results['kl_divergence'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Coarse-graining scale b', fontsize=12)
    axes[0].set_ylabel('D_KL(EW || KPZ)', fontsize=12)
    axes[0].set_title('KL Divergence vs Scale', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Plot 2: Symmetrized KL
    axes[1].plot(scales, results['symmetrized_kl'], 'o-', linewidth=2, markersize=8, color='C1')
    axes[1].set_xlabel('Coarse-graining scale b', fontsize=12)
    axes[1].set_ylabel('Symmetrized KL', fontsize=12)
    axes[1].set_title('Symmetrized KL Divergence vs Scale', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale('log', base=2)
    
    # Plot 3: Bhattacharyya Distance
    axes[2].plot(scales, results['bhattacharyya'], 'o-', linewidth=2, markersize=8, color='C2')
    axes[2].set_xlabel('Coarse-graining scale b', fontsize=12)
    axes[2].set_ylabel('Bhattacharyya Distance', fontsize=12)
    axes[2].set_title('Bhattacharyya Distance vs Scale', fontsize=14)
    axes[2].grid(alpha=0.3)
    axes[2].set_xscale('log', base=2)
    
    plt.tight_layout()
    
    output_dir = Path('results/exp47_information_geometry')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_dir / 'distances_vs_scale.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    
    # Check if distances increase or plateau
    kl_trend = np.polyfit(np.log2(scales), results['kl_divergence'], 1)[0]
    sym_kl_trend = np.polyfit(np.log2(scales), results['symmetrized_kl'], 1)[0]
    bhatt_trend = np.polyfit(np.log2(scales), results['bhattacharyya'], 1)[0]
    
    print("Distance trends (positive = increasing with scale):")
    print(f"  KL divergence slope: {kl_trend:.4f}")
    print(f"  Symmetrized KL slope: {sym_kl_trend:.4f}")
    print(f"  Bhattacharyya slope: {bhatt_trend:.4f}")
    print()
    
    if sym_kl_trend > 0:
        print("✅ DISTANCES INCREASE WITH SCALE")
        print("   → EW and KPZ become MORE distinguishable at larger scales")
        print("   → Gradient moments encode RG-RELEVANT structure")
        print("   → Consistent with universality: classes diverge under coarse-graining")
    else:
        print("❌ DISTANCES DECREASE WITH SCALE")
        print("   → EW and KPZ become LESS distinguishable at larger scales")
        print("   → Gradient moments are RG-IRRELEVANT")
        print("   → Features wash out under coarse-graining")
    
    print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results_path = output_dir / 'results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Saved results: {results_path}")
    
    print()
    print("=" * 70)
    print("✅ EXPERIMENT 47 COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
