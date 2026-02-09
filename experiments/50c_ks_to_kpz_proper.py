"""
Experiment 50c: Proper KS Generalization Test

CORRECT QUESTION: Does KS flow toward KPZ universality under coarse-graining?

Physical hypothesis:
- KS at small scales: deterministic chaos, different from KPZ
- KS at large scales: should show KPZ-like scaling (known from literature)

Framework test:
- Compute d(KS, KPZ) as function of coarse-graining scale b
- If KS ∈ KPZ universality at IR: distance should DECREASE with b
- If KS ≠ KPZ: distance flat or increases

This is the RG-flow-in-measure-space test mentioned in the roadmap.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator

sns.set_style("whitegrid")

# ============================================================================
# 1. GENERATE KS DATA
# ============================================================================

def generate_ks_data_fixed_params(n_trajectories=30, nu=1.0, kappa=2.0, lam=1.0, 
                                  L=256, T=2000):
    """
    Generate KS trajectories at FIXED parameters (chaotic regime).
    
    This matches the KPZ setup: fixed parameters, many realizations.
    """
    print("="*70)
    print("GENERATING KS DATA (FIXED PARAMETERS, MULTIPLE REALIZATIONS)")
    print("="*70)
    print(f"Parameters: ν={nu}, κ={kappa}, λ={lam}")
    print(f"Generating {n_trajectories} independent trajectories")
    
    simulator = KuramotoSivashinskySimulator(L=L, dt=0.01)
    
    all_features = []
    
    for traj_idx in range(n_trajectories):
        # Different random seed for each trajectory
        trajectory = simulator.simulate(
            T=T,
            nu=nu,
            kappa=kappa,
            lam=lam,
            noise_strength=0.1,
            record_interval=20
        )
        
        # Extract features from stationary part (skip transient)
        for t_idx in range(len(trajectory) // 2, len(trajectory)):
            h = trajectory[t_idx]
            
            # Gradient moments
            grad = np.gradient(h)
            features = np.array([
                np.mean(grad**2),
                np.mean(grad**3),
                np.mean(grad**4),
                np.mean(grad**5),
                np.mean(grad**6),
                np.mean(np.abs(grad)**7)
            ])
            
            all_features.append(features)
        
        if (traj_idx + 1) % 5 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories} trajectories")
    
    all_features = np.array(all_features)
    
    print(f"\nGenerated {len(all_features)} samples")
    print(f"  Feature range: {all_features.min():.2e} to {all_features.max():.2e}")
    print(f"  Mean amplitude: {np.mean(np.abs(all_features)):.2e}")
    
    return all_features

# ============================================================================
# 2. LOAD KPZ DATA
# ============================================================================

def load_kpz_data():
    """
    Load KPZ features from Exp 46b.
    """
    print("\n" + "="*70)
    print("LOADING KPZ DATA")
    print("="*70)
    
    kpz_path = Path('results/exp46_coupling_coordinate/coupling_coordinate_results.pkl')
    
    if not kpz_path.exists():
        print("❌ No KPZ data found. Run Exp 46 first.")
        return None
    
    with open(kpz_path, 'rb') as f:
        kpz_data = pickle.load(f)
    
    # Extract features (check structure)
    if 'features' in kpz_data:
        features = kpz_data['features']
    elif 'X_train' in kpz_data:
        features = kpz_data['X_train']
    else:
        print(f"Available keys: {kpz_data.keys()}")
        print("❌ Cannot find features in KPZ data")
        return None
    
    print(f"Loaded {len(features)} KPZ samples")
    print(f"  Feature range: {features.min():.2e} to {features.max():.2e}")
    
    return features

# ============================================================================
# 3. COARSE-GRAINING FUNCTION
# ============================================================================

def coarse_grain_features(features, block_size):
    """
    Coarse-grain by averaging over blocks.
    
    Parameters:
    -----------
    features : array (n_samples, n_features)
    block_size : int
        Number of samples to average
    
    Returns:
    --------
    features_coarse : array (n_blocks, n_features)
    """
    n_samples = len(features)
    n_blocks = n_samples // block_size
    
    if n_blocks == 0:
        return features
    
    # Truncate to multiple of block_size
    features_trunc = features[:n_blocks * block_size]
    
    # Reshape and average
    features_reshaped = features_trunc.reshape(n_blocks, block_size, -1)
    features_coarse = features_reshaped.mean(axis=1)
    
    return features_coarse

# ============================================================================
# 4. DISTANCE COMPUTATION (MMD-BASED)
# ============================================================================

def compute_mmd_distance(X, Y):
    """
    Compute Maximum Mean Discrepancy (MMD) between two datasets.
    
    MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    
    Uses RBF kernel with median heuristic for bandwidth.
    """
    # Combine for bandwidth estimation
    Z = np.vstack([X, Y])
    pairwise = pairwise_distances(Z, metric='euclidean')
    bandwidth = np.median(pairwise[pairwise > 0])
    
    if bandwidth == 0:
        bandwidth = 1.0
    
    # RBF kernel
    def rbf_kernel(A, B):
        pairwise_sq = pairwise_distances(A, B, metric='sqeuclidean')
        return np.exp(-pairwise_sq / (2 * bandwidth**2))
    
    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)
    
    # MMD²
    mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    
    return max(0, mmd_sq)**0.5  # Take sqrt, avoid negative due to numerical error

# ============================================================================
# 5. MAIN TEST: d(KS, KPZ) vs SCALE
# ============================================================================

def test_ks_kpz_convergence(ks_features, kpz_features):
    """
    Test if KS → KPZ under coarse-graining.
    
    Compute d(KS, KPZ) at scales b = 1, 2, 4, 8, 16, 32
    
    Expected:
    - If KS ∈ KPZ universality: distance decreases with b
    - If KS ≠ KPZ: distance flat or increases
    """
    print("\n" + "="*70)
    print("TEST: DOES KS → KPZ UNDER COARSE-GRAINING?")
    print("="*70)
    
    # Ensure equal sample sizes for fair comparison
    n_min = min(len(ks_features), len(kpz_features))
    ks_features = ks_features[:n_min]
    kpz_features = kpz_features[:n_min]
    
    print(f"\nUsing {n_min} samples from each dataset")
    
    # Standardize (important for distance comparison)
    scaler = StandardScaler()
    combined = np.vstack([ks_features, kpz_features])
    scaler.fit(combined)
    
    ks_scaled = scaler.transform(ks_features)
    kpz_scaled = scaler.transform(kpz_features)
    
    # Test at multiple scales
    scales = [1, 2, 4, 8, 16, 32]
    distances = []
    sample_sizes = []
    
    print("\nComputing distances at multiple scales:")
    
    for b in scales:
        ks_coarse = coarse_grain_features(ks_scaled, b)
        kpz_coarse = coarse_grain_features(kpz_scaled, b)
        
        if len(ks_coarse) < 10 or len(kpz_coarse) < 10:
            print(f"  b={b:2d}: Insufficient samples")
            break
        
        # Compute MMD distance
        dist = compute_mmd_distance(ks_coarse, kpz_coarse)
        
        distances.append(dist)
        sample_sizes.append(len(ks_coarse))
        
        print(f"  b={b:2d}: d(KS,KPZ) = {dist:.4f} ({len(ks_coarse)} samples)")
    
    # Analyze trend
    if len(distances) >= 3:
        from scipy.stats import linregress
        log_scales = np.log(scales[:len(distances)])
        slope, intercept, r, p, _ = linregress(log_scales, distances)
        
        print(f"\nLinear fit: d ~ {slope:.4f} * log(b) + {intercept:.4f}")
        print(f"  R² = {r**2:.3f}, p = {p:.3e}")
        
        print("\n" + "="*70)
        print("VERDICT:")
        
        if slope < -0.01 and p < 0.05:
            print("✅✅ DISTANCE DECREASES WITH SCALE")
            print("    → KS flows toward KPZ universality")
            print("    → Framework GENERALIZES to KS")
            verdict = "converges"
        elif slope < 0.01 and p < 0.05:
            print("➡️  DISTANCE ROUGHLY CONSTANT")
            print("    → KS and KPZ maintain separation")
            print("    → Different universality classes")
            verdict = "parallel"
        else:
            print("❌ DISTANCE INCREASES OR UNCLEAR")
            print("    → KS diverges from KPZ")
            print("    → Need different test or observables")
            verdict = "diverges"
    else:
        slope = np.nan
        verdict = "insufficient_data"
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Distance vs scale
    ax = axes[0]
    ax.plot(scales[:len(distances)], distances, 'o-', markersize=8, linewidth=2)
    if len(distances) >= 3:
        fitted = intercept + slope * np.log(scales[:len(distances)])
        ax.plot(scales[:len(distances)], fitted, '--', alpha=0.5, 
               label=f'slope={slope:.3f}')
    ax.set_xlabel('Coarse-graining scale b')
    ax.set_ylabel('d(KS, KPZ)')
    ax.set_title('Distance vs Scale')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    # Sample size
    ax = axes[1]
    ax.plot(scales[:len(sample_sizes)], sample_sizes, 's-', markersize=8)
    ax.set_xlabel('Scale b')
    ax.set_ylabel('Number of samples')
    ax.set_title('Sample Size After Coarse-Graining')
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    results = {
        'scales': scales[:len(distances)],
        'distances': distances,
        'sample_sizes': sample_sizes,
        'slope': slope,
        'verdict': verdict
    }
    
    return results, fig

# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50c: PROPER KS GENERALIZATION TEST")
    print("="*70)
    print("\nQuestion: Does KS → KPZ under coarse-graining?")
    print("="*70)
    
    output_dir = Path('results/exp50c_ks_to_kpz')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate KS data
    ks_features = generate_ks_data_fixed_params(
        n_trajectories=30,
        nu=1.0,
        kappa=2.0,
        lam=1.0,
        L=256,
        T=2000
    )
    
    # Load KPZ data
    kpz_features = load_kpz_data()
    
    if kpz_features is None:
        print("\n❌ Cannot proceed without KPZ data")
        return
    
    # Test KS → KPZ convergence
    results, fig = test_ks_kpz_convergence(ks_features, kpz_features)
    fig.savefig(output_dir / 'ks_to_kpz_distance.png', dpi=150, bbox_inches='tight')
    
    # Save results
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'ks_features': ks_features,
            'results': results
        }, f)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)
    
    plt.show()

if __name__ == '__main__':
    main()
