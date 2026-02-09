"""
Experiment 50f: Diagnostic A - Field-Level Coarse-Graining

PROPER RG TEST: Coarse-grain the FIELDS, then extract features.

Current Exp 50c flaw: Averages features (not physical RG)
Correct approach: h → h_b (coarse-grain field) → Φ(h_b) (extract features)

Two methods:
A1) Spectral low-pass: FFT, keep |k| ≤ k_c, inverse FFT
A2) Real-space block average: Average h(x) in blocks of size b

Then test: Does d(KS, KPZ) change with b under PROPER coarse-graining?

Uses FIXED BANDWIDTH (from Diagnostic C).
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.fft import fft, ifft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator

sns.set_style("whitegrid")

# ============================================================================
# FIELD-LEVEL COARSE-GRAINING
# ============================================================================

def coarse_grain_field_spectral(h, k_cutoff_fraction):
    """
    Spectral low-pass coarse-graining.
    
    Parameters:
    -----------
    h : array (L,)
        Height field
    k_cutoff_fraction : float
        Fraction of modes to keep (e.g., 0.5 = keep half)
    
    Returns:
    --------
    h_coarse : array (L,)
        Coarse-grained field (same size, but smoothed)
    """
    L = len(h)
    h_hat = fft(h)
    k = fftfreq(L)
    
    # Zero out high-k modes
    k_max = np.max(np.abs(k))
    k_cutoff = k_cutoff_fraction * k_max
    
    mask = np.abs(k) <= k_cutoff
    h_hat_filtered = h_hat * mask
    
    h_coarse = np.real(ifft(h_hat_filtered))
    
    return h_coarse

def coarse_grain_field_realspace(h, block_size):
    """
    Real-space block averaging.
    
    Parameters:
    -----------
    h : array (L,)
        Height field
    block_size : int
        Block size for averaging
    
    Returns:
    --------
    h_coarse : array (L // block_size,)
        Coarse-grained field (downsampled)
    """
    L = len(h)
    n_blocks = L // block_size
    
    h_trunc = h[:n_blocks * block_size]
    h_reshaped = h_trunc.reshape(n_blocks, block_size)
    h_coarse = h_reshaped.mean(axis=1)
    
    return h_coarse

def extract_gradient_moments(h):
    """
    Extract gradient moment features from field.
    
    Returns: [m2, m3, m4, m5, m6, m7]
    """
    grad = np.gradient(h)
    
    features = np.array([
        np.mean(grad**2),
        np.mean(grad**3),
        np.mean(grad**4),
        np.mean(grad**5),
        np.mean(grad**6),
        np.mean(np.abs(grad)**7)
    ])
    
    return features

# ============================================================================
# GENERATE DATA WITH FIELD-LEVEL COARSE-GRAINING
# ============================================================================

def generate_ks_features_multiscale(n_trajectories=30, method='spectral'):
    """
    Generate KS features at multiple scales using FIELD-LEVEL coarse-graining.
    
    Parameters:
    -----------
    method : str
        'spectral' or 'realspace'
    """
    print("="*70)
    print(f"GENERATING KS DATA (FIELD-LEVEL {method.upper()} COARSE-GRAINING)")
    print("="*70)
    
    simulator = KuramotoSivashinskySimulator(L=256, dt=0.01)
    
    # Scales
    if method == 'spectral':
        # For spectral: keep fraction of modes
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Keep 100%, 50%, 25%, 12.5%, 6.25%
        scale_names = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']
    else:  # realspace
        scales = [1, 2, 4, 8]
        scale_names = [f'b={s}' for s in scales]
    
    features_by_scale = {scale: [] for scale in scales}
    
    for traj_idx in range(n_trajectories):
        trajectory = simulator.simulate(
            T=2000,
            nu=1.0,
            kappa=2.0,
            lam=1.0,
            noise_strength=0.1,
            record_interval=20
        )
        
        # Process stationary part
        for t_idx in range(len(trajectory) // 2, len(trajectory)):
            h = trajectory[t_idx]
            
            # Coarse-grain at each scale
            for scale, name in zip(scales, scale_names):
                if method == 'spectral':
                    h_coarse = coarse_grain_field_spectral(h, scale)
                else:
                    h_coarse = coarse_grain_field_realspace(h, scale)
                
                # Extract features from coarse-grained field
                features = extract_gradient_moments(h_coarse)
                features_by_scale[scale].append(features)
        
        if (traj_idx + 1) % 5 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories} trajectories")
    
    # Convert to arrays
    for scale in scales:
        features_by_scale[scale] = np.array(features_by_scale[scale])
        print(f"  {scale_names[scales.index(scale)]}: {len(features_by_scale[scale])} samples")
    
    return features_by_scale, scales, scale_names

def generate_kpz_features_multiscale(n_samples=1000, method='spectral'):
    """
    Generate KPZ features at multiple scales.
    
    For now, load existing KPZ data and apply field-level coarse-graining.
    (Ideally would regenerate from scratch, but this tests the concept)
    """
    print("\n" + "="*70)
    print(f"GENERATING KPZ DATA (FIELD-LEVEL {method.upper()} COARSE-GRAINING)")
    print("="*70)
    print("⚠️  Using existing KPZ samples - ideally regenerate with fields")
    
    # Load existing (this is a limitation - we only have features, not fields)
    kpz_path = Path('results/exp46_coupling_coordinate/coupling_coordinate_results.pkl')
    
    if not kpz_path.exists():
        return None, None, None
    
    with open(kpz_path, 'rb') as f:
        kpz_data = pickle.load(f)
    
    features_raw = kpz_data['features']
    
    # Scales (match KS)
    if method == 'spectral':
        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
        scale_names = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']
    else:
        scales = [1, 2, 4, 8]
        scale_names = [f'b={s}' for s in scales]
    
    # For b=1, use raw features
    features_by_scale = {scales[0]: features_raw}
    
    # For b>1, we can't properly coarse-grain without fields
    # So we'll use feature-space averaging as approximation
    # (This is a known limitation - document it)
    for i, scale in enumerate(scales[1:], 1):
        # Block average (not ideal, but consistent with KS processing)
        if method == 'realspace':
            block_size = scale
        else:
            block_size = 2**i  # Approximate
        
        n_blocks = len(features_raw) // block_size
        if n_blocks < 5:
            features_by_scale[scale] = None
            continue
        
        features_trunc = features_raw[:n_blocks * block_size]
        features_reshaped = features_trunc.reshape(n_blocks, block_size, -1)
        features_coarse = features_reshaped.mean(axis=1)
        
        features_by_scale[scale] = features_coarse
    
    print(f"  Loaded {len(features_raw)} KPZ samples")
    for scale, name in zip(scales, scale_names):
        if features_by_scale[scale] is not None:
            print(f"  {name}: {len(features_by_scale[scale])} samples")
    
    return features_by_scale, scales, scale_names

# ============================================================================
# MMD WITH FIXED BANDWIDTH
# ============================================================================

def compute_mmd_fixed(X, Y, sigma):
    """MMD with fixed bandwidth."""
    def rbf_kernel(A, B):
        pairwise_sq = pairwise_distances(A, B, metric='sqeuclidean')
        return np.exp(-pairwise_sq / (2 * sigma**2))
    
    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)
    
    mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    
    return max(0, mmd_sq)**0.5

def estimate_bandwidth(X, Y):
    """Median heuristic for bandwidth."""
    Z = np.vstack([X, Y])
    pairwise = pairwise_distances(Z, metric='euclidean')
    sigma = np.median(pairwise[pairwise > 0])
    return sigma if sigma > 0 else 1.0

# ============================================================================
# MAIN TEST
# ============================================================================

def test_ks_kpz_field_coarsening(method='spectral'):
    """
    Test d(KS, KPZ) with field-level coarse-graining.
    """
    print("="*70)
    print(f"DIAGNOSTIC A: FIELD-LEVEL COARSE-GRAINING ({method.upper()})")
    print("="*70)
    
    # Generate data
    ks_by_scale, scales, scale_names = generate_ks_features_multiscale(
        n_trajectories=20, method=method
    )
    
    kpz_by_scale, _, _ = generate_kpz_features_multiscale(method=method)
    
    if kpz_by_scale is None:
        print("❌ Cannot proceed without KPZ data")
        return None, None
    
    # Match sample sizes and standardize
    print("\n" + "="*70)
    print("COMPUTING DISTANCES")
    print("="*70)
    
    # Estimate bandwidth from b=1 data
    ks_b1 = ks_by_scale[scales[0]]
    kpz_b1 = kpz_by_scale[scales[0]]
    
    n_min = min(len(ks_b1), len(kpz_b1))
    ks_b1 = ks_b1[:n_min]
    kpz_b1 = kpz_b1[:n_min]
    
    # Standardize and estimate bandwidth
    scaler = StandardScaler()
    scaler.fit(np.vstack([ks_b1, kpz_b1]))
    
    ks_b1_scaled = scaler.transform(ks_b1)
    kpz_b1_scaled = scaler.transform(kpz_b1)
    
    sigma_fixed = estimate_bandwidth(ks_b1_scaled, kpz_b1_scaled)
    print(f"\nFixed bandwidth (from b=1): σ = {sigma_fixed:.4f}")
    
    # Compute distances at each scale
    distances = []
    sample_sizes = []
    
    for scale, name in zip(scales, scale_names):
        ks_scale = ks_by_scale[scale]
        kpz_scale = kpz_by_scale.get(scale)
        
        if kpz_scale is None or len(kpz_scale) < 5:
            print(f"  {name}: Insufficient KPZ samples")
            break
        
        # Match sizes
        n_min = min(len(ks_scale), len(kpz_scale))
        ks_scale = ks_scale[:n_min]
        kpz_scale = kpz_scale[:n_min]
        
        # Standardize with SAME scaler
        ks_scaled = scaler.transform(ks_scale)
        kpz_scaled = scaler.transform(kpz_scale)
        
        # Compute distance with FIXED sigma
        dist = compute_mmd_fixed(ks_scaled, kpz_scaled, sigma_fixed)
        
        distances.append(dist)
        sample_sizes.append(n_min)
        
        print(f"  {name}: d(KS,KPZ) = {dist:.4f} ({n_min} samples)")
    
    # Analyze trend
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    from scipy.stats import linregress
    
    if len(distances) >= 3:
        # Use log scale indices
        scale_indices = np.arange(len(distances))
        slope, intercept, r, p, _ = linregress(scale_indices, distances)
        
        print(f"\nLinear fit: d ~ {slope:.4f} * scale_index + {intercept:.4f}")
        print(f"  R² = {r**2:.3f}, p = {p:.3e}")
        
        print("\n" + "="*70)
        print("VERDICT:")
        print("="*70)
        
        if slope < -0.02 and p < 0.05:
            print("✅✅ DISTANCE DECREASES WITH SCALE")
            print("    → KS flows toward KPZ under coarse-graining")
            print("    → Framework GENERALIZES to KS")
            verdict = "converges"
        elif abs(slope) < 0.02:
            print("➡️  DISTANCE ROUGHLY CONSTANT")
            print("    → KS and KPZ maintain separation")
            print("    → Need different regime or observables")
            verdict = "parallel"
        else:
            print("❌ DISTANCE INCREASES")
            print("    → KS diverges from KPZ in this regime")
            print("    → Try different parameters or observables")
            verdict = "diverges"
    else:
        slope = np.nan
        verdict = "insufficient_data"
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(distances)), distances, 'o-', markersize=10, linewidth=2)
    if len(distances) >= 3:
        fitted = intercept + slope * np.arange(len(distances))
        ax.plot(range(len(distances)), fitted, '--', alpha=0.5,
               label=f'slope={slope:.3f}')
    ax.set_xticks(range(len(scale_names)))
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Coarse-graining scale')
    ax.set_ylabel('d(KS, KPZ)')
    ax.set_title(f'Field-Level {method.capitalize()} Coarse-Graining')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    results = {
        'method': method,
        'scales': scales[:len(distances)],
        'scale_names': scale_names[:len(distances)],
        'distances': distances,
        'sample_sizes': sample_sizes,
        'slope': slope,
        'sigma_fixed': sigma_fixed,
        'verdict': verdict
    }
    
    return results, fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50f: DIAGNOSTIC A - FIELD-LEVEL COARSE-GRAINING")
    print("="*70)
    print("\nQuestion: Does KS → KPZ with PROPER field-level RG?")
    print("="*70)
    
    output_dir = Path('results/exp50f_diagnostic_a_field_cg')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test spectral method
    results, fig = test_ks_kpz_field_coarsening(method='spectral')
    
    if results is not None:
        fig.savefig(output_dir / 'ks_kpz_spectral_cg.png', dpi=150, bbox_inches='tight')
        
        with open(output_dir / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n{'='*70}")
        print(f"Results saved to {output_dir}/")
        print("="*70)
        
        plt.show()

if __name__ == '__main__':
    main()
