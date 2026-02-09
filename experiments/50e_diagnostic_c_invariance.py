"""
Experiment 50e: Diagnostic C - Pipeline Invariance Check

CRITICAL TEST: Does the pipeline produce stable distances for KPZ vs KPZ?

If d(KPZ1, KPZ2) drifts systematically with b, the Exp 50c result 
(distance increases) is likely an ARTIFACT of:
1. Bandwidth drift (median heuristic recomputed at each b)
2. Feature-averaging changing distribution shape
3. Scaler interaction with block averaging

This is the MOST DECISIVE test before interpreting any KS results.
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

sns.set_style("whitegrid")

# ============================================================================
# MMD WITH FIXED BANDWIDTH
# ============================================================================

def compute_mmd_distance_fixed_sigma(X, Y, sigma):
    """
    Compute MMD with FIXED bandwidth (no median heuristic).
    
    This prevents bandwidth drift across coarse-graining scales.
    """
    # RBF kernel
    def rbf_kernel(A, B):
        pairwise_sq = pairwise_distances(A, B, metric='sqeuclidean')
        return np.exp(-pairwise_sq / (2 * sigma**2))
    
    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)
    
    # MMD²
    mmd_sq = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    
    return max(0, mmd_sq)**0.5

def estimate_bandwidth(X, Y):
    """
    Estimate bandwidth from raw (b=1) data using median heuristic.
    
    This bandwidth will be FROZEN for all subsequent scales.
    """
    Z = np.vstack([X, Y])
    pairwise = pairwise_distances(Z, metric='euclidean')
    sigma = np.median(pairwise[pairwise > 0])
    
    if sigma == 0:
        sigma = 1.0
    
    return sigma

# ============================================================================
# COARSE-GRAINING
# ============================================================================

def coarse_grain_features(features, block_size):
    """Block averaging (same as Exp 50c)."""
    n_samples = len(features)
    n_blocks = n_samples // block_size
    
    if n_blocks == 0:
        return features
    
    features_trunc = features[:n_blocks * block_size]
    features_reshaped = features_trunc.reshape(n_blocks, block_size, -1)
    features_coarse = features_reshaped.mean(axis=1)
    
    return features_coarse

# ============================================================================
# LOAD KPZ DATA
# ============================================================================

def load_kpz_data():
    """Load KPZ features."""
    kpz_path = Path('results/exp46_coupling_coordinate/coupling_coordinate_results.pkl')
    
    if not kpz_path.exists():
        print("❌ No KPZ data found.")
        return None
    
    with open(kpz_path, 'rb') as f:
        kpz_data = pickle.load(f)
    
    features = kpz_data['features']
    
    print(f"Loaded {len(features)} KPZ samples")
    
    return features

# ============================================================================
# DIAGNOSTIC C: KPZ vs KPZ
# ============================================================================

def test_kpz_vs_kpz_stability(kpz_features):
    """
    Test if pipeline produces stable distances for KPZ vs KPZ.
    
    Split KPZ data into two halves (or bootstrap samples) and test
    if d(KPZ1, KPZ2) remains flat under coarse-graining.
    """
    print("="*70)
    print("DIAGNOSTIC C: PIPELINE INVARIANCE CHECK")
    print("="*70)
    print("\nTesting: d(KPZ1, KPZ2) vs scale")
    print("Expected: Should be flat (near zero) across all b")
    print("="*70)
    
    # Split into two independent samples
    n_half = len(kpz_features) // 2
    kpz1 = kpz_features[:n_half]
    kpz2 = kpz_features[n_half:2*n_half]
    
    print(f"\nKPZ1: {len(kpz1)} samples")
    print(f"KPZ2: {len(kpz2)} samples")
    
    # Standardize (fit on combined)
    scaler = StandardScaler()
    scaler.fit(np.vstack([kpz1, kpz2]))
    
    kpz1_scaled = scaler.transform(kpz1)
    kpz2_scaled = scaler.transform(kpz2)
    
    # Estimate bandwidth from raw data (b=1) - FREEZE THIS
    sigma_fixed = estimate_bandwidth(kpz1_scaled, kpz2_scaled)
    print(f"\nFixed bandwidth (from b=1): σ = {sigma_fixed:.4f}")
    
    # Test at multiple scales
    scales = [1, 2, 4, 8, 16]
    
    # Test 1: With RECOMPUTED bandwidth (like Exp 50c)
    print("\n" + "-"*70)
    print("TEST 1: RECOMPUTED BANDWIDTH (current Exp 50c method)")
    print("-"*70)
    
    distances_recomputed = []
    sigmas_recomputed = []
    
    for b in scales:
        kpz1_coarse = coarse_grain_features(kpz1_scaled, b)
        kpz2_coarse = coarse_grain_features(kpz2_scaled, b)
        
        if len(kpz1_coarse) < 5 or len(kpz2_coarse) < 5:
            break
        
        # Recompute sigma (as in Exp 50c)
        sigma_b = estimate_bandwidth(kpz1_coarse, kpz2_coarse)
        
        # Compute distance with this sigma
        dist = compute_mmd_distance_fixed_sigma(kpz1_coarse, kpz2_coarse, sigma_b)
        
        distances_recomputed.append(dist)
        sigmas_recomputed.append(sigma_b)
        
        print(f"  b={b:2d}: d={dist:.4f}, σ={sigma_b:.4f} ({len(kpz1_coarse)} samples)")
    
    # Test 2: With FIXED bandwidth
    print("\n" + "-"*70)
    print("TEST 2: FIXED BANDWIDTH (correct method)")
    print("-"*70)
    
    distances_fixed = []
    
    for b in scales:
        kpz1_coarse = coarse_grain_features(kpz1_scaled, b)
        kpz2_coarse = coarse_grain_features(kpz2_scaled, b)
        
        if len(kpz1_coarse) < 5 or len(kpz2_coarse) < 5:
            break
        
        # Use FIXED sigma from b=1
        dist = compute_mmd_distance_fixed_sigma(kpz1_coarse, kpz2_coarse, sigma_fixed)
        
        distances_fixed.append(dist)
        
        print(f"  b={b:2d}: d={dist:.4f}, σ={sigma_fixed:.4f} (fixed) ({len(kpz1_coarse)} samples)")
    
    # Analyze trends
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    from scipy.stats import linregress
    
    if len(distances_recomputed) >= 3:
        log_scales = np.log(scales[:len(distances_recomputed)])
        
        slope1, _, r1, p1, _ = linregress(log_scales, distances_recomputed)
        slope2, _, r2, p2, _ = linregress(log_scales, distances_fixed)
        
        print(f"\nRecomputed bandwidth:")
        print(f"  Slope: {slope1:+.4f} (R²={r1**2:.3f}, p={p1:.3e})")
        
        print(f"\nFixed bandwidth:")
        print(f"  Slope: {slope2:+.4f} (R²={r2**2:.3f}, p={p2:.3e})")
        
        print("\n" + "="*70)
        print("VERDICT:")
        print("="*70)
        
        if abs(slope1) > 0.01 and p1 < 0.1:
            print("⚠️  RECOMPUTED BANDWIDTH: Systematic drift detected")
            print("    → Exp 50c slope +0.085 likely includes bandwidth artifact")
        else:
            print("✅ RECOMPUTED BANDWIDTH: Stable (no drift)")
        
        if abs(slope2) > 0.01 and p2 < 0.1:
            print("⚠️  FIXED BANDWIDTH: Systematic drift detected")
            print("    → Feature-space averaging itself is problematic")
            print("    → Need field-level coarse-graining (Diagnostic A)")
        else:
            print("✅ FIXED BANDWIDTH: Stable (no drift)")
            print("    → Pipeline is valid for RG tests")
        
        verdict_recomputed = "drift" if (abs(slope1) > 0.01 and p1 < 0.1) else "stable"
        verdict_fixed = "drift" if (abs(slope2) > 0.01 and p2 < 0.1) else "stable"
    else:
        verdict_recomputed = "insufficient_data"
        verdict_fixed = "insufficient_data"
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test 1: Recomputed
    ax = axes[0]
    ax.plot(scales[:len(distances_recomputed)], distances_recomputed, 
           'o-', markersize=8, linewidth=2, color='C0')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coarse-graining scale b')
    ax.set_ylabel('d(KPZ1, KPZ2)')
    ax.set_title('Recomputed Bandwidth (Exp 50c method)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    if len(distances_recomputed) >= 3:
        ax.text(0.05, 0.95, f'slope={slope1:+.4f}\np={p1:.2e}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Test 2: Fixed
    ax = axes[1]
    ax.plot(scales[:len(distances_fixed)], distances_fixed, 
           'o-', markersize=8, linewidth=2, color='C2')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coarse-graining scale b')
    ax.set_ylabel('d(KPZ1, KPZ2)')
    ax.set_title('Fixed Bandwidth (correct)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    if len(distances_fixed) >= 3:
        ax.text(0.05, 0.95, f'slope={slope2:+.4f}\np={p2:.2e}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Bandwidth drift plot
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.plot(scales[:len(sigmas_recomputed)], sigmas_recomputed, 's-', 
           markersize=8, linewidth=2, label='Recomputed σ')
    ax.axhline(sigma_fixed, color='C2', linestyle='--', linewidth=2, 
              label=f'Fixed σ = {sigma_fixed:.3f}')
    ax.set_xlabel('Coarse-graining scale b')
    ax.set_ylabel('Kernel bandwidth σ')
    ax.set_title('Bandwidth Drift Check')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    results = {
        'scales': scales[:len(distances_recomputed)],
        'distances_recomputed': distances_recomputed,
        'distances_fixed': distances_fixed,
        'sigmas_recomputed': sigmas_recomputed,
        'sigma_fixed': sigma_fixed,
        'verdict_recomputed': verdict_recomputed,
        'verdict_fixed': verdict_fixed
    }
    
    return results, fig, fig2

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50e: DIAGNOSTIC C - PIPELINE INVARIANCE")
    print("="*70)
    print("\nQuestion: Is the Exp 50c pipeline stable for identical physics?")
    print("="*70)
    
    output_dir = Path('results/exp50e_diagnostic_c')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load KPZ data
    kpz_features = load_kpz_data()
    
    if kpz_features is None:
        print("\n❌ Cannot proceed without KPZ data")
        return
    
    # Test KPZ vs KPZ
    results, fig1, fig2 = test_kpz_vs_kpz_stability(kpz_features)
    
    fig1.savefig(output_dir / 'kpz_vs_kpz_distance.png', dpi=150, bbox_inches='tight')
    fig2.savefig(output_dir / 'bandwidth_drift.png', dpi=150, bbox_inches='tight')
    
    # Save results
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_dir}/")
    print("="*70)
    
    plt.show()

if __name__ == '__main__':
    main()
