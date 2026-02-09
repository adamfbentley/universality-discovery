"""
Experiment 50h: KS vs KPZ with Matched Data

DECISIVE TEST: Now that KPZ is regenerated with matched protocol,
rerun the comparison with symmetric data.

Previous result (Exp 50f):
- Slope: +0.0185 (distance 0.803 → 0.877)
- BUT: KPZ had only 5 samples at b=16, L=128
- Conclusion: Inconclusive due to asymmetry

This experiment:
- KS: 1000 samples at all b, L=256, field-level spectral CG
- KPZ: 1250 samples at all b, L=256, field-level spectral CG (NEW!)
- Fixed bandwidth (no median heuristic artifact)
- Symmetric protocols, fair comparison

INTERPRETATION GUIDE:
- Slope negative or ~0: KS → KPZ convergence! Framework generalizes.
- Slope positive but small (<0.05): Inconclusive, try Diagnostic B
- Slope positive and large (>0.1): Robust divergence, different physics
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.stats import linregress

sns.set_style("whitegrid")

# ============================================================================
# MMD DISTANCE (Fixed Bandwidth)
# ============================================================================

def rbf_kernel(X, Y, sigma):
    """RBF kernel: k(x,y) = exp(-||x-y||²/(2σ²))"""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    XY = X @ Y.T
    
    distances_sq = XX - 2*XY + YY
    return np.exp(-distances_sq / (2 * sigma**2))

def compute_mmd_distance_fixed_sigma(X, Y, sigma):
    """
    Compute MMD² with FIXED bandwidth (no recomputation).
    
    Parameters:
    -----------
    X, Y : arrays (n_samples, n_features)
        Two distributions
    sigma : float
        FIXED kernel bandwidth
    """
    n_x = len(X)
    n_y = len(Y)
    
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (n_x * (n_x - 1))
    mmd_sq += (np.sum(K_YY) - np.trace(K_YY)) / (n_y * (n_y - 1))
    mmd_sq -= 2 * np.mean(K_XY)
    
    return np.sqrt(max(0, mmd_sq))

def estimate_bandwidth(X, Y):
    """Median heuristic for bandwidth (computed ONCE at b=1)."""
    sample_X = X[np.random.choice(len(X), size=min(100, len(X)), replace=False)]
    sample_Y = Y[np.random.choice(len(Y), size=min(100, len(Y)), replace=False)]
    
    dists = []
    for x in sample_X:
        for y in sample_Y:
            dists.append(np.linalg.norm(x - y))
    
    return np.median(dists)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_ks_data():
    """Generate KS data with field-level spectral coarse-graining."""
    print("Generating KS data with field-level coarse-graining...")
    
    from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator
    from scipy.fft import fft, ifft
    
    def coarse_grain_field_spectral(h, k_cutoff_fraction):
        """Spectral low-pass coarse-graining."""
        L = len(h)
        h_hat = fft(h)
        k = np.fft.fftfreq(L)
        k_max = np.max(np.abs(k))
        k_cutoff = k_cutoff_fraction * k_max
        mask = np.abs(k) <= k_cutoff
        h_hat_filtered = h_hat * mask
        return np.real(ifft(h_hat_filtered))
    
    def extract_gradient_moments(h):
        """Extract gradient moment features."""
        grad = np.gradient(h)
        return np.array([
            np.mean(grad**2),
            np.mean(grad**3),
            np.mean(grad**4),
            np.mean(grad**5),
            np.mean(grad**6),
            np.mean(np.abs(grad)**7)
        ])
    
    # Generate KS trajectories
    n_trajs = 20  # 20 * 50 = 1000 samples
    all_fields = []
    
    print(f"  Generating {n_trajs} KS trajectories...")
    
    sim = KuramotoSivashinskySimulator(L=256, dt=0.01)
    
    for i in range(n_trajs):
        trajectory = sim.simulate(
            T=2000,
            nu=0.1,
            kappa=1.0,
            lam=1.0,
            noise_strength=0.5,
            record_interval=20
        )
        
        # Skip transient
        n_transient = len(trajectory) // 2
        for t in range(n_transient, len(trajectory)):
            all_fields.append(trajectory[t])
        
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{n_trajs} trajectories complete")
    
    all_fields = np.array(all_fields)
    print(f"  Generated {len(all_fields)} KS field samples")
    
    # Apply spectral coarse-graining at all scales
    scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
    features_by_scale = {}
    
    print("  Applying spectral coarse-graining...")
    for scale in scales:
        features_list = []
        for h in all_fields:
            h_coarse = coarse_grain_field_spectral(h, scale)
            features = extract_gradient_moments(h_coarse)
            features_list.append(features)
        features_by_scale[scale] = np.array(features_list)
        print(f"    Scale {scale:.3f}: {len(features_list)} samples")
    
    return features_by_scale

def load_kpz_matched():
    """Load newly regenerated KPZ data from Exp 50g."""
    print("Loading matched KPZ data...")
    path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    features_by_scale = data['features_by_scale']
    
    print(f"  Loaded KPZ: {len(features_by_scale[1.0])} samples at b=1")
    print(f"  Feature shape: {features_by_scale[1.0].shape}")
    
    return features_by_scale

# ============================================================================
# COMPARISON
# ============================================================================

def compare_ks_vs_kpz(ks_features, kpz_features, scales):
    """
    Compare KS vs KPZ across scales with FIXED bandwidth.
    """
    print("\n" + "="*70)
    print("COMPUTING MMD DISTANCES (FIXED BANDWIDTH)")
    print("="*70)
    
    # Estimate bandwidth at b=1 (FREEZE for all scales)
    sigma = estimate_bandwidth(ks_features[1.0], kpz_features[1.0])
    print(f"\nFixed bandwidth (estimated at b=1): σ = {sigma:.4f}")
    print("  (Will use this for ALL scales)")
    
    distances = []
    scale_names = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']
    
    print("\nScale-by-scale comparison:")
    for scale, name in zip(scales, scale_names):
        X = ks_features[scale]
        Y = kpz_features[scale]
        
        # Fixed bandwidth
        d = compute_mmd_distance_fixed_sigma(X, Y, sigma)
        distances.append(d)
        
        print(f"  {name}: d(KS,KPZ) = {d:.4f}  (n_KS={len(X)}, n_KPZ={len(Y)})")
    
    return np.array(distances), sigma, scale_names

def analyze_trend(distances, scale_names):
    """
    Analyze trend in distances.
    """
    print("\n" + "="*70)
    print("TREND ANALYSIS")
    print("="*70)
    
    x = np.arange(len(distances))
    slope, intercept, r_value, p_value, std_err = linregress(x, distances)
    
    print(f"\nLinear regression (distance vs scale index):")
    print(f"  Slope: {slope:+.5f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    
    # Relative change
    relative_change = (distances[-1] - distances[0]) / distances[0]
    print(f"\nRelative change (b=1 → b=16):")
    print(f"  Absolute: {distances[-1] - distances[0]:+.4f}")
    print(f"  Relative: {relative_change:+.2%}")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    if slope < -0.02:
        print("✅ CONVERGENCE: Distance decreases with coarse-graining")
        print("   → KS flows toward KPZ under RG")
        print("   → Framework generalizes! Gradient moments detect IR universality")
        interpretation = "convergence"
    elif abs(slope) < 0.02:
        print("⚠️  FLAT: Distance roughly constant across scales")
        print("   → No clear convergence with gradient moments")
        print("   → Next: Try Diagnostic B (spectral observables)")
        interpretation = "flat"
    elif slope < 0.05:
        print("⚠️  WEAK DIVERGENCE: Small positive slope")
        print("   → Slight increase, but not dramatic")
        print("   → Could be regime-specific or require different observables")
        print("   → Next: Diagnostic B or parameter sweep")
        interpretation = "weak_divergence"
    else:
        print("❌ DIVERGENCE: Distance increases significantly")
        print("   → KS and KPZ are robustly different with these observables")
        print("   → Try Diagnostic B or accept scope boundary")
        interpretation = "divergence"
    
    return slope, p_value, interpretation

# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(distances, sigma, scale_names, slope, interpretation):
    """
    Create comprehensive visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distance vs scale
    ax = axes[0, 0]
    x = np.arange(len(distances))
    ax.plot(x, distances, 'o-', markersize=10, linewidth=2, color='C0', label='d(KS,KPZ)')
    
    # Linear fit
    fit_line = slope * x + distances[0]
    ax.plot(x, fit_line, '--', alpha=0.5, color='C3', 
           label=f'Linear fit (slope={slope:+.4f})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Coarse-graining scale', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    ax.set_title(f'KS vs KPZ Comparison (Fixed σ={sigma:.3f})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Color code by interpretation
    color_map = {
        'convergence': 'green',
        'flat': 'orange',
        'weak_divergence': 'gold',
        'divergence': 'red'
    }
    ax.axhline(distances[0], color='gray', linestyle=':', alpha=0.3)
    
    # Plot 2: Log-log (check power law)
    ax = axes[0, 1]
    b_values = np.array([1, 2, 4, 8, 16])
    ax.loglog(b_values, distances, 'o-', markersize=10, linewidth=2, color='C0')
    ax.set_xlabel('Coarse-graining factor b', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    ax.set_title('Log-Log Plot (Power Law Check)', fontsize=13)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 3: Comparison to previous experiments
    ax = axes[1, 0]
    
    # Exp 50c (artifact-contaminated)
    exp50c_distances = [0.803, 0.843, 0.879, 0.930, 0.979]  # From previous
    ax.plot(x, exp50c_distances, 'x--', alpha=0.5, label='Exp 50c (recomputed σ)', 
           color='red', markersize=8)
    
    # Exp 50f (asymmetric data)
    exp50f_distances = [0.803, 0.827, 0.852, 0.865, 0.877]  # From previous
    ax.plot(x, exp50f_distances, 's--', alpha=0.5, label='Exp 50f (KPZ: L=128, n=5)', 
           color='orange', markersize=8)
    
    # This experiment (matched)
    ax.plot(x, distances, 'o-', label='Exp 50h (matched, L=256, n=1250)', 
           color='C0', markersize=10, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Coarse-graining scale', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    ax.set_title('Comparison Across Experiments', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Interpretation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
EXPERIMENT 50h: DECISIVE RESULT

Setup:
• KS: 1000 samples, L=256, field-level CG
• KPZ: 1250 samples, L=256, field-level CG
• Fixed bandwidth: σ = {sigma:.4f}
• Symmetric protocols ✓

Results:
• Initial distance: {distances[0]:.4f}
• Final distance: {distances[-1]:.4f}
• Slope: {slope:+.5f}
• Relative change: {(distances[-1]-distances[0])/distances[0]:+.2%}

Interpretation: {interpretation.upper().replace('_', ' ')}

Previous Results (for reference):
• Exp 50c: slope = +0.085 (artifact!)
• Exp 50f: slope = +0.0185 (asymmetric)
• Exp 50h: slope = {slope:+.4f} (THIS)

Change from 50f: {slope - 0.0185:+.4f}
→ {"Slope decreased!" if slope < 0.0185 else "Slope increased" if slope > 0.0185 else "Slope unchanged"}
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50h: KS vs KPZ WITH MATCHED DATA")
    print("="*70)
    print("\nDECISIVE TEST: Symmetric protocols, fair comparison")
    print("  - Both L=256, dt=0.01")
    print("  - Both use field-level spectral coarse-graining")
    print("  - Constant sample sizes (1000+ at all b)")
    print("  - Fixed bandwidth (no artifact)")
    print("="*70)
    
    output_dir = Path('results/exp50h_ks_vs_kpz_matched')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    ks_features = load_ks_data()
    kpz_features = load_kpz_matched()
    
    scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
    
    # Compare
    distances, sigma, scale_names = compare_ks_vs_kpz(ks_features, kpz_features, scales)
    
    # Analyze trend
    slope, p_value, interpretation = analyze_trend(distances, scale_names)
    
    # Plot
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    fig = plot_results(distances, sigma, scale_names, slope, interpretation)
    fig.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    
    # Save results
    results = {
        'distances': distances,
        'sigma': sigma,
        'scale_names': scale_names,
        'scales': scales,
        'slope': slope,
        'p_value': p_value,
        'interpretation': interpretation,
        'ks_sample_sizes': [len(ks_features[s]) for s in scales],
        'kpz_sample_sizes': [len(kpz_features[s]) for s in scales]
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nKey finding: Slope = {slope:+.5f} ({interpretation})")
    print(f"\nComparison to Exp 50f:")
    print(f"  Previous slope (asymmetric): +0.0185")
    print(f"  Current slope (matched): {slope:+.5f}")
    print(f"  Difference: {slope - 0.0185:+.5f}")
    
    if interpretation == "convergence":
        print("\n🎉 BREAKTHROUGH: Framework generalizes beyond KPZ!")
        print("   Gradient moments detect KS → KPZ crossover")
        print("   Next: Test more systems (MBE, anisotropic KPZ, Burgers)")
    elif interpretation == "flat":
        print("\n⏭️  NEXT STEP: Diagnostic B (spectral observables)")
        print("   Gradient moments may be KPZ-biased")
        print("   Try: Spectral slope, power fractions, structure functions")
    else:
        print("\n🔬 BOUNDARY FOUND: Consider Diagnostic B or parameter sweep")
        print("   Different observables or KS regime may show convergence")
    
    plt.show()

if __name__ == '__main__':
    main()
