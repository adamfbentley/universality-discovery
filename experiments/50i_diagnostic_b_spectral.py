"""
Experiment 50i: Diagnostic B - Spectral Shape Features

HYPOTHESIS: Gradient moments are KPZ-biased. Spectral shape features
(native to KS physics) might reveal KS→KPZ convergence.

Previous result (Exp 50h):
- With gradient moments: d(KS,KPZ) flat (slope = +0.00043)
- No convergence visible in gradient-moment observable space

This experiment:
- Use spectral SHAPE features (not gradient moments):
  * Low-k slope: α from S(k) ~ k^α 
  * Bandpower fractions: f_low, f_mid, f_high (normalized)
  * Spectral centroid: k_cent (center of mass)
- Multi-σ MMD: average over {σ/2, σ, 2σ, 4σ} to avoid saturation
- Physical k-bands: tied to wavelengths, not raw indices

DECISIVE OUTCOMES:
- d_b decreases: KS→KPZ convergence EXISTS (gradient moments were wrong tool)
- d_b flat: regime issue or genuine non-convergence
- d_b increases: KS spectral shape diverges from KPZ

This tests Fork A: Observable bottleneck hypothesis.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import linregress

sns.set_style("whitegrid")

# ============================================================================
# SPECTRAL SHAPE FEATURES
# ============================================================================

def compute_power_spectrum(h):
    """
    Compute power spectrum S(k) = <|h(k)|^2>.
    
    Returns:
    --------
    k : array
        Wavenumbers (positive only)
    S : array
        Power spectrum (positive k only)
    """
    L = len(h)
    h_fft = fft(h)
    
    # Power spectrum
    S = np.abs(h_fft)**2 / L
    
    # Get wavenumbers
    k = fftfreq(L, d=1.0)  # Assuming domain [0, 2π], dx = 2π/L
    
    # Keep only positive k (spectrum is symmetric)
    positive_k_mask = k > 0
    k_pos = k[positive_k_mask]
    S_pos = S[positive_k_mask]
    
    return k_pos, S_pos

def extract_spectral_shape_features(h, k_bands='auto'):
    """
    Extract spectral SHAPE features (amplitude-independent).
    
    Features:
    ---------
    1. Low-k slope: α from log(S(k)) ~ α*log(k) 
    2. Bandpower fractions: f_low, f_mid, f_high (normalized)
    3. Spectral centroid: k_cent (center of mass, normalized)
    
    Parameters:
    -----------
    h : array
        Height field
    k_bands : str or tuple
        If 'auto': choose physical bands
        If tuple: (k_low_cutoff, k_mid_cutoff, k_high_cutoff)
    
    Returns:
    --------
    features : array [5]
        [low_k_slope, f_low, f_mid, f_high, k_cent_norm]
    """
    k, S = compute_power_spectrum(h)
    
    # Total power (for normalization)
    total_power = np.sum(S)
    
    if total_power < 1e-12:
        # Degenerate case
        return np.zeros(5)
    
    # Define k-bands (PHYSICAL wavelengths, not indices)
    L = len(h)
    k_max = np.max(k)
    
    if k_bands == 'auto':
        # Low: k < 0.2*k_max (long wavelengths)
        # Mid: 0.2*k_max <= k < 0.5*k_max
        # High: k >= 0.5*k_max
        k_low_cutoff = 0.2 * k_max
        k_mid_cutoff = 0.5 * k_max
    else:
        k_low_cutoff, k_mid_cutoff = k_bands
    
    # Band masks
    low_mask = k < k_low_cutoff
    mid_mask = (k >= k_low_cutoff) & (k < k_mid_cutoff)
    high_mask = k >= k_mid_cutoff
    
    # Bandpower fractions (NORMALIZED by total power)
    f_low = np.sum(S[low_mask]) / total_power if np.any(low_mask) else 0.0
    f_mid = np.sum(S[mid_mask]) / total_power if np.any(mid_mask) else 0.0
    f_high = np.sum(S[high_mask]) / total_power if np.any(high_mask) else 0.0
    
    # Low-k slope (fit log(S) ~ α*log(k) over low-k band)
    # Use slightly wider band for fitting: k < 0.3*k_max
    fit_mask = (k > 0) & (k < 0.3 * k_max)
    
    if np.sum(fit_mask) >= 5:  # Need at least 5 points
        k_fit = k[fit_mask]
        S_fit = S[fit_mask]
        
        # Filter out zeros/negatives for log
        valid = S_fit > 1e-12
        if np.sum(valid) >= 3:
            log_k = np.log(k_fit[valid])
            log_S = np.log(S_fit[valid])
            
            slope, intercept, r_value, p_value, std_err = linregress(log_k, log_S)
            low_k_slope = slope
        else:
            low_k_slope = 0.0
    else:
        low_k_slope = 0.0
    
    # Spectral centroid (normalized by k_max so it's scale-independent)
    k_cent = np.sum(k * S) / total_power
    k_cent_norm = k_cent / k_max
    
    features = np.array([
        low_k_slope,
        f_low,
        f_mid,
        f_high,
        k_cent_norm
    ])
    
    return features

# ============================================================================
# MULTI-SIGMA MMD
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

def compute_mmd_single_sigma(X, Y, sigma):
    """Compute MMD² with single bandwidth."""
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
    """Median heuristic for initial bandwidth."""
    sample_X = X[np.random.choice(len(X), size=min(100, len(X)), replace=False)]
    sample_Y = Y[np.random.choice(len(Y), size=min(100, len(Y)), replace=False)]
    
    dists = []
    for x in sample_X[:50]:  # Subsample for speed
        for y in sample_Y[:50]:
            dists.append(np.linalg.norm(x - y))
    
    return np.median(dists)

def compute_mmd_multi_sigma(X, Y, sigma_base):
    """
    Compute MMD averaged over multiple bandwidths.
    
    Uses: {σ/2, σ, 2σ, 4σ} to avoid saturation artifacts.
    
    Returns:
    --------
    mmd_mean : float
        Average MMD across bandwidths
    mmd_by_sigma : dict
        {sigma: mmd} for each bandwidth
    """
    sigma_scales = [0.5, 1.0, 2.0, 4.0]
    mmds = []
    mmd_by_sigma = {}
    
    for scale in sigma_scales:
        sigma = sigma_base * scale
        mmd = compute_mmd_single_sigma(X, Y, sigma)
        mmds.append(mmd)
        mmd_by_sigma[scale] = mmd
    
    return np.mean(mmds), mmd_by_sigma

# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

def load_and_extract_features(data_path, system_name):
    """
    Load fields and extract spectral shape features at all scales.
    
    Parameters:
    -----------
    data_path : Path
        Path to .pkl file with 'fields' and 'features_by_scale'
    system_name : str
        'KS' or 'KPZ' for printing
    
    Returns:
    --------
    features_by_scale : dict
        {scale: features_array} for spectral shape features
    """
    print(f"\nLoading {system_name} data from {data_path.name}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    fields = data['fields']
    scales = data['scales']
    scale_names = data['scale_names']
    
    print(f"  Loaded {len(fields)} field samples")
    print(f"  Field shape: {fields.shape}")
    
    # Coarse-grain fields spectrally and extract features
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
    
    print("  Extracting spectral shape features at each scale...")
    features_by_scale = {}
    
    for scale, name in zip(scales, scale_names):
        features_list = []
        
        for h in fields:
            # Coarse-grain field
            h_coarse = coarse_grain_field_spectral(h, scale)
            
            # Extract spectral shape features
            features = extract_spectral_shape_features(h_coarse)
            features_list.append(features)
        
        features_by_scale[scale] = np.array(features_list)
        
        print(f"    {name}: {len(features_list)} samples, feature shape {features_by_scale[scale].shape}")
        print(f"      Feature ranges: [{features_by_scale[scale].min(axis=0)}, {features_by_scale[scale].max(axis=0)}]")
    
    return features_by_scale, scales, scale_names

# ============================================================================
# COMPARISON
# ============================================================================

def compare_with_spectral_features(ks_features, kpz_features, scales, scale_names):
    """
    Compare KS vs KPZ using spectral shape features with multi-σ MMD.
    """
    print("\n" + "="*70)
    print("COMPUTING MULTI-σ MMD DISTANCES")
    print("="*70)
    
    # Estimate base bandwidth at b=1
    sigma_base = estimate_bandwidth(ks_features[1.0], kpz_features[1.0])
    print(f"\nBase bandwidth (estimated at b=1): σ = {sigma_base:.4f}")
    print(f"  Will use: {{σ/2={sigma_base/2:.4f}, σ={sigma_base:.4f}, 2σ={2*sigma_base:.4f}, 4σ={4*sigma_base:.4f}}}")
    
    distances = []
    distances_by_sigma = {0.5: [], 1.0: [], 2.0: [], 4.0: []}
    
    print("\nScale-by-scale comparison:")
    for scale, name in zip(scales, scale_names):
        X = ks_features[scale]
        Y = kpz_features[scale]
        
        # Multi-σ MMD
        d_mean, d_dict = compute_mmd_multi_sigma(X, Y, sigma_base)
        distances.append(d_mean)
        
        for sigma_scale, d in d_dict.items():
            distances_by_sigma[sigma_scale].append(d)
        
        print(f"  {name}: d_mean = {d_mean:.4f}  (n_KS={len(X)}, n_KPZ={len(Y)})")
        print(f"    By σ: {{σ/2: {d_dict[0.5]:.4f}, σ: {d_dict[1.0]:.4f}, 2σ: {d_dict[2.0]:.4f}, 4σ: {d_dict[4.0]:.4f}}}")
    
    return np.array(distances), distances_by_sigma, sigma_base

def analyze_trend(distances, scale_names):
    """Analyze trend in distances."""
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
    
    if slope < -0.02 and relative_change < -0.05:
        print("✅ CONVERGENCE: Distance decreases with coarse-graining")
        print("   → KS flows toward KPZ in spectral shape space!")
        print("   → Gradient moments were WRONG TOOL (KPZ-biased)")
        print("   → Framework GENERALIZES (with right observables)")
        interpretation = "convergence"
    elif abs(slope) < 0.02 and abs(relative_change) < 0.05:
        print("⚠️  FLAT: Distance roughly constant across scales")
        print("   → Spectral shape also doesn't show convergence")
        print("   → Likely regime issue or genuine non-convergence")
        print("   → Next: Parameter sweep or accept boundary")
        interpretation = "flat"
    elif slope < 0.05:
        print("⚠️  WEAK CHANGE: Small trend")
        print(f"   → {'Slight convergence' if slope < 0 else 'Slight divergence'}")
        print("   → Borderline, may need larger L or different regime")
        interpretation = "weak"
    else:
        print("❌ DIVERGENCE: Distance increases significantly")
        print("   → KS spectral shape diverges from KPZ")
        print("   → Genuine physics difference in this regime")
        interpretation = "divergence"
    
    return slope, p_value, relative_change, interpretation

# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(distances, distances_by_sigma, sigma_base, scale_names, 
                slope, interpretation, comparison_to_50h):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Multi-σ MMD distances
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(distances))
    
    # Individual σ curves
    colors = ['C0', 'C1', 'C2', 'C3']
    sigma_labels = ['σ/2', 'σ', '2σ', '4σ']
    for (sigma_scale, dists), color, label in zip(distances_by_sigma.items(), colors, sigma_labels):
        ax.plot(x, dists, 'o--', alpha=0.5, color=color, 
               label=f'{label} (σ={sigma_base*sigma_scale:.3f})', markersize=6)
    
    # Mean curve (bold)
    ax.plot(x, distances, 'o-', color='black', linewidth=3, markersize=10,
           label=f'Mean (slope={slope:+.4f})', zorder=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Coarse-graining scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('MMD Distance', fontsize=12, fontweight='bold')
    ax.set_title('Multi-σ MMD: KS vs KPZ (Spectral Shape Features)', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Plot 2: Comparison to Exp 50h (gradient moments)
    ax = fig.add_subplot(gs[0, 2])
    
    # Exp 50h (gradient moments)
    exp50h_distances = comparison_to_50h
    ax.plot(x, exp50h_distances, 's-', label='Exp 50h (gradient)', 
           color='red', markersize=8, linewidth=2, alpha=0.7)
    
    # This experiment (spectral)
    ax.plot(x, distances, 'o-', label='Exp 50i (spectral)', 
           color='blue', markersize=8, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names, fontsize=8)
    ax.set_xlabel('Scale', fontsize=11)
    ax.set_ylabel('MMD', fontsize=11)
    ax.set_title('Observable Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Feature distributions at b=1 vs b=16
    # (Will be filled in by calling code - placeholder)
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.5, 'Sample spectral\nfeatures\n(see console output)', 
           ha='center', va='center', fontsize=10)
    ax.set_title('Feature Space (b=1)', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.text(0.5, 0.5, 'Sample spectral\nfeatures\n(see console output)', 
           ha='center', va='center', fontsize=10)
    ax.set_title('Feature Space (b=16)', fontsize=11)
    ax.axis('off')
    
    # Plot 4: Interpretation summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    summary_text = f"""
EXPERIMENT 50i: DIAGNOSTIC B

Observables: Spectral Shape
• Low-k slope (α)
• Bandpower fractions (normalized)
• Spectral centroid (shape)

Multi-σ MMD (σ={sigma_base:.3f}):
• {sigma_labels[0]}, {sigma_labels[1]}, {sigma_labels[2]}, {sigma_labels[3]}

Result:
• Slope: {slope:+.5f}
• Change: {((distances[-1]-distances[0])/distances[0]):+.2%}
• Interpretation: {interpretation.upper()}

Comparison to 50h:
• Gradient: {comparison_to_50h[-1]-comparison_to_50h[0]:+.4f}
• Spectral: {distances[-1]-distances[0]:+.4f}
• Observable matters: {"YES" if abs(slope - 0.00043) > 0.01 else "NO"}
"""
    
    ax.text(0.05, 0.5, summary_text, fontsize=9, verticalalignment='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 5: Log-log to check power law
    ax = fig.add_subplot(gs[2, 0])
    b_values = np.array([1, 2, 4, 8, 16])
    ax.loglog(b_values, distances, 'o-', markersize=10, linewidth=2, color='C0')
    ax.set_xlabel('Coarse-graining factor b', fontsize=11)
    ax.set_ylabel('MMD Distance', fontsize=11)
    ax.set_title('Log-Log Plot', fontsize=12)
    ax.grid(alpha=0.3, which='both')
    
    # Plot 6: Residuals from linear fit
    ax = fig.add_subplot(gs[2, 1])
    fit_line = slope * x + distances[0]
    residuals = distances - fit_line
    ax.plot(x, residuals, 'o-', markersize=8, linewidth=2, color='C1')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Scale', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residuals from Linear Fit', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 7: Decision tree
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    decision_text = """
DECISION TREE:

If CONVERGENCE:
→ Observable bottleneck confirmed
→ Framework generalizes!
→ Next: Test more systems

If FLAT:
→ Regime issue or non-convergence
→ Next: Parameter sweep
→ Or accept KPZ-family boundary

If DIVERGENCE:
→ Genuine physics difference
→ Accept boundary
→ Document systematically
"""
    
    ax.text(0.05, 0.5, decision_text, fontsize=9, verticalalignment='center',
           family='monospace')
    
    plt.suptitle(f'Diagnostic B: Spectral Shape Features | Result: {interpretation.upper()}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50i: DIAGNOSTIC B - SPECTRAL SHAPE FEATURES")
    print("="*70)
    print("\nHypothesis: Gradient moments are KPZ-biased.")
    print("Test: Do spectral shape features reveal KS→KPZ convergence?")
    print("\nFeatures: Low-k slope, bandpower fractions, spectral centroid")
    print("Method: Multi-σ MMD averaged over {σ/2, σ, 2σ, 4σ}")
    print("="*70)
    
    output_dir = Path('results/exp50i_diagnostic_b_spectral')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load KS data (regenerate features)
    ks_path = Path('results/exp50h_ks_vs_kpz_matched/ks_fields.pkl')
    if not ks_path.exists():
        print("\n⚠️  KS fields not saved in 50h. Loading from 50g KPZ structure...")
        # Generate KS fresh (same as 50h did)
        print("\nGenerating KS data...")
        from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator
        
        n_trajs = 20
        all_fields = []
        sim = KuramotoSivashinskySimulator(L=256, dt=0.01)
        
        for i in range(n_trajs):
            trajectory = sim.simulate(T=2000, nu=0.1, kappa=1.0, lam=1.0,
                                     noise_strength=0.5, record_interval=20)
            n_transient = len(trajectory) // 2
            for t in range(n_transient, len(trajectory)):
                all_fields.append(trajectory[t])
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{n_trajs} trajectories")
        
        ks_fields = np.array(all_fields)
        print(f"Generated {len(ks_fields)} KS fields")
        
        # Package for extraction
        ks_data = {
            'fields': ks_fields,
            'scales': [1.0, 0.5, 0.25, 0.125, 0.0625],
            'scale_names': ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']
        }
        
        # Save for future use
        with open(output_dir / 'ks_fields.pkl', 'wb') as f:
            pickle.dump(ks_data, f)
        
        ks_features, scales, scale_names = load_and_extract_features(
            output_dir / 'ks_fields.pkl', 'KS')
    else:
        ks_features, scales, scale_names = load_and_extract_features(ks_path, 'KS')
    
    # Load KPZ data
    kpz_path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')
    kpz_features, _, _ = load_and_extract_features(kpz_path, 'KPZ')
    
    # Compare
    distances, distances_by_sigma, sigma_base = compare_with_spectral_features(
        ks_features, kpz_features, scales, scale_names)
    
    # Analyze
    slope, p_value, relative_change, interpretation = analyze_trend(distances, scale_names)
    
    # Load Exp 50h for comparison
    with open(Path('results/exp50h_ks_vs_kpz_matched/results.pkl'), 'rb') as f:
        exp50h_data = pickle.load(f)
    exp50h_distances = exp50h_data['distances']
    
    # Plot
    print("\n" + "="*70)
    print("CREATING PLOTS")
    print("="*70)
    
    fig = plot_results(distances, distances_by_sigma, sigma_base, scale_names,
                      slope, interpretation, exp50h_distances)
    fig.savefig(output_dir / 'spectral_comparison.png', dpi=150, bbox_inches='tight')
    
    # Save results
    results = {
        'method': 'spectral_shape_features',
        'distances': distances,
        'distances_by_sigma': distances_by_sigma,
        'sigma_base': sigma_base,
        'scales': scales,
        'scale_names': scale_names,
        'slope': slope,
        'p_value': p_value,
        'relative_change': relative_change,
        'interpretation': interpretation,
        'exp50h_comparison': exp50h_distances,
        'ks_sample_sizes': [len(ks_features[s]) for s in scales],
        'kpz_sample_sizes': [len(kpz_features[s]) for s in scales]
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC B COMPLETE")
    print("="*70)
    print(f"\nKey finding: Slope = {slope:+.5f} ({interpretation})")
    print(f"Relative change: {relative_change:+.2%}")
    
    print(f"\nComparison to Exp 50h (gradient moments):")
    print(f"  Gradient slope: +0.00043 (flat)")
    print(f"  Spectral slope: {slope:+.5f}")
    print(f"  Difference: {slope - 0.00043:+.5f}")
    
    if interpretation == "convergence":
        print("\n🎉 BREAKTHROUGH: Observable bottleneck confirmed!")
        print("   → Framework GENERALIZES with right observables")
        print("   → Gradient moments were KPZ-biased")
        print("   → Spectral features reveal KS→KPZ flow")
        print("\n   Next: Test more systems (MBE, Burgers, anisotropic KPZ)")
    elif interpretation == "flat":
        print("\n⏭️  REGIME ISSUE: Spectral features also flat")
        print("   → Not just observable bottleneck")
        print("   → Likely wrong KS regime or genuine non-convergence")
        print("\n   Next options:")
        print("   1. Parameter sweep (vary ν, κ, λ)")
        print("   2. Larger system size (L > 256)")
        print("   3. Accept KPZ-family boundary")
    else:
        print("\n🔬 BOUNDARY FOUND: KS spectral shape diverges from KPZ")
        print("   → Genuine physics difference in this regime")
        print("   → Framework scope: KPZ-family growth models")
        print("\n   This is still valuable: knowing boundaries is science")
    
    plt.show()

if __name__ == '__main__':
    main()
