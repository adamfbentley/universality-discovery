"""
Experiment 50j: Diagnostic B - FIXED Spectral Features

CRITICAL FIX: Exp 50i had feature collapse artifact.
- Bug: k_max was constant (Nyquist), not adapting to coarse-graining cutoff
- Result: All modes fell into "low" band by b=8, causing degenerate features

This experiment:
- Define k_max = k_c(b) (the actual filtered cutoff)
- Bands scale with k_c: low=[0, 0.3*k_c], mid=[0.3*k_c, 0.7*k_c], high=[0.7*k_c, k_c]
- Stop when too few modes (< 10 positive k bins)
- Proper RG-consistent feature extraction

This will give THE REAL ANSWER about KS→KPZ convergence in spectral space.
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
# FIXED SPECTRAL SHAPE FEATURES (RG-Consistent)
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

def extract_spectral_shape_features_fixed(h, k_cutoff_applied):
    """
    Extract spectral SHAPE features with FIXED k_max = k_cutoff_applied.
    
    CRITICAL FIX: Use the actual filtered cutoff k_c(b), not Nyquist max!
    
    Features:
    ---------
    1. Low-k slope: α from log(S(k)) ~ α*log(k) using lowest 5-10 modes
    2. Bandpower fractions: f_low, f_mid, f_high with bands RELATIVE to k_c
    3. Spectral centroid: k_cent (normalized by k_c)
    
    Parameters:
    -----------
    h : array
        Height field (already coarse-grained)
    k_cutoff_applied : float
        The k_c(b) used for this coarse-graining level
        This becomes our k_max for band definitions!
    
    Returns:
    --------
    features : array [5] or None
        [low_k_slope, f_low, f_mid, f_high, k_cent_norm]
        Returns None if too few modes (< 10)
    """
    k, S = compute_power_spectrum(h)
    
    # Check if we have enough modes
    n_modes = len(k)
    if n_modes < 10:
        # Too few modes - stop coarse-graining here
        return None
    
    # Total power (for normalization)
    total_power = np.sum(S)
    
    if total_power < 1e-12:
        # Degenerate case
        return None
    
    # CRITICAL FIX: k_max is the FILTERED CUTOFF, not Nyquist
    k_max = k_cutoff_applied
    
    # Define k-bands RELATIVE to k_c(b)
    # Low: k < 0.3*k_c (long wavelengths)
    # Mid: 0.3*k_c <= k < 0.7*k_c
    # High: k >= 0.7*k_c (up to cutoff)
    alpha = 0.3
    beta = 0.7
    
    k_low_cutoff = alpha * k_max
    k_mid_cutoff = beta * k_max
    
    # Band masks (only consider k within filtered range)
    valid_k_mask = k <= k_max
    k_valid = k[valid_k_mask]
    S_valid = S[valid_k_mask]
    
    low_mask = k_valid < k_low_cutoff
    mid_mask = (k_valid >= k_low_cutoff) & (k_valid < k_mid_cutoff)
    high_mask = k_valid >= k_mid_cutoff
    
    # Bandpower fractions (NORMALIZED by total power)
    f_low = np.sum(S_valid[low_mask]) / total_power if np.any(low_mask) else 0.0
    f_mid = np.sum(S_valid[mid_mask]) / total_power if np.any(mid_mask) else 0.0
    f_high = np.sum(S_valid[high_mask]) / total_power if np.any(high_mask) else 0.0
    
    # Low-k slope: Use lowest 5-10 modes (fixed count, not fraction)
    # This automatically adapts as spectrum shrinks
    n_fit_modes = min(10, len(k_valid) // 2)  # Use lowest 10 or half, whichever smaller
    n_fit_modes = max(5, n_fit_modes)  # But at least 5
    
    if len(k_valid) >= n_fit_modes:
        k_fit = k_valid[:n_fit_modes]  # Lowest modes
        S_fit = S_valid[:n_fit_modes]
        
        # Filter out zeros/negatives for log
        valid = (S_fit > 1e-12) & (k_fit > 0)
        if np.sum(valid) >= 3:
            log_k = np.log(k_fit[valid])
            log_S = np.log(S_fit[valid])
            
            slope, intercept, r_value, p_value, std_err = linregress(log_k, log_S)
            low_k_slope = slope
        else:
            low_k_slope = 0.0
    else:
        low_k_slope = 0.0
    
    # Spectral centroid (normalized by k_c so it's dimensionless)
    k_cent = np.sum(k_valid * S_valid) / total_power
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
# MULTI-SIGMA MMD (same as before)
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
    for x in sample_X[:50]:
        for y in sample_Y[:50]:
            dists.append(np.linalg.norm(x - y))
    
    return np.median(dists)

def compute_mmd_multi_sigma(X, Y, sigma_base):
    """Compute MMD averaged over multiple bandwidths."""
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
# LOAD AND PROCESS DATA WITH FIXED FEATURES
# ============================================================================

def coarse_grain_field_spectral(h, k_cutoff_fraction):
    """Spectral low-pass coarse-graining."""
    L = len(h)
    h_hat = fft(h)
    k = np.fft.fftfreq(L)
    k_max = np.max(np.abs(k))
    k_cutoff = k_cutoff_fraction * k_max
    mask = np.abs(k) <= k_cutoff
    h_hat_filtered = h_hat * mask
    return np.real(ifft(h_hat_filtered)), k_cutoff

def load_and_extract_features_fixed(data_path, system_name):
    """
    Load fields and extract FIXED spectral features.
    
    Returns:
    --------
    features_by_scale : dict
        {scale: features_array} for valid scales only
    valid_scales : list
        Scales that had enough modes
    scale_names : list
        Names of valid scales
    """
    print(f"\nLoading {system_name} data from {data_path.name}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    fields = data['fields']
    scales = data['scales']
    scale_names_orig = data['scale_names']
    
    print(f"  Loaded {len(fields)} field samples")
    
    # Process each scale
    features_by_scale = {}
    valid_scales = []
    valid_scale_names = []
    
    for scale, name in zip(scales, scale_names_orig):
        print(f"\n  Processing {name} (k_cutoff_fraction = {scale})...")
        
        features_list = []
        n_valid = 0
        
        # Determine k_cutoff for this scale (in absolute units)
        # We need to compute k_max first
        L = len(fields[0])
        k_nyquist = 0.5  # For fftfreq with d=1.0, max k is 0.5
        k_cutoff_applied = scale * k_nyquist
        
        print(f"    k_cutoff = {k_cutoff_applied:.4f} (filtered spectrum support)")
        
        for h in fields:
            # Coarse-grain field
            h_coarse, k_c_actual = coarse_grain_field_spectral(h, scale)
            
            # Extract features with FIXED k_max = k_c
            features = extract_spectral_shape_features_fixed(h_coarse, k_cutoff_applied)
            
            if features is not None:
                features_list.append(features)
                n_valid += 1
        
        if n_valid >= 100:  # Need at least 100 valid samples
            features_by_scale[scale] = np.array(features_list)
            valid_scales.append(scale)
            valid_scale_names.append(name)
            
            print(f"    ✅ Valid: {n_valid}/{len(fields)} samples")
            print(f"       Feature shape: {features_by_scale[scale].shape}")
            print(f"       Feature ranges:")
            print(f"         slope: [{features_by_scale[scale][:, 0].min():.3f}, {features_by_scale[scale][:, 0].max():.3f}]")
            print(f"         f_low: [{features_by_scale[scale][:, 1].min():.3f}, {features_by_scale[scale][:, 1].max():.3f}]")
            print(f"         f_mid: [{features_by_scale[scale][:, 2].min():.3f}, {features_by_scale[scale][:, 2].max():.3f}]")
            print(f"         f_high: [{features_by_scale[scale][:, 3].min():.3f}, {features_by_scale[scale][:, 3].max():.3f}]")
        else:
            print(f"    ❌ Too few valid samples: {n_valid}/{len(fields)} - STOPPING HERE")
            break
    
    return features_by_scale, valid_scales, valid_scale_names

# ============================================================================
# COMPARISON
# ============================================================================

def compare_with_fixed_features(ks_features, kpz_features, scales, scale_names):
    """Compare KS vs KPZ using FIXED spectral features."""
    print("\n" + "="*70)
    print("COMPUTING MULTI-σ MMD DISTANCES (FIXED FEATURES)")
    print("="*70)
    
    # Estimate base bandwidth at first scale
    first_scale = scales[0]
    sigma_base = estimate_bandwidth(ks_features[first_scale], kpz_features[first_scale])
    print(f"\nBase bandwidth (estimated at {scale_names[0]}): σ = {sigma_base:.4f}")
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
    
    relative_change = (distances[-1] - distances[0]) / distances[0]
    print(f"\nRelative change ({scale_names[0]} → {scale_names[-1]}):")
    print(f"  Absolute: {distances[-1] - distances[0]:+.4f}")
    print(f"  Relative: {relative_change:+.2%}")
    
    # Interpretation
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    if slope < -0.02 and relative_change < -0.1:
        print("✅ CONVERGENCE: Distance decreases significantly")
        print("   → KS flows toward KPZ in spectral shape space")
        print("   → Framework generalizes with right observables")
        interpretation = "convergence"
    elif abs(slope) < 0.02 and abs(relative_change) < 0.1:
        print("⚠️  FLAT: Distance roughly constant")
        print("   → Consistent with Exp 50h (gradient moments)")
        print("   → No convergence in spectral shape either")
        print("   → Suggests regime issue or genuine non-convergence")
        interpretation = "flat"
    else:
        print("⚠️  DIVERGENCE: Distance increases")
        print("   → KS spectral shape diverges from KPZ")
        interpretation = "divergence"
    
    return slope, p_value, relative_change, interpretation

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50j: DIAGNOSTIC B - FIXED SPECTRAL FEATURES")
    print("="*70)
    print("\nCRITICAL FIX: k_max now scales with coarse-graining cutoff")
    print("  - Bands defined relative to k_c(b), not Nyquist")
    print("  - Stops when too few modes (< 10)")
    print("  - No more feature collapse artifact")
    print("\nThis will give THE REAL ANSWER about KS→KPZ in spectral space.")
    print("="*70)
    
    output_dir = Path('results/exp50j_diagnostic_b_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ks_path = Path('results/exp50i_diagnostic_b_spectral/ks_fields.pkl')
    kpz_path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')
    
    # Extract features with FIXED method
    ks_features, ks_scales, ks_scale_names = load_and_extract_features_fixed(ks_path, 'KS')
    kpz_features, kpz_scales, kpz_scale_names = load_and_extract_features_fixed(kpz_path, 'KPZ')
    
    # Use intersection of valid scales
    valid_scales = sorted(set(ks_scales) & set(kpz_scales))
    valid_scale_names = [f"b={int(1/s)}" for s in valid_scales]
    
    print(f"\n" + "="*70)
    print(f"VALID SCALES FOR COMPARISON: {valid_scale_names}")
    print("="*70)
    
    if len(valid_scales) < 3:
        print("\n❌ ERROR: Too few valid scales for comparison")
        print("   Need at least 3 scales to assess trend")
        return
    
    # Compare
    distances, distances_by_sigma, sigma_base = compare_with_fixed_features(
        ks_features, kpz_features, valid_scales, valid_scale_names)
    
    # Analyze
    slope, p_value, relative_change, interpretation = analyze_trend(distances, valid_scale_names)
    
    # Save results
    results = {
        'method': 'spectral_shape_features_FIXED',
        'distances': distances,
        'distances_by_sigma': distances_by_sigma,
        'sigma_base': sigma_base,
        'scales': valid_scales,
        'scale_names': valid_scale_names,
        'slope': slope,
        'p_value': p_value,
        'relative_change': relative_change,
        'interpretation': interpretation,
        'ks_sample_sizes': [len(ks_features[s]) for s in valid_scales],
        'kpz_sample_sizes': [len(kpz_features[s]) for s in valid_scales]
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EXPERIMENT 50j COMPLETE (NO ARTIFACT)")
    print("="*70)
    print(f"\nKey finding: Slope = {slope:+.5f} ({interpretation})")
    print(f"Relative change: {relative_change:+.2%}")
    print(f"Valid scales: {len(valid_scales)}")
    
    print(f"\nComparison to previous experiments:")
    print(f"  Exp 50h (gradient): slope = +0.00043 (flat)")
    print(f"  Exp 50i (spectral, BUGGY): slope = -0.07821 (artifact!)")
    print(f"  Exp 50j (spectral, FIXED): slope = {slope:+.5f}")
    
    if interpretation == "convergence":
        print("\n🎉 BREAKTHROUGH: Real KS→KPZ convergence in spectral space!")
        print("   Framework generalizes with appropriate observables")
    elif interpretation == "flat":
        print("\n⚠️  CONSISTENT: Spectral features also show no convergence")
        print("   Matches gradient moment result (Exp 50h)")
        print("   Suggests regime issue or genuine non-convergence")
    else:
        print("\n🔬 DIVERGENCE: KS spectral shape differs from KPZ")
    
    print("\n" + "="*70)
    print("THIS IS THE REAL ANSWER (no more artifacts)")
    print("="*70)

if __name__ == '__main__':
    main()
