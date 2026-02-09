"""
Experiment 50k: Diagnostic B - CLEAN Spectral Features

CRITICAL FIXES from 50j analysis:
1. Mean-subtract h before FFT (remove DC offset)
2. Drop k=0 from all computations
3. Replace 3-band fractions with log-binned spectrum (8 bins)
4. Slope uses fixed mode count (k=1..10)
5. Correct scale ordering for regression

This version should have NO degeneracy for KS at any scale.
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
# CLEAN SPECTRAL SHAPE FEATURES (No Degeneracy)
# ============================================================================

def compute_power_spectrum_clean(h):
    """
    Compute power spectrum S(k) with proper preprocessing.
    
    CRITICAL:
    - Mean-subtract h to remove DC offset
    - Drop k=0 from output (no DC in spectrum)
    
    Returns:
    --------
    k : array
        Wavenumbers (positive, excluding k=0)
    S : array
        Power spectrum (positive k only, no k=0)
    """
    L = len(h)
    
    # Mean-subtract (remove DC)
    h_centered = h - np.mean(h)
    
    # FFT
    h_fft = fft(h_centered)
    
    # Power spectrum
    S = np.abs(h_fft)**2 / L
    
    # Get wavenumbers
    k = fftfreq(L, d=1.0)
    
    # Keep only positive k, EXCLUDING k=0
    positive_k_mask = k > 0
    k_pos = k[positive_k_mask]
    S_pos = S[positive_k_mask]
    
    return k_pos, S_pos

def extract_spectral_shape_features_clean(h, k_cutoff_applied, min_total_modes=20, min_low_modes=10):
    """
    Extract spectral features with NO degeneracy.
    
    Features:
    ---------
    1. Low-k slope: α from log(S) ~ α*log(k), using k=1..10 (fixed count)
    2. Log-binned spectrum: 8 bins, logarithmically spaced in k
    3. Spectral centroid: k_cent (normalized by k_c)
    
    Total: 1 + 8 + 1 = 10 features
    
    Parameters:
    -----------
    h : array
        Height field (already coarse-grained)
    k_cutoff_applied : float
        The k_c(b) used for this coarse-graining level
    
    Returns:
    --------
    features : array [10] or None
        [low_k_slope, bin1, bin2, ..., bin8, k_cent_norm]
        Returns None if too few modes (< 20)
    """
    k, S = compute_power_spectrum_clean(h)
    
    # Check if we have enough modes
    n_modes = len(k)
    if n_modes < min_total_modes:
        # Too few modes - stop coarse-graining here
        return None
    
    # Total power (for normalization)
    total_power = np.sum(S)
    
    if total_power < 1e-12:
        # Degenerate case
        return None
    
    # Only use modes within filtered cutoff
    valid_k_mask = k <= k_cutoff_applied
    k_valid = k[valid_k_mask]
    S_valid = S[valid_k_mask]
    
    if len(k_valid) < min_total_modes:
        return None
    
    # ---- Feature 1: Low-k slope (fixed mode count) ----
    # Use lowest fixed number of modes (k=1..min_low_modes)
    if len(k_valid) < min_low_modes:
        # Too few surviving modes for a fixed-count slope
        return None
    
    k_fit = k_valid[:min_low_modes]
    S_fit = S_valid[:min_low_modes]
    
    # Log-log fit
    valid = (S_fit > 1e-12) & (k_fit > 0)
    if np.sum(valid) >= 3:
        log_k = np.log(k_fit[valid])
        log_S = np.log(S_fit[valid])
        slope, _, _, _, _ = linregress(log_k, log_S)
        low_k_slope = slope
    else:
        return None
    
    # ---- Feature 2-9: Log-binned spectrum (8 bins) ----
    # Create 8 logarithmic bins from k_min to k_max
    k_min = k_valid[0]
    k_max = k_cutoff_applied
    
    # Log-space bin edges
    n_bins = 8
    log_edges = np.linspace(np.log(k_min), np.log(k_max), n_bins + 1)
    bin_edges = np.exp(log_edges)
    
    # Compute power in each bin
    bin_powers = []
    for i in range(n_bins):
        bin_mask = (k_valid >= bin_edges[i]) & (k_valid < bin_edges[i+1])
        if i == n_bins - 1:
            # Last bin includes upper edge
            bin_mask = (k_valid >= bin_edges[i]) & (k_valid <= bin_edges[i+1])
        
        bin_power = np.sum(S_valid[bin_mask])
        bin_powers.append(bin_power)
    
    # Normalize to sum to 1 (spectral shape)
    bin_powers = np.array(bin_powers)
    total_power_valid = np.sum(S_valid)
    if total_power_valid < 1e-12:
        return None
    
    bin_powers_norm = bin_powers / (total_power_valid + 1e-12)
    
    # ---- Feature 10: Spectral centroid (normalized) ----
    k_cent = np.sum(k_valid * S_valid) / (total_power_valid + 1e-12)
    k_cent_norm = k_cent / k_max
    
    # Assemble features
    features = np.concatenate([
        [low_k_slope],
        bin_powers_norm,
        [k_cent_norm]
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
# LOAD AND PROCESS DATA WITH CLEAN FEATURES
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

def _features_non_degenerate(features, min_std=1e-6):
    """
    Check for degenerate features (near-zero variance).
    Returns True if all feature dimensions have std >= min_std.
    
    NOTE: Bins can legitimately have zero power (e.g., first bin at small k).
    We mainly want to catch collapse where ALL samples have identical feature vectors.
    """
    if features.size == 0:
        return False
    stds = np.std(features, axis=0)
    
    # Allow some dimensions to have low std (legitimate zero bins)
    # But if MORE THAN HALF have degenerate variance, that's an artifact
    n_degenerate = np.sum(stds < min_std)
    n_total = len(stds)
    
    if n_degenerate > n_total // 2:
        print(f"      WARNING: {n_degenerate}/{n_total} features have near-zero variance")
        print(f"      Feature stds: {stds}")
        return False
    
    return True

def load_and_extract_features_clean(data_path, system_name):
    """
    Load fields and extract CLEAN spectral features.
    
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
        
        # Determine k_cutoff for this scale (from actual filter)
        L = len(fields[0])
        k_nyquist = 0.5
        k_cutoff_applied = scale * k_nyquist
        
        print(f"    k_cutoff = {k_cutoff_applied:.4f}")
        
        for h in fields:
            # Coarse-grain field
            h_coarse, k_c_actual = coarse_grain_field_spectral(h, scale)
            
            # Extract features with CLEAN method
            features = extract_spectral_shape_features_clean(h_coarse, k_c_actual)
            
            if features is not None:
                features_list.append(features)
                n_valid += 1
        
        if n_valid >= 100:
            features_array = np.array(features_list)
            
            # Degeneracy check (must be non-degenerate across features)
            if not _features_non_degenerate(features_array):
                print("    ❌ Degenerate features detected (near-zero variance) - STOPPING")
                break
            
            features_by_scale[scale] = features_array
            valid_scales.append(scale)
            valid_scale_names.append(name)
            
            print(f"    ✅ Valid: {n_valid}/{len(fields)} samples")
            print(f"       Feature shape: {features_by_scale[scale].shape}")
            
            # Show first 3 features (slope + first 2 bins) to check non-degeneracy
            feat = features_by_scale[scale]
            print(f"       slope: [{feat[:, 0].min():.3f}, {feat[:, 0].max():.3f}]")
            print(f"       bin1:  [{feat[:, 1].min():.4f}, {feat[:, 1].max():.4f}]")
            print(f"       bin2:  [{feat[:, 2].min():.4f}, {feat[:, 2].max():.4f}]")
            print(f"       bin8:  [{feat[:, 8].min():.4f}, {feat[:, 8].max():.4f}]")
        else:
            print(f"    ❌ Too few valid samples: {n_valid}/{len(fields)} - STOPPING")
            break
    
    return features_by_scale, valid_scales, valid_scale_names

# ============================================================================
# COMPARISON
# ============================================================================

def compare_with_clean_features(ks_features, kpz_features, scales, scale_names):
    """Compare KS vs KPZ using CLEAN spectral features."""
    print("\n" + "="*70)
    print("COMPUTING MULTI-σ MMD DISTANCES (CLEAN FEATURES)")
    print("="*70)
    
    # Estimate base bandwidth at first scale
    first_scale = scales[0]
    sigma_base = estimate_bandwidth(ks_features[first_scale], kpz_features[first_scale])
    print(f"\nBase bandwidth (estimated at {scale_names[0]}): σ = {sigma_base:.4f}")
    
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
    
    return np.array(distances), distances_by_sigma, sigma_base

def analyze_trend_corrected(distances, scale_names):
    """
    Analyze trend with CORRECTED ordering.
    
    Distances are ordered from SMALL b to LARGE b (b=1,2,4,8,16).
    Positive slope means distance INCREASES with coarse-graining (b↑) = DIVERGENCE.
    Negative slope means distance DECREASES with coarse-graining (b↑) = CONVERGENCE.
    """
    print("\n" + "="*70)
    print("TREND ANALYSIS (CORRECTED ORDERING)")
    print("="*70)
    
    # x axis: b value (increasing with coarse-graining)
    # Parse b from scale_names like "b=1", "b=2", etc.
    b_values = []
    for name in scale_names:
        b_str = name.split('=')[1]
        b_values.append(int(b_str))
    
    x = np.array(b_values)
    
    # Regression: distance vs b
    slope, intercept, r_value, p_value, std_err = linregress(x, distances)
    
    print(f"\nLinear regression (distance vs b):")
    print(f"  Slope: {slope:+.6f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    
    relative_change = (distances[-1] - distances[0]) / distances[0]
    print(f"\nRelative change ({scale_names[0]} → {scale_names[-1]}):")
    print(f"  Absolute: {distances[-1] - distances[0]:+.4f}")
    print(f"  Relative: {relative_change:+.2%}")
    
    # Interpretation (CORRECTED)
    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    if slope < -0.01 and relative_change < -0.1:
        print("✅ CONVERGENCE: Distance decreases with coarse-graining (b↑)")
        print("   → KS flows toward KPZ in spectral shape space")
        print("   → Framework generalizes with appropriate observables")
        interpretation = "convergence"
    elif abs(slope) < 0.01 and abs(relative_change) < 0.1:
        print("⚠️  FLAT: Distance roughly constant")
        print("   → Consistent with Exp 50h (gradient moments)")
        print("   → No convergence in spectral shape")
        interpretation = "flat"
    else:
        print("⚠️  DIVERGENCE: Distance increases with coarse-graining (b↑)")
        print("   → KS spectral shape diverges from KPZ")
        interpretation = "divergence"
    
    return slope, p_value, relative_change, interpretation

# ============================================================================
# SANITY CHECK: KPZ vs KPZ
# ============================================================================

def sanity_check_kpz_vs_kpz(kpz_features, scales, scale_names):
    """
    Sanity check: Measure distance within KPZ at different scales.
    Should be roughly constant (no artificial convergence/divergence).
    """
    print("\n" + "="*70)
    print("SANITY CHECK: KPZ vs KPZ")
    print("="*70)
    
    # Split KPZ into two halves
    first_scale = scales[0]
    kpz_all = kpz_features[first_scale]
    n = len(kpz_all)
    
    kpz_half1 = {}
    kpz_half2 = {}
    
    for scale in scales:
        kpz_all = kpz_features[scale]
        n = len(kpz_all)
        kpz_half1[scale] = kpz_all[:n//2]
        kpz_half2[scale] = kpz_all[n//2:]
    
    # Estimate bandwidth
    sigma_base = estimate_bandwidth(kpz_half1[first_scale], kpz_half2[first_scale])
    
    distances = []
    print(f"\nBase bandwidth: σ = {sigma_base:.4f}")
    print("\nScale-by-scale comparison (KPZ half1 vs half2):")
    
    for scale, name in zip(scales, scale_names):
        X = kpz_half1[scale]
        Y = kpz_half2[scale]
        
        d_mean, _ = compute_mmd_multi_sigma(X, Y, sigma_base)
        distances.append(d_mean)
        
        print(f"  {name}: d_mean = {d_mean:.4f}")
    
    # Check for artificial trend
    b_values = [int(name.split('=')[1]) for name in scale_names]
    slope, _, _, _, _ = linregress(b_values, distances)
    
    print(f"\nSlope: {slope:+.6f}")
    if abs(slope) < 0.005:
        print("✅ PASS: KPZ vs KPZ is flat (no feature artifact)")
    else:
        print("⚠️  WARNING: KPZ vs KPZ shows trend - possible feature artifact")
    
    return distances, slope

# ============================================================================
# SANITY CHECK: KS vs KS
# ============================================================================

def sanity_check_ks_vs_ks(ks_features, scales, scale_names):
    """
    Sanity check: Measure distance within KS at different scales.
    Should be roughly constant (no artificial convergence/divergence).
    """
    print("\n" + "="*70)
    print("SANITY CHECK: KS vs KS")
    print("="*70)
    
    # Split KS into two halves
    first_scale = scales[0]
    ks_all = ks_features[first_scale]
    n = len(ks_all)
    
    ks_half1 = {}
    ks_half2 = {}
    
    for scale in scales:
        ks_all = ks_features[scale]
        n = len(ks_all)
        ks_half1[scale] = ks_all[:n//2]
        ks_half2[scale] = ks_all[n//2:]
    
    # Estimate bandwidth
    sigma_base = estimate_bandwidth(ks_half1[first_scale], ks_half2[first_scale])
    
    distances = []
    print(f"\nBase bandwidth: σ = {sigma_base:.4f}")
    print("\nScale-by-scale comparison (KS half1 vs half2):")
    
    for scale, name in zip(scales, scale_names):
        X = ks_half1[scale]
        Y = ks_half2[scale]
        
        d_mean, _ = compute_mmd_multi_sigma(X, Y, sigma_base)
        distances.append(d_mean)
        
        print(f"  {name}: d_mean = {d_mean:.4f}")
    
    # Check for artificial trend
    b_values = [int(name.split('=')[1]) for name in scale_names]
    slope, _, _, _, _ = linregress(b_values, distances)
    
    print(f"\nSlope: {slope:+.6f}")
    if abs(slope) < 0.005:
        print("✅ PASS: KS vs KS is flat (no feature artifact)")
    else:
        print("⚠️  WARNING: KS vs KS shows trend - possible feature artifact")
    
    return distances, slope

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50k: DIAGNOSTIC B - CLEAN SPECTRAL FEATURES")
    print("="*70)
    print("\nFIXES from 50j:")
    print("  - Mean-subtract h before FFT (remove DC)")
    print("  - Drop k=0 from all computations")
    print("  - 8 log-binned spectrum components (not 3-band fractions)")
    print("  - Low-k slope uses fixed mode count (k=1..10)")
    print("  - Corrected scale ordering in regression")
    print("\nThis should have NO degeneracy for KS at any scale.")
    print("="*70)
    
    output_dir = Path('results/exp50k_diagnostic_b_clean')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    ks_path = Path('results/exp50i_diagnostic_b_spectral/ks_fields.pkl')
    kpz_path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')
    
    # Extract features with CLEAN method
    ks_features, ks_scales, ks_scale_names = load_and_extract_features_clean(ks_path, 'KS')
    kpz_features, kpz_scales, kpz_scale_names = load_and_extract_features_clean(kpz_path, 'KPZ')
    
    # Use intersection of valid scales
    valid_scales = sorted(set(ks_scales) & set(kpz_scales))
    valid_scale_names = [f"b={int(1/s)}" for s in valid_scales]
    
    print(f"\n" + "="*70)
    print(f"VALID SCALES FOR COMPARISON: {valid_scale_names}")
    print("="*70)
    
    if len(valid_scales) < 3:
        print("\n❌ ERROR: Too few valid scales")
        return
    
    # Compare KS vs KPZ
    distances_ks_kpz, distances_by_sigma, sigma_base = compare_with_clean_features(
        ks_features, kpz_features, valid_scales, valid_scale_names)
    
    # Analyze trend (CORRECTED)
    slope, p_value, relative_change, interpretation = analyze_trend_corrected(
        distances_ks_kpz, valid_scale_names)
    
    # Sanity check: KPZ vs KPZ
    distances_kpz_kpz, slope_kpz = sanity_check_kpz_vs_kpz(
        kpz_features, valid_scales, valid_scale_names)
    
    # Sanity check: KS vs KS
    distances_ks_ks, slope_ks = sanity_check_ks_vs_ks(
        ks_features, valid_scales, valid_scale_names)
    
    # DIAGNOSTIC GATE: Check if invariance tests passed
    print("\n" + "="*70)
    print("DIAGNOSTIC GATE: INVARIANCE TEST EVALUATION")
    print("="*70)
    
    kpz_pass = abs(slope_kpz) < 0.005
    ks_pass = abs(slope_ks) < 0.005
    
    print(f"\nKPZ vs KPZ: {'✅ PASS' if kpz_pass else '❌ FAIL'} (slope = {slope_kpz:+.6f})")
    print(f"KS vs KS:   {'✅ PASS' if ks_pass else '❌ FAIL'} (slope = {slope_ks:+.6f})")
    
    diagnostic_passed = kpz_pass and ks_pass
    
    if not diagnostic_passed:
        print("\n⚠️  DIAGNOSTIC FAILURE: Invariance tests did not pass")
        print("   KS vs KPZ comparison is NOT VALID until feature artifacts are fixed")
        print("   The trend analysis below is UNRELIABLE.")
    else:
        print("\n✅ DIAGNOSTIC PASSED: Invariance tests confirm feature validity")
        print("   KS vs KPZ comparison is scientifically meaningful.")
    
    print("="*70)
    
    # Save results
    results = {
        'method': 'spectral_shape_features_CLEAN',
        'distances_ks_kpz': distances_ks_kpz,
        'distances_kpz_kpz': distances_kpz_kpz,
        'distances_ks_ks': distances_ks_ks,
        'distances_by_sigma': distances_by_sigma,
        'sigma_base': sigma_base,
        'scales': valid_scales,
        'scale_names': valid_scale_names,
        'slope': slope,
        'slope_kpz_kpz': slope_kpz,
        'slope_ks_ks': slope_ks,
        'p_value': p_value,
        'relative_change': relative_change,
        'interpretation': interpretation,
        'diagnostic_passed': diagnostic_passed,
        'kpz_invariance_pass': kpz_pass,
        'ks_invariance_pass': ks_pass,
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ EXPERIMENT 50k COMPLETE")
    print("="*70)
    print(f"\nKS vs KPZ:")
    print(f"  Slope: {slope:+.6f} ({interpretation})")
    print(f"  Relative change: {relative_change:+.2%}")
    print(f"  Diagnostic status: {'✅ VALID' if results['diagnostic_passed'] else '❌ INVALID'}")
    print(f"\nKPZ vs KPZ sanity check:")
    print(f"  Slope: {slope_kpz:+.6f} ({'PASS' if results['kpz_invariance_pass'] else 'FAIL'})")
    print(f"\nKS vs KS sanity check:")
    print(f"  Slope: {slope_ks:+.6f} ({'PASS' if results['ks_invariance_pass'] else 'FAIL'})")
    
    print(f"\nComparison to previous experiments:")
    print(f"  Exp 50h (gradient): slope = +0.00043 (flat)")
    print(f"  Exp 50j (spectral, 3-band): apparent convergence BUT degeneracy at small b")
    print(f"  Exp 50k (spectral, log-bins): slope = {slope:+.6f}")
    
    if not results['diagnostic_passed']:
        print("\n❌ DIAGNOSTIC FAILED: Results are UNRELIABLE")
        print("   Feature artifacts prevent valid interpretation")
    elif interpretation == "convergence":
        print("\n🎉 CONVERGENCE CONFIRMED: Real KS→KPZ flow in spectral space")
        print("   Framework generalizes with appropriate observables")
    elif interpretation == "flat":
        print("\n⚠️  CONSISTENT: No convergence in spectral shape")
        print("   Matches gradient moment result")
    else:
        print("\n🔬 DIVERGENCE: KS spectral shape differs from KPZ")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
