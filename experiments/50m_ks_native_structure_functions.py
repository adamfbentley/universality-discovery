"""
Exp 50m: KS-Native Observables - Structure Functions
=====================================================

Goal: Test if KS-native observables (structure functions) reveal different
      behavior than spectral shape features.

Structure Functions: S_p(r) = ⟨|h(x+r) - h(x)|^p⟩
- Natural for rough interfaces (no DC offset issues)
- Multi-scale by design (vary lag distance r)
- Used in KS/turbulence literature

Features extracted:
1. S_2 slope: α_2 from log(S_2) ~ α_2 * log(r) 
2. S_4 slope: α_4 from log(S_4) ~ α_4 * log(r)
3. Normalized structure function values at fixed r/L ratios
4. Structure function ratio: S_4 / S_2^2 (kurtosis-like)

Diagnostic Gate: Both KS-vs-KS AND KPZ-vs-KPZ must be flat (|slope| < 0.005)

Status: Tests KS generalization with native observables (Exp 50 sequence)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.fft import fft, ifft, fftfreq
import time

# ========== Configuration ==========
L = 256          # System size
dt = 0.01        # Time step
n_steps = 50000  # Simulation steps
n_save = 500     # Number of snapshots to save

# KS parameters (baseline regime from 50l)
nu_ks = 1.0      # Dispersion coefficient
kappa_ks = 1.0   # Anti-diffusion coefficient  
lambda_ks = 1.0  # Nonlinearity coefficient
noise_strength_ks = 0.1

# KPZ parameters (matched from earlier experiments)
nu_kpz = 1.0
lambda_kpz = 1.0
noise_strength_kpz = 0.1

# Coarse-graining scales
coarse_grain_factors = [1, 2, 4]  # b=1,2,4 (conservative, per 50k experience)

# MMD parameters
n_mmd_samples = 250  # Per system (total 500 for MMD)
sigma_base = None    # Will be set adaptively at b=1
sigma_multiples = [0.5, 1.0, 2.0, 4.0]  # Multi-sigma MMD

# Structure function parameters
r_fractions = [1/32, 1/16, 1/8, 1/4, 1/2]  # Lag distances as fractions of L
min_r_modes = 3  # Minimum number of lag points for slope fitting (reduced from 5)

# Output
output_dir = Path("results_exp50m")
output_dir.mkdir(exist_ok=True)

# ========== KS Simulation ==========
def ks_step_spectral(h_hat, k2, k4, k, dt, nu, kappa, lambda_val):
    """Single KS time step in Fourier space."""
    # Linear terms: -ν k^4 + κ k^2
    linear_term = (-nu * k4 + kappa * k2) * h_hat
    
    # Nonlinear term: -λ/2 (∇h)^2
    # ∇h in real space
    h = ifft(h_hat).real
    grad_h = ifft(1j * k * fft(h)).real
    nonlinear = -0.5 * lambda_val * grad_h**2
    nonlinear_hat = fft(nonlinear)
    
    # Euler step with dealiasing
    h_hat_new = h_hat + dt * (linear_term + nonlinear_hat)
    return h_hat_new

def simulate_ks(L, dt, n_steps, nu, kappa, lambda_val, noise_strength, n_save):
    """Simulate KS equation."""
    x = np.linspace(0, L, L, endpoint=False)
    dx = x[1] - x[0]
    
    k = 2 * np.pi * fftfreq(L, dx)
    k2 = k**2
    k4 = k**4
    
    # Initial condition: small random perturbation
    h = 0.1 * np.random.randn(L)
    h_hat = fft(h)
    
    # Storage
    save_interval = max(1, n_steps // n_save)
    h_snapshots = []
    
    # Time integration
    for step in range(n_steps):
        # Stochastic forcing
        noise = noise_strength * np.random.randn(L)
        h_hat += dt * fft(noise)
        
        # Deterministic step
        h_hat = ks_step_spectral(h_hat, k2, k4, k, dt, nu, kappa, lambda_val)
        
        # Save snapshot
        if step % save_interval == 0:
            h = ifft(h_hat).real
            h_snapshots.append(h.copy())
    
    return np.array(h_snapshots)

# ========== KPZ Simulation ==========
def kpz_step_spectral(h_hat, k2, k, dt, nu, lambda_val):
    """Single KPZ time step in Fourier space."""
    # Linear term: ν ∇²h
    linear_term = -nu * k2 * h_hat
    
    # Nonlinear term: λ/2 (∇h)²
    h = ifft(h_hat).real
    grad_h = ifft(1j * k * fft(h)).real
    nonlinear = 0.5 * lambda_val * grad_h**2
    nonlinear_hat = fft(nonlinear)
    
    # Euler step
    h_hat_new = h_hat + dt * (linear_term + nonlinear_hat)
    return h_hat_new

def simulate_kpz(L, dt, n_steps, nu, lambda_val, noise_strength, n_save):
    """Simulate KPZ equation."""
    x = np.linspace(0, L, L, endpoint=False)
    dx = x[1] - x[0]
    
    k = 2 * np.pi * fftfreq(L, dx)
    k2 = k**2
    
    h = 0.1 * np.random.randn(L)
    h_hat = fft(h)
    
    save_interval = max(1, n_steps // n_save)
    h_snapshots = []
    
    for step in range(n_steps):
        noise = noise_strength * np.random.randn(L)
        h_hat += dt * fft(noise)
        
        h_hat = kpz_step_spectral(h_hat, k2, k, dt, nu, lambda_val)
        
        if step % save_interval == 0:
            h = ifft(h_hat).real
            h_snapshots.append(h.copy())
    
    return np.array(h_snapshots)

# ========== Field-Level Coarse-Graining ==========
def coarse_grain_field_spectral(h, b):
    """
    Coarse-grain field by spectral low-pass filtering.
    Returns coarse-grained field and actual cutoff used.
    """
    L = len(h)
    k = fftfreq(L, 1.0)
    k_max = np.max(np.abs(k))
    k_cutoff = k_max / b
    
    h_hat = fft(h)
    mask = np.abs(k) <= k_cutoff
    h_hat_coarse = h_hat * mask
    h_coarse = ifft(h_hat_coarse).real
    
    k_c_actual = k_cutoff
    return h_coarse, k_c_actual

# ========== Structure Function Features ==========
def compute_structure_functions(h, r_fractions, L):
    """
    Compute structure functions S_p(r) = ⟨|h(x+r) - h(x)|^p⟩
    
    Returns:
        features: dict with slopes and normalized values
        diagnostic: dict with raw S_p arrays for validation
    """
    n_points = len(h)
    
    # Lag distances (in index units)
    r_lags = [max(1, int(r_frac * n_points)) for r_frac in r_fractions]
    r_lags = [r for r in r_lags if r < n_points // 2]  # Valid lags
    
    if len(r_lags) == 0:
        return None, {'valid': False, 'reason': 'no_valid_lags', 'n_points': n_points}
    
    # Compute S_2 and S_4
    S_2 = []
    S_4 = []
    
    for r in r_lags:
        dh = h[r:] - h[:-r]  # Finite differences
        S_2.append(np.mean(dh**2))
        S_4.append(np.mean(dh**4))
    
    S_2 = np.array(S_2)
    S_4 = np.array(S_4)
    r_array = np.array(r_lags) / n_points  # Normalize by field length
    
    # Feature extraction
    features = {}
    
    # 1. Log-log slopes (power-law exponents)
    if len(r_lags) >= min_r_modes:
        log_r = np.log(r_array)
        log_S2 = np.log(S_2 + 1e-12)  # Avoid log(0)
        log_S4 = np.log(S_4 + 1e-12)
        
        # Fit slopes
        slope_S2, intercept_S2, _, _, _ = stats.linregress(log_r, log_S2)
        slope_S4, intercept_S4, _, _, _ = stats.linregress(log_r, log_S4)
        
        features['slope_S2'] = slope_S2
        features['slope_S4'] = slope_S4
    else:
        return None, {'valid': False, 'reason': 'insufficient_r_modes', 'n_lags': len(r_lags), 'min_required': min_r_modes}
    
    # 2. Normalized structure function values (at r/L = 1/8, 1/4)
    idx_eighth = np.argmin(np.abs(r_array - 1/8))
    idx_quarter = np.argmin(np.abs(r_array - 1/4))
    
    total_S2 = np.sum(S_2)
    total_S4 = np.sum(S_4)
    
    features['S2_eighth'] = S_2[idx_eighth] / (total_S2 + 1e-12)
    features['S2_quarter'] = S_2[idx_quarter] / (total_S2 + 1e-12)
    features['S4_eighth'] = S_4[idx_eighth] / (total_S4 + 1e-12)
    features['S4_quarter'] = S_4[idx_quarter] / (total_S4 + 1e-12)
    
    # 3. Kurtosis-like ratio: S_4 / S_2^2
    S4_over_S2sq = S_4 / (S_2**2 + 1e-12)
    features['kurtosis_ratio'] = np.mean(S4_over_S2sq)
    
    # Diagnostic info
    diagnostic = {
        'r_lags': r_lags,
        'r_array': r_array,
        'S_2': S_2,
        'S_4': S_4,
        'valid': True
    }
    
    return features, diagnostic

def extract_features_for_field(h, b, L):
    """Extract structure function features from a single field."""
    h_coarse, k_c = coarse_grain_field_spectral(h, b)
    features, diagnostic = compute_structure_functions(h_coarse, r_fractions, L)
    
    if features is None:
        return None
    
    # Return as array: [slope_S2, slope_S4, S2_eighth, S2_quarter, S4_eighth, S4_quarter, kurtosis_ratio]
    feature_vec = np.array([
        features['slope_S2'] if features['slope_S2'] is not None else 0.0,
        features['slope_S4'] if features['slope_S4'] is not None else 0.0,
        features['S2_eighth'],
        features['S2_quarter'],
        features['S4_eighth'],
        features['S4_quarter'],
        features['kurtosis_ratio']
    ])
    
    return feature_vec, diagnostic

# ========== MMD Distance ==========
def rbf_kernel(X, Y, sigma):
    """RBF kernel between two sets of samples."""
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    XY = X @ Y.T
    dists = XX + YY - 2 * XY
    return np.exp(-dists / (2 * sigma**2))

def mmd_distance(X, Y, sigma):
    """Maximum Mean Discrepancy with RBF kernel."""
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    m = X.shape[0]
    n = Y.shape[0]
    
    mmd = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))
    mmd += (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))
    mmd -= 2 * np.mean(K_XY)
    
    return np.sqrt(max(0, mmd))

def multi_sigma_mmd(X, Y, sigma_base, sigma_multiples):
    """Average MMD over multiple kernel bandwidths."""
    mmds = []
    for mult in sigma_multiples:
        sigma = sigma_base * mult
        mmds.append(mmd_distance(X, Y, sigma))
    return np.mean(mmds)

# ========== Diagnostic Gate ==========
def compute_mmd_vs_scale(features_dict_a, features_dict_b, b_values, sigma_base):
    """
    Compute MMD distance vs coarse-graining scale.
    Returns distances and linear fit slope.
    """
    distances = []
    valid_b = []
    
    for b in b_values:
        if b not in features_dict_a or b not in features_dict_b:
            continue
        
        X = features_dict_a[b]
        Y = features_dict_b[b]
        
        if X is None or Y is None or len(X) == 0 or len(Y) == 0:
            continue
        
        # Multi-sigma MMD
        d = multi_sigma_mmd(X, Y, sigma_base, sigma_multiples)
        distances.append(d)
        valid_b.append(b)
    
    if len(valid_b) < 2:
        return None, None, None
    
    # Linear fit: distance ~ slope * log(b)
    log_b = np.log(valid_b)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_b, distances)
    
    return np.array(distances), np.array(valid_b), slope

def diagnostic_gate(ks_features, kpz_features, b_values, sigma_base, slope_threshold=0.005):
    """
    Diagnostic gate: Both KS-vs-KS and KPZ-vs-KPZ must be flat.
    
    Returns:
        diagnostic_passed: bool
        ks_invariance_pass: bool
        kpz_invariance_pass: bool
        ks_slope: float
        kpz_slope: float
    """
    # KS-vs-KS (sample half vs half)
    n_ks = len(ks_features[b_values[0]])
    idx_split = n_ks // 2
    
    ks_half1 = {b: ks_features[b][:idx_split] for b in b_values if b in ks_features}
    ks_half2 = {b: ks_features[b][idx_split:] for b in b_values if b in ks_features}
    
    _, _, ks_slope = compute_mmd_vs_scale(ks_half1, ks_half2, b_values, sigma_base)
    
    # KPZ-vs-KPZ
    n_kpz = len(kpz_features[b_values[0]])
    idx_split = n_kpz // 2
    
    kpz_half1 = {b: kpz_features[b][:idx_split] for b in b_values if b in kpz_features}
    kpz_half2 = {b: kpz_features[b][idx_split:] for b in b_values if b in kpz_features}
    
    _, _, kpz_slope = compute_mmd_vs_scale(kpz_half1, kpz_half2, b_values, sigma_base)
    
    # Check thresholds
    ks_pass = ks_slope is not None and abs(ks_slope) < slope_threshold
    kpz_pass = kpz_slope is not None and abs(kpz_slope) < slope_threshold
    
    diagnostic_passed = ks_pass and kpz_pass
    
    return {
        'diagnostic_passed': diagnostic_passed,
        'ks_invariance_pass': ks_pass,
        'kpz_invariance_pass': kpz_pass,
        'ks_slope': ks_slope if ks_slope is not None else np.nan,
        'kpz_slope': kpz_slope if kpz_slope is not None else np.nan
    }

# ========== Main Experiment ==========
def main():
    print("=" * 70)
    print("Exp 50m: KS-Native Observables (Structure Functions)")
    print("=" * 70)
    print()
    
    # ========== Generate KS Data ==========
    print("Generating KS data...")
    print(f"  Parameters: ν={nu_ks}, κ={kappa_ks}, λ={lambda_ks}")
    print(f"  Noise: {noise_strength_ks}")
    print(f"  System size: L={L}, steps={n_steps}, snapshots={n_save}")
    
    t_start = time.time()
    ks_fields = simulate_ks(L, dt, n_steps, nu_ks, kappa_ks, lambda_ks, 
                            noise_strength_ks, n_save)
    t_elapsed = time.time() - t_start
    print(f"  ✓ Generated {len(ks_fields)} KS fields in {t_elapsed:.1f}s")
    print()
    
    # ========== Load/Generate KPZ Data ==========
    kpz_data_path = Path("data/kpz_reference_L256_1250samples.npz")
    
    if kpz_data_path.exists():
        print(f"Loading KPZ reference data from {kpz_data_path}...")
        data = np.load(kpz_data_path)
        kpz_fields = data['fields']
        print(f"  ✓ Loaded {len(kpz_fields)} KPZ fields")
    else:
        print("Generating KPZ data...")
        print(f"  Parameters: ν={nu_kpz}, λ={lambda_kpz}")
        print(f"  Noise: {noise_strength_kpz}")
        
        t_start = time.time()
        kpz_fields = simulate_kpz(L, dt, n_steps, nu_kpz, lambda_kpz,
                                  noise_strength_kpz, n_save)
        t_elapsed = time.time() - t_start
        print(f"  ✓ Generated {len(kpz_fields)} KPZ fields in {t_elapsed:.1f}s")
    
    print()
    
    # ========== Extract Features at Multiple Scales ==========
    print("Extracting structure function features at multiple scales...")
    print(f"  Coarse-graining factors: {coarse_grain_factors}")
    print(f"  Lag fractions: {r_fractions}")
    print()
    
    ks_features = {b: [] for b in coarse_grain_factors}
    kpz_features = {b: [] for b in coarse_grain_factors}
    
    # Use subset for efficiency
    ks_subset = ks_fields[:n_mmd_samples]
    kpz_subset = kpz_fields[:n_mmd_samples]
    
    # KS features
    print("  Processing KS fields...")
    n_valid_ks = {b: 0 for b in coarse_grain_factors}
    debug_printed = False
    for i, h in enumerate(ks_subset):
        for b in coarse_grain_factors:
            feats, diag = extract_features_for_field(h, b, L)
            if feats is not None and diag['valid']:
                ks_features[b].append(feats)
                n_valid_ks[b] += 1
            elif not debug_printed and diag is not None:
                # Print first invalid diagnostic
                print(f"\n    DEBUG - First invalid sample (b={b}):")
                print(f"      Reason: {diag.get('reason', 'unknown')}")
                print(f"      Details: {diag}")
                debug_printed = True
        
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(ks_subset)} fields processed")
    
    print(f"  ✓ Valid KS samples per scale: {n_valid_ks}")
    
    # KPZ features
    print("  Processing KPZ fields...")
    n_valid_kpz = {b: 0 for b in coarse_grain_factors}
    for i, h in enumerate(kpz_subset):
        for b in coarse_grain_factors:
            feats, diag = extract_features_for_field(h, b, L)
            if feats is not None and diag['valid']:
                kpz_features[b].append(feats)
                n_valid_kpz[b] += 1
        
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(kpz_subset)} fields processed")
    
    print(f"  ✓ Valid KPZ samples per scale: {n_valid_kpz}")
    print()
    
    # Convert to arrays
    for b in coarse_grain_factors:
        ks_features[b] = np.array(ks_features[b]) if len(ks_features[b]) > 0 else None
        kpz_features[b] = np.array(kpz_features[b]) if len(kpz_features[b]) > 0 else None
    
    # Check if we have enough valid samples
    min_valid = min([n_valid_ks[b] for b in coarse_grain_factors] + 
                    [n_valid_kpz[b] for b in coarse_grain_factors])
    
    if min_valid < 10:
        print(f"❌ ERROR: Insufficient valid samples (min={min_valid})")
        print("   Cannot proceed with MMD computation.")
        return
    
    # ========== Estimate σ at b=1 ==========
    print("Estimating kernel bandwidth σ at b=1...")
    
    # Combine features for median heuristic
    X_b1 = ks_features[1]
    Y_b1 = kpz_features[1]
    
    if X_b1 is None or Y_b1 is None:
        print("❌ ERROR: No valid features at b=1")
        return
    
    combined = np.vstack([X_b1, Y_b1])
    n_samples = len(combined)
    
    # Pairwise distances (sample subset for efficiency)
    sample_size = min(1000, n_samples)
    idx = np.random.choice(n_samples, sample_size, replace=False)
    sample_data = combined[idx]
    
    dists = []
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            d = np.linalg.norm(sample_data[i] - sample_data[j])
            dists.append(d)
    
    global sigma_base
    sigma_base = np.median(dists)
    print(f"  σ = {sigma_base:.4f} (median pairwise distance at b=1)")
    print()
    
    # ========== Diagnostic Gate ==========
    print("Running diagnostic gate...")
    print("  Testing KS-vs-KS and KPZ-vs-KPZ invariance...")
    
    gate_results = diagnostic_gate(ks_features, kpz_features, coarse_grain_factors, sigma_base)
    
    print(f"  KS-vs-KS slope: {gate_results['ks_slope']:+.6f}")
    print(f"  KPZ-vs-KPZ slope: {gate_results['kpz_slope']:+.6f}")
    print(f"  KS invariance: {'✓ PASS' if gate_results['ks_invariance_pass'] else '✗ FAIL'}")
    print(f"  KPZ invariance: {'✓ PASS' if gate_results['kpz_invariance_pass'] else '✗ FAIL'}")
    print(f"  Diagnostic gate: {'✓ PASSED' if gate_results['diagnostic_passed'] else '✗ FAILED'}")
    print()
    
    # ========== Compute KS vs KPZ Distance ==========
    print("Computing KS vs KPZ distance across scales...")
    
    distances, valid_b, slope = compute_mmd_vs_scale(
        ks_features, kpz_features, coarse_grain_factors, sigma_base
    )
    
    if distances is None:
        print("❌ ERROR: Could not compute distance trend")
        return
    
    print(f"  Valid scales: b = {list(valid_b)}")
    print(f"  Distances: {distances}")
    print(f"  Slope (d/d(log b)): {slope:+.6f}")
    
    # Relative change
    rel_change = (distances[-1] - distances[0]) / distances[0] * 100
    print(f"  Relative change: {rel_change:+.2f}%")
    print()
    
    # ========== Interpretation ==========
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if not gate_results['diagnostic_passed']:
        print("⚠️  INVALID: Diagnostic gate FAILED")
        print(f"    KS-vs-KS slope: {gate_results['ks_slope']:+.6f} ({'PASS' if gate_results['ks_invariance_pass'] else 'FAIL'})")
        print(f"    KPZ-vs-KPZ slope: {gate_results['kpz_slope']:+.6f} ({'PASS' if gate_results['kpz_invariance_pass'] else 'FAIL'})")
        print()
        print("    Cannot interpret KS-vs-KPZ trend until invariance tests pass.")
        interpretation = "invalid"
    else:
        print("✅ VALID: Diagnostic gate PASSED")
        print()
        
        # Interpret slope
        if abs(slope) < 0.01:
            interpretation = "flat"
            print("INTERPRETATION: FLAT")
            print(f"  KS↔KPZ distance is constant across scales (slope={slope:+.6f})")
            print("  → No evidence of convergence with structure functions")
            print("  → Consistent with Exp 50k/50l spectral shape results")
        elif slope > 0.01:
            interpretation = "diverging"
            print("INTERPRETATION: DIVERGING")
            print(f"  KS↔KPZ distance increases with scale (slope={slope:+.6f})")
            print("  → Systems become MORE distinguishable")
            print("  → Structure functions emphasize differences")
        else:
            interpretation = "converging"
            print("INTERPRETATION: CONVERGING")
            print(f"  KS↔KPZ distance decreases with scale (slope={slope:+.6f})")
            print("  → Potential convergence behavior!")
            print("  → Structure functions may reveal hidden similarity")
    
    print()
    
    # ========== Save Results ==========
    results = {
        'distances': distances,
        'valid_b': valid_b,
        'slope': slope,
        'rel_change_percent': rel_change,
        'interpretation': interpretation,
        'diagnostic_passed': gate_results['diagnostic_passed'],
        'ks_invariance_pass': gate_results['ks_invariance_pass'],
        'kpz_invariance_pass': gate_results['kpz_invariance_pass'],
        'ks_slope': gate_results['ks_slope'],
        'kpz_slope': gate_results['kpz_slope'],
        'sigma_base': sigma_base,
        'n_valid_ks': n_valid_ks,
        'n_valid_kpz': n_valid_kpz,
        'parameters': {
            'L': L,
            'n_steps': n_steps,
            'nu_ks': nu_ks,
            'kappa_ks': kappa_ks,
            'lambda_ks': lambda_ks,
            'nu_kpz': nu_kpz,
            'lambda_kpz': lambda_kpz,
            'coarse_grain_factors': coarse_grain_factors,
            'r_fractions': r_fractions
        }
    }
    
    results_path = output_dir / "ks_vs_kpz_structure_functions.npz"
    np.savez(results_path, **results)
    print(f"Results saved to: {results_path}")
    
    # ========== Visualizations ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Distance vs scale
    ax = axes[0]
    ax.plot(valid_b, distances, 'o-', linewidth=2, markersize=8, label='KS vs KPZ')
    ax.set_xlabel('Coarse-graining factor b', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    ax.set_title(f'Distance vs Scale (slope={slope:+.5f})', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Diagnostic gate summary
    ax = axes[1]
    ax.axis('off')
    
    summary_text = f"""
Exp 50m: Structure Functions

KS-vs-KPZ Slope: {slope:+.6f}
Interpretation: {interpretation.upper()}

Diagnostic Gate:
  KS-vs-KS: {gate_results['ks_slope']:+.6f} {'✓' if gate_results['ks_invariance_pass'] else '✗'}
  KPZ-vs-KPZ: {gate_results['kpz_slope']:+.6f} {'✓' if gate_results['kpz_invariance_pass'] else '✗'}
  Overall: {'✓ PASSED' if gate_results['diagnostic_passed'] else '✗ FAILED'}

Valid Samples:
  KS: {list(n_valid_ks.values())}
  KPZ: {list(n_valid_kpz.values())}

σ (bandwidth): {sigma_base:.4f}
"""
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    fig_path = output_dir / "ks_vs_kpz_structure_functions.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    
    print()
    print("=" * 70)
    print("Exp 50m complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
