"""
Experiment 50l: KS Regime-Dependence Sweep

PURPOSE:
Test whether the flat KS vs KPZ result (Exp 50k) is regime-specific or robust.

APPROACH:
- Define 5 KS parameter regimes (vary ν, κ, λ systematically)
- Keep L=256, dt, sample counts FIXED
- Run the SAME validated Diagnostic B pipeline with gate enforced
- Compare slopes across regimes

REGIMES:
1. Baseline (current): ν=1, κ=1, λ=1
2. Strong dispersion: ν=2, κ=1, λ=1 (enhance -ν∇⁴h term)
3. Weak dispersion: ν=0.5, κ=1, λ=1
4. Strong anti-diffusion: ν=1, κ=2, λ=1 (enhance +κ∇²h term)
5. Strong nonlinearity: ν=1, κ=1, λ=2 (enhance -λ/2(∇h)² term)

If all regimes stay flat → KS↔KPZ separation is robust
If any converge → map the boundary, nuanced claim
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import linregress
from numba import jit
import time

# ============================================================================
# KS SIMULATION (same as before, parameterized)
# ============================================================================

@jit(nopython=True)
def ks_step_pseudospectral(h_hat, k, k2, k4, dt, nu, kappa, lam, noise_amp):
    """
    One timestep of KS equation using pseudo-spectral method.
    
    KS equation: ∂h/∂t = -ν∇⁴h + κ∇²h - (λ/2)(∇h)² + η
    
    Parameters:
    -----------
    nu : float - coefficient of -∇⁴h (dispersion/hyperviscosity)
    kappa : float - coefficient of +∇²h (anti-diffusion/instability)  
    lam : float - coefficient of -(∇h)²/2 (nonlinearity)
    """
    L = len(h_hat)
    
    # Linear part: -ν k⁴ + κ k² (note signs!)
    linear = -nu * k4 + kappa * k2
    
    # Nonlinear term in real space
    h = np.zeros(L)
    for i in range(L):
        h[i] = 0.0
    
    # Manual inverse FFT (Numba compatible)
    for x in range(L):
        for m in range(L):
            angle = 2 * np.pi * m * x / L
            h[x] += (h_hat[m].real * np.cos(angle) - h_hat[m].imag * np.sin(angle)) / L
    
    # Gradient in real space
    grad_h = np.zeros(L)
    ik_h_hat = 1j * k * h_hat
    for x in range(L):
        for m in range(L):
            angle = 2 * np.pi * m * x / L
            grad_h[x] += (ik_h_hat[m].real * np.cos(angle) - ik_h_hat[m].imag * np.sin(angle)) / L
    
    # Nonlinear term: -(λ/2)(∇h)²
    nonlin_real = -0.5 * lam * grad_h**2
    
    # FFT of nonlinear term
    nonlin_hat = np.zeros(L, dtype=np.complex128)
    for m in range(L):
        for x in range(L):
            angle = -2 * np.pi * m * x / L
            nonlin_hat[m] += nonlin_real[x] * (np.cos(angle) + 1j * np.sin(angle))
    
    # Noise in Fourier space
    noise_hat = np.zeros(L, dtype=np.complex128)
    for m in range(L):
        noise_hat[m] = noise_amp * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    noise_hat[0] = 0  # No noise at k=0
    
    # Semi-implicit update
    h_hat_new = (h_hat + dt * (nonlin_hat + noise_hat)) / (1 - dt * linear)
    
    return h_hat_new

def simulate_ks_regime(L, n_steps, dt, nu, kappa, lam, noise_amp, n_samples, sample_interval):
    """
    Simulate KS equation with given parameters.
    
    Returns sampled height fields.
    """
    # Setup
    k = 2 * np.pi * np.fft.fftfreq(L, d=2*np.pi/L)
    k2 = k**2
    k4 = k**4
    
    # Initial condition: small random perturbation
    h = 0.01 * np.random.randn(L)
    h_hat = fft(h)
    
    # Burn-in
    burn_in = n_steps // 5
    for _ in range(burn_in):
        h_hat = ks_step_pseudospectral(h_hat, k, k2, k4, dt, nu, kappa, lam, noise_amp)
    
    # Collect samples
    samples = []
    steps_per_sample = sample_interval
    
    for i in range(n_samples):
        for _ in range(steps_per_sample):
            h_hat = ks_step_pseudospectral(h_hat, k, k2, k4, dt, nu, kappa, lam, noise_amp)
        
        h_real = np.real(ifft(h_hat))
        samples.append(h_real.copy())
    
    return np.array(samples)

# ============================================================================
# SPECTRAL FEATURES (from 50k, validated)
# ============================================================================

def compute_power_spectrum_clean(h):
    """Compute power spectrum with mean subtraction, dropping k=0."""
    L = len(h)
    h_centered = h - np.mean(h)
    h_fft = fft(h_centered)
    S = np.abs(h_fft)**2 / L
    k = fftfreq(L, d=1.0)
    positive_k_mask = k > 0
    return k[positive_k_mask], S[positive_k_mask]

def extract_spectral_shape_features_clean(h, k_cutoff_applied, min_total_modes=20, min_low_modes=10):
    """Extract spectral features with no degeneracy (from 50k)."""
    k, S = compute_power_spectrum_clean(h)
    
    n_modes = len(k)
    if n_modes < min_total_modes:
        return None
    
    total_power = np.sum(S)
    if total_power < 1e-12:
        return None
    
    valid_k_mask = k <= k_cutoff_applied
    k_valid = k[valid_k_mask]
    S_valid = S[valid_k_mask]
    
    if len(k_valid) < min_total_modes:
        return None
    
    if len(k_valid) < min_low_modes:
        return None
    
    # Low-k slope
    k_fit = k_valid[:min_low_modes]
    S_fit = S_valid[:min_low_modes]
    valid = (S_fit > 1e-12) & (k_fit > 0)
    if np.sum(valid) >= 3:
        slope, _, _, _, _ = linregress(np.log(k_fit[valid]), np.log(S_fit[valid]))
        low_k_slope = slope
    else:
        return None
    
    # Log-binned spectrum
    k_min = k_valid[0]
    k_max = k_cutoff_applied
    n_bins = 8
    log_edges = np.linspace(np.log(k_min), np.log(k_max), n_bins + 1)
    bin_edges = np.exp(log_edges)
    
    bin_powers = []
    for i in range(n_bins):
        if i == n_bins - 1:
            bin_mask = (k_valid >= bin_edges[i]) & (k_valid <= bin_edges[i+1])
        else:
            bin_mask = (k_valid >= bin_edges[i]) & (k_valid < bin_edges[i+1])
        bin_powers.append(np.sum(S_valid[bin_mask]))
    
    bin_powers = np.array(bin_powers)
    total_power_valid = np.sum(S_valid)
    if total_power_valid < 1e-12:
        return None
    bin_powers_norm = bin_powers / (total_power_valid + 1e-12)
    
    # Spectral centroid
    k_cent = np.sum(k_valid * S_valid) / (total_power_valid + 1e-12)
    k_cent_norm = k_cent / k_max
    
    return np.concatenate([[low_k_slope], bin_powers_norm, [k_cent_norm]])

# ============================================================================
# MMD (from 50k)
# ============================================================================

def rbf_kernel(X, Y, sigma):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    XY = X @ Y.T
    distances_sq = XX - 2*XY + YY
    return np.exp(-distances_sq / (2 * sigma**2))

def compute_mmd_single_sigma(X, Y, sigma):
    n_x, n_y = len(X), len(Y)
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (n_x * (n_x - 1))
    mmd_sq += (np.sum(K_YY) - np.trace(K_YY)) / (n_y * (n_y - 1))
    mmd_sq -= 2 * np.mean(K_XY)
    return np.sqrt(max(0, mmd_sq))

def estimate_bandwidth(X, Y):
    sample_X = X[np.random.choice(len(X), size=min(100, len(X)), replace=False)]
    sample_Y = Y[np.random.choice(len(Y), size=min(100, len(Y)), replace=False)]
    dists = [np.linalg.norm(x - y) for x in sample_X[:50] for y in sample_Y[:50]]
    return np.median(dists)

def compute_mmd_multi_sigma(X, Y, sigma_base):
    sigma_scales = [0.5, 1.0, 2.0, 4.0]
    mmds = [compute_mmd_single_sigma(X, Y, sigma_base * s) for s in sigma_scales]
    return np.mean(mmds)

# ============================================================================
# COARSE-GRAINING
# ============================================================================

def coarse_grain_field_spectral(h, k_cutoff_fraction):
    L = len(h)
    h_hat = fft(h)
    k = np.fft.fftfreq(L)
    k_max = np.max(np.abs(k))
    k_cutoff = k_cutoff_fraction * k_max
    mask = np.abs(k) <= k_cutoff
    h_hat_filtered = h_hat * mask
    return np.real(ifft(h_hat_filtered)), k_cutoff

# ============================================================================
# DIAGNOSTIC B FOR ONE REGIME
# ============================================================================

def run_diagnostic_b_for_regime(ks_fields, kpz_features_by_scale, regime_name, 
                                 scales, scale_names):
    """
    Run full Diagnostic B pipeline for one KS regime.
    
    Returns:
    --------
    result : dict with slope, diagnostic_passed, etc.
    """
    print(f"\n{'='*70}")
    print(f"REGIME: {regime_name}")
    print(f"{'='*70}")
    
    # Extract features for KS at each scale
    ks_features_by_scale = {}
    
    for scale, name in zip(scales, scale_names):
        features_list = []
        L = len(ks_fields[0])
        k_nyquist = 0.5
        k_cutoff = scale * k_nyquist
        
        for h in ks_fields:
            h_coarse, k_c = coarse_grain_field_spectral(h, scale)
            features = extract_spectral_shape_features_clean(h_coarse, k_c)
            if features is not None:
                features_list.append(features)
        
        if len(features_list) >= 100:
            ks_features_by_scale[scale] = np.array(features_list)
            print(f"  {name}: {len(features_list)} valid samples")
        else:
            print(f"  {name}: Too few valid samples ({len(features_list)})")
    
    # Find valid scales for comparison
    valid_scales = sorted(set(ks_features_by_scale.keys()) & set(kpz_features_by_scale.keys()))
    valid_names = [f"b={int(1/s)}" for s in valid_scales]
    
    if len(valid_scales) < 3:
        print(f"  ❌ Too few valid scales ({len(valid_scales)})")
        return {'regime': regime_name, 'valid': False, 'reason': 'too_few_scales'}
    
    # KS vs KPZ distances
    first_scale = valid_scales[0]
    sigma_base = estimate_bandwidth(ks_features_by_scale[first_scale], 
                                     kpz_features_by_scale[first_scale])
    
    distances_ks_kpz = []
    for scale in valid_scales:
        d = compute_mmd_multi_sigma(ks_features_by_scale[scale], 
                                     kpz_features_by_scale[scale], sigma_base)
        distances_ks_kpz.append(d)
    
    # KS vs KS (split half)
    ks_half1 = {s: ks_features_by_scale[s][:len(ks_features_by_scale[s])//2] for s in valid_scales}
    ks_half2 = {s: ks_features_by_scale[s][len(ks_features_by_scale[s])//2:] for s in valid_scales}
    
    sigma_ks = estimate_bandwidth(ks_half1[first_scale], ks_half2[first_scale])
    distances_ks_ks = [compute_mmd_multi_sigma(ks_half1[s], ks_half2[s], sigma_ks) for s in valid_scales]
    
    # KPZ vs KPZ (split half)
    kpz_half1 = {s: kpz_features_by_scale[s][:len(kpz_features_by_scale[s])//2] for s in valid_scales}
    kpz_half2 = {s: kpz_features_by_scale[s][len(kpz_features_by_scale[s])//2:] for s in valid_scales}
    
    sigma_kpz = estimate_bandwidth(kpz_half1[first_scale], kpz_half2[first_scale])
    distances_kpz_kpz = [compute_mmd_multi_sigma(kpz_half1[s], kpz_half2[s], sigma_kpz) for s in valid_scales]
    
    # Compute slopes
    b_values = [int(1/s) for s in valid_scales]
    slope_ks_kpz, _, _, p_value, _ = linregress(b_values, distances_ks_kpz)
    slope_ks_ks, _, _, _, _ = linregress(b_values, distances_ks_ks)
    slope_kpz_kpz, _, _, _, _ = linregress(b_values, distances_kpz_kpz)
    
    # Diagnostic gate
    ks_pass = abs(slope_ks_ks) < 0.005
    kpz_pass = abs(slope_kpz_kpz) < 0.005
    diagnostic_passed = ks_pass and kpz_pass
    
    # Interpretation
    if slope_ks_kpz < -0.01:
        interpretation = "convergence"
    elif abs(slope_ks_kpz) < 0.01:
        interpretation = "flat"
    else:
        interpretation = "divergence"
    
    # Print summary
    print(f"\n  Results:")
    print(f"    KS vs KPZ slope: {slope_ks_kpz:+.6f} ({interpretation})")
    print(f"    KS vs KS slope:  {slope_ks_ks:+.6f} ({'PASS' if ks_pass else 'FAIL'})")
    print(f"    KPZ vs KPZ slope: {slope_kpz_kpz:+.6f} ({'PASS' if kpz_pass else 'FAIL'})")
    print(f"    Diagnostic: {'✅ PASSED' if diagnostic_passed else '❌ FAILED'}")
    
    relative_change = (distances_ks_kpz[-1] - distances_ks_kpz[0]) / distances_ks_kpz[0] if distances_ks_kpz[0] > 0 else 0
    
    return {
        'regime': regime_name,
        'valid': True,
        'diagnostic_passed': diagnostic_passed,
        'slope_ks_kpz': slope_ks_kpz,
        'slope_ks_ks': slope_ks_ks,
        'slope_kpz_kpz': slope_kpz_kpz,
        'p_value': p_value,
        'relative_change': relative_change,
        'interpretation': interpretation,
        'distances_ks_kpz': distances_ks_kpz,
        'valid_scales': valid_scales,
        'ks_pass': ks_pass,
        'kpz_pass': kpz_pass,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50l: KS REGIME-DEPENDENCE SWEEP")
    print("="*70)
    print("\nQuestion: Is the flat KS vs KPZ result regime-specific or robust?")
    print("\nApproach:")
    print("  - 5 KS parameter regimes")
    print("  - Same Diagnostic B pipeline with gate enforced")
    print("  - Compare slopes across regimes")
    print("="*70)
    
    output_dir = Path('results/exp50l_ks_regime_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed simulation parameters
    L = 256
    dt = 0.01
    n_steps = 50000
    n_samples = 500  # Per regime (smaller for speed)
    sample_interval = 50
    noise_amp = 0.1
    
    # Define regimes
    regimes = [
        {'name': 'Baseline (ν=1, κ=1, λ=1)', 'nu': 1.0, 'kappa': 1.0, 'lam': 1.0},
        {'name': 'Strong dispersion (ν=2)', 'nu': 2.0, 'kappa': 1.0, 'lam': 1.0},
        {'name': 'Weak dispersion (ν=0.5)', 'nu': 0.5, 'kappa': 1.0, 'lam': 1.0},
        {'name': 'Strong anti-diffusion (κ=2)', 'nu': 1.0, 'kappa': 2.0, 'lam': 1.0},
        {'name': 'Strong nonlinearity (λ=2)', 'nu': 1.0, 'kappa': 1.0, 'lam': 2.0},
    ]
    
    # Scales to test
    scales = [1.0, 0.5, 0.25]  # b=1, 2, 4
    scale_names = ['b=1', 'b=2', 'b=4']
    
    # Load KPZ reference data (precomputed)
    print("\nLoading KPZ reference data...")
    kpz_path = Path('results/kpz_fields_matched_L256/kpz_matched_data.pkl')
    with open(kpz_path, 'rb') as f:
        kpz_data = pickle.load(f)
    
    kpz_fields = kpz_data['fields']
    print(f"  Loaded {len(kpz_fields)} KPZ fields")
    
    # Extract KPZ features at each scale (once)
    print("\nExtracting KPZ features...")
    kpz_features_by_scale = {}
    for scale, name in zip(scales, scale_names):
        features_list = []
        k_cutoff = scale * 0.5
        for h in kpz_fields:
            h_coarse, k_c = coarse_grain_field_spectral(h, scale)
            features = extract_spectral_shape_features_clean(h_coarse, k_c)
            if features is not None:
                features_list.append(features)
        if len(features_list) >= 100:
            kpz_features_by_scale[scale] = np.array(features_list)
            print(f"  {name}: {len(features_list)} valid samples")
    
    # Run each regime
    all_results = []
    
    for regime in regimes:
        print(f"\n{'='*70}")
        print(f"Generating KS data for: {regime['name']}")
        print(f"{'='*70}")
        
        t0 = time.time()
        ks_fields = simulate_ks_regime(
            L=L, n_steps=n_steps, dt=dt,
            nu=regime['nu'], kappa=regime['kappa'], lam=regime['lam'],
            noise_amp=noise_amp, n_samples=n_samples, sample_interval=sample_interval
        )
        t1 = time.time()
        print(f"  Generated {len(ks_fields)} fields in {t1-t0:.1f}s")
        
        # Run Diagnostic B
        result = run_diagnostic_b_for_regime(
            ks_fields, kpz_features_by_scale, regime['name'],
            scales, scale_names
        )
        result['params'] = regime
        all_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("REGIME SWEEP SUMMARY")
    print("="*70)
    
    print("\n{:<35} {:>12} {:>12} {:>10}".format(
        "Regime", "Slope", "Interpretation", "Diagnostic"))
    print("-"*70)
    
    for r in all_results:
        if r['valid']:
            print("{:<35} {:>+12.6f} {:>12} {:>10}".format(
                r['regime'][:35], 
                r['slope_ks_kpz'], 
                r['interpretation'],
                '✅ PASS' if r['diagnostic_passed'] else '❌ FAIL'
            ))
        else:
            print("{:<35} {:>12} {:>12} {:>10}".format(
                r['regime'][:35], "N/A", r.get('reason', 'invalid'), "N/A"))
    
    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    valid_results = [r for r in all_results if r['valid'] and r['diagnostic_passed']]
    
    if len(valid_results) == 0:
        print("\n❌ No valid results - cannot draw conclusions")
    else:
        convergent = [r for r in valid_results if r['interpretation'] == 'convergence']
        flat = [r for r in valid_results if r['interpretation'] == 'flat']
        divergent = [r for r in valid_results if r['interpretation'] == 'divergence']
        
        print(f"\nOut of {len(valid_results)} valid regimes:")
        print(f"  - Convergent: {len(convergent)}")
        print(f"  - Flat: {len(flat)}")
        print(f"  - Divergent: {len(divergent)}")
        
        if len(flat) == len(valid_results):
            print("\n🔬 ROBUST RESULT: All regimes show flat KS vs KPZ distance")
            print("   → KS↔KPZ separation is robust in this framework")
        elif len(convergent) > 0:
            print("\n🎯 REGIME-DEPENDENT: Some regimes show convergence!")
            print("   Convergent regimes:")
            for r in convergent:
                print(f"     - {r['regime']}: slope = {r['slope_ks_kpz']:+.6f}")
        else:
            print("\n⚠️  MIXED RESULTS: No clear pattern")
    
    # Save results
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump({
            'regimes': regimes,
            'results': all_results,
            'scales': scales,
            'scale_names': scale_names,
        }, f)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
