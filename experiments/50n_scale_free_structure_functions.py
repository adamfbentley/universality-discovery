"""
Experiment 50n: Scale-Free Structure Functions for KS Generalization Test
==========================================================================

PURPOSE:
Fix invariance failure from Exp 50m by using scale-free structure function features.

CHANGES FROM 50m:
- Normalize structure functions: S2(r) / S2(r_ref) where r_ref = smallest valid r
- This removes absolute scale dependence → features should be scale-invariant
- Still extract slope α from log-log fit (now on normalized values)
- Optional: curvature from quadratic log-log fit if stable

METHOD:
- Field-level spectral coarse-graining (like 50k/50l): FFT → low-pass → iFFT
- Compute S2(r) = ⟨|h(x+r) - h(x)|²⟩ at lag fractions [1/32, 1/16, 1/8, 1/4, 1/2]
- Normalize: S2_norm(r) = S2(r) / S2(r_ref)
- Features: [α (slope), normalized ratios at multiple lags, optional curvature]
- Diagnostic gate: KS-vs-KS and KPZ-vs-KPZ must both have |slope| < 0.005

PARAMETERS:
- KS: ν=1.0, κ=1.0, λ=1.0, L=256, T=100, N=500
- KPZ: ν=1.0, λ=1.0, D=0.5, L=256, T=100, N=500
- Scales: b ∈ {1, 2, 4}
- MMD: Fixed σ (median at b=1) or multi-σ average

EXPECTED OUTCOME:
If scale-free normalization fixes invariance:
  → Diagnostic gate passes (both slopes < 0.005)
  → KS-vs-KPZ slope interpretable
  → Compare with 50h/50k/50l (expect flat if KS ≠ KPZ)

If gate still fails:
  → Mark INVALID, structure function approach needs further refinement
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import linregress
import json

# ============================================================================
# SIMULATION FUNCTIONS (KS and KPZ)
# ============================================================================

def simulate_ks(L, T, dt, nu, kappa, lambda_val, noise_strength):
    """
    Kuramoto-Sivashinsky: ∂h/∂t = -ν∇⁴h + κ∇²h - (λ/2)(∇h)² + η
    Spectral method (pseudo-spectral for nonlinear term).
    """
    N = L
    dx = 1.0
    x = np.arange(N) * dx
    
    # Initial condition: small random perturbation
    h = np.random.randn(N) * 0.01
    
    # Fourier space setup
    k = fftfreq(N, d=dx) * 2 * np.pi
    k2 = k**2
    k4 = k**4
    
    # Time evolution
    nsteps = int(T / dt)
    for _ in range(nsteps):
        h_hat = ks_step_spectral(h, k2, k4, k, dt, nu, kappa, lambda_val, noise_strength)
        h = ifft(h_hat).real
    
    # Remove mean
    h = h - h.mean()
    return h

def ks_step_spectral(h, k2, k4, k, dt, nu, kappa, lambda_val, noise_strength):
    """Single KS time step using spectral method."""
    h_hat = fft(h)
    
    # Linear terms: -ν∇⁴h + κ∇²h
    linear_hat = (-nu * k4 + kappa * k2) * h_hat
    
    # Nonlinear term: -(λ/2)(∇h)²
    grad_h = ifft(1j * k * h_hat).real
    nonlinear = -0.5 * lambda_val * grad_h**2
    nonlinear_hat = fft(nonlinear)
    
    # Noise
    noise = np.random.randn(len(h)) * noise_strength
    noise_hat = fft(noise)
    
    # Update
    h_hat_new = h_hat + dt * (linear_hat + nonlinear_hat + noise_hat)
    
    return h_hat_new

def simulate_kpz(L, T, dt, nu, lambda_val, D):
    """
    KPZ: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
    Exponential Time Differencing (ETD) for stability.
    """
    N = L
    dx = 1.0
    x = np.arange(N) * dx
    
    h = np.random.randn(N) * 0.01
    
    k = fftfreq(N, d=dx) * 2 * np.pi
    k2 = k**2
    
    # ETD coefficients: exp(-ν k² dt) for diffusion
    exp_factor = np.exp(-nu * k2 * dt)
    # For k=0: etd_coef = dt; for k≠0: etd_coef = (exp(-νk²dt) - 1) / (-νk²)
    # Use np.where to avoid divide-by-zero warnings
    etd_coef = np.where(np.abs(k2) > 1e-10, 
                        (exp_factor - 1.0) / (-nu * k2), 
                        dt)
    
    nsteps = int(T / dt)
    for step in range(nsteps):
        h = kpz_step_etd(h, k, k2, exp_factor, etd_coef, dt, nu, lambda_val, D)
        
        # Check for instability periodically
        if step % 500 == 0 and not np.all(np.isfinite(h)):
            return None  # Signal failure
    
    h = h - h.mean()
    return h

def kpz_step_etd(h, k, k2, exp_factor, etd_coef, dt, nu, lambda_val, D):
    """Single KPZ time step using ETD scheme."""
    h_hat = fft(h)
    
    # Nonlinear: (λ/2)(∇h)²
    grad_h = ifft(1j * k * h_hat).real
    nonlinear = 0.5 * lambda_val * grad_h**2
    nonlinear_hat = fft(nonlinear)
    
    # Conservative noise: η with ∫η dx = 0
    noise = np.random.randn(len(h)) * np.sqrt(2 * D / dt)
    noise = noise - noise.mean()
    noise_hat = fft(noise)
    
    # ETD update: h_new = exp(-νk²dt)*h + φ(νk²dt)*(nonlinear + noise)
    # where φ = (exp(-x)-1)/(-x) ≈ dt for small k
    h_hat_new = exp_factor * h_hat + etd_coef * (nonlinear_hat + noise_hat)
    
    h_new = ifft(h_hat_new).real
    return h_new

# ============================================================================
# SCALE-FREE STRUCTURE FUNCTION FEATURES
# ============================================================================

def compute_scale_free_structure_functions(h, lag_fractions=[1/32, 1/16, 1/8, 1/4, 1/2]):
    """
    Compute scale-free structure function features.
    
    For each lag fraction:
    - r = int(round(fraction * len(h))), constrained to r >= 1
    - S2(r) = ⟨|h(x+r) - h(x)|²⟩
    
    Normalize: S2_norm(r) = S2(r) / S2(r_ref) where r_ref = smallest valid r
    
    Features:
    - α: slope from log(S2_norm) vs log(r) (power-law exponent)
    - S2_norm values at each lag (ratios, scale-free by definition = 1 at r_ref)
    - Optional: curvature from quadratic fit in log-log space
    
    Returns:
    - feature_vector or None if insufficient data
    - diagnostic string if invalid
    """
    N = len(h)
    
    # Compute lags
    lags = []
    for frac in lag_fractions:
        r = max(1, int(round(frac * N)))
        if r < N // 2:  # Avoid wrapping issues
            lags.append(r)
    
    if len(lags) < 3:
        return None, "Insufficient valid lags (<3)"
    
    # Compute S2(r) for each lag
    S2_values = []
    for r in lags:
        diffs = h[r:] - h[:-r]  # |h(x+r) - h(x)|
        S2 = np.mean(diffs**2)
        S2_values.append(S2)
    
    S2_values = np.array(S2_values)
    lags = np.array(lags)
    
    # Check for zeros or negatives
    if np.any(S2_values <= 0):
        return None, "Non-positive S2 values"
    
    # Normalize by smallest lag
    S2_ref = S2_values[0]
    S2_norm = S2_values / S2_ref
    
    # Fit slope in log-log space
    log_r = np.log(lags)
    log_S2_norm = np.log(S2_norm)
    
    # Check for NaN/inf
    if not np.all(np.isfinite(log_S2_norm)):
        return None, "Non-finite log(S2_norm)"
    
    # Linear fit
    result = linregress(log_r, log_S2_norm)
    alpha = result.slope
    
    # Optional: curvature from quadratic fit
    # For now, skip curvature (can add if needed)
    
    # Feature vector: [alpha, S2_norm[1:]/S2_norm[0] ratios]
    # Note: S2_norm[0] = 1 by definition, so skip it
    features = [alpha] + list(S2_norm[1:])
    
    return np.array(features), None

# ============================================================================
# COARSE-GRAINING
# ============================================================================

def spectral_coarse_grain(h, block_size):
    """
    Field-level spectral coarse-graining (as in 50k/50l).
    FFT → low-pass filter at k_cutoff(b) → iFFT
    """
    if block_size == 1:
        return h
    
    N = len(h)
    h_hat = fft(h)
    k = fftfreq(N) * N
    
    # Cutoff: retain modes |k| <= N/(2*block_size)
    k_cutoff = N / (2 * block_size)
    mask = np.abs(k) <= k_cutoff
    
    h_hat_filtered = h_hat * mask
    h_coarse = ifft(h_hat_filtered).real
    
    # Remove mean
    h_coarse = h_coarse - h_coarse.mean()
    
    return h_coarse

# ============================================================================
# MMD DISTANCE
# ============================================================================

def multi_sigma_mmd(X, Y, sigma_base):
    """
    Multi-σ MMD: average over {σ/2, σ, 2σ, 4σ}.
    """
    sigmas = [sigma_base / 2, sigma_base, 2 * sigma_base, 4 * sigma_base]
    mmd_values = []
    
    for sig in sigmas:
        gamma = 1.0 / (2 * sig**2)
        
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
        
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        mmd_values.append(np.sqrt(max(0, mmd2)))
    
    return np.mean(mmd_values)

def estimate_sigma(features):
    """Estimate σ from median pairwise distance."""
    n = len(features)
    if n < 2:
        return 1.0
    
    # Sample pairs to avoid O(n²) cost
    n_pairs = min(1000, n * (n - 1) // 2)
    dists = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        dist = np.linalg.norm(features[i] - features[j])
        dists.append(dist)
    
    return np.median(dists)

# ============================================================================
# DIAGNOSTIC GATE
# ============================================================================

def diagnostic_gate(features_dict, sigma, threshold=0.005):
    """
    Test KS-vs-KS and KPZ-vs-KPZ invariance across scales.
    Both must have |slope| < threshold to pass gate.
    
    Returns:
    - ks_pass, kpz_pass, ks_slope, kpz_slope
    """
    scales = sorted(features_dict['ks'].keys())
    
    # KS-vs-KS: split each scale's KS data in half, compute distance
    ks_distances = []
    for b in scales:
        X_ks = features_dict['ks'][b]
        n = len(X_ks)
        half = n // 2
        
        X1 = X_ks[:half]
        X2 = X_ks[half:]
        
        dist = multi_sigma_mmd(X1, X2, sigma)
        ks_distances.append(dist)
    
    # KPZ-vs-KPZ
    kpz_distances = []
    for b in scales:
        X_kpz = features_dict['kpz'][b]
        n = len(X_kpz)
        half = n // 2
        
        X1 = X_kpz[:half]
        X2 = X_kpz[half:]
        
        dist = multi_sigma_mmd(X1, X2, sigma)
        kpz_distances.append(dist)
    
    # Fit slopes
    log_b = np.log2(scales)
    
    ks_result = linregress(log_b, ks_distances)
    ks_slope = ks_result.slope
    
    kpz_result = linregress(log_b, kpz_distances)
    kpz_slope = kpz_result.slope
    
    # Test against threshold
    ks_pass = abs(ks_slope) < threshold
    kpz_pass = abs(kpz_slope) < threshold
    
    return ks_pass, kpz_pass, ks_slope, kpz_slope, ks_distances, kpz_distances

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def generate_field_with_retries(sim_fn, max_retries, label, index):
    """
    Generate a single field with hard retry/skip.
    Returns (field, success_flag).
    """
    for attempt in range(1, max_retries + 1):
        h = sim_fn()
        if np.all(np.isfinite(h)):
            return h, True
        print(f"  Warning: NaN/Inf detected in {label} field {index} (attempt {attempt}/{max_retries})")
    print(f"  ❌ Skipping {label} field {index} after {max_retries} failed attempts.")
    return None, False

def main():
    print("=" * 70)
    print("Experiment 50n: Scale-Free Structure Functions (KS Generalization)")
    print("=" * 70)
    
    # Parameters
    L = 256
    T = 50  # Shorter integration time
    dt_ks = 0.01
    dt_kpz = 0.01  # Same timestep as KS - ETD handles stability
    N = 500  # fields per system
    max_retries = 3  # hard retry/skip for unstable fields
    max_skip_fraction = 0.1  # abort if >10% of fields fail
    max_skips = int(np.floor(N * max_skip_fraction))
    
    # KS parameters
    nu_ks = 1.0
    kappa_ks = 1.0
    lambda_ks = 1.0
    noise_ks = 0.1
    
    # KPZ parameters
    nu_kpz = 1.0
    lambda_kpz = 0.5  # Moderate nonlinearity
    D_kpz = 0.1  # Moderate noise
    
    # Scales
    scales = [1, 2, 4]
    
    print("\nParameters:")
    print(f"  KS: nu={nu_ks}, kappa={kappa_ks}, lambda={lambda_ks}, noise={noise_ks}")
    print(f"  KPZ: nu={nu_kpz}, lambda={lambda_kpz}, D={D_kpz}")
    print(f"  L={L}, T={T}, N={N} fields per system")
    print(f"  Scales: {scales}")
    print(f"  Features: Scale-free structure functions (normalized by r_min)")
    
    # ========================================================================
    # Generate Data
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Generating KS fields...")
    print("=" * 70)
    
    import time
    t0 = time.time()
    
    ks_fields = []
    ks_skipped = 0
    for i in range(N):
        h, ok = generate_field_with_retries(
            lambda: simulate_ks(L, T, dt_ks, nu_ks, kappa_ks, lambda_ks, noise_ks),
            max_retries,
            "KS",
            i + 1
        )
        if ok:
            ks_fields.append(h)
        else:
            ks_skipped += 1
            if ks_skipped > max_skips:
                print(f"❌ ERROR: KS skipped fields exceeded {max_skips} (> {max_skip_fraction:.0%} of {N}).")
                print("Aborting run due to instability.")
                return
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{N} KS fields (skipped: {ks_skipped})...")
    
    t1 = time.time()
    print(f"✓ Generated {N} KS fields in {t1-t0:.1f}s")
    if ks_skipped > 0:
        print(f"⚠️  KS skipped fields: {ks_skipped}")
    
    print("\n" + "=" * 70)
    print("Generating KPZ fields...")
    print("=" * 70)
    
    t0 = time.time()
    
    kpz_fields = []
    kpz_skipped = 0
    for i in range(N):
        h, ok = generate_field_with_retries(
            lambda: simulate_kpz(L, T, dt_kpz, nu_kpz, lambda_kpz, D_kpz),
            max_retries,
            "KPZ",
            i + 1
        )
        if ok:
            kpz_fields.append(h)
        else:
            kpz_skipped += 1
            if kpz_skipped > max_skips:
                print(f"❌ ERROR: KPZ skipped fields exceeded {max_skips} (> {max_skip_fraction:.0%} of {N}).")
                print("Aborting run due to instability.")
                return
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{N} KPZ fields (skipped: {kpz_skipped})...")
    
    t1 = time.time()
    print(f"✓ Generated {N} KPZ fields in {t1-t0:.1f}s")
    if kpz_skipped > 0:
        print(f"⚠️  KPZ skipped fields: {kpz_skipped}")
    
    # ========================================================================
    # Extract Features at Each Scale
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Extracting scale-free structure function features...")
    print("=" * 70)
    
    features_dict = {
        'ks': {b: [] for b in scales},
        'kpz': {b: [] for b in scales}
    }
    
    valid_counts = {
        'ks': {b: 0 for b in scales},
        'kpz': {b: 0 for b in scales}
    }
    
    for system, fields in [('ks', ks_fields), ('kpz', kpz_fields)]:
        for b in scales:
            for h in fields:
                h_coarse = spectral_coarse_grain(h, b)
                features, diagnostic = compute_scale_free_structure_functions(h_coarse)
                
                if features is not None:
                    features_dict[system][b].append(features)
                    valid_counts[system][b] += 1
        
        print(f"\n{system.upper()} valid samples: {valid_counts[system]}")
    
    # Convert to arrays
    for system in ['ks', 'kpz']:
        for b in scales:
            features_dict[system][b] = np.array(features_dict[system][b])
    
    # Check minimum valid samples
    min_valid = min([valid_counts[s][b] for s in ['ks', 'kpz'] for b in scales])
    if min_valid < 50:
        print(f"\n❌ ERROR: Insufficient valid samples (min={min_valid})")
        return
    
    print(f"\n✓ Feature extraction complete (min valid samples: {min_valid})")
    
    # ========================================================================
    # Estimate σ at b=1
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Estimating MMD bandwidth σ...")
    print("=" * 70)
    
    all_features_b1 = np.vstack([features_dict['ks'][1], features_dict['kpz'][1]])
    sigma = estimate_sigma(all_features_b1)
    print(f"σ = {sigma:.4f} (median pairwise distance at b=1)")
    
    # ========================================================================
    # Diagnostic Gate
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC GATE: Testing system-vs-system invariance")
    print("=" * 70)
    
    ks_pass, kpz_pass, ks_slope, kpz_slope, ks_dists, kpz_dists = diagnostic_gate(
        features_dict, sigma, threshold=0.005
    )
    
    print(f"\nKS-vs-KS (half vs half):")
    for b, d in zip(scales, ks_dists):
        print(f"  b={b}: distance = {d:.4f}")
    print(f"  → Slope: {ks_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if ks_pass else '❌ FAIL'}")
    
    print(f"\nKPZ-vs-KPZ (half vs half):")
    for b, d in zip(scales, kpz_dists):
        print(f"  b={b}: distance = {d:.4f}")
    print(f"  → Slope: {kpz_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if kpz_pass else '❌ FAIL'}")
    
    gate_passed = ks_pass and kpz_pass
    
    print(f"\n{'='*70}")
    if gate_passed:
        print("✅ DIAGNOSTIC GATE: PASSED")
        print("Both KS and KPZ observables are scale-invariant.")
        print("KS-vs-KPZ trend is interpretable.")
    else:
        print("❌ DIAGNOSTIC GATE: FAILED")
        print("Observables are not scale-invariant.")
        print("Cannot interpret KS-vs-KPZ trend until invariance tests pass.")
    print('='*70)
    
    # ========================================================================
    # KS-vs-KPZ Distance (only if gate passed)
    # ========================================================================
    
    if gate_passed:
        print("\n" + "=" * 70)
        print("Computing KS-vs-KPZ distances...")
        print("=" * 70)
        
        distances = []
        for b in scales:
            X_ks = features_dict['ks'][b]
            X_kpz = features_dict['kpz'][b]
            
            dist = multi_sigma_mmd(X_ks, X_kpz, sigma)
            distances.append(dist)
            print(f"  b={b}: d(KS, KPZ) = {dist:.4f}")
        
        # Fit slope
        log_b = np.log2(scales)
        result = linregress(log_b, distances)
        slope = result.slope
        
        # Relative change
        rel_change = (distances[-1] - distances[0]) / distances[0] * 100
        
        print(f"\nSlope: {slope:+.6f}")
        print(f"Relative change: {rel_change:+.2f}%")
        
        # Interpretation
        print("\n" + "=" * 70)
        if abs(slope) < 0.01:
            print("✅ INTERPRETATION: KS and KPZ do NOT converge")
            print("Distance remains flat across scales → different universality classes")
        elif slope < -0.01:
            print("⚠️ INTERPRETATION: Partial convergence observed")
            print(f"Distance decreases by {-rel_change:.1f}% → possible weak convergence")
        else:
            print("⚠️ INTERPRETATION: Distance increases (divergence)")
        print("=" * 70)
    else:
        distances = [np.nan] * len(scales)
        slope = np.nan
        rel_change = np.nan
        
        print("\n⚠️ Skipping KS-vs-KPZ distance (gate failed)")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    output_dir = Path("results_exp50n")
    output_dir.mkdir(exist_ok=True)
    
    # Metadata
    metadata = {
        'experiment': '50n',
        'description': 'Scale-free structure functions for KS generalization',
        'parameters': {
            'L': L, 'T': T, 'N': N,
            'ks': {'nu': nu_ks, 'kappa': kappa_ks, 'lambda': lambda_ks, 'noise': noise_ks},
            'kpz': {'nu': nu_kpz, 'lambda': lambda_kpz, 'D': D_kpz},
            'scales': scales,
        },
        'diagnostic_gate': {
            'passed': bool(gate_passed),
            'ks_invariance_pass': bool(ks_pass),
            'kpz_invariance_pass': bool(kpz_pass),
            'ks_slope': float(ks_slope),
            'kpz_slope': float(kpz_slope),
            'threshold': 0.005,
        },
        'ks_vs_kpz': {
            'distances': [float(d) for d in distances],
            'slope': float(slope) if not np.isnan(slope) else None,
            'relative_change_percent': float(rel_change) if not np.isnan(rel_change) else None,
        },
        'sigma': float(sigma),
        'valid_counts': valid_counts,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Arrays
    np.savez(
        output_dir / 'ks_vs_kpz_scale_free_structure_functions.npz',
        scales=scales,
        distances=distances,
        slope=slope,
        ks_slope=ks_slope,
        kpz_slope=kpz_slope,
        sigma=sigma,
        gate_passed=gate_passed,
        **{f'ks_b{b}': features_dict['ks'][b] for b in scales},
        **{f'kpz_b{b}': features_dict['kpz'][b] for b in scales}
    )
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Diagnostic gate
    ax = axes[0]
    ax.plot(scales, ks_dists, 'o-', label=f'KS-vs-KS (slope={ks_slope:+.6f})', color='blue')
    ax.plot(scales, kpz_dists, 's-', label=f'KPZ-vs-KPZ (slope={kpz_slope:+.6f})', color='green')
    ax.axhline(0.005, ls='--', color='red', alpha=0.5, label='Threshold')
    ax.set_xlabel('Scale b')
    ax.set_ylabel('MMD Distance (half vs half)')
    ax.set_title('Diagnostic Gate: System-vs-System Invariance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: KS-vs-KPZ (only if gate passed)
    ax = axes[1]
    if gate_passed:
        ax.plot(scales, distances, 'o-', color='purple', label=f'slope={slope:+.6f}')
        ax.set_xlabel('Scale b')
        ax.set_ylabel('MMD Distance')
        ax.set_title(f'KS vs KPZ (Gate: ✅ PASS)')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, '❌ Gate Failed\nKS-vs-KPZ not interpretable',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('KS vs KPZ (Gate: ❌ FAIL)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ks_vs_kpz_scale_free_structure_functions.png', dpi=150)
    print(f"✓ Figure saved to {output_dir}/ks_vs_kpz_scale_free_structure_functions.png")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 50n SUMMARY")
    print("=" * 70)
    print(f"Observable: Scale-free structure functions")
    print(f"Diagnostic gate: {'✅ PASSED' if gate_passed else '❌ FAILED'}")
    print(f"  KS-vs-KS slope: {ks_slope:+.6f} ({'pass' if ks_pass else 'FAIL'})")
    print(f"  KPZ-vs-KPZ slope: {kpz_slope:+.6f} ({'pass' if kpz_pass else 'FAIL'})")
    
    if gate_passed:
        print(f"KS-vs-KPZ slope: {slope:+.6f} ({rel_change:+.1f}%)")
        if abs(slope) < 0.01:
            print("→ KS does NOT converge to KPZ (flat distance)")
        elif slope < -0.01:
            print(f"→ Partial convergence ({-rel_change:.1f}% reduction)")
        else:
            print("→ Divergence observed")
    else:
        print("KS-vs-KPZ: Not interpretable (gate failed)")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
