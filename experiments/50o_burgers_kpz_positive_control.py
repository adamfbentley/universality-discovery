"""
Experiment 50o: Burgers → KPZ Convergence Test (Positive Control)
====================================================================

PURPOSE:
Test whether the framework can detect CONVERGENCE when systems ARE related.

THEORY:
Burgers equation: ∂u/∂t + u∇u = ν∇²u
Via Cole-Hopf transform u = ∇h, this becomes KPZ: ∂h/∂t = ν∇²h + (1/2)(∇h)² + η

EXPECTATION:
- Burgers and KPZ should CONVERGE under coarse-graining
- Distance should DECREASE across scales (negative slope)
- Diagnostic gate should pass (both systems scale-invariant)

CONTRASTS WITH:
- Exp 50h-50n: KS ≠ KPZ (distance flat, no convergence)
- This: Burgers → KPZ (distance decreases, convergence)

METHOD:
- Observable: Scale-free structure functions in height space (as in Exp 50n)
- Burgers u converted to height h via spectral integration
- Diagnostic gate: Burgers-vs-Burgers and KPZ-vs-KPZ must be flat
- Expected: Burgers-vs-KPZ slope < 0 (convergence)

PARAMETERS:
- Burgers: ν=1.0, L=256, T=600, N=500
- KPZ: ν=1.0, λ=1.0, D=0.1, L=256, T=600, N=500
- Scales: b ∈ {1, 2, 4, 8}
- MMD: Fixed σ (median at b=1)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import linregress
import json

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_burgers(L, T, dt, nu, D):
    """
    Stochastic Burgers equation with GRADIENT NOISE:
    du/dt + u*du/dx = nu*d^2u/dx^2 + d_x(eta)
    
    Gradient noise d_x(eta) is essential for Cole-Hopf equivalence to KPZ.
    Standard Burgers with additive noise is NOT equivalent to KPZ.
    
    ETD scheme with de-aliasing for stability.
    Returns velocity field u (not height h).
    """
    N = L
    dx = 1.0
    
    # Initial condition: small random perturbation
    u = np.random.randn(N) * 0.01
    
    # Fourier space setup
    k = fftfreq(N, d=dx) * 2 * np.pi
    k2 = k**2
    
    # ETD coefficients for diffusion term (nu*nabla^2 = -nu*k^2)
    exp_factor = np.exp(-nu * k2 * dt)
    etd_coef = np.zeros_like(k2)
    mask_nonzero = np.abs(k2) > 1e-10
    etd_coef[mask_nonzero] = (exp_factor[mask_nonzero] - 1.0) / (-nu * k2[mask_nonzero])
    etd_coef[~mask_nonzero] = dt  # For k=0, limit is dt
    
    # De-aliasing mask (2/3 rule)
    k_max = np.max(np.abs(k))
    dealias = np.abs(k) < (2.0/3.0) * k_max
    
    # Time evolution
    nsteps = int(T / dt)
    for step in range(nsteps):
        u = burgers_step_etd(u, k, k2, exp_factor, etd_coef, dealias, dt, nu, D)
        
        # Check for instability
        if step % 500 == 0 and not np.all(np.isfinite(u)):
            return None
    
    # Remove mean
    u = u - u.mean()
    
    # Return velocity field directly (for velocity-space comparison)
    return u

def burgers_step_etd(u, k, k2, exp_factor, etd_coef, dealias, dt, nu, D):
    """Single Burgers time step using ETD with de-aliasing and GRADIENT NOISE."""
    u_hat = fft(u)
    
    # Nonlinear term: -u*du/dx (advection)
    # De-alias before computing nonlinear term
    u_hat_dealiased = u_hat * dealias
    grad_u = ifft(1j * k * u_hat_dealiased).real
    u_real = ifft(u_hat_dealiased).real
    nonlinear = -u_real * grad_u
    nonlinear_hat = fft(nonlinear) * dealias  # De-alias result
    
    # GRADIENT NOISE: d_x(eta) for Cole-Hopf equivalence
    # Generate white noise eta, then take spatial derivative
    eta = np.random.randn(len(u)) * np.sqrt(2 * D / dt)
    eta = eta - eta.mean()  # Conservative
    eta_hat = fft(eta)
    # Gradient of noise: d_x(eta) = ik * eta_hat in Fourier space
    grad_noise_hat = 1j * k * eta_hat
    
    # ETD update: exp(-nu*k^2*dt) * [u + phi*(nonlinear + grad_noise)]
    u_hat_new = exp_factor * u_hat + etd_coef * (nonlinear_hat + grad_noise_hat)
    
    u_new = ifft(u_hat_new).real
    return u_new

def simulate_kpz(L, T, dt, nu, lambda_val, D):
    """
    KPZ: dh/dt = nu*d^2h/dx^2 + (lambda/2)*(dh/dx)^2 + eta
    Exponential Time Differencing (ETD) for stability.

    Returns height field h for structure-function features.
    """
    N = L
    dx = 1.0
    
    h = np.random.randn(N) * 0.01
    
    k = fftfreq(N, d=dx) * 2 * np.pi
    k2 = k**2
    
    # ETD coefficients
    exp_factor = np.exp(-nu * k2 * dt)
    etd_coef = np.zeros_like(k2)
    mask_nonzero = np.abs(k2) > 1e-10
    etd_coef[mask_nonzero] = (exp_factor[mask_nonzero] - 1.0) / (-nu * k2[mask_nonzero])
    etd_coef[~mask_nonzero] = dt  # For k=0, limit is dt
    
    nsteps = int(T / dt)
    for step in range(nsteps):
        h = kpz_step_etd(h, k, k2, exp_factor, etd_coef, dt, nu, lambda_val, D)
        
        # Check for instability
        if step % 500 == 0 and not np.all(np.isfinite(h)):
            return None
    
    h = h - h.mean()
    return h

def velocity_to_height(u):
    """
    Convert velocity field u = dh/dx to height field h by spectral integration.
    Sets k=0 mode to zero (height defined up to a constant).
    """
    N = len(u)
    u = u - u.mean()

    k = fftfreq(N) * 2 * np.pi
    u_hat = fft(u)

    h_hat = np.zeros_like(u_hat, dtype=np.complex128)
    mask_nonzero = np.abs(k) > 1e-10
    h_hat[mask_nonzero] = u_hat[mask_nonzero] / (1j * k[mask_nonzero])
    h_hat[~mask_nonzero] = 0.0

    h = ifft(h_hat).real
    h = h - h.mean()
    return h

def kpz_step_etd(h, k, k2, exp_factor, etd_coef, dt, nu, lambda_val, D):
    """Single KPZ time step using ETD scheme."""
    h_hat = fft(h)
    
    # Nonlinear: (λ/2)(∇h)²
    grad_h = ifft(1j * k * h_hat).real
    nonlinear = 0.5 * lambda_val * grad_h**2
    nonlinear_hat = fft(nonlinear)
    
    # Conservative noise
    noise = np.random.randn(len(h)) * np.sqrt(2 * D / dt)
    noise = noise - noise.mean()
    noise_hat = fft(noise)
    
    # ETD update
    h_hat_new = exp_factor * h_hat + etd_coef * (nonlinear_hat + noise_hat)
    
    h_new = ifft(h_hat_new).real
    return h_new

# ============================================================================
# SCALE-FREE STRUCTURE FUNCTION FEATURES (height space)
# ============================================================================

def spectral_coarse_grain(h, b, alpha=0.5):
    """
    Field-level spectral coarse-graining with RG rescaling for height.

    1. FFT -> low-pass filter -> iFFT (coarse-grain)
    2. Rescale height: h_rg = h_coarse / b^alpha  (alpha=1/2 in 1D KPZ)
    3. Ensure zero-mean and remove k=0 mode
    """
    if b == 1:
        h = h - h.mean()
        return h

    N = len(h)
    h = h - h.mean()

    h_hat = fft(h)
    k = fftfreq(N)

    # Remove k=0 mode explicitly
    h_hat[0] = 0

    # Low-pass filter
    k_cutoff = 0.5 / b
    h_hat[np.abs(k) > k_cutoff] = 0

    h_coarse = ifft(h_hat).real

    # RG rescaling: h scales as b^alpha in 1D KPZ (alpha=1/2)
    h_rg = h_coarse / (b ** alpha)

    # Ensure zero-mean after rescaling
    h_rg = h_rg - h_rg.mean()

    return h_rg

def compute_scale_free_structure_functions(h, lag_fractions=(1/32, 1/16, 1/8, 1/4, 1/2)):
    """
    Scale-free structure function features (same as Exp 50n).

    - S2(r) = <|h(x+r) - h(x)|^2>
    - Normalize by smallest lag: S2_norm(r) = S2(r) / S2(r_ref)
    - Features: slope alpha in log-log + normalized ratios at larger lags

    Returns (features, diagnostic) where features is None if invalid.
    """
    N = len(h)

    lags = []
    for frac in lag_fractions:
        r = max(1, int(round(frac * N)))
        if r < N // 2:
            lags.append(r)

    if len(lags) < 3:
        return None, "Insufficient valid lags (<3)"

    S2_values = []
    for r in lags:
        diffs = h[r:] - h[:-r]
        S2 = np.mean(diffs ** 2)
        S2_values.append(S2)

    S2_values = np.array(S2_values)
    lags = np.array(lags)

    if np.any(S2_values <= 0):
        return None, "Non-positive S2 values"

    S2_ref = S2_values[0]
    S2_norm = S2_values / S2_ref

    log_r = np.log(lags)
    log_S2_norm = np.log(S2_norm)

    if not np.all(np.isfinite(log_S2_norm)):
        return None, "Non-finite log(S2_norm)"

    result = linregress(log_r, log_S2_norm)
    alpha = result.slope

    # Feature vector: [alpha] + normalized ratios (skip S2_norm[0] == 1)
    features = [alpha] + list(S2_norm[1:])

    return np.array(features), None

# ============================================================================
# MMD DISTANCE
# ============================================================================

def mmd_distance(X, Y, sigma):
    """Maximum Mean Discrepancy with RBF kernel."""
    XX = rbf_kernel(X, X, gamma=1.0 / (2 * sigma**2))
    YY = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma**2))
    XY = rbf_kernel(X, Y, gamma=1.0 / (2 * sigma**2))
    
    mmd2 = XX.mean() + YY.mean() - 2 * XY.mean()
    return np.sqrt(max(0, mmd2))

def estimate_sigma(X, Y):
    """Estimate bandwidth as median pairwise distance."""
    combined = np.vstack([X[:100], Y[:100]])
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(combined)
    sigma = np.median(dists[dists > 0])
    return sigma

# ============================================================================
# DIAGNOSTIC GATE
# ============================================================================

def diagnostic_gate(features_dict, scales, sigma, threshold=0.005):
    """
    Test system-vs-system invariance (split-half test).
    Both Burgers-vs-Burgers and KPZ-vs-KPZ must be flat.
    """
    burgers_distances = []
    kpz_distances = []
    
    for b in scales:
        # Split each system in half
        burgers_feats = features_dict['burgers'][b]
        kpz_feats = features_dict['kpz'][b]
        
        mid_b = len(burgers_feats) // 2
        mid_k = len(kpz_feats) // 2
        
        # Burgers-vs-Burgers
        d_burgers = mmd_distance(burgers_feats[:mid_b], burgers_feats[mid_b:], sigma)
        burgers_distances.append(d_burgers)
        
        # KPZ-vs-KPZ
        d_kpz = mmd_distance(kpz_feats[:mid_k], kpz_feats[mid_k:], sigma)
        kpz_distances.append(d_kpz)
    
    # Fit slopes
    log_b = np.log2(scales)
    
    burgers_result = linregress(log_b, burgers_distances)
    burgers_slope = burgers_result.slope
    
    kpz_result = linregress(log_b, kpz_distances)
    kpz_slope = kpz_result.slope
    
    # Test against threshold
    burgers_pass = abs(burgers_slope) < threshold
    kpz_pass = abs(kpz_slope) < threshold
    
    return burgers_pass, kpz_pass, burgers_slope, kpz_slope, burgers_distances, kpz_distances

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def generate_field_with_retries(sim_fn, max_retries, label, index):
    """Generate a single field with retry logic."""
    for attempt in range(1, max_retries + 1):
        h = sim_fn()
        if h is not None and np.all(np.isfinite(h)):
            return h, True
        if attempt < max_retries:
            print(f"  Warning: Invalid {label} field {index} (attempt {attempt}/{max_retries})")
    print(f"  ❌ Skipping {label} field {index} after {max_retries} failed attempts.")
    return None, False

def main():
    print("=" * 70)
    print("Experiment 50o: Burgers → KPZ Convergence (Positive Control)")
    print("=" * 70)
    
    #Parameters
    L = 256
    T = 600  # Longer integration to approach stationary regime
    dt_burgers = 0.005
    dt_kpz = 0.01
    N = 500
    max_retries = 3
    max_skip_fraction = 0.1
    max_skips = int(np.floor(N * max_skip_fraction))
    
    # Burgers parameters
    nu_burgers = 1.0
    D_burgers = 0.1  # MATCHED to KPZ for fair comparison
    
    # KPZ parameters
    nu_kpz = 1.0
    lambda_kpz = 1.0  # Cole-Hopf: λ=1 for equivalence
    D_kpz = 0.1  # MATCHED to Burgers
    
    # Scales (extended to b=8)
    scales = [1, 2, 4, 8]
    
    print(f"\nParameters:")
    print(f"  Burgers: nu={nu_burgers}, D={D_burgers}")
    print(f"  KPZ: nu={nu_kpz}, lambda={lambda_kpz}, D={D_kpz}")
    print(f"  MATCHED NOISE: D_burgers = D_kpz = {D_burgers}")
    print(f"  L={L}, T={T}, N={N} fields per system")
    print(f"  Scales: {scales}")
    print(f"  Observable: Scale-free structure functions (height space)")
    print(f"    - RG rescaling: h_rg = coarse_grain(h) / b^(alpha), alpha=0.5")
    print(f"    - Features: slope alpha + normalized S2 ratios (scale-free)")
    print(f"    - Burgers: u evolved with gradient noise; convert to h via spectral integration")
    print(f"    - KPZ: height h directly")
    print(f"\n  EXPECTATION: Burgers->KPZ should CONVERGE (negative slope)")
    
    # ========================================================================
    # Generate Data
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Generating Burgers velocity fields (u), then converting to height h...")
    print("=" * 70)
    
    import time
    t0 = time.time()
    
    burgers_fields = []
    burgers_skipped = 0
    for i in range(N):
        u, ok = generate_field_with_retries(
            lambda: simulate_burgers(L, T, dt_burgers, nu_burgers, D_burgers),
            max_retries,
            "Burgers",
            i + 1
        )
        if ok:
            h = velocity_to_height(u)
            burgers_fields.append(h)
        else:
            burgers_skipped += 1
            if burgers_skipped > max_skips:
                print(f"\n❌ ERROR: Too many skipped Burgers fields ({burgers_skipped}/{N})")
                return
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {len(burgers_fields)}/{N} Burgers fields (skipped: {burgers_skipped})...")
    
    t1 = time.time()
    print(f"✓ Generated {len(burgers_fields)} Burgers fields in {t1-t0:.1f}s")
    
    print("\n" + "=" * 70)
    print("Generating KPZ height fields...")
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
                print(f"\n❌ ERROR: Too many skipped KPZ fields ({kpz_skipped}/{N})")
                return
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {len(kpz_fields)}/{N} KPZ fields (skipped: {kpz_skipped})...")
    
    t1 = time.time()
    print(f"✓ Generated {len(kpz_fields)} KPZ fields in {t1-t0:.1f}s")
    
    # ========================================================================
    # Extract Features at Each Scale
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Extracting scale-free structure function features (height space)...")
    print("=" * 70)
    
    features_dict = {
        'burgers': {b: [] for b in scales},
        'kpz': {b: [] for b in scales}
    }
    valid_counts = {
        'burgers': {b: 0 for b in scales},
        'kpz': {b: 0 for b in scales},
    }
    
    for system, fields in [('burgers', burgers_fields), ('kpz', kpz_fields)]:
        for b in scales:
            for h in fields:
                h_coarse = spectral_coarse_grain(h, b, alpha=0.5)
                features, diagnostic = compute_scale_free_structure_functions(h_coarse)
                if features is not None:
                    features_dict[system][b].append(features)
                    valid_counts[system][b] += 1
        
        print(f"\n{system.upper()}: {len(features_dict[system][1])} samples per scale")
    
    # Convert to arrays
    for system in ['burgers', 'kpz']:
        for b in scales:
            features_dict[system][b] = np.array(features_dict[system][b])
    
    min_valid = min([valid_counts[s][b] for s in ['burgers', 'kpz'] for b in scales])
    if min_valid < 50:
        print(f"\n❌ ERROR: Insufficient valid samples (min={min_valid})")
        return

    print(f"\n✓ Feature extraction complete (min valid samples: {min_valid})")
    
    # ========================================================================
    # Estimate Bandwidth
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Estimating MMD bandwidth σ...")
    print("=" * 70)
    
    sigma = estimate_sigma(features_dict['burgers'][1], features_dict['kpz'][1])
    print(f"σ = {sigma:.4f} (median pairwise distance at b=1)")
    
    # ========================================================================
    # Diagnostic Gate
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC GATE: Testing system-vs-system invariance")
    print("=" * 70)
    
    burgers_pass, kpz_pass, burgers_slope, kpz_slope, burgers_dists, kpz_dists = \
        diagnostic_gate(features_dict, scales, sigma)
    
    print(f"\nBurgers-vs-Burgers (half vs half):")
    for i, b in enumerate(scales):
        print(f"  b={b}: distance = {burgers_dists[i]:.4f}")
    print(f"  → Slope: {burgers_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if burgers_pass else '❌ FAIL'}")
    
    print(f"\nKPZ-vs-KPZ (half vs half):")
    for i, b in enumerate(scales):
        print(f"  b={b}: distance = {kpz_dists[i]:.4f}")
    print(f"  → Slope: {kpz_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if kpz_pass else '❌ FAIL'}")
    
    gate_passed = burgers_pass and kpz_pass
    
    print("\n" + "=" * 70)
    if gate_passed:
        print("✅ DIAGNOSTIC GATE: PASSED")
        print("Both Burgers and KPZ observables are scale-invariant.")
        print("Burgers-vs-KPZ trend is interpretable.")
    else:
        print("❌ DIAGNOSTIC GATE: FAILED")
        print("Observables are not scale-invariant.")
        print("Cannot interpret Burgers-vs-KPZ trend.")
    print("=" * 70)
    
    # ========================================================================
    # Compute Burgers-vs-KPZ Distance
    # ========================================================================
    
    if gate_passed:
        print("\n" + "=" * 70)
        print("Computing Burgers-vs-KPZ distances...")
        print("=" * 70)
        
        distances = []
        for b in scales:
            d = mmd_distance(features_dict['burgers'][b], features_dict['kpz'][b], sigma)
            distances.append(d)
            print(f"  b={b}: d(Burgers, KPZ) = {d:.4f}")
        
        # Fit slope
        log_b = np.log2(scales)
        result = linregress(log_b, distances)
        slope = result.slope
        
        # Relative change
        rel_change = 100 * (distances[-1] - distances[0]) / distances[0]
        
        print(f"\nSlope: {slope:+.6f}")
        print(f"Relative change: {rel_change:+.2f}%")
        
        # Interpretation
        print("\n" + "=" * 70)
        if slope < -0.01:
            print("✅ INTERPRETATION: Burgers CONVERGES to KPZ")
            print(f"Distance decreases by {-rel_change:.1f}% → convergence confirmed")
            print("⚠️  This validates the framework can detect true convergence!")
        elif abs(slope) < 0.01:
            print("⚠️ INTERPRETATION: Distance flat (unexpected)")
            print("Burgers and KPZ should converge (Cole-Hopf equivalence)")
        else:
            print("⚠️ INTERPRETATION: Distance increases (divergence)")
            print("Unexpected - Burgers should approach KPZ")
        print("=" * 70)
    else:
        distances = [np.nan] * len(scales)
        slope = np.nan
        rel_change = np.nan
        
        print("\n⚠️ Skipping Burgers-vs-KPZ distance (gate failed)")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    output_dir = Path("results_exp50o")
    output_dir.mkdir(exist_ok=True)
    
    # Metadata
    metadata = {
        'experiment': '50o',
        'description': 'Burgers -> KPZ positive control using scale-free structure functions in height space',
        'note': 'Matched noise (D=0.1), gradient noise d_x(eta) in Burgers, h from u via spectral integration, RG rescaling h/b^alpha',
        'parameters': {
            'L': L, 'T': T, 'N': N,
            'burgers': {'nu': nu_burgers, 'D': D_burgers},
            'kpz': {'nu': nu_kpz, 'lambda': lambda_kpz, 'D': D_kpz},
            'scales': scales,
        },
        'diagnostic_gate': {
            'passed': bool(gate_passed),
            'burgers_invariance_pass': bool(burgers_pass),
            'kpz_invariance_pass': bool(kpz_pass),
            'burgers_slope': float(burgers_slope),
            'kpz_slope': float(kpz_slope),
            'threshold': 0.005,
        },
        'burgers_vs_kpz': {
            'distances': [float(d) for d in distances],
            'slope': float(slope) if not np.isnan(slope) else None,
            'relative_change_percent': float(rel_change) if not np.isnan(rel_change) else None,
        },
        'sigma': float(sigma),
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature data
    np.savez(
        output_dir / 'burgers_vs_kpz_scale_free_structure_functions.npz',
        burgers_features_b1=features_dict['burgers'][1],
        burgers_features_b2=features_dict['burgers'][2],
        burgers_features_b4=features_dict['burgers'][4],
        burgers_features_b8=features_dict['burgers'][8],
        kpz_features_b1=features_dict['kpz'][1],
        kpz_features_b2=features_dict['kpz'][2],
        kpz_features_b4=features_dict['kpz'][4],
        kpz_features_b8=features_dict['kpz'][8],
        scales=scales,
        distances=distances,
        sigma=sigma,
    )
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # ========================================================================
    # Visualize
    # ========================================================================
    
    log_b = np.log2(scales)  # Define here for plotting
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Diagnostic gate: Burgers-vs-Burgers
    axes[0].plot(log_b, burgers_dists, 'o-', label='Burgers-vs-Burgers', color='blue')
    axes[0].axhline(burgers_dists[0], color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('log₂(scale b)')
    axes[0].set_ylabel('MMD distance')
    axes[0].set_title(f'Diagnostic: Burgers-vs-Burgers\nSlope: {burgers_slope:+.6f} ({"✅ PASS" if burgers_pass else "❌ FAIL"})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Diagnostic gate: KPZ-vs-KPZ
    axes[1].plot(log_b, kpz_dists, 'o-', label='KPZ-vs-KPZ', color='green')
    axes[1].axhline(kpz_dists[0], color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('log₂(scale b)')
    axes[1].set_ylabel('MMD distance')
    axes[1].set_title(f'Diagnostic: KPZ-vs-KPZ\nSlope: {kpz_slope:+.6f} ({"✅ PASS" if kpz_pass else "❌ FAIL"})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Main result: Burgers-vs-KPZ
    if gate_passed:
        axes[2].plot(log_b, distances, 'o-', label='Burgers-vs-KPZ', color='red', linewidth=2)
        axes[2].axhline(distances[0], color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('log₂(scale b)')
        axes[2].set_ylabel('MMD distance')
        status = "CONVERGES ✅" if slope < -0.01 else "FLAT ⚠️" if abs(slope) < 0.01 else "DIVERGES ⚠️"
        axes[2].set_title(f'Burgers-vs-KPZ Distance\nSlope: {slope:+.6f} ({status})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, '❌ GATE FAILED\nResults not interpretable', 
                     ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_xlabel('log₂(scale b)')
        axes[2].set_ylabel('MMD distance')
        axes[2].set_title('Burgers-vs-KPZ Distance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'burgers_vs_kpz_scale_free_structure_functions.png', dpi=150)
    print(f"✓ Figure saved to {output_dir}/burgers_vs_kpz_scale_free_structure_functions.png")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 50o SUMMARY")
    print("=" * 70)
    print("Observable: Scale-free structure functions (height space)")
    if gate_passed:
        print(f"Diagnostic gate: ✅ PASSED")
        print(f"  Burgers-vs-Burgers slope: {burgers_slope:+.6f} (pass)")
        print(f"  KPZ-vs-KPZ slope: {kpz_slope:+.6f} (pass)")
        print(f"Burgers-vs-KPZ slope: {slope:+.6f} ({rel_change:+.1f}%)")
        if slope < -0.01:
            print("→ Burgers CONVERGES to KPZ (positive control validated) ✅")
        else:
            print("→ Unexpected result (should converge) ⚠️")
    else:
        print(f"Diagnostic gate: ❌ FAILED")
        print("→ Results not interpretable")
    print("=" * 70)

if __name__ == '__main__':
    main()
