"""
Experiment 50r: KPZ Exponent-Only Positive Control (α-only)
=============================================================

PURPOSE:
Test universality with a truly universal observable: the roughness exponent α.
This is the decisive experiment to resolve ambiguity from Exp 50q.

MOTIVATION:
Scale-free structure functions (Exp 50n/50o/50q) pass the gate but encode
non-universal amplitudes. The normalized ratios S₂(r)/S₂(r_ref) preserve
parameter-dependent correction terms B/A that cause systems in the same
universality class to fail convergence.

SOLUTION:
Use ONLY the roughness exponent α = slope of log(S₂) vs log(r).
This is the canonical universal quantity in surface growth.

SYSTEMS:
- KPZ-A: (ν=1.0, λ=1.0, D=0.1) - baseline
- KPZ-B: (ν=1.0, λ=0.5, D=0.05) - reduced coupling
- Control: KS vs KPZ (different universality classes)

THEORY:
- KPZ: α = 1/2 in 1D (Family-Vicsek scaling)
- KS: α ≈ 1.0 (different universality class)
- Both KPZ-A and KPZ-B should have α ≈ 0.5

EXPECTED RESULTS:
- KPZ-A vs KPZ-B: Distance flat or decreasing (convergence within class)
- KS vs KPZ: Distance non-zero (different exponents)
- Gate passes for all systems (α should be scale-invariant)

CONTRASTS WITH:
- Exp 50q: Multi-dimensional structure function features → distance increases
- This: 1D exponent-only → should show universality

METHOD:
- Observable: 1D feature [α] where α = slope of log(S₂) vs log(r)
- RG rescaling: h_rg = h_coarse / b^0.5
- Diagnostic gate: A-vs-A and B-vs-B must be flat
- Distance: MMD on 1D feature space
- Also report mean ± std of α per system (direct comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import linregress
import json
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

PILOT = False  # Set via --pilot flag

def get_config(pilot=False):
    if pilot:
        return {
            'N': 200,
            'T': 300,
            'scales': [1, 2, 4],
            'workers': min(4, os.cpu_count() - 1),
            'checkpoint_interval': 25,
            'output_dir': 'results_exp50r_pilot'
        }
    else:
        return {
            'N': 500,
            'T': 600,
            'scales': [1, 2, 4, 8],
            'workers': min(6, os.cpu_count() - 1),
            'checkpoint_interval': 50,
            'output_dir': 'results_exp50r'
        }

# Common parameters
L = 256
dt = 0.01
max_retries = 3
max_skip_fraction = 0.1

# KPZ systems
params_KPZ_A = {'nu': 1.0, 'lambda': 1.0, 'D': 0.1, 'label': 'KPZ_A', 'type': 'kpz'}
params_KPZ_B = {'nu': 1.0, 'lambda': 0.5, 'D': 0.05, 'label': 'KPZ_B', 'type': 'kpz'}

# KS system (control)
params_KS = {'nu': 1.0, 'label': 'KS', 'type': 'ks'}

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_kpz(L, T, dt, nu, lam, D, seed=None):
    """
    KPZ: dh/dt = nu*d²h/dx² + (lambda/2)*(dh/dx)² + eta
    Returns height field h.
    """
    if seed is not None:
        np.random.seed(seed)
    
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
    etd_coef[~mask_nonzero] = dt
    
    # De-aliasing mask
    k_max = np.max(np.abs(k))
    dealias = np.abs(k) < (2.0/3.0) * k_max
    
    nsteps = int(T / dt)
    for step in range(nsteps):
        h_hat = fft(h)
        
        grad_h = ifft(1j * k * h_hat).real
        
        nonlinear = 0.5 * lam * grad_h**2
        nonlinear_hat = fft(nonlinear) * dealias
        
        noise = np.random.randn(N) * np.sqrt(2 * D / dt)
        noise = noise - noise.mean()
        noise_hat = fft(noise)
        
        h_hat_new = exp_factor * h_hat + etd_coef * (nonlinear_hat + noise_hat)
        h = ifft(h_hat_new).real
        
        if step % 500 == 0 and not np.all(np.isfinite(h)):
            return None
    
    h = h - h.mean()
    
    if not np.all(np.isfinite(h)) or np.max(np.abs(h)) > 1e3:
        return None
    
    return h

def simulate_ks(L, T, dt, nu, seed=None):
    """
    Kuramoto-Sivashinsky: h_t = -h_xx - h_xxxx - (1/2)(h_x)²
    Returns height field h.
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = L
    dx = 1.0
    
    h = np.random.randn(N) * 0.01
    
    k = fftfreq(N, d=dx) * 2 * np.pi
    k2 = k**2
    k4 = k**4
    
    # Linear operator: -k² - k⁴
    linear = -k2 - k4
    
    # ETD for KS
    exp_factor = np.exp(linear * dt)
    etd_coef = np.zeros_like(linear)
    mask_nonzero = np.abs(linear) > 1e-10
    etd_coef[mask_nonzero] = (exp_factor[mask_nonzero] - 1.0) / linear[mask_nonzero]
    etd_coef[~mask_nonzero] = dt
    
    # De-aliasing
    k_max = np.max(np.abs(k))
    dealias = np.abs(k) < (2.0/3.0) * k_max
    
    nsteps = int(T / dt)
    for step in range(nsteps):
        h_hat = fft(h)
        
        grad_h = ifft(1j * k * h_hat).real
        nonlinear = -0.5 * grad_h**2
        nonlinear_hat = fft(nonlinear) * dealias
        
        h_hat_new = exp_factor * h_hat + etd_coef * nonlinear_hat
        h = ifft(h_hat_new).real
        
        if step % 500 == 0 and not np.all(np.isfinite(h)):
            return None
    
    h = h - h.mean()
    
    if not np.all(np.isfinite(h)) or np.max(np.abs(h)) > 1e3:
        return None
    
    return h

# ============================================================================
# SPECTRAL COARSE-GRAINING
# ============================================================================

def spectral_coarse_grain(h, b, alpha=0.5):
    """
    Field-level spectral coarse-graining with RG rescaling.
    h_rg = h_coarse / b^alpha
    """
    if b == 1:
        h = h - h.mean()
        return h
    
    N = len(h)
    h = h - h.mean()
    
    h_hat = fft(h)
    k = fftfreq(N)
    
    h_hat[0] = 0  # Remove k=0 mode
    
    k_cutoff = 0.5 / b
    h_hat[np.abs(k) > k_cutoff] = 0
    
    h_coarse = ifft(h_hat).real
    
    h_rg = h_coarse / (b ** alpha)
    h_rg = h_rg - h_rg.mean()
    
    return h_rg

# ============================================================================
# ALPHA-ONLY FEATURE EXTRACTION
# ============================================================================

def compute_alpha_only(h, lag_fractions=(1/32, 1/16, 1/8, 1/4, 1/2)):
    """
    Extract ONLY the roughness exponent α from structure function.
    
    - S₂(r) = <|h(x+r) - h(x)|²>
    - Fit: log(S₂) = 2α * log(r) + const
    - Return: [α] (1D feature)
    
    This is the truly universal observable - no amplitude ratios.
    """
    N = len(h)
    
    lags = []
    for frac in lag_fractions:
        r = max(1, int(round(frac * N)))
        if r < N // 2:
            lags.append(r)
    
    if len(lags) < 3:
        return None, "Insufficient valid lags"
    
    S2_values = []
    for r in lags:
        diffs = h[r:] - h[:-r]
        S2 = np.mean(diffs ** 2)
        S2_values.append(S2)
    
    S2_values = np.array(S2_values)
    lags = np.array(lags)
    
    if np.any(S2_values <= 0):
        return None, "Non-positive S2 values"
    
    log_r = np.log(lags)
    log_S2 = np.log(S2_values)
    
    if not np.all(np.isfinite(log_S2)):
        return None, "Non-finite log(S2)"
    
    # Fit: log(S2) = 2α * log(r) + const
    result = linregress(log_r, log_S2)
    alpha = result.slope / 2  # S₂ ~ r^(2α), so slope = 2α
    
    # Return 1D feature
    return np.array([alpha]), None

# ============================================================================
# PARALLEL FIELD GENERATION
# ============================================================================

def generate_single_field(args):
    """Worker function for parallel field generation."""
    system_params, L, T, dt, index, base_seed = args
    
    seed = base_seed + index
    
    for attempt in range(max_retries):
        if system_params['type'] == 'kpz':
            h = simulate_kpz(
                L, T, dt,
                system_params['nu'],
                system_params['lambda'],
                system_params['D'],
                seed=seed + attempt * 10000
            )
        else:  # KS
            h = simulate_ks(
                L, T, dt,
                system_params['nu'],
                seed=seed + attempt * 10000
            )
        
        if h is not None and np.all(np.isfinite(h)):
            return h, True, system_params['label'], index
    
    return None, False, system_params['label'], index

def generate_fields_parallel(system_params, N, L, T, dt, workers, base_seed, config):
    """Generate N fields in parallel with checkpointing."""
    checkpoint_dir = Path(config['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"{system_params['label']}_checkpoint.npz"
    
    # Resume from checkpoint
    if checkpoint_file.exists():
        print(f"  Found checkpoint for {system_params['label']}, loading...")
        checkpoint = np.load(checkpoint_file)
        fields = list(checkpoint['fields'])
        start_idx = len(fields)
        print(f"  Resuming from {start_idx}/{N} fields")
    else:
        fields = []
        start_idx = 0
    
    if start_idx >= N:
        print(f"  {system_params['label']} already complete ({len(fields)}/{N})")
        return fields[:N]
    
    skipped = 0
    max_skips = int(N * max_skip_fraction)
    
    tasks = [
        (system_params, L, T, dt, i, base_seed)
        for i in range(start_idx, N + max_skips)
    ]
    
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_single_field, task): task 
                   for task in tasks[:N - start_idx + 10]}
        
        completed = 0
        for future in as_completed(futures):
            h, success, label, idx = future.result()
            
            if success:
                fields.append(h)
                completed += 1
                
                if completed % config['checkpoint_interval'] == 0:
                    elapsed = time.time() - t_start
                    rate = completed / elapsed
                    remaining = (N - start_idx - completed) / rate if rate > 0 else 0
                    np.savez(checkpoint_file, fields=np.array(fields))
                    print(f"  [{label}] {len(fields)}/{N} fields (ETA: {remaining/60:.1f} min)")
                
                if len(fields) >= N:
                    break
                    
            else:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f"Too many skipped fields ({skipped}/{N})")
    
    # Final checkpoint
    np.savez(checkpoint_file, fields=np.array(fields[:N]))
    
    return fields[:N]

# ============================================================================
# MMD DISTANCE
# ============================================================================

def mmd_distance(X, Y, sigma):
    """Maximum Mean Discrepancy with RBF kernel."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    XX = rbf_kernel(X, X, gamma=1.0 / (2 * sigma**2))
    YY = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma**2))
    XY = rbf_kernel(X, Y, gamma=1.0 / (2 * sigma**2))
    
    mmd2 = XX.mean() + YY.mean() - 2 * XY.mean()
    return np.sqrt(max(0, mmd2))

def estimate_sigma(X, Y):
    """Estimate bandwidth as median pairwise distance."""
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    combined = np.vstack([X[:100], Y[:100]])
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(combined)
    sigma = np.median(dists[dists > 0])
    return sigma

# ============================================================================
# DIAGNOSTIC GATE
# ============================================================================

def diagnostic_gate(features_dict, scales, sigma, system_key, threshold=0.005):
    """Test system-vs-system invariance (split-half test)."""
    distances = []
    
    for b in scales:
        feats = features_dict[system_key][b]
        mid = len(feats) // 2
        
        d = mmd_distance(feats[:mid], feats[mid:], sigma)
        distances.append(d)
    
    log_b = np.log2(scales)
    result = linregress(log_b, distances)
    slope = result.slope
    
    passed = abs(slope) < threshold
    
    return passed, slope, distances

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run in pilot mode')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-ks', action='store_true', help='Skip KS control')
    args = parser.parse_args()
    
    config = get_config(pilot=args.pilot)
    
    print("=" * 70)
    print("Experiment 50r: KPZ Exponent-Only Positive Control (alpha-only)")
    if args.pilot:
        print("               [PILOT MODE - Reduced Parameters]")
    print("=" * 70)
    
    N = config['N']
    T = config['T']
    scales = config['scales']
    workers = config['workers']
    
    print(f"\nParameters:")
    print(f"  KPZ-A: nu={params_KPZ_A['nu']}, lambda={params_KPZ_A['lambda']}, D={params_KPZ_A['D']}")
    print(f"  KPZ-B: nu={params_KPZ_B['nu']}, lambda={params_KPZ_B['lambda']}, D={params_KPZ_B['D']}")
    if not args.no_ks:
        print(f"  KS (control): nu={params_KS['nu']}")
    print(f"  L={L}, T={T}, dt={dt}, N={N} fields per system")
    print(f"  Scales: {scales}")
    print(f"  Workers: {workers}")
    print(f"  Observable: 1D [alpha] only (purely universal)")
    print(f"  RG rescaling: h_rg = coarse_grain(h) / b^0.5")
    print(f"\n  KPZ-A vs KPZ-B: Expected to CONVERGE (same alpha ~0.5)")
    if not args.no_ks:
        print(f"  KS vs KPZ: Expected to SEPARATE (different alpha)")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # Generate Fields
    # ========================================================================
    
    all_systems = [params_KPZ_A, params_KPZ_B]
    if not args.no_ks:
        all_systems.append(params_KS)
    
    all_fields = {}
    
    for system_params in all_systems:
        print("\n" + "=" * 70)
        print(f"Generating {system_params['label']} fields (parallel)...")
        print("=" * 70)
        
        base_seed = {
            'KPZ_A': 42,
            'KPZ_B': 100000,
            'KS': 200000
        }[system_params['label']]
        
        t0 = time.time()
        fields = generate_fields_parallel(
            system_params, N, L, T, dt, workers, base_seed, config
        )
        t1 = time.time()
        
        all_fields[system_params['label']] = fields
        print(f"✓ Generated {len(fields)} {system_params['label']} fields in {t1-t0:.1f}s")
    
    # ========================================================================
    # Extract Alpha Features
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Extracting alpha-only features...")
    print("=" * 70)
    
    features_dict = {label: {b: [] for b in scales} for label in all_fields.keys()}
    alpha_stats = {label: {b: [] for b in scales} for label in all_fields.keys()}
    
    for label, fields in all_fields.items():
        for b in scales:
            for h in fields:
                h_coarse = spectral_coarse_grain(h, b, alpha=0.5)
                features, diagnostic = compute_alpha_only(h_coarse)
                if features is not None:
                    features_dict[label][b].append(features)
                    alpha_stats[label][b].append(features[0])
        
        # Report alpha statistics
        b1_alphas = np.array(alpha_stats[label][1])
        print(f"\n{label}: alpha = {b1_alphas.mean():.4f} +/- {b1_alphas.std():.4f} (N={len(b1_alphas)})")
    
    # Convert to arrays
    for label in features_dict:
        for b in scales:
            features_dict[label][b] = np.array(features_dict[label][b])
    
    print("\n✓ Feature extraction complete")
    
    # ========================================================================
    # Estimate Bandwidth
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Estimating MMD bandwidth σ...")
    print("=" * 70)
    
    sigma = estimate_sigma(features_dict['KPZ_A'][1], features_dict['KPZ_B'][1])
    print(f"σ = {sigma:.6f} (median pairwise distance at b=1)")
    
    # ========================================================================
    # Diagnostic Gate
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC GATE: Testing system-vs-system invariance")
    print("=" * 70)
    
    gate_results = {}
    all_gate_passed = True
    
    for label in features_dict.keys():
        passed, slope, distances = diagnostic_gate(features_dict, scales, sigma, label)
        gate_results[label] = {'passed': passed, 'slope': slope, 'distances': distances}
        
        print(f"\n{label}-vs-{label} (half vs half):")
        for i, b in enumerate(scales):
            print(f"  b={b}: distance = {distances[i]:.6f}")
        print(f"  → Slope: {slope:+.6f}")
        print(f"  → Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if not passed:
            all_gate_passed = False
    
    print("\n" + "=" * 70)
    if all_gate_passed:
        print("✅ DIAGNOSTIC GATE: ALL PASSED")
        print("All systems have scale-invariant alpha.")
    else:
        print("❌ DIAGNOSTIC GATE: FAILED")
        print("Some systems have scale-dependent alpha.")
    print("=" * 70)
    
    # ========================================================================
    # Compute Cross-System Distances
    # ========================================================================
    
    cross_distances = {}
    
    if all_gate_passed:
        # KPZ-A vs KPZ-B
        print("\n" + "=" * 70)
        print("Computing KPZ-A vs KPZ-B distances...")
        print("=" * 70)
        
        AB_distances = []
        for b in scales:
            d = mmd_distance(features_dict['KPZ_A'][b], features_dict['KPZ_B'][b], sigma)
            AB_distances.append(d)
            print(f"  b={b}: d(KPZ-A, KPZ-B) = {d:.6f}")
        
        log_b = np.log2(scales)
        AB_result = linregress(log_b, AB_distances)
        AB_slope = AB_result.slope
        AB_change = (AB_distances[-1] - AB_distances[0]) / AB_distances[0] * 100 if AB_distances[0] > 0 else 0
        
        print(f"\nSlope: {AB_slope:+.6f}")
        print(f"Relative change: {AB_change:+.2f}%")
        
        cross_distances['KPZ_A_vs_KPZ_B'] = {
            'distances': AB_distances,
            'slope': AB_slope,
            'relative_change': AB_change
        }
        
        # Interpretation
        print("\n" + "=" * 70)
        if AB_slope < -0.0005:
            print("✅ CONVERGENCE: KPZ-A and KPZ-B converge (as expected for same class)")
        elif abs(AB_slope) < 0.0005:
            print("✅ FLAT: Alpha distributions overlap (same universality class)")
        else:
            print("⚠️ DIVERGENCE: Distance increases (unexpected)")
        print("=" * 70)
        
        # KS vs KPZ (control)
        if not args.no_ks and 'KS' in features_dict:
            print("\n" + "=" * 70)
            print("Computing KS vs KPZ-A distances (control)...")
            print("=" * 70)
            
            KS_distances = []
            for b in scales:
                d = mmd_distance(features_dict['KS'][b], features_dict['KPZ_A'][b], sigma)
                KS_distances.append(d)
                print(f"  b={b}: d(KS, KPZ-A) = {d:.6f}")
            
            log_b = np.log2(scales)
            KS_result = linregress(log_b, KS_distances)
            KS_slope = KS_result.slope
            
            print(f"\nSlope: {KS_slope:+.6f}")
            
            cross_distances['KS_vs_KPZ_A'] = {
                'distances': KS_distances,
                'slope': KS_slope
            }
            
            # Report alpha difference
            ks_alpha = np.mean(alpha_stats['KS'][1])
            kpz_alpha = np.mean(alpha_stats['KPZ_A'][1])
            print(f"\nAlpha comparison:")
            print(f"  KS: alpha = {ks_alpha:.4f}")
            print(f"  KPZ-A: alpha = {kpz_alpha:.4f}")
            print(f"  Difference: {abs(ks_alpha - kpz_alpha):.4f}")
            
            print("\n" + "=" * 70)
            if np.mean(KS_distances) > np.mean(AB_distances):
                print("✅ KS-vs-KPZ separated more than KPZ-A-vs-KPZ-B (as expected)")
            else:
                print("⚠️ Separation comparison unclear")
            print("=" * 70)
    
    else:
        print("\n⚠️ Skipping cross-system distances (gate failed)")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    # Compile alpha statistics
    alpha_summary = {}
    for label in alpha_stats:
        b1_alphas = np.array(alpha_stats[label][1])
        alpha_summary[label] = {
            'mean': float(b1_alphas.mean()),
            'std': float(b1_alphas.std()),
            'n': int(len(b1_alphas))
        }
    
    metadata = {
        'experiment': '50r',
        'description': 'KPZ exponent-only positive control (alpha-only)',
        'note': 'Truly universal observable: roughness exponent alpha from S2 scaling',
        'pilot_mode': args.pilot,
        'parameters': {
            'L': L,
            'T': T,
            'N': N,
            'scales': scales,
            'workers': workers,
            'KPZ_A': {'nu': params_KPZ_A['nu'], 'lambda': params_KPZ_A['lambda'], 'D': params_KPZ_A['D']},
            'KPZ_B': {'nu': params_KPZ_B['nu'], 'lambda': params_KPZ_B['lambda'], 'D': params_KPZ_B['D']},
            'KS': {'nu': params_KS['nu']} if not args.no_ks else None
        },
        'alpha_statistics': alpha_summary,
        'diagnostic_gate': {
            'all_passed': bool(all_gate_passed),
            **{label: {'passed': bool(gate_results[label]['passed']), 
                       'slope': float(gate_results[label]['slope'])}
               for label in gate_results}
        },
        'cross_distances': {
            k: {
                'distances': [float(d) for d in v['distances']],
                'slope': float(v['slope']),
                'relative_change': float(v.get('relative_change', 0))
            }
            for k, v in cross_distances.items()
        },
        'sigma': float(sigma)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save features
    save_dict = {'scales': scales, 'sigma': sigma}
    for label in features_dict:
        for b in scales:
            save_dict[f'{label}_b{b}'] = features_dict[label][b]
    
    np.savez(output_dir / 'kpzA_vs_kpzB_alpha_only.npz', **save_dict)
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # ========================================================================
    # Plot Results
    # ========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Alpha distributions
    ax1 = axes[0]
    colors = {'KPZ_A': 'C0', 'KPZ_B': 'C1', 'KS': 'C2'}
    
    for label in alpha_stats:
        alphas = np.array(alpha_stats[label][1])
        ax1.hist(alphas, bins=30, alpha=0.5, label=f'{label} (mean={alphas.mean():.3f})', 
                 color=colors.get(label, 'C3'))
        ax1.axvline(alphas.mean(), color=colors.get(label, 'C3'), linestyle='--', linewidth=2)
    
    ax1.axvline(0.5, color='black', linestyle=':', linewidth=2, label='KPZ theory (alpha=0.5)')
    ax1.set_xlabel('Roughness Exponent alpha', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Alpha Distributions (b=1)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Right: Distances vs scale
    ax2 = axes[1]
    log_b = np.log2(scales)
    
    # Gate tests
    for label in gate_results:
        dists = gate_results[label]['distances']
        slope = gate_results[label]['slope']
        ax2.plot(log_b, dists, 'o--', alpha=0.5, 
                 label=f'{label}-vs-{label} ({slope:+.5f})', markersize=6)
    
    # Cross-system
    if 'KPZ_A_vs_KPZ_B' in cross_distances:
        dists = cross_distances['KPZ_A_vs_KPZ_B']['distances']
        slope = cross_distances['KPZ_A_vs_KPZ_B']['slope']
        ax2.plot(log_b, dists, 'o-', color='C4', linewidth=2, markersize=8,
                 label=f'KPZ-A-vs-KPZ-B ({slope:+.5f})', zorder=10)
    
    if 'KS_vs_KPZ_A' in cross_distances:
        dists = cross_distances['KS_vs_KPZ_A']['distances']
        slope = cross_distances['KS_vs_KPZ_A']['slope']
        ax2.plot(log_b, dists, 's-', color='C5', linewidth=2, markersize=8,
                 label=f'KS-vs-KPZ-A ({slope:+.5f})', zorder=10)
    
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('log2(b)', fontsize=12)
    ax2.set_ylabel('MMD Distance', fontsize=12)
    
    title = 'Exp 50r: Alpha-Only Distances'
    if args.pilot:
        title += ' [PILOT]'
    ax2.set_title(title, fontsize=14)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3)
    
    # Status annotation
    if all_gate_passed and 'KPZ_A_vs_KPZ_B' in cross_distances:
        AB_slope = cross_distances['KPZ_A_vs_KPZ_B']['slope']
        if AB_slope < -0.0005:
            status = 'CONVERGE'
            color = 'green'
        elif abs(AB_slope) < 0.0005:
            status = 'FLAT'
            color = 'blue'
        else:
            status = 'DIVERGE'
            color = 'orange'
        ax2.text(0.98, 0.98, f'Gate: PASS\nKPZ-A vs B: {status}', 
                 transform=ax2.transAxes, ha='right', va='top', fontsize=11, 
                 color=color, weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    elif not all_gate_passed:
        ax2.text(0.98, 0.98, 'Gate: FAIL', transform=ax2.transAxes, 
                 ha='right', va='top', fontsize=11, color='red', weight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kpzA_vs_kpzB_alpha_only.png', dpi=150)
    print(f"✓ Figure saved to {output_dir}/kpzA_vs_kpzB_alpha_only.png")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 50r SUMMARY")
    if args.pilot:
        print("[PILOT MODE]")
    print("=" * 70)
    print(f"Observable: 1D roughness exponent alpha only")
    print(f"\nAlpha statistics (b=1):")
    for label, stats in alpha_summary.items():
        print(f"  {label}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"\nDiagnostic gate: {'✅ ALL PASSED' if all_gate_passed else '❌ FAILED'}")
    
    if all_gate_passed:
        for label in gate_results:
            print(f"  {label}: slope {gate_results[label]['slope']:+.6f}")
        
        if 'KPZ_A_vs_KPZ_B' in cross_distances:
            AB = cross_distances['KPZ_A_vs_KPZ_B']
            print(f"\nKPZ-A vs KPZ-B: slope {AB['slope']:+.6f} ({AB['relative_change']:+.1f}%)")
            if AB['slope'] < -0.0005:
                print("→ Convergence detected ✅")
            elif abs(AB['slope']) < 0.0005:
                print("→ Distance flat (same universality) ✅")
            else:
                print("→ Distance increasing ⚠️")
        
        if 'KS_vs_KPZ_A' in cross_distances:
            KS = cross_distances['KS_vs_KPZ_A']
            print(f"\nKS vs KPZ-A: slope {KS['slope']:+.6f}")
            ks_mean = np.mean([d for d in KS['distances']])
            ab_mean = np.mean([d for d in cross_distances.get('KPZ_A_vs_KPZ_B', {}).get('distances', [0])])
            if ks_mean > ab_mean * 1.5:
                print("→ KS clearly separated from KPZ (different class) ✅")
            else:
                print("→ Separation comparison uncertain")
    
    print("=" * 70)
    
    if args.pilot and all_gate_passed:
        print("\n✅ PILOT PASSED - Ready for full run")
        print("Run without --pilot flag for complete experiment")

if __name__ == '__main__':
    main()
