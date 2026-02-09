"""
Experiment 50q: KPZ-vs-KPZ Positive Control (Within-Class Convergence Test)
============================================================================

PURPOSE:
Test whether different KPZ parameter regimes converge under coarse-graining.
This is a positive control to verify the framework can detect convergence
when systems ARE in the same universality class.

SYSTEMS:
- System A: KPZ (nu=1.0, lambda=1.0, D=0.1)  [baseline from 50n/50o]
- System B: KPZ (nu=1.0, lambda=0.5, D=0.05) [reduced coupling]

THEORY:
Both are KPZ (same universality class), so should converge under RG flow.
Distance should DECREASE with scale (negative slope).

CONTRASTS WITH:
- Exp 50n: KS-vs-KPZ (flat) - different classes
- Exp 50o: Burgers-vs-KPZ (flat) - deterministic equivalence but statistical difference
- This: KPZ-vs-KPZ (should converge) - same universality class

METHOD:
- Observable: Scale-free structure functions in height space (same as 50n)
- RG rescaling: h_rg = h_coarse / b^alpha (alpha=0.5)
- Diagnostic gate: Both A-vs-A and B-vs-B must be flat before interpreting A-vs-B
- Expectation: Negative slope (convergence within KPZ class)

PERFORMANCE OPTIMIZATIONS:
- Parallel field generation with ProcessPoolExecutor
- Checkpointing every 50 fields for resume capability
- Hard retry logic with abort if >10% skipped

PILOT MODE:
- Run with reduced N=200, T=300, scales=[1,2,4] for quick validation
- If gate passes, indicates full run will likely succeed
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

# Pilot mode: quick test with reduced parameters
PILOT = False  # Set to True for quick testing, False for full run

if PILOT:
    CONFIG = {
        'N': 200,
        'T': 300,
        'scales': [1, 2, 4],
        'workers': min(4, os.cpu_count() - 1),
        'checkpoint_interval': 25,
        'output_dir': 'results_exp50q_pilot'
    }
else:
    CONFIG = {
        'N': 500,
        'T': 600,
        'scales': [1, 2, 4, 8],
        'workers': min(6, os.cpu_count() - 1),
        'checkpoint_interval': 50,
        'output_dir': 'results_exp50q'
    }

# Common parameters
L = 256
dt = 0.01
max_retries = 3
max_skip_fraction = 0.1

# System A: baseline KPZ
params_A = {'nu': 1.0, 'lambda': 1.0, 'D': 0.1, 'label': 'KPZ_A'}

# System B: reduced coupling KPZ
params_B = {'nu': 1.0, 'lambda': 0.5, 'D': 0.05, 'label': 'KPZ_B'}

# ============================================================================
# KPZ SIMULATION (from Exp 50o)
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
    
    # Initial condition
    h = np.random.randn(N) * 0.01
    
    # Fourier setup
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
    
    # Time evolution
    nsteps = int(T / dt)
    for step in range(nsteps):
        h_hat = fft(h)
        
        # Gradient
        grad_h = ifft(1j * k * h_hat).real
        
        # Nonlinear term
        nonlinear = 0.5 * lam * grad_h**2
        nonlinear_hat = fft(nonlinear) * dealias
        
        # Conservative noise
        noise = np.random.randn(N) * np.sqrt(2 * D / dt)
        noise = noise - noise.mean()
        noise_hat = fft(noise)
        
        # ETD update
        h_hat_new = exp_factor * h_hat + etd_coef * (nonlinear_hat + noise_hat)
        h = ifft(h_hat_new).real
        
        # Check for instability
        if step % 500 == 0 and not np.all(np.isfinite(h)):
            return None
    
    h = h - h.mean()
    
    if not np.all(np.isfinite(h)) or np.max(np.abs(h)) > 1e3:
        return None
    
    return h

# ============================================================================
# SPECTRAL COARSE-GRAINING WITH RG RESCALING (from Exp 50o)
# ============================================================================

def spectral_coarse_grain(h, b, alpha=0.5):
    """
    Field-level spectral coarse-graining with RG rescaling for height.
    
    1. FFT -> low-pass filter -> iFFT
    2. Rescale: h_rg = h_coarse / b^alpha (alpha=0.5 for 1D KPZ)
    3. Ensure zero-mean and remove k=0 mode
    """
    if b == 1:
        h = h - h.mean()
        return h
    
    N = len(h)
    h = h - h.mean()
    
    h_hat = fft(h)
    k = fftfreq(N)
    
    # Remove k=0 mode
    h_hat[0] = 0
    
    # Low-pass filter
    k_cutoff = 0.5 / b
    h_hat[np.abs(k) > k_cutoff] = 0
    
    h_coarse = ifft(h_hat).real
    
    # RG rescaling
    h_rg = h_coarse / (b ** alpha)
    
    # Ensure zero-mean
    h_rg = h_rg - h_rg.mean()
    
    return h_rg

# ============================================================================
# SCALE-FREE STRUCTURE FUNCTIONS (from Exp 50n)
# ============================================================================

def compute_scale_free_structure_functions(h, lag_fractions=(1/32, 1/16, 1/8, 1/4, 1/2)):
    """
    Scale-free structure function features.
    
    - S2(r) = <|h(x+r) - h(x)|²>
    - Normalize: S2_norm(r) = S2(r) / S2(r_ref)
    - Features: [slope alpha, normalized ratios]
    
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
    
    # Feature vector: [alpha] + normalized ratios (skip first = 1)
    features = [alpha] + list(S2_norm[1:])
    
    return np.array(features), None

# ============================================================================
# PARALLEL FIELD GENERATION
# ============================================================================

def generate_single_field(args):
    """
    Worker function for parallel field generation.
    Returns (field, success, system_label, index)
    """
    system_params, L, T, dt, index, base_seed = args
    
    # Unique seed per field
    seed = base_seed + index
    
    for attempt in range(max_retries):
        h = simulate_kpz(
            L, T, dt,
            system_params['nu'],
            system_params['lambda'],
            system_params['D'],
            seed=seed + attempt * 10000
        )
        
        if h is not None and np.all(np.isfinite(h)):
            return h, True, system_params['label'], index
    
    return None, False, system_params['label'], index

def generate_fields_parallel(system_params, N, L, T, dt, workers, base_seed=42):
    """
    Generate N fields in parallel with checkpointing.
    """
    checkpoint_dir = Path(CONFIG['output_dir']) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_dir / f"{system_params['label']}_checkpoint.npz"
    
    # Check for existing checkpoint
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
    
    # Generate remaining fields
    skipped = 0
    max_skips = int(N * max_skip_fraction)
    
    tasks = [
        (system_params, L, T, dt, i, base_seed)
        for i in range(start_idx, N + max_skips)
    ]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(generate_single_field, task): task for task in tasks[:N - start_idx + 10]}
        
        completed = 0
        for future in as_completed(futures):
            h, success, label, idx = future.result()
            
            if success:
                fields.append(h)
                completed += 1
                
                # Checkpoint every N fields
                if completed % CONFIG['checkpoint_interval'] == 0:
                    np.savez(checkpoint_file, fields=np.array(fields))
                    print(f"  [{label}] {len(fields)}/{N} fields generated (checkpoint saved)")
                
                # Done?
                if len(fields) >= N:
                    break
                
                # Submit one more task to maintain queue
                if len(fields) + skipped < N + max_skips:
                    new_idx = start_idx + completed + skipped + 10
                    new_task = (system_params, L, T, dt, new_idx, base_seed)
                    futures[executor.submit(generate_single_field, new_task)] = new_task
                    
            else:
                skipped += 1
                if skipped > max_skips:
                    raise RuntimeError(f"Too many skipped fields ({skipped}/{N})")
                
                # Submit replacement task
                if len(fields) + skipped < N + max_skips:
                    new_idx = start_idx + completed + skipped
                    new_task = (system_params, L, T, dt, new_idx, base_seed)
                    futures[executor.submit(generate_single_field, new_task)] = new_task
    
    # Final checkpoint
    np.savez(checkpoint_file, fields=np.array(fields[:N]))
    
    return fields[:N]

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
    Both A-vs-A and B-vs-B must be flat.
    """
    A_distances = []
    B_distances = []
    
    for b in scales:
        feats_A = features_dict['A'][b]
        feats_B = features_dict['B'][b]
        
        mid_A = len(feats_A) // 2
        mid_B = len(feats_B) // 2
        
        # A-vs-A
        d_A = mmd_distance(feats_A[:mid_A], feats_A[mid_A:], sigma)
        A_distances.append(d_A)
        
        # B-vs-B
        d_B = mmd_distance(feats_B[:mid_B], feats_B[mid_B:], sigma)
        B_distances.append(d_B)
    
    # Fit slopes
    log_b = np.log2(scales)
    
    A_result = linregress(log_b, A_distances)
    A_slope = A_result.slope
    
    B_result = linregress(log_b, B_distances)
    B_slope = B_result.slope
    
    # Test against threshold
    A_pass = abs(A_slope) < threshold
    B_pass = abs(B_slope) < threshold
    
    return A_pass, B_pass, A_slope, B_slope, A_distances, B_distances

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    global PILOT
    
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run in pilot mode (quick test)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if exists')
    args = parser.parse_args()
    
    if args.pilot:
        PILOT = True
        # Reconfigure
        CONFIG['N'] = 200
        CONFIG['T'] = 300
        CONFIG['scales'] = [1, 2, 4]
        CONFIG['workers'] = min(4, os.cpu_count() - 1)
        CONFIG['checkpoint_interval'] = 25
        CONFIG['output_dir'] = 'results_exp50q_pilot'
    
    print("=" * 70)
    print("Experiment 50q: KPZ-vs-KPZ Positive Control")
    if PILOT:
        print("               [PILOT MODE - Reduced Parameters]")
    print("=" * 70)
    
    N = CONFIG['N']
    T = CONFIG['T']
    scales = CONFIG['scales']
    workers = CONFIG['workers']
    
    print(f"\nParameters:")
    print(f"  System A: nu={params_A['nu']}, lambda={params_A['lambda']}, D={params_A['D']}")
    print(f"  System B: nu={params_B['nu']}, lambda={params_B['lambda']}, D={params_B['D']}")
    print(f"  L={L}, T={T}, dt={dt}, N={N} fields per system")
    print(f"  Scales: {scales}")
    print(f"  Workers: {workers}")
    print(f"  Observable: Scale-free structure functions (height space)")
    print(f"  RG rescaling: h_rg = coarse_grain(h) / b^0.5")
    print(f"  Expectation: Negative slope (KPZ-A and KPZ-B should converge)")
    
    # Output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # Generate Fields (Parallel)
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Generating KPZ-A fields (parallel)...")
    print("=" * 70)
    
    t0 = time.time()
    fields_A = generate_fields_parallel(params_A, N, L, T, dt, workers, base_seed=42)
    t1 = time.time()
    print(f"✓ Generated {len(fields_A)} KPZ-A fields in {t1-t0:.1f}s")
    
    print("\n" + "=" * 70)
    print("Generating KPZ-B fields (parallel)...")
    print("=" * 70)
    
    t0 = time.time()
    fields_B = generate_fields_parallel(params_B, N, L, T, dt, workers, base_seed=100000)
    t1 = time.time()
    print(f"✓ Generated {len(fields_B)} KPZ-B fields in {t1-t0:.1f}s")
    
    # ========================================================================
    # Extract Features at Each Scale
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Extracting scale-free structure function features...")
    print("=" * 70)
    
    features_dict = {
        'A': {b: [] for b in scales},
        'B': {b: [] for b in scales}
    }
    valid_counts = {
        'A': {b: 0 for b in scales},
        'B': {b: 0 for b in scales}
    }
    
    for system, fields in [('A', fields_A), ('B', fields_B)]:
        for b in scales:
            for h in fields:
                h_coarse = spectral_coarse_grain(h, b, alpha=0.5)
                features, diagnostic = compute_scale_free_structure_functions(h_coarse)
                if features is not None:
                    features_dict[system][b].append(features)
                    valid_counts[system][b] += 1
        
        print(f"\n{system}: {len(features_dict[system][1])} valid samples per scale")
    
    # Convert to arrays
    for system in ['A', 'B']:
        for b in scales:
            features_dict[system][b] = np.array(features_dict[system][b])
    
    min_valid = min([valid_counts[s][b] for s in ['A', 'B'] for b in scales])
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
    
    sigma = estimate_sigma(features_dict['A'][1], features_dict['B'][1])
    print(f"σ = {sigma:.4f} (median pairwise distance at b=1)")
    
    # ========================================================================
    # Diagnostic Gate
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC GATE: Testing system-vs-system invariance")
    print("=" * 70)
    
    A_pass, B_pass, A_slope, B_slope, A_dists, B_dists = \
        diagnostic_gate(features_dict, scales, sigma)
    
    print(f"\nKPZ-A-vs-A (half vs half):")
    for i, b in enumerate(scales):
        print(f"  b={b}: distance = {A_dists[i]:.4f}")
    print(f"  → Slope: {A_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if A_pass else '❌ FAIL'}")
    
    print(f"\nKPZ-B-vs-B (half vs half):")
    for i, b in enumerate(scales):
        print(f"  b={b}: distance = {B_dists[i]:.4f}")
    print(f"  → Slope: {B_slope:+.6f}")
    print(f"  → Threshold: 0.005")
    print(f"  → Status: {'✅ PASS' if B_pass else '❌ FAIL'}")
    
    gate_passed = A_pass and B_pass
    
    print("\n" + "=" * 70)
    if gate_passed:
        print("✅ DIAGNOSTIC GATE: PASSED")
        print("Both KPZ-A and KPZ-B observables are scale-invariant.")
        print("KPZ-A-vs-B trend is interpretable.")
    else:
        print("❌ DIAGNOSTIC GATE: FAILED")
        print("Observables are not scale-invariant.")
        print("Cannot interpret KPZ-A-vs-B trend.")
    print("=" * 70)
    
    # ========================================================================
    # Compute A-vs-B Distance (if gate passed)
    # ========================================================================
    
    AB_distances = []
    AB_slope = None
    AB_relative_change = None
    
    if gate_passed:
        print("\n" + "=" * 70)
        print("Computing KPZ-A-vs-B distances...")
        print("=" * 70)
        
        for b in scales:
            d_AB = mmd_distance(features_dict['A'][b], features_dict['B'][b], sigma)
            AB_distances.append(d_AB)
            print(f"  b={b}: d(KPZ-A, KPZ-B) = {d_AB:.4f}")
        
        # Fit slope
        log_b = np.log2(scales)
        AB_result = linregress(log_b, AB_distances)
        AB_slope = AB_result.slope
        AB_relative_change = (AB_distances[-1] - AB_distances[0]) / AB_distances[0] * 100
        
        print(f"\nSlope: {AB_slope:+.6f}")
        print(f"Relative change: {AB_relative_change:+.2f}%")
        
        print("\n" + "=" * 70)
        if AB_slope < -0.001:
            print("✅ CONVERGENCE DETECTED")
            print("KPZ-A and KPZ-B converge under coarse-graining (as expected)")
        elif abs(AB_slope) < 0.001:
            print("⚠️ DISTANCE FLAT")
            print("No convergence detected (unexpected for same universality class)")
        else:
            print("⚠️ DISTANCE INCREASING")
            print("Systems diverge under coarse-graining (unexpected)")
        print("=" * 70)
    else:
        print("\n⚠️ Skipping KPZ-A-vs-B distance (gate failed)")
        AB_distances = [float('nan')] * len(scales)
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    metadata = {
        'experiment': '50q',
        'description': 'KPZ-vs-KPZ positive control (within-class convergence test)',
        'note': 'Two KPZ parameter regimes with same universality class',
        'pilot_mode': PILOT,
        'parameters': {
            'L': L,
            'T': T,
            'N': N,
            'kpz_A': {'nu': params_A['nu'], 'lambda': params_A['lambda'], 'D': params_A['D']},
            'kpz_B': {'nu': params_B['nu'], 'lambda': params_B['lambda'], 'D': params_B['D']},
            'scales': scales,
            'workers': workers
        },
        'diagnostic_gate': {
            'passed': bool(gate_passed),
            'A_invariance_pass': bool(A_pass),
            'B_invariance_pass': bool(B_pass),
            'A_slope': float(A_slope),
            'B_slope': float(B_slope),
            'threshold': 0.005
        },
        'kpz_A_vs_B': {
            'distances': [float(d) if not np.isnan(d) else None for d in AB_distances],
            'slope': float(AB_slope) if AB_slope is not None else None,
            'relative_change_percent': float(AB_relative_change) if AB_relative_change is not None else None
        },
        'sigma': float(sigma)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save features
    np.savez(
        output_dir / 'kpzA_vs_kpzB_scale_free_structure_functions.npz',
        kpzA_features_b1=features_dict['A'][scales[0]],
        kpzA_features_b2=features_dict['A'][scales[1]],
        kpzA_features_b4=features_dict['A'][scales[2]],
        kpzA_features_b8=features_dict['A'][scales[3]] if len(scales) > 3 else np.array([]),
        kpzB_features_b1=features_dict['B'][scales[0]],
        kpzB_features_b2=features_dict['B'][scales[1]],
        kpzB_features_b4=features_dict['B'][scales[2]],
        kpzB_features_b8=features_dict['B'][scales[3]] if len(scales) > 3 else np.array([]),
        scales=scales,
        A_distances=A_dists,
        B_distances=B_dists,
        AB_distances=AB_distances,
        sigma=sigma
    )
    
    print(f"\n✓ Results saved to {output_dir}/")
    
    # ========================================================================
    # Plot Results
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    log_b = np.log2(scales)
    
    # Plot A-vs-A, B-vs-B
    ax.plot(log_b, A_dists, 'o-', color='C0', linewidth=2, markersize=8,
            label=f'KPZ-A-vs-A (slope: {A_slope:+.5f})')
    ax.plot(log_b, B_dists, 's-', color='C1', linewidth=2, markersize=8,
            label=f'KPZ-B-vs-B (slope: {B_slope:+.5f})')
    
    # Plot A-vs-B if gate passed
    if gate_passed and not np.all(np.isnan(AB_distances)):
        ax.plot(log_b, AB_distances, '^-', color='C2', linewidth=2, markersize=10,
                label=f'KPZ-A-vs-B (slope: {AB_slope:+.5f})', zorder=10)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('log₂(b)', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    
    title = f'Exp 50q: KPZ-vs-KPZ Positive Control'
    if PILOT:
        title += ' [PILOT]'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Status annotation
    if gate_passed:
        if AB_slope is not None and AB_slope < -0.001:
            status_text = f'Gate: PASS\nConvergence: YES'
            color = 'green'
        elif AB_slope is not None and abs(AB_slope) < 0.001:
            status_text = f'Gate: PASS\nConvergence: FLAT'
            color = 'orange'
        else:
            status_text = f'Gate: PASS\nConvergence: ?'
            color = 'blue'
    else:
        status_text = 'Gate: FAIL'
        color = 'red'
    
    ax.text(0.98, 0.98, status_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=11, color=color, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kpzA_vs_kpzB_scale_free_structure_functions.png', dpi=150)
    print(f"✓ Figure saved to {output_dir}/kpzA_vs_kpzB_scale_free_structure_functions.png")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 50q SUMMARY")
    if PILOT:
        print("[PILOT MODE]")
    print("=" * 70)
    print(f"Observable: Scale-free structure functions (height space)")
    print(f"Diagnostic gate: {'✅ PASSED' if gate_passed else '❌ FAILED'}")
    if gate_passed:
        print(f"  KPZ-A-vs-A slope: {A_slope:+.6f} ({'pass' if A_pass else 'fail'})")
        print(f"  KPZ-B-vs-B slope: {B_slope:+.6f} ({'pass' if B_pass else 'fail'})")
        print(f"KPZ-A-vs-B slope: {AB_slope:+.6f} ({AB_relative_change:+.1f}%)")
        if AB_slope < -0.001:
            print("→ Convergence detected ✅")
        elif abs(AB_slope) < 0.001:
            print("→ Distance flat (unexpected) ⚠️")
        else:
            print("→ Distance increasing (unexpected) ⚠️")
    else:
        print("→ Results not interpretable ❌")
    print("=" * 70)
    
    if PILOT and gate_passed:
        print("\n✅ PILOT PASSED - Ready for full run")
        print("Run without --pilot flag for complete experiment")

if __name__ == '__main__':
    main()
