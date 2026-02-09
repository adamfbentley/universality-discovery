"""
Experiment 50p: KPZ-Only Gate Test (Isolation Test)

Isolate KPZ to verify that:
1. Gradient-derived velocity u = dh/dx with RG rescaling passes gate
2. Dimensionless moments (skewness, kurtosis) are scale-invariant
3. Results are reproducible (not a fluke from Exp 50o)

If this passes reliably, we can focus on why Burgers with gradient noise fails.
If this fails, the observables themselves are problematic for KPZ.
"""

import numpy as np
from pathlib import Path
import json
import time
from scipy import fft
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import linregress
import matplotlib.pyplot as plt

# =============================================================================
# KPZ SIMULATION (SAME AS EXP 50o)
# =============================================================================

def simulate_kpz(L, T, dt, nu, lam, D, rng):
    """
    KPZ simulation: dh/dt = nu*d²h/dx² + (lambda/2)*(dh/dx)² + eta
    Returns velocity u = dh/dx (not height h)
    """
    # Grid
    N = L
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    
    # Wavenumbers
    k = 2 * np.pi * fft.fftfreq(N, d=dx)
    k2 = k**2
    
    # Initial condition (small random height)
    h = rng.normal(0, 0.1, N)
    h_hat = fft.fft(h)
    
    # ETD factors
    exp_factor = np.exp(-nu * k2 * dt)
    etd_coef = np.where(k != 0, (exp_factor - 1) / (-nu * k2), dt)
    
    # De-aliasing mask
    k_max = np.max(np.abs(k))
    dealias = np.abs(k) < (2.0/3.0) * k_max
    
    # Time evolution
    num_steps = int(T / dt)
    for step in range(num_steps):
        # Current height in real space
        h = np.real(fft.ifft(h_hat))
        
        # Gradient (dh/dx)
        grad_h = np.real(fft.ifft(1j * k * h_hat))
        
        # Nonlinear term (lambda/2)*(dh/dx)^2
        nonlinear = (lam / 2) * grad_h**2
        nonlinear_hat = fft.fft(nonlinear)
        nonlinear_hat *= dealias
        
        # Noise (white in space and time)
        noise = rng.normal(0, 1, N) * np.sqrt(2 * D / dt)
        noise_hat = fft.fft(noise)
        
        # ETD update
        h_hat = exp_factor * h_hat + etd_coef * (nonlinear_hat + noise_hat)
    
    # Return velocity u = dh/dx (not height)
    u = np.real(fft.ifft(1j * k * h_hat))
    
    # Sanity check
    if not np.all(np.isfinite(u)):
        return None
    if np.max(np.abs(u)) > 1e3:
        return None
    
    return u

# =============================================================================
# RG COARSE-GRAINING WITH RESCALING
# =============================================================================

def spectral_coarse_grain(u, b):
    """
    Coarse-grain u by factor b in Fourier space, then apply RG rescaling.
    
    For KPZ: alpha = 1/2, so velocity u ~ b^(alpha-1) = b^(-1/2)
    To make scale-invariant, multiply by b^(1/2) after coarse-graining.
    """
    N = len(u)
    L = N  # physical length
    
    # Fourier transform
    u_hat = fft.fft(u)
    
    # New grid size
    N_new = N // b
    
    # Extract low-k modes (band-limited to new Nyquist)
    u_hat_coarse = np.zeros(N_new, dtype=complex)
    u_hat_coarse[:N_new//2] = u_hat[:N_new//2]
    u_hat_coarse[N_new//2:] = u_hat[-N_new//2:]
    
    # Back to real space
    u_coarse = np.real(fft.ifft(u_hat_coarse))
    
    # RG RESCALING: multiply by b^(1/2) to compensate for u ~ b^(-1/2)
    u_rg = u_coarse * np.sqrt(b)
    
    # Enforce zero mean (remove k=0 mode)
    u_rg = u_rg - np.mean(u_rg)
    
    return u_rg

# =============================================================================
# DIMENSIONLESS VELOCITY MOMENTS
# =============================================================================

def compute_dimensionless_moments(u):
    """
    Extract purely dimensionless moments from velocity field u.
    
    Returns 2D feature vector: [skewness, kurtosis]
    Both are scale-invariant by construction.
    """
    # Remove mean (should already be zero, but ensure)
    u_centered = u - np.mean(u)
    
    # Remove k=0 in Fourier space (ensure truly zero mean)
    u_hat = fft.fft(u_centered)
    u_hat[0] = 0
    u_clean = np.real(fft.ifft(u_hat))
    
    # Standard deviation
    sigma = np.std(u_clean)
    if sigma < 1e-12:
        return np.array([0.0, 0.0])
    
    # Dimensionless moments
    u_norm = u_clean / sigma
    
    skewness = np.mean(u_norm**3)
    kurtosis = np.mean(u_norm**4)
    
    return np.array([skewness, kurtosis])

# =============================================================================
# MMD DISTANCE
# =============================================================================

def mmd_rbf(X, Y, sigma):
    """MMD with RBF kernel"""
    XX = rbf_kernel(X, X, gamma=1/(2*sigma**2))
    YY = rbf_kernel(Y, Y, gamma=1/(2*sigma**2))
    XY = rbf_kernel(X, Y, gamma=1/(2*sigma**2))
    return np.sqrt(max(0, XX.mean() + YY.mean() - 2*XY.mean()))

# =============================================================================
# DIAGNOSTIC GATE
# =============================================================================

def diagnostic_gate(features, scales, sigma, threshold=0.005):
    """
    Test if observables are scale-invariant by comparing halves of same system.
    
    Returns (passed, slope)
    """
    N = len(features[scales[0]])
    N_half = N // 2
    
    distances = []
    for b in scales:
        feats = features[b]
        # Split in half
        feats_A = feats[:N_half]
        feats_B = feats[N_half:]
        # MMD distance
        dist = mmd_rbf(feats_A, feats_B, sigma)
        distances.append(dist)
    
    # Linear regression: distance vs log(b)
    log_b = np.log(scales)
    slope, intercept, r_val, p_val, std_err = linregress(log_b, distances)
    
    passed = abs(slope) < threshold
    
    return passed, slope, distances

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("Experiment 50p: KPZ-Only Gate Test (Isolation)")
    print("=" * 70)
    print()
    
    # Parameters (match Exp 50o)
    L = 256
    T = 200
    dt_kpz = 0.01
    nu = 1.0
    lam = 1.0
    D = 0.1
    N = 500
    scales = [1, 2, 4, 8]
    
    print("Parameters:")
    print(f"  KPZ: nu={nu}, lambda={lam}, D={D}")
    print(f"  L={L}, T={T}, dt={dt_kpz}")
    print(f"  N={N} fields")
    print(f"  Scales: {scales}")
    print(f"  Observable: Dimensionless velocity moments (2D: skewness, kurtosis)")
    print(f"    - RG rescaling: u_rg = coarse_grain(u) * sqrt(b)")
    print(f"    - Features: skewness, kurtosis (purely scale-free)")
    print(f"    - KPZ: u = dh/dx (velocity from height)")
    print()
    print(f"  GOAL: Verify KPZ observables pass gate reliably")
    print()
    
    # Output directory
    output_dir = Path(__file__).parent.parent / 'results_exp50p'
    output_dir.mkdir(exist_ok=True)
    
    # Random seed
    rng = np.random.default_rng(42)
    
    # =============================================================================
    # GENERATE KPZ VELOCITY FIELDS
    # =============================================================================
    
    print("=" * 70)
    print("Generating KPZ velocity fields (u = dh/dx)...")
    print("=" * 70)
    
    kpz_fields = []
    skipped_kpz = 0
    
    t0 = time.time()
    for i in range(N):
        u_kpz = simulate_kpz(L, T, dt_kpz, nu, lam, D, rng)
        
        if u_kpz is None:
            skipped_kpz += 1
            continue
        
        kpz_fields.append(u_kpz)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{N} KPZ fields (skipped: {skipped_kpz})...")
        
        if skipped_kpz > N // 2:
            raise RuntimeError(f"Too many skipped KPZ fields ({skipped_kpz})")
    
    t1 = time.time()
    print(f"✓ Generated {len(kpz_fields)} KPZ fields in {t1-t0:.1f}s")
    print()
    
    if len(kpz_fields) < N:
        raise RuntimeError(f"Only {len(kpz_fields)} valid KPZ fields (need {N})")
    
    # Trim to exactly N
    kpz_fields = kpz_fields[:N]
    
    # =============================================================================
    # EXTRACT FEATURES AT MULTIPLE SCALES
    # =============================================================================
    
    print("=" * 70)
    print("Extracting dimensionless velocity moments (skewness, kurtosis)...")
    print("=" * 70)
    print()
    
    # Multi-scale features
    kpz_features = {b: [] for b in scales}
    
    for u_kpz in kpz_fields:
        for b in scales:
            u_rg = spectral_coarse_grain(u_kpz, b)
            feats = compute_dimensionless_moments(u_rg)
            kpz_features[b].append(feats)
    
    # Convert to arrays
    for b in scales:
        kpz_features[b] = np.array(kpz_features[b])
    
    print(f"KPZ: {len(kpz_features[1])} samples per scale")
    print()
    
    # Sanity check
    for b in scales:
        n_valid = np.sum(np.all(np.isfinite(kpz_features[b]), axis=1))
        if n_valid < len(kpz_features[b]):
            print(f"⚠ Warning: {len(kpz_features[b]) - n_valid} invalid KPZ samples at b={b}")
    
    print("✓ Feature extraction complete")
    print()
    
    # =============================================================================
    # ESTIMATE MMD BANDWIDTH
    # =============================================================================
    
    print("=" * 70)
    print("Estimating MMD bandwidth σ...")
    print("=" * 70)
    
    # Use b=1 features
    feats_b1 = kpz_features[1]
    
    # Median heuristic: σ = median pairwise distance
    n_samples = len(feats_b1)
    idx = np.random.choice(n_samples, size=min(200, n_samples), replace=False)
    feats_sample = feats_b1[idx]
    
    dists = []
    for i in range(len(feats_sample)):
        for j in range(i+1, len(feats_sample)):
            d = np.linalg.norm(feats_sample[i] - feats_sample[j])
            dists.append(d)
    
    sigma = np.median(dists)
    print(f"σ = {sigma:.4f} (median pairwise distance at b=1)")
    print()
    
    # =============================================================================
    # DIAGNOSTIC GATE: KPZ-VS-KPZ INVARIANCE TEST
    # =============================================================================
    
    print("=" * 70)
    print("DIAGNOSTIC GATE: Testing KPZ-vs-KPZ invariance")
    print("=" * 70)
    print()
    
    threshold = 0.005
    passed, slope, distances = diagnostic_gate(kpz_features, scales, sigma, threshold)
    
    print("KPZ-vs-KPZ (half vs half):")
    for i, b in enumerate(scales):
        print(f"  b={b}: distance = {distances[i]:.4f}")
    print(f"  → Slope: {slope:+.6f}")
    print(f"  → Threshold: {threshold}")
    if passed:
        print(f"  → Status: ✅ PASS")
    else:
        print(f"  → Status: ❌ FAIL")
    print()
    
    # =============================================================================
    # SAVE RESULTS
    # =============================================================================
    
    metadata = {
        'experiment': '50p',
        'description': 'KPZ-only gate test to verify observables are scale-invariant',
        'note': 'Isolation test: gradient noise, RG rescaling, skewness+kurtosis only',
        'parameters': {
            'L': L,
            'T': T,
            'N': N,
            'kpz': {
                'nu': nu,
                'lambda': lam,
                'D': D
            },
            'scales': scales
        },
        'diagnostic_gate': {
            'passed': bool(passed),
            'slope': float(slope),
            'threshold': float(threshold),
            'distances': [float(d) for d in distances]
        },
        'sigma': float(sigma)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save features
    np.savez(output_dir / 'features.npz',
             kpz_b1=kpz_features[1],
             kpz_b2=kpz_features[2],
             kpz_b4=kpz_features[4],
             kpz_b8=kpz_features[8])
    
    print(f"✓ Results saved to {output_dir}/")
    
    # =============================================================================
    # PLOT RESULTS
    # =============================================================================
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    log_b = np.log(scales)
    ax.plot(log_b, distances, 'o-', color='C0', linewidth=2, markersize=8, label='KPZ-vs-KPZ')
    
    # Fit line
    fit_line = slope * log_b + distances[0]
    ax.plot(log_b, fit_line, '--', color='C0', alpha=0.5)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('log(b)', fontsize=12)
    ax.set_ylabel('MMD Distance', fontsize=12)
    ax.set_title(f'Exp 50p: KPZ-Only Gate Test\nSlope: {slope:+.6f} (threshold: ±{threshold})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Status
    status = "PASS" if passed else "FAIL"
    color = 'green' if passed else 'red'
    ax.text(0.98, 0.98, f'Gate: {status}', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, color=color, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kpz_gate_test.png', dpi=150)
    print(f"✓ Figure saved to {output_dir}/kpz_gate_test.png")
    print()
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("=" * 70)
    print("EXPERIMENT 50p SUMMARY")
    print("=" * 70)
    print(f"KPZ-vs-KPZ gate: {'✅ PASSED' if passed else '❌ FAILED'}")
    print(f"Slope: {slope:+.6f} (threshold: ±{threshold})")
    print()
    if passed:
        print("✓ KPZ observables are scale-invariant")
        print("→ Issue in Exp 50o is with Burgers (not KPZ)")
    else:
        print("✗ KPZ observables are NOT scale-invariant")
        print("→ Problem with observables themselves, not Burgers")
    print("=" * 70)

if __name__ == '__main__':
    main()
