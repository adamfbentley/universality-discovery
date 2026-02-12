"""
Experiment 58: Ising Magnetic Sector Test
==========================================

PURPOSE: Test the Symmetry-Based Sufficiency Theorem by probing the
MAGNETIC (Z2-odd) sector of the 2D Ising model.

THEOREM PREDICTION:
  - Standard PCA-FSS uses Z2-EVEN features (|m|, boundary density, etc.)
  - These couple to the THERMAL operator → recover nu_thermal = 1.0  ✓
  - Z2-ODD features (signed m, m*|m|, etc.) couple to MAGNETIC operator
  - Prediction: PCA on Z2-odd features should recover different exponent
  - Specifically: the magnetic direction has Delta_sigma = 1/8 = beta/nu
  - If PCA picks up the magnetic scaling, we should see nu_mag or
    anomalous exponent behavior

APPROACH:
  A) Z2-EVEN features only (control, should reproduce Exp 52d): 
     var_local_m, |m_local|, |grad|, var_grad, boundary, corr
  B) Z2-ODD features: signed_m, signed_m^3, m*energy, local_m_mean (signed)
  C) MIXED features: both even and odd
  
  Apply PCA-FSS to each set. If theorem holds:
  - Set A: nu ≈ 1.0 (thermal, as in 52d)
  - Set B: different behavior (magnetic sector dominates)
  - Set C: thermal wins (larger eigenvalue) but we may see magnetic in PC2

SIMULATION: Wolff cluster (Numba) — same as 52d

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
from numba import jit
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import argparse
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Run pilot mode')
args = parser.parse_args()

PILOT = args.pilot

T_C = 2.0 / np.log(1 + np.sqrt(2))  # 2.269...
NU_EXACT = 1.0
BETA_NU = 0.125  # beta/nu for 2D Ising

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_temps': 12,
        'n_samples': 20,
        'n_equilibrate': 1500,
        'n_measure': 300,
        'n_measurements': 3,
        'T_half_width': 0.15,  # ±15% of T_c
        'output_dir': 'results_exp58_pilot',
    }
else:
    CONFIG = {
        'L_values': [32, 48, 64, 96],
        'n_temps': 15,
        'n_samples': 30,
        'n_equilibrate': 3000,
        'n_measure': 500,
        'n_measurements': 5,
        'T_half_width': 0.15,
        'output_dir': 'results_exp58',
    }

# ============================================================================
# ISING SIMULATION (Wolff, Numba)
# ============================================================================

@jit(nopython=True)
def wolff_step(spins, T, L):
    """Single Wolff cluster flip."""
    J = 1.0
    p_add = 1.0 - np.exp(-2.0 * J / T)
    
    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    
    cluster = np.zeros((L, L), dtype=np.int8)
    stack_i = np.zeros(L * L, dtype=np.int32)
    stack_j = np.zeros(L * L, dtype=np.int32)
    stack_i[0] = i0
    stack_j[0] = j0
    stack_ptr = 1
    cluster[i0, j0] = 1
    s0 = spins[i0, j0]
    
    while stack_ptr > 0:
        stack_ptr -= 1
        i = stack_i[stack_ptr]
        j = stack_j[stack_ptr]
        
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni = (i + di) % L
            nj = (j + dj) % L
            
            if cluster[ni, nj] == 0 and spins[ni, nj] == s0:
                if np.random.random() < p_add:
                    cluster[ni, nj] = 1
                    stack_i[stack_ptr] = ni
                    stack_j[stack_ptr] = nj
                    stack_ptr += 1
    
    for i in range(L):
        for j in range(L):
            if cluster[i, j] == 1:
                spins[i, j] *= -1
    
    return spins


@jit(nopython=True)
def run_ising(L, T, n_equilibrate, n_measure, n_measurements):
    """Run Ising simulation and collect configurations."""
    spins = np.ones((L, L), dtype=np.int8)
    for i in range(L):
        for j in range(L):
            if np.random.random() < 0.5:
                spins[i, j] = -1
    
    for _ in range(n_equilibrate):
        spins = wolff_step(spins, T, L)
    
    configs = np.zeros((n_measurements, L, L), dtype=np.int8)
    for m_idx in range(n_measurements):
        for _ in range(n_measure):
            spins = wolff_step(spins, T, L)
        for i in range(L):
            for j in range(L):
                configs[m_idx, i, j] = spins[i, j]
    
    return configs


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_even(config):
    """Z2-EVEN features (invariant under spin flip). Same as 52d."""
    L = config.shape[0]
    m = config.astype(np.float64)
    
    # Local magnetization (coarse-grained)
    block = max(2, L // 16)
    L_coarse = L // block
    m_coarse = np.zeros((L_coarse, L_coarse))
    for i in range(L_coarse):
        for j in range(L_coarse):
            m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
    
    var_local_m = np.var(m_coarse)
    abs_local_m = np.mean(np.abs(m_coarse))
    
    # Gradient features
    grad_x = np.roll(m, -1, axis=0) - m
    grad_y = np.roll(m, -1, axis=1) - m
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    mean_grad = np.mean(grad_mag)
    var_grad = np.var(grad_mag)
    
    # Domain boundary density
    n_boundaries = 0
    for i in range(L):
        for j in range(L):
            if config[i, j] != config[(i+1)%L, j]:
                n_boundaries += 1
            if config[i, j] != config[i, (j+1)%L]:
                n_boundaries += 1
    boundary = n_boundaries / (2 * L * L)
    
    # Correlation
    corr_1 = np.mean(m * np.roll(m, 1, axis=0))
    
    return np.array([var_local_m, abs_local_m, mean_grad, var_grad, boundary, corr_1])


def extract_features_odd(config):
    """Z2-ODD features (change sign under spin flip).
    These couple to the MAGNETIC sector."""
    L = config.shape[0]
    m = config.astype(np.float64)
    N = L * L
    
    # 1. Signed magnetization (the actual order parameter!)
    signed_m = np.mean(m)
    
    # 2. Signed local magnetization mean (not absolute value)
    block = max(2, L // 16)
    L_coarse = L // block
    m_coarse = np.zeros((L_coarse, L_coarse))
    for i in range(L_coarse):
        for j in range(L_coarse):
            m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
    signed_local_m = np.mean(m_coarse)  # Not |m_coarse|
    
    # 3. Cubic magnetization m^3 (odd under Z2)
    m_cubed = np.mean(m**3)
    
    # 4. Magnetization × energy coupling (m * E is odd)
    E = 0.0
    for i in range(L):
        for j in range(L):
            E -= m[i, j] * (m[(i+1)%L, j] + m[i, (j+1)%L])
    E_per_site = E / N
    m_times_E = signed_m * E_per_site
    
    # 5. Signed gradient sum (directional asymmetry, odd under flip)
    grad_x = np.roll(m, -1, axis=0) - m
    signed_grad = np.mean(grad_x * m)  # Odd: m * ∂m = m * ∂m changes sign
    
    # 6. Staggered-like (odd coupling between m and spatial structure)
    # Sum of m_i * m_j for j=neighbor weighted by sign
    m_nn_asym = np.mean(m * (np.roll(m, 1, axis=0) - np.roll(m, -1, axis=0)))
    
    return np.array([signed_m, signed_local_m, m_cubed, m_times_E, signed_grad, m_nn_asym])


def extract_features_all(config):
    """Combined even + odd features."""
    even = extract_features_even(config)
    odd = extract_features_odd(config)
    return np.concatenate([even, odd])


# ============================================================================
# PCA-FSS ANALYSIS
# ============================================================================

def compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu):
    """Quality metric for data collapse. Lower = better."""
    all_xi = []
    all_pc1 = []
    
    for L in L_values:
        xi = t_by_L[L] * (L ** (1.0 / nu))
        all_xi.extend(xi)
        all_pc1.extend(pc1_by_L[L])
    
    all_xi = np.array(all_xi)
    all_pc1 = np.array(all_pc1)
    
    sort_idx = np.argsort(all_xi)
    all_xi = all_xi[sort_idx]
    all_pc1 = all_pc1[sort_idx]
    
    n_windows = 20
    window_size = max(1, len(all_xi) // n_windows)
    
    total_var = 0.0
    n_valid = 0
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        if end > len(all_pc1):
            break
        window_pc1 = all_pc1[start:end]
        if len(window_pc1) > 1:
            total_var += np.var(window_pc1)
            n_valid += 1
    
    return total_var / n_valid if n_valid > 0 else np.inf


def find_optimal_nu(pc1_by_L, t_by_L, L_values, nu_range=(0.3, 3.0)):
    """Find ν that minimizes collapse spread."""
    def objective(nu):
        return compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu)
    result = minimize_scalar(objective, bounds=nu_range, method='bounded')
    return result.x, result.fun


def run_pca_fss(all_data, L_values, feature_set='even', n_features=None):
    """Run PCA-FSS pipeline on a given feature set."""
    
    # Extract features
    all_features = []
    all_t = []
    all_L = []
    
    for L in L_values:
        for i in range(len(all_data[L]['configs'])):
            config = all_data[L]['configs'][i]
            T = all_data[L]['temps'][i]
            t = (T - T_C) / T_C
            
            if feature_set == 'even':
                feat = extract_features_even(config)
            elif feature_set == 'odd':
                feat = extract_features_odd(config)
            else:
                feat = extract_features_all(config)
            
            all_features.append(feat)
            all_t.append(t)
            all_L.append(L)
    
    all_features = np.array(all_features)
    all_t = np.array(all_t)
    all_L = np.array(all_L)
    
    if n_features is None:
        n_features = all_features.shape[1]
    
    # Standardize + PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    pca = PCA(n_components=min(n_features, features_scaled.shape[1]))
    pca_coords = pca.fit_transform(features_scaled)
    
    # Split by L
    pc1_by_L = {}
    t_by_L = {}
    for L in L_values:
        mask = all_L == L
        pc1_by_L[L] = pca_coords[mask, 0]
        t_by_L[L] = all_t[mask]
    
    # Find optimal nu
    nu_opt, quality = find_optimal_nu(pc1_by_L, t_by_L, L_values)
    
    return {
        'nu_opt': nu_opt,
        'quality': quality,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pc1_by_L': pc1_by_L,
        't_by_L': t_by_L,
        'pca': pca,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 58: ISING MAGNETIC SECTOR TEST")
    print("=" * 70)
    print()
    print(f"Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"T_c = {T_C:.5f}, nu_exact = {NU_EXACT:.4f}")
    print(f"beta/nu = {BETA_NU}")
    print()
    
    L_values = CONFIG['L_values']
    T_min = T_C * (1 - CONFIG['T_half_width'])
    T_max = T_C * (1 + CONFIG['T_half_width'])
    temperatures = np.linspace(T_min, T_max, CONFIG['n_temps'])
    
    print(f"L values: {L_values}")
    print(f"T range: [{T_min:.3f}, {T_max:.3f}] ({CONFIG['n_temps']} pts)")
    print(f"Samples per point: {CONFIG['n_samples']}")
    print()
    
    # Generate data
    start_time = time.time()
    all_data = {}
    
    for L in L_values:
        print(f"\nL={L}:")
        configs_L = []
        temps_L = []
        
        for ti, T in enumerate(temperatures):
            t_reduced = (T - T_C) / T_C
            print(f"  T={T:.3f} (t={t_reduced:+.3f}): ", end="", flush=True)
            
            for s in range(CONFIG['n_samples']):
                np.random.seed(42 + L * 10000 + ti * 100 + s)
                configs = run_ising(L, T, CONFIG['n_equilibrate'],
                                    CONFIG['n_measure'], CONFIG['n_measurements'])
                # Average over sub-measurements
                avg_config = configs  # Keep all configs
                for c in configs:
                    configs_L.append(c)
                    temps_L.append(T)
                
                if (s + 1) % 10 == 0:
                    print(".", end="", flush=True)
            print(" done")
        
        all_data[L] = {
            'configs': configs_L,
            'temps': np.array(temps_L),
        }
    
    elapsed = time.time() - start_time
    print(f"\nData generation: {elapsed:.1f}s")
    
    # Run PCA-FSS for each feature set
    print("\n" + "=" * 70)
    print("PCA-FSS ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    for feature_set in ['even', 'odd', 'all']:
        print(f"\n--- Feature Set: {feature_set.upper()} ---")
        t0 = time.time()
        res = run_pca_fss(all_data, L_values, feature_set)
        dt = time.time() - t0
        
        nu = res['nu_opt']
        error_pct = 100 * abs(nu - NU_EXACT) / NU_EXACT
        
        print(f"  nu_opt = {nu:.4f} (error: {error_pct:.1f}%)")
        print(f"  Explained variance (PC1-3): {res['explained_variance'][:3]}")
        print(f"  Collapse quality: {res['quality']:.6f}")
        print(f"  [{dt:.1f}s]")
        
        results[feature_set] = {
            'nu_opt': float(nu),
            'error_pct': float(error_pct),
            'explained_variance': res['explained_variance'],
            'quality': float(res['quality']),
        }
    
    # Comparative analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS: SYMMETRY SECTORS")
    print("=" * 70)
    print()
    print(f"  Feature Set    | nu_opt  | Error   | Interpretation")
    print(f"  --------------|---------|---------|---------------")
    for fs in ['even', 'odd', 'all']:
        r = results[fs]
        if r['error_pct'] < 15:
            interp = "Thermal sector (nu=1)"
        elif r['error_pct'] < 30:
            interp = "Partial"
        else:
            interp = "Different sector?"
        print(f"  {fs:14s} | {r['nu_opt']:.4f}  | {r['error_pct']:5.1f}%  | {interp}")
    
    print()
    print("SYMMETRY THEOREM PREDICTION:")
    print("  Z2-even features → see thermal operator (nu ≈ 1.0)")
    print("  Z2-odd features  → see magnetic operator (different exponent)")
    print("  Mixed features   → thermal dominates (larger eigenvalue)")
    print()
    
    even_ok = results['even']['error_pct'] < 15
    odd_diff = abs(results['odd']['nu_opt'] - results['even']['nu_opt']) > 0.1
    
    if even_ok and odd_diff:
        print("✓ CONSISTENT with symmetry theorem:")
        print("  Even features recover thermal nu, odd features see something different")
    elif even_ok and not odd_diff:
        print("⚠ PARTIAL: Even features work, but odd features also give thermal nu")
        print("  This could mean Z2-odd features still couple to thermal sector")
    else:
        print("✗ UNEXPECTED: Even features don't recover thermal nu")
    
    # Save
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
