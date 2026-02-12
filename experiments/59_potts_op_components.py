"""
Experiment 59: Potts Order Parameter Components — THE CRITICAL TEST
===================================================================

PURPOSE: Test the central prediction of the Symmetry-Based Sufficiency
Theorem by including features that are LINEAR in the Potts order parameter.

THE THEOREM (conjectured):
  PCA-FSS on G-invariant local observables recovers nu of the leading
  G-invariant relevant operator.
  
  For 3-state Potts (S3 symmetry):
  - S3-invariant features (boundary density, correlation, etc.) couple
    QUADRATICALLY to the 2-component OP → effective exponent = 2*nu
  - Features LINEAR in the OP couple DIRECTLY → recover true nu

THE TEST:
  A) S3-invariant features only (control, should reproduce nu ≈ 1.66)
     Same as Exp 55b: boundary, correlation, gradient, cluster_var, etc.
  
  B) Potts OP components (sigma_1, sigma_2):
     sigma_1 = (2*n_1 - n_2 - n_3) / 3
     sigma_2 = (n_2 - n_3) / sqrt(3)
     These break S3 symmetry — they ARE the order parameter.
     PREDICTION: PCA-FSS on (sigma_1, sigma_2, + invariants) → nu ≈ 5/6
  
  C) Potts OP norm only |sigma| = sqrt(sigma_1^2 + sigma_2^2):
     This is S3-invariant but linear in |OP|.
     What does PCA see?

IF B gives nu ≈ 5/6: THEOREM PROVEN (at least numerically)
IF B gives nu ≈ 1.66: THEOREM FALSIFIED — symmetry isn't the issue

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
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

T_C = 1.0 / np.log(1 + np.sqrt(3))  # 0.9950
NU_EXACT = 5 / 6  # 0.8333

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_temps': 12,
        'n_samples': 20,
        'n_eq_base': 800,
        'n_decorr_base': 50,
        'n_measurements': 3,  # Sub-measurements per sample
        'T_half_width': 0.15,
        'output_dir': 'results_exp59_pilot',
    }
else:
    CONFIG = {
        'L_values': [24, 32, 48, 64],
        'n_temps': 15,
        'n_samples': 30,
        'n_eq_base': 1200,
        'n_decorr_base': 80,
        'n_measurements': 5,
        'T_half_width': 0.15,
        'output_dir': 'results_exp59',
    }

# ============================================================================
# POTTS SIMULATION (Wolff cluster, pure Python)
# ============================================================================

def wolff_cluster_potts(config, T, q=3):
    """Wolff cluster update for q-state Potts."""
    L = config.shape[0]
    p_add = 1 - np.exp(-1 / T)
    
    i, j = np.random.randint(0, L, 2)
    seed_spin = config[i, j]
    
    cluster = np.zeros((L, L), dtype=bool)
    stack = [(i, j)]
    cluster[i, j] = True
    
    while stack:
        x, y = stack.pop()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % L, (y + dy) % L
            if not cluster[nx, ny] and config[nx, ny] == seed_spin:
                if np.random.random() < p_add:
                    cluster[nx, ny] = True
                    stack.append((nx, ny))
    
    new_spin = (seed_spin + np.random.randint(1, q)) % q
    config[cluster] = new_spin
    return config


def simulate_potts(L, T, n_eq, n_decorr, n_measurements, seed=None):
    """Generate Potts configurations."""
    if seed is not None:
        np.random.seed(seed)
    
    config = np.random.randint(0, 3, size=(L, L))
    
    # Equilibrate
    for _ in range(n_eq):
        config = wolff_cluster_potts(config, T)
    
    # Collect configurations
    configs = []
    for _ in range(n_measurements):
        for _ in range(n_decorr):
            config = wolff_cluster_potts(config, T)
        configs.append(config.copy())
    
    return configs


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_potts_op_components(config, q=3):
    """
    Extract the 2-component Potts order parameter (Baxter representation).
    
    For q=3 Potts, the OP lives in 2D representation of S3:
      sigma_1 = (2*n_1 - n_2 - n_3) / 3
      sigma_2 = (n_2 - n_3) / sqrt(3)
    
    where n_s = fraction of sites in state s.
    
    These are LINEAR in the order parameter — they break S3 symmetry.
    """
    L = config.shape[0]
    N = L * L
    
    n = np.array([np.sum(config == s) / N for s in range(q)])
    
    sigma_1 = (2 * n[0] - n[1] - n[2]) / 3.0
    sigma_2 = (n[1] - n[2]) / np.sqrt(3.0)
    
    return sigma_1, sigma_2


def extract_features_invariant(config, q=3):
    """S3-INVARIANT features (same as Exp 55b). These are all symmetric
    under permutation of the q states."""
    L = config.shape[0]
    N = L * L
    
    # 1. Magnetization-like: max state fraction (S3-invariant)
    fracs = np.array([np.sum(config == s) / N for s in range(q)])
    max_frac = np.max(fracs)
    m_potts = (q * max_frac - 1) / (q - 1)
    
    # 2. Local magnetization variance (coarse-grained)
    block = max(2, L // 8)
    L_coarse = L // block
    local_m = np.zeros(L_coarse * L_coarse)
    for bi in range(L_coarse):
        for bj in range(L_coarse):
            block_config = config[bi*block:(bi+1)*block, bj*block:(bj+1)*block]
            block_fracs = np.array([np.sum(block_config == s) for s in range(q)]) / block**2
            local_m[bi * L_coarse + bj] = (q * np.max(block_fracs) - 1) / (q - 1)
    var_local_m = np.var(local_m)
    
    # 3. Domain boundary density
    n_boundaries = 0
    for i in range(L):
        for j in range(L):
            if config[i, j] != config[(i+1)%L, j]:
                n_boundaries += 1
            if config[i, j] != config[i, (j+1)%L]:
                n_boundaries += 1
    boundary = n_boundaries / (2 * N)
    
    # 4. Nearest-neighbor correlation (Potts-style: delta(s_i, s_j))
    corr = 0.0
    for i in range(L):
        for j in range(L):
            if config[i, j] == config[(i+1)%L, j]:
                corr += 1
            if config[i, j] == config[i, (j+1)%L]:
                corr += 1
    corr /= (2 * N)
    
    # 5. Gradient-like (entropy of neighbor disagreements)
    n_diff_neighbors = np.zeros(N)
    idx = 0
    for i in range(L):
        for j in range(L):
            count = 0
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                if config[i,j] != config[(i+dx)%L, (j+dy)%L]:
                    count += 1
            n_diff_neighbors[idx] = count / 4.0
            idx += 1
    mean_grad = np.mean(n_diff_neighbors)
    var_grad = np.var(n_diff_neighbors)
    
    return np.array([m_potts, var_local_m, boundary, corr, mean_grad, var_grad])


def extract_features_with_op(config, q=3):
    """Features including OP components (breaks S3 symmetry)."""
    invariant = extract_features_invariant(config, q)
    sigma_1, sigma_2 = extract_potts_op_components(config, q)
    
    # Also include |sigma| as a separate feature
    sigma_norm = np.sqrt(sigma_1**2 + sigma_2**2)
    
    # Combined: [sigma_1, sigma_2, |sigma|, + invariant features]
    return np.concatenate([[sigma_1, sigma_2, sigma_norm], invariant])


def extract_features_op_norm_only(config, q=3):
    """Features with |sigma| (S3-invariant) but not components."""
    invariant = extract_features_invariant(config, q)
    sigma_1, sigma_2 = extract_potts_op_components(config, q)
    sigma_norm = np.sqrt(sigma_1**2 + sigma_2**2)
    
    return np.concatenate([[sigma_norm], invariant])


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
    """Find nu that minimizes collapse spread."""
    def objective(nu):
        return compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu)
    result = minimize_scalar(objective, bounds=nu_range, method='bounded')
    return result.x, result.fun


def run_pca_fss(feature_matrix, t_array, L_array, L_values, label=""):
    """Run PCA-FSS on a pre-computed feature matrix."""
    
    # Standardize + PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    n_comp = min(feature_matrix.shape[1], feature_matrix.shape[0])
    pca = PCA(n_components=n_comp)
    pca_coords = pca.fit_transform(features_scaled)
    
    # Split by L
    pc1_by_L = {}
    t_by_L = {}
    for L in L_values:
        mask = L_array == L
        pc1_by_L[L] = pca_coords[mask, 0]
        t_by_L[L] = t_array[mask]
    
    # Find optimal nu
    nu_opt, quality = find_optimal_nu(pc1_by_L, t_by_L, L_values)
    error_pct = 100 * abs(nu_opt - NU_EXACT) / NU_EXACT
    
    return {
        'nu_opt': float(nu_opt),
        'error_pct': float(error_pct),
        'quality': float(quality),
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'loadings_pc1': pca.components_[0].tolist(),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 59: POTTS OP COMPONENTS — THE CRITICAL TEST")
    print("=" * 70)
    print()
    print(f"Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"T_c = {T_C:.6f}, exact nu = {NU_EXACT:.4f} (5/6)")
    print()
    print("THEOREM PREDICTION:")
    print("  A) S3-invariant features → nu ≈ 2 × 5/6 ≈ 1.67 (quadratic coupling)")
    print("  B) OP components (σ₁,σ₂) + invariants → nu ≈ 5/6 (linear coupling)")
    print("  C) |σ| only + invariants → nu ≈ ? (|σ| is S3-invariant)")
    print()
    
    L_values = CONFIG['L_values']
    T_min = T_C * (1 - CONFIG['T_half_width'])
    T_max = T_C * (1 + CONFIG['T_half_width'])
    temperatures = np.linspace(T_min, T_max, CONFIG['n_temps'])
    
    print(f"L values: {L_values}")
    print(f"T range: [{T_min:.4f}, {T_max:.4f}] ({CONFIG['n_temps']} pts)")
    print(f"Samples per (L,T): {CONFIG['n_samples']} × {CONFIG['n_measurements']} configs")
    print()
    
    # ========================================================================
    # DATA GENERATION
    # ========================================================================
    
    start_time = time.time()
    
    # Store features for all three variants
    features_inv = []     # S3-invariant only
    features_op = []      # With OP components
    features_norm = []    # With |sigma| only
    all_t = []
    all_L_arr = []
    
    for L in L_values:
        print(f"\nL={L}:")
        n_eq = CONFIG['n_eq_base'] + 15 * L
        n_decorr = CONFIG['n_decorr_base'] + 3 * L
        
        for ti, T in enumerate(temperatures):
            t_reduced = (T - T_C) / T_C
            print(f"  T={T:.4f} (t={t_reduced:+.3f}): ", end="", flush=True)
            
            for s in range(CONFIG['n_samples']):
                seed = 42 + L * 10000 + ti * 100 + s
                configs = simulate_potts(L, T, n_eq, n_decorr,
                                         CONFIG['n_measurements'], seed=seed)
                
                for config in configs:
                    f_inv = extract_features_invariant(config)
                    f_op = extract_features_with_op(config)
                    f_norm = extract_features_op_norm_only(config)
                    
                    features_inv.append(f_inv)
                    features_op.append(f_op)
                    features_norm.append(f_norm)
                    all_t.append(t_reduced)
                    all_L_arr.append(L)
                
                if (s + 1) % 10 == 0:
                    print(".", end="", flush=True)
            print(" done")
    
    features_inv = np.array(features_inv)
    features_op = np.array(features_op)
    features_norm = np.array(features_norm)
    all_t = np.array(all_t)
    all_L_arr = np.array(all_L_arr)
    
    elapsed = time.time() - start_time
    print(f"\nData generation: {elapsed:.1f}s")
    print(f"Total samples: {len(all_t)}")
    
    # ========================================================================
    # PCA-FSS ANALYSIS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PCA-FSS RESULTS")
    print("=" * 70)
    
    feature_names_inv = ['m_potts', 'var_local_m', 'boundary', 'corr',
                         'mean_grad', 'var_grad']
    feature_names_op = ['sigma_1', 'sigma_2', '|sigma|',
                        'm_potts', 'var_local_m', 'boundary', 'corr',
                        'mean_grad', 'var_grad']
    feature_names_norm = ['|sigma|', 'm_potts', 'var_local_m', 'boundary',
                          'corr', 'mean_grad', 'var_grad']
    
    print("\n--- A) S3-INVARIANT FEATURES ONLY (control) ---")
    res_inv = run_pca_fss(features_inv, all_t, all_L_arr, L_values, "invariant")
    print(f"  nu_opt = {res_inv['nu_opt']:.4f} (error: {res_inv['error_pct']:.1f}%)")
    print(f"  Expected: ~1.67 (2 × 5/6)")
    print(f"  Explained variance (PC1-3): "
          f"{res_inv['explained_variance'][:3]}")
    print(f"  PC1 loadings: {dict(zip(feature_names_inv, [f'{x:.3f}' for x in res_inv['loadings_pc1']]))}")
    
    print("\n--- B) WITH OP COMPONENTS σ₁, σ₂ (THE CRITICAL TEST) ---")
    res_op = run_pca_fss(features_op, all_t, all_L_arr, L_values, "with_op")
    print(f"  nu_opt = {res_op['nu_opt']:.4f} (error: {res_op['error_pct']:.1f}%)")
    print(f"  Expected if theorem TRUE:  ~0.833 (5/6)")
    print(f"  Expected if theorem FALSE: ~1.67")
    print(f"  Explained variance (PC1-3): "
          f"{res_op['explained_variance'][:3]}")
    print(f"  PC1 loadings: {dict(zip(feature_names_op, [f'{x:.3f}' for x in res_op['loadings_pc1']]))}")
    
    # Check if sigma_1, sigma_2 dominate PC1
    sigma_loading = np.sqrt(res_op['loadings_pc1'][0]**2 + res_op['loadings_pc1'][1]**2)
    inv_loading = np.sqrt(sum(x**2 for x in res_op['loadings_pc1'][3:]))
    print(f"  |σ components| loading on PC1: {sigma_loading:.3f}")
    print(f"  |invariant features| loading on PC1: {inv_loading:.3f}")
    
    print("\n--- C) |σ| ONLY (S3-invariant, but linear in |OP|) ---")
    res_norm = run_pca_fss(features_norm, all_t, all_L_arr, L_values, "norm_only")
    print(f"  nu_opt = {res_norm['nu_opt']:.4f} (error: {res_norm['error_pct']:.1f}%)")
    print(f"  Explained variance (PC1-3): "
          f"{res_norm['explained_variance'][:3]}")
    print(f"  PC1 loadings: {dict(zip(feature_names_norm, [f'{x:.3f}' for x in res_norm['loadings_pc1']]))}")
    
    # ========================================================================
    # VERDICT
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("VERDICT: SYMMETRY-BASED SUFFICIENCY THEOREM")
    print("=" * 70)
    print()
    print(f"  Feature Set              | nu_opt  | Error   | vs Exact 5/6")
    print(f"  -------------------------|---------|---------|-------------")
    for label, res in [("A: S3-invariant only", res_inv),
                        ("B: WITH OP (σ₁,σ₂)", res_op),
                        ("C: |σ| only", res_norm)]:
        ratio = res['nu_opt'] / NU_EXACT
        print(f"  {label:25s} | {res['nu_opt']:.4f}  | {res['error_pct']:5.1f}%  | {ratio:.2f}×")
    
    print()
    
    # Decision logic
    op_error = res_op['error_pct']
    inv_error = res_inv['error_pct']
    
    if op_error < 20 and inv_error > 50:
        print("🏆 THEOREM STRONGLY SUPPORTED!")
        print("   OP components (linear in sigma) recover nu ≈ 5/6")
        print("   Invariant features (quadratic) give nu ≈ 2 × 5/6")
        print("   The symmetry structure of features DETERMINES the exponent!")
        verdict = "SUPPORTED"
    elif op_error < 20 and inv_error < 30:
        print("⚠ THEOREM PARTIALLY SUPPORTED")
        print("   OP components work, but invariant features also give decent nu")
        print("   Need to investigate further")
        verdict = "PARTIAL"
    elif op_error > 50 and inv_error > 50:
        print("✗ THEOREM FALSIFIED")
        print("   Neither feature set recovers nu — problem is deeper than symmetry")
        verdict = "FALSIFIED"
    elif op_error > 50:
        print("✗ THEOREM FALSIFIED")
        print("   OP components don't help — symmetry isn't the issue")
        verdict = "FALSIFIED"
    else:
        print(f"? INCONCLUSIVE (op_error={op_error:.1f}%, inv_error={inv_error:.1f}%)")
        verdict = "INCONCLUSIVE"
    
    print()
    print(f"  Ratio test: nu_inv / nu_op = {res_inv['nu_opt'] / res_op['nu_opt']:.2f}")
    print(f"  If theorem true, this should be ≈ 2.0")
    
    # Save
    results = {
        'invariant': res_inv,
        'with_op': res_op,
        'norm_only': res_norm,
        'verdict': verdict,
        'nu_exact': float(NU_EXACT),
        'ratio_inv_op': float(res_inv['nu_opt'] / res_op['nu_opt']) if res_op['nu_opt'] > 0 else None,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    main()
