"""
Experiment 56: Potts FSS with S₃-SYMMETRIC Features

THEORETICAL MOTIVATION:
Exp 55b failed (ν = 1.08, 29.9% error) because the features broke S₃ → Z₂ symmetry.

The feature `local_m = one_hot[:,:,0] - 1/q` selects spin-0 arbitrarily.
This projects the 3-state Potts onto a binary-like variable, making it look 
like an Ising-like system — and indeed ν = 1.08 ≈ 1.0 (the Ising exponent!).

HYPOTHESIS: Using S₃-SYMMETRIC features (invariant under permutation of all q states)
will recover ν = 5/6 ≈ 0.833 because they preserve the full symmetry content.

S₃-SYMMETRIC OBSERVABLES:
- These are functions of the configuration that don't depend on which label 
  is called "spin 0", "spin 1", etc.
- Physically: domain structure, not spin identity.

THE DEEP POINT:
If this works, it proves the "Observable Sufficiency Theorem":
  PCA recovers scaling fields iff Φ is G-sufficient for the symmetry group G.

If it fails with symmetric features too, the issue is deeper (not symmetry-related).
Either way, we learn something fundamental.

FEATURES (all S₃-invariant):
1. Domain wall density: fraction of neighbor pairs with different spins
2. Mean cluster size: average connected component size / L²
3. Cluster size variance: variance of connected component sizes / L⁴
4. Number of clusters: n_clusters / L² (density)
5. Largest cluster fraction: max cluster size / L²
6. Pair correlation at r=1: P(s(x) = s(x+r)) for r=1 (NN)  
7. Pair correlation at r=2: P(s(x) = s(x+r)) for r=2

Note: ALL of these are invariant under relabeling spins 0↔1↔2.
None of them "pick out" a particular spin value.

CONTROL COMPARISON:
Also run with the Exp 55b features (Z₂-projected) side by side to confirm
the symmetry-breaking explanation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy import ndimage
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Quick pilot run')
args = parser.parse_args()
PILOT_MODE = args.pilot

# ============================================================================
# 3-state Potts model parameters (exact)
# ============================================================================
T_c = 1.0 / np.log(1 + np.sqrt(3))  # ≈ 0.99497
NU_EXACT = 5/6  # ≈ 0.8333
print(f"3-state Potts: T_c = {T_c:.6f}, ν = {NU_EXACT:.4f}")

# ============================================================================
# Simulation (Wolff cluster algorithm for q-state Potts)
# ============================================================================

def simulate_potts(L, T, n_samples=30, thermalization=1000, interval=100, q=3):
    """Simulate 3-state Potts with Wolff cluster algorithm."""
    lattice = np.random.randint(0, q, size=(L, L))
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-beta)
    
    def wolff_step():
        nonlocal lattice
        x0, y0 = np.random.randint(0, L, 2)
        cluster_spin = lattice[x0, y0]
        cluster = set()
        boundary = [(x0, y0)]
        cluster.add((x0, y0))
        
        while boundary:
            x, y = boundary.pop()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx) % L, (y+dy) % L
                if (nx, ny) not in cluster and lattice[nx, ny] == cluster_spin:
                    if np.random.random() < p_add:
                        cluster.add((nx, ny))
                        boundary.append((nx, ny))
        
        new_spin = np.random.randint(0, q - 1)
        if new_spin >= cluster_spin:
            new_spin += 1
        
        for (x, y) in cluster:
            lattice[x, y] = new_spin
        
        return len(cluster)
    
    # Thermalize
    for _ in range(thermalization):
        wolff_step()
    
    # Collect samples
    configs = []
    for _ in range(n_samples):
        for _ in range(interval):
            wolff_step()
        configs.append(lattice.copy())
    
    return configs


# ============================================================================
# S₃-SYMMETRIC features (invariant under spin permutation)
# ============================================================================

def compute_symmetric_features(config, q=3):
    """
    Extract 7D S₃-symmetric features.
    
    EVERY feature here is invariant under permutation of spin labels.
    This is the key difference from Exp 55b which projected onto spin-0.
    """
    L = config.shape[0]
    N = L * L
    
    # 1. Domain wall density: fraction of bonds with different spins
    same_x = (config == np.roll(config, 1, axis=0)).astype(float)
    same_y = (config == np.roll(config, 1, axis=1)).astype(float)
    domain_wall = 1.0 - 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    # 2-5. Cluster statistics (fully S₃-invariant)
    # Label connected components for EACH spin state
    all_cluster_sizes = []
    for s in range(q):
        mask = (config == s).astype(int)
        labeled, n_clusters = ndimage.label(mask)
        if n_clusters > 0:
            sizes = ndimage.sum(mask, labeled, range(1, n_clusters + 1))
            all_cluster_sizes.extend(sizes)
    
    all_cluster_sizes = np.array(all_cluster_sizes) if len(all_cluster_sizes) > 0 else np.array([1.0])
    
    mean_cluster = np.mean(all_cluster_sizes) / N
    var_cluster = np.var(all_cluster_sizes) / (N * N)
    n_clusters_density = len(all_cluster_sizes) / N
    max_cluster_frac = np.max(all_cluster_sizes) / N
    
    # 6. Nearest-neighbor pair correlation: P(s(x) = s(x+1))
    corr_r1 = 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    # 7. Next-nearest-neighbor pair correlation: P(s(x) = s(x+2))
    same_x2 = (config == np.roll(config, 2, axis=0)).astype(float)
    same_y2 = (config == np.roll(config, 2, axis=1)).astype(float)
    corr_r2 = 0.5 * (np.mean(same_x2) + np.mean(same_y2))
    
    return np.array([
        domain_wall,      # 1. Bond disorder
        mean_cluster,     # 2. Mean cluster size (normalized)
        var_cluster,      # 3. Cluster size variance (normalized)  
        n_clusters_density, # 4. Cluster density
        max_cluster_frac, # 5. Percolation-like order parameter
        corr_r1,          # 6. NN correlation
        corr_r2,          # 7. NNN correlation
    ])


# ============================================================================
# Z₂-PROJECTED features (from Exp 55b, for comparison)
# ============================================================================

def compute_projected_features(config, q=3):
    """
    Exp 55b features that BREAK S₃ → Z₂ by projecting onto spin-0.
    Included for direct comparison.
    """
    L = config.shape[0]
    
    one_hot = np.zeros((L, L, q))
    for s in range(q):
        one_hot[:, :, s] = (config == s).astype(float)
    local_m = one_hot[:, :, 0] - 1.0/q  # <-- THIS breaks S₃
    var_local_m = np.var(local_m)
    
    grad_x = np.roll(local_m, -1, axis=0) - local_m
    grad_y = np.roll(local_m, -1, axis=1) - local_m
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    mean_grad = np.mean(grad_mag)
    var_grad = np.var(grad_mag)
    
    same_x = (config == np.roll(config, 1, axis=0)).astype(float)
    same_y = (config == np.roll(config, 1, axis=1)).astype(float)
    boundary = 1.0 - 0.5 * (np.mean(same_x) + np.mean(same_y))
    corr = 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    probs = np.array([np.mean(config == s) for s in range(q)])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    labeled, n_cl = ndimage.label(config == config[0, 0])
    if n_cl > 1:
        sizes = ndimage.sum(np.ones_like(config), labeled, range(1, n_cl + 1))
        cluster_var = np.var(sizes) / (L * L)
    else:
        cluster_var = 0.0
    
    return np.array([var_local_m, mean_grad, var_grad, boundary, corr, entropy, cluster_var])


# ============================================================================
# FSS quality metric (identical to Exp 52d)
# ============================================================================

def compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu):
    """Window-based variance metric (from Exp 52d)."""
    all_xi = []
    all_pc1 = []
    
    for L in L_values:
        t_arr = np.array(t_by_L[L])
        xi = t_arr * (L ** (1.0 / nu))
        all_xi.extend(xi.tolist())
        all_pc1.extend(pc1_by_L[L])
    
    all_xi = np.array(all_xi)
    all_pc1 = np.array(all_pc1)
    
    sort_idx = np.argsort(all_xi)
    all_xi = all_xi[sort_idx]
    all_pc1 = all_pc1[sort_idx]
    
    n_windows = 20
    window_size = len(all_xi) // n_windows
    
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
    
    if n_valid == 0:
        return np.inf
    
    return total_var / n_valid


def find_optimal_nu(pc1_by_L, t_by_L, L_values, nu_range=(0.3, 2.5)):
    """Find optimal ν that minimizes collapse spread."""
    def objective(nu):
        return compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu)
    
    result = minimize_scalar(objective, bounds=nu_range, method='bounded')
    return result.x, result.fun


# ============================================================================
# Main experiment
# ============================================================================

def run_fss_analysis(feature_func, feature_label, L_values, temperatures, n_samples, T_c):
    """Run PCA + FSS analysis for a given feature function."""
    print(f"\n{'='*60}")
    print(f"  FSS Analysis: {feature_label}")
    print(f"{'='*60}")
    
    # Collect data
    all_features = []
    all_sizes = []
    all_temps = []
    
    for L in L_values:
        for T in temperatures:
            t_start = time.time()
            configs = simulate_potts(L, T, n_samples=n_samples)
            
            for config in configs:
                feat = feature_func(config)
                all_features.append(feat)
                all_sizes.append(L)
                all_temps.append(T)
            
            elapsed = time.time() - t_start
            print(f"  L={L:3d}, T={T:.4f}: {len(configs)} samples ({elapsed:.1f}s)")
    
    features = np.array(all_features)
    sizes = np.array(all_sizes)
    temps = np.array(all_temps)
    
    # PCA (once globally)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=min(3, features.shape[1]))
    pc_scores = pca.fit_transform(features_scaled)
    
    print(f"\n  PCA variance explained: {pca.explained_variance_ratio_[:3]}")
    print(f"  PC1 loadings: {pca.components_[0]}")
    
    # Organize by L
    pc1_by_L = {}
    t_by_L = {}
    for L in L_values:
        mask = sizes == L
        pc1_by_L[L] = pc_scores[mask, 0].tolist()
        t_by_L[L] = ((temps[mask] - T_c) / T_c).tolist()
    
    # ν scan
    nu_values = np.linspace(0.3, 2.5, 100)
    qualities = [compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu) for nu in nu_values]
    
    # Optimal ν
    nu_opt, q_opt = find_optimal_nu(pc1_by_L, t_by_L, L_values)
    error = abs(nu_opt - NU_EXACT) / NU_EXACT * 100
    
    print(f"\n  {'='*40}")
    print(f"  RESULT ({feature_label}):")
    print(f"  Optimal ν = {nu_opt:.3f}")
    print(f"  Exact ν   = {NU_EXACT:.4f}")
    print(f"  Error     = {error:.1f}%")
    print(f"  Quality   = {q_opt:.6f}")
    print(f"  {'='*40}")
    
    return {
        'nu_opt': nu_opt,
        'error_pct': error,
        'quality': q_opt,
        'nu_values': nu_values,
        'qualities': qualities,
        'pc1_by_L': pc1_by_L,
        't_by_L': t_by_L,
        'pca': pca,
        'features': features,
    }


def main():
    mode = "PILOT" if PILOT_MODE else "FULL"
    
    print("\n" + "="*70)
    print("Experiment 56: Potts FSS — S₃-Symmetric vs Z₂-Projected Features")
    print(f"               [{mode} MODE]")
    print(f"  Testing Observable Sufficiency Theorem")
    print("="*70)
    
    if PILOT_MODE:
        L_values = [24, 32, 48]
        n_temps = 10
        n_samples = 20
    else:
        L_values = [24, 32, 48, 64, 96]
        n_temps = 15
        n_samples = 30
    
    # Temperature window: ±15% around T_c
    t_min, t_max = -0.15, 0.15
    temperatures = T_c * (1 + np.linspace(t_min, t_max, n_temps))
    
    print(f"\nParameters:")
    print(f"  L values: {L_values}")
    print(f"  Temperatures: {n_temps} in [{temperatures[0]:.4f}, {temperatures[-1]:.4f}]")
    print(f"  Samples per (L,T): {n_samples}")
    print(f"  Total configs: {len(L_values) * n_temps * n_samples}")
    
    # ============================================================
    # RUN BOTH FEATURE SETS
    # ============================================================
    
    t0 = time.time()
    
    # 1. S₃-symmetric features (the hypothesis)
    print("\n" + "▓"*70)
    print("  PART 1: S₃-SYMMETRIC FEATURES (G-sufficient)")
    print("▓"*70)
    results_sym = run_fss_analysis(
        compute_symmetric_features, "S₃-symmetric",
        L_values, temperatures, n_samples, T_c
    )
    
    # 2. Z₂-projected features (Exp 55b control)
    print("\n" + "▓"*70)
    print("  PART 2: Z₂-PROJECTED FEATURES (symmetry-broken, Exp 55b)")
    print("▓"*70)
    results_proj = run_fss_analysis(
        compute_projected_features, "Z₂-projected",
        L_values, temperatures, n_samples, T_c
    )
    
    total_time = time.time() - t0
    
    # ============================================================
    # COMPARISON
    # ============================================================
    
    print("\n" + "="*70)
    print("  COMPARISON: Observable Sufficiency Test")
    print("="*70)
    print(f"\n  {'Feature Set':<25s} {'ν recovered':>12s} {'Error':>8s} {'Quality':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'S₃-symmetric':<25s} {results_sym['nu_opt']:>12.3f} {results_sym['error_pct']:>7.1f}% {results_sym['quality']:>10.6f}")
    print(f"  {'Z₂-projected (55b)':<25s} {results_proj['nu_opt']:>12.3f} {results_proj['error_pct']:>7.1f}% {results_proj['quality']:>10.6f}")
    print(f"  {'Exact':<25s} {NU_EXACT:>12.4f}")
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Interpretation
    sym_better = results_sym['error_pct'] < results_proj['error_pct']
    print(f"\n  INTERPRETATION:")
    if sym_better and results_sym['error_pct'] < 15:
        print(f"  ✅ S₃-symmetric features SUCCEED where Z₂-projected FAILED!")
        print(f"  ✅ This CONFIRMS the Observable Sufficiency Theorem:")
        print(f"     G-sufficient features recover ν; symmetry-broken features don't.")
    elif sym_better:
        print(f"  ⚠️ S₃-symmetric features are BETTER but still imprecise.")
        print(f"     Partial support for the theorem; may need larger L or more samples.")
    else:
        print(f"  ❌ S₃-symmetric features NOT better — the issue is deeper than symmetry.")
        print(f"     This falsifies the simple symmetry-sufficiency explanation.")
        print(f"     The failure has a different root cause (feature expressiveness, etc.).")
    
    # ============================================================
    # PLOTS
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Exp 56: Observable Sufficiency Test — Potts q=3', fontsize=14, fontweight='bold')
    
    # Row 1: S₃-symmetric
    # ν scan
    ax = axes[0, 0]
    ax.plot(results_sym['nu_values'], results_sym['qualities'], 'b-', linewidth=2)
    ax.axvline(NU_EXACT, color='r', linestyle='--', label=f'Exact ν = {NU_EXACT:.4f}')
    ax.axvline(results_sym['nu_opt'], color='g', linestyle=':', label=f'Optimal ν = {results_sym["nu_opt"]:.3f}')
    ax.set_xlabel('ν')
    ax.set_ylabel('Quality (lower=better)')
    ax.set_title('S₃-symmetric: ν scan')
    ax.legend()
    
    # PC1 vs T
    ax = axes[0, 1]
    for L in L_values:
        t_vals = np.array(results_sym['t_by_L'][L])
        pc1_vals = np.array(results_sym['pc1_by_L'][L])
        # Average over samples at each T
        unique_t = np.unique(np.round(t_vals, 5))
        mean_pc1 = [np.mean(pc1_vals[np.abs(t_vals - t) < 1e-4]) for t in unique_t]
        ax.plot(unique_t, mean_pc1, 'o-', markersize=3, label=f'L={L}')
    ax.set_xlabel('t = (T-Tc)/Tc')
    ax.set_ylabel('PC1 (mean)')
    ax.set_title('S₃-symmetric: PC1 vs reduced T')
    ax.legend()
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # Data collapse at optimal ν
    ax = axes[0, 2]
    for L in L_values:
        t_vals = np.array(results_sym['t_by_L'][L])
        pc1_vals = np.array(results_sym['pc1_by_L'][L])
        xi = t_vals * (L ** (1.0 / results_sym['nu_opt']))
        ax.scatter(xi, pc1_vals, s=5, alpha=0.3, label=f'L={L}')
    ax.set_xlabel(f'ξ = t × L^(1/{results_sym["nu_opt"]:.2f})')
    ax.set_ylabel('PC1')
    ax.set_title(f'S₃-symmetric: Collapse (ν={results_sym["nu_opt"]:.3f})')
    ax.legend()
    
    # Row 2: Z₂-projected
    ax = axes[1, 0]
    ax.plot(results_proj['nu_values'], results_proj['qualities'], 'r-', linewidth=2)
    ax.axvline(NU_EXACT, color='r', linestyle='--', label=f'Exact ν = {NU_EXACT:.4f}')
    ax.axvline(results_proj['nu_opt'], color='g', linestyle=':', label=f'Optimal ν = {results_proj["nu_opt"]:.3f}')
    ax.set_xlabel('ν')
    ax.set_ylabel('Quality (lower=better)')
    ax.set_title('Z₂-projected: ν scan')
    ax.legend()
    
    ax = axes[1, 1]
    for L in L_values:
        t_vals = np.array(results_proj['t_by_L'][L])
        pc1_vals = np.array(results_proj['pc1_by_L'][L])
        unique_t = np.unique(np.round(t_vals, 5))
        mean_pc1 = [np.mean(pc1_vals[np.abs(t_vals - t) < 1e-4]) for t in unique_t]
        ax.plot(unique_t, mean_pc1, 'o-', markersize=3, label=f'L={L}')
    ax.set_xlabel('t = (T-Tc)/Tc')
    ax.set_ylabel('PC1 (mean)')
    ax.set_title('Z₂-projected: PC1 vs reduced T')
    ax.legend()
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    ax = axes[1, 2]
    for L in L_values:
        t_vals = np.array(results_proj['t_by_L'][L])
        pc1_vals = np.array(results_proj['pc1_by_L'][L])
        xi = t_vals * (L ** (1.0 / results_proj['nu_opt']))
        ax.scatter(xi, pc1_vals, s=5, alpha=0.3, label=f'L={L}')
    ax.set_xlabel(f'ξ = t × L^(1/{results_proj["nu_opt"]:.2f})')
    ax.set_ylabel('PC1')
    ax.set_title(f'Z₂-projected: Collapse (ν={results_proj["nu_opt"]:.3f})')
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'exp56_sufficiency_test.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {fig_path}")
    plt.close()
    
    # Save data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp56_data.npz')
    np.savez(data_path,
        sym_nu=results_sym['nu_opt'],
        sym_error=results_sym['error_pct'],
        proj_nu=results_proj['nu_opt'],
        proj_error=results_proj['error_pct'],
        L_values=L_values,
    )
    print(f"  Data saved: {data_path}")
    
    return results_sym, results_proj


if __name__ == '__main__':
    main()
