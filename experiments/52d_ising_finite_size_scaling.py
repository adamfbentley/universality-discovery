"""
Experiment 52d: Ising Finite-Size Scaling Test (GOLD STANDARD)

QUESTION: Does PC1 show proper finite-size scaling collapse?

If PC1 genuinely captures RG-relevant structure, then:
1. PC1 values at different L should collapse when plotted vs t * L^(1/ν)
2. We should be able to recover ν ≈ 1 (exact 2D Ising exponent)

This is the GOLD STANDARD test for RG relevance:
- "Finding temperature" is easy over wide T window
- But proper scaling collapse requires genuine critical structure

PHYSICS:
- Near T_c, observables scale as: O(t, L) = L^(x/ν) * f(t * L^(1/ν))
- For magnetization-like quantities, x/ν = β/ν ≈ 0.125
- The scaling variable is: ξ = t * L^(1/ν) where ν = 1 for 2D Ising

SUCCESS CRITERION:
1. Data collapse when plotting PC1 vs t * L for ν = 1
2. Optimal ν from collapse quality ≈ 1.0 (within ~10%)
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import pearsonr
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys
import json
from datetime import datetime

# Critical temperature
J = 1.0
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269
NU_EXACT = 1.0  # Exact 2D Ising exponent

print(f"2D Ising: T_c = {T_C:.4f}, ν = {NU_EXACT}")

# =============================================================================
# Ising Model Simulation (Wolff Cluster Algorithm)
# =============================================================================

@jit(nopython=True)
def wolff_step(spins, T, L):
    """Single Wolff cluster flip."""
    p_add = 1.0 - np.exp(-2.0 * J / T)
    
    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    
    cluster = np.zeros((L, L), dtype=np.int8)
    stack = [(i0, j0)]
    cluster[i0, j0] = 1
    s0 = spins[i0, j0]
    
    while len(stack) > 0:
        i, j = stack.pop()
        
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni = (i + di) % L
            nj = (j + dj) % L
            
            if cluster[ni, nj] == 0 and spins[ni, nj] == s0:
                if np.random.random() < p_add:
                    cluster[ni, nj] = 1
                    stack.append((ni, nj))
    
    for i in range(L):
        for j in range(L):
            if cluster[i, j] == 1:
                spins[i, j] *= -1
    
    return spins


@jit(nopython=True)
def run_ising(L, T, n_equilibrate, n_measure, n_measurements):
    """Run Ising simulation and collect measurements."""
    spins = np.random.choice(np.array([-1, 1]), size=(L, L)).astype(np.int8)
    
    for _ in range(n_equilibrate):
        spins = wolff_step(spins, T, L)
    
    configs = np.zeros((n_measurements, L, L), dtype=np.int8)
    for m_idx in range(n_measurements):
        for _ in range(n_measure):
            spins = wolff_step(spins, T, L)
        configs[m_idx] = spins.copy()
    
    return configs


# =============================================================================
# Feature Extraction (Non-trivial features only, as in 52b)
# =============================================================================

def extract_ising_features(configs):
    """Extract 6D features (excluding |m| and E/N)."""
    features = []
    
    for config in configs:
        L = config.shape[0]
        m = config.astype(np.float64)
        
        # Local magnetization (coarse-grained)
        block = max(2, L // 16)  # Adaptive block size
        L_coarse = L // block
        m_coarse = np.zeros((L_coarse, L_coarse))
        for i in range(L_coarse):
            for j in range(L_coarse):
                m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
        
        m_local_var = np.var(m_coarse)
        m_local_abs_mean = np.mean(np.abs(m_coarse))
        
        # Gradient
        grad_x = np.roll(m, -1, axis=0) - m
        grad_y = np.roll(m, -1, axis=1) - m
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_var = np.var(grad_mag)
        grad_mean = np.mean(grad_mag)
        
        # Domain boundary density
        n_boundaries = 0
        for i in range(L):
            for j in range(L):
                if config[i, j] != config[(i+1)%L, j]:
                    n_boundaries += 1
                if config[i, j] != config[i, (j+1)%L]:
                    n_boundaries += 1
        boundary_density = n_boundaries / (2 * L * L)
        
        # Correlation
        corr_1 = np.mean(m * np.roll(m, 1, axis=0))
        
        feat = np.array([
            m_local_var,
            m_local_abs_mean,
            grad_mean,
            grad_var,
            boundary_density,
            corr_1,
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Finite-Size Scaling Analysis
# =============================================================================

def compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu):
    """
    Compute quality of data collapse for a given ν.
    
    Lower is better (measures spread in collapsed data).
    """
    # Combine all data with scaling variable ξ = t * L^(1/ν)
    all_xi = []
    all_pc1 = []
    
    for L in L_values:
        xi = t_by_L[L] * (L ** (1.0 / nu))
        all_xi.extend(xi)
        all_pc1.extend(pc1_by_L[L])
    
    all_xi = np.array(all_xi)
    all_pc1 = np.array(all_pc1)
    
    # Sort by xi and compute local variance
    sort_idx = np.argsort(all_xi)
    all_xi = all_xi[sort_idx]
    all_pc1 = all_pc1[sort_idx]
    
    # Window-based variance (measure of collapse quality)
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


def find_optimal_nu(pc1_by_L, t_by_L, L_values, nu_range=(0.5, 2.0)):
    """Find optimal ν that minimizes collapse spread."""
    
    def objective(nu):
        return compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu)
    
    result = minimize_scalar(objective, bounds=nu_range, method='bounded')
    return result.x, result.fun


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False):
    mode = "PILOT" if pilot else "FULL"
    
    print("\n" + "="*70)
    print("Experiment 52d: Ising Finite-Size Scaling (GOLD STANDARD)")
    print(f"               [{mode} MODE]")
    print("="*70)
    
    if pilot:
        L_values = [16, 24, 32, 48]
        n_temps = 12
        n_samples = 20
        n_equilibrate = 1500
        n_measure = 300
        n_measurements = 3
    else:
        L_values = [32, 48, 64, 96]
        n_temps = 15
        n_samples = 30
        n_equilibrate = 3000
        n_measure = 500
        n_measurements = 5
    
    # Focus on critical region
    T_min = 0.85 * T_C
    T_max = 1.15 * T_C
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    print(f"\nParameters:")
    print(f"  L values: {L_values}")
    print(f"  T range: [{T_min:.3f}, {T_max:.3f}] (±15% of T_c)")
    print(f"  n_temps={n_temps}, n_samples={n_samples}")
    print(f"  Expected ν = {NU_EXACT}")
    print("="*70)
    
    # Collect data for each L
    all_data = {}  # L -> {'features': [...], 'temps': [...], 't': [...]}
    
    for L in L_values:
        print(f"\n{'='*50}")
        print(f"L = {L}")
        print(f"{'='*50}")
        
        features_L = []
        temps_L = []
        t_L = []
        
        for ti, T in enumerate(temperatures):
            t_reduced = (T - T_C) / T_C
            
            print(f"  T={T:.3f} (t={t_reduced:+.4f}): ", end="", flush=True)
            
            for s in range(n_samples):
                configs = run_ising(L, T, n_equilibrate, n_measure, n_measurements)
                feats = extract_ising_features(configs)
                feat_mean = np.mean(feats, axis=0)
                
                features_L.append(feat_mean)
                temps_L.append(T)
                t_L.append(t_reduced)
                
                if (s + 1) % 10 == 0:
                    print(".", end="", flush=True)
            
            print(" done")
        
        all_data[L] = {
            'features': np.array(features_L),
            'temps': np.array(temps_L),
            't': np.array(t_L),
        }
    
    # ==========================================================================
    # PCA on combined data (standardize globally)
    # ==========================================================================
    
    print("\n" + "="*70)
    print("PCA Analysis (combined across all L)")
    print("="*70)
    
    # Combine all features
    all_features = np.vstack([all_data[L]['features'] for L in L_values])
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    pca = PCA(n_components=6)
    pca_coords_all = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance: {pca.explained_variance_ratio_[:3]}")
    
    # Split back by L
    pc1_by_L = {}
    t_by_L = {}
    idx = 0
    for L in L_values:
        n = len(all_data[L]['features'])
        pc1_by_L[L] = pca_coords_all[idx:idx+n, 0]
        t_by_L[L] = all_data[L]['t']
        idx += n
    
    # ==========================================================================
    # Finite-Size Scaling Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("Finite-Size Scaling Analysis")
    print("="*70)
    
    # Test collapse at ν = 1 (exact value)
    quality_exact = compute_collapse_quality(pc1_by_L, t_by_L, L_values, NU_EXACT)
    print(f"\nCollapse quality at ν = {NU_EXACT}: {quality_exact:.6f}")
    
    # Find optimal ν
    nu_opt, quality_opt = find_optimal_nu(pc1_by_L, t_by_L, L_values)
    print(f"Optimal ν from data: {nu_opt:.3f} (quality: {quality_opt:.6f})")
    print(f"Deviation from exact: {abs(nu_opt - NU_EXACT) / NU_EXACT * 100:.1f}%")
    
    # Test at various ν values
    print("\nCollapse quality vs ν:")
    nu_test = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    for nu in nu_test:
        q = compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu)
        marker = "← exact" if nu == 1.0 else ""
        marker = "← optimal" if abs(nu - nu_opt) < 0.05 else marker
        print(f"  ν = {nu:.1f}: quality = {q:.6f} {marker}")
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    
    deviation = abs(nu_opt - NU_EXACT) / NU_EXACT * 100
    
    if deviation < 10:
        print(f"\n✅ SUCCESS: Optimal ν = {nu_opt:.3f} ≈ 1.0 (deviation {deviation:.1f}%)")
        print("→ PC1 shows proper finite-size scaling!")
        print("→ Method genuinely captures RG-relevant structure")
        success = True
    elif deviation < 20:
        print(f"\n⚠️ PARTIAL: Optimal ν = {nu_opt:.3f} (deviation {deviation:.1f}%)")
        print("→ Scaling collapse is imperfect but trend is correct")
        success = False
    else:
        print(f"\n❌ FAILED: Optimal ν = {nu_opt:.3f} (deviation {deviation:.1f}%)")
        print("→ PC1 does not show proper finite-size scaling")
        success = False
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    mode_str = "pilot" if pilot else "full"
    results_dir = f"results_exp52d_{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Raw PC1 vs t for each L
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    for L, color in zip(L_values, colors):
        ax.scatter(t_by_L[L], pc1_by_L[L], c=[color], label=f'L={L}', alpha=0.6, s=20)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Reduced temperature t = (T - T_c) / T_c')
    ax.set_ylabel('PC1')
    ax.set_title('Raw: PC1 vs t (different L spread)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Collapsed PC1 vs t * L^(1/ν) at ν = 1
    ax = axes[0, 1]
    for L, color in zip(L_values, colors):
        xi = t_by_L[L] * (L ** (1.0 / NU_EXACT))
        ax.scatter(xi, pc1_by_L[L], c=[color], label=f'L={L}', alpha=0.6, s=20)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Scaling variable ξ = t × L^(1/ν), ν = {NU_EXACT}')
    ax.set_ylabel('PC1')
    ax.set_title(f'Collapsed: ν = {NU_EXACT} (exact)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Collapsed at optimal ν
    ax = axes[1, 0]
    for L, color in zip(L_values, colors):
        xi = t_by_L[L] * (L ** (1.0 / nu_opt))
        ax.scatter(xi, pc1_by_L[L], c=[color], label=f'L={L}', alpha=0.6, s=20)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(f'Scaling variable ξ = t × L^(1/ν), ν = {nu_opt:.3f}')
    ax.set_ylabel('PC1')
    ax.set_title(f'Collapsed: ν = {nu_opt:.3f} (optimal from data)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Quality vs ν curve
    ax = axes[1, 1]
    nu_range = np.linspace(0.4, 2.5, 50)
    qualities = [compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu) for nu in nu_range]
    ax.plot(nu_range, qualities, 'b-', linewidth=2)
    ax.axvline(NU_EXACT, color='r', linestyle='--', label=f'Exact ν = {NU_EXACT}')
    ax.axvline(nu_opt, color='g', linestyle='--', label=f'Optimal ν = {nu_opt:.3f}')
    ax.set_xlabel('ν')
    ax.set_ylabel('Collapse quality (lower = better)')
    ax.set_title('Collapse Quality vs ν')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/finite_size_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    output = {
        'experiment': '52d',
        'description': 'Ising finite-size scaling test (gold standard)',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'L_values': L_values,
            'n_temps': n_temps,
            'n_samples': n_samples,
            'T_range': [float(T_min), float(T_max)],
        },
        'results': {
            'nu_exact': NU_EXACT,
            'nu_optimal': float(nu_opt),
            'deviation_percent': float(deviation),
            'collapse_quality_exact': float(quality_exact),
            'collapse_quality_optimal': float(quality_opt),
            'success': bool(success),
        },
        'conclusion': 'SUCCESS - proper FSS' if success else 'NEEDS_INVESTIGATION',
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")
    
    return success, nu_opt


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    
    success, nu = main(pilot=pilot)
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    if success:
        print(f"✅ PC1 shows proper finite-size scaling with ν ≈ {nu:.2f}")
        print("   This is GOLD STANDARD evidence for RG-relevant structure!")
    else:
        print(f"⚠️ Optimal ν = {nu:.2f} deviates from exact value 1.0")
        print("   Need to investigate further")
