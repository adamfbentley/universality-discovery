"""
Experiment 55b: Potts FSS RERUN with corrected quality metric.

Bug fix: Exp 55 used polynomial-residual metric that has monotonic bias.
This version uses the SAME window-based variance metric that worked in Exp 52d (Ising).

Also restructures PCA to run ONCE globally (not per-ν candidate).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
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
# Simulation (copied from 55_potts_fss.py for self-containedness)
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


def compute_potts_features(config, q=3):
    """Extract 7D local features (no order parameter)."""
    L = config.shape[0]
    
    # 1. Local "magnetization" variance (using Potts encoding)
    one_hot = np.zeros((L, L, q))
    for s in range(q):
        one_hot[:, :, s] = (config == s).astype(float)
    local_m = one_hot[:, :, 0] - 1.0/q  # deviation from random
    var_local_m = np.var(local_m)
    
    # 2. Gradient magnitude
    grad_x = np.roll(local_m, -1, axis=0) - local_m
    grad_y = np.roll(local_m, -1, axis=1) - local_m
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    mean_grad = np.mean(grad_mag)
    var_grad = np.var(grad_mag)
    
    # 3. Domain boundary density
    same_x = (config == np.roll(config, 1, axis=0)).astype(float)
    same_y = (config == np.roll(config, 1, axis=1)).astype(float)
    boundary = 1.0 - 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    # 4. Nearest-neighbor correlation
    corr = 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    # 5. Local entropy
    probs = np.array([np.mean(config == s) for s in range(q)])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # 6. Cluster size variance (from connected components)
    from scipy import ndimage
    labeled, n_clusters = ndimage.label(config == config[0, 0])
    if n_clusters > 1:
        sizes = ndimage.sum(np.ones_like(config), labeled, range(1, n_clusters + 1))
        cluster_var = np.var(sizes) / (L * L)
    else:
        cluster_var = 0.0
    
    return np.array([var_local_m, mean_grad, var_grad, boundary, corr, entropy, cluster_var])


# ============================================================================
# CORRECT quality metric (from Exp 52d)
# ============================================================================

def compute_collapse_quality(pc1_by_L, t_by_L, L_values, nu):
    """
    Quality of FSS data collapse (SAME metric as Exp 52d Ising).
    Window-based variance: lower = better collapse.
    """
    all_xi = []
    all_pc1 = []
    
    for L in L_values:
        xi = t_by_L[L] * (L ** (1.0 / nu))
        all_xi.extend(xi)
        all_pc1.extend(pc1_by_L[L])
    
    all_xi = np.array(all_xi)
    all_pc1 = np.array(all_pc1)
    
    # Sort by xi
    sort_idx = np.argsort(all_xi)
    all_xi = all_xi[sort_idx]
    all_pc1 = all_pc1[sort_idx]
    
    # Window-based variance
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


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("EXPERIMENT 55b: POTTS FSS (CORRECTED METRIC)")
    print("="*70)
    print("Fix: Using Exp 52d window-variance metric instead of")
    print("     broken polynomial-residual metric from Exp 55")
    print("="*70)
    
    if PILOT_MODE:
        sizes = [24, 32, 48]
        n_temps = 9
        n_samples = 15
        therm = 500
        interval = 50
    else:
        sizes = [32, 48, 64, 96]
        n_temps = 15
        n_samples = 30
        therm = 1000
        interval = 100
    
    temp_range = 0.15
    temperatures = np.linspace(T_c * (1 - temp_range), T_c * (1 + temp_range), n_temps)
    
    total = len(sizes) * n_temps * n_samples
    print(f"\nSizes: {sizes}")
    print(f"Temperatures: {n_temps} points in [{temperatures[0]:.4f}, {temperatures[-1]:.4f}]")
    print(f"Samples: {n_samples} per (L,T), total = {total}")
    
    # ===== DATA COLLECTION =====
    print("\n" + "="*70)
    print("DATA COLLECTION")
    print("="*70)
    
    # Store per-L data (as in Exp 52d structure)
    all_data = {}
    
    for L in sizes:
        print(f"\nSystem size L = {L}")
        features_L = []
        t_L = []
        
        for T in temperatures:
            t = (T - T_c) / T_c
            print(f"  T = {T:.5f} (t = {t:+.4f})...", end=" ", flush=True)
            
            t0 = time.time()
            configs = simulate_potts(L, T, n_samples=n_samples, 
                                    thermalization=therm, interval=interval)
            
            for config in configs:
                feat = compute_potts_features(config)
                features_L.append(feat)
                t_L.append(t)
            
            elapsed = time.time() - t0
            print(f"[{elapsed:.1f}s]")
        
        all_data[L] = {
            'features': np.array(features_L),
            't': np.array(t_L),
        }
    
    # ===== PCA (GLOBAL, ONCE) =====
    print("\n" + "="*70)
    print("PCA ANALYSIS")
    print("="*70)
    
    # Combine all features globally
    all_features = np.vstack([all_data[L]['features'] for L in sizes])
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    pca = PCA(n_components=min(7, all_features.shape[1]))
    pca_all = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var*100:.2f}%")
    
    feature_names = ['Var(local_m)', '|∇m|', 'Var(|∇m|)', 'boundary', 'corr_1', 'entropy', 'cluster_var']
    print(f"\nPC1 loadings:")
    for name, loading in zip(feature_names, pca.components_[0]):
        print(f"  {name:15s}: {loading:+.4f}")
    
    # Split back into per-L dicts
    pc1_by_L = {}
    t_by_L = {}
    idx = 0
    for L in sizes:
        n = len(all_data[L]['features'])
        pc1_by_L[L] = pca_all[idx:idx+n, 0]
        t_by_L[L] = all_data[L]['t']
        idx += n
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.savez('data/exp55b_potts_data.npz',
             features=all_features,
             sizes=sizes,
             temperatures=temperatures,
             T_c=T_c,
             pc1_by_L={str(L): pc1_by_L[L] for L in sizes},
             t_by_L={str(L): t_by_L[L] for L in sizes})
    print("\n✓ Saved data: data/exp55b_potts_data.npz")
    
    # ===== RAW PC1 BEHAVIOR CHECK =====
    print("\n" + "="*70)
    print("DIAGNOSTIC: PC1 vs TEMPERATURE")
    print("="*70)
    
    fig_diag, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    for i, L in enumerate(sizes):
        # Group by temperature
        unique_t = np.unique(t_by_L[L])
        pc1_means = []
        pc1_stds = []
        for t_val in unique_t:
            mask = t_by_L[L] == t_val
            pc1_means.append(np.mean(pc1_by_L[L][mask]))
            pc1_stds.append(np.std(pc1_by_L[L][mask]))
        
        ax1.errorbar(unique_t, pc1_means, yerr=pc1_stds, 
                     marker='o', markersize=4, capsize=2, color=colors[i],
                     label=f'L={L}', alpha=0.8)
        
        # Also plot individual samples as scatter
        ax2.scatter(t_by_L[L], pc1_by_L[L], s=5, alpha=0.3, color=colors[i], label=f'L={L}')
    
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Reduced temperature t')
    ax1.set_ylabel('⟨PC1⟩')
    ax1.set_title('PC1 mean ± std vs Temperature')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Reduced temperature t')
    ax2.set_ylabel('PC1 (individual samples)')
    ax2.set_title('All PC1 samples vs Temperature')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_diag.savefig('figures/exp55b_pc1_vs_temp.png', dpi=150, bbox_inches='tight')
    print("  → Saved: figures/exp55b_pc1_vs_temp.png")
    
    # Check: Do curves separate with L? Do they cross near T_c?
    print("\nPC1 at T_c (t=0) for each L (should be size-independent if crossing):")
    for L in sizes:
        mask = np.abs(t_by_L[L]) < 0.02  # near T_c
        if mask.sum() > 0:
            print(f"  L={L:3d}: PC1 = {np.mean(pc1_by_L[L][mask]):+.4f} ± {np.std(pc1_by_L[L][mask]):.4f}")
    
    # ===== FSS COLLAPSE SCAN =====
    print("\n" + "="*70)
    print("FSS COLLAPSE SCAN (CORRECTED METRIC)")
    print("="*70)
    
    nu_candidates = np.linspace(0.5, 1.5, 41)  # Finer grid
    qualities = []
    
    print(f"Testing {len(nu_candidates)} values of ν in [{nu_candidates[0]:.2f}, {nu_candidates[-1]:.2f}]")
    print()
    
    for nu_test in nu_candidates:
        q = compute_collapse_quality(pc1_by_L, t_by_L, sizes, nu_test)
        qualities.append(q)
        
        marker = ""
        if abs(nu_test - NU_EXACT) < 0.015:
            marker = " ← EXACT"
        elif abs(nu_test - 1.0) < 0.015:
            marker = " (Ising)"
        
        print(f"  ν = {nu_test:.3f}:  quality = {q:.6f}{marker}")
    
    qualities = np.array(qualities)
    
    # Also use scipy optimizer for precise minimum
    def objective(nu):
        return compute_collapse_quality(pc1_by_L, t_by_L, sizes, nu)
    
    result = minimize_scalar(objective, bounds=(0.5, 1.5), method='bounded')
    nu_optimal = result.x
    quality_optimal = result.fun
    
    # ===== RESULTS =====
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    deviation = abs(nu_optimal - NU_EXACT) / NU_EXACT * 100
    
    print(f"Exact ν:                 {NU_EXACT:.4f}")
    print(f"Optimal ν (grid):        {nu_candidates[np.argmin(qualities)]:.4f}")
    print(f"Optimal ν (optimizer):   {nu_optimal:.4f}")
    print(f"Deviation from exact:    {deviation:.1f}%")
    print(f"Quality at exact:        {qualities[np.argmin(np.abs(nu_candidates - NU_EXACT))]:.6f}")
    print(f"Quality at optimal:      {quality_optimal:.6f}")
    print()
    
    # Quality landscape shape
    print("Quality landscape (is there a minimum?):")
    for i in range(0, len(nu_candidates), 4):
        nu = nu_candidates[i]
        q = qualities[i]
        bar = "█" * int(50 * (q - qualities.min()) / (qualities.max() - qualities.min() + 1e-10))
        marker = " ◄ exact" if abs(nu - NU_EXACT) < 0.02 else ""
        marker = " ◄ optimal" if abs(nu - nu_optimal) < 0.02 else marker
        print(f"  ν={nu:.2f}  {q:.6f}  {bar}{marker}")
    
    # ===== INTERPRETATION =====
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check if landscape is monotonic or has true minimum
    dq = np.diff(qualities)
    sign_changes = np.sum(np.diff(np.sign(dq)) != 0)
    
    if sign_changes == 0:
        print(f"\n⚠️  Quality is MONOTONIC (no true minimum found)")
        print(f"   This means FSS collapse doesn't discriminate ν for these observables")
        trend = "decreasing" if dq[0] < 0 else "increasing"
        print(f"   Quality is {trend} with ν")
    elif sign_changes <= 2:
        print(f"\n✓ Quality landscape has a clear minimum")
        if deviation < 10:
            print(f"\n✅ SUCCESS: ν_recovered = {nu_optimal:.3f} ≈ {NU_EXACT:.3f} ({deviation:.1f}% error)")
        elif deviation < 20:
            print(f"\n⚠️  PARTIAL: ν_recovered = {nu_optimal:.3f} ({deviation:.1f}% error)")
        else:
            print(f"\n❌ FAILED: ν_recovered = {nu_optimal:.3f} ({deviation:.1f}% error)")
    else:
        print(f"\n⚠️  Quality landscape is noisy ({sign_changes} sign changes)")
        print(f"   May need more statistics or larger systems")
    
    print(f"\nComparison:")
    print(f"  Exp 52d (Ising):  ν_exact=1.0,    ν_recovered=1.07   (7% error)")
    print(f"  Exp 55b (Potts):  ν_exact={NU_EXACT:.4f}, ν_recovered={nu_optimal:.4f} ({deviation:.1f}% error)")
    
    # ===== PLOTS =====
    
    # 1. Quality landscape
    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nu_candidates, qualities, 'o-', markersize=4, linewidth=2)
    ax.axvline(NU_EXACT, color='red', linestyle='--', linewidth=2, label=f'Exact ν = {NU_EXACT:.4f}')
    ax.axvline(nu_optimal, color='green', linestyle='--', linewidth=2, label=f'Optimal ν = {nu_optimal:.4f}')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='ν = 1 (Ising)')
    ax.set_xlabel('Test ν', fontsize=12)
    ax.set_ylabel('Collapse Quality (lower = better)', fontsize=12)
    ax.set_title('FSS Quality Landscape (Corrected Window-Variance Metric)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig1.savefig('figures/exp55b_nu_scan.png', dpi=150, bbox_inches='tight')
    print(f"\n  → Saved: figures/exp55b_nu_scan.png")
    
    # 2. FSS collapse at exact ν
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for plot_idx, (nu_plot, title) in enumerate([
        (NU_EXACT, f'Exact ν = {NU_EXACT:.4f}'),
        (nu_optimal, f'Optimal ν = {nu_optimal:.4f}'),
        (1.0, 'ν = 1.0 (Ising)')
    ]):
        ax = axes[plot_idx]
        for i, L in enumerate(sizes):
            xi = t_by_L[L] * (L ** (1.0 / nu_plot))
            ax.scatter(xi, pc1_by_L[L], s=8, alpha=0.4, color=colors[i], label=f'L={L}')
            
            # Also plot means
            unique_t = np.unique(t_by_L[L])
            for t_val in unique_t:
                mask = t_by_L[L] == t_val
                xi_mean = t_val * (L ** (1.0 / nu_plot))
                pc1_mean = np.mean(pc1_by_L[L][mask])
                ax.plot(xi_mean, pc1_mean, 'o', color=colors[i], markersize=6, 
                       markeredgecolor='black', markeredgewidth=0.5)
        
        q = compute_collapse_quality(pc1_by_L, t_by_L, sizes, nu_plot)
        ax.set_xlabel(f't × L^(1/ν)', fontsize=11)
        ax.set_ylabel('PC1', fontsize=11)
        ax.set_title(f'{title}\nquality = {q:.6f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('figures/exp55b_collapse_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: figures/exp55b_collapse_comparison.png")
    
    # 3. Before collapse (raw PC1 vs t)
    fig3, ax = plt.subplots(figsize=(8, 5))
    for i, L in enumerate(sizes):
        unique_t = np.unique(t_by_L[L])
        means = [np.mean(pc1_by_L[L][t_by_L[L] == t]) for t in unique_t]
        ax.plot(unique_t, means, 'o-', color=colors[i], label=f'L={L}', markersize=6)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='T_c')
    ax.set_xlabel('Reduced temperature t', fontsize=12)
    ax.set_ylabel('⟨PC1⟩', fontsize=12)
    ax.set_title('PC1 vs Temperature (Before Collapse)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig('figures/exp55b_raw_pc1.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: figures/exp55b_raw_pc1.png")
    
    print("\n" + "="*70)
    print("EXPERIMENT 55b COMPLETE")
    print("="*70)
    
    return nu_optimal, deviation


if __name__ == '__main__':
    nu_opt, dev = main()
