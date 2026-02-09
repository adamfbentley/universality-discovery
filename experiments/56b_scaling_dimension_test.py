"""
Experiment 56b: Test Observable Scaling Dimension at T_c

PREDICTION: If the Observable Scaling Dimension Theorem is correct:
  - For Potts at T_c: PC1(L) ~ L^x with x > 0
  - For Ising at T_c: PC1(L) ~ const (x = 0)

The factor-of-2 error (nu_fit = 2*nu_exact for Potts) suggests x ~ 0.5-0.6.

This experiment tests the prediction directly by measuring PC1 at T_c for multiple L.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# Parameters
# ============================================================================
T_c_potts = 1.0 / np.log(1 + np.sqrt(3))  # ~0.995
T_c_ising = 2.0 / np.log(1 + np.sqrt(2))  # ~2.269

L_values = [16, 24, 32, 48, 64, 96]
n_samples = 100  # samples per L at T_c

print("="*60)
print("Exp 56b: Observable Scaling Dimension at T_c")
print("="*60)
print(f"Testing PC1(L) at T=T_c for Potts and Ising")
print(f"L values: {L_values}")
print(f"Samples per L: {n_samples}")

# ============================================================================
# Potts simulation
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
    
    for _ in range(thermalization):
        wolff_step()
    
    configs = []
    for _ in range(n_samples):
        for _ in range(interval):
            wolff_step()
        configs.append(lattice.copy())
    
    return configs

def compute_potts_features(config, q=3):
    """S3-symmetric features (same as Exp 56)."""
    L = config.shape[0]
    N = L * L
    
    same_x = (config == np.roll(config, 1, axis=0)).astype(float)
    same_y = (config == np.roll(config, 1, axis=1)).astype(float)
    domain_wall = 1.0 - 0.5 * (np.mean(same_x) + np.mean(same_y))
    
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
    
    corr_r1 = 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    same_x2 = (config == np.roll(config, 2, axis=0)).astype(float)
    same_y2 = (config == np.roll(config, 2, axis=1)).astype(float)
    corr_r2 = 0.5 * (np.mean(same_x2) + np.mean(same_y2))
    
    return np.array([domain_wall, mean_cluster, var_cluster, n_clusters_density, max_cluster_frac, corr_r1, corr_r2])

# ============================================================================
# Ising simulation
# ============================================================================

def simulate_ising(L, T, n_samples=30, thermalization=1000, interval=100):
    """Simulate 2D Ising with Wolff cluster algorithm."""
    spins = np.random.choice([-1, 1], size=(L, L))
    p_add = 1.0 - np.exp(-2.0 / T)
    
    def wolff_step():
        nonlocal spins
        x0, y0 = np.random.randint(0, L, 2)
        cluster_spin = spins[x0, y0]
        cluster = set()
        boundary = [(x0, y0)]
        cluster.add((x0, y0))
        
        while boundary:
            x, y = boundary.pop()
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = (x+dx) % L, (y+dy) % L
                if (nx, ny) not in cluster and spins[nx, ny] == cluster_spin:
                    if np.random.random() < p_add:
                        cluster.add((nx, ny))
                        boundary.append((nx, ny))
        
        for (x, y) in cluster:
            spins[x, y] *= -1
    
    for _ in range(thermalization):
        wolff_step()
    
    configs = []
    for _ in range(n_samples):
        for _ in range(interval):
            wolff_step()
        configs.append(spins.copy())
    
    return configs

def compute_ising_features(config):
    """Same-style features as Potts (for comparison)."""
    L = config.shape[0]
    N = L * L
    m = config.astype(float)
    
    # Domain wall = boundary density
    same_x = (config == np.roll(config, 1, axis=0)).astype(float)
    same_y = (config == np.roll(config, 1, axis=1)).astype(float)
    domain_wall = 1.0 - 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    # Cluster statistics (for +1 spins)
    mask_plus = (config == 1).astype(int)
    labeled, n_plus = ndimage.label(mask_plus)
    if n_plus > 0:
        sizes_plus = ndimage.sum(mask_plus, labeled, range(1, n_plus + 1))
    else:
        sizes_plus = [1.0]
    
    mask_minus = (config == -1).astype(int)
    labeled, n_minus = ndimage.label(mask_minus)
    if n_minus > 0:
        sizes_minus = ndimage.sum(mask_minus, labeled, range(1, n_minus + 1))
    else:
        sizes_minus = [1.0]
    
    all_sizes = list(sizes_plus) + list(sizes_minus)
    
    mean_cluster = np.mean(all_sizes) / N
    var_cluster = np.var(all_sizes) / (N * N)
    n_clusters_density = len(all_sizes) / N
    max_cluster_frac = np.max(all_sizes) / N
    
    corr_r1 = 0.5 * (np.mean(same_x) + np.mean(same_y))
    
    same_x2 = (config == np.roll(config, 2, axis=0)).astype(float)
    same_y2 = (config == np.roll(config, 2, axis=1)).astype(float)
    corr_r2 = 0.5 * (np.mean(same_x2) + np.mean(same_y2))
    
    return np.array([domain_wall, mean_cluster, var_cluster, n_clusters_density, max_cluster_frac, corr_r1, corr_r2])

# ============================================================================
# Main
# ============================================================================

def collect_data_at_Tc(system, L_values, n_samples, T_c, feature_func, sim_func):
    """Collect features at T=T_c for multiple L."""
    all_features = []
    all_L = []
    
    for L in L_values:
        t0 = time.time()
        configs = sim_func(L, T_c, n_samples=n_samples)
        
        for config in configs:
            feat = feature_func(config)
            all_features.append(feat)
            all_L.append(L)
        
        elapsed = time.time() - t0
        print(f"  {system} L={L:3d}: {n_samples} samples ({elapsed:.1f}s)")
    
    return np.array(all_features), np.array(all_L)

print("\n" + "-"*60)
print("POTTS at T_c")
print("-"*60)
potts_features, potts_L = collect_data_at_Tc(
    "Potts", L_values, n_samples, T_c_potts, 
    compute_potts_features, simulate_potts
)

print("\n" + "-"*60)
print("ISING at T_c")
print("-"*60)
ising_features, ising_L = collect_data_at_Tc(
    "Ising", L_values, n_samples, T_c_ising,
    compute_ising_features, simulate_ising
)

# ============================================================================
# PCA and PC1 vs L analysis
# ============================================================================

def analyze_pc1_vs_L(features, L_arr, L_values, name):
    """Fit PC1 ~ L^x to determine scaling dimension."""
    # PCA on all data
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(feat_scaled)
    
    # Mean PC1 at each L
    pc1_means = []
    pc1_stds = []
    for L in L_values:
        mask = L_arr == L
        pc1_L = pc[mask, 0]
        # Take absolute value (sign is arbitrary)
        pc1_means.append(np.abs(np.mean(pc1_L)))
        pc1_stds.append(np.std(pc1_L))
    
    pc1_means = np.array(pc1_means)
    pc1_stds = np.array(pc1_stds)
    
    # Log-log fit: log(|PC1|) = x * log(L) + const
    log_L = np.log(L_values)
    log_pc1 = np.log(pc1_means + 1e-10)
    
    slope, intercept, r_value, p_value, std_err = linregress(log_L, log_pc1)
    
    print(f"\n{name}:")
    print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"  |PC1| at each L: {pc1_means}")
    print(f"  Log-log fit: |PC1| ~ L^{slope:.3f} (r={r_value:.3f})")
    print(f"  Prediction: x_O/nu = {slope:.3f}")
    
    return {
        'L': L_values,
        'pc1_mean': pc1_means,
        'pc1_std': pc1_stds,
        'exponent': slope,
        'r_value': r_value,
        'pca': pca,
    }

print("\n" + "="*60)
print("PC1 SCALING AT T_c")
print("="*60)

potts_result = analyze_pc1_vs_L(potts_features, potts_L, L_values, "POTTS (nu=5/6)")
ising_result = analyze_pc1_vs_L(ising_features, ising_L, L_values, "ISING (nu=1)")

# ============================================================================
# Interpretation
# ============================================================================

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

x_potts = potts_result['exponent']
x_ising = ising_result['exponent']

print(f"""
Observable Scaling Dimension Theorem Prediction:
  - Potts should have x_O/nu > 0 (explaining nu_fit = 2*nu_exact)
  - Ising should have x_O/nu ~ 0 (explaining nu_fit ~ nu_exact)

Measured:
  - Potts: |PC1| ~ L^{x_potts:.3f}
  - Ising: |PC1| ~ L^{x_ising:.3f}

If |x_potts| >> |x_ising|, the theory is supported.
If both are similar, the theory needs revision.
""")

# ============================================================================
# Plot
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Exp 56b: Observable Scaling Dimension at T_c', fontsize=14, fontweight='bold')

# Log-log plot
ax = axes[0]
ax.loglog(L_values, potts_result['pc1_mean'], 'bo-', markersize=8, label=f'Potts: ~L^{x_potts:.2f}')
ax.loglog(L_values, ising_result['pc1_mean'], 'rs-', markersize=8, label=f'Ising: ~L^{x_ising:.2f}')
ax.set_xlabel('L (system size)')
ax.set_ylabel('|PC1| at T_c')
ax.set_title('Scaling of |PC1| with L')
ax.legend()
ax.grid(True, alpha=0.3)

# Linear plot
ax = axes[1]
ax.plot(L_values, potts_result['pc1_mean'], 'bo-', markersize=8, label='Potts')
ax.fill_between(L_values, 
                potts_result['pc1_mean'] - potts_result['pc1_std'],
                potts_result['pc1_mean'] + potts_result['pc1_std'],
                alpha=0.2, color='blue')
ax.plot(L_values, ising_result['pc1_mean'], 'rs-', markersize=8, label='Ising')
ax.fill_between(L_values,
                ising_result['pc1_mean'] - ising_result['pc1_std'],
                ising_result['pc1_mean'] + ising_result['pc1_std'],
                alpha=0.2, color='red')
ax.set_xlabel('L (system size)')
ax.set_ylabel('|PC1| at T_c')
ax.set_title('Linear scale')
ax.legend()

# Comparison
ax = axes[2]
ax.bar(['Potts\n(nu=5/6)', 'Ising\n(nu=1)'], [x_potts, x_ising], 
       color=['blue', 'red'], alpha=0.7)
ax.axhline(0, color='gray', linestyle='--')
ax.set_ylabel('Scaling exponent x = d(log|PC1|)/d(log L)')
ax.set_title('Observable Scaling Dimension')

plt.tight_layout()
fig_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'exp56b_scaling_dimension.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {fig_path}")
plt.close()

print("\n" + "="*60)
print("DONE")
print("="*60)
