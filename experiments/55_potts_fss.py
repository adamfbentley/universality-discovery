"""
Experiment 55: 3-State Potts Model Finite-Size Scaling

GOAL: Test if FSS method generalizes to systems with ν ≠ 1
- 2D q=3 Potts model has exact ν = 5/6 ≈ 0.8333...
- Same pipeline as Exp 52d (Ising), but non-trivial exponent
- Success = recovering ν within 10-15% without being told the value

CRITICAL TEST: Exp 52d recovered ν=1.07 for 2D Ising (exact ν=1.0)
- But a skeptic could argue any monotonic coordinate gives ν≈1
- Recovering ν≈0.83 unambiguously proves the method works

PHYSICS:
- 2D 3-state Potts model: H = -J Σ_⟨ij⟩ δ(s_i, s_j)
- Continuous phase transition at T_c/J ≈ 1.005181... (exact)
- Critical exponents (exact): ν = 5/6, β = 1/9, γ = 13/9
- Second-order transition (unlike q≥5 Potts which is first-order)

METHOD:
1. Wolff cluster algorithm for efficient sampling near T_c
2. 7D local observables (analogous to Exp 52b: gradients, correlations, NO order parameter)
3. PCA → PC1
4. FSS collapse: test if PC1 collapses as function of t × L^(1/ν)
5. Scan ν ∈ [0.5, 1.5], find optimal collapse quality

Date: February 10, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import linregress
import sys
import time
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Quick pilot run (fewer samples)')
args = parser.parse_args()

PILOT_MODE = args.pilot

# Potts model critical temperature (exact for q=3 in 2D)
# From exact solution: T_c/J = 1/ln(1 + sqrt(q)) for q=3
T_c = 1.0 / np.log(1 + np.sqrt(3))  # ≈ 1.00518...
print(f"3-state Potts critical temperature: T_c/J = {T_c:.6f}")

if PILOT_MODE:
    print("\n⚠️  PILOT MODE: Reduced samples for quick validation")
    print("   For production run, use: python experiments/55_potts_fss.py")

# Exact critical exponents (conformal field theory)
NU_EXACT = 5.0 / 6.0  # ≈ 0.8333...
BETA_EXACT = 1.0 / 9.0
GAMMA_EXACT = 13.0 / 9.0

print(f"Exact critical exponents: ν = {NU_EXACT:.4f}, β = {BETA_EXACT:.4f}, γ = {GAMMA_EXACT:.4f}")


def wolff_cluster_step(config, T, J=1.0, q=3):
    """
    Wolff cluster algorithm for q-state Potts model.
    More efficient than single-spin flip near T_c.
    """
    L = config.shape[0]
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta * J)  # Bond probability
    
    # Pick random seed spin
    i, j = np.random.randint(0, L, size=2)
    seed_state = config[i, j]
    
    # Build cluster via BFS
    cluster = [(i, j)]
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[i, j] = True
    stack = [(i, j)]
    
    while stack:
        x, y = stack.pop()
        
        # Check 4 neighbors
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = (x + dx) % L, (y + dy) % L
            
            if not in_cluster[nx, ny] and config[nx, ny] == seed_state:
                if np.random.rand() < p_add:
                    in_cluster[nx, ny] = True
                    cluster.append((nx, ny))
                    stack.append((nx, ny))
    
    # Flip cluster to random new state (not seed_state)
    new_state = np.random.randint(0, q)
    while new_state == seed_state:
        new_state = np.random.randint(0, q)
    
    for x, y in cluster:
        config[x, y] = new_state
    
    return len(cluster)


def simulate_potts(L, T, n_samples=30, thermalization=1000, interval=100, J=1.0, q=3):
    """
    Simulate q-state Potts model and extract configurations.
    
    Returns:
        configs: (n_samples, L, L) array of configurations
    """
    # Initialize random configuration
    config = np.random.randint(0, q, size=(L, L))
    
    # Thermalization
    for _ in range(thermalization):
        wolff_cluster_step(config, T, J=J, q=q)
    
    # Collect samples
    configs = []
    for _ in range(n_samples):
        # Decorrelation sweeps
        for _ in range(interval):
            wolff_cluster_step(config, T, J=J, q=q)
        configs.append(config.copy())
    
    return np.array(configs)


def compute_potts_features(config, q=3):
    """
    Compute local observables for Potts model.
    Analogous to Exp 52b (Ising without |m|, E/N).
    
    7D features:
    1. Var(local_m): Local magnetization variance
    2. |∇m|: Gradient magnitude (domain wall structure)
    3. Var(|∇m|): Gradient variance
    4. boundary_density: Domain wall density
    5. corr_1: Nearest-neighbor correlation
    6. entropy_local: Local entropy (diversity of states)
    7. cluster_size_var: Variance of cluster sizes (coarse-grained at scale 2)
    
    CRITICAL: Excludes global order parameter m₀ = max(N_i/N) over states i
    """
    L = config.shape[0]
    
    # 1. Local magnetization (dominant state fraction in 3×3 window)
    local_m = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            window = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    window.append(config[(i+di)%L, (j+dj)%L])
            # Dominant state fraction
            counts = np.bincount(window, minlength=q)
            local_m[i, j] = counts.max() / 9.0
    
    # 2. Gradient magnitude (boundary detection)
    grad_mag = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            center = config[i, j]
            # Count different neighbors
            diff_count = 0
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                if config[(i+di)%L, (j+dj)%L] != center:
                    diff_count += 1
            grad_mag[i, j] = diff_count / 4.0
    
    # 3. Boundary density (fraction of boundary sites)
    boundary_density = np.mean(grad_mag > 0)
    
    # 4. Nearest-neighbor correlation
    corr_sum = 0
    count = 0
    for i in range(L):
        for j in range(L):
            center = config[i, j]
            for di, dj in [(1,0), (0,1)]:  # Count each pair once
                neighbor = config[(i+di)%L, (j+dj)%L]
                corr_sum += (center == neighbor)
                count += 1
    corr_1 = corr_sum / count
    
    # 5. Local entropy (Shannon entropy of 3×3 window)
    entropy_vals = []
    for i in range(L):
        for j in range(L):
            window = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    window.append(config[(i+di)%L, (j+dj)%L])
            counts = np.bincount(window, minlength=q)
            probs = counts / 9.0
            probs = probs[probs > 0]  # Remove zeros
            entropy_vals.append(-np.sum(probs * np.log(probs)))
    entropy_local = np.mean(entropy_vals)
    
    # 6. Cluster size variance (coarse-grain to 2×2 blocks)
    block_sizes = []
    for i in range(0, L-1, 2):
        for j in range(0, L-1, 2):
            block = [config[i, j], config[i+1, j], config[i, j+1], config[i+1, j+1]]
            # Size of dominant cluster in block
            counts = np.bincount(block, minlength=q)
            block_sizes.append(counts.max())
    cluster_size_var = np.var(block_sizes)
    
    # Pack features
    features = np.array([
        np.var(local_m),           # 1. Local magnetization variance
        np.mean(grad_mag),         # 2. Mean gradient magnitude
        np.var(grad_mag),          # 3. Gradient variance
        boundary_density,          # 4. Boundary density
        corr_1,                    # 5. NN correlation
        entropy_local,             # 6. Local entropy
        cluster_size_var           # 7. Cluster size variance
    ])
    
    return features


def collect_data(sizes, temperatures, n_samples_per_temp=30, thermalization=1000, interval=100):
    """
    Collect Potts configurations and features across sizes and temperatures.
    """
    print("\n" + "="*70)
    print("DATA COLLECTION")
    print("="*70)
    
    all_features = []
    all_labels = []
    
    for L in sizes:
        print(f"\nSystem size L = {L}")
        for T in temperatures:
            t = (T - T_c) / T_c  # Reduced temperature
            print(f"  T = {T:.5f} (t = {t:+.4f})...", end=" ")
            
            start = time.time()
            configs = simulate_potts(L, T, n_samples=n_samples_per_temp, 
                                   thermalization=thermalization, interval=interval)
            
            for config in configs:
                feat = compute_potts_features(config)
                all_features.append(feat)
                all_labels.append((L, T, t))
            
            elapsed = time.time() - start
            print(f"[{elapsed:.1f}s, {len(configs)} samples]")
    
    return np.array(all_features), all_labels


def finite_size_scaling_test(features, labels, nu_test):
    """
    Test FSS hypothesis: PC1 should collapse as function of t × L^(1/ν).
    
    Returns:
        quality: Collapse quality metric (lower = better)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=1)
    pc1_vals = pca.fit_transform(features_std).flatten()
    
    # Group by (L, t) and compute mean PC1
    unique_Lt = {}
    for i, (L, T, t) in enumerate(labels):
        key = (L, t)
        if key not in unique_Lt:
            unique_Lt[key] = []
        unique_Lt[key].append(pc1_vals[i])
    
    # Extract collapsed coordinate and PC1 values
    x_collapsed = []  # t × L^(1/ν)
    y_vals = []       # Mean PC1
    L_vals = []
    
    for (L, t), pc1_list in unique_Lt.items():
        x_collapsed.append(t * (L ** (1.0 / nu_test)))
        y_vals.append(np.mean(pc1_list))
        L_vals.append(L)
    
    x_collapsed = np.array(x_collapsed)
    y_vals = np.array(y_vals)
    L_vals = np.array(L_vals)
    
    # Quality metric: Variance of residuals after fitting master curve
    # Fit polynomial to collapsed data
    from numpy.polynomial import Polynomial
    
    # Sort by x_collapsed for fitting
    sort_idx = np.argsort(x_collapsed)
    x_sort = x_collapsed[sort_idx]
    y_sort = y_vals[sort_idx]
    
    # Fit 3rd order polynomial
    poly = Polynomial.fit(x_sort, y_sort, deg=3)
    y_fit = poly(x_sort)
    
    # Residuals
    residuals = y_sort - y_fit
    quality = np.std(residuals)
    
    return quality, (x_collapsed, y_vals, L_vals, poly)


def plot_fss_collapse(x_collapsed, y_vals, L_vals, poly, nu_test, quality, sizes):
    """
    Plot FSS collapse for given ν.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Raw PC1 vs t for each L
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    for i, L in enumerate(sizes):
        mask = (L_vals == L)
        x_L = x_collapsed[mask] / (L ** (1.0/nu_test))  # Recover t
        y_L = y_vals[mask]
        ax1.plot(x_L, y_L, 'o-', color=colors[i], label=f'L={L}', markersize=6)
    
    ax1.set_xlabel('Reduced temperature t = (T - T_c) / T_c', fontsize=12)
    ax1.set_ylabel('PC1', fontsize=12)
    ax1.set_title('Before FSS Collapse', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5, label='T_c')
    
    # Right: Collapsed data
    for i, L in enumerate(sizes):
        mask = (L_vals == L)
        ax2.plot(x_collapsed[mask], y_vals[mask], 'o', color=colors[i], 
                label=f'L={L}', markersize=6)
    
    # Master curve
    x_fit = np.linspace(x_collapsed.min(), x_collapsed.max(), 200)
    y_fit = poly(x_fit)
    ax2.plot(x_fit, y_fit, 'k-', linewidth=2, label='Master curve', alpha=0.7)
    
    ax2.set_xlabel(f'Scaled variable: t × L^(1/ν)  (ν = {nu_test:.3f})', fontsize=12)
    ax2.set_ylabel('PC1', fontsize=12)
    ax2.set_title(f'FSS Collapse (quality = {quality:.4f})', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def main():
    """
    Main FSS experiment for 3-state Potts model.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 55: 3-STATE POTTS MODEL FINITE-SIZE SCALING")
    print("="*70)
    print(f"Target: Recover ν = {NU_EXACT:.4f} from unsupervised PC1 + FSS collapse")
    print("="*70)
    
    # Parameters (adjust for pilot vs full)
    if PILOT_MODE:
        sizes = [24, 32, 48]  # Smaller, fewer sizes
        n_temps = 9           # Fewer temperatures
        temp_range = 0.15
        n_samples_per_temp = 15  # Half samples
        thermalization = 500     # Half thermalization
        interval = 50            # Half decorrelation
    else:
        sizes = [32, 48, 64, 96]
        n_temps = 15
        temp_range = 0.15  # ±15% of T_c
        n_samples_per_temp = 30
        thermalization = 1000
        interval = 100
    
    temperatures = np.linspace(T_c * (1 - temp_range), T_c * (1 + temp_range), n_temps)
    
    print(f"\nParameters:")
    print(f"  Sizes: {sizes}")
    print(f"  Temperatures: {n_temps} points in [{temperatures[0]:.5f}, {temperatures[-1]:.5f}]")
    print(f"  T_c = {T_c:.5f}")
    print(f"  Samples per (L, T): {n_samples_per_temp}")
    print(f"  Thermalization: {thermalization} Wolff steps")
    print(f"  Decorrelation: {interval} Wolff steps")
    
    total_samples = len(sizes) * n_temps * n_samples_per_temp
    print(f"\nTotal configurations: {total_samples}")
    
    # Collect data
    features, labels = collect_data(sizes, temperatures, n_samples_per_temp, 
                                   thermalization, interval)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature statistics:")
    for i, name in enumerate(['Var(local_m)', '|∇m|', 'Var(|∇m|)', 'boundary', 
                             'corr_1', 'entropy', 'cluster_var']):
        print(f"  {name:15s}: mean={features[:, i].mean():8.4f}, std={features[:, i].std():8.4f}")
    
    # Save data for diagnostics
    import os
    os.makedirs('data', exist_ok=True)
    np.savez('data/exp55_potts_data.npz',
             features=features,
             labels=labels,
             sizes=sizes,
             temperatures=temperatures,
             T_c=T_c)
    print("\n✓ Saved data: data/exp55_potts_data.npz")
    
    # Scan ν values
    print("\n" + "="*70)
    print("FINITE-SIZE SCALING SCAN")
    print("="*70)
    
    nu_candidates = np.linspace(0.5, 1.5, 21)
    qualities = []
    
    print(f"Testing {len(nu_candidates)} values of ν in [{nu_candidates[0]:.2f}, {nu_candidates[-1]:.2f}]")
    print()
    
    for nu_test in nu_candidates:
        quality, _ = finite_size_scaling_test(features, labels, nu_test)
        qualities.append(quality)
        
        # Highlight exact and key values
        marker = ""
        if abs(nu_test - NU_EXACT) < 0.01:
            marker = " ← EXACT"
        elif abs(nu_test - 1.0) < 0.01:
            marker = " (Ising)"
        
        print(f"  ν = {nu_test:.3f}:  quality = {quality:.4f}{marker}")
    
    # Find optimal ν
    qualities = np.array(qualities)
    best_idx = np.argmin(qualities)
    nu_optimal = nu_candidates[best_idx]
    quality_optimal = qualities[best_idx]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Exact ν:           {NU_EXACT:.4f}")
    print(f"Optimal ν (data):  {nu_optimal:.4f}")
    print(f"Deviation:         {abs(nu_optimal - NU_EXACT) / NU_EXACT * 100:.1f}%")
    print(f"Quality:           {quality_optimal:.4f}")
    print()
    
    # Comparison with nearby values
    print("Quality comparison:")
    for offset in [-0.2, -0.1, 0.0, +0.1, +0.2]:
        nu_test = NU_EXACT + offset
        if nu_test >= nu_candidates[0] and nu_test <= nu_candidates[-1]:
            idx = np.argmin(np.abs(nu_candidates - nu_test))
            print(f"  ν = {nu_candidates[idx]:.3f}:  {qualities[idx]:.4f}")
    
    # Check if optimal is close to exact
    deviation_pct = abs(nu_optimal - NU_EXACT) / NU_EXACT * 100
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if deviation_pct < 10:
        print("✅ SUCCESS: Recovered ν within 10% of exact value!")
        print(f"   This proves FSS method generalizes beyond ν=1 (Ising)")
        print(f"   The method correctly identifies critical exponents from unsupervised coordinates.")
    elif deviation_pct < 15:
        print("⚠️  PARTIAL SUCCESS: Recovered ν within 15% of exact value")
        print(f"   Close, but may need more samples or finer temperature resolution")
    else:
        print("❌ FAILURE: Optimal ν deviates > 15% from exact value")
        print(f"   This suggests limitations in either data quality or method")
    
    print()
    print(f"Note: Exp 52d (Ising) recovered ν=1.07 for exact ν=1.0 (7% error)")
    print(f"      This experiment: {deviation_pct:.1f}% error")
    
    # Generate plots
    print("\nGenerating FSS collapse plots...")
    
    # Plot for exact ν
    _, data_exact = finite_size_scaling_test(features, labels, NU_EXACT)
    fig1 = plot_fss_collapse(*data_exact, NU_EXACT, qualities[np.argmin(np.abs(nu_candidates - NU_EXACT))], sizes)
    fig1.savefig('figures/exp55_potts_fss_exact.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: figures/exp55_potts_fss_exact.png")
    
    # Plot for optimal ν
    _, data_optimal = finite_size_scaling_test(features, labels, nu_optimal)
    fig2 = plot_fss_collapse(*data_optimal, nu_optimal, quality_optimal, sizes)
    fig2.savefig('figures/exp55_potts_fss_optimal.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: figures/exp55_potts_fss_optimal.png")
    
    # Plot quality landscape
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(nu_candidates, qualities, 'o-', markersize=6, linewidth=2)
    ax.axvline(NU_EXACT, color='red', linestyle='--', linewidth=2, label=f'Exact ν = {NU_EXACT:.4f}')
    ax.axvline(nu_optimal, color='green', linestyle='--', linewidth=2, label=f'Optimal ν = {nu_optimal:.4f}')
    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='ν = 1 (Ising)')
    ax.set_xlabel('Test ν', fontsize=12)
    ax.set_ylabel('Collapse Quality (lower = better)', fontsize=12)
    ax.set_title('FSS Quality Landscape', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig3.savefig('figures/exp55_potts_nu_scan.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: figures/exp55_potts_nu_scan.png")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return nu_optimal, quality_optimal, deviation_pct


if __name__ == '__main__':
    nu_optimal, quality, deviation = main()
