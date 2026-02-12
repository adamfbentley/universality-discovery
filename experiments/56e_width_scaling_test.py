"""
Experiment 56e: Transition Width Scaling
=========================================

REVISED THEORY:
The FSS fit finds nu such that curves collapse when plotted vs t*L^(1/nu).

This works when the TRANSITION WIDTH scales as L^(-1/nu).
  Width(L) = dT/T_c over which PC1 goes from "ordered value" to "disordered value"

If Width(L) ~ L^(-1/nu_eff), then nu_fit = nu_eff.

TEST:
1. For each L, measure PC1(T) across the transition
2. Define width as: |T_high - T_low| / T_c where PC1 goes from 20% to 80% of its range
3. Fit Width ~ L^(-1/nu_eff)
4. Compare nu_eff to exact nu

PREDICTION:
- Ising: nu_eff ~ 1.0 (matches exact)
- Potts: nu_eff ~ 0.6 (explains observed nu_fit = 1.66 = 1/0.6)

Author: Adam (with Claude)  
Date: February 2026
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# System parameters
T_c_ising = 2.0 / np.log(1 + np.sqrt(2))  # 2.269...
T_c_potts = 1.0 / np.log(1 + np.sqrt(3))  # 0.9949...
NU_ISING = 1.0
NU_POTTS = 5/6

def wolff_cluster_ising(config, T):
    """Wolff cluster update for 2D Ising."""
    L = config.shape[0]
    p_add = 1 - np.exp(-2/T)
    
    i, j = np.random.randint(0, L, 2)
    seed_spin = config[i, j]
    
    cluster = np.zeros((L, L), dtype=bool)
    stack = [(i, j)]
    cluster[i, j] = True
    
    while stack:
        x, y = stack.pop()
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = (x + dx) % L, (y + dy) % L
            if not cluster[nx, ny] and config[nx, ny] == seed_spin:
                if np.random.random() < p_add:
                    cluster[nx, ny] = True
                    stack.append((nx, ny))
    
    config[cluster] *= -1
    return config

def wolff_cluster_potts(config, T, q=3):
    """Wolff cluster update for q-state Potts."""
    L = config.shape[0]
    p_add = 1 - np.exp(-1/T)
    
    i, j = np.random.randint(0, L, 2)
    seed_spin = config[i, j]
    
    cluster = np.zeros((L, L), dtype=bool)
    stack = [(i, j)]
    cluster[i, j] = True
    
    while stack:
        x, y = stack.pop()
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = (x + dx) % L, (y + dy) % L
            if not cluster[nx, ny] and config[nx, ny] == seed_spin:
                if np.random.random() < p_add:
                    cluster[nx, ny] = True
                    stack.append((nx, ny))
    
    new_spin = (seed_spin + np.random.randint(1, q)) % q
    config[cluster] = new_spin
    return config

def extract_features_ising(config):
    """Features for 2D Ising."""
    L = config.shape[0]
    m = np.abs(np.mean(config))
    horiz = np.sum(config != np.roll(config, 1, axis=1)) / (L*L)
    vert = np.sum(config != np.roll(config, 1, axis=0)) / (L*L)
    boundary = horiz + vert
    binary = (config > 0).astype(int)
    labeled, n_clusters = ndimage.label(binary)
    cluster_frac = n_clusters / (L*L)
    if n_clusters > 0:
        sizes = ndimage.sum(np.ones_like(binary), labeled, range(1, n_clusters+1))
        max_cluster = np.max(sizes) / (L*L)
    else:
        max_cluster = 0.0
    neighbors = (np.roll(config, 1, 0) + np.roll(config, -1, 0) + 
                 np.roll(config, 1, 1) + np.roll(config, -1, 1))
    field_var = np.var(config * neighbors)
    return np.array([m, boundary, cluster_frac, max_cluster, field_var])

def extract_features_potts(config, q=3):
    """Features for 3-state Potts."""
    L = config.shape[0]
    state_fracs = [np.mean(config == s) for s in range(q)]
    max_frac = max(state_fracs)
    horiz_walls = np.sum(config != np.roll(config, 1, axis=1)) / (L*L)
    vert_walls = np.sum(config != np.roll(config, 1, axis=0)) / (L*L)
    wall_density = horiz_walls + vert_walls
    dominant = np.argmax(state_fracs)
    binary = (config == dominant).astype(int)
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters > 0:
        sizes = ndimage.sum(np.ones_like(binary), labeled, range(1, n_clusters+1))
        max_cluster = np.max(sizes) / (L*L)
    else:
        max_cluster = 0.0
    total_clusters = 0
    for s in range(q):
        binary_s = (config == s).astype(int)
        _, n_c = ndimage.label(binary_s)
        total_clusters += n_c
    cluster_frac = total_clusters / (L*L)
    same_horiz = np.mean(config == np.roll(config, 1, axis=1))
    same_vert = np.mean(config == np.roll(config, 1, axis=0))
    correlation = (same_horiz + same_vert) / 2
    return np.array([max_frac, wall_density, max_cluster, cluster_frac, correlation])

def generate_samples(system, L, T, n_samples, n_eq, n_decorr):
    """Generate samples at given L, T."""
    if system == 'ising':
        config = np.random.choice([-1, 1], size=(L, L))
        wolff_fn = wolff_cluster_ising
        feature_fn = extract_features_ising
    else:
        config = np.random.randint(0, 3, size=(L, L))
        wolff_fn = lambda c, t: wolff_cluster_potts(c, t, q=3)
        feature_fn = extract_features_potts
    
    for _ in range(n_eq):
        config = wolff_fn(config, T)
    
    features = []
    for _ in range(n_samples):
        for _ in range(n_decorr):
            config = wolff_fn(config, T)
        features.append(feature_fn(config))
    
    return np.array(features)

def measure_transition_width(system, L, T_c, n_T=15, n_samples=50, n_eq=500, n_decorr=50):
    """
    Measure the PC1 transition width for a given L.
    
    Width = |t_80 - t_20| where t = (T - T_c) / T_c
    and t_20, t_80 are where PC1 reaches 20% and 80% of its range.
    """
    # Temperature range: 0.8 T_c to 1.2 T_c
    T_range = np.linspace(0.85 * T_c, 1.15 * T_c, n_T)
    
    # Collect features at each T
    all_features = []
    all_T = []
    for T in T_range:
        feats = generate_samples(system, L, T, n_samples, n_eq, n_decorr)
        all_features.append(feats)
        all_T.extend([T] * len(feats))
    
    all_features = np.vstack(all_features)
    all_T = np.array(all_T)
    
    # Standardize and PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    pca = PCA(n_components=1)
    pc1_all = pca.fit_transform(features_scaled).flatten()
    
    # Compute mean PC1 at each T
    T_vals = []
    pc1_means = []
    for T in T_range:
        mask = np.abs(all_T - T) < 1e-10
        if np.sum(mask) > 0:
            T_vals.append(T)
            pc1_means.append(np.mean(pc1_all[mask]))
    
    T_vals = np.array(T_vals)
    pc1_means = np.array(pc1_means)
    
    # Ensure PC1 is monotonic (increasing with T)
    if pc1_means[-1] < pc1_means[0]:
        pc1_means = -pc1_means
    
    # Normalize PC1 to [0, 1]
    pc1_min, pc1_max = pc1_means.min(), pc1_means.max()
    pc1_norm = (pc1_means - pc1_min) / (pc1_max - pc1_min)
    
    # Reduced temperature t = (T - T_c) / T_c
    t_vals = (T_vals - T_c) / T_c
    
    # Interpolate to find t_20 and t_80
    try:
        interp_fn = interp1d(pc1_norm, t_vals, kind='linear', bounds_error=False, fill_value='extrapolate')
        t_20 = interp_fn(0.2)
        t_80 = interp_fn(0.8)
        width = np.abs(t_80 - t_20)
    except:
        width = np.nan
    
    return width, t_vals, pc1_norm

def main():
    """Main experiment: measure transition width vs L."""
    
    L_values = [16, 24, 32, 48, 64]
    
    print("=" * 70)
    print("EXPERIMENT 56e: TRANSITION WIDTH SCALING")
    print("=" * 70)
    print()
    print("THEORY: If PC1 transition width ~ L^(-1/nu_eff), then nu_fit = nu_eff")
    print()
    print("This explains WHY FSS gives a particular nu_fit:")
    print("  - Ising: width ~ L^(-1), so nu_fit ~ 1.0")
    print("  - Potts: if width ~ L^(-0.6), then nu_fit ~ 1.67")
    print()
    
    results = {}
    
    for system, T_c, nu_exact in [('potts', T_c_potts, NU_POTTS), 
                                   ('ising', T_c_ising, NU_ISING)]:
        print(f"\n{'='*60}")
        print(f"SYSTEM: {system.upper()}")
        print(f"T_c = {T_c:.4f}, nu = {nu_exact:.4f}")
        print('='*60)
        
        widths = []
        for L in L_values:
            print(f"  L={L}: ", end="", flush=True)
            width, _, _ = measure_transition_width(
                system, L, T_c, 
                n_T=12, n_samples=40, n_eq=500, n_decorr=50
            )
            widths.append(width)
            print(f"width = {width:.4f}")
        
        widths = np.array(widths)
        
        # Fit: log(width) = -1/nu_eff * log(L) + const
        log_L = np.log(L_values)
        log_width = np.log(widths)
        
        slope, intercept, r, p, se = linregress(log_L, log_width)
        nu_eff = -1 / slope  # width ~ L^(-1/nu_eff)
        
        print(f"\n  RESULTS:")
        print(f"    L values: {L_values}")
        print(f"    Widths:   {[f'{w:.4f}' for w in widths]}")
        print(f"    Fit: width ~ L^{slope:.3f} (r = {r:.3f})")
        print(f"    nu_eff = -1/slope = {nu_eff:.3f}")
        print(f"    Expected nu = {nu_exact:.3f}")
        print(f"    Error = {100*abs(nu_eff - nu_exact)/nu_exact:.1f}%")
        
        results[system] = {
            'L_values': L_values,
            'widths': widths.tolist(),
            'slope': slope,
            'r': r,
            'nu_eff': nu_eff,
            'nu_exact': nu_exact,
            'error_pct': 100*abs(nu_eff - nu_exact)/nu_exact
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: TRANSITION WIDTH SCALING")
    print("=" * 70)
    print()
    print(f"{'System':<10} {'Width~L^?':<12} {'nu_eff':<10} {'nu_exact':<10} {'Error':<10}")
    print("-" * 52)
    
    for sys in ['potts', 'ising']:
        r = results[sys]
        print(f"{sys:<10} L^{r['slope']:<9.3f} {r['nu_eff']:<10.3f} {r['nu_exact']:<10.3f} {r['error_pct']:<10.1f}%")
    
    print()
    print("INTERPRETATION:")
    
    potts_nu = results['potts']['nu_eff']
    ising_nu = results['ising']['nu_eff']
    
    if np.abs(ising_nu - 1.0) < 0.15:
        print(f"  ✓ Ising: nu_eff = {ising_nu:.2f} ≈ 1.0 → FSS WORKS")
    else:
        print(f"  ? Ising: nu_eff = {ising_nu:.2f} ≠ 1.0 → unexpected")
    
    if potts_nu > 1.0:
        print(f"  ✓ Potts: nu_eff = {potts_nu:.2f} > nu_exact = 0.83")
        print(f"           Transition is TOO SHARP for true nu")
        print(f"           This explains the wrong FSS fit!")
    
    return results

if __name__ == "__main__":
    results = main()
