"""
Experiment 56c: Slope Scaling Test
===================================

HYPOTHESIS (Slope Scaling Criterion):
For FSS to recover correct nu, the T-derivative of PC1 at T_c must scale as L^(1/nu).

If d(PC1)/dt ~ L^a with a != 1/nu:
    nu_fit = 1/a

For Potts (nu=5/6):
    - 1/nu = 1.2
    - If a = 0.6, then nu_fit = 1.67 (matches observation!)
    
For Ising (nu=1):
    - 1/nu = 1.0
    - If a = 1.0, then nu_fit = 1.0 (matches observation!)

TEST:
1. Generate configs at T_c ± small dT for multiple L
2. Compute PC1(T) for each L near T_c
3. Measure slope = d(PC1)/dT at T_c
4. Fit slope vs L to get exponent a

PREDICTION:
- Potts: a ~ 0.6 (explains factor-of-2 error)
- Ising: a ~ 1.0 (explains success)

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# System parameters
T_c_ising = 2.0 / np.log(1 + np.sqrt(2))  # 2.269...
T_c_potts = 1.0 / np.log(1 + np.sqrt(3))  # 0.9949...
NU_ISING = 1.0
NU_POTTS = 5/6  # 0.8333...

def wolff_cluster_ising(config, T):
    """One Wolff cluster update for 2D Ising."""
    L = config.shape[0]
    p_add = 1 - np.exp(-2/T)
    
    # Random seed spin
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
    """One Wolff cluster update for q-state Potts."""
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
    
    # Flip to random different state
    new_spin = (seed_spin + np.random.randint(1, q)) % q
    config[cluster] = new_spin
    return config

def extract_features_ising(config):
    """Features for 2D Ising (same as Exp 55b)."""
    L = config.shape[0]
    
    # 1. Magnetization
    m = np.abs(np.mean(config))
    
    # 2. Energy proxy (boundary count)
    horiz = np.sum(config != np.roll(config, 1, axis=1)) / (L*L)
    vert = np.sum(config != np.roll(config, 1, axis=0)) / (L*L)
    boundary = horiz + vert
    
    # 3. Cluster statistics
    binary = (config > 0).astype(int)
    labeled, n_clusters = ndimage.label(binary)
    cluster_frac = n_clusters / (L*L)
    
    if n_clusters > 0:
        sizes = ndimage.sum(np.ones_like(binary), labeled, range(1, n_clusters+1))
        max_cluster = np.max(sizes) / (L*L)
    else:
        max_cluster = 0.0
    
    # 4. Local field variance
    neighbors = (np.roll(config, 1, 0) + np.roll(config, -1, 0) + 
                 np.roll(config, 1, 1) + np.roll(config, -1, 1))
    local_field = config * neighbors
    field_var = np.var(local_field)
    
    return np.array([m, boundary, cluster_frac, max_cluster, field_var])

def extract_features_potts(config, q=3):
    """Features for 3-state Potts (same as Exp 55b)."""
    L = config.shape[0]
    
    # 1. Order parameter (max state fraction)
    state_fracs = [np.mean(config == s) for s in range(q)]
    max_frac = max(state_fracs)
    
    # 2. Domain wall density
    horiz_walls = np.sum(config != np.roll(config, 1, axis=1)) / (L*L)
    vert_walls = np.sum(config != np.roll(config, 1, axis=0)) / (L*L)
    wall_density = horiz_walls + vert_walls
    
    # 3. Largest cluster (most common state)
    dominant = np.argmax(state_fracs)
    binary = (config == dominant).astype(int)
    labeled, n_clusters = ndimage.label(binary)
    if n_clusters > 0:
        sizes = ndimage.sum(np.ones_like(binary), labeled, range(1, n_clusters+1))
        max_cluster = np.max(sizes) / (L*L)
    else:
        max_cluster = 0.0
    
    # 4. Cluster count (all states)
    total_clusters = 0
    for s in range(q):
        binary_s = (config == s).astype(int)
        _, n_c = ndimage.label(binary_s)
        total_clusters += n_c
    cluster_frac = total_clusters / (L*L)
    
    # 5. Local correlation
    same_horiz = np.mean(config == np.roll(config, 1, axis=1))
    same_vert = np.mean(config == np.roll(config, 1, axis=0))
    correlation = (same_horiz + same_vert) / 2
    
    return np.array([max_frac, wall_density, max_cluster, cluster_frac, correlation])

def generate_and_extract(system, L, T, n_samples=50, n_eq=500, n_decorr=50):
    """Generate samples and extract features."""
    if system == 'ising':
        config = np.random.choice([-1, 1], size=(L, L))
        wolff_fn = wolff_cluster_ising
        feature_fn = extract_features_ising
    else:  # potts
        config = np.random.randint(0, 3, size=(L, L))
        wolff_fn = lambda c, t: wolff_cluster_potts(c, t, q=3)
        feature_fn = extract_features_potts
    
    # Equilibrate
    for _ in range(n_eq):
        config = wolff_fn(config, T)
    
    # Collect samples
    features = []
    for _ in range(n_samples):
        for _ in range(n_decorr):
            config = wolff_fn(config, T)
        features.append(feature_fn(config))
    
    return np.array(features)

def measure_slope(system, L, T_c, n_samples=100, n_eq=800, n_decorr=80):
    """
    Measure d(PC1)/dT at T_c using finite difference.
    
    PC1 is computed from the full T-scan, then we measure the local slope at T_c.
    """
    # Temperature range centered on T_c
    # Use narrow range for slope measurement
    dT = 0.02 * T_c  # 2% of T_c
    T_low = T_c - dT
    T_high = T_c + dT
    T_values = np.linspace(T_low, T_high, 5)  # 5 points for linear fit
    
    print(f"    L={L}: Generating at T in [{T_low:.4f}, {T_high:.4f}]")
    
    # Collect features at each T
    all_features = []
    all_T = []
    for T in T_values:
        feats = generate_and_extract(system, L, T, n_samples=n_samples, 
                                     n_eq=n_eq, n_decorr=n_decorr)
        all_features.append(feats)
        all_T.extend([T] * len(feats))
    
    all_features = np.vstack(all_features)
    all_T = np.array(all_T)
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    pca = PCA(n_components=1)
    pc1_all = pca.fit_transform(features_scaled).flatten()
    
    # Average PC1 at each T
    pc1_means = []
    for T in T_values:
        mask = np.abs(all_T - T) < 1e-10
        pc1_means.append(np.mean(pc1_all[mask]))
    
    pc1_means = np.array(pc1_means)
    
    # Linear fit to get slope
    # Use reduced temperature t = (T - T_c) / T_c
    t_values = (T_values - T_c) / T_c
    
    slope, intercept, r, p, se = linregress(t_values, pc1_means)
    
    print(f"    L={L}: slope = {slope:.3f}, r = {r:.3f}")
    
    return np.abs(slope)  # Use absolute value for log-log fit

def main():
    """Main experiment: measure slope vs L for both systems."""
    
    # System sizes - focus on larger L for cleaner scaling
    L_values = [24, 32, 48, 64, 96]
    
    print("=" * 70)
    print("EXPERIMENT 56c: SLOPE SCALING TEST (v2 - higher stats)")
    print("=" * 70)
    print()
    print("HYPOTHESIS: |d(PC1)/dT| at T_c scales as L^a")
    print("  - If a = 1/nu, FSS works")
    print("  - If a != 1/nu, FSS gives nu_fit = 1/a")
    print()
    print(f"PREDICTIONS:")
    print(f"  Potts: a < 1.2 (1/nu), observed nu_fit = 1.66")  
    print(f"  Ising: a ~ 1.0 (1/nu), observed nu_fit = 1.07")
    print()
    
    results = {}
    
    for system, T_c, nu_exact in [('potts', T_c_potts, NU_POTTS), 
                                    ('ising', T_c_ising, NU_ISING)]:
        print(f"\n{'='*60}")
        print(f"SYSTEM: {system.upper()}")
        print(f"T_c = {T_c:.4f}, nu = {nu_exact:.4f}, 1/nu = {1/nu_exact:.4f}")
        print('='*60)
        
        slopes = []
        for L in L_values:
            slope = measure_slope(system, L, T_c, 
                                  n_samples=120, n_eq=800, n_decorr=80)
            slopes.append(slope)
        
        slopes = np.array(slopes)
        
        # Log-log fit: log(|slope|) = a * log(L) + const
        log_L = np.log(L_values)
        log_slope = np.log(slopes)
        
        a, log_c, r, p, se = linregress(log_L, log_slope)
        
        print(f"\n  RESULTS:")
        print(f"    L values: {L_values}")
        print(f"    |Slopes|: {[f'{s:.3f}' for s in slopes]}")
        print(f"    Fit: |slope| ~ L^{a:.3f} (r = {r:.3f})")
        print(f"    Expected 1/nu = {1/nu_exact:.3f}")
        print(f"    Predicted nu_fit = 1/a = {1/a:.3f}")
        
        results[system] = {
            'L_values': L_values,
            'slopes': slopes.tolist(),
            'a': a,
            'r': r,
            'predicted_nu_fit': 1/a,
            'observed_nu_fit': 1.66 if system == 'potts' else 1.07,
            'nu_exact': nu_exact
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SLOPE SCALING TEST")
    print("=" * 70)
    print()
    print(f"{'System':<10} {'a':<10} {'1/a':<10} {'Obs nu':<10} {'Exact nu':<10} {'Match?':<10}")
    print("-" * 60)
    
    for sys in ['potts', 'ising']:
        r = results[sys]
        match = "YES" if np.abs(1/r['a'] - r['observed_nu_fit']) < 0.2 else "NO"
        print(f"{sys:<10} {r['a']:<10.3f} {1/r['a']:<10.3f} "
              f"{r['observed_nu_fit']:<10.2f} {r['nu_exact']:<10.3f} {match:<10}")
    
    print()
    print("INTERPRETATION:")
    if np.abs(results['potts']['a'] - 0.6) < 0.3:
        print("  ✓ Potts a ~ 0.6: Explains factor-of-2 error in nu recovery")
    else:
        print(f"  ? Potts a = {results['potts']['a']:.2f}: Unexpected")
    
    if np.abs(results['ising']['a'] - 1.0) < 0.3:
        print("  ✓ Ising a ~ 1.0: Explains correct nu recovery")
    else:
        print(f"  ? Ising a = {results['ising']['a']:.2f}: Unexpected")
    
    return results

if __name__ == "__main__":
    results = main()
