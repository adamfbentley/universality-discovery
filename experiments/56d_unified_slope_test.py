"""
Experiment 56d: Unified Slope Scaling Test
==========================================

ISSUE WITH EXP 56c:
PC1 direction changes between independent fits at different L.
Comparing slopes is meaningless when the coordinate system changes.

FIX:
Train PCA on ALL L at T_c, get unified PC1 direction.
Then project all data onto this fixed PC1 and measure slopes.

HYPOTHESIS (Slope Scaling Criterion):
For FSS collapse with exponent nu, the T-sensitivity of PC1 at T_c must scale as L^(1/nu).

If d(PC1)/dt at t=0 scales as L^a with a != 1/nu:
    nu_fit = 1/a

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
    local_field = config * neighbors
    field_var = np.var(local_field)
    
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

def generate_samples(system, L, T, n_samples=50, n_eq=500, n_decorr=50):
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

def unified_slope_test(system, L_values, T_c, n_samples=100, n_eq=800, n_decorr=80):
    """
    Unified slope scaling test:
    1. Train PCA on T_c data from ALL L together
    2. For each L, measure d(PC1)/dT using this fixed PCA direction
    3. Fit slope vs L
    """
    print(f"\n  Step 1: Generating T_c data for all L to train unified PCA...")
    
    # Collect features at T_c for all L
    all_Tc_features = []
    for L in L_values:
        feats = generate_samples(system, L, T_c, n_samples=n_samples//2, 
                                 n_eq=n_eq, n_decorr=n_decorr)
        all_Tc_features.append(feats)
        print(f"    L={L}: collected {len(feats)} samples at T_c")
    
    # Fit StandardScaler and PCA on ALL T_c data
    all_Tc = np.vstack(all_Tc_features)
    scaler = StandardScaler()
    all_Tc_scaled = scaler.fit_transform(all_Tc)
    
    pca = PCA(n_components=1)
    pca.fit(all_Tc_scaled)
    pc1_direction = pca.components_[0]
    print(f"\n  Unified PC1 direction: {pc1_direction}")
    
    # Step 2: For each L, measure d(PC1)/dT near T_c
    print(f"\n  Step 2: Measuring d(PC1)/dT for each L...")
    
    dT_frac = 0.02  # 2% of T_c
    T_offsets = [-dT_frac, 0, dT_frac]
    
    slopes = []
    for L in L_values:
        pc1_at_T = []
        for offset in T_offsets:
            T = T_c * (1 + offset)
            feats = generate_samples(system, L, T, n_samples=n_samples, 
                                     n_eq=n_eq, n_decorr=n_decorr)
            feats_scaled = scaler.transform(feats)
            pc1_vals = feats_scaled @ pc1_direction  # Project onto fixed PC1
            pc1_at_T.append(np.mean(pc1_vals))
        
        # Reduced temperature t = (T - T_c) / T_c = offset
        t_vals = np.array(T_offsets)
        pc1_means = np.array(pc1_at_T)
        
        # Linear fit for slope
        slope, _, r, _, _ = linregress(t_vals, pc1_means)
        slopes.append(np.abs(slope))
        print(f"    L={L}: |slope| = {np.abs(slope):.3f}, r = {r:.3f}")
    
    return np.array(slopes)

def main():
    """Main experiment."""
    L_values = [16, 24, 32, 48, 64]
    
    print("=" * 70)
    print("EXPERIMENT 56d: UNIFIED SLOPE SCALING TEST")
    print("=" * 70)
    print()
    print("KEY FIX: Use a single PCA direction (trained on T_c data from all L)")
    print("         to measure slopes consistently across L.")
    print()
    print("HYPOTHESIS: |d(PC1)/dt| scales as L^a")
    print("  - If a = 1/nu, FSS works")
    print("  - If a != 1/nu, FSS gives nu_fit = 1/a")
    print()
    
    results = {}
    
    for system, T_c, nu_exact in [('potts', T_c_potts, NU_POTTS), 
                                    ('ising', T_c_ising, NU_ISING)]:
        print(f"\n{'='*60}")
        print(f"SYSTEM: {system.upper()}")
        print(f"T_c = {T_c:.4f}, nu = {nu_exact:.4f}, 1/nu = {1/nu_exact:.4f}")
        print('='*60)
        
        slopes = unified_slope_test(system, L_values, T_c, 
                                    n_samples=80, n_eq=600, n_decorr=60)
        
        # Log-log fit
        log_L = np.log(L_values)
        log_slope = np.log(slopes)
        
        a, log_c, r, p, se = linregress(log_L, log_slope)
        
        print(f"\n  SUMMARY:")
        print(f"    L values: {L_values}")
        print(f"    |Slopes|: {[f'{s:.2f}' for s in slopes]}")
        print(f"    Fit: |slope| ~ L^{a:.3f} (r = {r:.3f})")
        print(f"    Expected 1/nu = {1/nu_exact:.3f}")
        print(f"    Predicted nu_fit = 1/a = {1/a:.3f}")
        
        results[system] = {
            'L_values': L_values,
            'slopes': slopes.tolist(),
            'a': a,
            'r': r,
            '1/a': 1/a if a != 0 else np.inf,
            '1/nu': 1/nu_exact,
            'nu_exact': nu_exact
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print()
    print(f"{'System':<10} {'a':<10} {'1/a':<10} {'1/nu':<10} {'Match?':<10}")
    print("-" * 50)
    
    for sys in ['potts', 'ising']:
        r = results[sys]
        # Check if a matches 1/nu
        match = "YES" if np.abs(r['a'] - r['1/nu']) < 0.2 else "NO"
        print(f"{sys:<10} {r['a']:<10.3f} {r['1/a']:<10.3f} {r['1/nu']:<10.3f} {match:<10}")
    
    print()
    potts_a = results['potts']['a']
    ising_a = results['ising']['a']
    
    print("INTERPRETATION:")
    if potts_a < 0.8:  # Significantly less than 1/nu = 1.2
        print(f"  Potts: a = {potts_a:.2f} << 1/nu = 1.20")
        print(f"         Observable T-sensitivity scales too slowly with L")
        print(f"         This explains systematic error in FSS")
    
    if np.abs(ising_a - 1.0) < 0.3:
        print(f"  Ising: a = {ising_a:.2f} ≈ 1/nu = 1.00")
        print(f"         Observable T-sensitivity scales correctly")
        print(f"         This explains FSS success")
    
    return results

if __name__ == "__main__":
    results = main()
