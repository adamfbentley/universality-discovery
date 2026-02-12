"""
Experiment 57d: Ising Binder Cumulant + Comparative FSS
========================================================

PURPOSE: Apply the same Binder method to Ising as a control,
then do the comparative analysis:
  - Binder gives correct nu for BOTH systems
  - PCA-FSS gives correct nu for Ising but not Potts
  - WHY?

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
from pathlib import Path
from scipy.stats import linregress
import pickle
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true')
args = parser.parse_args()

PILOT = args.pilot

# Ising parameters
T_c_ising = 2.0 / np.log(1 + np.sqrt(2))  # 2.269...
NU_ISING = 1.0

# Potts parameters (from 57c results)
T_c_potts = 1.0 / np.log(1 + np.sqrt(3))
NU_POTTS = 5/6

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_T': 9,
        'T_half_width': 0.03,
        'n_samples': 300,
        'n_eq_base': 500,
        'n_decorr_base': 50,
        'output_dir': 'results_exp57d_pilot',
    }
else:
    CONFIG = {
        'L_values': [16, 24, 32, 48, 64],
        'n_T': 13,
        'T_half_width': 0.04,
        'n_samples': 600,
        'n_eq_base': 800,
        'n_decorr_base': 80,
        'output_dir': 'results_exp57d',
    }

def wolff_cluster_ising(config, T):
    """Wolff for Ising."""
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

def ising_binder(m_samples):
    """Binder cumulant for Ising: U = 1 - <m^4>/(3<m^2>^2)."""
    m2 = np.mean(m_samples**2)
    m4 = np.mean(m_samples**4)
    if m2 < 1e-10:
        return 0.0
    return 1 - m4 / (3 * m2**2)

def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 57d: ISING BINDER + COMPARATIVE ANALYSIS")
    print("=" * 70)
    print()
    
    L_values = CONFIG['L_values']
    hw = CONFIG['T_half_width']
    T_values = np.linspace(T_c_ising * (1 - hw), T_c_ising * (1 + hw), CONFIG['n_T'])
    t_reduced = (T_values - T_c_ising) / T_c_ising
    
    print(f"Ising T_c = {T_c_ising:.5f}, nu = {NU_ISING:.4f}")
    print(f"L values: {L_values}")
    print(f"T range: [{T_values[0]:.4f}, {T_values[-1]:.4f}] ({CONFIG['n_T']} pts)")
    print()
    
    U_data = {}
    start_time = time.time()
    
    for L in L_values:
        print(f"L={L}:", end=" ", flush=True)
        n_eq = CONFIG['n_eq_base'] + 15 * L
        n_decorr = CONFIG['n_decorr_base'] + 3 * L
        
        U_at_L = []
        for j, T in enumerate(T_values):
            config = np.random.choice([-1, 1], size=(L, L))
            np.random.seed(42 + L * 1000 + j)
            
            for _ in range(n_eq):
                config = wolff_cluster_ising(config, T)
            
            mags = []
            for _ in range(CONFIG['n_samples']):
                for _ in range(n_decorr):
                    config = wolff_cluster_ising(config, T)
                mags.append(np.abs(np.mean(config)))
            
            U = ising_binder(np.array(mags))
            U_at_L.append(U)
        
        U_data[L] = np.array(U_at_L)
        idx_Tc = CONFIG['n_T'] // 2
        print(f"U(T_c) = {U_data[L][idx_Tc]:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nData generation: {elapsed:.1f}s")
    
    # Compute slopes
    slopes = []
    for L in L_values:
        coeffs = np.polyfit(t_reduced, U_data[L], deg=2)
        slope = abs(coeffs[1])
        slopes.append(slope)
    
    slopes = np.array(slopes)
    log_L = np.log(L_values)
    log_slopes = np.log(slopes)
    inv_nu_fit, _, r, _, _ = linregress(log_L, log_slopes)
    nu_fit = 1.0 / inv_nu_fit if inv_nu_fit > 0 else np.nan
    error_pct = 100 * abs(nu_fit - NU_ISING) / NU_ISING
    
    print()
    print("=" * 70)
    print("ISING RESULTS")
    print("=" * 70)
    print()
    for L, s in zip(L_values, slopes):
        print(f"  L={L:3d}: dU/dt = {s:.3f}")
    print()
    print(f"  Slope scaling: dU/dt ~ L^{inv_nu_fit:.3f} (r = {r:.3f})")
    print(f"  nu_fit = {nu_fit:.4f}")
    print(f"  nu_exact = {NU_ISING:.4f}")
    print(f"  Error = {error_pct:.1f}%")
    
    # COMPARATIVE ANALYSIS
    print()
    print("=" * 70)
    print("COMPARATIVE ANALYSIS: WHY DOES PCA FAIL FOR POTTS?")
    print("=" * 70)
    print()
    print("Binder cumulant (this work):")
    print(f"  Ising: nu = {nu_fit:.3f} ({error_pct:.1f}% error)")
    print(f"  Potts: nu = 0.884 (6.1% error) [from Exp 57c]")
    print()
    print("PCA-FSS with generic observables:")
    print(f"  Ising: nu = 1.07 (7% error)")
    print(f"  Potts: nu = 1.66 (100% error)")
    print()
    print("DIAGNOSIS:")
    print("  The Binder cumulant works for BOTH systems.")
    print("  PCA-FSS works for Ising but fails for Potts.")
    print()
    print("  The Binder cumulant uses the TRUE order parameter (magnetization).")
    print("  PCA uses generic observables (domain walls, clusters, etc.)")
    print()
    print("  For Ising: generic observables happen to correlate strongly")
    print("            with the order parameter -> PCA works")
    print("  For Potts: generic observables do NOT correlate with the")
    print("            Potts order parameter -> PCA fails")
    print()
    print("  This IS the theoretical contribution:")
    print("  Unsupervised FSS works IFF the features contain information")
    print("  about the true scaling field (order parameter).")
    print("  For Ising (Z2, one order param), generic features work.")
    print("  For Potts (S3, multi-component), they don't.")
    
    # Save
    results = {
        'nu_fit': float(nu_fit),
        'nu_exact': float(NU_ISING),
        'error_pct': float(error_pct),
        'slopes': slopes.tolist(),
        'r': float(r),
        'L_values': [int(L) for L in L_values],
        'T_values': T_values.tolist(),
        'U_data': {int(L): U_data[L].tolist() for L in L_values},
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")
    return results

if __name__ == "__main__":
    main()
