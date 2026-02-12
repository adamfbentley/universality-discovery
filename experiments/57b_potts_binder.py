"""
Experiment 57b: Potts FSS with Binder Cumulant
==============================================

The Binder cumulant is the GOLD STANDARD for FSS because:
1. It's dimensionless (no beta/nu prefactor)
2. Curves for different L cross at T_c
3. The slope at crossing scales as L^(1/nu)

For 3-state Potts:
  U_L = 1 - <m^4> / (3 <m^2>^2)  [normalized for q=3]

At T_c, U_L(T_c) is L-independent → crossing point
Near T_c: dU_L/dT ~ L^(1/nu)

This should DEFINITIVELY recover nu = 5/6.

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import interp1d
from scipy.stats import linregress
import pickle
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Run pilot mode')
args = parser.parse_args()

PILOT = args.pilot

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_T': 15,
        'T_range': (0.92, 1.08),
        'n_samples': 200,
        'n_eq_base': 500,
        'n_decorr_base': 50,
        'output_dir': 'results_exp57b_pilot',
    }
else:
    CONFIG = {
        'L_values': [16, 24, 32, 48, 64, 96],
        'n_T': 25,
        'T_range': (0.90, 1.10),
        'n_samples': 500,
        'n_eq_base': 800,
        'n_decorr_base': 80,
        'output_dir': 'results_exp57b',
    }

# Physical parameters
T_c = 1.0 / np.log(1 + np.sqrt(3))  # 0.9949...
NU_EXACT = 5/6

# ============================================================================
# POTTS SIMULATION
# ============================================================================

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

def measure_magnetization(config, q=3):
    """Potts order parameter."""
    L = config.shape[0]
    fractions = [np.sum(config == s) / (L*L) for s in range(q)]
    max_frac = max(fractions)
    m = (q * max_frac - 1) / (q - 1)
    return m

def compute_binder_cumulant(m_samples, q=3):
    """
    Binder cumulant for q-state Potts.
    U = 1 - <m^4> / (c * <m^2>^2)
    where c = 3 for q=3 (normalization constant)
    """
    m2 = np.mean(m_samples**2)
    m4 = np.mean(m_samples**4)
    
    if m2 < 1e-10:
        return 0.0
    
    # For q=3 Potts, the normalization is c = 3
    U = 1 - m4 / (3 * m2**2)
    return U

def generate_binder_data(L, T, n_samples, n_eq, n_decorr, seed=None):
    """Generate magnetization samples and compute Binder cumulant."""
    if seed is not None:
        np.random.seed(seed)
    
    config = np.random.randint(0, 3, size=(L, L))
    
    # Equilibrate
    for _ in range(n_eq):
        config = wolff_cluster_potts(config, T)
    
    # Collect samples
    mags = []
    for _ in range(n_samples):
        for _ in range(n_decorr):
            config = wolff_cluster_potts(config, T)
        mags.append(measure_magnetization(config))
    
    mags = np.array(mags)
    U = compute_binder_cumulant(mags)
    
    return U, mags

# ============================================================================
# BINDER CUMULANT ANALYSIS
# ============================================================================

def find_crossing_temperature(L1, L2, T_values, U_data):
    """Find temperature where U(L1) = U(L2)."""
    U1 = U_data[L1]
    U2 = U_data[L2]
    
    # Interpolate
    f1 = interp1d(T_values, U1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(T_values, U2, kind='linear', fill_value='extrapolate')
    
    # Find crossing
    def diff(T):
        return f1(T) - f2(T)
    
    # Look for sign change
    T_min, T_max = T_values.min(), T_values.max()
    T_test = np.linspace(T_min, T_max, 100)
    diffs = [diff(T) for T in T_test]
    
    crossings = []
    for i in range(len(diffs)-1):
        if diffs[i] * diffs[i+1] < 0:
            try:
                T_cross = brentq(diff, T_test[i], T_test[i+1])
                crossings.append(T_cross)
            except:
                pass
    
    if crossings:
        # Return crossing closest to T_c
        return min(crossings, key=lambda T: abs(T - T_c))
    return None

def estimate_nu_from_slopes(L_values, T_values, U_data, T_cross):
    """
    Estimate nu from slope scaling.
    dU/dT at T_cross scales as L^(1/nu)
    """
    slopes = []
    
    for L in L_values:
        U = U_data[L]
        
        # Find slope at T_cross using linear fit near crossing
        idx = np.argmin(np.abs(T_values - T_cross))
        # Use 3 points around crossing
        i_min = max(0, idx - 1)
        i_max = min(len(T_values), idx + 2)
        
        T_local = T_values[i_min:i_max]
        U_local = U[i_min:i_max]
        
        if len(T_local) >= 2:
            slope, _, _, _, _ = linregress(T_local, U_local)
            slopes.append(abs(slope))
        else:
            slopes.append(np.nan)
    
    slopes = np.array(slopes)
    valid = ~np.isnan(slopes) & (slopes > 0)
    
    if np.sum(valid) < 3:
        return None, None, None
    
    # Fit log(slope) vs log(L)
    log_L = np.log(np.array(L_values)[valid])
    log_slope = np.log(slopes[valid])
    
    a, b, r, _, _ = linregress(log_L, log_slope)
    
    nu_fit = 1 / a if a > 0 else None
    
    return nu_fit, slopes, r

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment."""
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 57b: POTTS FSS WITH BINDER CUMULANT")
    print("=" * 70)
    print()
    print(f"Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"T_c = {T_c:.6f}")
    print(f"Exact nu = {NU_EXACT:.4f} = 5/6")
    print()
    print("METHOD: Binder cumulant crossing + slope scaling")
    print("  - Curves cross at T_c")
    print("  - Slope dU/dT at crossing ~ L^(1/nu)")
    print()
    
    L_values = CONFIG['L_values']
    n_T = CONFIG['n_T']
    T_min = CONFIG['T_range'][0] * T_c
    T_max = CONFIG['T_range'][1] * T_c
    T_values = np.linspace(T_min, T_max, n_T)
    
    print(f"L values: {L_values}")
    print(f"T range: [{T_min:.4f}, {T_max:.4f}] ({n_T} points)")
    print()
    
    # Generate data
    U_data = {}
    mag_data = {}
    
    start_time = time.time()
    
    for L in L_values:
        print(f"Generating data for L={L}...")
        
        n_eq = CONFIG['n_eq_base'] + 15 * L
        n_decorr = CONFIG['n_decorr_base'] + 3 * L
        
        U_at_L = []
        mags_at_L = []
        
        for j, T in enumerate(T_values):
            U, mags = generate_binder_data(
                L, T,
                n_samples=CONFIG['n_samples'],
                n_eq=n_eq,
                n_decorr=n_decorr,
                seed=42 + L * 1000 + j
            )
            U_at_L.append(U)
            mags_at_L.append(np.mean(mags))
        
        U_data[L] = np.array(U_at_L)
        mag_data[L] = np.array(mags_at_L)
        
        # Find U at T closest to T_c
        idx_Tc = np.argmin(np.abs(T_values - T_c))
        print(f"  U(T_c) = {U_data[L][idx_Tc]:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nData generation: {elapsed:.1f}s")
    
    # Find crossing temperatures
    print("\nFinding crossings...")
    crossings = []
    for i in range(len(L_values)-1):
        L1, L2 = L_values[i], L_values[i+1]
        T_cross = find_crossing_temperature(L1, L2, T_values, U_data)
        if T_cross is not None:
            crossings.append(T_cross)
            print(f"  L={L1}-{L2}: T_cross = {T_cross:.5f}")
    
    if crossings:
        T_cross_mean = np.mean(crossings)
        T_cross_std = np.std(crossings)
        print(f"\n  Mean crossing: T_c = {T_cross_mean:.5f} +/- {T_cross_std:.5f}")
        print(f"  Exact T_c:     {T_c:.5f}")
        print(f"  Error: {100*abs(T_cross_mean - T_c)/T_c:.2f}%")
    else:
        T_cross_mean = T_c
        print("\n  No crossings found, using exact T_c")
    
    # Estimate nu from slope scaling
    print("\nEstimating nu from slope scaling...")
    nu_fit, slopes, r = estimate_nu_from_slopes(L_values, T_values, U_data, T_cross_mean)
    
    if nu_fit is not None:
        error_pct = 100 * abs(nu_fit - NU_EXACT) / NU_EXACT
        
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"  Slopes: {[f'{s:.2f}' for s in slopes]}")
        print(f"  Slope fit: r = {r:.3f}")
        print()
        print(f"  Fitted nu = {nu_fit:.4f}")
        print(f"  Exact nu  = {NU_EXACT:.4f}")
        print(f"  Error     = {error_pct:.1f}%")
        print()
        
        if error_pct < 15:
            print("SUCCESS! Binder cumulant method recovers nu.")
        elif error_pct < 30:
            print(f"PARTIAL SUCCESS ({error_pct:.1f}% error)")
        else:
            print(f"STILL FAILS with {error_pct:.1f}% error")
    else:
        print("Could not estimate nu from slopes")
        nu_fit = None
        error_pct = None
    
    # Print U(T) table
    print()
    print("Binder cumulant U(T):")
    header = f"{'T':<10} " + " ".join([f"L={L:<6}" for L in L_values])
    print(header)
    print("-" * len(header))
    for i, T in enumerate(T_values):
        row = f"{T:<10.4f} "
        for L in L_values:
            row += f"{U_data[L][i]:<8.4f}"
        print(row)
    
    # Save results
    results = {
        'nu_fit': nu_fit,
        'nu_exact': NU_EXACT,
        'error_pct': error_pct,
        'T_cross_mean': T_cross_mean,
        'crossings': crossings,
        'slopes': slopes.tolist() if slopes is not None else None,
        'L_values': L_values,
        'T_values': T_values.tolist(),
        'U_data': {L: U_data[L].tolist() for L in L_values},
        'mag_data': {L: mag_data[L].tolist() for L in L_values},
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    results = main()
