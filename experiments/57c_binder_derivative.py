"""
Experiment 57c: Potts nu from Binder Cumulant Derivative
========================================================

Exp 57b showed:
- T_c crossing works perfectly (0.1% error)  
- But slope scaling gave nu = 0.40 (52% error)

ISSUE: T-grid was too coarse and samples too few for accurate slopes.

FIX:
- Fine T-grid focused around T_c (many points in narrow window)
- More samples for each point
- Compute dU/dT using polynomial fit, not 3-point difference
- Use derivative at T_c directly, not at crossing (which we know is T_c)

METHOD (textbook Binder cumulant FSS):
1. Compute U(T) for each L with dense T near T_c
2. Compute dU/dT at T_c via polynomial fit
3. Fit dU/dT(T_c) vs L: slope ~ L^(1/nu)

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

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Run pilot mode')
args = parser.parse_args()

PILOT = args.pilot

# Physical parameters
T_c = 1.0 / np.log(1 + np.sqrt(3))  # 0.9949...
NU_EXACT = 5/6

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_T': 9,             # Dense T points near T_c
        'T_half_width': 0.03, # +/- 3% of T_c
        'n_samples': 300,     # More samples per point
        'n_eq_base': 500,
        'n_decorr_base': 50,
        'output_dir': 'results_exp57c_pilot',
    }
else:
    CONFIG = {
        'L_values': [16, 24, 32, 48, 64],
        'n_T': 13,
        'T_half_width': 0.04,
        'n_samples': 600,
        'n_eq_base': 800,
        'n_decorr_base': 80,
        'output_dir': 'results_exp57c',
    }

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

def compute_binder_cumulant(m_samples):
    """Binder cumulant U = 1 - <m^4>/(3<m^2>^2)."""
    m2 = np.mean(m_samples**2)
    m4 = np.mean(m_samples**4)
    if m2 < 1e-10:
        return 0.0
    return 1 - m4 / (3 * m2**2)

def compute_binder_with_bootstrap(m_samples, n_bootstrap=50):
    """Compute Binder cumulant with bootstrap error estimate."""
    U = compute_binder_cumulant(m_samples)
    
    # Bootstrap
    n = len(m_samples)
    U_boots = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        U_boots.append(compute_binder_cumulant(m_samples[idx]))
    
    return U, np.std(U_boots)

def generate_data(L, T, n_samples, n_eq, n_decorr, seed=None):
    """Generate magnetization samples."""
    if seed is not None:
        np.random.seed(seed)
    
    config = np.random.randint(0, 3, size=(L, L))
    
    for _ in range(n_eq):
        config = wolff_cluster_potts(config, T)
    
    mags = []
    for _ in range(n_samples):
        for _ in range(n_decorr):
            config = wolff_cluster_potts(config, T)
        mags.append(measure_magnetization(config))
    
    return np.array(mags)

# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(output_dir, data):
    """Save checkpoint."""
    with open(output_dir / 'checkpoint.pkl', 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(output_dir):
    """Load checkpoint if exists."""
    cp = output_dir / 'checkpoint.pkl'
    if cp.exists():
        with open(cp, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 57c: BINDER CUMULANT DERIVATIVE FOR nu")
    print("=" * 70)
    print()
    print(f"Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"T_c = {T_c:.6f}, exact nu = {NU_EXACT:.4f}")
    print()
    
    L_values = CONFIG['L_values']
    hw = CONFIG['T_half_width']
    T_values = np.linspace(T_c * (1 - hw), T_c * (1 + hw), CONFIG['n_T'])
    
    print(f"L values: {L_values}")
    print(f"T range: [{T_values[0]:.5f}, {T_values[-1]:.5f}] ({CONFIG['n_T']} pts)")
    print(f"T spacing: dT = {(T_values[1]-T_values[0]):.5f}")
    print(f"Samples per point: {CONFIG['n_samples']}")
    print()
    
    # Check checkpoint
    checkpoint = load_checkpoint(output_dir)
    if checkpoint is not None:
        print("Resuming from checkpoint...")
        U_data = checkpoint['U_data']
        U_err = checkpoint['U_err']
        L_done = checkpoint['L_done']
    else:
        U_data = {}
        U_err = {}
        L_done = []
    
    # Generate data
    start_time = time.time()
    
    for L in L_values:
        if L in L_done:
            print(f"L={L}: already done, skipping")
            continue
        
        print(f"\nL={L}:")
        n_eq = CONFIG['n_eq_base'] + 15 * L
        n_decorr = CONFIG['n_decorr_base'] + 3 * L
        
        U_at_L = []
        Uerr_at_L = []
        
        for j, T in enumerate(T_values):
            t0 = time.time()
            mags = generate_data(L, T, CONFIG['n_samples'], n_eq, n_decorr,
                                seed=42 + L * 1000 + j)
            U, err = compute_binder_with_bootstrap(mags)
            U_at_L.append(U)
            Uerr_at_L.append(err)
            
            dt = time.time() - t0
            if j == CONFIG['n_T'] // 2:
                print(f"  T={T:.5f} (T_c): U = {U:.4f} +/- {err:.4f}  [{dt:.1f}s/pt]")
        
        U_data[L] = np.array(U_at_L)
        U_err[L] = np.array(Uerr_at_L)
        L_done.append(L)
        
        save_checkpoint(output_dir, {
            'U_data': U_data, 'U_err': U_err, 'L_done': L_done,
            'T_values': T_values, 'config': CONFIG
        })
    
    elapsed = time.time() - start_time
    print(f"\nData generation: {elapsed:.1f}s")
    
    # Compute dU/dT at T_c for each L
    print("\n" + "=" * 70)
    print("COMPUTING dU/dT AT T_c")
    print("=" * 70)
    
    # Use reduced temperature t = (T - T_c) / T_c
    t_reduced = (T_values - T_c) / T_c
    
    slopes = []
    slopes_err = []
    
    for L in L_values:
        U = U_data[L]
        
        # Polynomial fit (degree 2) to U(t) near T_c
        coeffs = np.polyfit(t_reduced, U, deg=2)
        # dU/dt at t=0 = coeffs[1] (linear coefficient)
        slope = abs(coeffs[1])
        
        # Also compute via simple linear regression for comparison
        slope_lin, _, r, _, se = linregress(t_reduced, U)
        slope_lin = abs(slope_lin)
        
        slopes.append(slope)
        slopes_err.append(se)
        
        print(f"  L={L:3d}: dU/dt = {slope:.3f} (poly2), {slope_lin:.3f} (linear, r={r:.3f})")
    
    slopes = np.array(slopes)
    
    # Fit: log(slope) = (1/nu) * log(L) + const
    log_L = np.log(L_values)
    log_slopes = np.log(slopes)
    
    inv_nu_fit, intercept, r, p, se = linregress(log_L, log_slopes)
    nu_fit = 1.0 / inv_nu_fit if inv_nu_fit > 0 else np.nan
    
    error_pct = 100 * abs(nu_fit - NU_EXACT) / NU_EXACT
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Slope scaling: dU/dt ~ L^{inv_nu_fit:.3f} (r = {r:.3f})")
    print()
    print(f"  1/nu_fit = {inv_nu_fit:.4f}")
    print(f"  nu_fit   = {nu_fit:.4f}")
    print(f"  nu_exact = {NU_EXACT:.4f}")
    print(f"  Error    = {error_pct:.1f}%")
    print()
    
    if error_pct < 15:
        print("SUCCESS!")
    elif error_pct < 30:
        print(f"PARTIAL ({error_pct:.1f}% error)")
    else:
        print(f"FAILED ({error_pct:.1f}% error)")
    
    # Print U(T) table
    print()
    print("U(T) data:")
    header = f"{'t':<10} " + " ".join([f"L={L:<8}" for L in L_values])
    print(header)
    print("-" * len(header))
    for i, t in enumerate(t_reduced):
        row = f"{t:<10.5f} "
        for L in L_values:
            row += f"{U_data[L][i]:<10.4f}"
        print(row)
    
    # Save
    results = {
        'nu_fit': float(nu_fit),
        'inv_nu_fit': float(inv_nu_fit),
        'nu_exact': float(NU_EXACT),
        'error_pct': float(error_pct),
        'slopes': slopes.tolist(),
        'r': float(r),
        'L_values': [int(L) for L in L_values],
        'T_values': T_values.tolist(),
        'U_data': {int(L): U_data[L].tolist() for L in L_values},
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'U_data'}, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    return results

if __name__ == "__main__":
    main()
