"""
Experiment 57: Potts FSS with True Order Parameter
===================================================

QUESTION: Do generic observables fail because they're not the ORDER PARAMETER?

For 3-state Potts, the true order parameter is the magnetization:
  m = (q * max_frac - 1) / (q - 1)
  where max_frac = max(n_0, n_1, n_2) / L^2

Standard FSS: m(t, L) = L^(-beta/nu) * f(t * L^(1/nu))
             where t = (T - T_c) / T_c

TEST:
1. Use ONLY the magnetization as the feature
2. Apply FSS collapse to m(T, L) curves
3. Does this recover nu = 5/6?

If YES: The issue is that generic observables don't capture the order parameter
If NO: Something deeper is wrong

Features:
- Checkpointing for resume capability
- Data saving for analysis
- L-scaled equilibration

Author: Adam (with Claude)
Date: February 2026
"""

import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar
import pickle
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--pilot', action='store_true', help='Run pilot mode with reduced parameters')
args = parser.parse_args()

PILOT = args.pilot

if PILOT:
    CONFIG = {
        'L_values': [16, 24, 32, 48],
        'n_T': 12,
        'T_range': (0.90, 1.10),  # Fraction of T_c
        'n_samples': 50,
        'n_eq_base': 400,
        'n_decorr_base': 40,
        'output_dir': 'results_exp57_pilot',
        'checkpoint_interval': 2,  # Save after every 2 L values
    }
else:
    CONFIG = {
        'L_values': [16, 24, 32, 48, 64, 96],
        'n_T': 20,
        'T_range': (0.88, 1.12),  # Fraction of T_c
        'n_samples': 150,
        'n_eq_base': 600,
        'n_decorr_base': 60,
        'output_dir': 'results_exp57',
        'checkpoint_interval': 1,  # Save after every L value
    }

# Physical parameters
T_c = 1.0 / np.log(1 + np.sqrt(3))  # 0.9949...
NU_EXACT = 5/6  # 0.8333...
BETA = 1/9  # order parameter exponent

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
    """
    True Potts order parameter.
    m = (q * max_fraction - 1) / (q - 1)
    Ranges from 0 (disordered) to 1 (fully ordered)
    """
    L = config.shape[0]
    fractions = [np.sum(config == s) / (L*L) for s in range(q)]
    max_frac = max(fractions)
    m = (q * max_frac - 1) / (q - 1)
    return m

def generate_magnetization_samples(L, T, n_samples, n_eq, n_decorr, seed=None):
    """Generate magnetization samples at given L, T with proper equilibration."""
    if seed is not None:
        np.random.seed(seed)
    
    config = np.random.randint(0, 3, size=(L, L))
    
    # Equilibrate (scale with L^2 for critical slowing down)
    for _ in range(n_eq):
        config = wolff_cluster_potts(config, T)
    
    # Collect samples
    mags = []
    for _ in range(n_samples):
        for _ in range(n_decorr):
            config = wolff_cluster_potts(config, T)
        mags.append(measure_magnetization(config))
    
    return np.array(mags)

# ============================================================================
# FSS ANALYSIS
# ============================================================================

def fss_quality_metric(nu, L_values, T_values, mag_data):
    """
    FSS quality metric using window variance (from Exp 52d).
    
    For each window in rescaled-T space, measure variance of m values
    across different L. Good collapse = low variance.
    """
    if nu <= 0.1 or nu > 5:
        return 1e10
    
    # Rescaled temperature for each L
    scaled_data = {}
    for L in L_values:
        t_reduced = (T_values - T_c) / T_c
        x_scaled = t_reduced * (L ** (1/nu))
        scaled_data[L] = (x_scaled, mag_data[L])
    
    # Find overlap region
    x_min = max(min(scaled_data[L][0]) for L in L_values)
    x_max = min(max(scaled_data[L][0]) for L in L_values)
    
    if x_min >= x_max:
        return 1e10
    
    # Window analysis
    n_windows = 15
    window_centers = np.linspace(x_min * 0.9, x_max * 0.9, n_windows)
    window_width = 0.3 * (x_max - x_min) / n_windows
    
    total_variance = 0
    n_valid = 0
    
    for x_center in window_centers:
        values_in_window = []
        for L in L_values:
            x_vals, y_vals = scaled_data[L]
            # Find points within window
            mask = np.abs(x_vals - x_center) < window_width
            if np.any(mask):
                # Interpolate to get value at x_center
                idx = np.argmin(np.abs(x_vals - x_center))
                values_in_window.append(y_vals[idx])
        
        if len(values_in_window) >= 3:
            total_variance += np.var(values_in_window)
            n_valid += 1
    
    if n_valid == 0:
        return 1e10
    
    return total_variance / n_valid

def fit_fss(L_values, T_values, mag_data):
    """Find best nu by minimizing collapse metric."""
    def objective(nu):
        return fss_quality_metric(nu, L_values, T_values, mag_data)
    
    result = minimize_scalar(objective, bounds=(0.3, 3.0), method='bounded')
    return result.x, result.fun

# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(output_dir, L_completed, T_values, mag_data, raw_samples):
    """Save checkpoint for resume capability."""
    checkpoint_file = output_dir / 'checkpoint.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'L_completed': L_completed,
            'T_values': T_values,
            'mag_data': mag_data,
            'raw_samples': raw_samples,
            'config': CONFIG
        }, f)
    print(f"  [Checkpoint saved: {len(L_completed)} L values completed]")

def load_checkpoint(output_dir):
    """Load checkpoint if exists."""
    checkpoint_file = output_dir / 'checkpoint.pkl'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_results(output_dir, results):
    """Save final results."""
    results_file = output_dir / 'results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save summary as JSON for easy viewing
    summary = {
        'nu_fit': float(results['nu_fit']),
        'nu_exact': float(NU_EXACT),
        'error_pct': float(results['error_pct']),
        'quality': float(results['quality']),
        'L_values': [int(L) for L in results['L_values']],
        'success': bool(results['error_pct'] < 15.0)
    }
    with open(output_dir / 'summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main experiment."""
    
    # Setup output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 57: POTTS FSS WITH TRUE ORDER PARAMETER")
    print("=" * 70)
    print()
    print(f"Mode: {'PILOT' if PILOT else 'FULL'}")
    print(f"T_c = {T_c:.6f}")
    print(f"Exact nu = {NU_EXACT:.4f} = 5/6")
    print(f"Exact beta = {BETA:.4f} = 1/9")
    print()
    print("QUESTION: If we use the TRUE magnetization (not generic features),")
    print("          does FSS recover the correct nu?")
    print()
    
    L_values = CONFIG['L_values']
    n_T = CONFIG['n_T']
    T_min = CONFIG['T_range'][0] * T_c
    T_max = CONFIG['T_range'][1] * T_c
    T_values = np.linspace(T_min, T_max, n_T)
    
    print(f"L values: {L_values}")
    print(f"T range: [{T_min:.4f}, {T_max:.4f}] ({n_T} points)")
    print(f"Output: {output_dir}")
    print()
    
    # Check for checkpoint
    checkpoint = load_checkpoint(output_dir)
    if checkpoint is not None:
        print("Found checkpoint, resuming...")
        L_completed = checkpoint['L_completed']
        mag_data = checkpoint['mag_data']
        raw_samples = checkpoint['raw_samples']
        print(f"  Completed L values: {L_completed}")
    else:
        L_completed = []
        mag_data = {}
        raw_samples = {}
    
    # Generate data for remaining L values
    start_time = time.time()
    
    for i, L in enumerate(L_values):
        if L in L_completed:
            print(f"L={L}: Already completed, skipping")
            continue
        
        print(f"\nGenerating data for L={L}...")
        
        # Scale equilibration with L for critical slowing down
        # Wolff reduces slowing, but still need more for larger L
        n_eq = CONFIG['n_eq_base'] + 10 * L
        n_decorr = CONFIG['n_decorr_base'] + 2 * L
        
        mags_at_L = []
        samples_at_L = {}
        
        for j, T in enumerate(T_values):
            t0 = time.time()
            samples = generate_magnetization_samples(
                L, T, 
                n_samples=CONFIG['n_samples'],
                n_eq=n_eq, 
                n_decorr=n_decorr,
                seed=42 + L * 1000 + j
            )
            mags_at_L.append(np.mean(samples))
            samples_at_L[T] = samples
            
            if j == n_T // 2:  # Print at T_c
                print(f"  m(T_c) = {mags_at_L[-1]:.4f} (took {time.time()-t0:.1f}s)")
        
        mag_data[L] = np.array(mags_at_L)
        raw_samples[L] = samples_at_L
        L_completed.append(L)
        
        # Checkpoint
        if (i + 1) % CONFIG['checkpoint_interval'] == 0:
            save_checkpoint(output_dir, L_completed, T_values, mag_data, raw_samples)
    
    elapsed = time.time() - start_time
    print(f"\nData generation completed in {elapsed:.1f}s")
    
    # FSS fit
    print("\nFitting FSS collapse...")
    nu_fit, quality = fit_fss(L_values, T_values, mag_data)
    
    error_pct = 100 * abs(nu_fit - NU_EXACT) / NU_EXACT
    
    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  Fitted nu = {nu_fit:.4f}")
    print(f"  Exact nu  = {NU_EXACT:.4f}")
    print(f"  Error     = {error_pct:.1f}%")
    print(f"  Quality   = {quality:.6f}")
    print()
    
    if error_pct < 15:
        print("SUCCESS! Using the true order parameter recovers correct nu.")
        print("IMPLICATION: Generic observables fail because they're not")
        print("             coupled to the relevant scaling field.")
    elif error_pct < 30:
        print(f"PARTIAL SUCCESS with {error_pct:.1f}% error.")
        print("Much better than generic observables (~100% error)")
    else:
        print(f"STILL FAILS with {error_pct:.1f}% error.")
        print("The issue is NOT just about using generic observables.")
    
    # Print m(T) table
    print()
    print("Magnetization curves:")
    header = f"{'T':<10} " + " ".join([f"L={L:<6}" for L in L_values])
    print(header)
    print("-" * len(header))
    for i, T in enumerate(T_values):
        row = f"{T:<10.4f} "
        for L in L_values:
            row += f"{mag_data[L][i]:<8.4f}"
        print(row)
    
    # Save results
    results = {
        'nu_fit': nu_fit,
        'nu_exact': NU_EXACT,
        'error_pct': error_pct,
        'quality': quality,
        'L_values': L_values,
        'T_values': T_values.tolist(),
        'mag_data': {L: mag_data[L].tolist() for L in L_values},
        'config': CONFIG
    }
    save_results(output_dir, results)
    print(f"\nResults saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    results = main()
