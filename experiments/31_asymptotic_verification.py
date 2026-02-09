"""
Experiment 31: Asymptotic Regime Verification

Critical Question: Are we actually in the asymptotic scaling regime?

Problem from review:
- EW skewness = -0.2988, KPZ skewness = -0.2966 (essentially identical!)
- Theory: EW → 0 (Gaussian), KPZ → -0.29 (Tracy-Widom)
- This suggests we haven't reached asymptotic regime

Test:
- Run VERY long simulations (T up to 50,000)
- Track skewness vs time for both EW and KPZ
- They should DIVERGE: EW → 0, KPZ → -0.29

Also track gradient variance separation vs time to see when it stabilizes.

Date: January 21, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from numba import jit
import time as timer

@jit(nopython=True)
def simulate_ew_step(interface, nu=1.0, D=1.0, dt=0.05):
    """Single EW step: dh/dt = ν∇²h + η"""
    L = len(interface)
    new_interface = interface.copy()
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        noise = np.sqrt(2*D*dt) * np.random.randn()
        new_interface[x] = center + nu * laplacian * dt + noise
    return new_interface

@jit(nopython=True)
def simulate_kpz_step(interface, nu=1.0, lam=1.0, D=1.0, dt=0.05):
    """Single KPZ step: dh/dt = ν∇²h + (λ/2)(∇h)² + η"""
    L = len(interface)
    new_interface = interface.copy()
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        gradient = (right - left) / 2.0
        nonlinear = lam * 0.5 * gradient**2
        noise = np.sqrt(2*D*dt) * np.random.randn()
        new_interface[x] = center + (nu * laplacian + nonlinear) * dt + noise
    return new_interface

def run_long_simulation(model, L, T_max, dt=0.05, snapshot_times=None):
    """Run simulation and collect snapshots at specified times."""
    if snapshot_times is None:
        snapshot_times = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    snapshot_times = [t for t in snapshot_times if t <= T_max]
    
    # Droplet IC (curved) - this is where we expect Tracy-Widom GUE
    x = np.arange(L)
    interface = -np.abs(x - L/2).astype(np.float64)
    interface = interface - np.mean(interface)
    
    n_steps = int(T_max / dt)
    snapshots = {}
    
    step = 0
    current_time = 0.0
    snapshot_idx = 0
    
    while step <= n_steps and snapshot_idx < len(snapshot_times):
        # Check if we should save a snapshot
        if current_time >= snapshot_times[snapshot_idx]:
            snapshots[snapshot_times[snapshot_idx]] = interface.copy()
            snapshot_idx += 1
        
        # Evolve
        if step < n_steps:
            if model == 'EW':
                interface = simulate_ew_step(interface, dt=dt)
            else:
                interface = simulate_kpz_step(interface, dt=dt)
            interface = interface - np.mean(interface)  # Remove drift
        
        step += 1
        current_time = step * dt
    
    return snapshots

def compute_statistics(interface):
    """Compute height and gradient statistics."""
    L = len(interface)
    
    # Height statistics (center point normalized)
    h_mean = np.mean(interface)
    h_std = np.std(interface)
    h_center = interface[L // 2]
    h_normalized = (h_center - h_mean) / (h_std + 1e-10)
    
    # Full height distribution stats
    h_skew = stats.skew(interface)
    
    # Gradient statistics
    grad = (np.roll(interface, -1) - np.roll(interface, 1)) / 2.0
    grad_var = np.var(grad)
    grad_skew = stats.skew(grad)
    grad_kurt = stats.kurtosis(grad)
    
    return {
        'h_center_normalized': h_normalized,
        'h_skew': h_skew,
        'h_std': h_std,
        'grad_var': grad_var,
        'grad_skew': grad_skew,
        'grad_kurt': grad_kurt
    }

def main():
    print("=" * 78)
    print("EXPERIMENT 31: ASYMPTOTIC REGIME VERIFICATION")
    print("=" * 78)
    print()
    print("Testing whether simulations reach asymptotic scaling regime")
    print("Key prediction: EW skewness → 0, KPZ skewness → -0.29")
    print()
    
    # Parameters
    L = 256  # Moderate size for speed
    T_max = 30000  # Very long time
    n_samples = 50  # Samples per model
    dt = 0.05
    
    # Snapshot times (logarithmically spaced focus on late times)
    snapshot_times = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
    
    print(f"Parameters: L={L}, T_max={T_max}, n_samples={n_samples}")
    print(f"Snapshot times: {snapshot_times}")
    print()
    
    # Results storage
    results = {t: {'EW': [], 'KPZ': []} for t in snapshot_times}
    
    # Run simulations
    for model in ['EW', 'KPZ']:
        print(f"\n{model} simulations:")
        start = timer.time()
        
        for i in range(n_samples):
            if (i + 1) % 10 == 0:
                elapsed = timer.time() - start
                print(f"  Sample {i+1}/{n_samples} ({elapsed:.1f}s elapsed)")
            
            np.random.seed(42 + i * 1000)
            snapshots = run_long_simulation(model, L, T_max, dt, snapshot_times)
            
            for t, interface in snapshots.items():
                stats_dict = compute_statistics(interface)
                results[t][model].append(stats_dict)
        
        print(f"  Completed in {timer.time() - start:.1f}s")
    
    # Analysis
    print("\n" + "=" * 78)
    print("RESULTS: SKEWNESS EVOLUTION")
    print("=" * 78)
    print(f"\n{'Time':>8} | {'EW h_skew':>12} | {'KPZ h_skew':>12} | {'Difference':>12} | {'Theory OK?':>10}")
    print("-" * 70)
    
    times = []
    ew_skews = []
    kpz_skews = []
    ew_grad_vars = []
    kpz_grad_vars = []
    
    for t in snapshot_times:
        ew_h_skew = np.mean([s['h_skew'] for s in results[t]['EW']])
        kpz_h_skew = np.mean([s['h_skew'] for s in results[t]['KPZ']])
        ew_h_skew_std = np.std([s['h_skew'] for s in results[t]['EW']])
        kpz_h_skew_std = np.std([s['h_skew'] for s in results[t]['KPZ']])
        
        ew_gv = np.mean([s['grad_var'] for s in results[t]['EW']])
        kpz_gv = np.mean([s['grad_var'] for s in results[t]['KPZ']])
        
        diff = kpz_h_skew - ew_h_skew
        
        # Theory: EW → 0, KPZ → -0.29
        # So difference should approach -0.29
        theory_ok = "Yes" if abs(ew_h_skew) < 0.15 and abs(kpz_h_skew + 0.29) < 0.1 else "No"
        
        print(f"{t:>8} | {ew_h_skew:>+.4f}±{ew_h_skew_std:.2f} | {kpz_h_skew:>+.4f}±{kpz_h_skew_std:.2f} | {diff:>+.4f} | {theory_ok:>10}")
        
        times.append(t)
        ew_skews.append(ew_h_skew)
        kpz_skews.append(kpz_h_skew)
        ew_grad_vars.append(ew_gv)
        kpz_grad_vars.append(kpz_gv)
    
    # Gradient variance ratio
    print("\n" + "=" * 78)
    print("GRADIENT VARIANCE EVOLUTION")
    print("=" * 78)
    print(f"\n{'Time':>8} | {'EW grad_var':>12} | {'KPZ grad_var':>12} | {'KPZ/EW ratio':>12}")
    print("-" * 55)
    
    for i, t in enumerate(times):
        ratio = kpz_grad_vars[i] / (ew_grad_vars[i] + 1e-10)
        print(f"{t:>8} | {ew_grad_vars[i]:>12.4f} | {kpz_grad_vars[i]:>12.4f} | {ratio:>12.1f}x")
    
    # Save results
    out_dir = Path(__file__).parent.parent / 'results' / 'exp31_asymptotic_verification'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Skewness evolution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(times, ew_skews, 'b-o', label='EW', linewidth=2, markersize=6)
    ax1.plot(times, kpz_skews, 'r-s', label='KPZ', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='EW theory (0)')
    ax1.axhline(y=-0.29, color='red', linestyle='--', alpha=0.5, label='KPZ theory (-0.29)')
    ax1.set_xlabel('Time T', fontsize=11)
    ax1.set_ylabel('Height Skewness', fontsize=11)
    ax1.set_title('Skewness Evolution: Do They Diverge?', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ratios = [kpz_grad_vars[i] / (ew_grad_vars[i] + 1e-10) for i in range(len(times))]
    ax2.plot(times, ratios, 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Time T', fontsize=11)
    ax2.set_ylabel('KPZ / EW Gradient Variance Ratio', fontsize=11)
    ax2.set_title('Gradient Variance Separation vs Time', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'asymptotic_evolution.png', dpi=150)
    print(f"\nSaved: {out_dir / 'asymptotic_evolution.png'}")
    
    # Save summary
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write("Experiment 31: Asymptotic Regime Verification\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Parameters: L={L}, T_max={T_max}, n_samples={n_samples}\n\n")
        f.write("Skewness Evolution:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Time':>8} {'EW_skew':>12} {'KPZ_skew':>12} {'KPZ/EW_gvar':>12}\n")
        for i, t in enumerate(times):
            ratio = kpz_grad_vars[i] / (ew_grad_vars[i] + 1e-10)
            f.write(f"{t:>8} {ew_skews[i]:>+12.4f} {kpz_skews[i]:>+12.4f} {ratio:>12.1f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Key Questions:\n")
        f.write(f"1. Does EW skewness → 0? Final value: {ew_skews[-1]:.4f}\n")
        f.write(f"2. Does KPZ skewness → -0.29? Final value: {kpz_skews[-1]:.4f}\n")
        f.write(f"3. Gradient variance ratio at T={T_max}: {ratios[-1]:.1f}x\n")
    
    print(f"Saved: {out_dir / 'summary.txt'}")
    
    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    
    final_ew = ew_skews[-1]
    final_kpz = kpz_skews[-1]
    
    if abs(final_ew) < 0.1 and abs(final_kpz + 0.29) < 0.1:
        print("✓ ASYMPTOTIC REGIME REACHED")
        print(f"  EW skewness: {final_ew:.3f} (theory: 0)")
        print(f"  KPZ skewness: {final_kpz:.3f} (theory: -0.29)")
    elif abs(final_kpz - final_ew) > 0.15:
        print("~ PARTIAL: Skewnesses diverging but not yet at theory values")
        print(f"  EW skewness: {final_ew:.3f}")
        print(f"  KPZ skewness: {final_kpz:.3f}")
        print("  → Need longer times or larger L")
    else:
        print("✗ NOT IN ASYMPTOTIC REGIME")
        print(f"  EW and KPZ skewnesses still similar: {final_ew:.3f} vs {final_kpz:.3f}")
        print("  → Need much longer simulations")

if __name__ == "__main__":
    main()
