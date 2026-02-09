"""
Experiment 28: Time-Resolved Analysis for Flat IC

Critical test following Exp 27's IC-dependence finding.

Key question: Does EW/KPZ separation exist in the GROWTH REGIME for flat IC,
and then vanish at stationarity?

From Exp 27:
- Flat IC at T=1000: r = -0.06 (NO separation)
- Droplet IC at T=1000: r = -0.98 (PERFECT separation)

Hypothesis:
- Flat IC separation should be present at early T (growth regime)
- Should decay/vanish as T → stationary

KPZ theory context:
- Flat IC → GOE Tracy-Widom (Airy₁ process)
- The universal signal is in PROPERLY RESCALED height fluctuations
- Gradient moments may not capture GOE structure well

This experiment maps r(PC1, model) vs T to find:
1. T_onset: when separation emerges
2. T_peak: when separation is maximal  
3. T_decay: when separation vanishes (stationarity)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numba import jit

# JIT-compiled simulation functions
@jit(nopython=True)
def simulate_ew_step(interface, diffusion=1.0, noise_strength=1.0, dt=0.05):
    """Single EW time step: dh/dt = ν∇²h + η"""
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        dhdt = diffusion * laplacian + noise
        new_interface[x] = center + dt * dhdt
    
    return new_interface

@jit(nopython=True)
def simulate_kpz_step(interface, diffusion=1.0, nonlinearity=1.0, noise_strength=1.0, dt=0.05):
    """Single KPZ time step: dh/dt = ν∇²h + (λ/2)(∇h)² + η"""
    L = len(interface)
    new_interface = interface.copy()
    
    for x in range(L):
        left = interface[(x-1) % L]
        center = interface[x]
        right = interface[(x+1) % L]
        laplacian = left - 2*center + right
        gradient = (right - left) / 2.0
        nonlinear_term = nonlinearity * 0.5 * gradient**2
        noise = noise_strength * np.sqrt(dt) * np.random.randn()
        dhdt = diffusion * laplacian + nonlinear_term + noise
        new_interface[x] = center + dt * dhdt
    
    return new_interface

def simulate_growth_trajectory(L, T_max, model='EW', h0=None, nu=1.0, lam=1.0, D=1.0, dt=0.05, save_interval=10):
    """
    Simulate surface growth and save snapshots at regular intervals.
    
    Returns: dict with times and corresponding interfaces
    """
    if h0 is None:
        h0 = np.zeros(L)
    
    interface = h0.copy()
    n_steps = int(T_max / dt)
    save_every = int(save_interval / dt)
    
    trajectory = {'times': [], 'interfaces': []}
    
    for step in range(n_steps + 1):
        t = step * dt
        
        # Save at intervals
        if step % save_every == 0:
            trajectory['times'].append(t)
            trajectory['interfaces'].append(interface.copy())
        
        # Evolve
        if step < n_steps:
            if model == 'EW':
                interface = simulate_ew_step(interface, diffusion=nu, noise_strength=D, dt=dt)
            else:  # KPZ
                interface = simulate_kpz_step(interface, diffusion=nu, nonlinearity=lam, 
                                             noise_strength=D, dt=dt)
            # Remove global tilt
            interface = interface - np.mean(interface)
    
    return trajectory

def extract_gradient_moments(h, L):
    """Extract 6D gradient moment features (matching Exp 21)."""
    # Periodic boundary gradients
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2.0
    
    # Laplacian with periodic BC
    lap = np.roll(h, -1) + np.roll(h, 1) - 2*h
    
    # Moments
    grad_var = np.var(grad)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)
    
    if grad_std > 1e-10:
        grad_skew = np.mean((grad - grad_mean)**3) / (grad_std**3)
        grad_kurt = np.mean((grad - grad_mean)**4) / (grad_std**4) - 3
    else:
        grad_skew = 0
        grad_kurt = 0
    
    lap_var = np.var(lap)
    grad_lap_cov = np.cov(np.abs(grad), lap)[0, 1] if len(grad) > 1 else 0
    h_var = np.var(h)
    
    return np.array([grad_var, grad_skew, grad_kurt, lap_var, grad_lap_cov, h_var])

def main():
    print("=" * 80)
    print("EXPERIMENT 28: TIME-RESOLVED FLAT IC ANALYSIS")
    print("=" * 80)
    print()
    print("Testing whether EW/KPZ separation exists in growth regime for flat IC")
    print()
    
    # Parameters
    L = 128
    T_max = 3000  # Long simulation to capture growth → stationary
    n_samples = 40  # Per model
    save_interval = 50  # Save every 50 time units
    
    # Physics parameters
    nu = 1.0
    lam = 1.0
    D = 1.0
    dt = 0.05
    
    # Characteristic times for KPZ at L=128:
    # Saturation time ~ L^z with z=3/2 → ~1450
    # Growth regime: T << 1450
    # Stationary regime: T >> 1450
    
    print(f"Parameters:")
    print(f"  L = {L}")
    print(f"  T_max = {T_max}")
    print(f"  n_samples = {n_samples}")
    print(f"  save_interval = {save_interval}")
    print(f"  Expected saturation time: ~{int(L**1.5)} (L^(3/2))")
    print()
    
    # Generate trajectories
    print("Generating EW trajectories with flat IC...")
    ew_trajectories = []
    for i in range(n_samples):
        if (i+1) % 10 == 0:
            print(f"  EW sample {i+1}/{n_samples}")
        h0 = np.zeros(L)  # FLAT IC
        traj = simulate_growth_trajectory(L, T_max, model='EW', h0=h0, 
                                         nu=nu, D=D, dt=dt, save_interval=save_interval)
        ew_trajectories.append(traj)
    
    print("\nGenerating KPZ trajectories with flat IC...")
    kpz_trajectories = []
    for i in range(n_samples):
        if (i+1) % 10 == 0:
            print(f"  KPZ sample {i+1}/{n_samples}")
        h0 = np.zeros(L)  # FLAT IC
        traj = simulate_growth_trajectory(L, T_max, model='KPZ', h0=h0,
                                         nu=nu, lam=lam, D=D, dt=dt, save_interval=save_interval)
        kpz_trajectories.append(traj)
    
    # Get time points
    times = np.array(ew_trajectories[0]['times'])
    n_times = len(times)
    print(f"\nAnalyzing {n_times} time points from T=0 to T={T_max}")
    
    # Compute separation at each time
    results = {
        'times': times,
        'r_pc1': [],
        'p_pc1': [],
        'cohens_d': [],
        'pc1_loadings': []
    }
    
    print("\nComputing PC1 separation at each time point...")
    
    feature_names = ['grad_var', 'grad_skew', 'grad_kurt', 'lap_var', 'grad_lap_cov', 'h_var']
    
    for t_idx in range(n_times):
        # Extract features at this time
        ew_features = []
        kpz_features = []
        
        for traj in ew_trajectories:
            h = traj['interfaces'][t_idx]
            features = extract_gradient_moments(h, L)
            ew_features.append(features)
        
        for traj in kpz_trajectories:
            h = traj['interfaces'][t_idx]
            features = extract_gradient_moments(h, L)
            kpz_features.append(features)
        
        ew_features = np.array(ew_features)
        kpz_features = np.array(kpz_features)
        
        # Combine and run PCA
        X_combined = np.vstack([ew_features, kpz_features])
        labels = np.array([0]*len(ew_features) + [1]*len(kpz_features))
        
        # Handle edge cases (T=0 all zeros)
        if np.std(X_combined) < 1e-10:
            results['r_pc1'].append(0)
            results['p_pc1'].append(1)
            results['cohens_d'].append(0)
            results['pc1_loadings'].append(np.zeros(6))
            continue
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Correlation
        r, p = pearsonr(X_pca[:, 0], labels)
        
        # Cohen's d
        pc1_ew = X_pca[labels == 0, 0]
        pc1_kpz = X_pca[labels == 1, 0]
        pooled_std = np.sqrt((np.var(pc1_ew) + np.var(pc1_kpz)) / 2)
        if pooled_std > 1e-10:
            d = (np.mean(pc1_kpz) - np.mean(pc1_ew)) / pooled_std
        else:
            d = 0
        
        results['r_pc1'].append(r)
        results['p_pc1'].append(p)
        results['cohens_d'].append(d)
        results['pc1_loadings'].append(pca.components_[0])
        
        if t_idx % 10 == 0:
            print(f"  T={times[t_idx]:.0f}: r={r:.3f}, d={d:.2f}")
    
    # Convert to arrays
    for key in ['r_pc1', 'p_pc1', 'cohens_d']:
        results[key] = np.array(results[key])
    results['pc1_loadings'] = np.array(results['pc1_loadings'])
    
    # Analysis
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Find key time points
    abs_r = np.abs(results['r_pc1'])
    t_peak_idx = np.argmax(abs_r)
    t_peak = times[t_peak_idx]
    r_peak = results['r_pc1'][t_peak_idx]
    
    # Find when r first exceeds 0.5 (onset)
    onset_mask = abs_r > 0.5
    if np.any(onset_mask):
        t_onset_idx = np.argmax(onset_mask)
        t_onset = times[t_onset_idx]
    else:
        t_onset = None
    
    # Find when r drops below 0.3 after peak (decay)
    if t_peak_idx < len(times) - 1:
        post_peak = abs_r[t_peak_idx:]
        decay_mask = post_peak < 0.3
        if np.any(decay_mask):
            t_decay_idx = t_peak_idx + np.argmax(decay_mask)
            t_decay = times[t_decay_idx]
        else:
            t_decay = None
    else:
        t_decay = None
    
    print(f"\nKey Time Points:")
    print(f"  T_onset (|r| > 0.5): {t_onset if t_onset else 'Never reached'}")
    print(f"  T_peak (max |r|): {t_peak:.0f} (r = {r_peak:.3f})")
    print(f"  T_decay (|r| < 0.3): {t_decay if t_decay else 'Not reached'}")
    print(f"  T_saturation (expected): ~{int(L**1.5)}")
    
    # Final values
    print(f"\nFinal state (T={T_max}):")
    print(f"  r(PC1, model) = {results['r_pc1'][-1]:.4f}")
    print(f"  Cohen's d = {results['cohens_d'][-1]:.2f}")
    
    # Compare to Exp 27
    print(f"\nComparison to Exp 27 (T=1000, flat IC):")
    t1000_idx = np.argmin(np.abs(times - 1000))
    print(f"  Exp 28 at T=1000: r = {results['r_pc1'][t1000_idx]:.4f}")
    print(f"  Exp 27 at T=1000: r = -0.060")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if abs(r_peak) > 0.7:
        print("\n✅ GROWTH REGIME SEPARATION EXISTS")
        print(f"   Peak separation at T={t_peak:.0f} with r={r_peak:.3f}")
        print("   The GOE/Airy₁ structure IS visible in gradient moments")
        print("   during the growth regime, before stationary equilibrium.")
    elif abs(r_peak) > 0.4:
        print("\n⚠️ WEAK GROWTH REGIME SEPARATION")
        print(f"   Peak separation at T={t_peak:.0f} with r={r_peak:.3f}")
        print("   Some signal present but gradient moments don't strongly")
        print("   capture the Airy₁/GOE structure for flat IC.")
    else:
        print("\n❌ NO SEPARATION AT ANY TIME")
        print("   Gradient moments don't distinguish EW/KPZ for flat IC")
        print("   even in the growth regime. The Airy₁/GOE structure")
        print("   requires different observables (height fluctuations?).")
    
    # Visualization
    print("\n" + "=" * 80)
    print("Generating visualization...")
    
    output_dir = Path(__file__).parent.parent / 'results' / 'exp28_time_resolved'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: r(PC1, model) vs T
    ax1 = axes[0, 0]
    ax1.plot(times, results['r_pc1'], 'b-', linewidth=2, label='r(PC1, model)')
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(0.5, color='g', linestyle='--', linewidth=1, alpha=0.5, label='|r|=0.5 threshold')
    ax1.axhline(-0.5, color='g', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(L**1.5, color='r', linestyle=':', linewidth=2, label=f'T_sat ~ L^(3/2) = {int(L**1.5)}')
    if t_peak:
        ax1.axvline(t_peak, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'T_peak = {t_peak:.0f}')
    ax1.set_xlabel('Time T', fontsize=12)
    ax1.set_ylabel('r(PC1, model label)', fontsize=12)
    ax1.set_title('PC1 Correlation vs Time (Flat IC)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T_max)
    
    # Panel 2: |r| vs T (log scale for time)
    ax2 = axes[0, 1]
    ax2.plot(times[1:], np.abs(results['r_pc1'][1:]), 'b-', linewidth=2)  # Skip T=0
    ax2.axhline(0.5, color='g', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(0.8, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Strong (|r|=0.8)')
    ax2.axvline(L**1.5, color='r', linestyle=':', linewidth=2)
    ax2.set_xlabel('Time T', fontsize=12)
    ax2.set_ylabel('|r(PC1, model)|', fontsize=12)
    ax2.set_title('Separation Strength vs Time', fontsize=13, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Cohen's d vs T
    ax3 = axes[1, 0]
    ax3.plot(times, results['cohens_d'], 'purple', linewidth=2)
    ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(0.8, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Large effect (d=0.8)')
    ax3.axhline(-0.8, color='g', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(L**1.5, color='r', linestyle=':', linewidth=2, label=f'T_sat')
    ax3.set_xlabel('Time T', fontsize=12)
    ax3.set_ylabel("Cohen's d", fontsize=12)
    ax3.set_title('Effect Size vs Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, T_max)
    
    # Panel 4: PC1 loadings evolution
    ax4 = axes[1, 1]
    for i, name in enumerate(feature_names):
        loadings_over_time = results['pc1_loadings'][:, i]
        ax4.plot(times, loadings_over_time, linewidth=1.5, label=name, alpha=0.8)
    ax4.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax4.axvline(L**1.5, color='r', linestyle=':', linewidth=2)
    ax4.set_xlabel('Time T', fontsize=12)
    ax4.set_ylabel('PC1 Loading', fontsize=12)
    ax4.set_title('PC1 Loadings vs Time', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, T_max)
    
    plt.tight_layout()
    
    fig_path = output_dir / 'time_resolved_flat_ic.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {fig_path}")
    
    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT 28: TIME-RESOLVED FLAT IC ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Parameters: L={L}, T_max={T_max}, n_samples={n_samples}\n\n")
        f.write("Key Results:\n")
        f.write(f"  T_onset (|r| > 0.5): {t_onset if t_onset else 'Never'}\n")
        f.write(f"  T_peak: {t_peak:.0f} (r = {r_peak:.3f})\n")
        f.write(f"  T_decay (|r| < 0.3): {t_decay if t_decay else 'Not reached'}\n")
        f.write(f"  T_saturation (expected): {int(L**1.5)}\n\n")
        f.write("Time series data:\n")
        f.write("T\tr(PC1)\tCohen's_d\n")
        for i in range(0, len(times), max(1, len(times)//20)):
            f.write(f"{times[i]:.0f}\t{results['r_pc1'][i]:.4f}\t{results['cohens_d'][i]:.2f}\n")
    
    print(f"✓ Saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 28 COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
