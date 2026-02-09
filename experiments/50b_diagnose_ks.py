"""
Experiment 50b: Diagnose KS Simulation Issues

The features from Exp 50 are ~10^-10, which is a red flag.
This script checks:
1. Are KS dynamics actually evolving?
2. What are the actual field amplitudes?
3. Where does the feature extraction fail?
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator

# ============================================================================
# DIAGNOSTIC 1: Check raw KS output
# ============================================================================

def diagnose_simulation():
    """Check if KS simulation produces reasonable amplitudes."""
    print("="*70)
    print("DIAGNOSTIC 1: KS SIMULATION OUTPUT")
    print("="*70)
    
    sim = KuramotoSivashinskySimulator(L=256, dt=0.01)
    
    # Run with same parameters as Exp 50
    trajectory = sim.simulate(
        T=1000,
        nu=1.0,
        kappa=1.0,
        lam=1.0,
        noise_strength=0.5,
        record_interval=10
    )
    
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"  {trajectory.shape[0]} snapshots × {trajectory.shape[1]} spatial points")
    
    # Check field statistics
    print("\nField statistics:")
    print(f"  Mean: {np.mean(trajectory):.6e}")
    print(f"  Std:  {np.std(trajectory):.6e}")
    print(f"  Min:  {np.min(trajectory):.6e}")
    print(f"  Max:  {np.max(trajectory):.6e}")
    print(f"  Range: {np.ptp(trajectory):.6e}")
    
    # Check evolution
    initial_std = np.std(trajectory[0])
    final_std = np.std(trajectory[-1])
    print(f"\nEvolution:")
    print(f"  Initial std: {initial_std:.6e}")
    print(f"  Final std:   {final_std:.6e}")
    print(f"  Ratio:       {final_std/initial_std:.2f}")
    
    # Check gradients
    h = trajectory[-1]
    grad = np.gradient(h)
    print(f"\nGradient statistics (final snapshot):")
    print(f"  Mean: {np.mean(grad):.6e}")
    print(f"  Std:  {np.std(grad):.6e}")
    print(f"  Min:  {np.min(grad):.6e}")
    print(f"  Max:  {np.max(grad):.6e}")
    
    # Check gradient moments (as computed in Exp 50)
    print(f"\nGradient moments (as computed in Exp 50):")
    for k in [2, 3, 4, 5, 6, 7]:
        if k == 7:
            m_k = np.mean(np.abs(grad)**k)
        else:
            m_k = np.mean(grad**k)
        print(f"  m{k} = {m_k:.6e}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Spacetime
    ax = axes[0, 0]
    im = ax.imshow(trajectory.T, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Space')
    ax.set_title('Spacetime Diagram')
    plt.colorbar(im, ax=ax)
    
    # Final profile
    ax = axes[0, 1]
    ax.plot(sim.x, h)
    ax.set_xlabel('x')
    ax.set_ylabel('h(x)')
    ax.set_title(f'Final Height Profile (std={final_std:.2e})')
    ax.grid(alpha=0.3)
    
    # Gradient
    ax = axes[1, 0]
    ax.plot(sim.x, grad)
    ax.set_xlabel('x')
    ax.set_ylabel('∂h/∂x')
    ax.set_title(f'Final Gradient (std={np.std(grad):.2e})')
    ax.grid(alpha=0.3)
    
    # Evolution of std
    ax = axes[1, 1]
    stds = [np.std(trajectory[i]) for i in range(len(trajectory))]
    ax.plot(stds)
    ax.set_xlabel('Time step')
    ax.set_ylabel('std(h)')
    ax.set_title('Field Standard Deviation Over Time')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return trajectory, fig

# ============================================================================
# DIAGNOSTIC 2: Compare to KPZ features
# ============================================================================

def compare_to_kpz_features():
    """Load KPZ features from Exp 46b and compare magnitudes."""
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: COMPARE KS vs KPZ FEATURE MAGNITUDES")
    print("="*70)
    
    # Check if KPZ data exists
    kpz_path = Path('results/exp46b_log_scale_coupling/data.pkl')
    if not kpz_path.exists():
        print("⚠️  KPZ data not found, skipping comparison")
        return
    
    import pickle
    with open(kpz_path, 'rb') as f:
        kpz_data = pickle.load(f)
    
    kpz_features = kpz_data['features']
    
    print(f"\nKPZ features (from Exp 46b):")
    print(f"  Shape: {kpz_features.shape}")
    print(f"  Mean: {np.mean(kpz_features):.6e}")
    print(f"  Std:  {np.std(kpz_features):.6e}")
    print(f"  Min:  {np.min(kpz_features):.6e}")
    print(f"  Max:  {np.max(kpz_features):.6e}")
    print(f"  Range: {np.ptp(kpz_features):.6e}")
    
    print("\nKPZ feature columns:")
    for i in range(kpz_features.shape[1]):
        col = kpz_features[:, i]
        print(f"  Column {i}: mean={np.mean(col):.3e}, std={np.std(col):.3e}, "
              f"range=[{np.min(col):.3e}, {np.max(col):.3e}]")
    
    print("\n" + "="*70)
    print("EXPECTED: KS features should have similar magnitude to KPZ")
    print("  KPZ: ~10^-1 to 10^1 (typical)")
    print("  KS (Exp 50): ~10^-15 to 10^-10 ❌ WAY TOO SMALL")
    print("="*70)

# ============================================================================
# DIAGNOSTIC 3: Test with larger initial amplitude
# ============================================================================

def test_different_initial_conditions():
    """Test if initial condition amplitude matters."""
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: TEST DIFFERENT INITIAL CONDITIONS")
    print("="*70)
    
    sim = KuramotoSivashinskySimulator(L=256, dt=0.01)
    
    initial_amplitudes = [0.01, 0.1, 1.0, 10.0]
    
    for amp in initial_amplitudes:
        # Custom initial condition
        h0 = amp * np.random.randn(sim.L)
        
        trajectory = sim.simulate(
            T=500,
            nu=1.0,
            kappa=1.0,
            lam=1.0,
            noise_strength=0.5,
            record_interval=10,
            initial_condition=h0
        )
        
        final_std = np.std(trajectory[-1])
        grad = np.gradient(trajectory[-1])
        m2 = np.mean(grad**2)
        
        print(f"\nInitial amplitude: {amp:.2f}")
        print(f"  Initial std(h): {np.std(h0):.3e}")
        print(f"  Final std(h):   {final_std:.3e}")
        print(f"  Final m2:       {m2:.3e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    output_dir = Path('results/exp50b_ks_diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Diagnostic 1: Check simulation output
    trajectory, fig1 = diagnose_simulation()
    fig1.savefig(output_dir / 'diagnostic_1_simulation.png', dpi=150, bbox_inches='tight')
    
    # Diagnostic 2: Compare to KPZ
    compare_to_kpz_features()
    
    # Diagnostic 3: Test initial conditions
    test_different_initial_conditions()
    
    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to {output_dir}/")
    
    plt.show()

if __name__ == '__main__':
    main()
