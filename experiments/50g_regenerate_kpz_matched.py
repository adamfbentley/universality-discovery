"""
Experiment 50g: Regenerate KPZ with Field-Level Matching

CRITICAL FIX: Generate KPZ data matching KS setup exactly.

Current problem (Exp 50f):
- KPZ: L=128, 95 samples → 5 at b=16 (underpowered)
- KS: L=256, 1000 samples at all b
- KPZ uses feature-averaging (not field-level coarse-graining)
- Creates asymmetry that makes slope untrustworthy

This experiment:
- Generate KPZ with L=256 (match KS)
- Store RAW FIELDS (not just features)
- Generate ~1000-1500 samples (match KS)
- Apply SAME spectral coarse-graining as KS
- Keep sample count constant across all b

Estimated time: ~1 hour to generate
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.fft import fft, ifft
from numba import jit

sns.set_style("whitegrid")

# ============================================================================
# KPZ SIMULATOR (Numba-optimized, matching KS setup)
# ============================================================================

@jit(nopython=True)
def simulate_kpz_trajectory(L=256, T=2000, lambda_=1.0, nu=1.0, D=1.0, 
                            dt=0.01, save_interval=20):
    """
    Simulate KPZ equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
    
    Parameters match KS setup:
    - L = 256 (spatial grid points)
    - dt = 0.01 (time step)
    
    Parameters:
    -----------
    L : int
        System size (spatial lattice points)
    T : int
        Number of time steps to simulate
    lambda_ : float
        Nonlinearity strength (KPZ parameter)
    nu : float
        Diffusion coefficient
    D : float
        Noise strength
    dt : float
        Time step size
    save_interval : int
        Save every N steps
        
    Returns:
    --------
    trajectory : array (n_saves, L)
        Height field evolution
    """
    interface = np.zeros(L)  # Flat initial condition
    n_saves = T // save_interval
    trajectory = np.zeros((n_saves, L))
    save_idx = 0
    
    for t in range(T):
        # Compute derivatives
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Laplacian: ∇²h
            laplacian = left - 2*center + right
            
            # Gradient: ∇h (central difference)
            gradient = (right - left) / 2.0
            
            # Noise
            noise = np.sqrt(D * dt) * np.random.randn()
            
            # KPZ equation
            dhdt = nu * laplacian + (lambda_ / 2.0) * gradient**2 + noise
            new_interface[x] = center + dt * dhdt
        
        interface = new_interface
        
        # Save if at interval
        if t % save_interval == 0:
            trajectory[save_idx] = interface.copy()
            save_idx += 1
    
    return trajectory

# ============================================================================
# FIELD-LEVEL COARSE-GRAINING (same as KS)
# ============================================================================

def coarse_grain_field_spectral(h, k_cutoff_fraction):
    """
    Spectral low-pass coarse-graining.
    
    Parameters:
    -----------
    h : array (L,)
        Height field
    k_cutoff_fraction : float
        Fraction of modes to keep (e.g., 0.5 = keep half)
    
    Returns:
    --------
    h_coarse : array (L,)
        Coarse-grained field (same size, but smoothed)
    """
    L = len(h)
    h_hat = fft(h)
    
    # Get wavenumbers
    k = np.fft.fftfreq(L)
    k_max = np.max(np.abs(k))
    k_cutoff = k_cutoff_fraction * k_max
    
    # Zero out high-k modes
    mask = np.abs(k) <= k_cutoff
    h_hat_filtered = h_hat * mask
    
    h_coarse = np.real(ifft(h_hat_filtered))
    
    return h_coarse

def extract_gradient_moments(h):
    """
    Extract gradient moment features from field.
    
    Returns: [m2, m3, m4, m5, m6, m7]
    """
    grad = np.gradient(h)
    
    features = np.array([
        np.mean(grad**2),
        np.mean(grad**3),
        np.mean(grad**4),
        np.mean(grad**5),
        np.mean(grad**6),
        np.mean(np.abs(grad)**7)
    ])
    
    return features

# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_kpz_matched(n_trajectories=25, L=256, T=2000,
                        lambda_=1.0, nu=1.0, D=1.0):
    """
    Generate KPZ data matching KS setup.
    
    Parameters:
    -----------
    n_trajectories : int
        Number of independent trajectories (25 × 50 snapshots = ~1250 samples)
    L : int
        System size (match KS: 256)
    T : int
        Time steps per trajectory
    lambda_, nu, D : float
        KPZ parameters (neutral regime: all 1.0)
    
    Returns:
    --------
    all_fields : array (n_samples, L)
        Raw height fields from stationary regime
    """
    print("="*70)
    print("GENERATING KPZ FIELDS (MATCHED TO KS SETUP)")
    print("="*70)
    print(f"\nParameters:")
    print(f"  L = {L} (spatial grid)")
    print(f"  T = {T} (timesteps per trajectory)")
    print(f"  dt = 0.01")
    print(f"  λ = {lambda_}, ν = {nu}, D = {D}")
    print(f"  Generating {n_trajectories} independent trajectories")
    print(f"  Save interval: 20 (every 20 timesteps)")
    
    all_fields = []
    
    for traj_idx in range(n_trajectories):
        # Simulate trajectory
        trajectory = simulate_kpz_trajectory(
            L=L,
            T=T,
            lambda_=lambda_,
            nu=nu,
            D=D,
            dt=0.01,  # Match KS
            save_interval=20  # Match KS
        )
        
        # Skip transient (first half)
        n_transient = len(trajectory) // 2
        
        for t_idx in range(n_transient, len(trajectory)):
            all_fields.append(trajectory[t_idx])
        
        if (traj_idx + 1) % 5 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories} trajectories")
    
    all_fields = np.array(all_fields)
    
    print(f"\nGenerated {len(all_fields)} field samples")
    print(f"  Field shape: {all_fields.shape}")
    print(f"  Field range: [{all_fields.min():.2f}, {all_fields.max():.2f}]")
    print(f"  Mean std per field: {np.mean([np.std(h) for h in all_fields]):.2f}")
    
    return all_fields

def apply_spectral_coarsening(fields, scales=[1.0, 0.5, 0.25, 0.125, 0.0625]):
    """
    Apply spectral coarse-graining at multiple scales.
    
    Parameters:
    -----------
    fields : array (n_samples, L)
        Raw height fields
    scales : list
        k_cutoff fractions (1.0=no filtering, 0.5=half modes, etc.)
    
    Returns:
    --------
    features_by_scale : dict
        {scale: features_array} for each scale
    """
    print("\n" + "="*70)
    print("APPLYING SPECTRAL COARSE-GRAINING")
    print("="*70)
    
    scale_names = ['b=1', 'b=2', 'b=4', 'b=8', 'b=16']
    features_by_scale = {}
    
    for scale, name in zip(scales, scale_names):
        print(f"\n{name} (keep {scale*100:.1f}% of modes):")
        
        features_list = []
        
        for h in fields:
            # Coarse-grain field
            h_coarse = coarse_grain_field_spectral(h, scale)
            
            # Extract features from coarse-grained field
            features = extract_gradient_moments(h_coarse)
            features_list.append(features)
        
        features_by_scale[scale] = np.array(features_list)
        
        print(f"  Extracted {len(features_list)} feature vectors")
        print(f"  Feature shape: {features_by_scale[scale].shape}")
        print(f"  Feature range: [{features_by_scale[scale].min():.2e}, {features_by_scale[scale].max():.2e}]")
    
    return features_by_scale, scales, scale_names

# ============================================================================
# VALIDATION PLOTS
# ============================================================================

def plot_validation(fields, features_by_scale, scales, scale_names):
    """
    Create validation plots showing:
    1. Sample fields at different coarse-graining scales
    2. Feature distributions
    3. Gradient statistics
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plot 1: Sample fields at different scales
    ax = axes[0, 0]
    sample_idx = len(fields) // 2
    h_raw = fields[sample_idx]
    ax.plot(h_raw, alpha=0.7, linewidth=1)
    ax.set_xlabel('Position x')
    ax.set_ylabel('Height h(x)')
    ax.set_title('Raw Field (b=1)')
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    h_coarse = coarse_grain_field_spectral(h_raw, 0.125)  # b=8
    ax.plot(h_coarse, alpha=0.7, linewidth=2, color='C2')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Height h(x)')
    ax.set_title('Coarse-Grained Field (b=8)')
    ax.grid(alpha=0.3)
    
    # Plot 2: Gradient distributions
    ax = axes[1, 0]
    for scale, name in zip(scales[:3], scale_names[:3]):
        h = coarse_grain_field_spectral(h_raw, scale)
        grad = np.gradient(h)
        ax.hist(grad, bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Gradient ∂h/∂x')
    ax.set_ylabel('Density')
    ax.set_title('Gradient Distributions')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Feature evolution across scales
    ax = axes[1, 1]
    feature_names = ['m2', 'm3', 'm4', 'm5', 'm6', 'm7']
    
    for feat_idx in [0, 2, 5]:  # Plot m2, m4, m7
        means = [features_by_scale[s][:, feat_idx].mean() for s in scales]
        ax.plot(range(len(means)), means, 'o-', label=feature_names[feat_idx], 
               markersize=8, linewidth=2)
    
    ax.set_xticks(range(len(scale_names)))
    ax.set_xticklabels(scale_names)
    ax.set_xlabel('Coarse-graining scale')
    ax.set_ylabel('Mean feature value')
    ax.set_title('Feature Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 4: Power spectrum
    ax = axes[2, 0]
    h_fft = np.abs(fft(h_raw))
    freqs = np.fft.fftfreq(len(h_raw))
    mask = freqs > 0
    ax.loglog(freqs[mask], h_fft[mask]**2, alpha=0.7)
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Power |h(k)|²')
    ax.set_title('Power Spectrum (b=1)')
    ax.grid(alpha=0.3)
    
    # Plot 5: Roughness across samples
    ax = axes[2, 1]
    roughness_by_scale = []
    for scale in scales:
        roughness = []
        for h in fields[:100]:  # First 100 samples
            h_coarse = coarse_grain_field_spectral(h, scale)
            roughness.append(np.std(h_coarse))
        roughness_by_scale.append(roughness)
    
    ax.boxplot(roughness_by_scale, labels=scale_names)
    ax.set_xlabel('Coarse-graining scale')
    ax.set_ylabel('Interface width (std)')
    ax.set_title('Roughness Distribution')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50g: REGENERATE KPZ WITH MATCHED PROTOCOL")
    print("="*70)
    print("\nGoal: Generate KPZ data matching KS setup exactly")
    print("  - L=256 (match KS)")
    print("  - dt=0.01 (match KS)")
    print("  - Field storage (not just features)")
    print("  - ~1000-1500 samples (match KS)")
    print("  - Spectral coarse-graining (match KS)")
    print("="*70)
    
    output_dir = Path('results/kpz_fields_matched_L256')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fields
    fields = generate_kpz_matched(
        n_trajectories=25,  # 25 × 50 snapshots = 1250 samples
        L=256,
        T=2000,
        lambda_=1.0,
        nu=1.0,
        D=1.0
    )
    
    # Apply spectral coarse-graining
    features_by_scale, scales, scale_names = apply_spectral_coarsening(fields)
    
    # Create validation plots
    print("\n" + "="*70)
    print("CREATING VALIDATION PLOTS")
    print("="*70)
    
    fig = plot_validation(fields, features_by_scale, scales, scale_names)
    fig.savefig(output_dir / 'validation.png', dpi=150, bbox_inches='tight')
    
    # Save data
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    data = {
        'fields': fields,
        'features_by_scale': features_by_scale,
        'scales': scales,
        'scale_names': scale_names,
        'parameters': {
            'L': 256,
            'T': 2000,
            'dt': 0.01,
            'lambda': 1.0,
            'nu': 1.0,
            'D': 1.0,
            'n_trajectories': 25,
            'save_interval': 20
        }
    }
    
    with open(output_dir / 'kpz_matched_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to {output_dir}/")
    print(f"  Fields: {fields.shape}")
    print(f"  Features at each scale: {len(features_by_scale)}")
    for scale, name in zip(scales, scale_names):
        print(f"    {name}: {features_by_scale[scale].shape}")
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Rerun Exp 50f with this matched KPZ data")
    print("  2. Check if slope changes from +0.0185")
    print("  3. If still flat → Move to Diagnostic B (spectral observables)")
    
    plt.show()

if __name__ == '__main__':
    main()
