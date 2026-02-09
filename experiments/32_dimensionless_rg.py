"""
Experiment 32: Dimensionless Features RG Test

The Problem:
Exp24/29/30 showed that gradient-moment features FAIL under RG coarse-graining
because they're dimensionful (grad_var scales as ~1/b², etc.)

The Solution:
Use ONLY dimensionless observables that should be RG-invariant:
1. Skewness (3rd moment / std³) - scale-free
2. Kurtosis (4th moment / std⁴ - 3) - scale-free
3. Ratios: lap_var/grad_var², h_var/grad_var, etc.

Prediction:
If dimensionless features work, we should see:
- BD→KPZ: STRONG contraction (>50%)
- EW↔KPZ: STABLE separation (drift <20%)

This would validate that universality IS geometric, just in the right coordinates.

Date: January 21, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import euclidean
from scipy import stats
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.physics_simulation import GrowthModelSimulator


def extract_dimensionless_features(h):
    """
    Extract ONLY scale-invariant (dimensionless) features.
    
    These should be RG-invariant under proper coarse-graining.
    """
    # Gradient and Laplacian
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2.0
    lap = np.roll(h, -1) + np.roll(h, 1) - 2.0 * h
    
    # Standardized moments (dimensionless!)
    grad_skew = stats.skew(grad)
    grad_kurt = stats.kurtosis(grad)  # Excess kurtosis
    lap_skew = stats.skew(lap)
    lap_kurt = stats.kurtosis(lap)
    h_skew = stats.skew(h)
    h_kurt = stats.kurtosis(h)
    
    # Dimensionless ratios
    grad_var = np.var(grad) + 1e-12
    lap_var = np.var(lap) + 1e-12
    h_var = np.var(h) + 1e-12
    
    ratio_lap_grad = lap_var / grad_var  # Should be ~4 for smooth surfaces
    ratio_h_grad = h_var / grad_var
    
    # Correlation coefficient (dimensionless)
    grad_lap_corr = np.corrcoef(grad, lap)[0, 1] if len(grad) > 1 else 0
    
    # 9D dimensionless feature vector
    return np.array([
        grad_skew,
        grad_kurt,
        lap_skew,
        lap_kurt,
        h_skew,
        h_kurt,
        ratio_lap_grad,
        ratio_h_grad,
        grad_lap_corr
    ])


def block_average_1d(h, block_size):
    """Apply block averaging (spatial coarse-graining)."""
    if block_size == 1:
        return h
    L = len(h)
    L_new = L // block_size
    return np.array([np.mean(h[i*block_size:(i+1)*block_size]) for i in range(L_new)])


def mean_pairwise_distance(A, B):
    """Mean pairwise Euclidean distance between feature sets."""
    return float(np.mean([euclidean(a, b) for a in A for b in B]))


MODEL_SPECS = {
    "EW": ("edwards_wilkinson", {"diffusion": 1.0, "noise_strength": 1.0, "dt": 0.1}),
    "KPZ": ("kpz_equation", {"diffusion": 1.0, "nonlinearity": 1.0, "noise_strength": 1.0, "dt": 0.05}),
    "BD": ("ballistic_deposition", {"noise_strength": 0.2}),
}


def main():
    print("=" * 78)
    print("EXPERIMENT 32: DIMENSIONLESS FEATURES RG TEST")
    print("=" * 78)
    print()
    print("Testing whether DIMENSIONLESS features exhibit proper RG behavior")
    print("(Unlike Exp24/29/30 which used dimensionful features)")
    print()
    
    # Parameters
    L = 512
    T = 3000
    n_samples = 30
    block_sizes = [1, 2, 4, 8, 16, 32]
    
    print(f"Parameters: L={L}, T={T}, n_samples={n_samples}")
    print(f"Block sizes: {block_sizes}")
    print(f"Features: 9D dimensionless (skew, kurt, ratios, correlations)")
    print()
    
    out_dir = project_root / 'results' / 'exp32_dimensionless_rg'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate trajectories once
    print("Phase 1: Generating trajectories...")
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    trajectories = {"EW": [], "KPZ": [], "BD": []}
    
    for model in ["EW", "KPZ", "BD"]:
        model_type, kwargs = MODEL_SPECS[model]
        print(f"  {model}...")
        for i in range(n_samples):
            traj = simulator.generate_trajectory(model_type, **kwargs)
            trajectories[model].append(traj)
    
    # Compute whitening from b=1 features (fixed metric)
    print("\nPhase 2: Computing fixed whitening from b=1...")
    features_b1 = {"EW": [], "KPZ": [], "BD": []}
    for model in ["EW", "KPZ", "BD"]:
        for traj in trajectories[model]:
            h = traj[-1]  # Final time
            features_b1[model].append(extract_dimensionless_features(h))
        features_b1[model] = np.array(features_b1[model])
    
    all_b1 = np.vstack([features_b1["EW"], features_b1["KPZ"], features_b1["BD"]])
    whiten_mean = np.mean(all_b1, axis=0)
    whiten_std = np.std(all_b1, axis=0)
    
    print(f"  Whitening computed from {len(all_b1)} samples")
    
    # Main RG sweep
    print("\nPhase 3: RG coarse-graining sweep...")
    print("-" * 78)
    
    distances = {"b": [], "d_bd_kpz": [], "d_ew_kpz": [], "d_bd_ew": []}
    
    for b in block_sizes:
        print(f"\nBlock size b={b}:")
        
        features_by_model = {}
        for model in ["EW", "KPZ", "BD"]:
            feats = []
            for traj in trajectories[model]:
                h = traj[-1]
                h_coarse = block_average_1d(h, b)
                f = extract_dimensionless_features(h_coarse)
                # Apply fixed whitening
                f_white = (f - whiten_mean) / (whiten_std + 1e-10)
                feats.append(f_white)
            features_by_model[model] = np.array(feats)
        
        # Compute distances
        d_bd_kpz = mean_pairwise_distance(features_by_model["BD"], features_by_model["KPZ"])
        d_ew_kpz = mean_pairwise_distance(features_by_model["EW"], features_by_model["KPZ"])
        d_bd_ew = mean_pairwise_distance(features_by_model["BD"], features_by_model["EW"])
        
        distances["b"].append(b)
        distances["d_bd_kpz"].append(d_bd_kpz)
        distances["d_ew_kpz"].append(d_ew_kpz)
        distances["d_bd_ew"].append(d_bd_ew)
        
        print(f"  d(BD,KPZ) = {d_bd_kpz:.3f}")
        print(f"  d(EW,KPZ) = {d_ew_kpz:.3f}")
        print(f"  d(BD,EW)  = {d_bd_ew:.3f}")
    
    # Analysis
    print("\n" + "=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)
    
    b_arr = np.array(distances["b"])
    d_bd_kpz = np.array(distances["d_bd_kpz"])
    d_ew_kpz = np.array(distances["d_ew_kpz"])
    
    # Contraction metrics
    bd_kpz_contraction = 1.0 - d_bd_kpz[-1] / d_bd_kpz[0]
    ew_kpz_drift = d_ew_kpz[-1] / d_ew_kpz[0] - 1.0
    
    print(f"\nBD→KPZ contraction: {bd_kpz_contraction*100:.1f}%")
    print(f"EW↔KPZ drift: {ew_kpz_drift*100:+.1f}%")
    
    # Success criteria
    print("\n" + "-" * 78)
    if bd_kpz_contraction > 0.5 and abs(ew_kpz_drift) < 0.2:
        verdict = "✓ SUCCESS: Differential contraction achieved with dimensionless features!"
        success = True
    elif bd_kpz_contraction > 0.3 and abs(ew_kpz_drift) < 0.3:
        verdict = "~ PARTIAL: Some contraction, moderate drift"
        success = False
    else:
        verdict = "✗ FAILURE: Dimensionless features also don't show clean RG behavior"
        success = False
    
    print(verdict)
    
    # Comparison to Exp24 (dimensionful features)
    print("\n" + "-" * 78)
    print("COMPARISON TO EXP24 (dimensionful features):")
    print("  Exp24: BD→KPZ contraction = 14.9%, EW↔KPZ drift = +44.7%")
    print(f"  Exp32: BD→KPZ contraction = {bd_kpz_contraction*100:.1f}%, EW↔KPZ drift = {ew_kpz_drift*100:+.1f}%")
    
    if bd_kpz_contraction > 0.149:
        print("  → Dimensionless features show MORE contraction than Exp24")
    else:
        print("  → Dimensionless features show similar or less contraction")
    
    # Save plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(b_arr, d_bd_kpz, 'r-o', label='d(BD,KPZ) [within-class]', linewidth=2)
    ax1.plot(b_arr, d_ew_kpz, 'b-s', label='d(EW,KPZ) [between-class]', linewidth=2)
    ax1.set_xlabel('Block size b', fontsize=11)
    ax1.set_ylabel('Mean pairwise distance', fontsize=11)
    ax1.set_title('Dimensionless Features: RG Coarse-Graining', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(b_arr, d_bd_kpz / d_bd_kpz[0], 'r-o', label='BD→KPZ (normalized)', linewidth=2)
    ax2.plot(b_arr, d_ew_kpz / d_ew_kpz[0], 'b-s', label='EW↔KPZ (normalized)', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Block size b', fontsize=11)
    ax2.set_ylabel('Normalized distance', fontsize=11)
    ax2.set_title('Normalized: Contraction vs Stability', fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'dimensionless_rg.png', dpi=150)
    print(f"\nSaved: {out_dir / 'dimensionless_rg.png'}")
    
    # Save summary
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write("Experiment 32: Dimensionless Features RG Test\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Parameters: L={L}, T={T}, n_samples={n_samples}\n")
        f.write(f"Features: 9D dimensionless (skew, kurt, ratios, correlations)\n\n")
        f.write("Distance Evolution:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'b':>4} {'d(BD,KPZ)':>12} {'d(EW,KPZ)':>12} {'d(BD,EW)':>12}\n")
        for i in range(len(b_arr)):
            f.write(f"{int(b_arr[i]):>4} {distances['d_bd_kpz'][i]:>12.4f} {distances['d_ew_kpz'][i]:>12.4f} {distances['d_bd_ew'][i]:>12.4f}\n")
        f.write("\n")
        f.write(f"BD→KPZ contraction: {bd_kpz_contraction*100:.1f}%\n")
        f.write(f"EW↔KPZ drift: {ew_kpz_drift*100:+.1f}%\n")
        f.write(f"\nVerdict: {verdict}\n")
    
    print(f"Saved: {out_dir / 'summary.txt'}")
    
    return success

if __name__ == "__main__":
    main()
