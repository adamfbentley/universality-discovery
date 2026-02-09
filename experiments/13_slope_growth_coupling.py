"""
Experiment 13: Slope-Growth Coupling Diagnostic

GOAL: Test whether KPZ nonlinearity λ(∇h)² is detectable in our simulations.

THEORETICAL BASIS:
The KPZ equation is: ∂h/∂t = ν∇²h + λ(∇h)² + η(x,t)
                              ↑          ↑
                         diffusion   nonlinearity

For EW (Edwards-Wilkinson): λ = 0, only diffusion
For KPZ class: λ ≠ 0, nonlinearity present

DIAGNOSTIC TEST:
If we regress local growth g = Δₜh against slope squared s² = (∇ₓh)²:
    g ≈ a + b·s² + noise

Then:
- KPZ-class (KPZ, BD, EDEN): b > 0 (nonlinearity contributes to growth)
- EW-class: b ≈ 0 (no nonlinear term)
- RD: b ≈ 0 (purely random, no spatial correlations)

This is the most direct test of whether the nonlinear signature exists in the data.

EXPECTED OUTCOME:
If b > 0 for KPZ-class → proceed with using b as ML feature (Φ_λ)
If b ≈ 0 for ALL → simulations are fundamentally broken
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Add parent directory and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation.physics_simulation import GrowthModelSimulator


def compute_slope_growth_coupling(surface_evolution: np.ndarray, 
                                   smooth_sigma: float = 1.0) -> dict:
    """
    Compute slope-growth coupling coefficient from surface evolution.
    
    Args:
        surface_evolution: Array of shape (T, L) - height profile at each time
        smooth_sigma: Gaussian smoothing to reduce noise (default 1.0)
    
    Returns:
        dict with regression results and statistics
    """
    T, L = surface_evolution.shape
    
    # Apply mild smoothing to reduce discrete noise
    if smooth_sigma > 0:
        smoothed = np.array([gaussian_filter1d(surface_evolution[t], smooth_sigma) 
                           for t in range(T)])
    else:
        smoothed = surface_evolution
    
    # Compute local slopes: s = ∇ₓh (spatial gradient)
    # Use central differences for interior points
    slopes = np.zeros((T, L))
    slopes[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
    slopes[:, 0] = smoothed[:, 1] - smoothed[:, 0]
    slopes[:, -1] = smoothed[:, -1] - smoothed[:, -2]
    
    # Compute local growth: g = Δₜh (temporal derivative)
    # Use forward differences
    growth = np.zeros((T-1, L))
    growth = smoothed[1:, :] - smoothed[:-1, :]
    
    # Align: use slope at time t to predict growth from t to t+1
    slopes_aligned = slopes[:-1, :]  # Shape: (T-1, L)
    
    # Flatten for regression (exclude boundaries)
    s_flat = slopes_aligned[:, 5:-5].flatten()  # Exclude boundary effects
    g_flat = growth[:, 5:-5].flatten()
    s_squared = s_flat ** 2
    
    # Remove outliers (beyond 3 sigma)
    g_mean, g_std = np.mean(g_flat), np.std(g_flat)
    s2_mean, s2_std = np.mean(s_squared), np.std(s_squared)
    
    mask = (np.abs(g_flat - g_mean) < 3 * g_std) & (np.abs(s_squared - s2_mean) < 3 * s2_std)
    s_squared_clean = s_squared[mask]
    g_clean = g_flat[mask]
    
    # Linear regression: g = a + b * s²
    slope_coeff, intercept, r_value, p_value, std_err = stats.linregress(s_squared_clean, g_clean)
    
    # Also compute correlation
    correlation = np.corrcoef(s_squared_clean, g_clean)[0, 1]
    
    # Compute mean values for normalization
    mean_s_squared = np.mean(s_squared_clean)
    mean_growth = np.mean(g_clean)
    std_growth = np.std(g_clean)
    
    # Normalized coefficient (dimensionless)
    if mean_growth != 0:
        b_normalized = slope_coeff * mean_s_squared / np.abs(mean_growth)
    else:
        b_normalized = slope_coeff * mean_s_squared / (std_growth + 1e-10)
    
    return {
        'b_coefficient': slope_coeff,      # Raw regression slope
        'b_normalized': b_normalized,       # Dimensionless version
        'intercept': intercept,             # Regression intercept
        'r_squared': r_value ** 2,          # Coefficient of determination
        'p_value': p_value,                 # Statistical significance
        'std_err': std_err,                 # Standard error of slope
        'correlation': correlation,         # Pearson correlation
        'mean_slope_sq': mean_s_squared,    # Mean s²
        'mean_growth': mean_growth,         # Mean g
        'n_points': len(s_squared_clean),   # Number of data points
    }


def analyze_model(model_name: str, model_type: str, n_samples: int = 20, 
                  L: int = 256, T: int = 1000, smooth_sigma: float = 1.0) -> dict:
    """
    Analyze slope-growth coupling for a single model type.
    """
    print(f"\n  Analyzing {model_name}...")
    
    results_list = []
    
    for i in range(n_samples):
        # Generate surface evolution
        simulator = GrowthModelSimulator(width=L, height=T, random_state=42 + i)
        surface = simulator.generate_trajectory(model_type)
        
        # Compute coupling
        result = compute_slope_growth_coupling(surface, smooth_sigma=smooth_sigma)
        results_list.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"    Sample {i+1}/{n_samples}: b = {result['b_coefficient']:.6f}, r² = {result['r_squared']:.4f}")
    
    # Aggregate statistics
    b_values = [r['b_coefficient'] for r in results_list]
    b_norm_values = [r['b_normalized'] for r in results_list]
    r2_values = [r['r_squared'] for r in results_list]
    corr_values = [r['correlation'] for r in results_list]
    
    return {
        'model': model_name,
        'b_mean': np.mean(b_values),
        'b_std': np.std(b_values),
        'b_norm_mean': np.mean(b_norm_values),
        'b_norm_std': np.std(b_norm_values),
        'r2_mean': np.mean(r2_values),
        'r2_std': np.std(r2_values),
        'corr_mean': np.mean(corr_values),
        'corr_std': np.std(corr_values),
        'individual_results': results_list,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 13: SLOPE-GROWTH COUPLING DIAGNOSTIC")
    print("=" * 70)
    print("\nTesting whether KPZ nonlinearity λ(∇h)² is detectable in simulations")
    print("Regression: growth g ≈ a + b·(slope)²")
    print("Expected: b > 0 for KPZ-class, b ≈ 0 for EW/RD")
    
    # Simulation parameters
    L = 256      # Larger system for better statistics
    T = 1000     # Longer time for more data points
    n_samples = 20  # Multiple realizations for error bars
    smooth_sigma = 1.0  # Mild smoothing
    
    print(f"\nParameters: L={L}, T={T}, n_samples={n_samples}, smooth_σ={smooth_sigma}")
    
    # Define models to test
    models = [
        ("Edwards-Wilkinson", "edwards_wilkinson"),
        ("KPZ equation", "kpz_equation"),
        ("Ballistic Deposition", "ballistic_deposition"),
        ("EDEN", "eden"),
        ("Random Deposition", "random_deposition"),
    ]
    
    # Analyze each model
    results = {}
    for name, model_type in models:
        results[name] = analyze_model(name, model_type, n_samples=n_samples, 
                                       L=L, T=T, smooth_sigma=smooth_sigma)
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: SLOPE-GROWTH COUPLING COEFFICIENTS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'b (raw)':<18} {'b (norm)':<18} {'r²':<12} {'Expected'}")
    print("-" * 90)
    
    expected = {
        "Edwards-Wilkinson": "≈ 0 (no λ term)",
        "KPZ equation": "> 0 (λ ≠ 0)",
        "Ballistic Deposition": "> 0 (KPZ class)",
        "EDEN": "> 0 (KPZ class)",
        "Random Deposition": "≈ 0 (uncorrelated)",
    }
    
    for name, r in results.items():
        b_str = f"{r['b_mean']:.6f} ± {r['b_std']:.6f}"
        b_norm_str = f"{r['b_norm_mean']:.4f} ± {r['b_norm_std']:.4f}"
        r2_str = f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}"
        print(f"{name:<25} {b_str:<18} {b_norm_str:<18} {r2_str:<12} {expected[name]}")
    
    # Diagnostic interpretation
    print("\n" + "=" * 70)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 70)
    
    ew_b = results["Edwards-Wilkinson"]['b_mean']
    kpz_b = results["KPZ equation"]['b_mean']
    bd_b = results["Ballistic Deposition"]['b_mean']
    eden_b = results["EDEN"]['b_mean']
    rd_b = results["Random Deposition"]['b_mean']
    
    # Compute KPZ-class average
    kpz_class_avg = (kpz_b + bd_b + eden_b) / 3
    ew_class_avg = (ew_b + rd_b) / 2
    
    print(f"\nKPZ-class average b: {kpz_class_avg:.6f}")
    print(f"EW-class average b:  {ew_class_avg:.6f}")
    print(f"Ratio (KPZ/EW): {kpz_class_avg / (ew_class_avg + 1e-10):.2f}x")
    
    # Statistical tests
    print("\n--- Statistical Tests ---")
    
    # Test: Is KPZ b significantly > 0?
    kpz_b_values = [r['b_coefficient'] for r in results["KPZ equation"]['individual_results']]
    t_stat, p_val = stats.ttest_1samp(kpz_b_values, 0)
    print(f"\nKPZ b > 0 test: t={t_stat:.3f}, p={p_val:.2e} {'✓ SIGNIFICANT' if p_val < 0.05 else '✗ NOT significant'}")
    
    # Test: Is EW b ≈ 0?
    ew_b_values = [r['b_coefficient'] for r in results["Edwards-Wilkinson"]['individual_results']]
    t_stat_ew, p_val_ew = stats.ttest_1samp(ew_b_values, 0)
    print(f"EW b ≈ 0 test: t={t_stat_ew:.3f}, p={p_val_ew:.2e} {'✓ consistent with 0' if p_val_ew > 0.05 else '✗ NOT zero'}")
    
    # Test: KPZ-class vs EW-class difference
    kpz_class_b = kpz_b_values + [r['b_coefficient'] for r in results["Ballistic Deposition"]['individual_results']] + \
                  [r['b_coefficient'] for r in results["EDEN"]['individual_results']]
    ew_class_b = ew_b_values + [r['b_coefficient'] for r in results["Random Deposition"]['individual_results']]
    
    t_stat_diff, p_val_diff = stats.ttest_ind(kpz_class_b, ew_class_b)
    print(f"KPZ-class vs EW-class: t={t_stat_diff:.3f}, p={p_val_diff:.2e} {'✓ DIFFERENT' if p_val_diff < 0.05 else '✗ NOT different'}")
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)
    
    if kpz_class_avg > 2 * ew_class_avg and p_val_diff < 0.05:
        print("\n✅ SUCCESS: KPZ nonlinearity IS detectable!")
        print("   → The (∇h)² term contributes positively to growth in KPZ-class models")
        print("   → Recommendation: Use b coefficient as ML feature (Φ_λ)")
        print("   → Proceed to Priority 2: Explicit Φ_λ feature extraction")
        diagnosis = "SUCCESS"
    elif kpz_class_avg > ew_class_avg:
        print("\n⚠️ PARTIAL: Weak nonlinearity signal detected")
        print("   → KPZ-class shows higher b, but difference is marginal")
        print("   → May need longer simulations or different analysis")
        print("   → Recommendation: Try without smoothing, or increase T")
        diagnosis = "PARTIAL"
    else:
        print("\n❌ FAILURE: No nonlinearity detected!")
        print("   → KPZ and EW have similar slope-growth coupling")
        print("   → Possible causes:")
        print("      1. Simulations too short for nonlinearity to dominate")
        print("      2. Discretization artifacts mask the signal")
        print("      3. Need different diagnostic approach")
        print("   → Recommendation: Check simulation implementation")
        diagnosis = "FAILURE"
    
    # Create visualization
    print("\n\nGenerating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    model_names = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: b coefficients comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(model_names))
    b_means = [results[m]['b_mean'] for m in model_names]
    b_stds = [results[m]['b_std'] for m in model_names]
    bars = ax1.bar(x_pos, b_means, yerr=b_stds, capsize=5, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m[:10] for m in model_names], rotation=45, ha='right')
    ax1.set_ylabel('Slope-Growth Coefficient (b)')
    ax1.set_title('KPZ Nonlinearity Diagnostic: b coefficient\n(g = a + b·s²)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: r² values
    ax2 = axes[0, 1]
    r2_means = [results[m]['r2_mean'] for m in model_names]
    r2_stds = [results[m]['r2_std'] for m in model_names]
    ax2.bar(x_pos, r2_means, yerr=r2_stds, capsize=5, color=colors, alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m[:10] for m in model_names], rotation=45, ha='right')
    ax2.set_ylabel('R² (variance explained)')
    ax2.set_title('Regression Quality')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Normalized b
    ax3 = axes[0, 2]
    b_norm_means = [results[m]['b_norm_mean'] for m in model_names]
    b_norm_stds = [results[m]['b_norm_std'] for m in model_names]
    ax3.bar(x_pos, b_norm_means, yerr=b_norm_stds, capsize=5, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([m[:10] for m in model_names], rotation=45, ha='right')
    ax3.set_ylabel('Normalized b')
    ax3.set_title('Dimensionless Coupling Strength')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-5: Scatter plots for representative models (EW vs KPZ)
    # Generate one sample each for visualization
    for idx, (ax, model_name, model_type) in enumerate([
        (axes[1, 0], "Edwards-Wilkinson", "edwards_wilkinson"),
        (axes[1, 1], "KPZ equation", "kpz_equation"),
        (axes[1, 2], "Ballistic Deposition", "ballistic_deposition"),
    ]):
        simulator = GrowthModelSimulator(width=L, height=T, random_state=999+idx)
        surface = simulator.generate_trajectory(model_type)
        
        # Compute slopes and growth
        smoothed = np.array([gaussian_filter1d(surface[t], smooth_sigma) for t in range(T)])
        slopes = np.zeros((T, L))
        slopes[:, 1:-1] = (smoothed[:, 2:] - smoothed[:, :-2]) / 2.0
        growth = smoothed[1:, :] - smoothed[:-1, :]
        
        s_flat = slopes[:-1, 10:-10].flatten()
        g_flat = growth[:, 10:-10].flatten()
        
        # Subsample for plotting
        n_plot = min(5000, len(s_flat))
        idx_plot = np.random.choice(len(s_flat), n_plot, replace=False)
        
        ax.scatter(s_flat[idx_plot]**2, g_flat[idx_plot], alpha=0.1, s=1, c=colors[list(results.keys()).index(model_name)])
        
        # Fit line
        s2 = s_flat**2
        slope, intercept, _, _, _ = stats.linregress(s2, g_flat)
        x_line = np.linspace(0, np.percentile(s2, 99), 100)
        ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2, label=f'b = {slope:.4f}')
        
        ax.set_xlabel('Slope² (s²)')
        ax.set_ylabel('Growth (g)')
        ax.set_title(f'{model_name}\nb = {slope:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.percentile(s2, 99))
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'exp13_slope_growth_coupling.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    
    plt.close()
    
    # Summary statistics for EXPERIMENT_LOG
    print("\n" + "=" * 70)
    print("SUMMARY FOR EXPERIMENT LOG")
    print("=" * 70)
    print(f"""
| Model | b coefficient | b (normalized) | r² | Interpretation |
|-------|---------------|----------------|-----|----------------|
| Edwards-Wilkinson | {results['Edwards-Wilkinson']['b_mean']:.6f} ± {results['Edwards-Wilkinson']['b_std']:.6f} | {results['Edwards-Wilkinson']['b_norm_mean']:.4f} | {results['Edwards-Wilkinson']['r2_mean']:.4f} | λ = 0 expected |
| KPZ equation | {results['KPZ equation']['b_mean']:.6f} ± {results['KPZ equation']['b_std']:.6f} | {results['KPZ equation']['b_norm_mean']:.4f} | {results['KPZ equation']['r2_mean']:.4f} | λ ≠ 0 expected |
| Ballistic Deposition | {results['Ballistic Deposition']['b_mean']:.6f} ± {results['Ballistic Deposition']['b_std']:.6f} | {results['Ballistic Deposition']['b_norm_mean']:.4f} | {results['Ballistic Deposition']['r2_mean']:.4f} | KPZ class |
| EDEN | {results['EDEN']['b_mean']:.6f} ± {results['EDEN']['b_std']:.6f} | {results['EDEN']['b_norm_mean']:.4f} | {results['EDEN']['r2_mean']:.4f} | KPZ class |
| Random Deposition | {results['Random Deposition']['b_mean']:.6f} ± {results['Random Deposition']['b_std']:.6f} | {results['Random Deposition']['b_norm_mean']:.4f} | {results['Random Deposition']['r2_mean']:.4f} | Uncorrelated |

KPZ-class vs EW-class t-test: p = {p_val_diff:.2e}
Diagnosis: {diagnosis}
""")
    
    return results, diagnosis


if __name__ == "__main__":
    results, diagnosis = main()
