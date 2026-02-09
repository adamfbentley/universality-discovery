"""
Experiment 14: Coarse-Grained Slope-Growth Coupling

GOAL: Test whether the KPZ nonlinearity λ(∇h)² EMERGES in discrete models 
(BD, EDEN) after coarse-graining, while remaining absent in EW/RD.

THEORETICAL BASIS (from Exp 13 + Grok's assessment):
- Exp 13 showed: b > 0 for continuum KPZ, b ≈ 0 for discrete BD/EDEN
- Hypothesis: Discrete models are in KPZ universality class, but the effective
  nonlinearity only emerges at COARSE SCALES after RG flow
- Coarse-graining (Gaussian blur σ) acts as RG transformation
- If b(σ) increases for BD/EDEN as σ increases → universality confirmed

PREDICTION:
- EW: b(σ) ≈ 0 for all σ (no nonlinearity at any scale)
- KPZ: b(σ) > 0 for all σ (explicit nonlinearity)
- BD/EDEN: b(σ=0) ≈ 0 → b(σ→∞) > 0 (emergent nonlinearity via RG)
- RD: b(σ) ≈ 0 for all σ (uncorrelated, no structure)

This tests the core idea: universality = emergent nonlinearity in measures,
where local stats (gradients) are RG-relevant geodesics (Cotler-Rezchikov).
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
                                   coarse_sigma: float = 0.0) -> dict:
    """
    Compute slope-growth coupling coefficient from surface evolution.
    
    Args:
        surface_evolution: Array of shape (T, L) - height profile at each time
        coarse_sigma: Gaussian coarse-graining scale (0 = no coarse-graining)
    
    Returns:
        dict with regression results and statistics
    """
    T, L = surface_evolution.shape
    
    # Apply coarse-graining (RG transformation)
    if coarse_sigma > 0:
        coarsened = np.array([gaussian_filter1d(surface_evolution[t], coarse_sigma) 
                             for t in range(T)])
    else:
        coarsened = surface_evolution.copy()
    
    # Compute local slopes: s = ∇ₓh (spatial gradient)
    # Use central differences for interior points
    slopes = np.zeros((T, L))
    slopes[:, 1:-1] = (coarsened[:, 2:] - coarsened[:, :-2]) / 2.0
    slopes[:, 0] = coarsened[:, 1] - coarsened[:, 0]
    slopes[:, -1] = coarsened[:, -1] - coarsened[:, -2]
    
    # Compute local growth: g = Δₜh (temporal derivative)
    growth = coarsened[1:, :] - coarsened[:-1, :]
    
    # Align: use slope at time t to predict growth from t to t+1
    slopes_aligned = slopes[:-1, :]
    
    # Flatten for regression (exclude boundaries)
    margin = max(5, int(2 * coarse_sigma))  # Larger margin for larger σ
    s_flat = slopes_aligned[:, margin:-margin].flatten()
    g_flat = growth[:, margin:-margin].flatten()
    s_squared = s_flat ** 2
    
    # Remove outliers (beyond 3 sigma)
    g_mean, g_std = np.mean(g_flat), np.std(g_flat)
    s2_mean, s2_std = np.mean(s_squared), np.std(s_squared)
    
    mask = (np.abs(g_flat - g_mean) < 3 * g_std) & (np.abs(s_squared - s2_mean) < 3 * s2_std)
    s_squared_clean = s_squared[mask]
    g_clean = g_flat[mask]
    
    # Linear regression: g = a + b * s²
    slope_coeff, intercept, r_value, p_value, std_err = stats.linregress(s_squared_clean, g_clean)
    
    # Compute correlation
    correlation = np.corrcoef(s_squared_clean, g_clean)[0, 1]
    
    # Mean values for normalization
    mean_s_squared = np.mean(s_squared_clean)
    mean_growth = np.mean(g_clean)
    std_growth = np.std(g_clean)
    
    # Normalized coefficient (dimensionless)
    if np.abs(mean_growth) > 1e-10:
        b_normalized = slope_coeff * mean_s_squared / np.abs(mean_growth)
    else:
        b_normalized = slope_coeff * mean_s_squared / (std_growth + 1e-10)
    
    return {
        'b_coefficient': slope_coeff,
        'b_normalized': b_normalized,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'correlation': correlation,
        'mean_slope_sq': mean_s_squared,
        'mean_growth': mean_growth,
        'n_points': len(s_squared_clean),
        'coarse_sigma': coarse_sigma,
    }


def analyze_model_across_scales(model_name: str, model_type: str, 
                                 sigmas: list, n_samples: int = 15,
                                 L: int = 256, T: int = 1000) -> dict:
    """
    Analyze slope-growth coupling across multiple coarse-graining scales.
    """
    print(f"\n  Analyzing {model_name} across σ = {sigmas}...")
    
    results_by_sigma = {sigma: [] for sigma in sigmas}
    
    for i in range(n_samples):
        # Generate surface evolution once
        simulator = GrowthModelSimulator(width=L, height=T, random_state=42 + i)
        surface = simulator.generate_trajectory(model_type)
        
        # Compute coupling at each scale
        for sigma in sigmas:
            result = compute_slope_growth_coupling(surface, coarse_sigma=sigma)
            results_by_sigma[sigma].append(result)
        
        if (i + 1) % 5 == 0:
            # Show progress for σ=0 and max σ
            b0 = results_by_sigma[sigmas[0]][i]['b_coefficient']
            b_max = results_by_sigma[sigmas[-1]][i]['b_coefficient']
            print(f"    Sample {i+1}/{n_samples}: b(σ=0)={b0:.5f}, b(σ={sigmas[-1]})={b_max:.5f}")
    
    # Aggregate statistics per sigma
    aggregated = {}
    for sigma in sigmas:
        b_values = [r['b_coefficient'] for r in results_by_sigma[sigma]]
        b_norm_values = [r['b_normalized'] for r in results_by_sigma[sigma]]
        r2_values = [r['r_squared'] for r in results_by_sigma[sigma]]
        
        aggregated[sigma] = {
            'b_mean': np.mean(b_values),
            'b_std': np.std(b_values),
            'b_norm_mean': np.mean(b_norm_values),
            'b_norm_std': np.std(b_norm_values),
            'r2_mean': np.mean(r2_values),
            'r2_std': np.std(r2_values),
            'individual': results_by_sigma[sigma],
        }
    
    return {
        'model': model_name,
        'model_type': model_type,
        'results': aggregated,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 14: COARSE-GRAINED SLOPE-GROWTH COUPLING")
    print("=" * 70)
    print("\nTesting whether KPZ nonlinearity EMERGES in discrete models via RG")
    print("Hypothesis: b(σ) increases for BD/EDEN as coarse-graining σ increases")
    
    # Parameters
    L = 256
    T = 1000
    n_samples = 15
    sigmas = [0, 1, 2, 4, 8, 16]  # Coarse-graining scales
    
    print(f"\nParameters: L={L}, T={T}, n_samples={n_samples}")
    print(f"Coarse-graining scales σ: {sigmas}")
    
    # Models to test
    models = [
        ("Edwards-Wilkinson", "edwards_wilkinson"),
        ("KPZ equation", "kpz_equation"),
        ("Ballistic Deposition", "ballistic_deposition"),
        ("EDEN", "eden"),
        ("Random Deposition", "random_deposition"),
    ]
    
    # Analyze each model
    all_results = {}
    for name, model_type in models:
        all_results[name] = analyze_model_across_scales(
            name, model_type, sigmas, n_samples=n_samples, L=L, T=T
        )
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: b(σ) ACROSS COARSE-GRAINING SCALES")
    print("=" * 70)
    
    # Header
    header = f"{'Model':<20}"
    for sigma in sigmas:
        header += f" σ={sigma:<8}"
    print(header)
    print("-" * (20 + 10 * len(sigmas)))
    
    # Data rows
    for name in all_results:
        row = f"{name:<20}"
        for sigma in sigmas:
            b_mean = all_results[name]['results'][sigma]['b_mean']
            row += f" {b_mean:+.5f} "
        print(row)
    
    # Print with standard deviations
    print("\n--- With Standard Deviations ---")
    for name in all_results:
        print(f"\n{name}:")
        for sigma in sigmas:
            b_mean = all_results[name]['results'][sigma]['b_mean']
            b_std = all_results[name]['results'][sigma]['b_std']
            print(f"  σ={sigma:2d}: b = {b_mean:+.5f} ± {b_std:.5f}")
    
    # Analyze trends
    print("\n" + "=" * 70)
    print("TREND ANALYSIS: Does b increase with σ?")
    print("=" * 70)
    
    trends = {}
    for name in all_results:
        b_values = [all_results[name]['results'][sigma]['b_mean'] for sigma in sigmas]
        
        # Compute correlation of b with log(σ+1) to detect RG scaling
        log_sigmas = np.log(np.array(sigmas) + 1)
        corr, p_val = stats.pearsonr(log_sigmas, b_values)
        
        # Linear regression for slope
        slope, intercept, r, p, se = stats.linregress(log_sigmas, b_values)
        
        trends[name] = {
            'correlation': corr,
            'p_value': p_val,
            'slope': slope,
            'b_initial': b_values[0],
            'b_final': b_values[-1],
            'change': b_values[-1] - b_values[0],
        }
        
        trend_str = "↑ INCREASING" if slope > 0.001 else ("↓ DECREASING" if slope < -0.001 else "→ FLAT")
        sig_str = "✓" if p_val < 0.1 else ""
        print(f"{name:<20}: {trend_str} (slope={slope:.5f}, p={p_val:.3f}) {sig_str}")
    
    # Key diagnostic: Do discrete models show emergent nonlinearity?
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTIC: EMERGENT NONLINEARITY IN DISCRETE MODELS")
    print("=" * 70)
    
    # Compare b at σ=0 vs σ=16 for each model
    print("\n| Model | b(σ=0) | b(σ=16) | Change | Interpretation |")
    print("|-------|--------|---------|--------|----------------|")
    
    interpretations = {
        "Edwards-Wilkinson": "λ=0 at all scales",
        "KPZ equation": "λ explicit, should persist",
        "Ballistic Deposition": "λ emergent via RG?",
        "EDEN": "λ emergent via RG?",
        "Random Deposition": "No structure at any scale",
    }
    
    for name in all_results:
        b0 = all_results[name]['results'][0]['b_mean']
        b16 = all_results[name]['results'][16]['b_mean']
        change = b16 - b0
        interp = interpretations[name]
        print(f"| {name:<20} | {b0:+.4f} | {b16:+.4f} | {change:+.4f} | {interp} |")
    
    # Statistical test: Are BD/EDEN at σ=16 distinguishable from EW?
    print("\n--- Statistical Tests at σ=16 ---")
    
    bd_b16 = [r['b_coefficient'] for r in all_results["Ballistic Deposition"]['results'][16]['individual']]
    eden_b16 = [r['b_coefficient'] for r in all_results["EDEN"]['results'][16]['individual']]
    ew_b16 = [r['b_coefficient'] for r in all_results["Edwards-Wilkinson"]['results'][16]['individual']]
    kpz_b16 = [r['b_coefficient'] for r in all_results["KPZ equation"]['results'][16]['individual']]
    
    # Combine KPZ-class discrete at σ=16
    kpz_class_discrete_b16 = bd_b16 + eden_b16
    
    # Test: KPZ-class discrete vs EW at σ=16
    t_stat, p_val = stats.ttest_ind(kpz_class_discrete_b16, ew_b16)
    print(f"KPZ-class discrete (BD+EDEN) vs EW at σ=16: t={t_stat:.3f}, p={p_val:.4f}")
    
    # Test: Do BD/EDEN at σ=16 have b > 0?
    t_bd, p_bd = stats.ttest_1samp(bd_b16, 0)
    t_eden, p_eden = stats.ttest_1samp(eden_b16, 0)
    print(f"BD at σ=16: b > 0? t={t_bd:.3f}, p={p_bd:.4f}")
    print(f"EDEN at σ=16: b > 0? t={t_eden:.3f}, p={p_eden:.4f}")
    
    # Final diagnosis
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)
    
    # Check if discrete models show increasing b
    bd_increasing = trends["Ballistic Deposition"]['slope'] > 0
    eden_increasing = trends["EDEN"]['slope'] > 0
    ew_flat = abs(trends["Edwards-Wilkinson"]['slope']) < 0.005
    
    bd_b16_mean = np.mean(bd_b16)
    eden_b16_mean = np.mean(eden_b16)
    ew_b16_mean = np.mean(ew_b16)
    
    if (bd_increasing or eden_increasing) and ew_flat:
        if bd_b16_mean > ew_b16_mean and eden_b16_mean > ew_b16_mean:
            print("\n✅ SUCCESS: EMERGENT NONLINEARITY DETECTED!")
            print("   → BD and/or EDEN show increasing b with coarse-graining")
            print("   → EW remains flat (no nonlinearity at any scale)")
            print("   → This confirms: universality = emergent λ via RG flow")
            print("   → Discrete models ARE in KPZ class, just need coarse-graining to see it")
            diagnosis = "SUCCESS"
        else:
            print("\n⚠️ PARTIAL: Trends suggest emergence, but magnitude weak")
            diagnosis = "PARTIAL"
    elif bd_b16_mean > 0.01 or eden_b16_mean > 0.01:
        print("\n⚠️ PARTIAL: Some positive b at coarse scales, but trend unclear")
        print("   → May need even coarser scales or longer simulations")
        diagnosis = "PARTIAL"
    else:
        print("\n❌ INCONCLUSIVE: No clear emergent nonlinearity")
        print("   → Possible causes:")
        print("      1. Coarse-graining scales still too fine")
        print("      2. System size L=256 insufficient")
        print("      3. Discrete growth rules fundamentally different")
        diagnosis = "INCONCLUSIVE"
    
    # Visualization
    print("\n\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        "Edwards-Wilkinson": 'blue',
        "KPZ equation": 'red',
        "Ballistic Deposition": 'green',
        "EDEN": 'orange',
        "Random Deposition": 'purple',
    }
    
    # Plot 1: b(σ) for all models
    ax1 = axes[0, 0]
    for name in all_results:
        b_means = [all_results[name]['results'][sigma]['b_mean'] for sigma in sigmas]
        b_stds = [all_results[name]['results'][sigma]['b_std'] for sigma in sigmas]
        ax1.errorbar(sigmas, b_means, yerr=b_stds, marker='o', label=name, 
                    color=colors[name], capsize=3, linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Coarse-graining scale σ')
    ax1.set_ylabel('Slope-growth coefficient b')
    ax1.set_title('b(σ): Does nonlinearity emerge with coarse-graining?')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: b(σ) on log scale for σ
    ax2 = axes[0, 1]
    for name in all_results:
        b_means = [all_results[name]['results'][sigma]['b_mean'] for sigma in sigmas]
        log_sigmas_plot = [np.log(s + 1) for s in sigmas]
        ax2.plot(log_sigmas_plot, b_means, marker='o', label=name, 
                color=colors[name], linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('log(σ + 1)')
    ax2.set_ylabel('Slope-growth coefficient b')
    ax2.set_title('b vs log(σ+1): RG scaling behavior')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Change in b from σ=0 to σ=16
    ax3 = axes[1, 0]
    model_names = list(all_results.keys())
    changes = [trends[name]['change'] for name in model_names]
    bars = ax3.bar(range(len(model_names)), changes, 
                   color=[colors[name] for name in model_names], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels([n[:10] for n in model_names], rotation=45, ha='right')
    ax3.set_ylabel('Δb = b(σ=16) - b(σ=0)')
    ax3.set_title('Change in b with coarse-graining')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Comparison at σ=0 vs σ=16
    ax4 = axes[1, 1]
    x = np.arange(len(model_names))
    width = 0.35
    
    b0_means = [all_results[name]['results'][0]['b_mean'] for name in model_names]
    b16_means = [all_results[name]['results'][16]['b_mean'] for name in model_names]
    b0_stds = [all_results[name]['results'][0]['b_std'] for name in model_names]
    b16_stds = [all_results[name]['results'][16]['b_std'] for name in model_names]
    
    ax4.bar(x - width/2, b0_means, width, yerr=b0_stds, label='σ=0 (microscopic)', 
            color='lightblue', capsize=3, edgecolor='blue')
    ax4.bar(x + width/2, b16_means, width, yerr=b16_stds, label='σ=16 (coarse)', 
            color='lightcoral', capsize=3, edgecolor='red')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels([n[:10] for n in model_names], rotation=45, ha='right')
    ax4.set_ylabel('Slope-growth coefficient b')
    ax4.set_title('Microscopic vs Coarse-grained coupling')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    fig_path = figures_dir / 'exp14_coarsegrained_slope_growth.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()
    
    # Summary table for experiment log
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR EXPERIMENT LOG")
    print("=" * 70)
    
    print("\n### Results: b(σ) across coarse-graining scales\n")
    print("| Model | σ=0 | σ=1 | σ=2 | σ=4 | σ=8 | σ=16 | Trend |")
    print("|-------|-----|-----|-----|-----|-----|------|-------|")
    for name in all_results:
        row = f"| {name} |"
        for sigma in sigmas:
            b = all_results[name]['results'][sigma]['b_mean']
            row += f" {b:+.4f} |"
        trend = "↑" if trends[name]['slope'] > 0.001 else ("↓" if trends[name]['slope'] < -0.001 else "→")
        row += f" {trend} |"
        print(row)
    
    print(f"\nDiagnosis: {diagnosis}")
    print(f"KPZ-class discrete vs EW at σ=16: p = {p_val:.4f}")
    
    return all_results, trends, diagnosis


if __name__ == "__main__":
    all_results, trends, diagnosis = main()
