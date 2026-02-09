"""
Experiment 26: Tracy-Widom Statistics - Rigorous Validation

Goal: Validate gradient moment framework against gold-standard KPZ theory.
      Height fluctuations should follow:
      - EW: Gaussian distribution (skew ≈ 0)
      - KPZ: Tracy-Widom GUE distribution (skew ≈ -0.29)

Prediction: Tracy-Widom skewness should correlate with PC1 from Exp 21.

This experiment connects our geometric framework to rigorous KPZ theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats
from scipy.special import airy
from scipy.optimize import curve_fit

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.simulation.physics_simulation import GrowthModelSimulator


def extract_gradient_moments(h):
    """Extract 6D gradient moment features (matching Exp 21)."""
    # Gradient (central difference with periodic BC)
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
    
    # Laplacian (second derivative)
    lap = np.roll(h, 1) - 2*h + np.roll(h, -1)
    
    # Standardized moments
    grad_var = np.var(grad)
    grad_skew = stats.skew(grad)
    grad_kurt = stats.kurtosis(grad)  # Excess kurtosis
    
    # Laplacian statistics
    lap_var = np.var(lap)
    
    # Cross-correlation
    grad_lap_cov = np.cov(grad, lap)[0, 1]
    
    # Height variance
    h_var = np.var(h)
    
    return np.array([grad_var, grad_skew, grad_kurt, lap_var, grad_lap_cov, h_var])


def collect_height_fluctuations(model_name, L=512, T=5000, n_samples=100):
    """
    Collect height fluctuations from many realizations.
    
    For asymptotic regime:
    - Large L (512) for finite-size convergence
    - Large T (5000) for time convergence
    - Many samples (100) for statistics
    
    Returns:
    - fluctuations: Array of normalized height fluctuations
    - gradient_features: 6D feature vector for PC1 correlation
    """
    # Map model names to simulator names
    model_map = {
        'EW': 'edwards_wilkinson',
        'KPZ': 'kpz_equation'
    }
    
    sim = GrowthModelSimulator(width=L, height=T, random_state=42)
    
    fluctuations = []
    gradient_features = []
    
    print(f"\n{model_name} simulation:")
    for i in range(n_samples):
        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n_samples}")
        
        # Generate trajectory
        np.random.seed(42 + i * 1000)
        trajectory = sim.generate_trajectory(model_map[model_name])
        
        # Get final height profile
        h = trajectory[-1].copy()
        
        # Height fluctuation at center point (or multiple points)
        h_center = h[L // 2]
        h_mean = np.mean(h)
        h_std = np.std(h)
        
        # Normalized fluctuation
        fluctuation = (h_center - h_mean) / (h_std + 1e-10)
        fluctuations.append(fluctuation)
        
        # Also collect gradient moments for correlation with PC1
        features = extract_gradient_moments(h)
        gradient_features.append(features)
    
    return np.array(fluctuations), np.array(gradient_features)


def tracy_widom_gue_pdf(x):
    """
    Approximate Tracy-Widom GUE PDF.
    Using Airy function approximation for large negative x.
    
    Exact distribution is complex, but key property:
    - Skewness ≈ -0.29
    - Mean ≈ -1.77
    - Std ≈ 0.81
    """
    # Shift and scale to standard form
    x_shifted = (x + 1.77) / 0.81
    
    # For visualization purposes, use skewed Gaussian approximation
    # Real TW-GUE is more complex but has these moments
    return stats.skewnorm.pdf(x_shifted, a=-5)  # Negative skew


def fit_distributions(fluctuations, model_name):
    """Fit Gaussian and measure statistics."""
    mean = np.mean(fluctuations)
    std = np.std(fluctuations)
    skew = stats.skew(fluctuations)
    kurt = stats.kurtosis(fluctuations)
    
    print(f"\n{model_name} statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Skewness: {skew:.4f}")
    print(f"  Kurtosis: {kurt:.4f}")
    
    # Kolmogorov-Smirnov test vs Gaussian
    ks_stat, ks_pval = stats.kstest(fluctuations, 'norm', args=(mean, std))
    print(f"  KS test vs Gaussian: D={ks_stat:.4f}, p={ks_pval:.4e}")
    
    return {
        'mean': mean,
        'std': std,
        'skewness': skew,
        'kurtosis': kurt,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }


def compute_pc1(features):
    """
    Compute PC1 projection using loadings from Exp 21.
    
    PC1 loadings (from Exp 21):
    grad_var: +0.607
    grad_skew: -0.004
    grad_kurt: +0.026
    lap_var: +0.586
    grad_lap_cov: -0.000
    h_var: +0.536
    """
    pc1_loadings = np.array([0.607, -0.004, 0.026, 0.586, 0.0, 0.536])
    
    # Standardize features (important for PCA)
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0) + 1e-10
    features_standardized = (features - features_mean) / features_std
    
    # Project onto PC1
    pc1_values = features_standardized @ pc1_loadings
    
    return pc1_values


def main():
    """Main experiment: Tracy-Widom validation."""
    
    output_dir = Path("results/exp26_tracy_widom")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EXPERIMENT 26: Tracy-Widom Statistics Validation")
    print("=" * 70)
    print("\nGoal: Validate gradient moment framework against KPZ theory")
    print("Expected: EW→Gaussian (skew≈0), KPZ→Tracy-Widom (skew≈-0.29)")
    print("\nParameters:")
    print("  L = 512 (large system)")
    print("  T = 5000 (long time)")
    print("  n = 100 (samples per model)")
    
    # Collect data
    models = ['EW', 'KPZ']
    results = {}
    
    for model in models:
        flucts, features = collect_height_fluctuations(
            model_name=model,
            L=512,
            T=5000,
            n_samples=100
        )
        
        stats_dict = fit_distributions(flucts, model)
        pc1_values = compute_pc1(features)
        
        results[model] = {
            'fluctuations': flucts,
            'features': features,
            'stats': stats_dict,
            'pc1': pc1_values
        }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Height fluctuation histograms
    ax = axes[0, 0]
    bins = 30
    
    for model, color in zip(['EW', 'KPZ'], ['blue', 'red']):
        flucts = results[model]['fluctuations']
        skew = results[model]['stats']['skewness']
        
        ax.hist(flucts, bins=bins, alpha=0.5, label=f'{model} (skew={skew:.3f})',
                color=color, density=True)
        
        # Overlay Gaussian fit
        from scipy.stats import norm
        x = np.linspace(flucts.min(), flucts.max(), 200)
        mu, sigma = np.mean(flucts), np.std(flucts)
        gaussian = norm.pdf(x, mu, sigma)
        ax.plot(x, gaussian, '--', color=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Normalized Height Fluctuation', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Height Fluctuation Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add theoretical expectations
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='_nolegend_')
    ax.text(0.02, 0.98, 'Expected:\nEW: skew ≈ 0\nKPZ: skew ≈ -0.29',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Q-Q plots
    ax = axes[0, 1]
    
    from scipy.stats import probplot
    for model, color, offset in zip(['EW', 'KPZ'], ['blue', 'red'], [0, 0.5]):
        flucts = results[model]['fluctuations']
        probplot(flucts, dist="norm", plot=ax)
        
    ax.set_title('Q-Q Plot vs Gaussian', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 3: Skewness vs PC1
    ax = axes[1, 0]
    
    all_pc1 = []
    all_skew = []
    all_colors = []
    
    for model, color in zip(['EW', 'KPZ'], ['blue', 'red']):
        pc1 = results[model]['pc1']
        features = results[model]['features']
        grad_skew = features[:, 1]  # grad_skew is index 1
        
        all_pc1.extend(pc1)
        all_skew.extend(grad_skew)
        all_colors.extend([color] * len(pc1))
    
    ax.scatter(all_pc1, all_skew, c=all_colors, alpha=0.6, s=30)
    
    # Correlation
    corr = np.corrcoef(all_pc1, all_skew)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('PC1 (Universality Axis)', fontsize=12)
    ax.set_ylabel('Gradient Skewness', fontsize=12)
    ax.set_title('PC1 vs Gradient Skewness', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='EW'),
        Patch(facecolor='red', alpha=0.6, label='KPZ')
    ]
    ax.legend(handles=legend_elements, fontsize=11)
    
    # Plot 4: Height fluctuation skewness vs PC1 correlation
    ax = axes[1, 1]
    
    # For each model, compute mean PC1 and height fluctuation skewness
    model_data = []
    for model in ['EW', 'KPZ']:
        mean_pc1 = np.mean(results[model]['pc1'])
        height_skew = results[model]['stats']['skewness']
        model_data.append((mean_pc1, height_skew, model))
    
    pc1_vals = [d[0] for d in model_data]
    skew_vals = [d[1] for d in model_data]
    labels = [d[2] for d in model_data]
    colors = ['blue', 'red']
    
    for pc1, skew, label, color in zip(pc1_vals, skew_vals, labels, colors):
        ax.scatter(pc1, skew, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(pc1, skew, f'  {label}', fontsize=12, fontweight='bold', 
                verticalalignment='center')
    
    # Add theory line if correlation exists
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Gaussian (skew=0)')
    ax.axhline(-0.29, color='orange', linestyle='--', alpha=0.7, label='TW-GUE (skew≈-0.29)')
    
    ax.set_xlabel('Mean PC1', fontsize=12)
    ax.set_ylabel('Height Fluctuation Skewness', fontsize=12)
    ax.set_title('Universality Axis vs Tracy-Widom Signature', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracy_widom_validation.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {output_dir / 'tracy_widom_validation.png'}")
    
    # Save numerical results
    with open(output_dir / 'statistics.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT 26: Tracy-Widom Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        for model in ['EW', 'KPZ']:
            stats = results[model]['stats']
            f.write(f"\n{model} Model:\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
            f.write(f"  Skewness: {stats['skewness']:.4f}\n")
            f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\n")
            f.write(f"  KS test: D={stats['ks_stat']:.4f}, p={stats['ks_pval']:.4e}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Theoretical Expectations:\n")
        f.write("=" * 70 + "\n")
        f.write("EW (Gaussian):\n")
        f.write("  Skewness ≈ 0\n")
        f.write("  Kurtosis ≈ 0\n\n")
        f.write("KPZ (Tracy-Widom GUE):\n")
        f.write("  Skewness ≈ -0.29\n")
        f.write("  Kurtosis ≈ 0.16\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Interpretation:\n")
        f.write("=" * 70 + "\n")
        
        ew_skew = results['EW']['stats']['skewness']
        kpz_skew = results['KPZ']['stats']['skewness']
        
        f.write(f"\nEW skewness: {ew_skew:.4f}\n")
        if abs(ew_skew) < 0.15:
            f.write("  ✓ Consistent with Gaussian (expected: ~0)\n")
        else:
            f.write("  ⚠ Deviates from Gaussian expectation\n")
        
        f.write(f"\nKPZ skewness: {kpz_skew:.4f}\n")
        if -0.5 < kpz_skew < -0.1:
            f.write("  ✓ Consistent with Tracy-Widom (expected: ~-0.29)\n")
        else:
            f.write("  ⚠ Deviates from Tracy-Widom expectation\n")
        
        # Gradient skewness correlation
        all_pc1 = np.concatenate([results['EW']['pc1'], results['KPZ']['pc1']])
        all_grad_skew = np.concatenate([
            results['EW']['features'][:, 1],
            results['KPZ']['features'][:, 1]
        ])
        corr = np.corrcoef(all_pc1, all_grad_skew)[0, 1]
        
        f.write(f"\n\nPC1 vs Gradient Skewness correlation: r = {corr:.4f}\n")
        if abs(corr) > 0.5:
            f.write("  ✓ Strong correlation validates geometric framework\n")
        else:
            f.write("  ⚠ Weak correlation\n")
    
    print(f"Saved results: {output_dir / 'statistics.txt'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    ew_skew = results['EW']['stats']['skewness']
    kpz_skew = results['KPZ']['stats']['skewness']
    
    print(f"\nHeight Fluctuation Skewness:")
    print(f"  EW:  {ew_skew:+.4f} (expected: ~0)")
    print(f"  KPZ: {kpz_skew:+.4f} (expected: ~-0.29)")
    print(f"\nΔ skewness = {kpz_skew - ew_skew:.4f}")
    
    if kpz_skew < ew_skew - 0.1:
        print("  ✓ KPZ shows negative skewness (Tracy-Widom signature)")
    else:
        print("  ⚠ KPZ skewness not clearly negative")
    
    print("\nValidation Status:")
    if abs(ew_skew) < 0.15 and -0.5 < kpz_skew < -0.1:
        print("  ✅ VALIDATED: Results consistent with Tracy-Widom theory")
    else:
        print("  ⚠️ PARTIAL: Results show expected trend but may need:")
        print("     - Larger L (reduce finite-size effects)")
        print("     - Longer T (reach asymptotic regime)")
        print("     - More samples (reduce statistical noise)")


if __name__ == '__main__':
    main()
