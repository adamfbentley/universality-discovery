"""
Experiment 7: EW vs KPZ Discrimination with Discrete-Trained Model
===================================================================

Key Question: Can the discrete-trained autoencoder (from Exp 6) distinguish
EW (different universality class) from KPZ (same class as BD/EDEN)?

This tests whether the model respects universality class boundaries, or
only detects "ease of reconstruction."

Hypothesis:
- If EW and KPZ scores are statistically different → Model respects universality
- If indistinguishable → Model only captures discrete/continuum, not class structure

From Exp 6:
- EW: 0.0106 ± 0.0004
- KPZ: 0.0070 ± 0.0002

These LOOK different, but are they statistically significant?
We need larger samples and proper statistical tests.

Links to Conjectures:
- Conjecture 3.1 (Separation): Different classes should have disjoint supports
- Conjecture 3.3 (Geometric Universality): Same class → same limit measure
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from scipy import stats

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
WIDTH = 128
TIME_STEPS = 500
N_TRAIN_PER_CLASS = 200  # BD + EDEN = 400 total
N_TEST = 200  # Larger samples for statistical power
EPOCHS = 25
LATENT_DIM = 32
SIGMA = 2  # Optimal blur from Exp 5


def compute_gradient_field(surface):
    """Compute spatial gradient of surface."""
    grad_x = np.roll(surface, -1, axis=0) - surface
    return grad_x


def apply_coarse_graining(field, sigma):
    """Apply Gaussian blur for coarse-graining."""
    if sigma == 0:
        return field
    return gaussian_filter(field, sigma=sigma, mode='wrap')


def preprocess_surface(surface, sigma=2):
    """Transform to gradient space and apply coarse-graining."""
    grad = compute_gradient_field(surface)
    return apply_coarse_graining(grad, sigma)


def generate_dataset(model_types, n_per_class, sigma=2, desc="Generating"):
    """Generate preprocessed surfaces for multiple model types."""
    all_surfaces = []
    
    for model_type in tqdm(model_types, desc=desc):
        surfaces = []
        for _ in range(n_per_class):
            sim = GrowthModelSimulator(WIDTH, TIME_STEPS)
            
            if model_type == 'edwards_wilkinson':
                surface = sim.generate_trajectory('edwards_wilkinson', diffusion=1.0)
            elif model_type == 'kpz_equation':
                surface = sim.generate_trajectory('kpz_equation', diffusion=1.0, nonlinearity=1.0)
            elif model_type == 'ballistic_deposition':
                surface = sim.generate_trajectory('ballistic_deposition')
            elif model_type == 'eden':
                surface = sim.generate_trajectory('eden')
            elif model_type == 'random_deposition':
                surface = sim.generate_trajectory('random_deposition')
            
            # Preprocess: gradient + coarse-graining
            processed = preprocess_surface(surface, sigma)
            surfaces.append(processed)
        
        all_surfaces.extend(surfaces)
    
    return np.array(all_surfaces)


def train_autoencoder(data, epochs=25, verbose=True):
    """Train autoencoder on preprocessed data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transpose to match model expectations: (N, time_steps, width) -> (N, width, time_steps)
    data_t = np.transpose(data, (0, 2, 1))
    
    # Normalize data
    mean = data_t.mean()
    std = data_t.std()
    data_norm = (data_t - mean) / (std + 1e-8)
    
    # Create model
    model = SurfaceAutoencoder(width=WIDTH, time_steps=TIME_STEPS, latent_dim=LATENT_DIM)
    model = model.to(device)
    
    # Prepare data
    tensor_data = torch.FloatTensor(data_norm).unsqueeze(1).to(device)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    for epoch in iterator:
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, z = model(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model, mean, std


def compute_anomaly_scores(model, data, mean, std):
    """Compute reconstruction error for each surface."""
    device = next(model.parameters()).device
    model.eval()
    
    # Transpose to match model expectations
    data_t = np.transpose(data, (0, 2, 1))
    
    data_norm = (data_t - mean) / (std + 1e-8)
    tensor_data = torch.FloatTensor(data_norm).unsqueeze(1).to(device)
    
    scores = []
    with torch.no_grad():
        for i in range(len(data)):
            x = tensor_data[i:i+1]
            recon, z = model(x)
            score = ((x - recon) ** 2).mean().item()
            scores.append(score)
    
    return np.array(scores)


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std


def main():
    print("=" * 70)
    print("Experiment 7: EW vs KPZ Discrimination with Discrete-Trained Model")
    print("=" * 70)
    print("\nKey Question: Does the model respect universality class boundaries?")
    print("- EW: Different universality class (α=1/2, β=1/4, z=2)")
    print("- KPZ: Same universality class as BD/EDEN (α=1/2, β=1/3, z=3/2)")
    print()
    
    print(f"Configuration:")
    print(f"  Grid: {WIDTH} x {TIME_STEPS}")
    print(f"  Coarse-graining σ = {SIGMA}")
    print(f"  Training: {N_TRAIN_PER_CLASS} BD + {N_TRAIN_PER_CLASS} EDEN (discrete)")
    print(f"  Testing: {N_TEST} EW, {N_TEST} KPZ (for statistical power)")
    print()
    
    # =========================================================================
    # Phase 1: Train on Discrete Models (BD + EDEN)
    # =========================================================================
    print("=" * 50)
    print("Phase 1: Training on Discrete Models")
    print("=" * 50)
    
    print("\nGenerating training data (BD + EDEN)...")
    train_data = generate_dataset(
        ['ballistic_deposition', 'eden'], 
        N_TRAIN_PER_CLASS, 
        sigma=SIGMA,
        desc="Training"
    )
    print(f"  Training shape: {train_data.shape}")
    
    print("\nTraining autoencoder...")
    model, mean, std = train_autoencoder(train_data, epochs=EPOCHS)
    
    # =========================================================================
    # Phase 2: Generate Test Data
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 2: Generating Test Data")
    print("=" * 50)
    
    print(f"\nGenerating {N_TEST} samples each for EW, KPZ, BD, EDEN, RD...")
    
    test_data = {}
    for model_name, model_type in [
        ('EW', 'edwards_wilkinson'),
        ('KPZ', 'kpz_equation'),
        ('BD', 'ballistic_deposition'),
        ('EDEN', 'eden'),
        ('RD', 'random_deposition')
    ]:
        test_data[model_name] = generate_dataset(
            [model_type], 
            N_TEST, 
            sigma=SIGMA,
            desc=f"  {model_name}"
        )
        print(f"  {model_name}: {test_data[model_name].shape}")
    
    # =========================================================================
    # Phase 3: Compute Anomaly Scores
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 3: Computing Anomaly Scores")
    print("=" * 50)
    
    scores = {}
    for name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        scores[name] = compute_anomaly_scores(model, test_data[name], mean, std)
        print(f"  {name}: {scores[name].mean():.6f} ± {scores[name].std():.6f}")
    
    # =========================================================================
    # Phase 4: Statistical Analysis - EW vs KPZ
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 4: Statistical Analysis - EW vs KPZ")
    print("=" * 50)
    
    ew_scores = scores['EW']
    kpz_scores = scores['KPZ']
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(f"  EW:  mean = {ew_scores.mean():.6f}, std = {ew_scores.std():.6f}, n = {len(ew_scores)}")
    print(f"  KPZ: mean = {kpz_scores.mean():.6f}, std = {kpz_scores.std():.6f}, n = {len(kpz_scores)}")
    
    # Bootstrap confidence intervals
    ew_ci = bootstrap_ci(ew_scores)
    kpz_ci = bootstrap_ci(kpz_scores)
    print(f"\n95% Bootstrap Confidence Intervals:")
    print(f"  EW:  [{ew_ci[0]:.6f}, {ew_ci[1]:.6f}]")
    print(f"  KPZ: [{kpz_ci[0]:.6f}, {kpz_ci[1]:.6f}]")
    
    # Check if CIs overlap
    ci_overlap = ew_ci[0] < kpz_ci[1] and kpz_ci[0] < ew_ci[1]
    print(f"  CIs overlap: {ci_overlap}")
    
    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(ew_scores, kpz_scores, equal_var=False)
    print(f"\nWelch's t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Significant at α=0.05: {p_value < 0.05}")
    print(f"  Significant at α=0.01: {p_value < 0.01}")
    print(f"  Significant at α=0.001: {p_value < 0.001}")
    
    # Effect size (Cohen's d)
    d = cohens_d(ew_scores, kpz_scores)
    print(f"\nEffect Size (Cohen's d): {d:.4f}")
    if abs(d) < 0.2:
        effect_interp = "negligible"
    elif abs(d) < 0.5:
        effect_interp = "small"
    elif abs(d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Interpretation: {effect_interp}")
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(ew_scores, kpz_scores, alternative='two-sided')
    print(f"\nMann-Whitney U test (non-parametric):")
    print(f"  U-statistic: {u_stat:.1f}")
    print(f"  p-value: {u_pvalue:.2e}")
    
    # =========================================================================
    # Phase 5: Context - Compare to Discrete Models
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 5: Full Context - All Models")
    print("=" * 50)
    
    # Use EDEN as baseline (highest reconstruction error in Exp 6)
    baseline = scores['EDEN'].mean()
    
    print(f"\nAnomaly Scores (relative to EDEN baseline):")
    print("-" * 50)
    print(f"{'Model':<10} {'Score':>12} {'Std':>10} {'Relative':>10}")
    print("-" * 50)
    
    for name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        s = scores[name]
        rel = s.mean() / baseline
        print(f"{name:<10} {s.mean():>12.6f} {s.std():>10.6f} {rel:>10.2f}x")
    
    # =========================================================================
    # Phase 6: Interpretation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 6: Interpretation")
    print("=" * 50)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ KEY RESULT                                                      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    if p_value < 0.001:
        print("│ ✓ EW and KPZ are STATISTICALLY DISTINGUISHABLE (p < 0.001)     │")
        print("│                                                                 │")
        print("│ The discrete-trained autoencoder RESPECTS universality class   │")
        print("│ boundaries! It can distinguish:                                │")
        print("│   - EW class (β=1/4) from KPZ class (β=1/3)                    │")
        print("│                                                                 │")
        print("│ This supports Conjecture 3.1 (Asymptotic Separation):          │")
        print("│ Different universality classes have disjoint supports          │")
        print("│ even in the discrete-trained representation space.             │")
    elif p_value < 0.05:
        print("│ ⚠ EW and KPZ are WEAKLY DISTINGUISHABLE (p < 0.05)             │")
        print("│                                                                 │")
        print("│ Some separation exists, but evidence is not overwhelming.      │")
    else:
        print("│ ✗ EW and KPZ are NOT STATISTICALLY DISTINGUISHABLE             │")
        print("│                                                                 │")
        print("│ The model only captures discrete/continuum structure,          │")
        print("│ not universality class boundaries.                             │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # =========================================================================
    # Phase 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 50)
    print("Phase 7: Generating Visualizations")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: EW vs KPZ distributions
    ax1 = axes[0]
    ax1.hist(ew_scores, bins=30, alpha=0.7, label=f'EW (μ={ew_scores.mean():.4f})', color='blue', density=True)
    ax1.hist(kpz_scores, bins=30, alpha=0.7, label=f'KPZ (μ={kpz_scores.mean():.4f})', color='red', density=True)
    ax1.axvline(ew_scores.mean(), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(kpz_scores.mean(), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Density')
    ax1.set_title(f'EW vs KPZ Distribution\n(p = {p_value:.2e}, d = {d:.2f})')
    ax1.legend()
    
    # Plot 2: All models box plot
    ax2 = axes[1]
    box_data = [scores['EW'], scores['KPZ'], scores['BD'], scores['EDEN'], scores['RD']]
    bp = ax2.boxplot(box_data, labels=['EW', 'KPZ', 'BD', 'EDEN', 'RD'], patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightgray']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('All Models - Box Plot')
    ax2.axhline(baseline, color='orange', linestyle='--', label='EDEN baseline')
    
    # Plot 3: Hierarchical structure visualization
    ax3 = axes[2]
    means = [scores[name].mean() for name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']]
    stds = [scores[name].std() for name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']]
    x_pos = range(5)
    colors = ['blue', 'red', 'green', 'orange', 'gray']
    ax3.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['EW\n(diff class)', 'KPZ\n(same class)', 'BD\n(training)', 'EDEN\n(training)', 'RD\n(diff class)'])
    ax3.set_ylabel('Reconstruction Error')
    ax3.set_title('Hierarchical Structure')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'figures', 'exp07_ew_kpz_discrimination.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Experiment 7 Results:
─────────────────────
Training: BD + EDEN (discrete KPZ-class models)
Testing: EW (n={N_TEST}), KPZ (n={N_TEST})

Statistical Tests:
  • Welch's t-test p-value: {p_value:.2e}
  • Cohen's d effect size: {d:.4f} ({effect_interp})
  • Mann-Whitney U p-value: {u_pvalue:.2e}

Conclusion:
  {'EW and KPZ ARE distinguishable - model respects universality!' if p_value < 0.05 else 'EW and KPZ are NOT distinguishable - model only sees discrete/continuum'}

Implications for Conjectures:
  • Conjecture 3.1 (Separation): {'SUPPORTED' if p_value < 0.05 else 'NOT SUPPORTED'} at this scale
  • The discrete-trained representation {'captures' if p_value < 0.05 else 'does not capture'} universality class structure
""")


if __name__ == "__main__":
    main()
