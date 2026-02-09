"""
Experiment 6: Train on Discrete Models
======================================

Reverse the paradigm: Train autoencoder on BD + EDEN (discrete KPZ-class models).
Test whether continuum KPZ appears as anomalous.

Key Questions:
1. Does KPZ appear "normal" when the model learns discrete dynamics?
2. Is the asymmetry fundamental, or an artifact of training choice?
3. Can we find a representation where universality class matters more than discreteness?
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

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
WIDTH = 128
TIME_STEPS = 500
N_TRAIN_PER_CLASS = 200  # BD + EDEN = 400 total
N_TEST_PER_CLASS = 40
EPOCHS = 25
LATENT_DIM = 32


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


def main():
    print("=" * 60)
    print("Experiment 6: Train on Discrete Models")
    print("=" * 60)
    print("\nParadigm reversal: Learn 'normal' = BD + EDEN (discrete)")
    print("Test: Does continuum KPZ appear anomalous?")
    print()
    
    # Use optimal blur from Exp 5
    SIGMA = 2
    
    print(f"Configuration:")
    print(f"  Grid: {WIDTH} x {TIME_STEPS}")
    print(f"  Coarse-graining σ = {SIGMA}")
    print(f"  Training: {N_TRAIN_PER_CLASS} BD + {N_TRAIN_PER_CLASS} EDEN")
    print(f"  Testing: {N_TEST_PER_CLASS} per class")
    print()
    
    # Generate training data (discrete models only)
    print("Generating training data (BD + EDEN)...")
    train_data = generate_dataset(
        ['ballistic_deposition', 'eden'], 
        N_TRAIN_PER_CLASS, 
        sigma=SIGMA,
        desc="Training data"
    )
    print(f"  Training shape: {train_data.shape}")
    
    # Generate test data for all models
    print("\nGenerating test data (all models)...")
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
            N_TEST_PER_CLASS, 
            sigma=SIGMA,
            desc=f"  {model_name}"
        )
        print(f"  {model_name}: {test_data[model_name].shape}")
    
    # Train autoencoder on discrete models
    print("\nTraining autoencoder on BD + EDEN...")
    model, mean, std = train_autoencoder(train_data, epochs=EPOCHS)
    
    # Compute baseline from training-like data (BD + EDEN test)
    print("\nComputing anomaly scores...")
    bd_scores = compute_anomaly_scores(model, test_data['BD'], mean, std)
    eden_scores = compute_anomaly_scores(model, test_data['EDEN'], mean, std)
    baseline = np.concatenate([bd_scores, eden_scores]).mean()
    
    # Results table
    results = {}
    print("\n" + "=" * 60)
    print("RESULTS: Training on Discrete (BD + EDEN)")
    print("=" * 60)
    print(f"\n{'Model':<12} {'Score':>12} {'Separation':>12}")
    print("-" * 40)
    
    for model_name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        scores = compute_anomaly_scores(model, test_data[model_name], mean, std)
        results[model_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'separation': scores.mean() / baseline if baseline > 0 else 0
        }
        print(f"{model_name:<12} {results[model_name]['mean']:.4f} ± {results[model_name]['std']:.4f}  {results[model_name]['separation']:>8.1f}x")
    
    # Key metrics
    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    
    kpz_sep = results['KPZ']['separation']
    ew_sep = results['EW']['separation']
    rd_sep = results['RD']['separation']
    bd_sep = results['BD']['separation']
    eden_sep = results['EDEN']['separation']
    
    print(f"\nContinuum models (should be anomalous if discreteness is key):")
    print(f"  EW separation:  {ew_sep:.2f}x")
    print(f"  KPZ separation: {kpz_sep:.2f}x")
    
    print(f"\nDiscrete models (should be normal, trained on these):")
    print(f"  BD separation:   {bd_sep:.2f}x")
    print(f"  EDEN separation: {eden_sep:.2f}x")
    
    print(f"\nRandom deposition (different class):")
    print(f"  RD separation:  {rd_sep:.2f}x")
    
    print(f"\nCritical ratio (KPZ/discrete_baseline):")
    print(f"  KPZ / BD:   {kpz_sep/bd_sep:.2f}x")
    print(f"  KPZ / EDEN: {kpz_sep/eden_sep:.2f}x")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if kpz_sep > 5 and ew_sep > 5:
        print("✗ Continuum models ARE anomalous to discrete-trained model")
        print("  → Confirms: Discreteness is dominant feature, not universality")
        print("  → Asymmetry is NOT an artifact of training choice")
    elif kpz_sep < 2:
        print("✓ KPZ is NOT anomalous to discrete-trained model!")
        print("  → Suggests: Universality CAN be detected with right training")
        print("  → Previous experiments were biased by continuum training")
    else:
        print("~ Mixed results: moderate anomaly detected")
        print(f"  KPZ separation = {kpz_sep:.1f}x (neither clearly normal nor anomalous)")
    
    if rd_sep > max(bd_sep, eden_sep) * 2:
        print(f"\n✓ RD is more anomalous ({rd_sep:.1f}x) than BD/EDEN")
        print("  → Model distinguishes universality classes within discrete models")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All separations
    ax = axes[0, 0]
    models = list(results.keys())
    seps = [results[m]['separation'] for m in models]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    bars = ax.bar(models, seps, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Baseline')
    ax.set_ylabel('Separation from Baseline')
    ax.set_title('Anomaly Detection (Trained on BD + EDEN)')
    ax.set_yscale('log')
    ax.legend()
    
    # Add value labels
    for bar, sep in zip(bars, seps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{sep:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Score distributions
    ax = axes[0, 1]
    positions = range(len(results))
    for i, (name, color) in enumerate(zip(results.keys(), colors)):
        scores = compute_anomaly_scores(model, test_data[name], mean, std)
        parts = ax.violinplot([scores], positions=[i], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(results.keys())
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Score Distributions')
    ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Comparison with Exp 1 (trained on EW+KPZ)
    ax = axes[1, 0]
    
    # Data from Exp 1 (approximate values from earlier results)
    exp1_seps = {'EW': 1.8, 'KPZ': 0.2, 'BD': 1522, 'EDEN': 22042, 'RD': 33358}
    exp6_seps = {m: results[m]['separation'] for m in models}
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [exp1_seps.get(m, 0) for m in models], width, 
                   label='Exp 5 σ=0 (EW+KPZ train)', alpha=0.7)
    bars2 = ax.bar(x + width/2, [exp6_seps[m] for m in models], width,
                   label='Exp 6 (BD+EDEN train)', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Separation from Baseline')
    ax.set_title('Comparison: Training on Continuum vs Discrete')
    ax.set_yscale('log')
    ax.legend()
    
    # Plot 4: Summary diagram
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    EXPERIMENT 6 SUMMARY
    ════════════════════════════════════════════════════════════
    
    Training: BD + EDEN (discrete KPZ-class models)
    
    Results:
    ────────────────────────────────────────────────────────────
    • EW (continuum):     {ew_sep:.1f}x separation
    • KPZ (continuum):    {kpz_sep:.1f}x separation  
    • BD (trained):       {bd_sep:.1f}x separation
    • EDEN (trained):     {eden_sep:.1f}x separation
    • RD (different):     {rd_sep:.1f}x separation
    
    Key Insight:
    ────────────────────────────────────────────────────────────
    """
    
    if kpz_sep > 5:
        summary_text += """
    The asymmetry is REAL:
    • Discrete → Continuum is anomalous
    • Continuum → Discrete is also anomalous
    • Discreteness dominates universality class
    """
    else:
        summary_text += """
    Universality DOES emerge:
    • KPZ (continuum) recognized by discrete-trained model
    • Shared class structure visible across discreteness boundary
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 
                            'exp06_discrete_training.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Experiment 6 complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
