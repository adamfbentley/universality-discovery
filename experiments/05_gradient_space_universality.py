"""
Experiment 5: Gradient-Space Coarse-Grained Autoencoder

HYPOTHESIS (from Grok's analysis):
- Local gradients sample RG-relevant operators (λ(∇h)² in KPZ) directly
- Coarse-graining removes discrete lattice artifacts
- In gradient space, EDEN should cluster with KPZ (same universality class)

APPROACH:
1. Compute gradient field: g(x,t) = ∇h(x,t) 
2. Apply Gaussian blur to remove discrete artifacts
3. Train autoencoder on gradient fields from EW + KPZ (continuum)
4. Test if EDEN (KPZ class) has LOWER anomaly than BD (non-KPZ)

SUCCESS CRITERION:
- EDEN anomaly score << BD anomaly score
- EDEN approaches KPZ-like scores as blur increases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from tqdm import tqdm

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
CONFIG = {
    'width': 128,
    'time_steps': 500,
    'n_train_per_class': 200,  # EW + KPZ = 400 total
    'n_test_per_class': 40,
    'epochs': 25,
    'blur_sigmas': [0, 1, 2, 4, 8],  # Test multiple blur levels
    'latent_dim': 16,
    'seed': 42
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])


def compute_gradient_field(surface):
    """
    Convert height field h(x,t) to gradient field ∇h(x,t).
    
    This captures the RG-relevant operator λ(∇h)² directly.
    Removes absolute height information (where discreteness lives).
    """
    # Spatial gradient along x-axis (periodic boundary)
    grad_x = np.roll(surface, -1, axis=0) - surface
    return grad_x


def apply_coarse_graining(field, sigma):
    """
    Apply Gaussian blur to remove discrete lattice artifacts.
    
    At scale σ, features smaller than σ are smoothed out.
    This should eliminate the discrete "staircase" structure in BD/EDEN.
    """
    if sigma == 0:
        return field
    # Apply 2D Gaussian filter (spatial + temporal smoothing)
    return gaussian_filter(field, sigma=sigma, mode='wrap')


def preprocess_surface(surface, blur_sigma):
    """Full preprocessing pipeline: gradient + coarse-graining."""
    gradient = compute_gradient_field(surface)
    smoothed = apply_coarse_graining(gradient, blur_sigma)
    return smoothed


def generate_training_data(blur_sigma):
    """Generate preprocessed EW + KPZ surfaces for training."""
    print(f"\nGenerating training data (blur σ={blur_sigma})...")
    
    surfaces = []
    
    # EW surfaces
    for i in range(CONFIG['n_train_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        diffusion = np.random.uniform(0.5, 2.0)
        h = sim.generate_trajectory('edwards_wilkinson', diffusion=diffusion)
        g = preprocess_surface(h, blur_sigma)
        surfaces.append(g)
    
    # KPZ surfaces  
    for i in range(CONFIG['n_train_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        diffusion = np.random.uniform(0.5, 2.0)
        nonlinearity = np.random.uniform(0.5, 2.0)
        h = sim.generate_trajectory('kpz_equation', diffusion=diffusion, nonlinearity=nonlinearity)
        g = preprocess_surface(h, blur_sigma)
        surfaces.append(g)
    
    return np.array(surfaces)


def generate_test_data(blur_sigma):
    """Generate preprocessed test surfaces for all model classes."""
    print(f"\nGenerating test data (blur σ={blur_sigma})...")
    
    test_data = {}
    
    # EW (continuum, EW class)
    surfaces = []
    for i in range(CONFIG['n_test_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        h = sim.generate_trajectory('edwards_wilkinson', diffusion=1.0)
        surfaces.append(preprocess_surface(h, blur_sigma))
    test_data['EW'] = np.array(surfaces)
    
    # KPZ (continuum, KPZ class)
    surfaces = []
    for i in range(CONFIG['n_test_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        h = sim.generate_trajectory('kpz_equation', diffusion=1.0, nonlinearity=1.0)
        surfaces.append(preprocess_surface(h, blur_sigma))
    test_data['KPZ'] = np.array(surfaces)
    
    # Ballistic Deposition (discrete, KPZ class theoretically)
    surfaces = []
    for i in range(CONFIG['n_test_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        h = sim.generate_trajectory('ballistic_deposition')
        surfaces.append(preprocess_surface(h, blur_sigma))
    test_data['BD'] = np.array(surfaces)
    
    # EDEN (discrete, KPZ class theoretically)
    surfaces = []
    for i in range(CONFIG['n_test_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        h = sim.generate_trajectory('eden')
        surfaces.append(preprocess_surface(h, blur_sigma))
    test_data['EDEN'] = np.array(surfaces)
    
    # Random Deposition (discrete, NO lateral correlations - not KPZ)
    surfaces = []
    for i in range(CONFIG['n_test_per_class']):
        sim = GrowthModelSimulator(CONFIG['width'], CONFIG['time_steps'])
        h = sim.generate_trajectory('random_deposition')
        surfaces.append(preprocess_surface(h, blur_sigma))
    test_data['RD'] = np.array(surfaces)
    
    return test_data


def compute_anomaly_scores(model, data):
    """Compute reconstruction error as anomaly score."""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(data).unsqueeze(1)  # Add channel dim
        recon, z = model(x)  # Forward returns (reconstruction, latent)
        mse = ((x - recon) ** 2).mean(dim=(1, 2, 3))  # Per-sample MSE
    return mse.numpy()


def run_experiment_for_blur(blur_sigma):
    """Run full experiment for a given blur level."""
    print(f"\n{'='*60}")
    print(f"BLUR SIGMA = {blur_sigma}")
    print('='*60)
    
    # Generate data
    train_data = generate_training_data(blur_sigma)
    test_data = generate_test_data(blur_sigma)
    
    print(f"  Training shape: {train_data.shape}")
    for name, data in test_data.items():
        print(f"  {name}: {data.shape}")
    
    # Create and train autoencoder
    # Note: gradient field has shape (time_steps, width) - need to transpose for model
    print("\nTraining autoencoder...")
    
    # Transpose to match model expectations (width x time_steps)
    train_data_t = np.transpose(train_data, (0, 2, 1))  # (N, width, time_steps)
    
    model = SurfaceAutoencoder(
        width=CONFIG['width'],
        time_steps=CONFIG['time_steps'],
        latent_dim=CONFIG['latent_dim']
    )
    
    train_tensor = torch.FloatTensor(train_data_t).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in tqdm(range(CONFIG['epochs']), desc="Training"):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, z = model(x)  # Forward returns (reconstruction, latent)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Compute anomaly scores
    print("\nComputing anomaly scores...")
    results = {}
    for name, data in test_data.items():
        # Transpose to match model (N, width, time_steps)
        data_t = np.transpose(data, (0, 2, 1))
        scores = compute_anomaly_scores(model, data_t)
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Compute baseline (average of EW and KPZ)
    baseline = (results['EW']['mean'] + results['KPZ']['mean']) / 2
    
    print(f"\n  Results for blur σ={blur_sigma}:")
    print(f"  {'Model':<10} {'Score':>12} {'Separation':>12}")
    print(f"  {'-'*36}")
    for name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        r = results[name]
        sep = r['mean'] / baseline if baseline > 0 else 0
        print(f"  {name:<10} {r['mean']:>8.4f} ± {r['std']:>5.4f} {sep:>8.1f}x")
    
    return results, baseline


def main():
    print("="*60)
    print("EXPERIMENT 5: Gradient-Space Coarse-Grained Autoencoder")
    print("="*60)
    print("\nHypothesis: In gradient space with coarse-graining,")
    print("EDEN should approach KPZ (same universality class)")
    print("while BD remains distinct (different local dynamics).")
    
    all_results = {}
    
    for sigma in CONFIG['blur_sigmas']:
        results, baseline = run_experiment_for_blur(sigma)
        all_results[sigma] = {
            'results': results,
            'baseline': baseline
        }
    
    # Summary analysis
    print("\n" + "="*60)
    print("SUMMARY: How blur affects universality detection")
    print("="*60)
    
    print(f"\n{'σ':>4} | {'EW':>8} | {'KPZ':>8} | {'BD':>8} | {'EDEN':>8} | {'RD':>8} | EDEN/BD")
    print("-"*70)
    
    eden_bd_ratios = []
    for sigma in CONFIG['blur_sigmas']:
        r = all_results[sigma]['results']
        baseline = all_results[sigma]['baseline']
        
        ew_sep = r['EW']['mean'] / baseline
        kpz_sep = r['KPZ']['mean'] / baseline
        bd_sep = r['BD']['mean'] / baseline
        eden_sep = r['EDEN']['mean'] / baseline
        rd_sep = r['RD']['mean'] / baseline
        
        eden_bd_ratio = eden_sep / bd_sep if bd_sep > 0 else 0
        eden_bd_ratios.append(eden_bd_ratio)
        
        print(f"{sigma:>4} | {ew_sep:>7.2f}x | {kpz_sep:>7.2f}x | {bd_sep:>7.1f}x | {eden_sep:>7.1f}x | {rd_sep:>7.1f}x | {eden_bd_ratio:.3f}")
    
    # Key interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    # Check if EDEN/BD ratio decreases with blur (EDEN approaching KPZ)
    if eden_bd_ratios[-1] < eden_bd_ratios[0]:
        print("✓ EDEN/BD ratio DECREASES with blur")
        print("  → Coarse-graining reveals EDEN is closer to KPZ than BD")
        print("  → Supports universality hypothesis!")
    else:
        print("✗ EDEN/BD ratio does NOT decrease with blur")
        print("  → Discreteness signature persists even in gradient space")
        print("  → May need different approach")
    
    # Check if EDEN approaches baseline (EW/KPZ average)
    final_eden_sep = all_results[CONFIG['blur_sigmas'][-1]]['results']['EDEN']['mean'] / all_results[CONFIG['blur_sigmas'][-1]]['baseline']
    if final_eden_sep < 5:
        print(f"\n✓ At high blur (σ={CONFIG['blur_sigmas'][-1]}), EDEN separation = {final_eden_sep:.1f}x")
        print("  → EDEN is approaching continuum-like behavior!")
    else:
        print(f"\n✗ Even at high blur, EDEN separation = {final_eden_sep:.1f}x")
        print("  → Significant anomaly persists")
    
    # Create visualization
    create_visualization(all_results)
    
    print("\n" + "="*60)
    print("Experiment 5 complete!")
    print("="*60)


def create_visualization(all_results):
    """Create figure showing how blur affects universality detection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sigmas = CONFIG['blur_sigmas']
    
    # Panel 1: Separation vs blur for each model
    ax1 = axes[0]
    for name, color in [('EW', 'blue'), ('KPZ', 'green'), ('BD', 'red'), ('EDEN', 'orange'), ('RD', 'purple')]:
        seps = []
        for sigma in sigmas:
            r = all_results[sigma]['results'][name]
            baseline = all_results[sigma]['baseline']
            seps.append(r['mean'] / baseline)
        ax1.plot(sigmas, seps, 'o-', color=color, label=name, linewidth=2, markersize=8)
    
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xlabel('Blur σ', fontsize=12)
    ax1.set_ylabel('Separation (×baseline)', fontsize=12)
    ax1.set_title('Anomaly Separation vs Coarse-Graining', fontsize=14)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: EDEN/BD ratio vs blur
    ax2 = axes[1]
    eden_bd_ratios = []
    for sigma in sigmas:
        r = all_results[sigma]['results']
        baseline = all_results[sigma]['baseline']
        eden_sep = r['EDEN']['mean'] / baseline
        bd_sep = r['BD']['mean'] / baseline
        eden_bd_ratios.append(eden_sep / bd_sep)
    
    ax2.plot(sigmas, eden_bd_ratios, 'o-', color='darkgreen', linewidth=2, markersize=10)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Blur σ', fontsize=12)
    ax2.set_ylabel('EDEN/BD Ratio', fontsize=12)
    ax2.set_title('Does EDEN Approach KPZ Relative to BD?', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation
    if eden_bd_ratios[-1] < eden_bd_ratios[0]:
        ax2.annotate('↓ EDEN closer\nto KPZ', xy=(sigmas[-1], eden_bd_ratios[-1]),
                    xytext=(sigmas[-1]-2, eden_bd_ratios[-1]*1.5),
                    fontsize=10, color='green',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Panel 3: Score distributions at max blur
    ax3 = axes[2]
    max_sigma = sigmas[-1]
    r = all_results[max_sigma]['results']
    
    positions = [0, 1, 2, 3, 4]
    names = ['EW', 'KPZ', 'BD', 'EDEN', 'RD']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (name, color) in enumerate(zip(names, colors)):
        scores = r[name]['scores']
        parts = ax3.violinplot([scores], positions=[i], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('Anomaly Score', fontsize=12)
    ax3.set_title(f'Score Distributions (σ={max_sigma})', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'figures', 'exp05_gradient_space_universality.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
