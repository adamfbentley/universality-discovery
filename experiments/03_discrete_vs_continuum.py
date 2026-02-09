"""
Experiment 03: Discrete vs Continuum Classification
====================================================
Test whether ALL discrete models separate from continuum PDEs,
or if some discrete models truly become indistinguishable.

Models tested:
- Continuum (training): EW, KPZ
- Discrete (test): BD, Random Deposition (RD), EDEN-like

Hypothesis: If the autoencoder is detecting "discreteness" rather than
universality class, then ALL discrete models should appear anomalous.
If it's detecting universality, then BD should cluster with KPZ.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.autoencoder import SurfaceAutoencoder, prepare_surface_batch
from simulation.physics_simulation import GrowthModelSimulator

# Configuration
CONFIG = {
    'width': 128,
    'time_steps': 500,  # Balanced: enough signal, reasonable compute
    'latent_dim': 32,
    'n_training_samples': 400,  # EW + KPZ
    'n_test_samples_per_class': 60,
    'batch_size': 16,
    'epochs': 30,
    'learning_rate': 1e-3,
    'random_seed': 42,
}


def generate_random_deposition(width, time_steps, seed=None):
    """
    Random Deposition (RD): Particles fall straight down, no lateral interaction.
    This is the SIMPLEST growth model - should be very different from KPZ.
    Exponents: β = 1/2, α = undefined (uncorrelated)
    """
    if seed is not None:
        np.random.seed(seed)
    
    surface = np.zeros(width)
    history = [surface.copy()]
    
    for t in range(time_steps - 1):
        # Random column gets a particle
        col = np.random.randint(0, width)
        surface[col] += 1
        history.append(surface.copy())
    
    return np.array(history)  # (time_steps, width)


def generate_eden_model(width, time_steps, seed=None):
    """
    EDEN Model: Growth from perimeter sites.
    Known to be in KPZ universality class.
    Should cluster WITH KPZ if autoencoder sees universality.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with a flat surface
    surface = np.zeros(width)
    history = [surface.copy()]
    
    for t in range(time_steps - 1):
        # Find all valid growth sites (perimeter)
        perimeter = []
        for i in range(width):
            left = (i - 1) % width
            right = (i + 1) % width
            # Can grow at position i if it's adjacent to or below neighbors
            max_neighbor = max(surface[left], surface[right])
            if surface[i] <= max_neighbor:
                perimeter.append(i)
        
        if not perimeter:
            perimeter = list(range(width))
        
        # Pick random perimeter site and grow
        col = np.random.choice(perimeter)
        surface[col] += 1
        history.append(surface.copy())
    
    return np.array(history)  # (time_steps, width)


def generate_surfaces(model_type, n_samples, width, time_steps, seed_base=0):
    """Generate surfaces for a given model type."""
    surfaces = []
    
    for i in range(n_samples):
        seed = seed_base + i
        
        if model_type == 'random_deposition':
            trajectory = generate_random_deposition(width, time_steps, seed=seed)
        elif model_type == 'eden':
            trajectory = generate_eden_model(width, time_steps, seed=seed)
        else:
            # Use GrowthModelSimulator for EW, KPZ, BD
            sim = GrowthModelSimulator(
                width=width,
                height=time_steps,
                random_state=seed
            )
            trajectory = sim.generate_trajectory(model_type)
        
        # trajectory is (time_steps, width), transpose to (width, time_steps)
        surfaces.append(trajectory.T)
    
    return np.array(surfaces)


def train_autoencoder(model, train_loader, epochs, lr):
    """Train the autoencoder."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        total_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed, _ = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    return avg_loss


def compute_anomaly_scores(model, surfaces):
    """Compute reconstruction error for each surface."""
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        x = prepare_surface_batch(surfaces)
        reconstructed, _ = model(x)
        errors = criterion(reconstructed, x)
        scores = errors.mean(dim=(1, 2, 3)).numpy()
    
    return scores


def main():
    print("=" * 60)
    print("EXPERIMENT 3: DISCRETE VS CONTINUUM")
    print("=" * 60)
    print(f"Config: {CONFIG['width']}x{CONFIG['time_steps']}, "
          f"{CONFIG['epochs']} epochs")
    print()
    
    # Generate TRAINING data (continuum models only)
    print("Generating training data (CONTINUUM only)...")
    n_each = CONFIG['n_training_samples'] // 2
    
    print(f"  Generating {n_each} EW surfaces...")
    ew_train = generate_surfaces('edwards_wilkinson', n_each, 
                                  CONFIG['width'], CONFIG['time_steps'], 
                                  seed_base=0)
    
    print(f"  Generating {n_each} KPZ surfaces...")
    kpz_train = generate_surfaces('kpz_equation', n_each,
                                   CONFIG['width'], CONFIG['time_steps'],
                                   seed_base=10000)
    
    train_surfaces = np.vstack([ew_train, kpz_train])
    np.random.seed(CONFIG['random_seed'])
    np.random.shuffle(train_surfaces)
    print(f"Training surfaces shape: {train_surfaces.shape}")
    
    # Generate TEST data (all model types)
    print("\nGenerating test data (ALL models)...")
    n_test = CONFIG['n_test_samples_per_class']
    
    test_data = {}
    seed_offsets = {
        'edwards_wilkinson': 20000,
        'kpz_equation': 30000,
        'ballistic_deposition': 40000,
        'random_deposition': 50000,
        'eden': 60000
    }
    
    for model_type, seed_base in seed_offsets.items():
        print(f"  Generating {n_test} {model_type} surfaces...")
        test_data[model_type] = generate_surfaces(
            model_type, n_test, CONFIG['width'], CONFIG['time_steps'], 
            seed_base=seed_base
        )
    
    # Create model
    model = SurfaceAutoencoder(
        width=CONFIG['width'],
        time_steps=CONFIG['time_steps'],
        latent_dim=CONFIG['latent_dim']
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_tensor = prepare_surface_batch(train_surfaces)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    
    print("\nTraining autoencoder on CONTINUUM models only...")
    final_loss = train_autoencoder(
        model, train_loader, CONFIG['epochs'], CONFIG['learning_rate']
    )
    
    # Compute anomaly scores for all test sets
    print("\nComputing anomaly scores...")
    results = {}
    for model_type, surfaces in test_data.items():
        scores = compute_anomaly_scores(model, surfaces)
        results[model_type] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    # Calculate baseline (training distribution)
    continuum_scores = np.concatenate([
        results['edwards_wilkinson']['scores'],
        results['kpz_equation']['scores']
    ])
    baseline_mean = continuum_scores.mean()
    baseline_std = continuum_scores.std()
    
    # Display results
    print("\n" + "=" * 60)
    print("ANOMALY SCORES BY MODEL TYPE")
    print("=" * 60)
    
    model_labels = {
        'edwards_wilkinson': ('EW', 'CONTINUUM'),
        'kpz_equation': ('KPZ', 'CONTINUUM'),
        'ballistic_deposition': ('BD', 'DISCRETE'),
        'random_deposition': ('RD', 'DISCRETE'),
        'eden': ('EDEN', 'DISCRETE')
    }
    
    for model_type, (abbrev, category) in model_labels.items():
        r = results[model_type]
        separation = r['mean'] / baseline_mean if baseline_mean > 0 else 0
        status = "✓ NORMAL" if separation < 2 else "✗ ANOMALOUS"
        print(f"{abbrev:6s} [{category:9s}] : {r['mean']:.4f} ± {r['std']:.4f}  "
              f"({separation:.1f}x) {status}")
    
    print("-" * 60)
    print(f"Continuum baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    
    # Visualize with UMAP
    print("\nVisualizing latent space...")
    from umap import UMAP
    
    model.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for model_type, surfaces in test_data.items():
            x = prepare_surface_batch(surfaces)
            _, latent = model(x)
            all_latents.append(latent.numpy())
            all_labels.extend([model_type] * len(surfaces))
    
    all_latents = np.vstack(all_latents)
    
    # UMAP reduction
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(all_latents)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    style_map = {
        'edwards_wilkinson': {'color': '#2ecc71', 'marker': 'o', 'label': 'EW (continuum)', 's': 100},
        'kpz_equation': {'color': '#3498db', 'marker': 's', 'label': 'KPZ (continuum)', 's': 100},
        'ballistic_deposition': {'color': '#e74c3c', 'marker': '^', 'label': 'BD (discrete)', 's': 120},
        'random_deposition': {'color': '#9b59b6', 'marker': 'D', 'label': 'RD (discrete)', 's': 100},
        'eden': {'color': '#f39c12', 'marker': 'v', 'label': 'EDEN (discrete)', 's': 120},
    }
    
    for model_type, style in style_map.items():
        mask = np.array(all_labels) == model_type
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=style['color'], marker=style['marker'],
            s=style['s'], alpha=0.7, label=style['label'],
            edgecolors='black', linewidths=0.5
        )
    
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('Latent Space: Discrete vs Continuum Models\n'
                 f'(t={CONFIG["time_steps"]} steps, trained on EW+KPZ only)',
                 fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/exp03_discrete_vs_continuum.png', dpi=150, bbox_inches='tight')
    print("Saved to figures/exp03_discrete_vs_continuum.png")
    
    # Summary interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    discrete_models = ['ballistic_deposition', 'random_deposition', 'eden']
    discrete_separations = {m: results[m]['mean'] / baseline_mean for m in discrete_models}
    
    if all(s > 5 for s in discrete_separations.values()):
        print("ALL discrete models are strongly anomalous!")
        print("→ Autoencoder detects 'discreteness', not universality class")
    elif discrete_separations['eden'] < 2:
        print("EDEN clusters with continuum models!")
        print("→ Autoencoder may be detecting TRUE universality")
    else:
        print("Mixed results - some discrete models more anomalous than others")
        print("→ Complex interplay between discreteness and universality")
    
    # Which discrete model is LEAST anomalous?
    least_anomalous = min(discrete_models, key=lambda m: results[m]['mean'])
    print(f"\nLeast anomalous discrete model: {least_anomalous}")
    print(f"  Separation: {discrete_separations[least_anomalous]:.1f}x")
    
    most_anomalous = max(discrete_models, key=lambda m: results[m]['mean'])
    print(f"Most anomalous discrete model: {most_anomalous}")
    print(f"  Separation: {discrete_separations[most_anomalous]:.1f}x")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
