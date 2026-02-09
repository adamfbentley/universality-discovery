"""
Experiment 1: Representation Learning
======================================

Goal: Can autoencoders rediscover known universality class structure without supervision?

Method:
1. Train autoencoder on unlabeled EW + KPZ surfaces
2. Encode EW, KPZ, MBE, BD surfaces
3. Visualize latent space with UMAP
4. Check if known classes form clusters

Success criterion: EW/KPZ overlap (training distribution); BD/MBE clearly separated
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.autoencoder import SurfaceAutoencoder, SurfaceVAE, prepare_surface_batch
from simulation.physics_simulation import GrowthModelSimulator


# Configuration
CONFIG = {
    'width': 128,
    'time_steps': 200,
    'latent_dim': 32,
    'n_training_samples': 1000,  # 500 EW + 500 KPZ
    'n_test_samples_per_class': 100,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_seed': 42
}


def generate_surfaces(model_type: str, n_samples: int, config: dict, seed_offset: int = 0):
    """
    Generate surfaces using the physics simulator.
    
    Args:
        model_type: 'edwards_wilkinson', 'kpz_equation', 'ballistic_deposition'
        n_samples: Number of surfaces to generate
        config: Configuration dict
        seed_offset: Offset for random seed (for different train/test sets)
    
    Returns:
        surfaces: (n_samples, width, time_steps) array
    """
    surfaces = []
    
    for i in range(n_samples):
        sim = GrowthModelSimulator(
            width=config['width'],
            height=config['time_steps'],
            random_state=config['random_seed'] + seed_offset + i
        )
        
        # Generate trajectory - comes out as (time_steps, width)
        trajectory = sim.generate_trajectory(model_type)
        # Transpose to (width, time_steps) for the autoencoder
        surfaces.append(trajectory.T)
    
    return np.array(surfaces)


def generate_training_data(config: dict):
    """
    Generate unlabeled EW + KPZ training surfaces.
    """
    print("Generating training data...")
    n_each = config['n_training_samples'] // 2
    
    print(f"  Generating {n_each} Edwards-Wilkinson surfaces...")
    ew_surfaces = generate_surfaces('edwards_wilkinson', n_each, config, seed_offset=0)
    
    print(f"  Generating {n_each} KPZ surfaces...")
    kpz_surfaces = generate_surfaces('kpz_equation', n_each, config, seed_offset=10000)
    
    # Combine without labels (unsupervised)
    all_surfaces = np.concatenate([ew_surfaces, kpz_surfaces], axis=0)
    
    # Shuffle
    np.random.seed(config['random_seed'])
    indices = np.random.permutation(len(all_surfaces))
    
    return all_surfaces[indices]


def generate_test_data(config: dict):
    """
    Generate labeled test surfaces from multiple universality classes.
    """
    print("Generating test data...")
    n_each = config['n_test_samples_per_class']
    
    surfaces = []
    labels = []
    
    class_configs = [
        ('edwards_wilkinson', 'EW'),
        ('kpz_equation', 'KPZ'),
        ('ballistic_deposition', 'BD'),
    ]
    
    # Try to add MBE if available
    try:
        test_sim = GrowthModelSimulator(width=64, height=50, random_state=0)
        test_sim.generate_trajectory('mbe')
        class_configs.append(('mbe', 'MBE'))
    except:
        print("  (MBE model not available, skipping)")
    
    class_names = [c[1] for c in class_configs]
    
    for class_idx, (model_type, name) in enumerate(class_configs):
        print(f"  Generating {n_each} {name} surfaces...")
        class_surfaces = generate_surfaces(
            model_type, n_each, config, 
            seed_offset=20000 + class_idx * 1000
        )
        surfaces.append(class_surfaces)
        labels.extend([class_idx] * n_each)
    
    return np.concatenate(surfaces, axis=0), np.array(labels), class_names


def train_autoencoder(model, train_loader, config: dict):
    """Train autoencoder and return loss history."""
    device = config['device']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    loss_history = []
    
    pbar = tqdm(range(config['epochs']), desc='Training')
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            x_recon, z = model(x)
            loss = model.reconstruction_loss(x, x_recon)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    return loss_history


def visualize_latent_space(model, surfaces, labels, class_names, config: dict, save_path: str):
    """Project surfaces to latent space and visualize with UMAP."""
    try:
        import umap
    except ImportError:
        print("Install umap-learn for visualization: pip install umap-learn")
        return None
    
    device = config['device']
    model = model.to(device)
    model.eval()
    
    # Encode all surfaces
    print("Encoding surfaces to latent space...")
    surface_tensor = prepare_surface_batch(surfaces).to(device)
    with torch.no_grad():
        latent = model.encode(surface_tensor).cpu().numpy()
    
    print(f"Latent space shape: {latent.shape}")
    
    # UMAP reduction
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent)
    
    # Plot with distinct markers, colors, and outlines for each class
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define distinct visual styles for each class
    class_styles = {
        'EW': {'color': '#2ecc71', 'marker': 'o', 'edgecolor': '#1a5f3a', 'label': 'EW (Edwards-Wilkinson)'},
        'KPZ': {'color': '#3498db', 'marker': 's', 'edgecolor': '#1a4a6e', 'label': 'KPZ (Kardar-Parisi-Zhang)'},
        'BD': {'color': '#e74c3c', 'marker': '^', 'edgecolor': '#7a1f14', 'label': 'BD (Ballistic Deposition) - UNKNOWN'},
        'MBE': {'color': '#9b59b6', 'marker': 'D', 'edgecolor': '#4a2a5a', 'label': 'MBE (Molecular Beam Epitaxy)'},
    }
    
    # Fallback styles for additional classes
    fallback_colors = ['#f39c12', '#1abc9c', '#e91e63', '#795548']
    fallback_markers = ['v', 'p', 'h', '*']
    
    for i, name in enumerate(class_names):
        mask = labels == i
        
        if name in class_styles:
            style = class_styles[name]
        else:
            style = {
                'color': fallback_colors[i % len(fallback_colors)],
                'marker': fallback_markers[i % len(fallback_markers)],
                'edgecolor': 'black',
                'label': name
            }
        
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=style['color'], 
            marker=style['marker'],
            label=style['label'], 
            s=120,  # Larger points
            alpha=0.85,
            edgecolors=style['edgecolor'], 
            linewidth=2.0  # Thick outlines
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.set_title('Latent Space Representation of Surface Growth Models\n(Autoencoder trained on EW + KPZ only)', fontsize=16, fontweight='bold')
    
    # Enhanced legend
    legend = ax.legend(fontsize=12, loc='best', framealpha=0.9, edgecolor='gray')
    legend.get_frame().set_linewidth(1.5)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved latent space visualization to {save_path}")
    plt.show()
    
    return embedding


def analyze_anomaly_scores(model, surfaces, labels, class_names, config: dict):
    """Compute and report anomaly scores by class."""
    device = config['device']
    model = model.to(device)
    model.eval()
    
    surface_tensor = prepare_surface_batch(surfaces).to(device)
    anomaly_scores = model.anomaly_score(surface_tensor).cpu().numpy()
    
    print("\n" + "=" * 50)
    print("ANOMALY SCORES BY CLASS")
    print("=" * 50)
    print("(Higher score = more different from training distribution)")
    print()
    
    results = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        scores = anomaly_scores[mask]
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        # Mark training vs test classes
        if name in ['EW', 'KPZ']:
            marker = '[TRAINING]'
        else:
            marker = '[UNKNOWN]'
        
        print(f"{name:4s} {marker:12s}: {scores.mean():.4f} ± {scores.std():.4f}  (range: {scores.min():.4f} - {scores.max():.4f})")
    
    # Success check
    print("\n" + "-" * 50)
    training_mean = (results['EW']['mean'] + results['KPZ']['mean']) / 2
    unknown_classes = [n for n in class_names if n not in ['EW', 'KPZ']]
    
    if unknown_classes:
        unknown_mean = np.mean([results[n]['mean'] for n in unknown_classes])
        separation = unknown_mean / training_mean if training_mean > 0 else float('inf')
        print(f"Training class mean score: {training_mean:.4f}")
        print(f"Unknown class mean score:  {unknown_mean:.4f}")
        print(f"Separation ratio:          {separation:.2f}x")
        
        if separation > 2.0:
            print("\n✅ SUCCESS: Unknown classes are clearly more anomalous!")
        elif separation > 1.5:
            print("\n⚠️  PARTIAL: Some separation, but could be better")
        else:
            print("\n❌ NEEDS WORK: Not enough separation between known and unknown")
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Representation Learning")
    print("=" * 60)
    print(f"\nDevice: {CONFIG['device']}")
    print(f"Latent dimension: {CONFIG['latent_dim']}")
    print(f"Training samples: {CONFIG['n_training_samples']} (EW + KPZ)")
    print(f"Test samples: {CONFIG['n_test_samples_per_class']} per class")
    print()
    
    # Create output directories
    Path('figures').mkdir(exist_ok=True)
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Generate data
    train_surfaces = generate_training_data(CONFIG)
    print(f"Training surfaces shape: {train_surfaces.shape}")
    
    test_surfaces, test_labels, class_names = generate_test_data(CONFIG)
    print(f"Test surfaces shape: {test_surfaces.shape}")
    print(f"Classes: {class_names}")
    
    # Prepare data loaders
    train_tensor = prepare_surface_batch(train_surfaces)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize model
    print("\nInitializing autoencoder...")
    model = SurfaceAutoencoder(
        width=CONFIG['width'],
        time_steps=CONFIG['time_steps'],
        latent_dim=CONFIG['latent_dim']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\nTraining autoencoder...")
    loss_history = train_autoencoder(model, train_loader, CONFIG)
    
    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Autoencoder Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/exp01_training_loss.png', dpi=150)
    plt.close()
    print("Saved training curve to figures/exp01_training_loss.png")
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/autoencoder_exp01.pt')
    print("Saved model to checkpoints/autoencoder_exp01.pt")
    
    # Visualize latent space
    print("\nVisualizing latent space...")
    visualize_latent_space(
        model, test_surfaces, test_labels, class_names, CONFIG,
        save_path='figures/exp01_latent_space.png'
    )
    
    # Analyze anomaly scores
    analyze_anomaly_scores(model, test_surfaces, test_labels, class_names, CONFIG)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
