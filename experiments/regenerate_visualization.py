"""
Regenerate visualization using saved model (no retraining needed).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import from main experiment
from models.autoencoder import SurfaceAutoencoder

CONFIG = {
    'width': 128,
    'time_steps': 200,
    'latent_dim': 32,
    'n_test_samples_per_class': 100,
    'device': 'cpu'
}

def prepare_surface_batch(surfaces):
    """Prepare surfaces for neural network input."""
    if surfaces.ndim == 3:
        surfaces = surfaces[:, np.newaxis, :, :]
    return torch.FloatTensor(surfaces)

def generate_surfaces(model_type: str, n_samples: int, config: dict, seed_offset: int = 0):
    """Generate surface growth trajectories."""
    from simulation.physics_simulation import GrowthModelSimulator
    
    surfaces = []
    for i in range(n_samples):
        sim = GrowthModelSimulator(
            width=config['width'],
            height=config['time_steps'],
            random_state=seed_offset + i
        )
        trajectory = sim.generate_trajectory(model_type)
        surfaces.append(trajectory)
    
    return np.array(surfaces)

def generate_test_data(config: dict):
    """Generate labeled test surfaces from multiple universality classes."""
    print("Generating test data...")
    n_each = config['n_test_samples_per_class']
    
    surfaces = []
    labels = []
    
    class_configs = [
        ('edwards_wilkinson', 'EW'),
        ('kpz_equation', 'KPZ'),
        ('ballistic_deposition', 'BD'),
    ]
    
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


def visualize_latent_space(model, surfaces, labels, class_names, config: dict, save_path: str):
    """Project surfaces to latent space and visualize with UMAP."""
    import umap
    
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
    
    for i, name in enumerate(class_names):
        mask = labels == i
        
        if name in class_styles:
            style = class_styles[name]
        else:
            style = {
                'color': '#f39c12',
                'marker': 'v',
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
    plt.close()
    
    return embedding


def main():
    print("=" * 60)
    print("REGENERATING VISUALIZATION WITH IMPROVED STYLING")
    print("=" * 60)
    
    # Load saved model
    model_path = 'checkpoints/autoencoder_exp01.pt'
    if not os.path.exists(model_path):
        model_path = '../checkpoints/autoencoder_exp01.pt'
    if not os.path.exists(model_path):
        model_path = 'C:/Users/adamf/Desktop/pp/checkpoints/autoencoder_exp01.pt'
    
    print(f"Loading model from {model_path}...")
    model = SurfaceAutoencoder(
        width=CONFIG['width'],
        time_steps=CONFIG['time_steps'],
        latent_dim=CONFIG['latent_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded successfully!")
    
    # Generate test data (same seeds as original)
    test_surfaces, test_labels, class_names = generate_test_data(CONFIG)
    print(f"Test surfaces shape: {test_surfaces.shape}")
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate improved visualization
    visualize_latent_space(
        model, test_surfaces, test_labels, class_names,
        CONFIG, 'figures/exp01_latent_space_improved.png'
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
