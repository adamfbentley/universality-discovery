"""
Experiment 2: Test with 1000 time steps to see if:
- BD moves closer to KPZ (crossover toward universality)
- EW separates from KPZ (different scaling becomes apparent)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator

# Key change: 1000 time steps instead of 200
CONFIG = {
    'width': 128,
    'time_steps': 1000,  # 5x longer!
    'latent_dim': 32,
    'n_training_samples': 500,  # Reduced for faster training
    'n_test_samples_per_class': 80,
    'batch_size': 16,  # Smaller batch for larger surfaces
    'epochs': 30,  # Fewer epochs
    'learning_rate': 1e-3,
    'device': 'cpu'
}


def prepare_surface_batch(surfaces):
    # Input: (batch, time_steps, width) 
    # Model expects: (batch, 1, width, time_steps)
    # So we need to transpose the last two dimensions
    if surfaces.ndim == 3:
        # Transpose time_steps and width: (batch, time, width) -> (batch, width, time)
        surfaces = surfaces.transpose(0, 2, 1)  # Now (batch, width, time_steps)
        surfaces = surfaces[:, np.newaxis, :, :]  # Add channel: (batch, 1, width, time_steps)
    return torch.FloatTensor(surfaces)


def generate_surfaces(model_type: str, n_samples: int, config: dict, seed_offset: int = 0):
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


def generate_training_data(config: dict):
    print("Generating training data (1000 time steps)...")
    n_each = config['n_training_samples'] // 2
    
    print(f"  Generating {n_each} EW surfaces...")
    ew_surfaces = generate_surfaces('edwards_wilkinson', n_each, config, seed_offset=0)
    
    print(f"  Generating {n_each} KPZ surfaces...")
    kpz_surfaces = generate_surfaces('kpz_equation', n_each, config, seed_offset=10000)
    
    return np.concatenate([ew_surfaces, kpz_surfaces], axis=0)


def generate_test_data(config: dict):
    print("Generating test data...")
    n_each = config['n_test_samples_per_class']
    
    surfaces = []
    labels = []
    class_names = ['EW', 'KPZ', 'BD']
    
    for class_idx, (model_type, name) in enumerate([
        ('edwards_wilkinson', 'EW'),
        ('kpz_equation', 'KPZ'),
        ('ballistic_deposition', 'BD'),
    ]):
        print(f"  Generating {n_each} {name} surfaces...")
        class_surfaces = generate_surfaces(model_type, n_each, config, seed_offset=20000 + class_idx * 1000)
        surfaces.append(class_surfaces)
        labels.extend([class_idx] * n_each)
    
    return np.concatenate(surfaces, axis=0), np.array(labels), class_names


def train_autoencoder(model, train_loader, config: dict):
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
    import umap
    
    device = config['device']
    model = model.to(device)
    model.eval()
    
    print("Encoding surfaces to latent space...")
    surface_tensor = prepare_surface_batch(surfaces).to(device)
    with torch.no_grad():
        latent = model.encode(surface_tensor).cpu().numpy()
    
    print(f"Latent space shape: {latent.shape}")
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    class_styles = {
        'EW': {'color': '#2ecc71', 'marker': 'o', 'edgecolor': '#1a5f3a', 'label': 'EW (Edwards-Wilkinson)'},
        'KPZ': {'color': '#3498db', 'marker': 's', 'edgecolor': '#1a4a6e', 'label': 'KPZ (Kardar-Parisi-Zhang)'},
        'BD': {'color': '#e74c3c', 'marker': '^', 'edgecolor': '#7a1f14', 'label': 'BD (Ballistic Deposition)'},
    }
    
    for i, name in enumerate(class_names):
        mask = labels == i
        style = class_styles[name]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=style['color'], marker=style['marker'],
            label=style['label'], s=120, alpha=0.85,
            edgecolors=style['edgecolor'], linewidth=2.0
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=14)
    ax.set_ylabel('UMAP Dimension 2', fontsize=14)
    ax.set_title(f'Latent Space at t={config["time_steps"]} steps\n(5x longer than Experiment 1)', fontsize=16, fontweight='bold')
    legend = ax.legend(fontsize=12, loc='best', framealpha=0.9, edgecolor='gray')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved to {save_path}")
    plt.close()
    
    return embedding


def analyze_anomaly_scores(model, surfaces, labels, class_names, config: dict):
    device = config['device']
    model = model.to(device)
    model.eval()
    
    surface_tensor = prepare_surface_batch(surfaces).to(device)
    anomaly_scores = model.anomaly_score(surface_tensor).cpu().numpy()
    
    print("\n" + "=" * 50)
    print(f"ANOMALY SCORES (t={config['time_steps']} steps)")
    print("=" * 50)
    
    training_scores = []
    unknown_scores = []
    
    for i, name in enumerate(class_names):
        mask = labels == i
        scores = anomaly_scores[mask]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if name in ['EW', 'KPZ']:
            label = '[TRAINING]'
            training_scores.extend(scores)
        else:
            label = '[UNKNOWN]'
            unknown_scores.extend(scores)
        
        print(f"{name:4s} {label:12s}: {mean_score:.4f} ± {std_score:.4f}")
    
    if training_scores and unknown_scores:
        train_mean = np.mean(training_scores)
        unknown_mean = np.mean(unknown_scores)
        ratio = unknown_mean / train_mean
        
        print("-" * 50)
        print(f"Training mean: {train_mean:.4f}")
        print(f"BD mean:       {unknown_mean:.4f}")
        print(f"Separation:    {ratio:.2f}x")
        
        if ratio < 1.3:
            print("\n✅ BD is SIMILAR to training data (approaching KPZ universality)")
        elif ratio < 2.0:
            print("\n⚠️  BD shows PARTIAL separation")
        else:
            print("\n🔴 BD is ANOMALOUS")


def main():
    print("=" * 60)
    print("EXPERIMENT 2: LONGER TIMESCALE (1000 steps)")
    print("=" * 60)
    print(f"Config: {CONFIG['width']}x{CONFIG['time_steps']}, {CONFIG['n_training_samples']} training samples")
    
    os.makedirs('figures', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Generate data
    train_surfaces = generate_training_data(CONFIG)
    print(f"Training surfaces shape: {train_surfaces.shape}")
    
    test_surfaces, test_labels, class_names = generate_test_data(CONFIG)
    print(f"Test surfaces shape: {test_surfaces.shape}")
    
    # Create model
    model = SurfaceAutoencoder(
        width=CONFIG['width'],
        time_steps=CONFIG['time_steps'],
        latent_dim=CONFIG['latent_dim']
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    train_tensor = prepare_surface_batch(train_surfaces)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=CONFIG['batch_size'], shuffle=True)
    
    print("\nTraining autoencoder...")
    loss_history = train_autoencoder(model, train_loader, CONFIG)
    
    # Save
    torch.save(model.state_dict(), 'checkpoints/autoencoder_exp02_1000steps.pt')
    
    # Visualize
    print("\nVisualizing latent space...")
    visualize_latent_space(model, test_surfaces, test_labels, class_names, CONFIG, 
                          'figures/exp02_latent_space_1000steps.png')
    
    # Analyze
    analyze_anomaly_scores(model, test_surfaces, test_labels, class_names, CONFIG)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
