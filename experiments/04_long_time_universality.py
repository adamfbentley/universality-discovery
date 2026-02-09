"""
Experiment 4: Long-Time Universality Test

CRITICAL QUESTION:
At long times, does the EDEN model's universality class (KPZ) become detectable,
or does discreteness ALWAYS dominate the autoencoder's representation?

HYPOTHESIS:
If universality is real and the autoencoder can detect it, EDEN should:
- At short times: Cluster with discrete models (BD, RD) - dominated by discreteness
- At long times: Converge toward KPZ - universality emerges

EXPERIMENT DESIGN:
- Train on continuum models (EW + KPZ) at LONG times (1000+ steps)
- Test multiple time horizons: 500, 1000, 2000 steps
- Track how EDEN's anomaly score changes relative to KPZ

SUCCESS CRITERIA:
- If EDEN score approaches KPZ at long times → Universality detectable!
- If EDEN stays with BD/RD at all times → Discreteness always dominates
"""

import sys
import os
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.autoencoder import SurfaceAutoencoder
from simulation.physics_simulation import GrowthModelSimulator


def generate_surfaces(model_type: str, n_samples: int, width: int, time_steps: int, 
                      seed_offset: int = 0) -> np.ndarray:
    """Generate surfaces for a specific model type."""
    surfaces = []
    for i in range(n_samples):
        sim = GrowthModelSimulator(width, time_steps, random_state=seed_offset + i * 100)
        trajectory = sim.generate_trajectory(model_type)
        surfaces.append(trajectory.T)  # Transpose to (width, time_steps)
    return np.array(surfaces)


def train_autoencoder(model, train_data, epochs=30, batch_size=16, lr=1e-3):
    """Train autoencoder and return loss history."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    # Normalize data
    mean = train_data.mean()
    std = train_data.std() + 1e-8
    train_normalized = (train_data - mean) / std
    
    # Convert to tensor
    X = torch.FloatTensor(train_normalized).unsqueeze(1).to(device)
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    losses = []
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        epoch_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        pbar.set_postfix(loss=f"{avg_loss:.6f}")
    
    return losses, mean, std


def compute_anomaly_scores(model, data, mean, std):
    """Compute reconstruction error for each sample."""
    device = next(model.parameters()).device
    normalized = (data - mean) / std
    X = torch.FloatTensor(normalized).unsqueeze(1).to(device)
    
    model.eval()
    with torch.no_grad():
        recon, _ = model(X)
        errors = ((recon - X) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    
    return errors


def main():
    print("=" * 70)
    print("EXPERIMENT 4: LONG-TIME UNIVERSALITY TEST")
    print("=" * 70)
    print("\nQuestion: Does universality emerge at long times,")
    print("          or does discreteness always dominate?\n")
    
    # Configuration
    WIDTH = 128
    TIME_STEPS_LIST = [500, 1000, 1500]  # Test multiple time horizons
    N_TRAIN = 200  # Per model type (EW + KPZ)
    N_TEST = 40    # Per model type
    EPOCHS = 25
    LATENT_DIM = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Store results across time scales
    results = {ts: {} for ts in TIME_STEPS_LIST}
    
    for time_steps in TIME_STEPS_LIST:
        print(f"\n{'='*70}")
        print(f"TIME HORIZON: {time_steps} steps")
        print(f"{'='*70}")
        
        # Generate training data (continuum only)
        print(f"\nGenerating training data (EW + KPZ)...")
        ew_train = generate_surfaces('edwards_wilkinson', N_TRAIN, WIDTH, time_steps, seed_offset=0)
        kpz_train = generate_surfaces('kpz_equation', N_TRAIN, WIDTH, time_steps, seed_offset=10000)
        train_data = np.concatenate([ew_train, kpz_train], axis=0)
        print(f"  Training shape: {train_data.shape}")
        
        # Generate test data (all models including discrete)
        print(f"\nGenerating test data...")
        test_data = {}
        
        # Continuum models
        test_data['EW'] = generate_surfaces('edwards_wilkinson', N_TEST, WIDTH, time_steps, seed_offset=50000)
        test_data['KPZ'] = generate_surfaces('kpz_equation', N_TEST, WIDTH, time_steps, seed_offset=60000)
        
        # Discrete models
        test_data['BD'] = generate_surfaces('ballistic_deposition', N_TEST, WIDTH, time_steps, seed_offset=70000)
        test_data['EDEN'] = generate_surfaces('eden', N_TEST, WIDTH, time_steps, seed_offset=80000)
        
        for name, data in test_data.items():
            print(f"  {name}: {data.shape}")
        
        # Create and train model
        model = SurfaceAutoencoder(width=WIDTH, time_steps=time_steps, latent_dim=LATENT_DIM).to(device)
        
        print(f"\nTraining autoencoder...")
        losses, mean, std = train_autoencoder(model, train_data, epochs=EPOCHS)
        print(f"  Final loss: {losses[-1]:.6f}")
        
        # Compute anomaly scores
        print(f"\nComputing anomaly scores...")
        for name, data in test_data.items():
            scores = compute_anomaly_scores(model, data, mean, std)
            results[time_steps][name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        # Print results for this time horizon
        baseline = (results[time_steps]['EW']['mean'] + results[time_steps]['KPZ']['mean']) / 2
        
        print(f"\n  {'Model':<8} {'Score':>12} {'Separation':>12}")
        print(f"  {'-'*36}")
        for name in ['EW', 'KPZ', 'BD', 'EDEN']:
            r = results[time_steps][name]
            sep = r['mean'] / baseline
            print(f"  {name:<8} {r['mean']:>8.4f} ± {r['std']:.4f} {sep:>8.1f}x")
    
    # =========================================================================
    # CRITICAL ANALYSIS: How does EDEN evolve with time?
    # =========================================================================
    print("\n" + "=" * 70)
    print("CRITICAL ANALYSIS: EDEN TRAJECTORY ACROSS TIME SCALES")
    print("=" * 70)
    
    print(f"\n{'Time':>8} {'EDEN Score':>14} {'KPZ Score':>14} {'EDEN/KPZ Ratio':>16} {'Interpretation':>20}")
    print("-" * 76)
    
    interpretations = []
    for ts in TIME_STEPS_LIST:
        eden_score = results[ts]['EDEN']['mean']
        kpz_score = results[ts]['KPZ']['mean']
        ratio = eden_score / kpz_score
        
        if ratio < 1.3:
            interp = "✓ CONVERGING!"
        elif ratio < 1.8:
            interp = "~ PARTIAL"
        else:
            interp = "✗ DISCRETE"
        
        interpretations.append((ts, ratio, interp))
        print(f"{ts:>8} {eden_score:>10.4f} ± {results[ts]['EDEN']['std']:.3f} "
              f"{kpz_score:>10.4f} ± {results[ts]['KPZ']['std']:.3f} "
              f"{ratio:>12.2f}x     {interp}")
    
    # Determine overall conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    ratios = [r[1] for r in interpretations]
    if ratios[-1] < ratios[0] * 0.8:  # Decreasing trend
        print("\n🎯 UNIVERSALITY EMERGES AT LONG TIMES!")
        print("   The EDEN model converges toward KPZ as time increases.")
        print("   The autoencoder CAN detect universality class at long times.")
    elif ratios[-1] > ratios[0] * 1.2:  # Increasing trend
        print("\n⚠️ DISCRETENESS PERSISTS!")
        print("   The EDEN model remains distinct from KPZ even at long times.")
        print("   The autoencoder primarily detects microscopic features.")
    else:
        print("\n📊 INCONCLUSIVE - Need longer time scales")
        print("   The trend is unclear at these time horizons.")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Anomaly scores across time for each model
    ax1 = axes[0]
    colors = {'EW': 'blue', 'KPZ': 'green', 'BD': 'red', 'EDEN': 'purple'}
    markers = {'EW': 'o', 'KPZ': 's', 'BD': '^', 'EDEN': 'D'}
    
    for name in ['EW', 'KPZ', 'BD', 'EDEN']:
        scores = [results[ts][name]['mean'] for ts in TIME_STEPS_LIST]
        stds = [results[ts][name]['std'] for ts in TIME_STEPS_LIST]
        ax1.errorbar(TIME_STEPS_LIST, scores, yerr=stds, 
                    label=name, color=colors[name], marker=markers[name],
                    linewidth=2, markersize=8, capsize=5)
    
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Anomaly Score', fontsize=12)
    ax1.set_title('Anomaly Score vs Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: EDEN/KPZ ratio over time (key metric!)
    ax2 = axes[1]
    eden_kpz_ratios = [results[ts]['EDEN']['mean'] / results[ts]['KPZ']['mean'] 
                       for ts in TIME_STEPS_LIST]
    bd_kpz_ratios = [results[ts]['BD']['mean'] / results[ts]['KPZ']['mean'] 
                     for ts in TIME_STEPS_LIST]
    
    ax2.plot(TIME_STEPS_LIST, eden_kpz_ratios, 'D-', color='purple', 
             linewidth=2, markersize=10, label='EDEN/KPZ')
    ax2.plot(TIME_STEPS_LIST, bd_kpz_ratios, '^-', color='red',
             linewidth=2, markersize=10, label='BD/KPZ')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='KPZ baseline')
    
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Score Ratio (vs KPZ)', fontsize=12)
    ax2.set_title('Convergence to KPZ Universality?', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Score distributions at longest time
    ax3 = axes[2]
    longest_ts = TIME_STEPS_LIST[-1]
    positions = [1, 2, 3.5, 4.5]  # Gap between continuum and discrete
    
    box_data = [results[longest_ts][name]['scores'] for name in ['EW', 'KPZ', 'BD', 'EDEN']]
    bp = ax3.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)
    
    face_colors = ['lightblue', 'lightgreen', 'lightcoral', 'plum']
    for patch, color in zip(bp['boxes'], face_colors):
        patch.set_facecolor(color)
    
    ax3.set_xticks(positions)
    ax3.set_xticklabels(['EW', 'KPZ', 'BD', 'EDEN'])
    ax3.set_ylabel('Anomaly Score', fontsize=12)
    ax3.set_title(f'Score Distribution (t={longest_ts})', fontsize=14, fontweight='bold')
    ax3.axvline(x=2.75, color='gray', linestyle=':', linewidth=2)
    ax3.text(1.5, ax3.get_ylim()[1]*0.95, 'Continuum', ha='center', fontsize=10)
    ax3.text(4.0, ax3.get_ylim()[1]*0.95, 'Discrete', ha='center', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/exp04_long_time_universality.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: figures/exp04_long_time_universality.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
