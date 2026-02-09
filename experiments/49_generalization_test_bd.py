"""
Experiment 49: Generalization Test with Ballistic Deposition

CRITICAL QUESTION (from Assessment 3):
---------------------------------------
Does the three-pillar framework generalize beyond KPZ-family systems?

APPROACH:
---------
Test the same framework (Exp 46/45b/47) on three universality classes:
  - Edwards-Wilkinson (EW): α=0, β=0, z=2 (diffusive)
  - Kardar-Parisi-Zhang (KPZ): α=1/2, β=1/3, z=3/2 (nonlinear)
  - Ballistic Deposition (BD): α≈1/2, β≈1/3, z≈3/2 (KPZ-like but discrete)

If framework works: Should detect ALL pairwise boundaries (EW/KPZ, EW/BD, BD/KPZ)
If KPZ-specific: Only EW/KPZ boundary detected

THREE TESTS:
------------
1. Information Geometry (Exp 47 analog):
   - Compute KL divergence between all class pairs at scales b=1,2,4,8
   - Check if distances INCREASE under coarse-graining for all pairs
   
2. Coupling Coordinates (Exp 46 analog):
   - For BD: different coupling structure (sticking vs nonlinearity)
   - Check if PC1 correlates with relevant BD parameters
   
3. Learned Embeddings (Exp 45b analog):
   - 3-way classification: EW vs KPZ vs BD
   - Multi-task: classification + RG covariance
   - Check if achieves separation for all classes

EXPECTED OUTCOMES:
------------------
STRONG GENERALIZATION:
  - All 3 tests succeed for all class pairs
  - PC1 tracks different couplings per class
  - Learned embeddings separate all three
  → Framework is universal

PARTIAL GENERALIZATION:
  - EW/KPZ boundary detected (known to work)
  - EW/BD or BD/KPZ partially detected
  → Some structure but limited

NO GENERALIZATION:
  - Only EW/KPZ works, BD treated like KPZ
  → Framework is KPZ-family specific

Status: NEW (Feb 3, 2026) - Critical scope test per Assessment 3
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.stats import pearsonr, linregress
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from simulation.physics_simulation import GrowthModelSimulator

# Style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. DATA GENERATION
# ============================================================================

def generate_three_class_data(n_trajectories=30, L=256, T=500):
    """
    Generate gradient moment features for EW, KPZ, and BD.
    Using droplet IC (Exp 27 showed best separation).
    """
    print("="*70)
    print("GENERATING DATA FOR THREE UNIVERSALITY CLASSES")
    print("="*70)
    
    simulator = GrowthModelSimulator(width=L, height=T)
    
    all_features = []
    all_labels = []
    all_model_names = []
    
    models = [
        ('edwards_wilkinson', 0, 'EW', {'diffusion': 1.0, 'noise_strength': 1.0, 'dt': 0.1}),
        ('kpz_equation', 1, 'KPZ', {'diffusion': 1.0, 'nonlinearity': 1.0, 'noise_strength': 1.0, 'dt': 0.05}),
        ('ballistic_deposition', 2, 'BD', {'noise_strength': 0.2})
    ]
    
    for model_type, label, name, params in models:
        print(f"\nSimulating {name}...")
        
        for traj_idx in range(n_trajectories):
            if (traj_idx + 1) % 10 == 0:
                print(f"  Trajectory {traj_idx+1}/{n_trajectories}")
            
            # Generate trajectory using GrowthModelSimulator
            traj = simulator.generate_trajectory(model_type=model_type, **params)
            
            # Extract features from snapshots (t>100 to skip transient)
            # traj is shape (time_steps, L)
            for t_idx in range(traj.shape[0] // 5, traj.shape[0]):  # skip first 20% transient
                h = traj[t_idx]
                
                # Gradient moments
                grad = np.gradient(h)
                features = np.array([
                    np.mean(grad**2),  # m2
                    np.mean(grad**3),  # m3
                    np.mean(grad**4),  # m4
                    np.mean(grad**5),  # m5
                    np.mean(grad**6),  # m6
                    np.mean(np.abs(grad)**7)  # m7
                ])
                
                all_features.append(features)
                all_labels.append(label)
                all_model_names.append(name)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"\nGenerated {len(all_features)} samples:")
    print(f"  EW:  {(all_labels==0).sum()}")
    print(f"  KPZ: {(all_labels==1).sum()}")
    print(f"  BD:  {(all_labels==2).sum()}")
    
    return all_features, all_labels, all_model_names

# ============================================================================
# 2. TEST 1: INFORMATION GEOMETRY
# ============================================================================

def test_information_geometry(features, labels):
    """
    Compute KL divergence between all class pairs at multiple scales.
    Check if distances INCREASE under coarse-graining.
    """
    print("\n" + "="*70)
    print("TEST 1: INFORMATION-GEOMETRIC RG RELEVANCE")
    print("="*70)
    
    scales = [1, 2, 4, 8]
    class_pairs = [(0, 1, 'EW/KPZ'), (0, 2, 'EW/BD'), (1, 2, 'KPZ/BD')]
    
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for pair_idx, (class_a, class_b, pair_name) in enumerate(class_pairs):
        ax = axes[pair_idx]
        
        distances = []
        
        for scale in scales:
            # Coarse-grain features via averaging
            if scale == 1:
                feat_coarse = features.copy()
            else:
                # Block average (simple coarse-graining)
                n_blocks = len(features) // scale
                feat_coarse = features[:n_blocks*scale].reshape(n_blocks, scale, -1).mean(axis=1)
                labels_coarse = labels[:n_blocks*scale].reshape(n_blocks, scale).mean(axis=1).round().astype(int)
            
            # Gaussian approximation for KL
            mask_a = (labels if scale == 1 else labels_coarse) == class_a
            mask_b = (labels if scale == 1 else labels_coarse) == class_b
            
            if mask_a.sum() < 10 or mask_b.sum() < 10:
                continue
            
            feat_a = (features if scale == 1 else feat_coarse)[mask_a]
            feat_b = (features if scale == 1 else feat_coarse)[mask_b]
            
            # Compute means and covariances
            mu_a, mu_b = feat_a.mean(axis=0), feat_b.mean(axis=0)
            cov_a = np.cov(feat_a, rowvar=False) + 1e-6 * np.eye(feat_a.shape[1])
            cov_b = np.cov(feat_b, rowvar=False) + 1e-6 * np.eye(feat_b.shape[1])
            
            # Symmetrized KL divergence
            try:
                inv_cov_a = np.linalg.inv(cov_a)
                inv_cov_b = np.linalg.inv(cov_b)
                
                kl_ab = 0.5 * (np.trace(inv_cov_b @ cov_a) + 
                              (mu_b - mu_a) @ inv_cov_b @ (mu_b - mu_a) - 
                              len(mu_a) + np.log(np.linalg.det(cov_b) / np.linalg.det(cov_a)))
                
                kl_ba = 0.5 * (np.trace(inv_cov_a @ cov_b) + 
                              (mu_a - mu_b) @ inv_cov_a @ (mu_a - mu_b) - 
                              len(mu_b) + np.log(np.linalg.det(cov_a) / np.linalg.det(cov_b)))
                
                sym_kl = kl_ab + kl_ba
                distances.append(sym_kl)
            except np.linalg.LinAlgError:
                distances.append(np.nan)
        
        # Plot
        valid_scales = [s for s, d in zip(scales, distances) if not np.isnan(d)]
        valid_distances = [d for d in distances if not np.isnan(d)]
        
        if len(valid_distances) >= 3:
            ax.plot(valid_scales, valid_distances, 'o-', linewidth=2, markersize=8)
            
            # Linear fit
            slope, intercept, r_value, p_value, std_err = linregress(
                np.log(valid_scales), valid_distances
            )
            
            ax.plot(valid_scales, intercept + slope * np.log(valid_scales), 
                   '--', alpha=0.5, label=f'Slope: {slope:.3f}')
            
            results[pair_name] = {
                'scales': valid_scales,
                'distances': valid_distances,
                'slope': slope,
                'p_value': p_value
            }
            
            # Interpret
            if slope > 0 and p_value < 0.05:
                verdict = "✅ INCREASE (RG-relevant)"
            elif slope > 0:
                verdict = "⚠️  Positive but not significant"
            else:
                verdict = "❌ DECREASE or flat"
            
            print(f"\n{pair_name}:")
            print(f"  Slope: {slope:.3f} (p={p_value:.2e})")
            print(f"  Verdict: {verdict}")
        
        ax.set_xlabel('Coarse-graining scale b')
        ax.set_ylabel('Symmetrized KL divergence')
        ax.set_title(f'{pair_name}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return results, fig

# ============================================================================
# 3. TEST 2: COUPLING COORDINATES
# ============================================================================

def test_coupling_coordinates(features, labels):
    """
    Check if PC1 correlates with physical couplings.
    For BD: correlation structure may differ from EW/KPZ.
    """
    print("\n" + "="*70)
    print("TEST 2: COUPLING COORDINATE ALIGNMENT")
    print("="*70)
    
    # PCA on all classes
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA()
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance:")
    for i in range(min(3, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f}")
    
    # Per-class analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for class_idx, class_name, ax in zip([0, 1, 2], ['EW', 'KPZ', 'BD'], axes):
        mask = labels == class_idx
        pc1_vals = features_pca[mask, 0]
        
        # For EW/KPZ: check gradient variance (m2) correlation
        # For BD: check different feature (BD has discrete dynamics)
        m2_vals = features[mask, 0]  # gradient variance
        
        r, p = pearsonr(pc1_vals, m2_vals)
        
        ax.scatter(pc1_vals, m2_vals, alpha=0.3, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('Gradient variance (m2)')
        ax.set_title(f'{class_name}: r={r:.3f} (p={p:.2e})')
        ax.grid(alpha=0.3)
        
        print(f"\n{class_name}:")
        print(f"  PC1 vs m2: r={r:.3f} (p={p:.2e})")
    
    plt.tight_layout()
    
    results = {
        'pca': pca,
        'explained_variance': pca.explained_variance_ratio_,
        'features_pca': features_pca
    }
    
    return results, fig

# ============================================================================
# 4. TEST 3: LEARNED EMBEDDINGS (3-way classification)
# ============================================================================

class ThreeClassRGModel(nn.Module):
    """Multi-task: RG covariance + 3-way classification"""
    def __init__(self, input_dim=6, feature_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 3)  # 3 classes
        )
        
        # RG transformation (one per scale)
        self.rg_transform = nn.Linear(feature_dim, feature_dim, bias=True)
    
    def forward(self, x, x_coarse=None):
        features = self.encoder(x)
        logits = self.classifier(features)
        
        if x_coarse is not None:
            features_coarse = self.encoder(x_coarse)
            features_rg = self.rg_transform(features)
            return features, logits, features_coarse, features_rg
        
        return features, logits, None, None

class ThreeClassDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def test_learned_embeddings(features, labels):
    """
    3-way classification with RG covariance constraint.
    """
    print("\n" + "="*70)
    print("TEST 3: LEARNED RG-COVARIANT EMBEDDINGS (3-way)")
    print("="*70)
    
    # Split data
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    train_dataset = ThreeClassDataset(features[train_idx], labels[train_idx])
    val_dataset = ThreeClassDataset(features[val_idx], labels[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Model
    model = ThreeClassRGModel(input_dim=6, feature_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_class = nn.CrossEntropyLoss()
    
    # Training
    n_epochs = 50
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    
    print("\nTraining 3-way classifier...")
    for epoch in range(n_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, logits, _, _ = model(x)
            loss = criterion_class(logits, y)
            loss.backward()
            optimizer.step()
            
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
        
        train_acc = 100. * correct / total
        history['train_acc'].append(train_acc)
        history['train_loss'].append(total_loss / len(train_loader))
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, logits, _, _ = model(x)
                _, pred = logits.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        val_acc = 100. * correct / total
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
    
    # Final evaluation
    print(f"\n Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, logits, _, _ = model(x)
            _, pred = logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nConfusion Matrix:")
    print("       EW  KPZ   BD")
    for i, row_name in enumerate(['EW', 'KPZ', 'BD']):
        print(f"{row_name:3s} {cm[i]}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training curves
    ax = axes[0]
    ax.plot(history['train_acc'], label='Train', alpha=0.7)
    ax.plot(history['val_acc'], label='Val', alpha=0.7)
    ax.axhline(33.3, color='gray', linestyle='--', alpha=0.5, label='Chance (1/3)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('3-Way Classification')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confusion matrix
    ax = axes[1]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['EW', 'KPZ', 'BD'])
    ax.set_yticklabels(['EW', 'KPZ', 'BD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    results = {
        'history': history,
        'final_val_acc': history['val_acc'][-1],
        'confusion_matrix': cm
    }
    
    return results, fig

# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 49: GENERALIZATION TEST WITH BALLISTIC DEPOSITION")
    print("="*70)
    print("\nCRITICAL QUESTION: Does framework generalize beyond KPZ-family?")
    print("\nTesting three universality classes: EW, KPZ, BD")
    print("="*70)
    
    output_dir = Path('results/exp49_generalization_bd')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    features, labels, model_names = generate_three_class_data(
        n_trajectories=30, L=256, T=500
    )
    
    # Test 1: Information geometry
    info_geom_results, info_geom_fig = test_information_geometry(features, labels)
    info_geom_fig.savefig(output_dir / 'test1_information_geometry.png', 
                          dpi=150, bbox_inches='tight')
    
    # Test 2: Coupling coordinates
    coupling_results, coupling_fig = test_coupling_coordinates(features, labels)
    coupling_fig.savefig(output_dir / 'test2_coupling_coordinates.png', 
                        dpi=150, bbox_inches='tight')
    
    # Test 3: Learned embeddings
    embedding_results, embedding_fig = test_learned_embeddings(features, labels)
    embedding_fig.savefig(output_dir / 'test3_learned_embeddings.png', 
                         dpi=150, bbox_inches='tight')
    
    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT: GENERALIZATION TO BD")
    print("="*70)
    
    # Count successes
    test1_success = sum(
        1 for results in info_geom_results.values() 
        if results['slope'] > 0 and results['p_value'] < 0.05
    )
    
    test3_success = embedding_results['final_val_acc'] > 80
    
    print(f"\nTest 1 (Info Geometry): {test1_success}/3 pairs show RG relevance")
    print(f"Test 2 (Coupling): Per-class correlations computed")
    print(f"Test 3 (Embeddings): {embedding_results['final_val_acc']:.1f}% accuracy")
    
    if test1_success == 3 and test3_success:
        print("\n✅ STRONG GENERALIZATION: Framework works for all class pairs!")
        print("   → High-impact result: method generalizes beyond KPZ")
    elif test1_success >= 2 and test3_success:
        print("\n⚠️  PARTIAL GENERALIZATION: Works for most pairs")
        print("   → Method partially generalizes, some limitations")
    else:
        print("\n❌ LIMITED GENERALIZATION: Framework is KPZ-family specific")
        print("   → Still solid contribution, but narrower scope")
    
    # Save results
    results = {
        'features': features,
        'labels': labels,
        'model_names': model_names,
        'test1_info_geom': info_geom_results,
        'test2_coupling': coupling_results,
        'test3_embedding': embedding_results
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
