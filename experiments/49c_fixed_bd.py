"""
Experiment 49c: Fixed BD Generalization Test

FIXES APPLIED:
--------------
1. Feature standardization (z-score per feature globally)
2. Shrinkage covariance estimation (Ledoit-Wolf)
3. Alternative distance: Maximum Mean Discrepancy (MMD)
4. Balanced training with proper loss weighting
5. Diagnostic checks for numerical stability

This addresses all issues found in Exp 49b diagnostics:
- Singular covariances (λ_min ~ 10^-18)
- Non-Gaussian features
- Scale mismatch (m2 ~ 1, m7 ~ 10^4)
- Classifier collapse (never predicting KPZ)
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.stats import pearsonr, linregress
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from simulation.physics_simulation import GrowthModelSimulator

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# LOAD DATA WITH STANDARDIZATION
# ============================================================================

def load_and_standardize_data():
    """Load Exp 49 data and apply global standardization"""
    results_path = Path('results/exp49_generalization_bd/results.pkl')
    
    with open(results_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    labels = data['labels']
    
    print(f"Loaded {len(features)} samples")
    print(f"  EW:  {(labels==0).sum()}")
    print(f"  KPZ: {(labels==1).sum()}")
    print(f"  BD:  {(labels==2).sum()}")
    
    # Global standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print("\nFeature scales after standardization:")
    print(f"  Mean: {features_scaled.mean(axis=0).round(3)}")
    print(f"  Std:  {features_scaled.std(axis=0).round(3)}")
    
    return features_scaled, labels, scaler

# ============================================================================
# TEST 1: INFORMATION GEOMETRY WITH ROBUST ESTIMATION
# ============================================================================

def compute_shrinkage_kl(feat_a, feat_b):
    """
    Compute symmetrized KL with Ledoit-Wolf shrinkage covariance.
    Guarantees positive-definite Σ.
    """
    # Ledoit-Wolf shrinkage
    lw_a = LedoitWolf()
    lw_b = LedoitWolf()
    
    cov_a = lw_a.fit(feat_a).covariance_
    cov_b = lw_b.fit(feat_b).covariance_
    
    # Means
    mu_a = feat_a.mean(axis=0)
    mu_b = feat_b.mean(axis=0)
    
    # Check condition numbers
    eigvals_a = np.linalg.eigvalsh(cov_a)
    eigvals_b = np.linalg.eigvalsh(cov_b)
    cond_a = eigvals_a.max() / eigvals_a.min()
    cond_b = eigvals_b.max() / eigvals_b.min()
    
    # Symmetrized KL
    inv_cov_a = np.linalg.inv(cov_a)
    inv_cov_b = np.linalg.inv(cov_b)
    
    kl_ab = 0.5 * (np.trace(inv_cov_b @ cov_a) + 
                  (mu_b - mu_a) @ inv_cov_b @ (mu_b - mu_a) - 
                  len(mu_a) + np.log(np.linalg.det(cov_b) / np.linalg.det(cov_a)))
    
    kl_ba = 0.5 * (np.trace(inv_cov_a @ cov_b) + 
                  (mu_a - mu_b) @ inv_cov_a @ (mu_a - mu_b) - 
                  len(mu_b) + np.log(np.linalg.det(cov_a) / np.linalg.det(cov_b)))
    
    sym_kl = kl_ab + kl_ba
    
    return sym_kl, cond_a, cond_b

def compute_mmd(feat_a, feat_b, gamma=1.0):
    """
    Maximum Mean Discrepancy with RBF kernel.
    Nonparametric, no Gaussian assumption.
    """
    # Subsample for speed if needed
    if len(feat_a) > 1000:
        idx_a = np.random.choice(len(feat_a), 1000, replace=False)
        feat_a = feat_a[idx_a]
    if len(feat_b) > 1000:
        idx_b = np.random.choice(len(feat_b), 1000, replace=False)
        feat_b = feat_b[idx_b]
    
    # RBF kernel
    def rbf_kernel(X, Y, gamma):
        dists = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * dists)
    
    K_aa = rbf_kernel(feat_a, feat_a, gamma).mean()
    K_bb = rbf_kernel(feat_b, feat_b, gamma).mean()
    K_ab = rbf_kernel(feat_a, feat_b, gamma).mean()
    
    mmd = K_aa + K_bb - 2 * K_ab
    return np.sqrt(max(0, mmd))

def test_information_geometry_robust(features, labels):
    """
    Test with both shrinkage KL and MMD.
    """
    print("\n" + "="*70)
    print("TEST 1: ROBUST INFORMATION-GEOMETRIC RG RELEVANCE")
    print("="*70)
    
    scales = [1, 2, 4, 8]
    class_pairs = [(0, 1, 'EW/KPZ'), (0, 2, 'EW/BD'), (1, 2, 'KPZ/BD')]
    
    results = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for pair_idx, (class_a, class_b, pair_name) in enumerate(class_pairs):
        ax_kl = axes[0, pair_idx]
        ax_mmd = axes[1, pair_idx]
        
        kl_distances = []
        mmd_distances = []
        kl_conds = []
        
        for scale in scales:
            # Coarse-grain
            if scale == 1:
                feat_coarse = features.copy()
                labels_coarse = labels.copy()
            else:
                n_blocks = len(features) // scale
                feat_coarse = features[:n_blocks*scale].reshape(n_blocks, scale, -1).mean(axis=1)
                labels_coarse = labels[:n_blocks*scale].reshape(n_blocks, scale).mean(axis=1).round().astype(int)
            
            mask_a = labels_coarse == class_a
            mask_b = labels_coarse == class_b
            
            if mask_a.sum() < 50 or mask_b.sum() < 50:
                continue
            
            feat_a = feat_coarse[mask_a]
            feat_b = feat_coarse[mask_b]
            
            # Shrinkage KL
            try:
                sym_kl, cond_a, cond_b = compute_shrinkage_kl(feat_a, feat_b)
                kl_distances.append(sym_kl)
                kl_conds.append(max(cond_a, cond_b))
            except:
                kl_distances.append(np.nan)
                kl_conds.append(np.nan)
            
            # MMD
            mmd = compute_mmd(feat_a, feat_b, gamma=0.1)
            mmd_distances.append(mmd)
        
        # Plot KL
        valid_scales_kl = [s for s, d in zip(scales, kl_distances) if not np.isnan(d)]
        valid_kl = [d for d in kl_distances if not np.isnan(d)]
        
        if len(valid_kl) >= 3:
            ax_kl.plot(valid_scales_kl, valid_kl, 'o-', linewidth=2, markersize=8, label='Shrinkage KL')
            
            slope_kl, intercept_kl, r_kl, p_kl, _ = linregress(np.log(valid_scales_kl), valid_kl)
            ax_kl.plot(valid_scales_kl, intercept_kl + slope_kl * np.log(valid_scales_kl), 
                      '--', alpha=0.5, label=f'slope={slope_kl:.3f}')
            
            verdict_kl = "✅ INCREASE" if (slope_kl > 0 and p_kl < 0.05) else "❌ DECREASE/FLAT"
            
            print(f"\n{pair_name} (Shrinkage KL):")
            print(f"  Slope: {slope_kl:.4f} (p={p_kl:.2e})")
            print(f"  Max cond(Σ): {max(kl_conds):.2e}")
            print(f"  Verdict: {verdict_kl}")
        
        ax_kl.set_xlabel('Coarse-graining scale b')
        ax_kl.set_ylabel('Symmetrized KL (shrinkage)')
        ax_kl.set_title(f'{pair_name} - Shrinkage KL')
        ax_kl.legend()
        ax_kl.grid(alpha=0.3)
        
        # Plot MMD
        if len(mmd_distances) >= 3:
            ax_mmd.plot(scales[:len(mmd_distances)], mmd_distances, 's-', linewidth=2, markersize=8, 
                       color='green', label='MMD')
            
            slope_mmd, intercept_mmd, r_mmd, p_mmd, _ = linregress(
                np.log(scales[:len(mmd_distances)]), mmd_distances
            )
            ax_mmd.plot(scales[:len(mmd_distances)], 
                       intercept_mmd + slope_mmd * np.log(scales[:len(mmd_distances)]), 
                       '--', alpha=0.5, color='green', label=f'slope={slope_mmd:.3f}')
            
            verdict_mmd = "✅ INCREASE" if (slope_mmd > 0 and p_mmd < 0.05) else "❌ DECREASE/FLAT"
            
            print(f"\n{pair_name} (MMD):")
            print(f"  Slope: {slope_mmd:.4f} (p={p_mmd:.2e})")
            print(f"  Verdict: {verdict_mmd}")
            
            results[pair_name] = {
                'kl_slope': slope_kl if len(valid_kl) >= 3 else np.nan,
                'kl_pvalue': p_kl if len(valid_kl) >= 3 else np.nan,
                'mmd_slope': slope_mmd,
                'mmd_pvalue': p_mmd,
                'max_condition': max(kl_conds) if len(valid_kl) >= 3 else np.nan
            }
        
        ax_mmd.set_xlabel('Coarse-graining scale b')
        ax_mmd.set_ylabel('MMD distance')
        ax_mmd.set_title(f'{pair_name} - MMD')
        ax_mmd.legend()
        ax_mmd.grid(alpha=0.3)
    
    plt.tight_layout()
    return results, fig

# ============================================================================
# TEST 2: COUPLING COORDINATES (unchanged, for reference)
# ============================================================================

def test_coupling_coordinates(features, labels):
    """PCA analysis per class"""
    print("\n" + "="*70)
    print("TEST 2: COUPLING COORDINATE ALIGNMENT")
    print("="*70)
    
    # PCA on standardized features
    pca = PCA()
    features_pca = pca.fit_transform(features)
    
    print(f"\nExplained variance:")
    for i in range(min(3, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for class_idx, class_name, ax in zip([0, 1, 2], ['EW', 'KPZ', 'BD'], axes):
        mask = labels == class_idx
        pc1_vals = features_pca[mask, 0]
        m2_vals = features[mask, 0]  # Standardized m2
        
        r, p = pearsonr(pc1_vals, m2_vals)
        
        ax.scatter(pc1_vals, m2_vals, alpha=0.3, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('Standardized m2')
        ax.set_title(f'{class_name}: r={r:.3f} (p={p:.2e})')
        ax.grid(alpha=0.3)
        
        print(f"\n{class_name}: PC1 vs m2: r={r:.3f} (p={p:.2e})")
    
    plt.tight_layout()
    
    return {'pca': pca, 'explained_variance': pca.explained_variance_ratio_}, fig

# ============================================================================
# TEST 3: FIXED LEARNED EMBEDDINGS (balanced loss, proper training)
# ============================================================================

class BalancedThreeClassModel(nn.Module):
    """3-way classifier with balanced training"""
    def __init__(self, input_dim=6, hidden_dim=32, feature_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.classifier = nn.Linear(feature_dim, 3)
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

class BalancedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def test_learned_embeddings_fixed(features, labels):
    """
    3-way classification with balanced sampling and class weights.
    """
    print("\n" + "="*70)
    print("TEST 3: BALANCED 3-WAY CLASSIFICATION")
    print("="*70)
    
    # Split data
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    
    train_dataset = BalancedDataset(X_train, y_train)
    val_dataset = BalancedDataset(X_val, y_val)
    
    # Balanced sampling
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Train: EW={( y_train==0).sum()}, KPZ={(y_train==1).sum()}, BD={(y_train==2).sum()}")
    print(f"Val:   EW={( y_val==0).sum()}, KPZ={(y_val==1).sum()}, BD={(y_val==2).sum()}")
    
    # Model with class weights
    model = BalancedThreeClassModel(input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Class-weighted loss
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training
    n_epochs = 100
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    best_val_acc = 0
    
    print("\nTraining with balanced sampling and class weights...")
    for epoch in range(n_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            _, logits = model(x)
            loss = criterion(logits, y)
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
                _, logits = model(x)
                _, pred = logits.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)
        
        val_acc = 100. * correct / total
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # Final confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, logits = model(x)
            _, pred = logits.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())
    
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nConfusion Matrix:")
    print("       EW  KPZ   BD")
    for i, row_name in enumerate(['EW', 'KPZ', 'BD']):
        print(f"{row_name:3s} {cm[i]}")
    
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(['EW', 'KPZ', 'BD']):
        class_acc = 100 * cm[i, i] / cm[i].sum()
        print(f"  {class_name}: {class_acc:.1f}%")
    
    # Check if all classes predicted
    preds_unique, preds_counts = np.unique(all_preds, return_counts=True)
    print(f"\nPrediction distribution:")
    for pred_class, count in zip(preds_unique, preds_counts):
        print(f"  Class {pred_class}: {count} ({100*count/len(all_preds):.1f}%)")
    
    if len(preds_unique) < 3:
        print("⚠️  WARNING: Not all classes are being predicted!")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training curves
    ax = axes[0]
    ax.plot(history['train_acc'], label='Train', alpha=0.7)
    ax.plot(history['val_acc'], label='Val', alpha=0.7)
    ax.axhline(33.3, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Balanced Training (Best: {best_val_acc:.1f}%)')
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
    
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    results = {
        'history': history,
        'best_val_acc': best_val_acc,
        'final_val_acc': history['val_acc'][-1],
        'confusion_matrix': cm
    }
    
    return results, fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 49c: FIXED BD GENERALIZATION TEST")
    print("="*70)
    
    output_dir = Path('results/exp49c_fixed_bd')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and standardize
    features, labels, scaler = load_and_standardize_data()
    
    # Test 1: Robust information geometry
    info_results, info_fig = test_information_geometry_robust(features, labels)
    info_fig.savefig(output_dir / 'test1_robust_info_geom.png', dpi=150, bbox_inches='tight')
    
    # Test 2: Coupling coordinates
    coupling_results, coupling_fig = test_coupling_coordinates(features, labels)
    coupling_fig.savefig(output_dir / 'test2_coupling.png', dpi=150, bbox_inches='tight')
    
    # Test 3: Fixed embeddings
    embedding_results, embedding_fig = test_learned_embeddings_fixed(features, labels)
    embedding_fig.savefig(output_dir / 'test3_balanced_classifier.png', dpi=150, bbox_inches='tight')
    
    # Final assessment
    print("\n" + "="*70)
    print("FINAL ASSESSMENT: BD GENERALIZATION (FIXED)")
    print("="*70)
    
    # Count successes
    test1_kl_success = sum(
        1 for r in info_results.values()
        if not np.isnan(r.get('kl_slope', np.nan)) and 
           r['kl_slope'] > 0 and r['kl_pvalue'] < 0.05
    )
    
    test1_mmd_success = sum(
        1 for r in info_results.values()
        if r['mmd_slope'] > 0 and r['mmd_pvalue'] < 0.05
    )
    
    test3_success = embedding_results['best_val_acc'] > 80
    
    print(f"\nTest 1 (Shrinkage KL): {test1_kl_success}/3 pairs show RG relevance")
    print(f"Test 1 (MMD): {test1_mmd_success}/3 pairs show RG relevance")
    print(f"Test 2: Coupling coordinate correlations computed")
    print(f"Test 3: {embedding_results['best_val_acc']:.1f}% best accuracy")
    
    # Interpret
    if test1_mmd_success >= 2 and test3_success:
        print("\n✅ STRONG GENERALIZATION: Framework detects BD boundaries")
        verdict = "generalizes"
    elif test1_mmd_success >= 1 or embedding_results['best_val_acc'] > 60:
        print("\n⚠️  PARTIAL GENERALIZATION: Some evidence of BD detection")
        verdict = "partial"
    else:
        print("\n❌ LIMITED GENERALIZATION: BD not cleanly detected")
        verdict = "limited"
    
    # Save
    results = {
        'features': features,
        'labels': labels,
        'scaler': scaler,
        'test1_info_geom': info_results,
        'test2_coupling': coupling_results,
        'test3_embedding': embedding_results,
        'verdict': verdict
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
