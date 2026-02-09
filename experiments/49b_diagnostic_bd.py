"""
Experiment 49b: Diagnostic Analysis of BD Test Failures

PURPOSE:
--------
Diagnose why Exp 49 gave invalid results:
1. Check covariance condition numbers (expect singularity/near-singularity)
2. Verify feature distributions (Gaussian assumption may fail for BD)
3. Check training data labels (KPZ never predicted → label bug?)
4. Test simple baseline classifier (can it separate EW/KPZ/BD at all?)

This will tell us whether negative results are real physics or numerical artifacts.
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy.stats import pearsonr, linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from simulation.physics_simulation import GrowthModelSimulator

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# LOAD DATA FROM EXP 49
# ============================================================================

def load_exp49_data():
    """Load features/labels from Exp 49 run"""
    results_path = Path('results/exp49_generalization_bd/results.pkl')
    
    if results_path.exists():
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
        print("✅ Loaded Exp 49 data")
        return data['features'], data['labels']
    else:
        print("⚠️  No Exp 49 data found, generating fresh...")
        return None, None

# ============================================================================
# DIAGNOSTIC 1: COVARIANCE CONDITION NUMBERS
# ============================================================================

def diagnose_covariance_stability(features, labels):
    """
    Check if covariance matrices are numerically stable.
    Report eigenvalues, condition numbers, det values.
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC 1: COVARIANCE STABILITY")
    print("="*70)
    
    scales = [1, 2, 4, 8]
    class_pairs = [(0, 1, 'EW/KPZ'), (0, 2, 'EW/BD'), (1, 2, 'KPZ/BD')]
    
    diagnostics = {}
    
    for pair_idx, (class_a, class_b, pair_name) in enumerate(class_pairs):
        print(f"\n{pair_name}:")
        print("-" * 40)
        
        for scale in scales:
            # Coarse-grain via block averaging
            if scale == 1:
                feat_coarse = features.copy()
                labels_coarse = labels.copy()
            else:
                n_blocks = len(features) // scale
                feat_coarse = features[:n_blocks*scale].reshape(n_blocks, scale, -1).mean(axis=1)
                labels_coarse = labels[:n_blocks*scale].reshape(n_blocks, scale).mean(axis=1).round().astype(int)
            
            mask_a = labels_coarse == class_a
            mask_b = labels_coarse == class_b
            
            if mask_a.sum() < 10 or mask_b.sum() < 10:
                continue
            
            feat_a = feat_coarse[mask_a]
            feat_b = feat_coarse[mask_b]
            
            # Compute covariances
            cov_a = np.cov(feat_a, rowvar=False)
            cov_b = np.cov(feat_b, rowvar=False)
            
            # Eigenvalues
            eigvals_a = np.linalg.eigvalsh(cov_a)
            eigvals_b = np.linalg.eigvalsh(cov_b)
            
            # Condition numbers
            cond_a = eigvals_a.max() / (eigvals_a.min() + 1e-15)
            cond_b = eigvals_b.max() / (eigvals_b.min() + 1e-15)
            
            # Determinants
            det_a = np.linalg.det(cov_a)
            det_b = np.linalg.det(cov_b)
            
            print(f"\n  Scale b={scale}:")
            print(f"    Class {class_a} ({['EW','KPZ','BD'][class_a]}):")
            print(f"      λ_min:     {eigvals_a.min():.2e}")
            print(f"      λ_max:     {eigvals_a.max():.2e}")
            print(f"      cond(Σ):   {cond_a:.2e}")
            print(f"      det(Σ):    {det_a:.2e}")
            
            print(f"    Class {class_b} ({['EW','KPZ','BD'][class_b]}):")
            print(f"      λ_min:     {eigvals_b.min():.2e}")
            print(f"      λ_max:     {eigvals_b.max():.2e}")
            print(f"      cond(Σ):   {cond_b:.2e}")
            print(f"      det(Σ):    {det_b:.2e}")
            
            # Flag problems
            if cond_a > 1e10 or cond_b > 1e10:
                print(f"    ⚠️  ILL-CONDITIONED: cond(Σ) > 10^10")
            if eigvals_a.min() < 1e-10 or eigvals_b.min() < 1e-10:
                print(f"    ⚠️  NEAR-SINGULAR: λ_min < 10^-10")
            if det_a < 1e-50 or det_b < 1e-50:
                print(f"    ⚠️  UNDERFLOW: det(Σ) < 10^-50")
            
            diagnostics[(pair_name, scale)] = {
                'eigvals_a': eigvals_a,
                'eigvals_b': eigvals_b,
                'cond_a': cond_a,
                'cond_b': cond_b,
                'det_a': det_a,
                'det_b': det_b
            }
    
    return diagnostics

# ============================================================================
# DIAGNOSTIC 2: FEATURE DISTRIBUTIONS (GAUSSIANITY CHECK)
# ============================================================================

def diagnose_gaussianity(features, labels):
    """Check if features are approximately Gaussian per class"""
    print("\n" + "="*70)
    print("DIAGNOSTIC 2: GAUSSIANITY CHECK")
    print("="*70)
    
    from scipy.stats import shapiro, normaltest
    
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    
    for class_idx, class_name in enumerate(['EW', 'KPZ', 'BD']):
        mask = labels == class_idx
        feat_class = features[mask]
        
        print(f"\n{class_name} (n={mask.sum()}):")
        print("-" * 40)
        
        for feat_idx in range(6):
            ax = axes[class_idx, feat_idx]
            
            vals = feat_class[:, feat_idx]
            
            # Histogram
            ax.hist(vals, bins=50, density=True, alpha=0.6, edgecolor='black')
            ax.set_title(f'{class_name} m{feat_idx+2}')
            ax.set_xlabel('Value')
            if feat_idx == 0:
                ax.set_ylabel('Density')
            
            # Normality test (using normaltest which is more robust for large n)
            stat, p = normaltest(vals)
            
            # Add text with p-value
            verdict = "✅ Gaussian" if p > 0.01 else "❌ Non-Gaussian"
            ax.text(0.05, 0.95, f'p={p:.2e}\n{verdict}', 
                   transform=ax.transAxes, va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            print(f"  m{feat_idx+2}: p={p:.2e} {'(Gaussian)' if p > 0.01 else '(Non-Gaussian)'}")
    
    plt.tight_layout()
    return fig

# ============================================================================
# DIAGNOSTIC 3: TRAINING DATA SANITY CHECK
# ============================================================================

def diagnose_training_data(features, labels):
    """Check for label bugs, imbalance, or separability issues"""
    print("\n" + "="*70)
    print("DIAGNOSTIC 3: TRAINING DATA SANITY")
    print("="*70)
    
    # Label counts
    print("\nLabel distribution:")
    for class_idx, class_name in enumerate(['EW', 'KPZ', 'BD']):
        count = (labels == class_idx).sum()
        print(f"  {class_name}: {count} ({100*count/len(labels):.1f}%)")
    
    # Check for duplicate/constant features
    print("\nFeature statistics:")
    for feat_idx in range(features.shape[1]):
        vals = features[:, feat_idx]
        print(f"  m{feat_idx+2}: min={vals.min():.2e}, max={vals.max():.2e}, "
              f"std={vals.std():.2e}")
        
        if vals.std() < 1e-10:
            print(f"    ⚠️  CONSTANT FEATURE")
    
    # Check feature correlations
    print("\nFeature correlation matrix (absolute):")
    corr = np.corrcoef(features.T)
    print(np.abs(corr).round(2))
    
    # High correlation warning
    high_corr = np.where((np.abs(corr) > 0.99) & (np.abs(corr) < 1.0))
    if len(high_corr[0]) > 0:
        print(f"  ⚠️  {len(high_corr[0])} pairs with |r| > 0.99 (near-collinear)")

# ============================================================================
# DIAGNOSTIC 4: BASELINE CLASSIFIER
# ============================================================================

def diagnose_baseline_classifier(features, labels):
    """Test if simple classifier can separate all 3 classes"""
    print("\n" + "="*70)
    print("DIAGNOSTIC 4: BASELINE CLASSIFIER")
    print("="*70)
    
    # Split data
    n_train = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    
    print(f"\nDataset: {len(X_train)} train, {len(X_val)} val")
    print(f"Train labels: EW={( y_train==0).sum()}, KPZ={(y_train==1).sum()}, BD={(y_train==2).sum()}")
    print(f"Val labels:   EW={( y_val==0).sum()}, KPZ={(y_val==1).sum()}, BD={(y_val==2).sum()}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train simple logistic regression
    print("\nTraining logistic regression (max_iter=1000)...")
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train_scaled, y_train)
    
    train_acc = clf.score(X_train_scaled, y_train)
    val_acc = clf.score(X_val_scaled, y_val)
    
    print(f"\nResults:")
    print(f"  Train accuracy: {100*train_acc:.2f}%")
    print(f"  Val accuracy:   {100*val_acc:.2f}%")
    
    # Confusion matrix
    y_pred = clf.predict(X_val_scaled)
    cm = confusion_matrix(y_val, y_pred)
    
    print("\nConfusion Matrix:")
    print("       EW  KPZ   BD")
    for i, row_name in enumerate(['EW', 'KPZ', 'BD']):
        print(f"{row_name:3s} {cm[i]}")
    
    # Check if KPZ is being predicted
    kpz_predicted = (y_pred == 1).sum()
    print(f"\nKPZ predictions: {kpz_predicted} / {len(y_val)} ({100*kpz_predicted/len(y_val):.1f}%)")
    
    if kpz_predicted == 0:
        print("⚠️  BASELINE NEVER PREDICTS KPZ - Label or dataset bug likely!")
    
    # Per-class report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['EW', 'KPZ', 'BD'], digits=3))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['EW', 'KPZ', 'BD'])
    ax.set_yticklabels(['EW', 'KPZ', 'BD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Baseline Logistic Regression\nVal Acc: {100*val_acc:.1f}%')
    
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig, val_acc, cm

# ============================================================================
# MAIN DIAGNOSTIC RUN
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 49b: DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    output_dir = Path('results/exp49b_diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    features, labels = load_exp49_data()
    
    if features is None:
        print("\n⚠️  No cached data, generating minimal dataset...")
        simulator = GrowthModelSimulator(width=256, height=500)
        
        all_features = []
        all_labels = []
        
        models = [
            ('edwards_wilkinson', 0, 'EW', {'diffusion': 1.0, 'noise_strength': 1.0, 'dt': 0.1}),
            ('kpz_equation', 1, 'KPZ', {'diffusion': 1.0, 'nonlinearity': 1.0, 'noise_strength': 1.0, 'dt': 0.05}),
            ('ballistic_deposition', 2, 'BD', {'noise_strength': 0.2})
        ]
        
        for model_type, label, name, params in models:
            print(f"\nGenerating {name} (5 trajectories)...")
            for _ in range(5):
                traj = simulator.generate_trajectory(model_type=model_type, **params)
                
                for t_idx in range(traj.shape[0] // 5, traj.shape[0]):
                    h = traj[t_idx]
                    grad = np.gradient(h)
                    features_sample = np.array([
                        np.mean(grad**2),
                        np.mean(grad**3),
                        np.mean(grad**4),
                        np.mean(grad**5),
                        np.mean(grad**6),
                        np.mean(np.abs(grad)**7)
                    ])
                    all_features.append(features_sample)
                    all_labels.append(label)
        
        features = np.array(all_features)
        labels = np.array(all_labels)
        
        print(f"\nGenerated {len(features)} samples")
    
    # Run diagnostics
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTICS")
    print("="*70)
    
    # 1. Covariance stability
    cov_diag = diagnose_covariance_stability(features, labels)
    
    # 2. Gaussianity
    gauss_fig = diagnose_gaussianity(features, labels)
    gauss_fig.savefig(output_dir / 'diagnostic_gaussianity.png', dpi=150, bbox_inches='tight')
    
    # 3. Training data sanity
    diagnose_training_data(features, labels)
    
    # 4. Baseline classifier
    baseline_fig, baseline_acc, baseline_cm = diagnose_baseline_classifier(features, labels)
    baseline_fig.savefig(output_dir / 'diagnostic_baseline.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print(f"\n1. Covariance stability: Check printed eigenvalues above")
    print(f"2. Gaussianity: See {output_dir / 'diagnostic_gaussianity.png'}")
    print(f"3. Training data: Check printed statistics above")
    print(f"4. Baseline classifier: {100*baseline_acc:.1f}% accuracy")
    
    if baseline_acc < 0.5:
        print("\n⚠️  BASELINE POOR: Data may not be separable or has serious issues")
    elif baseline_acc > 0.8:
        print("\n✅ BASELINE GOOD: Data is separable, Exp 49 training issue likely")
    else:
        print("\n⚠️  BASELINE WEAK: Data partially separable, mixed issues")
    
    # Save diagnostics
    diag_results = {
        'covariance_diagnostics': cov_diag,
        'baseline_accuracy': baseline_acc,
        'baseline_confusion': baseline_cm,
        'features': features,
        'labels': labels
    }
    
    with open(output_dir / 'diagnostics.pkl', 'wb') as f:
        pickle.dump(diag_results, f)
    
    print(f"\nDiagnostics saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
