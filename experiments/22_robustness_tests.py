"""
EXPERIMENT 22: Robustness Tests for Universality Coordinates
============================================================

ChatGPT identified three critical tests to convince skeptics:

TEST 1: Generalization to untrained KPZ-class models
  - Project BD, Eden onto SAME PCA (fit only on EW/KPZ)
  - Do they land on the KPZ side without retraining?

TEST 2: Nuisance parameter invariance
  - Sweep noise strength, discretization
  - Show class separation survives, nuisance moves orthogonally

TEST 3: Coordinate-free verification
  - Logistic regression AUC on raw features
  - Wasserstein distance in feature space
  - Confirm it's not a PCA artifact

THE KEY STATEMENT WE'RE DEFENDING:
"Universality can be detected as a geometric coordinate in a low-dimensional 
space of RG-relevant local observables."
"""

import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_dir)

from simulation.physics_simulation import GrowthModelSimulator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION (consistent with Exp 21)
# ============================================================================

def extract_gradient_features(surface):
    """Extract gradient-based feature vector from a single surface."""
    h = surface
    L = len(h)
    
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
    lap = np.roll(h, -1) + np.roll(h, 1) - 2*h
    
    grad_var = np.var(grad)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)
    
    if grad_std > 1e-10:
        grad_skew = np.mean((grad - grad_mean)**3) / grad_std**3
        grad_kurt = np.mean((grad - grad_mean)**4) / grad_std**4 - 3
    else:
        grad_skew = 0
        grad_kurt = 0
    
    lap_var = np.var(lap)
    grad_lap_cov = np.mean((grad - np.mean(grad)) * (lap - np.mean(lap)))
    h_var = np.var(h - np.mean(h))
    
    return np.array([grad_var, grad_skew, grad_kurt, lap_var, grad_lap_cov, h_var])


def generate_surfaces(model_type, n_surfaces, L=128, T=1000, seed_offset=0):
    """Generate surfaces from specified model."""
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    surfaces = []
    
    for i in range(n_surfaces):
        np.random.seed(seed_offset + i)
        traj = simulator.generate_trajectory(model_type)
        surfaces.append(traj[-1].copy())
    
    return surfaces


# ============================================================================
# TEST 1: GENERALIZATION TO UNTRAINED MODELS
# ============================================================================

def test_generalization(output_dir):
    """
    Test: Do other KPZ-class models (BD, Eden) land on the KPZ side
    when projected onto PCA fit ONLY on EW/KPZ?
    """
    print("\n" + "=" * 70)
    print("TEST 1: Generalization to Untrained Models")
    print("=" * 70)
    
    n_train = 200
    n_test = 100
    
    # Generate training data (EW and KPZ only)
    print("  Generating EW/KPZ for PCA fitting...")
    ew_surfaces = generate_surfaces('edwards_wilkinson', n_train, seed_offset=0)
    kpz_surfaces = generate_surfaces('kpz_equation', n_train, seed_offset=1000)
    
    ew_features = np.array([extract_gradient_features(s) for s in ew_surfaces])
    kpz_features = np.array([extract_gradient_features(s) for s in kpz_surfaces])
    
    # Fit PCA on EW + KPZ only
    X_train = np.vstack([ew_features, kpz_features])
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1
    X_train_norm = (X_train - X_mean) / X_std
    
    pca = PCA(n_components=2)
    pca.fit(X_train_norm)
    
    ew_pca = pca.transform((ew_features - X_mean) / X_std)
    kpz_pca = pca.transform((kpz_features - X_mean) / X_std)
    
    # Generate TEST data: BD and Eden (never seen by PCA)
    print("  Generating BD/Eden for projection...")
    bd_surfaces = generate_surfaces('ballistic_deposition', n_test, seed_offset=2000)
    eden_surfaces = generate_surfaces('eden', n_test, seed_offset=3000)
    
    bd_features = np.array([extract_gradient_features(s) for s in bd_surfaces])
    eden_features = np.array([extract_gradient_features(s) for s in eden_surfaces])
    
    # Project onto SAME PCA
    bd_pca = pca.transform((bd_features - X_mean) / X_std)
    eden_pca = pca.transform((eden_features - X_mean) / X_std)
    
    # Analysis: Do BD/Eden land on KPZ side?
    ew_pc1_mean = np.mean(ew_pca[:, 0])
    kpz_pc1_mean = np.mean(kpz_pca[:, 0])
    bd_pc1_mean = np.mean(bd_pca[:, 0])
    eden_pc1_mean = np.mean(eden_pca[:, 0])
    
    # Decision boundary (midpoint)
    boundary = (ew_pc1_mean + kpz_pc1_mean) / 2
    
    print(f"\n  PC1 means:")
    print(f"    EW:   {ew_pc1_mean:>8.3f}")
    print(f"    KPZ:  {kpz_pc1_mean:>8.3f}")
    print(f"    BD:   {bd_pc1_mean:>8.3f}")
    print(f"    Eden: {eden_pc1_mean:>8.3f}")
    print(f"    Boundary: {boundary:.3f}")
    
    # Classification accuracy (which side of boundary)
    bd_on_kpz_side = np.mean((bd_pca[:, 0] > boundary) == (kpz_pc1_mean > boundary))
    eden_on_kpz_side = np.mean((eden_pca[:, 0] > boundary) == (kpz_pc1_mean > boundary))
    
    print(f"\n  Fraction landing on KPZ side:")
    print(f"    BD:   {bd_on_kpz_side:.1%}")
    print(f"    Eden: {eden_on_kpz_side:.1%}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(ew_pca[:, 0], ew_pca[:, 1], c='blue', alpha=0.5, s=30, label='EW (train)')
    ax.scatter(kpz_pca[:, 0], kpz_pca[:, 1], c='red', alpha=0.5, s=30, label='KPZ (train)')
    ax.scatter(bd_pca[:, 0], bd_pca[:, 1], c='orange', alpha=0.7, s=50, marker='s', label='BD (test)')
    ax.scatter(eden_pca[:, 0], eden_pca[:, 1], c='green', alpha=0.7, s=50, marker='^', label='Eden (test)')
    
    ax.axvline(boundary, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
    
    ax.set_xlabel('PC1 (Universality Axis)')
    ax.set_ylabel('PC2')
    ax.set_title('TEST 1: Do BD/Eden land on KPZ side?\n(PCA fit only on EW+KPZ)')
    ax.legend()
    
    # Add result text
    result = "✓ PASS" if (bd_on_kpz_side > 0.9 and eden_on_kpz_side > 0.9) else "? PARTIAL" if (bd_on_kpz_side > 0.5 or eden_on_kpz_side > 0.5) else "✗ FAIL"
    ax.text(0.02, 0.98, f'BD on KPZ side: {bd_on_kpz_side:.0%}\nEden on KPZ side: {eden_on_kpz_side:.0%}\n{result}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    fig_path = output_dir / 'test1_generalization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.savefig(str(fig_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")
    
    return {
        'bd_on_kpz_side': bd_on_kpz_side,
        'eden_on_kpz_side': eden_on_kpz_side,
        'pass': bd_on_kpz_side > 0.9 and eden_on_kpz_side > 0.9
    }


# ============================================================================
# TEST 2: NUISANCE PARAMETER INVARIANCE
# ============================================================================

def test_nuisance_invariance(output_dir):
    """
    Test: Does class separation survive nuisance parameter changes?
    Sweep: system size L, evolution time T
    """
    print("\n" + "=" * 70)
    print("TEST 2: Nuisance Parameter Invariance")
    print("=" * 70)
    
    n_per_condition = 50
    
    # Nuisance parameter ranges
    L_values = [64, 128, 256, 512]
    T_values = [500, 1000, 2000]
    
    all_data = []
    
    print("  Generating surfaces across parameter sweep...")
    
    for L in L_values:
        for T in T_values:
            for model in ['edwards_wilkinson', 'kpz_equation']:
                seed_base = L * 1000 + T + (0 if model == 'edwards_wilkinson' else 10000)
                surfaces = generate_surfaces(model, n_per_condition, L=L, T=T, seed_offset=seed_base)
                
                for s in surfaces:
                    features = extract_gradient_features(s)
                    all_data.append({
                        'model': 'EW' if model == 'edwards_wilkinson' else 'KPZ',
                        'L': L,
                        'T': T,
                        'features': features
                    })
    
    print(f"  Total samples: {len(all_data)}")
    
    # Fit PCA on all data
    X = np.array([d['features'] for d in all_data])
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-10] = 1
    X_norm = (X - X_mean) / X_std
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_norm)
    
    # Compute class separation for each (L, T) condition
    print("\n  Class separation (PC1 mean difference) by condition:")
    print(f"  {'L':>5} {'T':>5} {'|ΔPC1|':>8} {'Separable?':>12}")
    print("  " + "-" * 35)
    
    separations = []
    for L in L_values:
        for T in T_values:
            ew_mask = np.array([(d['model'] == 'EW' and d['L'] == L and d['T'] == T) for d in all_data])
            kpz_mask = np.array([(d['model'] == 'KPZ' and d['L'] == L and d['T'] == T) for d in all_data])
            
            if np.sum(ew_mask) > 0 and np.sum(kpz_mask) > 0:
                ew_pc1 = coords[ew_mask, 0]
                kpz_pc1 = coords[kpz_mask, 0]
                
                sep = abs(np.mean(kpz_pc1) - np.mean(ew_pc1))
                separations.append(sep)
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(ew_pc1) + np.var(kpz_pc1)) / 2)
                cohens_d = sep / pooled_std if pooled_std > 0 else 0
                
                status = "✓" if cohens_d > 2 else "~" if cohens_d > 1 else "✗"
                print(f"  {L:>5} {T:>5} {sep:>8.2f} {status:>12} (d={cohens_d:.1f})")
    
    # Invariance test: is separation consistent?
    sep_mean = np.mean(separations)
    sep_std = np.std(separations)
    cv = sep_std / sep_mean if sep_mean > 0 else np.inf
    
    print(f"\n  Separation across conditions: {sep_mean:.2f} ± {sep_std:.2f} (CV={cv:.2f})")
    
    # Create figure: separation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: All points colored by model
    ax = axes[0]
    models = np.array([d['model'] for d in all_data])
    colors = ['blue' if m == 'EW' else 'red' for m in models]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, alpha=0.3, s=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('All conditions (blue=EW, red=KPZ)')
    
    # Panel B: Separation by L (aggregate over T)
    ax = axes[1]
    for L in L_values:
        ew_mask = np.array([(d['model'] == 'EW' and d['L'] == L) for d in all_data])
        kpz_mask = np.array([(d['model'] == 'KPZ' and d['L'] == L) for d in all_data])
        
        ew_pc1_mean = np.mean(coords[ew_mask, 0])
        kpz_pc1_mean = np.mean(coords[kpz_mask, 0])
        
        ax.scatter(L, ew_pc1_mean, c='blue', s=100, marker='o')
        ax.scatter(L, kpz_pc1_mean, c='red', s=100, marker='^')
        ax.plot([L, L], [ew_pc1_mean, kpz_pc1_mean], 'k-', alpha=0.5)
    
    ax.set_xlabel('System Size L')
    ax.set_ylabel('PC1 Mean')
    ax.set_title('Class separation vs system size')
    ax.legend(['EW', 'KPZ'], loc='upper right')
    
    plt.tight_layout()
    fig_path = output_dir / 'test2_nuisance_invariance.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")
    
    return {
        'separation_mean': sep_mean,
        'separation_std': sep_std,
        'cv': cv,
        'pass': cv < 0.3  # Low coefficient of variation = invariant
    }


# ============================================================================
# TEST 3: COORDINATE-FREE VERIFICATION
# ============================================================================

def test_coordinate_free(output_dir):
    """
    Test: Is separation a PCA artifact or real?
    - Logistic regression AUC on raw features
    - Wasserstein distance in each feature dimension
    """
    print("\n" + "=" * 70)
    print("TEST 3: Coordinate-Free Verification")
    print("=" * 70)
    
    n_samples = 300
    
    # Generate data
    print("  Generating surfaces...")
    ew_surfaces = generate_surfaces('edwards_wilkinson', n_samples, seed_offset=5000)
    kpz_surfaces = generate_surfaces('kpz_equation', n_samples, seed_offset=6000)
    
    ew_features = np.array([extract_gradient_features(s) for s in ew_surfaces])
    kpz_features = np.array([extract_gradient_features(s) for s in kpz_surfaces])
    
    X = np.vstack([ew_features, kpz_features])
    y = np.array([0] * n_samples + [1] * n_samples)
    
    # Standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std < 1e-10] = 1
    X_norm = (X - X_mean) / X_std
    
    # Test A: Logistic Regression
    print("\n  A) Logistic Regression (5-fold CV):")
    
    clf = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(clf, X_norm, y, cv=5, scoring='accuracy')
    
    clf.fit(X_norm, y)
    y_prob = clf.predict_proba(X_norm)[:, 1]
    auc = roc_auc_score(y, y_prob)
    
    print(f"    Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print(f"    AUC: {auc:.3f}")
    
    # Feature importance from logistic regression
    feature_names = ['grad_var', 'grad_skew', 'grad_kurt', 'lap_var', 'grad_lap_cov', 'h_var']
    print(f"\n    Logistic Regression Coefficients:")
    for i, name in enumerate(feature_names):
        print(f"      {name:<15}: {clf.coef_[0, i]:>8.3f}")
    
    # Test B: Wasserstein distances per feature
    print("\n  B) Wasserstein Distance per Feature:")
    
    w_distances = []
    for i, name in enumerate(feature_names):
        w = wasserstein_distance(ew_features[:, i], kpz_features[:, i])
        w_distances.append(w)
        print(f"    {name:<15}: W₁ = {w:.4f}")
    
    # Test C: Combined Wasserstein in normalized space
    # Use sum of marginal Wassersteins as proxy
    total_w = sum(w_distances)
    print(f"\n    Total (sum of marginals): {total_w:.4f}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Panel A: Feature importance
    ax = axes[0]
    importance = np.abs(clf.coef_[0])
    bars = ax.barh(feature_names, importance)
    ax.set_xlabel('|Coefficient|')
    ax.set_title('Logistic Regression Feature Importance')
    
    # Highlight top features
    max_idx = np.argmax(importance)
    bars[max_idx].set_color('red')
    
    # Panel B: Wasserstein distances
    ax = axes[1]
    bars = ax.barh(feature_names, w_distances)
    ax.set_xlabel('Wasserstein-1 Distance')
    ax.set_title('Feature-wise EW-KPZ Separation')
    
    max_idx = np.argmax(w_distances)
    bars[max_idx].set_color('red')
    
    # Panel C: ROC-like visualization
    ax = axes[2]
    # Sort by predicted probability
    sorted_idx = np.argsort(y_prob)
    ax.scatter(range(len(y)), y[sorted_idx], c=y_prob[sorted_idx], cmap='RdBu_r', alpha=0.5, s=10)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Sample (sorted by P(KPZ))')
    ax.set_ylabel('True Label (0=EW, 1=KPZ)')
    ax.set_title(f'Classification (AUC={auc:.3f})')
    
    plt.tight_layout()
    fig_path = output_dir / 'test3_coordinate_free.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {fig_path}")
    
    return {
        'accuracy': np.mean(cv_scores),
        'auc': auc,
        'top_feature': feature_names[np.argmax(importance)],
        'wasserstein_distances': dict(zip(feature_names, w_distances)),
        'pass': auc > 0.99
    }


# ============================================================================
# MAIN
# ============================================================================

def run_experiment():
    """Run all three robustness tests."""
    
    print("=" * 70)
    print("EXPERIMENT 22: Robustness Tests for Universality Coordinates")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = Path(__file__).parent.parent / 'results' / 'exp22_robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run tests
    results['test1'] = test_generalization(output_dir)
    results['test2'] = test_nuisance_invariance(output_dir)
    results['test3'] = test_coordinate_free(output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n  Test 1 (Generalization):     {'✓ PASS' if results['test1']['pass'] else '✗ FAIL'}")
    print(f"    BD on KPZ side: {results['test1']['bd_on_kpz_side']:.0%}")
    print(f"    Eden on KPZ side: {results['test1']['eden_on_kpz_side']:.0%}")
    
    print(f"\n  Test 2 (Nuisance Invariance): {'✓ PASS' if results['test2']['pass'] else '✗ FAIL'}")
    print(f"    Separation CV: {results['test2']['cv']:.2f}")
    
    print(f"\n  Test 3 (Coordinate-Free):    {'✓ PASS' if results['test3']['pass'] else '✗ FAIL'}")
    print(f"    AUC: {results['test3']['auc']:.3f}")
    print(f"    Top feature: {results['test3']['top_feature']}")
    
    all_pass = all(r['pass'] for r in results.values())
    
    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
        print("=" * 70)
        print("""
The defensible statement is now empirically grounded:

  "Universality can be detected as a geometric coordinate in a 
   low-dimensional space of RG-relevant local observables."

This statement survives:
  ✓ Generalization to untrained models (BD, Eden → KPZ side)
  ✓ Nuisance parameter variation (L, T sweep)
  ✓ Coordinate-free verification (Logistic AUC, Wasserstein)
""")
    else:
        print("SOME TESTS FAILED - CLAIMS NEED QUALIFICATION")
        print("=" * 70)
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
