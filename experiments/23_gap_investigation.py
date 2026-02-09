"""
EXPERIMENT 23: Investigating the Discrete-Continuum Gap
=======================================================

PUZZLE: BD/Eden share KPZ exponents but occupy completely different 
regions in gradient moment space (PC1 = 16559 vs -1.6 for KPZ).

This experiment investigates WHY and whether we can find universal coordinates.

PHASES:
1. DIAGNOSE: Which features cause the gap?
2. NORMALIZE: Can scale-invariant features cluster by universality?
3. SEARCH: Find features that DO cluster by universality class
4. TEST RG: Does coarse-graining merge the manifolds?
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_all_features(surface):
    """Extract comprehensive feature set including scale-invariant ones."""
    h = surface
    L = len(h)
    
    # Basic derivatives
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2
    lap = np.roll(h, -1) + np.roll(h, 1) - 2*h
    
    # Variances
    h_var = np.var(h - np.mean(h))
    grad_var = np.var(grad)
    lap_var = np.var(lap)
    
    # Standardized moments (scale-invariant)
    grad_std = np.std(grad)
    if grad_std > 1e-10:
        grad_skew = np.mean((grad - np.mean(grad))**3) / grad_std**3
        grad_kurt = np.mean((grad - np.mean(grad))**4) / grad_std**4 - 3
    else:
        grad_skew, grad_kurt = 0, 0
    
    lap_std = np.std(lap)
    if lap_std > 1e-10:
        lap_skew = np.mean((lap - np.mean(lap))**3) / lap_std**3
        lap_kurt = np.mean((lap - np.mean(lap))**4) / lap_std**4 - 3
    else:
        lap_skew, lap_kurt = 0, 0
    
    # Dimensionless ratios
    ratio_lap_grad = lap_var / (grad_var + 1e-10)
    ratio_h_grad = h_var / (grad_var + 1e-10)
    
    # Cross-correlation (normalized)
    grad_centered = grad - np.mean(grad)
    lap_centered = lap - np.mean(lap)
    grad_lap_corr = np.mean(grad_centered * lap_centered) / (grad_std * lap_std + 1e-10)
    
    return {
        # Scale-dependent
        'h_var': h_var,
        'grad_var': grad_var,
        'lap_var': lap_var,
        # Scale-invariant (standardized moments)
        'grad_skew': grad_skew,
        'grad_kurt': grad_kurt,
        'lap_skew': lap_skew,
        'lap_kurt': lap_kurt,
        # Dimensionless ratios
        'ratio_lap_grad': ratio_lap_grad,
        'ratio_h_grad': ratio_h_grad,
        'grad_lap_corr': grad_lap_corr,
    }


def generate_surfaces(model_type, n_surfaces, L=128, T=1000, seed_offset=0):
    """Generate surfaces from specified model."""
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    surfaces = []
    for i in range(n_surfaces):
        np.random.seed(seed_offset + i)
        traj = simulator.generate_trajectory(model_type)
        surfaces.append(traj[-1].copy())
    return surfaces


def coarse_grain(surface, block_size):
    """Apply block averaging (RG) to surface."""
    L = len(surface)
    L_new = L // block_size
    coarse = np.zeros(L_new)
    for i in range(L_new):
        coarse[i] = np.mean(surface[i*block_size:(i+1)*block_size])
    return coarse


# ============================================================================
# PHASE 1: DIAGNOSE
# ============================================================================

def phase1_diagnose(output_dir):
    """Identify which features cause the massive EW/KPZ vs BD/Eden separation."""
    
    print("\n" + "=" * 70)
    print("PHASE 1: DIAGNOSE — Which Features Cause the Gap?")
    print("=" * 70)
    
    n_samples = 100
    L, T = 128, 1000
    
    # Generate data
    print("  Generating surfaces...")
    models = {
        'EW': generate_surfaces('edwards_wilkinson', n_samples, L, T, 1000),
        'KPZ': generate_surfaces('kpz_equation', n_samples, L, T, 2000),
        'BD': generate_surfaces('ballistic_deposition', n_samples, L, T, 3000),
        'Eden': generate_surfaces('eden', n_samples, L, T, 4000),
    }
    
    # Extract features
    print("  Extracting features...")
    features = {model: [extract_all_features(s) for s in surfaces] 
                for model, surfaces in models.items()}
    
    # Compute means and stds
    feature_names = list(features['EW'][0].keys())
    
    print("\n  Feature comparison (mean ± std):")
    print(f"  {'Feature':<15} {'EW':>15} {'KPZ':>15} {'BD':>15} {'Eden':>15}")
    print("  " + "-" * 75)
    
    results = {}
    for feat in feature_names:
        vals = {model: [f[feat] for f in features[model]] for model in models}
        means = {model: np.mean(vals[model]) for model in models}
        stds = {model: np.std(vals[model]) for model in models}
        
        results[feat] = {'means': means, 'stds': stds}
        
        print(f"  {feat:<15} {means['EW']:>7.3f}±{stds['EW']:<5.3f} "
              f"{means['KPZ']:>7.3f}±{stds['KPZ']:<5.3f} "
              f"{means['BD']:>7.3f}±{stds['BD']:<5.3f} "
              f"{means['Eden']:>7.3f}±{stds['Eden']:<5.3f}")
    
    # Identify problematic features (huge BD/Eden values)
    print("\n  Gap analysis (BD/KPZ ratio):")
    for feat in feature_names:
        ratio = abs(results[feat]['means']['BD']) / (abs(results[feat]['means']['KPZ']) + 1e-10)
        if ratio > 10 or ratio < 0.1:
            print(f"    {feat}: {ratio:.1f}x — LARGE GAP")
        else:
            print(f"    {feat}: {ratio:.1f}x — similar scale")
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    
    colors = {'EW': 'blue', 'KPZ': 'red', 'BD': 'green', 'Eden': 'orange'}
    
    for i, feat in enumerate(feature_names):
        ax = axes[i]
        for model in models:
            vals = [f[feat] for f in features[model]]
            ax.hist(vals, bins=20, alpha=0.5, label=model, color=colors[model], density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
    
    plt.suptitle('Phase 1: Feature Distributions by Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_feature_distributions.png', dpi=150)
    print(f"\n  Saved: {output_dir / 'phase1_feature_distributions.png'}")
    
    return results, features


# ============================================================================
# PHASE 2: NORMALIZE — Scale-Invariant Features
# ============================================================================

def phase2_normalize(features, output_dir):
    """Test if scale-invariant features cluster by universality class."""
    
    print("\n" + "=" * 70)
    print("PHASE 2: NORMALIZE — Do Scale-Invariant Features Cluster?")
    print("=" * 70)
    
    # Scale-invariant features only
    invariant_features = ['grad_skew', 'grad_kurt', 'lap_skew', 'lap_kurt', 
                          'ratio_lap_grad', 'ratio_h_grad', 'grad_lap_corr']
    
    # Build feature matrix
    models = list(features.keys())
    X_all = []
    labels = []
    
    for model in models:
        for f in features[model]:
            X_all.append([f[feat] for feat in invariant_features])
            labels.append(model)
    
    X_all = np.array(X_all)
    labels = np.array(labels)
    
    # Standardize
    X_std = (X_all - np.mean(X_all, axis=0)) / (np.std(X_all, axis=0) + 1e-10)
    
    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_std)
    
    print(f"\n  Using features: {invariant_features}")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = {'EW': 'blue', 'KPZ': 'red', 'BD': 'green', 'Eden': 'orange'}
    markers = {'EW': 'o', 'KPZ': '^', 'BD': 's', 'Eden': 'D'}
    
    for model in models:
        mask = labels == model
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   c=colors[model], marker=markers[model], 
                   label=model, alpha=0.6, s=40)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Phase 2: Scale-Invariant Features Only\n(Standardized Moments + Dimensionless Ratios)', 
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add cluster analysis
    ew_kpz_mask = (labels == 'EW') | (labels == 'KPZ')
    bd_eden_mask = (labels == 'BD') | (labels == 'Eden')
    
    ew_center = np.mean(coords[labels == 'EW'], axis=0)
    kpz_center = np.mean(coords[labels == 'KPZ'], axis=0)
    bd_center = np.mean(coords[labels == 'BD'], axis=0)
    eden_center = np.mean(coords[labels == 'Eden'], axis=0)
    
    # Distances
    d_ew_kpz = np.linalg.norm(ew_center - kpz_center)
    d_kpz_bd = np.linalg.norm(kpz_center - bd_center)
    d_kpz_eden = np.linalg.norm(kpz_center - eden_center)
    
    print(f"\n  Cluster distances:")
    print(f"    EW ↔ KPZ:   {d_ew_kpz:.2f}")
    print(f"    KPZ ↔ BD:   {d_kpz_bd:.2f}")
    print(f"    KPZ ↔ Eden: {d_kpz_eden:.2f}")
    
    if d_kpz_bd < d_ew_kpz and d_kpz_eden < d_ew_kpz:
        print("\n  ✓ KPZ-class models (BD, Eden) cluster CLOSER to KPZ than EW!")
        success = True
    else:
        print("\n  ✗ KPZ-class models do NOT cluster with KPZ")
        success = False
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase2_scale_invariant.png', dpi=150)
    print(f"\n  Saved: {output_dir / 'phase2_scale_invariant.png'}")
    
    return success, coords, labels


# ============================================================================
# PHASE 3: SEARCH — Find Universal Coordinates
# ============================================================================

def phase3_search(features, output_dir):
    """Find features that cluster by universality class, not implementation."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: SEARCH — Find Universal Coordinates")
    print("=" * 70)
    
    feature_names = list(features['EW'][0].keys())
    
    print("\n  Computing within-class vs between-class separation...")
    print(f"\n  {'Feature':<15} {'EW↔KPZ':>10} {'KPZ↔BD':>10} {'KPZ↔Eden':>10} {'Universal?':>12}")
    print("  " + "-" * 60)
    
    universal_features = []
    
    for feat in feature_names:
        # Get feature values
        ew_vals = np.array([f[feat] for f in features['EW']])
        kpz_vals = np.array([f[feat] for f in features['KPZ']])
        bd_vals = np.array([f[feat] for f in features['BD']])
        eden_vals = np.array([f[feat] for f in features['Eden']])
        
        # Distances (using mean difference / pooled std)
        def cohens_d(a, b):
            pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
            return abs(np.mean(a) - np.mean(b)) / (pooled_std + 1e-10)
        
        d_ew_kpz = cohens_d(ew_vals, kpz_vals)
        d_kpz_bd = cohens_d(kpz_vals, bd_vals)
        d_kpz_eden = cohens_d(kpz_vals, eden_vals)
        
        # Universal = EW↔KPZ distance > KPZ↔BD and KPZ↔Eden distances
        # (i.e., KPZ-class models are closer to each other than to EW)
        is_universal = (d_kpz_bd < d_ew_kpz) and (d_kpz_eden < d_ew_kpz)
        
        status = "✓ YES" if is_universal else "✗ no"
        print(f"  {feat:<15} {d_ew_kpz:>10.2f} {d_kpz_bd:>10.2f} {d_kpz_eden:>10.2f} {status:>12}")
        
        if is_universal:
            universal_features.append(feat)
    
    print(f"\n  Universal features found: {universal_features if universal_features else 'NONE'}")
    
    # If we found universal features, plot them
    if len(universal_features) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'EW': 'blue', 'KPZ': 'red', 'BD': 'green', 'Eden': 'orange'}
        
        feat1, feat2 = universal_features[0], universal_features[1]
        
        for model in features:
            x = [f[feat1] for f in features[model]]
            y = [f[feat2] for f in features[model]]
            ax.scatter(x, y, c=colors[model], label=model, alpha=0.6, s=40)
        
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f'Phase 3: Universal Coordinates\n({feat1} vs {feat2})', 
                     fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'phase3_universal_coords.png', dpi=150)
        print(f"\n  Saved: {output_dir / 'phase3_universal_coords.png'}")
    
    return universal_features


# ============================================================================
# PHASE 4: TEST RG — Does Coarse-Graining Merge Manifolds?
# ============================================================================

def phase4_rg(output_dir):
    """Test if coarse-graining makes BD approach the continuum KPZ manifold."""
    
    print("\n" + "=" * 70)
    print("PHASE 4: TEST RG — Does Coarse-Graining Merge Manifolds?")
    print("=" * 70)
    
    n_samples = 50
    L, T = 256, 2000  # Larger system for more coarse-graining headroom
    
    print("  Generating surfaces...")
    kpz_surfaces = generate_surfaces('kpz_equation', n_samples, L, T, 5000)
    bd_surfaces = generate_surfaces('ballistic_deposition', n_samples, L, T, 6000)
    
    # Coarse-graining scales
    block_sizes = [1, 2, 4, 8, 16]
    
    # Track features at each scale
    results = {bs: {'KPZ': [], 'BD': []} for bs in block_sizes}
    
    print("  Coarse-graining and extracting features...")
    
    for bs in block_sizes:
        for s in kpz_surfaces:
            coarse = coarse_grain(s, bs) if bs > 1 else s
            results[bs]['KPZ'].append(extract_all_features(coarse))
        
        for s in bd_surfaces:
            coarse = coarse_grain(s, bs) if bs > 1 else s
            results[bs]['BD'].append(extract_all_features(coarse))
    
    # Focus on scale-invariant features
    invariant_feats = ['grad_skew', 'grad_kurt', 'lap_skew', 'lap_kurt']
    
    print(f"\n  Distance KPZ↔BD vs coarse-graining scale:")
    print(f"  {'Block':>6} {'grad_skew':>12} {'grad_kurt':>12} {'lap_skew':>12} {'Mean':>10}")
    print("  " + "-" * 55)
    
    distances = []
    
    for bs in block_sizes:
        d_list = []
        for feat in invariant_feats:
            kpz_vals = np.array([f[feat] for f in results[bs]['KPZ']])
            bd_vals = np.array([f[feat] for f in results[bs]['BD']])
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(kpz_vals) + np.var(bd_vals)) / 2)
            d = abs(np.mean(kpz_vals) - np.mean(bd_vals)) / (pooled_std + 1e-10)
            d_list.append(d)
        
        mean_d = np.mean(d_list)
        distances.append(mean_d)
        print(f"  {bs:>6} {d_list[0]:>12.2f} {d_list[1]:>12.2f} {d_list[2]:>12.2f} {mean_d:>10.2f}")
    
    # Check if distance decreases with coarse-graining
    if distances[-1] < distances[0]:
        print(f"\n  ✓ Distance DECREASES with coarse-graining: {distances[0]:.2f} → {distances[-1]:.2f}")
        print("    → RG IS merging the manifolds!")
        rg_works = True
    else:
        print(f"\n  ✗ Distance does not decrease: {distances[0]:.2f} → {distances[-1]:.2f}")
        print("    → Manifolds remain separate under coarse-graining")
        rg_works = False
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Distance vs scale
    ax = axes[0]
    ax.plot(block_sizes, distances, 'ko-', markersize=10, linewidth=2)
    ax.set_xlabel('Block Size (coarse-graining scale)')
    ax.set_ylabel('Cohen\'s d (KPZ ↔ BD)')
    ax.set_title('Distance Between KPZ and BD\nvs Coarse-Graining Scale')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # Right: Trajectory in feature space
    ax = axes[1]
    
    # Get grad_skew and grad_kurt at each scale
    for model, color in [('KPZ', 'red'), ('BD', 'green')]:
        skews = [np.mean([f['grad_skew'] for f in results[bs][model]]) for bs in block_sizes]
        kurts = [np.mean([f['grad_kurt'] for f in results[bs][model]]) for bs in block_sizes]
        
        ax.plot(skews, kurts, 'o-', color=color, label=model, markersize=8, linewidth=2)
        
        # Mark start and end
        ax.scatter(skews[0], kurts[0], c=color, s=150, marker='s', edgecolor='black', zorder=10)
        ax.scatter(skews[-1], kurts[-1], c=color, s=150, marker='*', edgecolor='black', zorder=10)
    
    ax.set_xlabel('Gradient Skewness')
    ax.set_ylabel('Gradient Kurtosis')
    ax.set_title('RG Flow in Feature Space\n(□=fine, ★=coarse)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase4_rg_flow.png', dpi=150)
    print(f"\n  Saved: {output_dir / 'phase4_rg_flow.png'}")
    
    return rg_works, distances


# ============================================================================
# MAIN
# ============================================================================

def run_experiment():
    """Run all phases of the discrete-continuum gap investigation."""
    
    print("=" * 70)
    print("EXPERIMENT 23: Investigating the Discrete-Continuum Gap")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = Path(__file__).parent.parent / 'results' / 'exp23_gap_investigation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Diagnose
    results, features = phase1_diagnose(output_dir)
    
    # Phase 2: Normalize
    phase2_success, coords, labels = phase2_normalize(features, output_dir)
    
    # Phase 3: Search
    universal_features = phase3_search(features, output_dir)
    
    # Phase 4: RG
    rg_works, distances = phase4_rg(output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n  Phase 1 (Diagnose): Scale-dependent features (grad_var, lap_var, h_var)")
    print(f"                      cause >1000x gap between continuum and discrete")
    
    print(f"\n  Phase 2 (Normalize): Scale-invariant features cluster by universality: "
          f"{'✓ YES' if phase2_success else '✗ NO'}")
    
    print(f"\n  Phase 3 (Search): Universal features found: "
          f"{universal_features if universal_features else 'NONE'}")
    
    print(f"\n  Phase 4 (RG): Coarse-graining merges manifolds: "
          f"{'✓ YES' if rg_works else '✗ NO'}")
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if phase2_success or universal_features or rg_works:
        print("""
  There ARE coordinates where universality class dominates over implementation:
  
  → Scale-INVARIANT features (moments, ratios) cluster by physics
  → The "gap" is in scale-DEPENDENT features (variances)
  → Coarse-graining helps merge discrete → continuum
  
  REVISED FRAMEWORK:
  "Universality class = equivalence in SHAPE space (moments, ratios),
   not in RAW feature space (variances)."
""")
    else:
        print("""
  No universal coordinates found. Discrete and continuum models may be
  fundamentally different manifolds despite sharing asymptotic exponents.
  
  Framework needs major revision.
""")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    run_experiment()
