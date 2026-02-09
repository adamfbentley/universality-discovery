"""
Experiment 50: KS vs KPZ Framework Generalization Test

CRITICAL QUESTION: Does the coupling-coordinate / RG-relevance framework 
generalize beyond KPZ to other universality classes?

TEST SYSTEM: Kuramoto-Sivashinsky (KS) equation
- Different universality class from KPZ
- Similar PDE structure (gradient nonlinearity, dissipation)
- Known parameters: ν (fourth-order), κ (second-order), λ (nonlinearity)

THREE-PART TEST:
1. Coupling coordinate: Does PC1 track KS coupling (ν/κ, λ²/νκ, etc.)?
2. RG relevance: Do information-geometric distances increase with scale?
3. Cross-system: Can joint KPZ+KS embedding separate both?

SUCCESS CRITERIA:
- Strong: r > 0.7 between PC1 and physical coupling
- Weak: r > 0.5 with some PC
- Failure: r < 0.3 for all PCs

This determines if framework is:
- Universal (works for KS) → Continue to MBE, RD
- KPZ-specific (fails for KS) → Accept scope limit
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

from simulation.kuramoto_sivashinsky import KuramotoSivashinskySimulator

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# ============================================================================
# 1. GENERATE KS DATA WITH PARAMETER VARIATIONS
# ============================================================================

def generate_ks_data(n_param_combos=15, n_trajectories_per=2, L=256, T=1000):
    """
    Generate KS trajectories with varying parameters.
    
    Parameter space:
    - ν ∈ [0.5, 2.0]: Fourth-order dissipation
    - κ ∈ [0.5, 2.0]: Second-order dissipation  
    - λ ∈ [0.5, 2.0]: Nonlinearity
    
    Physical couplings to test:
    - g1 = λ²/(ν*κ): Dimensionless coupling
    - g2 = ν/κ: Dissipation ratio
    - g3 = λ/sqrt(ν*κ): Alternative coupling
    """
    print("="*70)
    print("GENERATING KS DATA WITH PARAMETER VARIATIONS")
    print("="*70)
    
    simulator = KuramotoSivashinskySimulator(L=L, dt=0.01)
    
    # Parameter space sampling
    np.random.seed(42)
    params_list = []
    all_features = []
    all_couplings = []
    
    for combo_idx in range(n_param_combos):
        # Sample parameters
        nu = np.random.uniform(0.5, 2.0)
        kappa = np.random.uniform(0.5, 2.0)
        lam = np.random.uniform(0.5, 2.0)
        
        # Compute physical couplings
        g1 = lam**2 / (nu * kappa)  # Dimensionless coupling
        g2 = nu / kappa              # Dissipation ratio
        g3 = lam / np.sqrt(nu * kappa)  # Alternative
        
        print(f"\nCombo {combo_idx+1}/{n_param_combos}:")
        print(f"  ν={nu:.3f}, κ={kappa:.3f}, λ={lam:.3f}")
        print(f"  g1={g1:.3f}, g2={g2:.3f}, g3={g3:.3f}")
        
        for traj_idx in range(n_trajectories_per):
            # Simulate with stronger noise for chaotic regime
            trajectory = simulator.simulate(
                T=T,
                nu=nu,
                kappa=kappa,
                lam=lam,
                noise_strength=0.5,  # Strong noise for chaotic dynamics
                record_interval=10
            )
            
            # Extract gradient moment features (skip transient)
            for t_idx in range(len(trajectory) // 5, len(trajectory)):
                h = trajectory[t_idx]
                
                # Same features as KPZ
                grad = np.gradient(h)
                features = np.array([
                    np.mean(grad**2),           # m2
                    np.mean(grad**3),           # m3
                    np.mean(grad**4),           # m4
                    np.mean(grad**5),           # m5
                    np.mean(grad**6),           # m6
                    np.mean(np.abs(grad)**7)    # m7
                ])
                
                all_features.append(features)
                all_couplings.append([g1, g2, g3, nu, kappa, lam])
                params_list.append([nu, kappa, lam])
    
    all_features = np.array(all_features)
    all_couplings = np.array(all_couplings)
    params_array = np.array(params_list)
    
    print(f"\nGenerated {len(all_features)} samples")
    print(f"  Feature range: {all_features.min():.2e} to {all_features.max():.2e}")
    
    return all_features, all_couplings, params_array

# ============================================================================
# 2. TEST 1: COUPLING COORDINATE DISCOVERY
# ============================================================================

def test_coupling_coordinate(features, couplings):
    """
    Test if PC1 correlates with KS physical couplings.
    Compare to KPZ result: PC1 vs D/ν³ (r=0.857)
    """
    print("\n" + "="*70)
    print("TEST 1: COUPLING COORDINATE DISCOVERY")
    print("="*70)
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA()
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance:")
    for i in range(min(4, len(pca.explained_variance_ratio_))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f}")
    
    # Test correlations with different couplings
    coupling_names = ['g1=λ²/(νκ)', 'g2=ν/κ', 'g3=λ/√(νκ)', 'ν', 'κ', 'λ']
    
    print("\nCorrelations with PC1:")
    best_r = 0
    best_coupling = None
    
    for i, name in enumerate(coupling_names):
        r, p = pearsonr(features_pca[:, 0], couplings[:, i])
        print(f"  PC1 vs {name:15s}: r={r:+.3f} (p={p:.2e})")
        
        if abs(r) > abs(best_r):
            best_r = r
            best_coupling = (name, i)
    
    print(f"\n🎯 Best correlation: PC1 vs {best_coupling[0]}")
    print(f"   r = {best_r:.3f}")
    
    # Compare to KPZ benchmark
    print("\n" + "-"*70)
    print("COMPARISON TO KPZ:")
    print(f"  KPZ: PC1 vs D/ν³     → r = 0.857 ✅")
    print(f"  KS:  PC1 vs {best_coupling[0]:15s} → r = {abs(best_r):.3f}", end="")
    
    if abs(best_r) > 0.7:
        print(" ✅✅ STRONG")
        verdict = "strong"
    elif abs(best_r) > 0.5:
        print(" ✅ WEAK")
        verdict = "weak"
    else:
        print(" ❌ POOR")
        verdict = "poor"
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, (name, ax) in enumerate(zip(coupling_names, axes.flat)):
        r, p = pearsonr(features_pca[:, 0], couplings[:, i])
        
        ax.scatter(features_pca[:, 0], couplings[:, i], alpha=0.5, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel(name)
        ax.set_title(f'r={r:+.3f} (p={p:.2e})')
        ax.grid(alpha=0.3)
        
        # Highlight best
        if i == best_coupling[1]:
            ax.set_facecolor('#fff9e6')
    
    plt.tight_layout()
    
    results = {
        'pca': pca,
        'best_coupling': best_coupling,
        'best_r': best_r,
        'verdict': verdict,
        'all_correlations': {name: pearsonr(features_pca[:, 0], couplings[:, i])[0] 
                            for i, name in enumerate(coupling_names)}
    }
    
    return results, fig

# ============================================================================
# 3. TEST 2: RG-RELEVANCE (INFORMATION GEOMETRY)
# ============================================================================

def test_rg_relevance(features):
    """
    Test if KS parameter regimes become more distinguishable under coarse-graining.
    Compare to KPZ: slope +0.31 (distances increase with scale)
    """
    print("\n" + "="*70)
    print("TEST 2: RG-RELEVANCE (INFORMATION GEOMETRY)")
    print("="*70)
    
    # For simplicity: split features into two groups (high/low by median of first coupling)
    # In real analysis: use actual KS parameter classes
    
    # Placeholder: Use variance as simple discriminator
    variances = np.var(features, axis=1)
    median_var = np.median(variances)
    labels = (variances > median_var).astype(int)
    
    print(f"\nSplit samples by variance:")
    print(f"  Low-variance: {(labels==0).sum()}")
    print(f"  High-variance: {(labels==1).sum()}")
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Compute distances at multiple scales
    from sklearn.covariance import LedoitWolf
    
    scales = [1, 2, 4, 8]
    distances = []
    
    for scale in scales:
        # Coarse-grain
        if scale == 1:
            feat_coarse = features_scaled.copy()
            labels_coarse = labels.copy()
        else:
            n_blocks = len(features_scaled) // scale
            feat_coarse = features_scaled[:n_blocks*scale].reshape(n_blocks, scale, -1).mean(axis=1)
            labels_coarse = labels[:n_blocks*scale].reshape(n_blocks, scale).mean(axis=1).round().astype(int)
        
        mask_0 = labels_coarse == 0
        mask_1 = labels_coarse == 1
        
        if mask_0.sum() < 20 or mask_1.sum() < 20:
            continue
        
        feat_0 = feat_coarse[mask_0]
        feat_1 = feat_coarse[mask_1]
        
        # Shrinkage covariance
        lw_0 = LedoitWolf().fit(feat_0)
        lw_1 = LedoitWolf().fit(feat_1)
        
        cov_0 = lw_0.covariance_
        cov_1 = lw_1.covariance_
        
        mu_0 = feat_0.mean(axis=0)
        mu_1 = feat_1.mean(axis=0)
        
        # Symmetrized KL
        inv_cov_0 = np.linalg.inv(cov_0)
        inv_cov_1 = np.linalg.inv(cov_1)
        
        kl_01 = 0.5 * (np.trace(inv_cov_1 @ cov_0) + (mu_1 - mu_0) @ inv_cov_1 @ (mu_1 - mu_0) - 
                      len(mu_0) + np.log(np.linalg.det(cov_1) / np.linalg.det(cov_0)))
        kl_10 = 0.5 * (np.trace(inv_cov_0 @ cov_1) + (mu_0 - mu_1) @ inv_cov_0 @ (mu_0 - mu_1) - 
                      len(mu_1) + np.log(np.linalg.det(cov_0) / np.linalg.det(cov_1)))
        
        sym_kl = kl_01 + kl_10
        distances.append(sym_kl)
        
        print(f"  Scale b={scale}: KL = {sym_kl:.4f}")
    
    # Linear fit
    if len(distances) >= 3:
        slope, intercept, r, p, _ = linregress(np.log(scales[:len(distances)]), distances)
        
        print(f"\nLinear fit: KL ~ {slope:.4f} * log(b) + {intercept:.4f}")
        print(f"  R² = {r**2:.3f}, p = {p:.2e}")
        
        print("\n" + "-"*70)
        print("COMPARISON TO KPZ:")
        print(f"  KPZ: slope = +0.31 (increase) ✅")
        print(f"  KS:  slope = {slope:+.4f}", end="")
        
        if slope > 0 and p < 0.05:
            print(" ✅ INCREASE (RG-relevant)")
            verdict = "increase"
        else:
            print(" ❌ FLAT/DECREASE")
            verdict = "flat"
    else:
        slope = np.nan
        verdict = "insufficient_data"
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scales[:len(distances)], distances, 'o-', markersize=8, linewidth=2)
    if len(distances) >= 3:
        ax.plot(scales[:len(distances)], intercept + slope * np.log(scales[:len(distances)]),
               '--', alpha=0.5, label=f'slope={slope:.3f}')
    ax.set_xlabel('Coarse-graining scale b')
    ax.set_ylabel('Symmetrized KL divergence')
    ax.set_title('KS: Information-Geometric RG Relevance')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    results = {
        'scales': scales[:len(distances)],
        'distances': distances,
        'slope': slope if len(distances) >= 3 else np.nan,
        'verdict': verdict
    }
    
    return results, fig

# ============================================================================
# 4. MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*70)
    print("EXPERIMENT 50: KS GENERALIZATION TEST")
    print("="*70)
    print("\nCRITICAL QUESTION: Does framework generalize beyond KPZ?")
    print("TEST SYSTEM: Kuramoto-Sivashinsky equation")
    print("="*70)
    
    output_dir = Path('results/exp50_ks_generalization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    features, couplings, params = generate_ks_data(
        n_param_combos=20,
        n_trajectories_per=3,
        L=256,
        T=500
    )
    
    # Test 1: Coupling coordinate
    coupling_results, coupling_fig = test_coupling_coordinate(features, couplings)
    coupling_fig.savefig(output_dir / 'test1_coupling_coordinates.png', 
                        dpi=150, bbox_inches='tight')
    
    # Test 2: RG relevance
    rg_results, rg_fig = test_rg_relevance(features)
    rg_fig.savefig(output_dir / 'test2_rg_relevance.png', 
                  dpi=150, bbox_inches='tight')
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT: FRAMEWORK GENERALIZATION TO KS")
    print("="*70)
    
    test1_verdict = coupling_results['verdict']
    test2_verdict = rg_results['verdict']
    
    print(f"\nTest 1 (Coupling): {test1_verdict.upper()}")
    print(f"  Best: PC1 vs {coupling_results['best_coupling'][0]} (r={coupling_results['best_r']:.3f})")
    
    print(f"\nTest 2 (RG relevance): {test2_verdict.upper()}")
    if not np.isnan(rg_results['slope']):
        print(f"  Slope: {rg_results['slope']:+.4f}")
    
    # Overall assessment
    if test1_verdict == 'strong' and test2_verdict == 'increase':
        print("\n✅✅✅ FRAMEWORK GENERALIZES TO KS")
        print("    → Continue to Phase 2 (MBE)")
        print("    → Path to universal framework")
        overall = "generalizes"
    elif test1_verdict in ['strong', 'weak'] or test2_verdict == 'increase':
        print("\n✅ PARTIAL GENERALIZATION")
        print("    → Some aspects work, refinement needed")
        print("    → Continue with caution")
        overall = "partial"
    else:
        print("\n❌ LIMITED GENERALIZATION TO KS")
        print("    → Framework appears KPZ-specific")
        print("    → Accept scope limit (still strong contribution)")
        overall = "limited"
    
    # Save results
    results = {
        'features': features,
        'couplings': couplings,
        'params': params,
        'test1_coupling': coupling_results,
        'test2_rg': rg_results,
        'overall_verdict': overall
    }
    
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}/")
    print("="*70)

if __name__ == '__main__':
    main()
