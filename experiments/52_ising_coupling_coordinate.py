"""
Experiment 52: Ising Model Coupling Coordinate Discovery

QUESTION: Does unsupervised PCA on local observables recover the RG-relevant 
coupling (reduced temperature) in the 2D Ising model?

If PC1 correlates strongly with t = (T - T_c) / T_c, this validates that
the coupling coordinate discovery generalizes beyond KPZ.

PHYSICS:
- 2D Ising: H = -J Σ_{<ij>} s_i s_j
- Critical temperature: T_c = 2J / ln(1 + √2) ≈ 2.269J
- Near T_c, relevant coupling is reduced temperature t
- RG flow: t → b^{1/ν} t where ν = 1 (2D Ising)

OBSERVABLES (analogous to gradient moments for KPZ):
- Magnetization moments: ⟨|m|^k⟩ for k = 1,2,3,4
- Local gradient moments: ⟨|∇m|^k⟩ (discrete gradient of spin field)
- Domain boundary density: fraction of unlike neighbor pairs
- Correlation length proxy: integrated autocorrelation

SUCCESS CRITERION:
If PC1 vs reduced temperature t shows r > 0.8, the approach generalizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import sys
import json
from datetime import datetime

# Parameters
L = 64  # Lattice size
N_TEMPS = 15  # Number of temperatures
N_SAMPLES = 50  # Samples per temperature
N_EQUILIBRATE = 5000  # MC sweeps for equilibration
N_MEASURE = 1000  # MC sweeps between measurements
N_MEASUREMENTS = 10  # Measurements per sample

# Critical temperature
J = 1.0
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269

print(f"2D Ising critical temperature: T_c = {T_C:.4f}")

# =============================================================================
# Ising Model Simulation (Wolff Cluster Algorithm for efficiency)
# =============================================================================

@jit(nopython=True)
def wolff_step(spins, T, L):
    """Single Wolff cluster flip."""
    p_add = 1.0 - np.exp(-2.0 * J / T)
    
    # Random starting spin
    i0 = np.random.randint(0, L)
    j0 = np.random.randint(0, L)
    
    # Cluster to flip
    cluster = np.zeros((L, L), dtype=np.int8)
    stack = [(i0, j0)]
    cluster[i0, j0] = 1
    s0 = spins[i0, j0]
    
    while len(stack) > 0:
        i, j = stack.pop()
        
        # Check neighbors
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni = (i + di) % L
            nj = (j + dj) % L
            
            if cluster[ni, nj] == 0 and spins[ni, nj] == s0:
                if np.random.random() < p_add:
                    cluster[ni, nj] = 1
                    stack.append((ni, nj))
    
    # Flip cluster
    for i in range(L):
        for j in range(L):
            if cluster[i, j] == 1:
                spins[i, j] *= -1
    
    return spins


@jit(nopython=True)
def metropolis_sweep(spins, T, L):
    """One Metropolis sweep (L^2 single-spin updates)."""
    beta = 1.0 / T
    
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        # Energy change for flipping spin (i,j)
        s = spins[i, j]
        neighbors = (spins[(i+1)%L, j] + spins[(i-1)%L, j] + 
                     spins[i, (j+1)%L] + spins[i, (j-1)%L])
        dE = 2.0 * J * s * neighbors
        
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] *= -1
    
    return spins


@jit(nopython=True)
def run_ising(L, T, n_equilibrate, n_measure, n_measurements):
    """Run Ising simulation and collect measurements."""
    # Initialize random configuration
    spins = np.random.choice(np.array([-1, 1]), size=(L, L)).astype(np.int8)
    
    # Equilibrate using Wolff algorithm
    for _ in range(n_equilibrate):
        spins = wolff_step(spins, T, L)
    
    # Collect measurements
    configs = np.zeros((n_measurements, L, L), dtype=np.int8)
    for m in range(n_measurements):
        for _ in range(n_measure):
            spins = wolff_step(spins, T, L)
        configs[m] = spins.copy()
    
    return configs


# =============================================================================
# Feature Extraction (analogous to gradient moments)
# =============================================================================

def extract_ising_features(configs):
    """
    Extract features from Ising configurations.
    
    Analogous to gradient moments for KPZ:
    - Magnetization statistics ↔ Height statistics
    - Gradient of spin field ↔ Gradient of height field
    - Domain boundaries ↔ Interface roughness
    """
    features = []
    
    for config in configs:
        L = config.shape[0]
        m = config.astype(np.float64)
        
        # 1. Magnetization moments
        mag = np.mean(m)
        mag_abs = np.abs(mag)
        m2 = np.mean(m**2)  # Always 1 for ±1 spins
        m4 = np.mean(m**4)  # Always 1 for ±1 spins
        
        # 2. Local magnetization fluctuations (coarse-grained)
        # Block average to get local magnetization field
        block = 4
        L_coarse = L // block
        m_coarse = np.zeros((L_coarse, L_coarse))
        for i in range(L_coarse):
            for j in range(L_coarse):
                m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
        
        m_local_var = np.var(m_coarse)
        m_local_abs_mean = np.mean(np.abs(m_coarse))
        
        # 3. Gradient of spin field (discrete gradient)
        grad_x = np.roll(m, -1, axis=0) - m
        grad_y = np.roll(m, -1, axis=1) - m
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_var = np.var(grad_mag)
        grad_mean = np.mean(grad_mag)
        grad_max = np.max(grad_mag)
        
        # 4. Domain boundary density
        # Count unlike neighbor pairs
        n_boundaries = 0
        for i in range(L):
            for j in range(L):
                if config[i, j] != config[(i+1)%L, j]:
                    n_boundaries += 1
                if config[i, j] != config[i, (j+1)%L]:
                    n_boundaries += 1
        boundary_density = n_boundaries / (2 * L * L)
        
        # 5. Correlation length proxy (exponential decay rate)
        # Use autocorrelation at lag 1
        corr_1 = np.mean(m * np.roll(m, 1, axis=0))
        
        # 6. Energy per spin
        energy = 0.0
        for i in range(L):
            for j in range(L):
                s = config[i, j]
                neighbors = config[(i+1)%L, j] + config[i, (j+1)%L]
                energy -= J * s * neighbors
        energy_per_spin = energy / (L * L)
        
        # 7. Susceptibility proxy (magnetization variance)
        # Can't compute from single config, use |m| as proxy
        
        # Compile features (8D, similar to gradient moments for KPZ)
        feat = np.array([
            mag_abs,           # 0: |⟨m⟩|
            m_local_var,       # 1: Var(m_local)
            m_local_abs_mean,  # 2: ⟨|m_local|⟩
            grad_mean,         # 3: ⟨|∇m|⟩
            grad_var,          # 4: Var(|∇m|)
            boundary_density,  # 5: domain boundary fraction
            corr_1,            # 6: nearest-neighbor correlation
            energy_per_spin    # 7: ⟨E⟩/N
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False):
    if pilot:
        print("\n" + "="*70)
        print("Experiment 52: Ising Coupling Coordinate Discovery")
        print("               [PILOT MODE]")
        print("="*70)
        n_temps = 8
        n_samples = 20
        n_equilibrate = 2000
        n_measure = 500
        n_measurements = 5
        L_use = 32
    else:
        print("\n" + "="*70)
        print("Experiment 52: Ising Coupling Coordinate Discovery")
        print("               [FULL MODE]")
        print("="*70)
        n_temps = N_TEMPS
        n_samples = N_SAMPLES
        n_equilibrate = N_EQUILIBRATE
        n_measure = N_MEASURE
        n_measurements = N_MEASUREMENTS
        L_use = L
    
    # Temperature range around T_c
    T_min = 0.7 * T_C  # Below T_c (ordered)
    T_max = 1.3 * T_C  # Above T_c (disordered)
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    print(f"\nParameters:")
    print(f"  L={L_use}, T_c={T_C:.4f}")
    print(f"  T range: [{T_min:.3f}, {T_max:.3f}]")
    print(f"  n_temps={n_temps}, n_samples={n_samples}")
    print(f"  n_equilibrate={n_equilibrate}, n_measure={n_measure}")
    print("="*70)
    
    # Collect data
    all_features = []
    all_temps = []
    all_reduced_temps = []
    
    print("\nGenerating Ising configurations...")
    
    for ti, T in enumerate(temperatures):
        t_reduced = (T - T_C) / T_C  # Reduced temperature
        
        print(f"  T={T:.3f} (t={t_reduced:+.3f}): ", end="", flush=True)
        
        temp_features = []
        for s in range(n_samples):
            # Run simulation
            configs = run_ising(L_use, T, n_equilibrate, n_measure, n_measurements)
            
            # Extract features (average over measurements)
            feats = extract_ising_features(configs)
            feat_mean = np.mean(feats, axis=0)
            
            temp_features.append(feat_mean)
            all_temps.append(T)
            all_reduced_temps.append(t_reduced)
            
            if (s + 1) % 10 == 0:
                print(".", end="", flush=True)
        
        all_features.extend(temp_features)
        print(f" done ({n_samples} samples)")
    
    all_features = np.array(all_features)
    all_temps = np.array(all_temps)
    all_reduced_temps = np.array(all_reduced_temps)
    
    print(f"\nTotal samples: {len(all_features)}")
    print(f"Feature dimension: {all_features.shape[1]}")
    
    # ==========================================================================
    # PCA and Coupling Coordinate Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("PCA Analysis")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    pca = PCA(n_components=min(8, all_features.shape[1]))
    pca_coords = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f"  PC{i+1}: {var:.3f}")
    
    # ==========================================================================
    # Correlation with Physical Couplings
    # ==========================================================================
    
    print("\n" + "="*70)
    print("Coupling Coordinate Discovery")
    print("="*70)
    
    # Test correlations
    candidates = {
        'T': all_temps,
        't = (T-T_c)/T_c': all_reduced_temps,
        '|t|': np.abs(all_reduced_temps),
        '1/T': 1.0 / all_temps,
        'T - T_c': all_temps - T_C,
    }
    
    print("\nPC1 correlations with candidate couplings:")
    print("-" * 50)
    
    best_corr = 0
    best_name = ""
    
    for name, values in candidates.items():
        r, p = pearsonr(pca_coords[:, 0], values)
        print(f"  PC1 vs {name:20s}: r = {r:+.3f} (p = {p:.2e})")
        if abs(r) > abs(best_corr):
            best_corr = r
            best_name = name
    
    print("-" * 50)
    print(f"\n  BEST: PC1 vs {best_name}: r = {best_corr:+.3f}")
    
    # Also check PC2
    print("\nPC2 correlations:")
    for name, values in candidates.items():
        r, p = pearsonr(pca_coords[:, 1], values)
        print(f"  PC2 vs {name:20s}: r = {r:+.3f}")
    
    # ==========================================================================
    # Feature Loadings
    # ==========================================================================
    
    feature_names = ['|m|', 'Var(m_local)', '|m_local|', '|∇m|', 
                     'Var(|∇m|)', 'boundary', 'corr_1', 'E/N']
    
    print("\n" + "="*70)
    print("PC1 Loadings (what physical quantities drive PC1)")
    print("="*70)
    
    loadings = pca.components_[0]
    sorted_idx = np.argsort(np.abs(loadings))[::-1]
    
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:15s}: {loadings[idx]:+.3f}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    mode_str = "pilot" if pilot else "full"
    results_dir = f"results_exp52_{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PC1 vs reduced temperature
    ax = axes[0, 0]
    scatter = ax.scatter(all_reduced_temps, pca_coords[:, 0], 
                         c=all_temps, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='T')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='T = T_c')
    ax.set_xlabel('Reduced temperature t = (T - T_c) / T_c')
    ax.set_ylabel('PC1')
    ax.set_title(f'PC1 vs Reduced Temperature (r = {best_corr:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PC1 vs PC2 colored by temperature
    ax = axes[0, 1]
    scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                         c=all_reduced_temps, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='t')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Space (colored by reduced temperature)')
    ax.grid(True, alpha=0.3)
    
    # 3. Feature evolution with temperature
    ax = axes[1, 0]
    for i, name in enumerate(feature_names):
        temp_means = []
        for T in np.unique(all_temps):
            mask = all_temps == T
            temp_means.append(np.mean(features_scaled[mask, i]))
        ax.plot(np.unique(all_temps), temp_means, 'o-', label=name, alpha=0.7)
    ax.axvline(T_C, color='k', linestyle='--', alpha=0.5, label='T_c')
    ax.set_xlabel('Temperature T')
    ax.set_ylabel('Feature value (standardized)')
    ax.set_title('Feature Evolution with Temperature')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 4. PC1 loadings bar chart
    ax = axes[1, 1]
    x = np.arange(len(feature_names))
    colors = ['tab:blue' if l > 0 else 'tab:red' for l in loadings]
    ax.bar(x, loadings, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('PC1 Loading')
    ax.set_title('PC1 Loading by Feature')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ising_coupling_coordinate.png", dpi=150)
    plt.close()
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 52 SUMMARY")
    print("="*70)
    
    # Success criterion: r > 0.8 with reduced temperature
    r_with_t, _ = pearsonr(pca_coords[:, 0], all_reduced_temps)
    
    print(f"\nKey Result: PC1 vs reduced temperature t")
    print(f"  Correlation: r = {r_with_t:+.3f}")
    
    if abs(r_with_t) > 0.8:
        print(f"\n  ✅ SUCCESS: |r| > 0.8")
        print(f"  → Unsupervised PCA recovers RG-relevant coupling in Ising model")
        print(f"  → Coupling coordinate discovery GENERALIZES beyond KPZ!")
    elif abs(r_with_t) > 0.6:
        print(f"\n  ⚠️ PARTIAL: 0.6 < |r| < 0.8")
        print(f"  → Moderate correlation with reduced temperature")
        print(f"  → May need feature refinement or more data")
    else:
        print(f"\n  ❌ WEAK: |r| < 0.6")
        print(f"  → PC1 does not strongly track reduced temperature")
        print(f"  → Method may be KPZ-specific or needs different observables")
    
    print("-" * 70)
    
    # Comparison to KPZ result
    print("\nComparison to KPZ (Exp 46):")
    print("  KPZ: PC1 vs D/ν³       r = 0.857")
    print(f"  Ising: PC1 vs t        r = {abs(r_with_t):.3f}")
    
    if abs(r_with_t) > 0.8:
        print("\n  → BOTH systems show PC1 ≈ relevant RG coupling")
        print("  → Supports: 'Unsupervised ML discovers RG structure'")
    
    # Save results
    results = {
        'experiment': 'Exp 52: Ising Coupling Coordinate',
        'timestamp': datetime.now().isoformat(),
        'mode': 'pilot' if pilot else 'full',
        'parameters': {
            'L': L_use,
            'T_c': T_C,
            'n_temps': n_temps,
            'n_samples': n_samples,
        },
        'correlations': {
            'PC1_vs_t': float(r_with_t),
            'PC1_vs_T': float(pearsonr(pca_coords[:, 0], all_temps)[0]),
        },
        'explained_variance': pca.explained_variance_ratio_.tolist(),
        'pc1_loadings': dict(zip(feature_names, loadings.tolist())),
        'success': bool(abs(r_with_t) > 0.8),
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    print("="*70)
    
    return results


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    main(pilot=pilot)
