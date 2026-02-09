"""
Experiment 52b: Ising Coupling Coordinate - WITHOUT Trivial Features

CRITICAL VALIDATION: Remove |m| (order parameter) and E/N (conjugate to T)
to test if coupling coordinate discovery is REAL or CIRCULAR.

If Exp 52 was trivial:
  - Including |m| guarantees finding T-correlation (|m| IS order parameter)
  - Including E/N guarantees finding T-correlation (E = ∂F/∂β)

If method is genuine:
  - PC1 should STILL correlate with t using only:
    * Var(m_local), |m_local|, |∇m|, Var(|∇m|), boundary_density, corr_1
  - These are "indirect" probes of the thermal transition

OBSERVABLES (6D - excludes |m| and E/N):
  0. Var(m_local)   - local magnetization variance
  1. |m_local|      - mean local magnetization magnitude  
  2. |∇m|           - gradient magnitude
  3. Var(|∇m|)      - gradient variance
  4. boundary       - domain boundary density
  5. corr_1         - nearest-neighbor correlation

SUCCESS CRITERION:
If PC1 vs t still shows |r| > 0.8, the Ising result is VALIDATED.
If correlation collapses, Exp 52 was driven by trivial features.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
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
# Ising Model Simulation (Wolff Cluster Algorithm)
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
# Feature Extraction - EXCLUDES |m| AND E/N
# =============================================================================

def extract_ising_features_no_trivial(configs):
    """
    Extract features EXCLUDING trivial ones:
    - EXCLUDES |m| (order parameter itself)
    - EXCLUDES E/N (conjugate to temperature)
    
    Keeps only indirect probes of thermal transition.
    """
    features = []
    
    for config in configs:
        L = config.shape[0]
        m = config.astype(np.float64)
        
        # EXCLUDED: mag_abs = np.abs(np.mean(m))  # This IS the order parameter
        
        # 1. Local magnetization fluctuations (coarse-grained)
        block = 4
        L_coarse = L // block
        m_coarse = np.zeros((L_coarse, L_coarse))
        for i in range(L_coarse):
            for j in range(L_coarse):
                m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
        
        m_local_var = np.var(m_coarse)
        m_local_abs_mean = np.mean(np.abs(m_coarse))
        
        # 2. Gradient of spin field (discrete gradient)
        grad_x = np.roll(m, -1, axis=0) - m
        grad_y = np.roll(m, -1, axis=1) - m
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_var = np.var(grad_mag)
        grad_mean = np.mean(grad_mag)
        
        # 3. Domain boundary density
        n_boundaries = 0
        for i in range(L):
            for j in range(L):
                if config[i, j] != config[(i+1)%L, j]:
                    n_boundaries += 1
                if config[i, j] != config[i, (j+1)%L]:
                    n_boundaries += 1
        boundary_density = n_boundaries / (2 * L * L)
        
        # 4. Correlation length proxy (autocorrelation at lag 1)
        corr_1 = np.mean(m * np.roll(m, 1, axis=0))
        
        # EXCLUDED: energy_per_spin  # Conjugate to T
        
        # Compile features (6D - non-trivial only)
        feat = np.array([
            m_local_var,       # 0: Var(m_local)
            m_local_abs_mean,  # 1: ⟨|m_local|⟩
            grad_mean,         # 2: ⟨|∇m|⟩
            grad_var,          # 3: Var(|∇m|)
            boundary_density,  # 4: domain boundary fraction
            corr_1,            # 5: nearest-neighbor correlation
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False, narrow_window=False):
    mode = "PILOT" if pilot else "FULL"
    window = "NARROW" if narrow_window else "WIDE"
    
    print("\n" + "="*70)
    print("Experiment 52b: Ising WITHOUT Trivial Features (|m|, E/N excluded)")
    print(f"               [{mode} MODE, {window} TEMPERATURE WINDOW]")
    print("="*70)
    
    if pilot:
        n_temps = 8
        n_samples = 30
        n_equilibrate = 2000
        n_measure = 500
        n_measurements = 5
        L_use = 32
    else:
        n_temps = N_TEMPS
        n_samples = N_SAMPLES
        n_equilibrate = N_EQUILIBRATE
        n_measure = N_MEASURE
        n_measurements = N_MEASUREMENTS
        L_use = L
    
    # Temperature range
    if narrow_window:
        # Narrow: ±10% around T_c (tests critical regime)
        T_min = 0.90 * T_C
        T_max = 1.10 * T_C
    else:
        # Wide: ±30% around T_c (original Exp 52 range)
        T_min = 0.70 * T_C
        T_max = 1.30 * T_C
    
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    print(f"\nParameters:")
    print(f"  L={L_use}, T_c={T_C:.4f}")
    print(f"  T range: [{T_min:.3f}, {T_max:.3f}] ({window})")
    print(f"  n_temps={n_temps}, n_samples={n_samples}")
    print(f"  Features: 6D (EXCLUDES |m| and E/N)")
    print("="*70)
    
    # Collect data
    all_features = []
    all_temps = []
    all_reduced_temps = []
    
    print("\nGenerating Ising configurations...")
    
    for ti, T in enumerate(temperatures):
        t_reduced = (T - T_C) / T_C
        
        print(f"  T={T:.3f} (t={t_reduced:+.4f}): ", end="", flush=True)
        
        temp_features = []
        for s in range(n_samples):
            # Run simulation
            configs = run_ising(L_use, T, n_equilibrate, n_measure, n_measurements)
            
            # Extract features (WITHOUT trivial ones)
            feats = extract_ising_features_no_trivial(configs)
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
    print(f"Feature dimension: {all_features.shape[1]} (6D - no |m|, no E/N)")
    
    # ==========================================================================
    # PCA and Coupling Coordinate Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("PCA Analysis (on non-trivial features only)")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # PCA
    pca = PCA(n_components=min(6, all_features.shape[1]))
    pca_coords = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance ratios:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    
    # ==========================================================================
    # Correlation with Physical Couplings
    # ==========================================================================
    
    print("\n" + "="*70)
    print("Coupling Coordinate Discovery (WITHOUT trivial features)")
    print("="*70)
    
    # Test correlations
    candidates = {
        't = (T-T_c)/T_c': all_reduced_temps,
        'T': all_temps,
        '|t|': np.abs(all_reduced_temps),
        '1/T (β)': 1.0 / all_temps,
    }
    
    print("\nPC1 correlations with candidate couplings:")
    print("-" * 60)
    
    results = {}
    best_corr = 0
    best_name = ""
    
    for name, values in candidates.items():
        r, p = pearsonr(pca_coords[:, 0], values)
        print(f"  PC1 vs {name:20s}: r = {r:+.4f} (p = {p:.2e})")
        results[f'PC1_vs_{name}'] = {'r': float(r), 'p': float(p)}
        if abs(r) > abs(best_corr):
            best_corr = r
            best_name = name
    
    r_with_t = results['PC1_vs_t = (T-T_c)/T_c']['r']
    
    print("-" * 60)
    
    # ==========================================================================
    # CRITICAL VALIDATION
    # ==========================================================================
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    
    # Compare with Exp 52 (with |m|)
    exp52_r = 0.971  # From Exp 52 pilot
    
    print(f"\n  Exp 52  (with |m|, E/N): r = {exp52_r:.3f}")
    print(f"  Exp 52b (without them):  r = {abs(r_with_t):.3f}")
    print(f"  Change: {(abs(r_with_t) - exp52_r) / exp52_r * 100:+.1f}%")
    
    if abs(r_with_t) > 0.8:
        print(f"\n✅ SUCCESS: |r| = {abs(r_with_t):.3f} > 0.8")
        print("→ Coupling coordinate discovery is GENUINE, not circular!")
        print("→ PC1 tracks RG-relevant coupling WITHOUT trivial features")
        success = True
    elif abs(r_with_t) > 0.5:
        print(f"\n⚠️ PARTIAL: |r| = {abs(r_with_t):.3f} (moderate)")
        print("→ Some signal remains without trivial features")
        print("→ But weaker than with |m| - partial validation")
        success = False
    else:
        print(f"\n❌ FAILED: |r| = {abs(r_with_t):.3f} < 0.5")
        print("→ Correlation collapsed without |m| and E/N")
        print("→ Exp 52 result was driven by TRIVIAL features")
        success = False
    
    # ==========================================================================
    # Feature Loadings
    # ==========================================================================
    
    feature_names = ['Var(m_local)', '|m_local|', '|∇m|', 
                     'Var(|∇m|)', 'boundary', 'corr_1']
    
    print("\n" + "="*70)
    print("PC1 Loadings (non-trivial features only)")
    print("="*70)
    
    loadings = pca.components_[0]
    sorted_idx = np.argsort(np.abs(loadings))[::-1]
    
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:15s}: {loadings[idx]:+.4f}")
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    
    mode_str = "pilot" if pilot else "full"
    window_str = "narrow" if narrow_window else "wide"
    results_dir = f"results_exp52b_{mode_str}_{window_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save JSON
    output = {
        'experiment': '52b',
        'description': 'Ising coupling coordinate WITHOUT trivial features',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'L': L_use,
            'n_temps': n_temps,
            'n_samples': n_samples,
            'T_range': [float(T_min), float(T_max)],
            'window': window_str,
            'features_excluded': ['|m|', 'E/N'],
            'features_used': feature_names,
        },
        'results': {
            'PC1_vs_t': float(r_with_t),
            'comparison_with_exp52': {
                'exp52_r': exp52_r,
                'exp52b_r': float(abs(r_with_t)),
                'change_percent': float((abs(r_with_t) - exp52_r) / exp52_r * 100),
            },
            'success': bool(success),
            'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
            'pc1_loadings': {name: float(loadings[i]) for i, name in enumerate(feature_names)},
        },
        'conclusion': 'GENUINE' if success else 'TRIVIAL' if abs(r_with_t) < 0.5 else 'PARTIAL',
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PC1 vs reduced temperature
    ax = axes[0, 0]
    scatter = ax.scatter(all_reduced_temps, pca_coords[:, 0], 
                         c=all_temps, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='T')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='T = T_c')
    ax.set_xlabel('Reduced temperature t = (T - T_c) / T_c')
    ax.set_ylabel('PC1')
    ax.set_title(f'PC1 vs t (r = {r_with_t:.3f}) - NO |m|, NO E/N')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PC1 vs PC2
    ax = axes[0, 1]
    scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                         c=all_reduced_temps, cmap='coolwarm', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=ax, label='t')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1 vs PC2 (colored by reduced temperature)')
    ax.grid(True, alpha=0.3)
    
    # 3. Loadings comparison
    ax = axes[1, 0]
    x = np.arange(len(feature_names))
    bars = ax.bar(x, loadings, color=['steelblue']*len(feature_names))
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('PC1 Loading')
    ax.set_title('PC1 Loadings (non-trivial features)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 4. Validation summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
VALIDATION SUMMARY (Exp 52b)
{'='*40}

Temperature range: [{T_min:.2f}, {T_max:.2f}] ({window})
Features: 6D (EXCLUDES |m| and E/N)

CORRELATIONS:
  PC1 vs t: r = {r_with_t:.4f}

COMPARISON WITH Exp 52:
  Exp 52  (with |m|, E/N): r = {exp52_r:.3f}
  Exp 52b (without them):  r = {abs(r_with_t):.3f}
  Change: {(abs(r_with_t) - exp52_r) / exp52_r * 100:+.1f}%

VERDICT: {'✅ GENUINE' if success else '❌ TRIVIAL/PARTIAL'}
{'Coupling coordinate discovery validated!' if success else 'Result was driven by trivial features.'}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/coupling_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {results_dir}/")
    print("\n" + "="*70)
    
    return success, r_with_t


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    narrow = "--narrow" in sys.argv
    
    success, r = main(pilot=pilot, narrow_window=narrow)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    if success:
        print("✅ Ising coupling coordinate discovery is GENUINE")
        print("   PC1 tracks RG-relevant coupling WITHOUT trivial |m| or E/N")
    else:
        print("⚠️ Result needs further investigation")
        print(f"   Correlation without trivial features: r = {r:.3f}")
