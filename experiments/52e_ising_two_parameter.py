"""
Experiment 52e: Ising Two-Parameter Test (t, h)

QUESTION: Does PC space show 2D structure corresponding to RG-relevant directions?

The 2D Ising model has TWO relevant operators:
1. Thermal: t = (T - T_c) / T_c  (even under Z_2)
2. Magnetic: h = external field   (odd under Z_2)

If PC1/PC2 capture RG structure, we should see:
- PC1 aligns with one direction (likely thermal, even)
- PC2 aligns with the other (likely magnetic, odd)
- 2D manifold structure in (t, h) parameter space

This is a DEFINITIVE test:
- If we see 2D RG structure → genuine discovery
- PC space coordinates correspond to relevant operators

PHYSICS:
- Near critical point, free energy scales as: f(t, h) ~ |t|^(2-α) * Φ±(h/|t|^Δ)
- Where Δ = βδ = 15/8 for 2D Ising
- β = 1/8, δ = 15, ν = 1, η = 1/4

SUCCESS CRITERION:
1. PC1 correlates strongly with t (or linear combo of t, h)
2. PC2 correlates with orthogonal direction
3. 2D structure visible in PC1-PC2 plane
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

# Critical temperature
J = 1.0
T_C = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269

print(f"2D Ising: T_c = {T_C:.4f}")

# =============================================================================
# Ising Model with External Field (Metropolis Algorithm)
# =============================================================================

@jit(nopython=True)
def metropolis_step(spins, T, h, L):
    """Single Metropolis sweep with external field."""
    beta = 1.0 / T
    
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        s = spins[i, j]
        
        # Sum of neighbors
        neighbors = (spins[(i+1)%L, j] + spins[(i-1)%L, j] + 
                     spins[i, (j+1)%L] + spins[i, (j-1)%L])
        
        # Energy change: ΔE = 2*s*(J*neighbors + h)
        dE = 2.0 * s * (J * neighbors + h)
        
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = -s
    
    return spins


@jit(nopython=True)
def run_ising_with_field(L, T, h, n_equilibrate, n_measure, n_measurements):
    """Run Ising simulation with external field."""
    # Initialize aligned with field direction if h != 0
    if h > 0:
        spins = np.ones((L, L), dtype=np.int8)
    elif h < 0:
        spins = -np.ones((L, L), dtype=np.int8)
    else:
        spins = np.random.choice(np.array([-1, 1]), size=(L, L)).astype(np.int8)
    
    # Equilibrate
    for _ in range(n_equilibrate):
        spins = metropolis_step(spins, T, h, L)
    
    # Collect measurements
    configs = np.zeros((n_measurements, L, L), dtype=np.int8)
    for m_idx in range(n_measurements):
        for _ in range(n_measure):
            spins = metropolis_step(spins, T, h, L)
        configs[m_idx] = spins.copy()
    
    return configs


# =============================================================================
# Feature Extraction (Same as 52b)
# =============================================================================

def extract_ising_features(configs):
    """Extract 6D features (excluding |m| and E/N)."""
    features = []
    
    for config in configs:
        L = config.shape[0]
        m = config.astype(np.float64)
        
        # Local magnetization (coarse-grained)
        block = max(2, L // 16)
        L_coarse = L // block
        m_coarse = np.zeros((L_coarse, L_coarse))
        for i in range(L_coarse):
            for j in range(L_coarse):
                m_coarse[i, j] = np.mean(m[i*block:(i+1)*block, j*block:(j+1)*block])
        
        m_local_var = np.var(m_coarse)
        m_local_abs_mean = np.mean(np.abs(m_coarse))
        
        # Gradient
        grad_x = np.roll(m, -1, axis=0) - m
        grad_y = np.roll(m, -1, axis=1) - m
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        grad_var = np.var(grad_mag)
        grad_mean = np.mean(grad_mag)
        
        # Domain boundary density
        n_boundaries = 0
        for i in range(L):
            for j in range(L):
                if config[i, j] != config[(i+1)%L, j]:
                    n_boundaries += 1
                if config[i, j] != config[i, (j+1)%L]:
                    n_boundaries += 1
        boundary_density = n_boundaries / (2 * L * L)
        
        # Correlation
        corr_1 = np.mean(m * np.roll(m, 1, axis=0))
        
        feat = np.array([
            m_local_var,
            m_local_abs_mean,
            grad_mean,
            grad_var,
            boundary_density,
            corr_1,
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False):
    mode = "PILOT" if pilot else "FULL"
    
    print("\n" + "="*70)
    print("Experiment 52e: Ising Two-Parameter Test (t, h)")
    print(f"               [{mode} MODE]")
    print("="*70)
    
    L = 32 if pilot else 48
    
    if pilot:
        n_temps = 7
        n_fields = 7
        n_samples = 8
        n_equilibrate = 2000
        n_measure = 200
        n_measurements = 3
    else:
        n_temps = 9
        n_fields = 9
        n_samples = 15
        n_equilibrate = 4000
        n_measure = 400
        n_measurements = 5
    
    # Temperature range around T_c
    T_min = 0.90 * T_C
    T_max = 1.10 * T_C
    temperatures = np.linspace(T_min, T_max, n_temps)
    
    # Field range (symmetric around 0)
    h_max = 0.3  # Modest field strength
    fields = np.linspace(-h_max, h_max, n_fields)
    
    print(f"\nParameters:")
    print(f"  L = {L}")
    print(f"  T range: [{T_min:.3f}, {T_max:.3f}] (±10% of T_c)")
    print(f"  h range: [{-h_max}, {h_max}]")
    print(f"  Grid: {n_temps} × {n_fields} = {n_temps * n_fields} points")
    print(f"  Samples per point: {n_samples}")
    print("="*70)
    
    # Collect data on (t, h) grid
    all_features = []
    all_t = []
    all_h = []
    
    total_points = n_temps * n_fields
    point_idx = 0
    
    for T in temperatures:
        t_reduced = (T - T_C) / T_C
        
        for h in fields:
            point_idx += 1
            print(f"[{point_idx}/{total_points}] T={T:.3f} (t={t_reduced:+.3f}), h={h:+.3f}: ", 
                  end="", flush=True)
            
            for s in range(n_samples):
                configs = run_ising_with_field(L, T, h, n_equilibrate, n_measure, n_measurements)
                feats = extract_ising_features(configs)
                feat_mean = np.mean(feats, axis=0)
                
                all_features.append(feat_mean)
                all_t.append(t_reduced)
                all_h.append(h)
            
            print("done")
    
    all_features = np.array(all_features)
    all_t = np.array(all_t)
    all_h = np.array(all_h)
    
    # ==========================================================================
    # PCA Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("PCA Analysis")
    print("="*70)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    pca = PCA(n_components=6)
    pca_coords = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance ratios: {pca.explained_variance_ratio_[:4]}")
    print(f"Cumulative (PC1+PC2): {pca.explained_variance_ratio_[:2].sum():.3f}")
    
    PC1 = pca_coords[:, 0]
    PC2 = pca_coords[:, 1]
    
    # ==========================================================================
    # Correlation Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("Correlation with RG Parameters")
    print("="*70)
    
    # PC1 correlations
    r_pc1_t, p_pc1_t = pearsonr(PC1, all_t)
    r_pc1_h, p_pc1_h = pearsonr(PC1, all_h)
    r_pc1_h_abs, _ = pearsonr(PC1, np.abs(all_h))
    
    # PC2 correlations
    r_pc2_t, p_pc2_t = pearsonr(PC2, all_t)
    r_pc2_h, p_pc2_h = pearsonr(PC2, all_h)
    r_pc2_h_abs, _ = pearsonr(PC2, np.abs(all_h))
    
    print(f"\nPC1 correlations:")
    print(f"  vs t:   r = {r_pc1_t:+.3f} (p = {p_pc1_t:.2e})")
    print(f"  vs h:   r = {r_pc1_h:+.3f} (p = {p_pc1_h:.2e})")
    print(f"  vs |h|: r = {r_pc1_h_abs:+.3f}")
    
    print(f"\nPC2 correlations:")
    print(f"  vs t:   r = {r_pc2_t:+.3f} (p = {p_pc2_t:.2e})")
    print(f"  vs h:   r = {r_pc2_h:+.3f} (p = {p_pc2_h:.2e})")
    print(f"  vs |h|: r = {r_pc2_h_abs:+.3f}")
    
    # Check for 2D structure
    # If PC1 ~ t and PC2 ~ h (or vice versa), we have 2D RG structure
    
    # Try linear combinations
    print("\n" + "="*70)
    print("Optimal Linear Combinations")
    print("="*70)
    
    # Find rotation that maximizes |r| with t for one axis and h for other
    from scipy.optimize import minimize_scalar
    
    def rotation_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
    
    def objective(theta):
        R = rotation_matrix(theta)
        rotated = pca_coords[:, :2] @ R.T
        r_t = abs(pearsonr(rotated[:, 0], all_t)[0])
        r_h = abs(pearsonr(rotated[:, 1], all_h)[0])
        return -(r_t + r_h)  # Maximize sum
    
    result = minimize_scalar(objective, bounds=(0, np.pi/2), method='bounded')
    theta_opt = result.x
    R_opt = rotation_matrix(theta_opt)
    
    rotated_coords = pca_coords[:, :2] @ R_opt.T
    r_rot0_t, _ = pearsonr(rotated_coords[:, 0], all_t)
    r_rot0_h, _ = pearsonr(rotated_coords[:, 0], all_h)
    r_rot1_t, _ = pearsonr(rotated_coords[:, 1], all_t)
    r_rot1_h, _ = pearsonr(rotated_coords[:, 1], all_h)
    
    print(f"\nOptimal rotation: θ = {np.degrees(theta_opt):.1f}°")
    print(f"\nAfter rotation:")
    print(f"  Axis 0: r(t) = {r_rot0_t:+.3f}, r(h) = {r_rot0_h:+.3f}")
    print(f"  Axis 1: r(t) = {r_rot1_t:+.3f}, r(h) = {r_rot1_h:+.3f}")
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    
    # Check if we have 2D structure
    # Criterion: One axis correlates with t, other with h
    has_t_axis = max(abs(r_rot0_t), abs(r_rot1_t)) > 0.8
    has_h_axis = max(abs(r_rot0_h), abs(r_rot1_h)) > 0.5
    
    # Also check if PC2 explains significant variance
    pc2_significant = pca.explained_variance_ratio_[1] > 0.05
    
    if has_t_axis and has_h_axis and pc2_significant:
        print(f"\n✅ SUCCESS: Found 2D RG structure!")
        print(f"   Thermal axis: r(t) = {max(abs(r_rot0_t), abs(r_rot1_t)):.3f}")
        print(f"   Magnetic axis: r(h) = {max(abs(r_rot0_h), abs(r_rot1_h)):.3f}")
        print("   → PC space captures both relevant operators!")
        success = True
    elif has_t_axis:
        print(f"\n⚠️ PARTIAL: Found thermal axis only")
        print(f"   r(t) = {max(abs(r_rot0_t), abs(r_rot1_t)):.3f}")
        print(f"   But h-axis weak: r(h) = {max(abs(r_rot0_h), abs(r_rot1_h)):.3f}")
        print("   → Replicates 52b result, no new 2D structure")
        success = False
    else:
        print(f"\n❌ FAILED: No clear RG structure")
        success = False
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    mode_str = "pilot" if pilot else "full"
    results_dir = f"results_exp52e_{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. PC1 vs t (colored by h)
    ax = axes[0, 0]
    sc = ax.scatter(all_t, PC1, c=all_h, cmap='coolwarm', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='h')
    ax.set_xlabel('t = (T - T_c) / T_c')
    ax.set_ylabel('PC1')
    ax.set_title(f'PC1 vs t (r = {r_pc1_t:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 2. PC2 vs h (colored by t)
    ax = axes[0, 1]
    sc = ax.scatter(all_h, PC2, c=all_t, cmap='RdYlBu', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='t')
    ax.set_xlabel('h (external field)')
    ax.set_ylabel('PC2')
    ax.set_title(f'PC2 vs h (r = {r_pc2_h:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 3. PC1-PC2 plane (colored by t)
    ax = axes[0, 2]
    sc = ax.scatter(PC1, PC2, c=all_t, cmap='RdYlBu', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='t')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1-PC2 plane (color = t)')
    ax.grid(True, alpha=0.3)
    
    # 4. PC1-PC2 plane (colored by h)
    ax = axes[1, 0]
    sc = ax.scatter(PC1, PC2, c=all_h, cmap='coolwarm', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='h')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1-PC2 plane (color = h)')
    ax.grid(True, alpha=0.3)
    
    # 5. Rotated axes
    ax = axes[1, 1]
    sc = ax.scatter(rotated_coords[:, 0], rotated_coords[:, 1], c=all_t, cmap='RdYlBu', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='t')
    ax.set_xlabel(f'Rotated axis 0 (r(t)={r_rot0_t:.2f})')
    ax.set_ylabel(f'Rotated axis 1 (r(t)={r_rot1_t:.2f})')
    ax.set_title(f'After optimal rotation (θ = {np.degrees(theta_opt):.1f}°)')
    ax.grid(True, alpha=0.3)
    
    # 6. (t, h) grid colored by PC1
    ax = axes[1, 2]
    # Average PC1 at each grid point
    t_unique = np.unique(all_t)
    h_unique = np.unique(all_h)
    PC1_grid = np.zeros((len(t_unique), len(h_unique)))
    for i, t in enumerate(t_unique):
        for j, h in enumerate(h_unique):
            mask = (np.abs(all_t - t) < 0.001) & (np.abs(all_h - h) < 0.001)
            if mask.sum() > 0:
                PC1_grid[i, j] = np.mean(PC1[mask])
    
    im = ax.imshow(PC1_grid.T, extent=[t_unique.min(), t_unique.max(), h_unique.min(), h_unique.max()],
                   origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='PC1')
    ax.set_xlabel('t')
    ax.set_ylabel('h')
    ax.set_title('PC1 on (t, h) grid')
    ax.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax.axvline(0, color='white', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/two_parameter_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    output = {
        'experiment': '52e',
        'description': 'Ising two-parameter (t, h) test',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'L': L,
            'n_temps': n_temps,
            'n_fields': n_fields,
            'n_samples': n_samples,
            'T_range': [float(T_min), float(T_max)],
            'h_range': [-h_max, h_max],
        },
        'results': {
            'explained_variance': pca.explained_variance_ratio_[:4].tolist(),
            'PC1_correlations': {
                't': float(r_pc1_t),
                'h': float(r_pc1_h),
            },
            'PC2_correlations': {
                't': float(r_pc2_t),
                'h': float(r_pc2_h),
            },
            'optimal_rotation_deg': float(np.degrees(theta_opt)),
            'rotated_correlations': {
                'axis0_t': float(r_rot0_t),
                'axis0_h': float(r_rot0_h),
                'axis1_t': float(r_rot1_t),
                'axis1_h': float(r_rot1_h),
            },
            'success': bool(success),
        },
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")
    
    return success


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    
    success = main(pilot=pilot)
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    if success:
        print("✅ PC space captures 2D RG structure (thermal + magnetic)")
        print("   This is DEFINITIVE evidence for RG-relevant embedding!")
    else:
        print("⚠️ Only 1D structure found (replicates 52b)")
        print("   Need to investigate why magnetic direction is weak/absent")
