"""
Experiment 53b: Vicsek WITHOUT Order-Parameter-Like Features

Same as Exp 53 but excluding local polarization features.
This tests if PC1 is just tracking φ (trivial) or captures independent structure.

EXCLUDED FEATURES:
- local_φ_mean (directly related to order parameter)
- local_φ_var (fluctuations of order parameter)

KEPT FEATURES (7D):
- density_var
- grad_v_mean, grad_v_var
- vorticity_mean, vorticity_var  
- divergence_mean, divergence_var
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

print("Vicsek Model - Without Order Parameter Features")

# =============================================================================
# Vicsek Model Simulation (same as 53)
# =============================================================================

@jit(nopython=True)
def vicsek_step(positions, thetas, v, R, eta, L, N):
    """Single Vicsek time step."""
    new_thetas = np.zeros(N)
    
    for i in range(N):
        sum_sin = 0.0
        sum_cos = 0.0
        
        for j in range(N):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            if dx > L/2: dx -= L
            if dx < -L/2: dx += L
            if dy > L/2: dy -= L
            if dy < -L/2: dy += L
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < R:
                sum_sin += np.sin(thetas[j])
                sum_cos += np.cos(thetas[j])
        
        avg_theta = np.arctan2(sum_sin, sum_cos)
        new_thetas[i] = avg_theta + eta * (np.random.random() - 0.5)
    
    new_positions = np.zeros_like(positions)
    for i in range(N):
        new_positions[i, 0] = (positions[i, 0] + v * np.cos(new_thetas[i])) % L
        new_positions[i, 1] = (positions[i, 1] + v * np.sin(new_thetas[i])) % L
    
    return new_positions, new_thetas


@jit(nopython=True)
def compute_order_parameter(thetas):
    """Compute global polarization φ = |⟨e^{iθ}⟩|."""
    sum_sin = np.sum(np.sin(thetas))
    sum_cos = np.sum(np.cos(thetas))
    N = len(thetas)
    return np.sqrt(sum_sin**2 + sum_cos**2) / N


@jit(nopython=True)
def run_vicsek(N, L, v, R, eta, n_equilibrate, n_measure, n_measurements):
    """Run Vicsek simulation and collect configurations."""
    positions = np.random.random((N, 2)) * L
    thetas = np.random.random(N) * 2 * np.pi - np.pi
    
    for _ in range(n_equilibrate):
        positions, thetas = vicsek_step(positions, thetas, v, R, eta, L, N)
    
    configs = []
    order_params = np.zeros(n_measurements)
    
    for m in range(n_measurements):
        for _ in range(n_measure):
            positions, thetas = vicsek_step(positions, thetas, v, R, eta, L, N)
        configs.append((positions.copy(), thetas.copy()))
        order_params[m] = compute_order_parameter(thetas)
    
    return configs, order_params


# =============================================================================
# Feature Extraction (EXCLUDING polarization features)
# =============================================================================

def extract_vicsek_features_no_phi(configs, L, n_grid=8):
    """
    Extract features EXCLUDING local polarization (analogous to Exp 52b).
    Only keeps: density, velocity gradients, vorticity, divergence.
    """
    features = []
    
    for positions, thetas in configs:
        N = len(thetas)
        
        # Grid for coarse-graining
        grid_count = np.zeros((n_grid, n_grid))
        
        for i in range(N):
            gx = int(positions[i, 0] / L * n_grid) % n_grid
            gy = int(positions[i, 1] / L * n_grid) % n_grid
            grid_count[gx, gy] += 1
        
        # 1. Density fluctuations
        density = grid_count / (L/n_grid)**2
        density_var = np.var(density)
        
        # 2. Velocity field on grid
        vx_grid = np.zeros((n_grid, n_grid))
        vy_grid = np.zeros((n_grid, n_grid))
        
        for i in range(N):
            gx = int(positions[i, 0] / L * n_grid) % n_grid
            gy = int(positions[i, 1] / L * n_grid) % n_grid
            vx_grid[gx, gy] += np.cos(thetas[i])
            vy_grid[gx, gy] += np.sin(thetas[i])
        
        # Normalize (velocity per particle, not total)
        for i in range(n_grid):
            for j in range(n_grid):
                if grid_count[i, j] > 0:
                    vx_grid[i, j] /= grid_count[i, j]
                    vy_grid[i, j] /= grid_count[i, j]
        
        # 3. Velocity gradients
        dvx_dx = np.roll(vx_grid, -1, axis=0) - vx_grid
        dvx_dy = np.roll(vx_grid, -1, axis=1) - vx_grid
        dvy_dx = np.roll(vy_grid, -1, axis=0) - vy_grid
        dvy_dy = np.roll(vy_grid, -1, axis=1) - vy_grid
        
        grad_v = np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
        grad_v_mean = np.mean(grad_v)
        grad_v_var = np.var(grad_v)
        
        # 4. Vorticity (curl)
        vorticity = dvy_dx - dvx_dy
        vorticity_mean = np.mean(np.abs(vorticity))
        vorticity_var = np.var(vorticity)
        
        # 5. Divergence
        divergence = dvx_dx + dvy_dy
        divergence_mean = np.mean(np.abs(divergence))
        divergence_var = np.var(divergence)
        
        # Feature vector (7D, NO local_φ_mean or local_φ_var)
        feat = np.array([
            density_var,         # 0
            grad_v_mean,         # 1
            grad_v_var,          # 2
            vorticity_mean,      # 3
            vorticity_var,       # 4
            divergence_mean,     # 5
            divergence_var,      # 6
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False):
    mode = "PILOT" if pilot else "FULL"
    
    print("\n" + "="*70)
    print("Experiment 53b: Vicsek WITHOUT Order-Parameter Features")
    print(f"               [{mode} MODE]")
    print("="*70)
    
    if pilot:
        N = 300
        L = 10.0
        n_etas = 12
        n_samples = 15
        n_equilibrate = 500
        n_measure = 100
        n_measurements = 3
    else:
        N = 500
        L = 15.0
        n_etas = 15
        n_samples = 25
        n_equilibrate = 1000
        n_measure = 200
        n_measurements = 5
    
    v = 0.5
    R = 1.0
    
    eta_min = 0.1
    eta_max = 3.0
    etas = np.linspace(eta_min, eta_max, n_etas)
    
    print(f"\nParameters:")
    print(f"  N = {N} particles, L = {L}")
    print(f"  Features: 7D (EXCLUDING local_φ_mean, local_φ_var)")
    print(f"  η range: [{eta_min}, {eta_max}]")
    print("="*70)
    
    all_features = []
    all_etas = []
    all_phi = []
    
    for i, eta in enumerate(etas):
        print(f"\n[{i+1}/{n_etas}] η = {eta:.3f}: ", end="", flush=True)
        
        for s in range(n_samples):
            configs, order_params = run_vicsek(
                N, L, v, R, eta, n_equilibrate, n_measure, n_measurements
            )
            
            feats = extract_vicsek_features_no_phi(configs, L)
            feat_mean = np.mean(feats, axis=0)
            phi_mean = np.mean(order_params)
            
            all_features.append(feat_mean)
            all_etas.append(eta)
            all_phi.append(phi_mean)
            
            if (s + 1) % 5 == 0:
                print(".", end="", flush=True)
        
        print(" done")
    
    all_features = np.array(all_features)
    all_etas = np.array(all_etas)
    all_phi = np.array(all_phi)
    
    print(f"\nFeature shape: {all_features.shape} (7D without φ-related)")
    
    # PCA
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    pca = PCA(n_components=7)
    pca_coords = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance: {pca.explained_variance_ratio_[:4]}")
    
    PC1 = pca_coords[:, 0]
    
    # Correlations
    r_pc1_eta, p_pc1_eta = pearsonr(PC1, all_etas)
    r_pc1_phi, _ = pearsonr(PC1, all_phi)
    r_phi_eta, _ = pearsonr(all_phi, all_etas)
    
    print("\n" + "="*70)
    print("RESULTS (Without Order-Parameter Features)")
    print("="*70)
    print(f"\nReference: φ vs η: r = {r_phi_eta:.3f}")
    print(f"\nPC1 (7D features, no local_φ):")
    print(f"  vs η:  r = {r_pc1_eta:+.3f}")
    print(f"  vs φ:  r = {r_pc1_phi:+.3f}")
    
    # Loadings
    feature_names = [
        'density_var', 'grad_v_mean', 'grad_v_var',
        'vorticity_mean', 'vorticity_var',
        'divergence_mean', 'divergence_var'
    ]
    
    loadings = pca.components_[0]
    print("\nPC1 loadings:")
    for name, load in zip(feature_names, loadings):
        print(f"  {name:20s}: {load:+.3f}")
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    if abs(r_pc1_eta) > 0.9:
        print(f"\n✅ STRONG: PC1 vs η = {r_pc1_eta:.3f} WITHOUT local polarization!")
        print("   → Not just tracking order parameter")
        print("   → Velocity gradients/vorticity capture coupling independently")
        success = True
    elif abs(r_pc1_eta) > 0.7:
        print(f"\n⚠️ MODERATE: PC1 vs η = {r_pc1_eta:.3f}")
        print("   → Weaker than with φ-features, but still present")
        success = True
    else:
        print(f"\n❌ WEAK: PC1 vs η = {r_pc1_eta:.3f}")
        print("   → Exp 53 result WAS driven by local polarization")
        success = False
    
    # Save results
    mode_str = "pilot" if pilot else "full"
    results_dir = f"results_exp53b_{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    output = {
        'experiment': '53b',
        'description': 'Vicsek WITHOUT order-parameter features',
        'results': {
            'r_PC1_eta': float(r_pc1_eta),
            'r_PC1_phi': float(r_pc1_phi),
            'comparison_with_53': 'Does NOT use local_φ_mean, local_φ_var',
            'success': success,
        },
    }
    
    with open(f"{results_dir}/results.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")
    
    return success, r_pc1_eta


if __name__ == "__main__":
    pilot = "--pilot" in sys.argv
    success, r = main(pilot=pilot)
    
    print("\n" + "="*70)
    if success:
        print(f"✅ Exp 53b VALIDATES: r = {r:.3f} without φ-features")
        print("   Method captures genuine coupling structure, not just order parameter")
    else:
        print(f"⚠️ Exp 53b: Weaker signal without φ-features (r = {r:.3f})")
