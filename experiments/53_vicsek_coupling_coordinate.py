"""
Experiment 53: Vicsek Model Coupling Coordinate Discovery

QUESTION: Can we discover coupling coordinates in a system with less characterized RG structure?

The Vicsek model is a canonical active matter system with:
- Order-disorder phase transition at noise η_c
- Flocking behavior below η_c (ordered)
- Disordered gas above η_c
- RG structure is less understood than Ising/KPZ

If PC1 discovers a meaningful coupling coordinate:
- Would demonstrate PREDICTIVE power (not just validation)
- Upgrade method from "confirms known physics" to "discovers new physics"

THE MODEL:
- N particles on L×L torus with periodic boundaries
- Each particle i has position (x_i, y_i) and velocity direction θ_i
- At each step:
  1. θ_i = average of neighbors' θ within radius R, plus noise η
  2. Move with speed v in direction θ_i
- Order parameter: φ = |⟨e^{iθ}⟩| (polarization)

CONTROL PARAMETER: Noise strength η ∈ [0, 2π]
- Low η → ordered (all particles align)
- High η → disordered (random directions)

FEATURES (analogous to Ising):
- Local polarization statistics
- Velocity gradients
- Density fluctuations
- Correlation lengths
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

print("Vicsek Active Matter Model")

# =============================================================================
# Vicsek Model Simulation
# =============================================================================

@jit(nopython=True)
def get_neighbors(x, y, positions, R, L):
    """Get indices of particles within radius R (with periodic boundaries)."""
    neighbors = []
    for j in range(len(positions)):
        dx = positions[j, 0] - x
        dy = positions[j, 1] - y
        # Periodic boundaries
        if dx > L/2: dx -= L
        if dx < -L/2: dx += L
        if dy > L/2: dy -= L
        if dy < -L/2: dy += L
        
        dist = np.sqrt(dx**2 + dy**2)
        if dist < R:
            neighbors.append(j)
    return neighbors


@jit(nopython=True)
def vicsek_step(positions, thetas, v, R, eta, L, N):
    """Single Vicsek time step."""
    new_thetas = np.zeros(N)
    
    for i in range(N):
        # Find neighbors within radius R
        sum_sin = 0.0
        sum_cos = 0.0
        count = 0
        
        for j in range(N):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            # Periodic boundaries
            if dx > L/2: dx -= L
            if dx < -L/2: dx += L
            if dy > L/2: dy -= L
            if dy < -L/2: dy += L
            
            dist = np.sqrt(dx**2 + dy**2)
            if dist < R:
                sum_sin += np.sin(thetas[j])
                sum_cos += np.cos(thetas[j])
                count += 1
        
        # Average direction plus noise
        avg_theta = np.arctan2(sum_sin, sum_cos)
        new_thetas[i] = avg_theta + eta * (np.random.random() - 0.5)
    
    # Update positions
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
    # Initialize randomly
    positions = np.random.random((N, 2)) * L
    thetas = np.random.random(N) * 2 * np.pi - np.pi
    
    # Equilibrate
    for _ in range(n_equilibrate):
        positions, thetas = vicsek_step(positions, thetas, v, R, eta, L, N)
    
    # Collect measurements
    configs = []
    order_params = np.zeros(n_measurements)
    
    for m in range(n_measurements):
        for _ in range(n_measure):
            positions, thetas = vicsek_step(positions, thetas, v, R, eta, L, N)
        
        # Store configuration
        configs.append((positions.copy(), thetas.copy()))
        order_params[m] = compute_order_parameter(thetas)
    
    return configs, order_params


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_vicsek_features(configs, L, n_grid=8):
    """
    Extract local features from Vicsek configurations.
    Analogous to Ising features but for continuous angles.
    """
    features = []
    
    for positions, thetas in configs:
        N = len(thetas)
        
        # 1. Global order parameter (will exclude later like |m| in Ising)
        phi = compute_order_parameter(thetas)
        
        # 2. Local polarization on grid
        grid_phi = np.zeros((n_grid, n_grid))
        grid_count = np.zeros((n_grid, n_grid))
        
        for i in range(N):
            gx = int(positions[i, 0] / L * n_grid) % n_grid
            gy = int(positions[i, 1] / L * n_grid) % n_grid
            grid_count[gx, gy] += 1
        
        # Compute local polarization in each cell
        grid_sin = np.zeros((n_grid, n_grid))
        grid_cos = np.zeros((n_grid, n_grid))
        
        for i in range(N):
            gx = int(positions[i, 0] / L * n_grid) % n_grid
            gy = int(positions[i, 1] / L * n_grid) % n_grid
            grid_sin[gx, gy] += np.sin(thetas[i])
            grid_cos[gx, gy] += np.cos(thetas[i])
        
        # Local polarization magnitude
        local_phi = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                if grid_count[i, j] > 0:
                    local_phi[i, j] = np.sqrt(grid_sin[i, j]**2 + grid_cos[i, j]**2) / grid_count[i, j]
        
        local_phi_mean = np.mean(local_phi[grid_count > 0]) if np.any(grid_count > 0) else 0
        local_phi_var = np.var(local_phi[grid_count > 0]) if np.any(grid_count > 0) else 0
        
        # 3. Density fluctuations
        density = grid_count / (L/n_grid)**2  # Particles per unit area
        density_mean = np.mean(density)
        density_var = np.var(density)
        
        # 4. Velocity field gradients
        # Compute average velocity in each cell
        vx_grid = np.zeros((n_grid, n_grid))
        vy_grid = np.zeros((n_grid, n_grid))
        
        for i in range(N):
            gx = int(positions[i, 0] / L * n_grid) % n_grid
            gy = int(positions[i, 1] / L * n_grid) % n_grid
            vx_grid[gx, gy] += np.cos(thetas[i])
            vy_grid[gx, gy] += np.sin(thetas[i])
        
        # Normalize
        for i in range(n_grid):
            for j in range(n_grid):
                if grid_count[i, j] > 0:
                    vx_grid[i, j] /= grid_count[i, j]
                    vy_grid[i, j] /= grid_count[i, j]
        
        # Gradient of velocity field
        dvx_dx = np.roll(vx_grid, -1, axis=0) - vx_grid
        dvx_dy = np.roll(vx_grid, -1, axis=1) - vx_grid
        dvy_dx = np.roll(vy_grid, -1, axis=0) - vy_grid
        dvy_dy = np.roll(vy_grid, -1, axis=1) - vy_grid
        
        # Velocity gradient magnitude
        grad_v = np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
        grad_v_mean = np.mean(grad_v)
        grad_v_var = np.var(grad_v)
        
        # 5. Vorticity (curl of velocity field)
        vorticity = dvy_dx - dvx_dy
        vorticity_mean = np.mean(np.abs(vorticity))
        vorticity_var = np.var(vorticity)
        
        # 6. Divergence (compression)
        divergence = dvx_dx + dvy_dy
        divergence_mean = np.mean(np.abs(divergence))
        divergence_var = np.var(divergence)
        
        # Feature vector (excluding global φ for now)
        feat = np.array([
            local_phi_mean,      # 0
            local_phi_var,       # 1
            density_var,         # 2
            grad_v_mean,         # 3
            grad_v_var,          # 4
            vorticity_mean,      # 5
            vorticity_var,       # 6
            divergence_mean,     # 7
            divergence_var,      # 8
        ])
        
        features.append(feat)
    
    return np.array(features)


# =============================================================================
# Main Experiment
# =============================================================================

def main(pilot=False):
    mode = "PILOT" if pilot else "FULL"
    
    print("\n" + "="*70)
    print("Experiment 53: Vicsek Model Coupling Coordinate Discovery")
    print(f"               [{mode} MODE]")
    print("="*70)
    
    # Parameters
    if pilot:
        N = 300       # Number of particles
        L = 10.0      # Box size
        n_etas = 12   # Number of noise values
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
    
    v = 0.5       # Particle speed
    R = 1.0       # Interaction radius
    
    # Noise range (spans ordered to disordered)
    eta_min = 0.1
    eta_max = 3.0
    etas = np.linspace(eta_min, eta_max, n_etas)
    
    print(f"\nParameters:")
    print(f"  N = {N} particles, L = {L}")
    print(f"  v = {v}, R = {R}")
    print(f"  η range: [{eta_min}, {eta_max}]")
    print(f"  n_etas = {n_etas}, n_samples = {n_samples}")
    print("="*70)
    
    # Collect data
    all_features = []
    all_etas = []
    all_phi = []  # Order parameters for reference
    
    for i, eta in enumerate(etas):
        print(f"\n[{i+1}/{n_etas}] η = {eta:.3f}: ", end="", flush=True)
        
        for s in range(n_samples):
            configs, order_params = run_vicsek(
                N, L, v, R, eta, n_equilibrate, n_measure, n_measurements
            )
            
            feats = extract_vicsek_features(configs, L)
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
    
    print(f"\nCollected {len(all_features)} samples")
    print(f"Feature shape: {all_features.shape}")
    
    # ==========================================================================
    # PCA Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("PCA Analysis")
    print("="*70)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    pca = PCA(n_components=min(9, all_features.shape[1]))
    pca_coords = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance ratios: {pca.explained_variance_ratio_[:4]}")
    
    PC1 = pca_coords[:, 0]
    PC2 = pca_coords[:, 1] if pca_coords.shape[1] > 1 else np.zeros_like(PC1)
    
    # ==========================================================================
    # Correlation Analysis
    # ==========================================================================
    
    print("\n" + "="*70)
    print("Correlation with Control Parameter")
    print("="*70)
    
    # PC1 correlations
    r_pc1_eta, p_pc1_eta = pearsonr(PC1, all_etas)
    r_pc1_phi, p_pc1_phi = pearsonr(PC1, all_phi)
    
    # PC2 correlations
    r_pc2_eta, p_pc2_eta = pearsonr(PC2, all_etas)
    r_pc2_phi, p_pc2_phi = pearsonr(PC2, all_phi)
    
    # Phi vs eta (for reference)
    r_phi_eta, _ = pearsonr(all_phi, all_etas)
    
    print(f"\nOrder parameter φ vs η: r = {r_phi_eta:.3f}")
    print(f"\nPC1 correlations:")
    print(f"  vs η:  r = {r_pc1_eta:+.3f} (p = {p_pc1_eta:.2e})")
    print(f"  vs φ:  r = {r_pc1_phi:+.3f} (p = {p_pc1_phi:.2e})")
    
    print(f"\nPC2 correlations:")
    print(f"  vs η:  r = {r_pc2_eta:+.3f} (p = {p_pc2_eta:.2e})")
    print(f"  vs φ:  r = {r_pc2_phi:+.3f} (p = {p_pc2_phi:.2e})")
    
    # Feature loadings
    print("\n" + "="*70)
    print("PC1 Loadings (Physical Interpretation)")
    print("="*70)
    
    feature_names = [
        'local_φ_mean', 'local_φ_var', 'density_var',
        'grad_v_mean', 'grad_v_var',
        'vorticity_mean', 'vorticity_var',
        'divergence_mean', 'divergence_var'
    ]
    
    loadings = pca.components_[0]
    sorted_idx = np.argsort(np.abs(loadings))[::-1]
    
    print("\nPC1 loadings (sorted by magnitude):")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:20s}: {loadings[idx]:+.3f}")
    
    # ==========================================================================
    # Validation
    # ==========================================================================
    
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)
    
    if abs(r_pc1_eta) > 0.9:
        print(f"\n✅ STRONG: PC1 tracks η with r = {r_pc1_eta:.3f}")
        print("   → Method discovers coupling coordinate in active matter!")
        success = True
        strength = "STRONG"
    elif abs(r_pc1_eta) > 0.7:
        print(f"\n⚠️ MODERATE: PC1 tracks η with r = {r_pc1_eta:.3f}")
        print("   → Clear coupling structure but some noise")
        success = True
        strength = "MODERATE"
    else:
        print(f"\n❌ WEAK: PC1 vs η has r = {r_pc1_eta:.3f}")
        print("   → Features may not capture the relevant structure")
        success = False
        strength = "WEAK"
    
    # Check if PC1 is just tracking φ (trivial case)
    if abs(r_pc1_phi) > 0.95 and abs(r_pc1_eta) < 0.9:
        print("\n⚠️ CAVEAT: PC1 may just be tracking order parameter φ")
        print("   → Need to exclude φ-like features (like Exp 52b)")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    
    mode_str = "pilot" if pilot else "full"
    results_dir = f"results_exp53_{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Order parameter vs η
    ax = axes[0, 0]
    ax.scatter(all_etas, all_phi, c=all_etas, cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel('Noise η')
    ax.set_ylabel('Order parameter φ')
    ax.set_title(f'φ vs η (r = {r_phi_eta:.3f})')
    ax.grid(True, alpha=0.3)
    
    # 2. PC1 vs η
    ax = axes[0, 1]
    ax.scatter(all_etas, PC1, c=all_phi, cmap='coolwarm', alpha=0.6, s=30)
    ax.set_xlabel('Noise η')
    ax.set_ylabel('PC1')
    ax.set_title(f'PC1 vs η (r = {r_pc1_eta:.3f})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='φ')
    
    # 3. PC1 vs φ
    ax = axes[0, 2]
    ax.scatter(all_phi, PC1, c=all_etas, cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel('Order parameter φ')
    ax.set_ylabel('PC1')
    ax.set_title(f'PC1 vs φ (r = {r_pc1_phi:.3f})')
    ax.grid(True, alpha=0.3)
    plt.colorbar(ax.collections[0], ax=ax, label='η')
    
    # 4. PC1-PC2 plane
    ax = axes[1, 0]
    sc = ax.scatter(PC1, PC2, c=all_etas, cmap='viridis', alpha=0.6, s=30)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PC1-PC2 plane (color = η)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='η')
    
    # 5. Explained variance
    ax = axes[1, 1]
    n_comp = min(9, len(pca.explained_variance_ratio_))
    ax.bar(range(1, n_comp+1), pca.explained_variance_ratio_[:n_comp])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.grid(True, alpha=0.3)
    
    # 6. PC1 loadings
    ax = axes[1, 2]
    x_pos = np.arange(len(loadings))
    colors = ['green' if l > 0 else 'red' for l in loadings]
    ax.barh(x_pos, loadings, color=colors, alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel('Loading')
    ax.set_title('PC1 Feature Loadings')
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/vicsek_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    output = {
        'experiment': '53',
        'description': 'Vicsek model coupling coordinate discovery',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'N': N,
            'L': L,
            'v': v,
            'R': R,
            'eta_range': [eta_min, eta_max],
            'n_etas': n_etas,
            'n_samples': n_samples,
        },
        'results': {
            'r_PC1_eta': float(r_pc1_eta),
            'r_PC1_phi': float(r_pc1_phi),
            'r_PC2_eta': float(r_pc2_eta),
            'r_phi_eta': float(r_phi_eta),
            'explained_variance': pca.explained_variance_ratio_[:4].tolist(),
            'PC1_loadings': dict(zip(feature_names, loadings.tolist())),
            'strength': strength,
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
    print("FINAL VERDICT")
    print("="*70)
    if success:
        print(f"✅ PC1 discovers coupling coordinate in Vicsek model (r = {r:.3f})")
        print("   → Method has PREDICTIVE power for less-characterized systems!")
    else:
        print(f"⚠️ PC1 shows weak coupling (r = {r:.3f})")
        print("   → May need different features for active matter")
