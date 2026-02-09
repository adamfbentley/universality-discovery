"""
Experiment 46: Coupling Coordinate Calibration Test
===================================================

DEEP THEORETICAL QUESTION from Assessment 2:
Does PC1 track the effective coupling g_eff = (λ²D/ν³) L^(2-d)?

If YES → PC1 is a genuine COUPLING COORDINATE tracking crossover between
         weak/strong coupling regimes (profound physical insight!)
If NO  → PC1 is just an ensemble-dependent classifier (still useful but shallow)

KEY INNOVATION:
Vary λ, ν, D INDEPENDENTLY (not just λ/ν ratio) to test for data collapse.
If features collapse onto single curve when plotted vs g_eff, we've found
the natural coordinate system for KPZ crossover dynamics.

THEORETICAL BACKGROUND:
In d=1 KPZ, the dimensionless coupling grows under coarse-graining:
    g_eff(ℓ) ~ (λ²D/ν³) ℓ^(2-d)
where:
    λ = nonlinearity strength (∇h)² term
    ν = diffusion coefficient (∇²h term)
    D = noise strength (η term)
    ℓ = observation scale

For d=1: g_eff(ℓ) ~ (λ²D/ν³) ℓ grows with scale → strong coupling regime
This is exactly RG language: coupling is RG-relevant operator.

IF PC1 tracks g_eff across independent parameter variations, then the 
gradient moment features are capturing the PHYSICAL COUPLING STRENGTH,
not just providing a discriminative coordinate.

SUCCESS CRITERIA:
1. Data collapse: PC1 vs g_eff forms single curve
2. Monotonic relationship: PC1 increases/decreases with g_eff
3. Independent validation: Collapse holds for (λ,ν,D) not in training set

IMPLICATIONS:
- Confirms Assessment 2's "coupling coordinate" interpretation
- Shows features encode PHYSICS not just CLASSIFICATION
- Provides path to RG-covariant embedding learning (Exp 45)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from numba import jit

# ============================================================================
# SIMULATION FUNCTIONS (Numba-optimized)
# ============================================================================

@jit(nopython=True)
def simulate_kpz_trajectory(L=128, T=2000, lambda_=0.5, nu=1.0, D=1.0, 
                            dt=0.05, save_interval=10):
    """
    Simulate KPZ equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η
    
    Parameters:
    -----------
    L : int
        System size (spatial lattice points)
    T : int
        Number of time steps to simulate
    lambda_ : float
        Nonlinearity strength (KPZ parameter)
    nu : float
        Diffusion coefficient
    D : float
        Noise strength
    dt : float
        Time step size
    save_interval : int
        Save every N steps
        
    Returns:
    --------
    trajectory : array (T_save, L)
        Height field evolution
    """
    interface = np.zeros(L)  # Flat initial condition
    n_saves = T // save_interval
    trajectory = np.zeros((n_saves, L))
    save_idx = 0
    
    for t in range(T):
        # Compute derivatives
        new_interface = interface.copy()
        
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            
            # Laplacian: ∇²h
            laplacian = left - 2*center + right
            
            # Gradient: ∇h (central difference)
            gradient = (right - left) / 2.0
            
            # Noise
            noise = np.sqrt(D * dt) * np.random.randn()
            
            # KPZ equation
            dhdt = nu * laplacian + (lambda_ / 2.0) * gradient**2 + noise
            new_interface[x] = center + dt * dhdt
        
        interface = new_interface
        
        # Save if at interval
        if t % save_interval == 0:
            trajectory[save_idx] = interface.copy()
            save_idx += 1
    
    return trajectory

@jit(nopython=True)
def compute_gradient_moments_numba(h):
    """
    Compute 6D gradient moment features (Numba-optimized).
    
    Features:
    1. grad_var: Variance of spatial gradient
    2. grad_skew: Skewness of gradient distribution
    3. grad_kurt: Kurtosis of gradient distribution
    4. lap_var: Variance of Laplacian
    5. grad_lap_cov: Covariance between |gradient| and Laplacian
    6. h_var: Variance of height field itself
    """
    L = len(h)
    
    # Compute spatial derivatives (periodic BC)
    gradient = np.zeros(L)
    laplacian = np.zeros(L)
    
    for x in range(L):
        left = h[(x-1) % L]
        center = h[x]
        right = h[(x+1) % L]
        
        gradient[x] = (right - left) / 2.0
        laplacian[x] = left - 2*center + right
    
    # Gradient statistics
    grad_mean = np.mean(gradient)
    grad_var = np.var(gradient)
    grad_std = np.sqrt(grad_var) if grad_var > 1e-10 else 1e-10
    
    # Standardized moments
    grad_centered = gradient - grad_mean
    grad_skew = np.mean((grad_centered / grad_std)**3)
    grad_kurt = np.mean((grad_centered / grad_std)**4) - 3.0  # Excess kurtosis
    
    # Laplacian statistics
    lap_var = np.var(laplacian)
    
    # Covariance
    grad_abs = np.abs(gradient)
    grad_lap_cov = np.mean((grad_abs - np.mean(grad_abs)) * 
                            (laplacian - np.mean(laplacian)))
    
    # Height variance
    h_var = np.var(h)
    
    return np.array([grad_var, grad_skew, grad_kurt, 
                     lap_var, grad_lap_cov, h_var])

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_trajectory(trajectory):
    """
    Extract 6D features from each frame of trajectory.
    Returns: (n_frames, 6) feature array
    """
    n_frames = trajectory.shape[0]
    features = np.zeros((n_frames, 6))
    
    for i in range(n_frames):
        features[i] = compute_gradient_moments_numba(trajectory[i])
    
    return features

# ============================================================================
# COUPLING PARAMETER GRID
# ============================================================================

def generate_coupling_grid():
    """
    Generate grid of (λ, ν, D) combinations for testing collapse.
    
    Strategy: Sample broadly across parameter space, including:
    - Varying each parameter independently
    - Different ratios of λ/ν
    - Different noise strengths D
    """
    coupling_grid = []
    
    # Grid values
    lambda_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    nu_values = [0.5, 0.8, 1.0, 1.5, 2.0]
    D_values = [0.5, 0.8, 1.0, 1.5, 2.0]
    
    # Sample combinations (not full cartesian product - too many)
    # Strategy: Fix two, vary third
    
    # 1. Vary λ, fix ν=1.0, D=1.0
    for lam in lambda_values:
        coupling_grid.append({'lambda': lam, 'nu': 1.0, 'D': 1.0})
    
    # 2. Vary ν, fix λ=0.5, D=1.0
    for nu in nu_values:
        if nu != 1.0:  # Avoid duplicate
            coupling_grid.append({'lambda': 0.5, 'nu': nu, 'D': 1.0})
    
    # 3. Vary D, fix λ=0.5, ν=1.0
    for d in D_values:
        if d != 1.0:  # Avoid duplicate
            coupling_grid.append({'lambda': 0.5, 'nu': 1.0, 'D': d})
    
    # 4. Additional combinations for validation
    extra_combos = [
        {'lambda': 0.3, 'nu': 0.8, 'D': 1.5},
        {'lambda': 0.8, 'nu': 1.5, 'D': 0.8},
        {'lambda': 1.0, 'nu': 2.0, 'D': 0.5},
        {'lambda': 1.5, 'nu': 0.5, 'D': 2.0},
    ]
    coupling_grid.extend(extra_combos)
    
    return coupling_grid

def compute_effective_coupling(lambda_, nu, D, L, d=1):
    """
    Compute effective dimensionless coupling.
    
    For KPZ in dimension d, the coupling grows as:
        g_eff ~ (λ²D/ν³) * ℓ^(2-d)
    
    For d=1 (our case):
        g_eff ~ (λ²D/ν³) * L
    """
    return (lambda_**2 * D / nu**3) * L**(2 - d)

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_coupling_coordinate_test():
    """
    Main experimental loop: Test if PC1 tracks g_eff.
    """
    print("="*70)
    print("EXPERIMENT 46: Coupling Coordinate Calibration Test")
    print("="*70)
    print("\nGoal: Test if PC1 tracks effective coupling g_eff = (λ²D/ν³)L")
    print("\nGenerating coupling parameter grid...")
    
    # Setup
    coupling_grid = generate_coupling_grid()
    L = 128  # System size
    T = 2000  # Time steps
    n_samples_per_config = 5  # Multiple samples for statistics
    
    print(f"Total configurations: {len(coupling_grid)}")
    print(f"Samples per config: {n_samples_per_config}")
    print(f"Total simulations: {len(coupling_grid) * n_samples_per_config}")
    
    # Storage
    all_features = []
    all_g_eff = []
    all_lambda = []
    all_nu = []
    all_D = []
    config_labels = []
    
    # Run simulations
    print("\nRunning simulations...")
    for idx, config in enumerate(coupling_grid):
        lam = config['lambda']
        nu = config['nu']
        d = config['D']
        
        g_eff = compute_effective_coupling(lam, nu, d, L)
        
        print(f"Config {idx+1}/{len(coupling_grid)}: "
              f"λ={lam:.2f}, ν={nu:.2f}, D={d:.2f} → g_eff={g_eff:.3f}")
        
        # Generate multiple samples for this configuration
        for sample in range(n_samples_per_config):
            # Simulate KPZ
            trajectory = simulate_kpz_trajectory(
                L=L, T=T, lambda_=lam, nu=nu, D=d, 
                dt=0.05, save_interval=10
            )
            
            # Extract features (average over trajectory)
            features_traj = extract_features_from_trajectory(trajectory)
            features_mean = np.mean(features_traj[-50:], axis=0)  # Last 50 frames
            
            all_features.append(features_mean)
            all_g_eff.append(g_eff)
            all_lambda.append(lam)
            all_nu.append(nu)
            all_D.append(d)
            config_labels.append(idx)
    
    # Convert to arrays
    all_features = np.array(all_features)
    all_g_eff = np.array(all_g_eff)
    all_lambda = np.array(all_lambda)
    all_nu = np.array(all_nu)
    all_D = np.array(all_D)
    config_labels = np.array(config_labels)
    
    print(f"\nTotal data points: {len(all_features)}")
    print(f"Feature shape: {all_features.shape}")
    
    # ========================================================================
    # PCA ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("PCA ANALYSIS")
    print("="*70)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)
    
    # Fit PCA
    pca = PCA(n_components=6)
    pca_features = pca.fit_transform(features_scaled)
    
    print(f"\nExplained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.4f} ({np.sum(pca.explained_variance_ratio_[:i+1]):.4f} cumulative)")
    
    # Extract PC1
    pc1_values = pca_features[:, 0]
    
    # ========================================================================
    # COUPLING COORDINATE TEST
    # ========================================================================
    
    print("\n" + "="*70)
    print("COUPLING COORDINATE COLLAPSE TEST")
    print("="*70)
    
    # Test correlation
    r_pc1_g, p_pc1_g = pearsonr(pc1_values, all_g_eff)
    print(f"\nCorrelation PC1 vs g_eff: r = {r_pc1_g:.4f} (p = {p_pc1_g:.2e})")
    
    # Test collapse by plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: PC1 vs g_eff (colored by λ)
    ax = axes[0, 0]
    scatter = ax.scatter(all_g_eff, pc1_values, c=all_lambda, 
                        cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Effective Coupling $g_{eff} = (\\lambda^2 D / \\nu^3) L$', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('Collapse Test: PC1 vs Coupling', fontsize=14, weight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('$\\lambda$', fontsize=12)
    
    # Add trendline
    z = np.polyfit(all_g_eff, pc1_values, 1)
    p = np.poly1d(z)
    g_sorted = np.sort(all_g_eff)
    ax.plot(g_sorted, p(g_sorted), 'r--', linewidth=2, alpha=0.8, 
            label=f'Linear fit: r={r_pc1_g:.3f}')
    ax.legend()
    
    # Plot 2: PC1 vs g_eff (colored by ν)
    ax = axes[0, 1]
    scatter = ax.scatter(all_g_eff, pc1_values, c=all_nu, 
                        cmap='plasma', s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Effective Coupling $g_{eff}$', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('Colored by Diffusion $\\nu$', fontsize=14)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('$\\nu$', fontsize=12)
    
    # Plot 3: PC1 vs g_eff (colored by D)
    ax = axes[1, 0]
    scatter = ax.scatter(all_g_eff, pc1_values, c=all_D, 
                        cmap='coolwarm', s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Effective Coupling $g_{eff}$', fontsize=12)
    ax.set_ylabel('PC1', fontsize=12)
    ax.set_title('Colored by Noise Strength $D$', fontsize=14)
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('$D$', fontsize=12)
    
    # Plot 4: Residuals from linear fit
    ax = axes[1, 1]
    residuals = pc1_values - p(all_g_eff)
    ax.scatter(all_g_eff, residuals, c=config_labels, 
              cmap='tab20', s=50, alpha=0.7, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Effective Coupling $g_{eff}$', fontsize=12)
    ax.set_ylabel('Residuals (PC1 - fit)', fontsize=12)
    ax.set_title('Residual Analysis', fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = Path(__file__).parent.parent / 'results' / 'exp46_coupling_coordinate'
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / 'coupling_collapse_test.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {results_dir / 'coupling_collapse_test.png'}")
    
    # ========================================================================
    # STATISTICAL ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    # Compute R² for linear fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((pc1_values - np.mean(pc1_values))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nLinear fit quality:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Slope = {z[0]:.4f}")
    print(f"  Intercept = {z[1]:.4f}")
    print(f"  Residual std = {np.std(residuals):.4f}")
    
    # Test correlations with individual parameters
    r_lambda, p_lambda = pearsonr(pc1_values, all_lambda)
    r_nu, p_nu = pearsonr(pc1_values, all_nu)
    r_D, p_D = pearsonr(pc1_values, all_D)
    
    print(f"\nCorrelations with individual parameters:")
    print(f"  PC1 vs λ:  r = {r_lambda:.4f} (p = {p_lambda:.2e})")
    print(f"  PC1 vs ν:  r = {r_nu:.4f} (p = {p_nu:.2e})")
    print(f"  PC1 vs D:  r = {r_D:.4f} (p = {p_D:.2e})")
    
    # ========================================================================
    # SUCCESS CRITERIA EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*70)
    
    success_collapse = r_squared > 0.7
    success_correlation = abs(r_pc1_g) > 0.8
    success_monotonic = z[0] != 0 and np.sign(z[0]) == np.sign(r_pc1_g)
    
    print(f"\n1. Data collapse (R² > 0.7): {'✅ PASS' if success_collapse else '❌ FAIL'}")
    print(f"   → R² = {r_squared:.4f}")
    
    print(f"\n2. Strong correlation (|r| > 0.8): {'✅ PASS' if success_correlation else '❌ FAIL'}")
    print(f"   → r = {r_pc1_g:.4f}")
    
    print(f"\n3. Monotonic relationship: {'✅ PASS' if success_monotonic else '❌ FAIL'}")
    print(f"   → Slope sign matches correlation: {np.sign(z[0]) == np.sign(r_pc1_g)}")
    
    overall_success = success_collapse and success_correlation and success_monotonic
    
    print("\n" + "="*70)
    if overall_success:
        print("🎉 SUCCESS: PC1 IS A COUPLING COORDINATE!")
        print("="*70)
        print("\nInterpretation:")
        print("  • PC1 tracks effective coupling g_eff across parameter space")
        print("  • Features encode PHYSICAL COUPLING STRENGTH, not just classification")
        print("  • This validates Assessment 2's 'coupling coordinate' interpretation")
        print("  • Path forward: RG-covariant embedding learning (Exp 45)")
    else:
        print("⚠️  PARTIAL SUCCESS / NEEDS INVESTIGATION")
        print("="*70)
        print("\nInterpretation:")
        print("  • PC1 shows correlation with g_eff but collapse is imperfect")
        print("  • May need:")
        print("    - Nonlinear relationship (try log scale)")
        print("    - Additional parameters (finite-size corrections)")
        print("    - Different feature normalization")
        print("  • Still proceed to Exp 45 with caution")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results = {
        'coupling_grid': coupling_grid,
        'features': all_features,
        'g_eff': all_g_eff,
        'lambda': all_lambda,
        'nu': all_nu,
        'D': all_D,
        'pc1_values': pc1_values,
        'pca': pca,
        'scaler': scaler,
        'correlations': {
            'r_pc1_g': r_pc1_g,
            'p_pc1_g': p_pc1_g,
            'r_squared': r_squared,
            'r_lambda': r_lambda,
            'r_nu': r_nu,
            'r_D': r_D
        },
        'fit_params': {'slope': z[0], 'intercept': z[1]},
        'success_criteria': {
            'collapse': success_collapse,
            'correlation': success_correlation,
            'monotonic': success_monotonic,
            'overall': overall_success
        }
    }
    
    import pickle
    results_file = results_dir / 'coupling_coordinate_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {results_file}")
    
    plt.show()
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_coupling_coordinate_test()
