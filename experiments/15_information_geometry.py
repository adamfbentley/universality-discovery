"""
Experiment 15: Information Geometry of Universality Classes

Hypothesis: Universality classes are classified by information-geometric curvature.
- EW: R ≈ 0 at all scales (observables decouple)
- KPZ: R > 0, invariant under RG (curvature IS the signature)
- BD/EDEN: R ≈ 0 microscale → R_KPZ under coarse-graining

Methods:
1. Compute observables (g, s², ∇²h) from surfaces
2. Estimate Fisher information matrix F for joint distribution P(g, s², ∇²h)
3. Compute Ricci scalar R from F
4. Track R across scales (proper block RG with rescaling)
5. Compare across universality classes
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from simulation.physics_simulation import GrowthModelSimulator

@dataclass
class FisherAnalysis:
    """Container for Fisher information analysis"""
    model_name: str
    scale: int  # block size for coarse-graining
    observables: np.ndarray  # shape (n_samples, 3) for (g, s2, lap)
    fisher_matrix: np.ndarray  # 3x3 Fisher information
    ricci_scalar: float  # Information-geometric curvature
    eigenvalues: np.ndarray  # Eigenvalues of Fisher matrix
    condition_number: float  # Conditioning of F
    decorrelated: bool  # Are variables approximately independent?

def compute_observables_windowed(surface, dt=1):
    """
    Compute local observables from a single surface snapshot.
    Returns flattened array of (g, s², ∇²h) triplets at each spatial point.
    
    Args:
        surface: (T, L) height field
        dt: time step between consecutive frames
    
    Returns:
        (n_points, 3) array of observables
    """
    T, L = surface.shape
    
    # Temporal difference (growth rate)
    g = (surface[1:] - surface[:-1]) / dt  # (T-1, L)
    
    # Spatial slope (forward differences, periodic)
    s = (np.roll(surface[:-1], -1, axis=1) - np.roll(surface[:-1], 1, axis=1)) / 2.0
    s_sq = s**2  # (T-1, L)
    
    # Laplacian (discrete second derivative, periodic)
    lap = (np.roll(surface[:-1], -1, axis=1) + np.roll(surface[:-1], 1, axis=1) - 2*surface[:-1])  # (T-1, L)
    
    # Stack into (n_points, 3)
    observables = np.column_stack([g.flatten(), s_sq.flatten(), lap.flatten()])
    
    return observables

def compute_fisher_matrix(observables, bandwidth=None, max_samples=10000):
    """
    Compute Fisher information matrix for observables using kernel density estimation.
    
    F_ij = E[partial_i log P * partial_j log P]
    
    Approximated numerically:
    1. Estimate P via KDE (subsampled for speed)
    2. Compute score function gradient of log P(x)
    3. Compute F = E[score * score^T]
    
    Args:
        observables: (n_samples, n_dims) array
        bandwidth: KDE bandwidth (auto if None)
        max_samples: Maximum samples for KDE (subsamples if larger)
    
    Returns:
        (n_dims, n_dims) Fisher information matrix
    """
    n_samples, n_dims = observables.shape
    
    # Subsample if too large (for KDE efficiency)
    if n_samples > max_samples:
        idx = np.random.choice(n_samples, max_samples, replace=False)
        obs_subsample = observables[idx]
    else:
        obs_subsample = observables
    
    n_eval = len(obs_subsample)
    
    # Standardize observables for KDE
    obs_mean = obs_subsample.mean(axis=0)
    obs_std = obs_subsample.std(axis=0)
    obs_std[obs_std < 1e-10] = 1.0  # Avoid division by zero
    obs_norm = (obs_subsample - obs_mean) / obs_std
    
    # Fit KDE on subset
    try:
        kde = gaussian_kde(obs_norm.T, bw_method='silverman')
    except:
        # Fallback if KDE fails
        kde = gaussian_kde(obs_norm.T, bw_method=0.1)
    
    # Compute score function (gradient of log P) via finite differences
    # Evaluate on a further subsample for speed
    n_score_eval = min(3000, n_eval)
    score_idx = np.random.choice(n_eval, n_score_eval, replace=False)
    obs_score = obs_norm[score_idx]
    
    eps = 1e-4
    scores = np.zeros((n_score_eval, n_dims))
    
    for i in range(n_dims):
        obs_plus = obs_score.copy()
        obs_plus[:, i] += eps
        obs_minus = obs_score.copy()
        obs_minus[:, i] -= eps
        
        log_p_plus = kde.logpdf(obs_plus.T)
        log_p_minus = kde.logpdf(obs_minus.T)
        
        scores[:, i] = (log_p_plus - log_p_minus) / (2 * eps)
    
    # Fisher = E[score * score^T]
    fisher = scores.T @ scores / n_score_eval
    
    # Rescale back to original coordinates
    scale_matrix = np.diag(1.0 / obs_std)
    fisher = scale_matrix @ fisher @ scale_matrix
    
    return fisher

def compute_ricci_scalar_3d(fisher_matrix):
    """
    Compute Ricci scalar from Fisher information metric for 3D manifold.
    
    For a 3D Riemannian manifold with metric g_ij = F_ij,
    the Ricci scalar R encodes the mean curvature.
    
    Approximation: Use eigenvalue decomposition
    R ≈ sum of normalized principal curvatures
    
    Args:
        fisher_matrix: (3, 3) Fisher information matrix (metric tensor)
    
    Returns:
        float: Ricci scalar (information-geometric curvature)
    """
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(fisher_matrix)
    
    # Filter out near-zero eigenvalues (rank deficiency)
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Approximate Ricci scalar via eigenvalue sum
    # R ~ 1/n * sum(1/λ_i) - trace penalty for degeneracy
    ricci = np.sum(1.0 / eigvals) / len(eigvals)
    
    # Alternative: Use trace ratio (more stable)
    # R ~ log(det(F) / trace(F))
    if np.linalg.matrix_rank(fisher_matrix) == 3:
        det_f = np.linalg.det(fisher_matrix)
        if det_f > 0:
            ricci_alt = np.log(det_f) / np.trace(fisher_matrix)
        else:
            ricci_alt = 0.0
    else:
        ricci_alt = 0.0
    
    # Use eigenvalue-based measure (more interpretable)
    return float(np.mean(1.0 / eigvals)), eigvals

def is_decorrelated(fisher_matrix, threshold=0.3):
    """
    Check if variables are approximately independent (uncorrelated in info-geom sense).
    Returns True if off-diagonal elements of F are small relative to diagonal.
    """
    diag = np.diag(np.diag(fisher_matrix))
    off_diag = fisher_matrix - diag
    
    coupling = np.linalg.norm(off_diag) / (np.linalg.norm(diag) + 1e-10)
    return coupling < threshold

def block_coarse_grain(surface, block_size):
    """
    Block coarse-grain a surface with proper height rescaling.
    
    Procedure:
    1. Average over blocks of size block_size
    2. Rescale height by 1/sqrt(block_size) (variance is additive for noise)
    
    Args:
        surface: (T, L) original surface
        block_size: size of coarse-graining blocks
    
    Returns:
        (T, L_coarse) coarse-grained surface
    """
    T, L = surface.shape
    L_coarse = L // block_size
    
    # Reshape and average
    h_reshaped = surface[:, :L_coarse*block_size].reshape(T, L_coarse, block_size)
    h_coarse = h_reshaped.mean(axis=2)
    
    # Rescale height fluctuations (variance rescaling)
    h_coarse = h_coarse / np.sqrt(block_size)
    
    return h_coarse

def analyze_model_across_scales(model_name, model_type, max_scale=8, L=256, T=500, n_samples=20):
    """
    Analyze a single model across multiple coarse-graining scales.
    
    Args:
        model_name: e.g., 'KPZ', 'EW', etc.
        model_type: string key for model ('ballistic_deposition', 'edwards_wilkinson', etc.)
        max_scale: maximum block size (2, 4, 8, ...)
        L: system size
        T: time steps
        n_samples: samples per scale
    
    Returns:
        list of FisherAnalysis objects
    """
    results = []
    scales = [1] + [2**i for i in range(1, int(np.log2(max_scale))+1)]
    
    for scale in scales:
        print(f"  {model_name} scale={scale}...", end=' ', flush=True)
        
        all_observables = []
        
        for sample in range(n_samples):
            # Generate surface
            simulator = GrowthModelSimulator(width=L, height=T, random_state=42 + sample)
            surface = simulator.generate_trajectory(model_type)
            
            # Coarse-grain if scale > 1
            if scale > 1:
                surface = block_coarse_grain(surface, scale)
            
            # Extract observables
            obs = compute_observables_windowed(surface, dt=1)
            all_observables.append(obs)
        
        # Concatenate all samples
        all_obs = np.vstack(all_observables)
        
        # Remove outliers (>5 sigma)
        mean = all_obs.mean(axis=0)
        std = all_obs.std(axis=0)
        mask = np.all(np.abs(all_obs - mean) < 5*std, axis=1)
        all_obs_clean = all_obs[mask]
        
        # Compute Fisher matrix
        fisher = compute_fisher_matrix(all_obs_clean)
        
        # Compute Ricci scalar
        ricci, eigvals = compute_ricci_scalar_3d(fisher)
        
        # Check decorrelation
        decorrelated = is_decorrelated(fisher, threshold=0.5)
        
        # Condition number
        cond = np.linalg.cond(fisher)
        
        # Store result
        result = FisherAnalysis(
            model_name=model_name,
            scale=scale,
            observables=all_obs_clean,
            fisher_matrix=fisher,
            ricci_scalar=ricci,
            eigenvalues=eigvals,
            condition_number=cond,
            decorrelated=decorrelated
        )
        results.append(result)
        print(f"R={ricci:.4f}, cond={cond:.2e}, decorr={decorrelated}")
    
    return results

def plot_results(all_results):
    """
    Visualize Ricci scalar across scales for all models.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by model
    by_model = {}
    for res in all_results:
        if res.model_name not in by_model:
            by_model[res.model_name] = []
        by_model[res.model_name].append(res)
    
    # Plot 1: Ricci scalar vs scale
    ax = axes[0]
    colors = {'EW': 'blue', 'KPZ': 'red', 'BD': 'green', 'EDEN': 'orange', 'RD': 'purple'}
    
    for model_name, results in sorted(by_model.items()):
        scales = [r.scale for r in results]
        riccius = [r.ricci_scalar for r in results]
        ax.plot(scales, riccius, 'o-', label=model_name, color=colors.get(model_name, 'black'), linewidth=2, markersize=8)
    
    ax.set_xlabel('Coarse-graining Scale', fontsize=12)
    ax.set_ylabel('Ricci Scalar (Information-Geometric Curvature)', fontsize=12)
    ax.set_title('Exp 15: Information Geometry Across Scales', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 2: Eigenvalue spectrum at scale=1
    ax = axes[1]
    x_pos = 0
    for model_name, results in sorted(by_model.items()):
        # Find scale=1 result
        scale_1 = [r for r in results if r.scale == 1][0]
        eigvals = scale_1.eigenvalues
        
        ax.bar([x_pos, x_pos+0.25, x_pos+0.5], eigvals, width=0.2, label=model_name, color=colors.get(model_name, 'black'))
        x_pos += 1
    
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Fisher Matrix Eigenvalues (scale=1)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('exp15_information_geometry.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to exp15_information_geometry.png")
    plt.close()

def main():
    print("="*70)
    print("EXPERIMENT 15: Information Geometry of Universality Classes")
    print("="*70)
    print()
    print("Testing hypothesis: Universality classes differ in information-geometric curvature")
    print("  - EW class: R ~ 0 (decoupled observables)")
    print("  - KPZ class: R > 0 (coupled observables)")
    print("  - Discrete models: R evolves from ~0 to R_KPZ under coarse-graining")
    print()
    
    models = {
        'EW': 'edwards_wilkinson',
        'KPZ': 'kpz_equation',
        'BD': 'ballistic_deposition',
        'EDEN': 'eden',
        'RD': 'random_deposition',
    }
    
    all_results = []
    
    for model_name, model_type in models.items():
        print(f"\n{model_name}:")
        results = analyze_model_across_scales(
            model_name, model_type,
            max_scale=8,
            L=128, T=300, n_samples=8
        )
        all_results.extend(results)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Group by model
    by_model = {}
    for res in all_results:
        if res.model_name not in by_model:
            by_model[res.model_name] = []
        by_model[res.model_name].append(res)
    
    for model_name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        if model_name in by_model:
            print(f"\n{model_name}:")
            results = by_model[model_name]
            
            # Show Ricci evolution
            print(f"  Ricci scalar evolution:")
            for r in results:
                print(f"    scale={r.scale:2d}: R={r.ricci_scalar:8.5f}, decorr={r.decorrelated}, cond={r.condition_number:.2e}")
            
            # Trend analysis
            scales = np.array([r.scale for r in results])
            riccius = np.array([r.ricci_scalar for r in results])
            
            if len(scales) > 1:
                # Linear regression in log-log
                coeffs = np.polyfit(np.log(scales), riccius, 1)
                trend = "increasing" if coeffs[0] > 0.01 else "decreasing" if coeffs[0] < -0.01 else "flat"
                print(f"  Trend: {trend} (slope in log-log: {coeffs[0]:.4f})")
    
    # Cross-class comparison at microscale (scale=1)
    print("\n" + "-"*70)
    print("Cross-class comparison at scale=1 (microscale):")
    for model_name in ['EW', 'KPZ', 'BD', 'EDEN', 'RD']:
        if model_name in by_model:
            result = [r for r in by_model[model_name] if r.scale == 1][0]
            print(f"  {model_name:5s}: R={result.ricci_scalar:.5f}, decorr={result.decorrelated}")
    
    print("\n" + "-"*70)
    print("Theoretical interpretation:")
    print()
    print("Expected pattern:")
    print("  • EW:       R ~ 0 always (no coupling of g, s^2, Laplacian)")
    print("  • KPZ:      R > 0 always (coupled via nonlinearity)")
    print("  • BD/EDEN:  R ~ 0 at scale=1, R -> R_KPZ as scale increases")
    print()
    print("If pattern holds -> universality classes ARE topological invariants")
    print("                   of the information manifold.")
    print()
    
    # Visualization
    plot_results(all_results)
    
    print("\n" + "="*70)
    print("Experiment 15 complete!")
    print("="*70)

if __name__ == '__main__':
    main()
