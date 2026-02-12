"""
Experiment 60: Sensitivity Matrix Analysis (Fisher-Approximation)
==================================================================

WHAT THIS COMPUTES:
    A Gaussian-approximation sensitivity matrix (Machta et al. 2013):
        F_hat = J^T * Sigma_ref^{-1} * J
    where J = d(mu)/d(theta) is the Jacobian of mean features w.r.t. parameters
    and Sigma_ref is a reference covariance (averaged over parameter grid).

    This equals the true Fisher information ONLY when:
    - Feature distribution is approximately Gaussian
    - theta-dependence is concentrated in the mean
    - Sigma is approximately constant across theta

    For KPZ (Gaussian gradient measure = exponential family), this is exact.
    For Ising/Potts, it's an approximation.

    See docs/FISHER_RG_THEORY.md for the theoretical framework.

IMPORTANT - THREE DIFFERENT VECTORS:
    1. PCA PC1 = eigenvector of Cov_pooled (max variance direction)
    2. Mean-tangent = J * v1 (direction mu moves along stiffest theta)
    3. Whitened tangent = Sigma^{-1} * (J * v1) (distinguishability direction)
    Parts A/C compare (2) vs historical PC1. Do NOT conflate with (3).

THIS EXPERIMENT TESTS:
    PART A: KPZ sensitivity matrix
        - Compute F_hat numerically by varying (D, nu, lam)
        - Predict: dominant eigenvector aligns with D/nu direction
        - Predict: lam-direction eigenvalue ~ 0
        - Compare mean-tangent J*v1 to PCA loadings from Exp 46/54

    PART B: Sensitivity matrix under coarse-graining  
        - Recompute F_hat at scales b = 1, 2, 4
        - RESULT: eigenvalue ratio DECREASES (original prediction falsified)
        - Direction stability is the meaningful signal

    PART C: Ising sensitivity matrix
        - Compute F_hat by varying (T, h) near T_c
        - Compare mean-tangent J*v1 to Exp 52b PC1 loadings

    PART D: Cross-system comparison (TODO - not yet implemented)
        - Must use identical comparison objects across systems
        - See docs/FISHER_PCA_CONSISTENCY_CHECKLIST.md

PREDICTIONS (falsifiable):
    P1: KPZ sensitivity eigenvalue for D/nu >> eigenvalue for lam  [CONFIRMED]
    P2: Mean-tangent J*v1 aligns with PCA PC1 (cosine > 0.7)      [CONFIRMED ~0.80]
    P3: Eigenvalue ratio increases under coarse-graining            [FALSIFIED]
    P4: Ising mean-tangent = thermal direction                      [CONFIRMED ~0.75]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Fix Unicode output on Windows (cp1252 consoles can't print ν, λ, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from numba import jit

RESULTS_DIR = Path(__file__).parent.parent / 'results_exp60'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = ['grad_var', 'grad_skew', 'grad_kurt', 'lap_var', 'grad_lap_cov', 'h_var']

# ============================================================================
# KPZ SIMULATION (from Exp 54)
# ============================================================================

@jit(nopython=True)
def simulate_kpz_trajectory(L=128, T=2000, lambda_=0.5, nu=1.0, D=1.0, 
                            dt=0.05, save_interval=10):
    """Simulate KPZ equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η"""
    interface = np.zeros(L)
    n_saves = T // save_interval
    trajectory = np.zeros((n_saves, L))
    save_idx = 0
    for t in range(T):
        new_interface = interface.copy()
        for x in range(L):
            left = interface[(x-1) % L]
            center = interface[x]
            right = interface[(x+1) % L]
            laplacian = left - 2*center + right
            gradient = (right - left) / 2.0
            noise = np.sqrt(D * dt) * np.random.randn()
            dhdt = nu * laplacian + (lambda_ / 2.0) * gradient**2 + noise
            new_interface[x] = center + dt * dhdt
        interface = new_interface
        if t % save_interval == 0:
            trajectory[save_idx] = interface.copy()
            save_idx += 1
    return trajectory

@jit(nopython=True)
def compute_gradient_moments_numba(h):
    """Compute 6D gradient moment features."""
    L = len(h)
    gradient = np.zeros(L)
    laplacian = np.zeros(L)
    for x in range(L):
        left = h[(x-1) % L]
        center = h[x]
        right = h[(x+1) % L]
        gradient[x] = (right - left) / 2.0
        laplacian[x] = left - 2*center + right
    grad_mean = np.mean(gradient)
    grad_var = np.var(gradient)
    grad_std = np.sqrt(grad_var) if grad_var > 1e-10 else 1e-10
    grad_centered = gradient - grad_mean
    grad_skew = np.mean((grad_centered / grad_std)**3)
    grad_kurt = np.mean((grad_centered / grad_std)**4) - 3.0
    lap_var = np.var(laplacian)
    grad_abs = np.abs(gradient)
    grad_lap_cov = np.mean((grad_abs - np.mean(grad_abs)) * 
                           (laplacian - np.mean(laplacian)))
    h_var = np.var(h)
    return np.array([grad_var, grad_skew, grad_kurt, 
                     lap_var, grad_lap_cov, h_var])


def coarse_grain(h, factor):
    """Coarse-grain interface by block averaging."""
    L = len(h)
    new_L = L // factor
    return np.array([h[i*factor:(i+1)*factor].mean() for i in range(new_L)])


def extract_features_ensemble(L, T, lambda_, nu, D, n_samples, dt=0.05,
                              save_interval=10, late_frac=0.3, cg_factor=1):
    """Generate n_samples feature vectors at given parameters.
    
    Returns: (n_samples, 6) array of feature vectors.
    Each sample = one trajectory's late-time averaged features.
    """
    features = []
    for _ in range(n_samples):
        traj = simulate_kpz_trajectory(L=L, T=T, lambda_=lambda_, 
                                        nu=nu, D=D, dt=dt, 
                                        save_interval=save_interval)
        n_frames = traj.shape[0]
        start = int(n_frames * (1 - late_frac))
        
        sample_feats = []
        for i in range(start, n_frames):
            h = traj[i]
            if cg_factor > 1:
                h = coarse_grain(h, cg_factor)
            sample_feats.append(compute_gradient_moments_numba(h))
        features.append(np.mean(sample_feats, axis=0))
    
    return np.array(features)


# ============================================================================
# 2D ISING SIMULATION (from Exp 52b)
# ============================================================================

@jit(nopython=True)
def ising_wolff_step(spins, beta):
    """One Wolff cluster flip."""
    L = spins.shape[0]
    x0 = np.random.randint(0, L)
    y0 = np.random.randint(0, L)
    s0 = spins[x0, y0]
    
    p_add = 1.0 - np.exp(-2.0 * beta)  # Ising bond probability
    
    cluster = [(x0, y0)]
    in_cluster = np.zeros((L, L), dtype=np.int8)
    in_cluster[x0, y0] = 1
    
    idx = 0
    while idx < len(cluster):
        cx, cy = cluster[idx]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = (cx + dx) % L
            ny = (cy + dy) % L
            if in_cluster[nx, ny] == 0 and spins[nx, ny] == s0:
                if np.random.random() < p_add:
                    cluster.append((nx, ny))
                    in_cluster[nx, ny] = 1
        idx += 1
    
    for cx, cy in cluster:
        spins[cx, cy] *= -1
    
    return spins


@jit(nopython=True)
def ising_local_features(spins, h_field=0.0):
    """Compute 6D local features for Ising (same as Exp 52b, no |m| or E/N).
    
    Features:
        0: |∇m| — mean absolute gradient of magnetization
        1: corr_1 — nearest-neighbor spin correlation
        2: boundary_density — fraction of unlike-neighbor pairs
        3: |m_local| — mean absolute local magnetization (2x2 blocks)
        4: Var(|∇m|) — variance of local gradient
        5: Var(m_local) — variance of local magnetization
    """
    L = spins.shape[0]
    
    # Gradient magnitude
    grad_vals = np.zeros(L * L)
    idx = 0
    for i in range(L):
        for j in range(L):
            gx = (spins[(i+1) % L, j] - spins[(i-1) % L, j]) / 2.0
            gy = (spins[i, (j+1) % L] - spins[i, (j-1) % L]) / 2.0
            grad_vals[idx] = np.sqrt(gx**2 + gy**2)
            idx += 1
    grad_mean = np.mean(grad_vals)
    grad_var = np.var(grad_vals)
    
    # Nearest-neighbor correlation
    corr = 0.0
    count = 0
    for i in range(L):
        for j in range(L):
            corr += spins[i, j] * spins[(i+1) % L, j]
            corr += spins[i, j] * spins[i, (j+1) % L]
            count += 2
    corr /= count
    
    # Boundary density
    bd = 0.0
    count2 = 0
    for i in range(L):
        for j in range(L):
            if spins[i, j] != spins[(i+1) % L, j]:
                bd += 1
            if spins[i, j] != spins[i, (j+1) % L]:
                bd += 1
            count2 += 2
    bd /= count2
    
    # Local magnetization (2x2 blocks)
    n_blocks = (L // 2) ** 2
    m_local = np.zeros(n_blocks)
    bidx = 0
    for i in range(0, L, 2):
        for j in range(0, L, 2):
            m_local[bidx] = (spins[i, j] + spins[i+1, j] + 
                            spins[i, j+1] + spins[i+1, j+1]) / 4.0
            bidx += 1
    m_local_abs_mean = np.mean(np.abs(m_local))
    m_local_var = np.var(m_local)
    
    return np.array([grad_mean, corr, bd, m_local_abs_mean, grad_var, m_local_var])


def generate_ising_features(L, T, h_field, n_samples, n_therm=500, n_between=5):
    """Generate feature samples for Ising at given (T, h).
    
    Note: h_field is an external field. For h≠0, the energy is:
        E = -J Σ s_i s_j - h Σ s_i
    We implement this via modified Wolff (only valid for h=0) or Metropolis.
    For h≠0, we use single-spin Metropolis.
    """
    beta = 1.0 / T
    spins = np.ones((L, L), dtype=np.float64)  # Start ordered
    
    if abs(h_field) < 1e-10:
        # Wolff cluster for h=0
        for _ in range(n_therm):
            spins = ising_wolff_step(spins, beta)
        
        features = []
        for _ in range(n_samples):
            for _ in range(n_between):
                spins = ising_wolff_step(spins, beta)
            features.append(ising_local_features(spins))
        return np.array(features)
    else:
        # Metropolis for h≠0
        spins = _ising_metropolis_ensemble(spins, beta, h_field, 
                                           n_therm, n_samples, n_between)
        features = []
        for config in spins:
            features.append(ising_local_features(config))
        return np.array(features)


@jit(nopython=True)
def _ising_metropolis_sweep(spins, beta, h_field):
    """One Metropolis sweep with external field."""
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        s = spins[i, j]
        nn_sum = (spins[(i+1) % L, j] + spins[(i-1) % L, j] + 
                  spins[i, (j+1) % L] + spins[i, (j-1) % L])
        dE = 2.0 * s * (nn_sum + h_field)
        if dE <= 0 or np.random.random() < np.exp(-beta * dE):
            spins[i, j] = -s
    return spins


@jit(nopython=True)
def _ising_metropolis_ensemble(spins, beta, h_field, n_therm, n_samples, n_between):
    """Generate ensemble via Metropolis."""
    for _ in range(n_therm):
        spins = _ising_metropolis_sweep(spins, beta, h_field)
    
    configs = np.zeros((n_samples, spins.shape[0], spins.shape[1]))
    for s in range(n_samples):
        for _ in range(n_between):
            spins = _ising_metropolis_sweep(spins, beta, h_field)
        configs[s] = spins.copy()
    return configs


# ============================================================================
# FISHER INFORMATION ESTIMATION
# ============================================================================

def estimate_fisher_matrix(param_grid, feature_fn, param_names, 
                           normalize=True):
    """Estimate Fisher information matrix via parameter sensitivity.
    
    The Fisher information can be approximated as:
        F_{ij} ≈ Σ_k (∂⟨Φ_k⟩/∂θ_i) · (1/Var[Φ_k]) · (∂⟨Φ_k⟩/∂θ_j)
    
    More precisely, using the full covariance:
        F_{ij} = (∂μ/∂θ_i)^T Σ^{-1} (∂μ/∂θ_j)
    
    where μ = E[Φ] and Σ = Cov[Φ] at a reference point.
    
    This is the "sensitivity matrix" approach from Machta et al. (2013).
    
    Args:
        param_grid: list of dicts, each with parameter values and features
            [{'params': [θ1, θ2, ...], 'mean_features': array, 'cov_features': array}, ...]
        feature_fn: not used here (features pre-computed in param_grid)
        param_names: list of parameter names
        normalize: whether to normalize features before computing F
    
    Returns:
        F: (n_params, n_params) Fisher information matrix
        jacobian: (n_features, n_params) sensitivity matrix ∂μ/∂θ
    """
    n_params = len(param_names)
    
    # Extract parameter values and mean features
    params_array = np.array([g['params'] for g in param_grid])
    means_array = np.array([g['mean_features'] for g in param_grid])
    n_features = means_array.shape[1]
    
    # Use reference covariance (average across all parameter points)
    cov_ref = np.mean([g['cov_features'] for g in param_grid], axis=0)
    if normalize:
        # Normalize features by their std
        stds = np.sqrt(np.diag(cov_ref))
        stds[stds < 1e-10] = 1.0
        means_array = means_array / stds
        cov_ref = cov_ref / np.outer(stds, stds)
    
    # Estimate Jacobian ∂μ/∂θ via least squares on the parameter grid
    # For each feature, regress mean_feature ~ θ
    # J[k, i] = ∂⟨Φ_k⟩/∂θ_i
    jacobian = np.zeros((n_features, n_params))
    
    # Center parameters
    param_center = np.mean(params_array, axis=0)
    dparams = params_array - param_center
    
    for k in range(n_features):
        # Linear regression: mean_feature_k = a + J_k · dθ
        # Using lstsq
        A = np.column_stack([dparams, np.ones(len(dparams))])
        result = np.linalg.lstsq(A, means_array[:, k], rcond=None)
        jacobian[k, :] = result[0][:n_params]
    
    # Fisher matrix: F = J^T Σ^{-1} J
    cov_reg = cov_ref + 1e-6 * np.eye(n_features)
    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_reg)
    
    F = jacobian.T @ cov_inv @ jacobian
    
    return F, jacobian


def compare_eigenvectors(v1, v2):
    """Cosine similarity between two vectors (absolute, since sign is arbitrary)."""
    return abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ============================================================================
# PART A: KPZ FISHER MATRIX
# ============================================================================

def run_part_a(pilot=False):
    """Compute Fisher information matrix for KPZ in (D, ν, λ) space."""
    print("=" * 70)
    print("PART A: KPZ FISHER INFORMATION MATRIX")
    print("=" * 70)
    print()
    print("Theory predicts:")
    print("  - Dominant Fisher eigenvalue in D/ν direction")
    print("  - λ-direction eigenvalue ≈ 0 (λ invisible to gradient moments)")
    print("  - Fisher eigenvector 1 ∝ PCA PC1 loadings")
    print()
    
    L = 128
    T = 3000 if not pilot else 1500
    n_samples = 20 if not pilot else 8
    
    # Parameter grid: vary D, ν, λ around reference point
    # Reference: D=1.0, ν=1.0, λ=1.0
    if pilot:
        D_vals = [0.5, 1.0, 2.0]
        nu_vals = [0.5, 1.0, 2.0]
        lam_vals = [0.5, 1.0, 2.0]
    else:
        D_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
        nu_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
        lam_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"Parameter grid: {len(D_vals)} D × {len(nu_vals)} ν × {len(lam_vals)} λ "
          f"= {len(D_vals)*len(nu_vals)*len(lam_vals)} points")
    print(f"Samples per point: {n_samples}")
    print()
    
    # Compile Numba
    print("Compiling Numba functions...")
    _ = simulate_kpz_trajectory(L=16, T=100, lambda_=1.0, nu=1.0, D=1.0, dt=0.05)
    _ = compute_gradient_moments_numba(np.random.randn(16))
    print("Done.\n")
    
    # Collect features at each parameter point
    param_grid = []
    total = len(D_vals) * len(nu_vals) * len(lam_vals)
    count = 0
    t_start = time.time()
    
    for D in D_vals:
        for nu in nu_vals:
            for lam in lam_vals:
                count += 1
                features = extract_features_ensemble(
                    L=L, T=T, lambda_=lam, nu=nu, D=D, n_samples=n_samples
                )
                
                # Log scale for parameters (natural for multiplicative)
                param_grid.append({
                    'params': [np.log(D), np.log(nu), np.log(lam)],
                    'params_raw': [D, nu, lam],
                    'mean_features': np.mean(features, axis=0),
                    'cov_features': np.cov(features.T) if features.shape[0] > 1 
                                    else np.diag(np.var(features, axis=0)),
                    'features_all': features
                })
                
                elapsed = time.time() - t_start
                rate = elapsed / count
                remaining = rate * (total - count)
                
                if count % 5 == 0 or count == total:
                    print(f"  [{count}/{total}] D={D:.1f}, ν={nu:.1f}, λ={lam:.1f} | "
                          f"grad_var={np.mean(features, axis=0)[0]:.6f} | "
                          f"ETA: {remaining:.0f}s")
    
    print(f"\nData collection complete in {time.time()-t_start:.0f}s\n")
    
    # Compute Fisher matrix in (log D, log ν, log λ) coordinates
    F, J = estimate_fisher_matrix(param_grid, None, ['log_D', 'log_nu', 'log_lam'])
    
    print("Fisher Information Matrix F(log D, log ν, log λ):")
    print(f"  F =")
    for i in range(3):
        print(f"    [{F[i,0]:10.4f} {F[i,1]:10.4f} {F[i,2]:10.4f}]")
    print()
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    # Sort descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("Fisher eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {ev:.6f}  ({ev/eigenvalues[0]*100:.1f}% of dominant)")
    print()
    
    print("Fisher eigenvectors (in log D, log ν, log λ basis):")
    for i in range(3):
        v = eigenvectors[:, i]
        print(f"  v_{i+1} = [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]")
        # Interpret: how much does this direction correspond to D/ν?
        # D/ν in log space = log(D) - log(ν), so the "D/ν direction" is [1, -1, 0]/√2
        dnu_direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        lam_direction = np.array([0.0, 0.0, 1.0])
        cos_dnu = compare_eigenvectors(v, dnu_direction)
        cos_lam = compare_eigenvectors(v, lam_direction)
        print(f"         |cos(v, D/ν direction)| = {cos_dnu:.4f}")
        print(f"         |cos(v, λ direction)|   = {cos_lam:.4f}")
    print()
    
    # Compute Jacobian interpretation
    print("Jacobian (sensitivity of features to parameters):")
    print(f"  {'Feature':<15} {'∂/∂(log D)':>12} {'∂/∂(log ν)':>12} {'∂/∂(log λ)':>12}")
    for k in range(6):
        print(f"  {FEATURE_NAMES[k]:<15} {J[k,0]:12.6f} {J[k,1]:12.6f} {J[k,2]:12.6f}")
    print()
    
    # PREDICTION CHECK: Is the dominant eigenvector the D/ν direction?
    dnu_direction = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    cos_sim = compare_eigenvectors(eigenvectors[:, 0], dnu_direction)
    print(f"PREDICTION P1: Dominant eigenvector ≈ D/ν direction")
    print(f"  |cos(v₁, [1,-1,0]/√2)| = {cos_sim:.4f}")
    print(f"  Status: {'✅ CONFIRMED' if cos_sim > 0.8 else '⚠️ PARTIAL' if cos_sim > 0.5 else '❌ FAILED'}")
    print()
    
    # PREDICTION CHECK: λ eigenvalue ≈ 0?
    lam_projection = eigenvectors[:, -1]  # smallest eigenvalue
    cos_lam = compare_eigenvectors(lam_projection, np.array([0., 0., 1.]))
    ratio = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else float('inf')
    print(f"PREDICTION P2: λ-direction eigenvalue ≈ 0")
    print(f"  Eigenvalue ratio (smallest/largest) = {ratio:.6f}")
    print(f"  Smallest eigenvector alignment with λ: |cos| = {cos_lam:.4f}")
    print(f"  Status: {'✅ CONFIRMED' if ratio < 0.1 else '⚠️ PARTIAL' if ratio < 0.3 else '❌ FAILED'}")
    print()
    
    # Compare to PCA loadings from Exp 46/54
    # PC1 loadings: grad_var (+0.607), grad_skew (−0.004), grad_kurt (+0.026), 
    #               lap_var (+0.586), grad_lap_cov (0.000), h_var (+0.536)
    pca_pc1 = np.array([0.607, -0.004, 0.026, 0.586, 0.000, 0.536])
    
    # The Fisher "stiffest direction in feature space" = J @ v₁
    fisher_feature_direction = J @ eigenvectors[:, 0]
    fisher_feature_direction /= np.linalg.norm(fisher_feature_direction)
    pca_pc1_norm = pca_pc1 / np.linalg.norm(pca_pc1)
    
    cos_pca = compare_eigenvectors(fisher_feature_direction, pca_pc1_norm)
    print(f"PREDICTION P3: Fisher feature direction ≈ PCA PC1")
    print(f"  Fisher feature direction: [{', '.join(f'{x:+.3f}' for x in fisher_feature_direction)}]")
    print(f"  PCA PC1 (Exp 46):         [{', '.join(f'{x:+.3f}' for x in pca_pc1_norm)}]")
    print(f"  |cos similarity| = {cos_pca:.4f}")
    print(f"  Status: {'✅ CONFIRMED' if cos_pca > 0.8 else '⚠️ PARTIAL' if cos_pca > 0.5 else '❌ FAILED'}")
    print()
    
    # Also do PCA on the pooled features and compare
    all_features = np.vstack([g['features_all'] for g in param_grid])
    all_features_centered = all_features - np.mean(all_features, axis=0)
    cov_pca = np.cov(all_features_centered.T)
    pca_evals, pca_evecs = np.linalg.eigh(cov_pca)
    pca_idx = np.argsort(-pca_evals)
    pca_pc1_empirical = pca_evecs[:, pca_idx[0]]
    
    cos_pca_emp = compare_eigenvectors(fisher_feature_direction, pca_pc1_empirical)
    print(f"  PCA PC1 (this data):      [{', '.join(f'{x:+.3f}' for x in pca_pc1_empirical)}]")
    print(f"  |cos(Fisher, PCA_emp)|  = {cos_pca_emp:.4f}")
    print()
    
    return {
        'fisher_matrix': F.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'jacobian': J.tolist(),
        'dnu_alignment': float(cos_sim),
        'lambda_ratio': float(ratio),
        'pca_alignment': float(cos_pca),
        'pca_alignment_empirical': float(cos_pca_emp),
        'param_grid_summary': {
            'n_points': len(param_grid),
            'D_vals': D_vals,
            'nu_vals': nu_vals,
            'lam_vals': lam_vals
        }
    }


# ============================================================================
# PART B: FISHER UNDER COARSE-GRAINING
# ============================================================================

def run_part_b(pilot=False):
    """Test if Fisher eigenvalue ratio increases under coarse-graining."""
    print("=" * 70)
    print("PART B: FISHER INFORMATION UNDER COARSE-GRAINING")
    print("=" * 70)
    print()
    print("Theory predicts:")
    print("  - Under RG (coarse-graining), relevant eigenvalues persist/grow")
    print("  - Irrelevant eigenvalues shrink")
    print("  - Eigenvalue ratio λ₁/λ₂ INCREASES with scale b")
    print("  - Explains why Exp 47 found KL divergence increasing with b")
    print()
    
    L = 256  # Need large L for meaningful coarse-graining
    T = 2000 if not pilot else 1000
    n_samples = 15 if not pilot else 6
    cg_scales = [1, 2, 4]
    
    # Smaller parameter grid for this test
    D_vals = [0.5, 1.0, 2.0]
    nu_vals = [0.5, 1.0, 2.0]
    lam_vals = [0.5, 1.0, 2.0]
    
    print(f"L={L}, scales b={cg_scales}")
    print(f"Grid: {len(D_vals)}×{len(nu_vals)}×{len(lam_vals)} = "
          f"{len(D_vals)*len(nu_vals)*len(lam_vals)} points\n")
    
    results_by_scale = {}
    
    for b in cg_scales:
        print(f"\n--- Scale b = {b} ---")
        param_grid = []
        total = len(D_vals) * len(nu_vals) * len(lam_vals)
        count = 0
        t_start = time.time()
        
        for D in D_vals:
            for nu in nu_vals:
                for lam in lam_vals:
                    count += 1
                    features = extract_features_ensemble(
                        L=L, T=T, lambda_=lam, nu=nu, D=D, 
                        n_samples=n_samples, cg_factor=b
                    )
                    param_grid.append({
                        'params': [np.log(D), np.log(nu), np.log(lam)],
                        'mean_features': np.mean(features, axis=0),
                        'cov_features': np.cov(features.T) if features.shape[0] > 1 
                                        else np.diag(np.var(features, axis=0))
                    })
                    if count % 9 == 0 or count == total:
                        print(f"  [{count}/{total}] {time.time()-t_start:.0f}s")
        
        F, J = estimate_fisher_matrix(param_grid, None, ['log_D', 'log_nu', 'log_lam'])
        eigenvalues, eigenvectors = np.linalg.eigh(F)
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        dnu_dir = np.array([1., -1., 0.]) / np.sqrt(2)
        cos_dnu = compare_eigenvectors(eigenvectors[:, 0], dnu_dir)
        
        results_by_scale[b] = {
            'eigenvalues': eigenvalues.tolist(),
            'ratio_12': float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else float('inf'),
            'ratio_13': float(eigenvalues[0] / eigenvalues[2]) if eigenvalues[2] > 0 else float('inf'),
            'dnu_alignment': float(cos_dnu)
        }
        
        print(f"  Eigenvalues: [{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}]")
        print(f"  Ratio λ₁/λ₂ = {results_by_scale[b]['ratio_12']:.4f}")
        print(f"  D/ν alignment = {cos_dnu:.4f}")
    
    print(f"\n{'='*50}")
    print("PREDICTION P4: Eigenvalue ratio increases with scale")
    print(f"{'='*50}")
    for b in cg_scales:
        r = results_by_scale[b]
        print(f"  b={b}: λ₁/λ₂ = {r['ratio_12']:.4f}, D/ν align = {r['dnu_alignment']:.4f}")
    
    ratios = [results_by_scale[b]['ratio_12'] for b in cg_scales]
    increasing = all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1))
    print(f"\n  Ratios monotonically increasing: {'✅ YES' if increasing else '❌ NO'}")
    if len(ratios) > 1:
        print(f"  Amplification factor (b=4 vs b=1): {ratios[-1]/ratios[0]:.4f}x")
    
    return results_by_scale


# ============================================================================
# PART C: ISING FISHER MATRIX 
# ============================================================================

def run_part_c(pilot=False):
    """Compute Fisher information matrix for 2D Ising in (T, h) space."""
    print("\n" + "=" * 70)
    print("PART C: ISING FISHER INFORMATION MATRIX")
    print("=" * 70)
    print()
    print("Theory predicts:")
    print("  - Dominant Fisher eigenvalue in thermal direction (T)")
    print("  - Fisher eigenvector 1 should align with PC1 from Exp 52b")
    print("  - At T_c, thermal direction is the RG-relevant direction")
    print()
    
    T_c = 2.0 / np.log(1 + np.sqrt(2))
    L = 32  
    n_samples = 50 if not pilot else 20
    
    # Temperature grid near T_c  
    if pilot:
        T_vals = [0.90 * T_c, 0.95 * T_c, T_c, 1.05 * T_c, 1.10 * T_c]
        h_vals = [0.0, 0.05, 0.1]
    else:
        T_vals = [0.88*T_c, 0.92*T_c, 0.96*T_c, T_c, 1.04*T_c, 1.08*T_c, 1.12*T_c]
        h_vals = [0.0, 0.02, 0.05, 0.1, 0.2]
    
    print(f"T_c = {T_c:.4f}")
    print(f"Grid: {len(T_vals)} T × {len(h_vals)} h = {len(T_vals)*len(h_vals)} points")
    print(f"L={L}, {n_samples} samples per point\n")
    
    # Compile
    print("Compiling Ising functions...")
    test_spins = np.ones((8, 8), dtype=np.float64)
    _ = ising_wolff_step(test_spins, 0.5)
    _ = ising_local_features(test_spins)
    _ = _ising_metropolis_sweep(test_spins, 0.5, 0.0)
    print("Done.\n")
    
    param_grid = []
    total = len(T_vals) * len(h_vals)
    count = 0
    t_start = time.time()
    
    ISING_FEATURE_NAMES = ['|∇m|', 'corr_1', 'boundary', '|m_local|', 'Var(|∇m|)', 'Var(m_local)']
    
    for T_val in T_vals:
        for h_val in h_vals:
            count += 1
            t_reduced = (T_val - T_c) / T_c
            
            features = generate_ising_features(L, T_val, h_val, n_samples,
                                                n_therm=300 if not pilot else 200, 
                                                n_between=5)
            
            param_grid.append({
                'params': [t_reduced, h_val],  # Use reduced temperature
                'params_raw': [T_val, h_val],
                'mean_features': np.mean(features, axis=0),
                'cov_features': np.cov(features.T),
                'features_all': features
            })
            
            if count % 3 == 0 or count == total:
                elapsed = time.time() - t_start
                print(f"  [{count}/{total}] T={T_val:.3f} (t={t_reduced:+.3f}), "
                      f"h={h_val:.2f} | {elapsed:.0f}s")
    
    print(f"\nData collection: {time.time()-t_start:.0f}s\n")
    
    # Fisher matrix in (t, h) coordinates
    F, J = estimate_fisher_matrix(param_grid, None, ['t', 'h'])
    
    print("Fisher Information Matrix F(t, h):")
    print(f"  F = [{F[0,0]:10.4f} {F[0,1]:10.4f}]")
    print(f"      [{F[1,0]:10.4f} {F[1,1]:10.4f}]")
    print()
    
    eigenvalues, eigenvectors = np.linalg.eigh(F)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("Fisher eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i+1} = {ev:.6f}")
    print(f"  Ratio λ₁/λ₂ = {eigenvalues[0]/eigenvalues[1]:.4f}" if eigenvalues[1] > 0 else "  λ₂ = 0")
    print()
    
    print("Fisher eigenvectors (in t, h basis):")
    for i in range(2):
        v = eigenvectors[:, i]
        print(f"  v_{i+1} = [{v[0]:+.4f}, {v[1]:+.4f}]")
        cos_t = abs(v[0])  # alignment with pure thermal direction [1, 0]
        cos_h = abs(v[1])  # alignment with pure field direction [0, 1]
        print(f"         thermal alignment: {cos_t:.4f}")
        print(f"         field alignment:   {cos_h:.4f}")
    print()
    
    # Dominant eigenvector should be thermal direction
    v1 = eigenvectors[:, 0]
    thermal_alignment = abs(v1[0]) / np.linalg.norm(v1)
    print(f"PREDICTION P5: Dominant eigenvector = thermal direction")
    print(f"  Thermal alignment of v₁ = {thermal_alignment:.4f}")
    print(f"  Status: {'✅ CONFIRMED' if thermal_alignment > 0.8 else '⚠️ PARTIAL' if thermal_alignment > 0.5 else '❌ FAILED'}")
    print()
    
    # Jacobian tells us: which features are most sensitive to T vs h?
    print("Jacobian (sensitivity of features to parameters):")
    print(f"  {'Feature':<15} {'∂/∂t':>12} {'∂/∂h':>12}  {'ratio |∂t/∂h|':>15}")
    for k in range(6):
        ratio_str = f"{abs(J[k,0]/J[k,1]):.2f}" if abs(J[k,1]) > 1e-10 else "∞"
        print(f"  {ISING_FEATURE_NAMES[k]:<15} {J[k,0]:12.6f} {J[k,1]:12.6f}  {ratio_str:>15}")
    print()
    
    # Compare Fisher feature direction to PC1 loadings from Exp 52b
    # Exp 52b PC1 loadings: [+0.413, -0.413, +0.413, -0.413, +0.411, +0.384]
    pca_pc1_52b = np.array([0.413, -0.413, 0.413, -0.413, 0.411, 0.384])
    fisher_feat_dir = J @ eigenvectors[:, 0]
    fisher_feat_dir_norm = fisher_feat_dir / np.linalg.norm(fisher_feat_dir)
    pca_pc1_norm = pca_pc1_52b / np.linalg.norm(pca_pc1_52b)
    
    cos_pca = compare_eigenvectors(fisher_feat_dir_norm, pca_pc1_norm)
    print(f"PREDICTION P6: Fisher feature direction ≈ Ising PC1")
    print(f"  Fisher dir: [{', '.join(f'{x:+.3f}' for x in fisher_feat_dir_norm)}]")
    print(f"  PC1 (52b):  [{', '.join(f'{x:+.3f}' for x in pca_pc1_norm)}]")
    print(f"  |cos| = {cos_pca:.4f}")
    print(f"  Status: {'✅ CONFIRMED' if cos_pca > 0.8 else '⚠️ PARTIAL' if cos_pca > 0.5 else '❌ FAILED'}")
    print()
    
    return {
        'fisher_matrix': F.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'jacobian': J.tolist(),
        'thermal_alignment': float(thermal_alignment),
        'pca_alignment': float(cos_pca),
        'eigenvalue_ratio': float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else float('inf')
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true', help='Run quick pilot')
    parser.add_argument('--part', type=str, default='all', 
                        help='Which part to run: a, b, c, or all')
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 60: Fisher Information ↔ PCA ↔ RG Relevance           ║")
    print("║  'PCA finds the stiffest direction of the Fisher metric'           ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Mode: {'PILOT' if args.pilot else 'FULL'}  |  Parts: {args.part.upper():<42}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    all_results = {}
    
    if args.part in ['a', 'all']:
        results_a = run_part_a(pilot=args.pilot)
        all_results['part_a'] = results_a
    
    if args.part in ['b', 'all']:
        results_b = run_part_b(pilot=args.pilot)
        all_results['part_b'] = results_b
    
    if args.part in ['c', 'all']:
        results_c = run_part_c(pilot=args.pilot)
        all_results['part_c'] = results_c
    
    # Save results
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'results.json'}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Fisher–PCA–RG Connection Tests")
    print("=" * 70)
    
    if 'part_a' in all_results:
        r = all_results['part_a']
        print(f"\nPart A (KPZ):")
        print(f"  P1 (D/ν dominance):     alignment = {r['dnu_alignment']:.4f} "
              f"{'✅' if r['dnu_alignment'] > 0.8 else '❌'}")
        print(f"  P2 (λ invisible):       ratio = {r['lambda_ratio']:.6f} "
              f"{'✅' if r['lambda_ratio'] < 0.1 else '❌'}")
        print(f"  P3 (Fisher ≈ PCA):      |cos| = {r['pca_alignment']:.4f} "
              f"{'✅' if r['pca_alignment'] > 0.8 else '❌'}")
    
    if 'part_b' in all_results:
        r = all_results['part_b']
        scales = sorted(r.keys())
        print(f"\nPart B (Coarse-graining):")
        for b in scales:
            print(f"  b={b}: λ₁/λ₂ = {r[b]['ratio_12']:.4f}")
        ratios = [r[b]['ratio_12'] for b in scales]
        print(f"  Ratio increasing: {'✅' if all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1)) else '❌'}")
    
    if 'part_c' in all_results:
        r = all_results['part_c']
        print(f"\nPart C (Ising):")
        print(f"  P5 (thermal dominant):  alignment = {r['thermal_alignment']:.4f} "
              f"{'✅' if r['thermal_alignment'] > 0.8 else '❌'}")
        print(f"  P6 (Fisher ≈ PC1):      |cos| = {r['pca_alignment']:.4f} "
              f"{'✅' if r['pca_alignment'] > 0.8 else '❌'}")
    
    print()
    print("If P1–P6 all pass: Fisher information IS the bridge between PCA and RG.")
    print("This would be the theoretical foundation for Paper C (PRL).")


if __name__ == '__main__':
    main()
