"""
EXPERIMENT 17: Block RG + Total Correlation Flow
=================================================


1. PROPER RG: Block averaging + space-time-height rescaling
   - h_b(x,t) = (1/b) * Σ h(bx+i, t)  (spatial coarse-grain)
   - t_b = t / b^z                     (time rescale)
   - h̃_b = h_b / b^α                  (height rescale)
   
2. DEFENSIBLE METRIC: Total Correlation (multi-information)
   TC(X) = Σ_i H(X_i) - H(X)
   Measures statistical dependence among observables
   
3. PREDICTIONS:
   - TC decreases under RG (irrelevant couplings die)
   - KPZ-class (KPZ, BD, EDEN) flows collapse to same attractor
   - EW flows to different attractor
   - RD either doesn't converge or goes somewhere distinct

Key exponents:
- KPZ: z=3/2, α=1/2 (1D)
- EW: z=2, α=1/2 (1D)

This experiment aims to produce a publishable result showing
"universality as basin structure on an information manifold"
"""

import numpy as np
import sys
import os

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_dir)

from simulation.physics_simulation import GrowthModelSimulator
from scipy.stats import gaussian_kde
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Known scaling exponents
EXPONENTS = {
    'edwards_wilkinson': {'z': 2.0, 'alpha': 0.5},
    'kpz_equation': {'z': 1.5, 'alpha': 0.5},
    'ballistic_deposition': {'z': 1.5, 'alpha': 0.5},  # KPZ class
    'eden': {'z': 1.5, 'alpha': 0.5},  # KPZ class
    'random_deposition': {'z': None, 'alpha': 0.5},  # No lateral correlation
}

MODEL_SHORTNAMES = {
    'edwards_wilkinson': 'EW',
    'kpz_equation': 'KPZ',
    'ballistic_deposition': 'BD',
    'eden': 'EDEN',
    'random_deposition': 'RD'
}


def block_average_surface(h, block_size):
    """
    Apply block RG to surface: h_b(x) = (1/b) * Σ h(bx+i)
    
    Args:
        h: 1D height profile
        block_size: coarse-graining factor b
    
    Returns:
        Coarse-grained height profile
    """
    L = len(h)
    L_new = L // block_size
    h_coarse = np.zeros(L_new)
    
    for i in range(L_new):
        h_coarse[i] = np.mean(h[i*block_size:(i+1)*block_size])
    
    return h_coarse


def apply_block_rg(surfaces, block_size, z_exponent, alpha_exponent):
    """
    Apply proper block RG transformation with rescaling.
    
    For a trajectory of surfaces at different times:
    - Block average spatially
    - Rescale heights by b^alpha
    - Select times corresponding to t_b = t / b^z
    
    Args:
        surfaces: array of shape (T, L) - time series of height profiles
        block_size: coarse-graining factor b
        z_exponent: dynamic exponent z
        alpha_exponent: roughness exponent α
    
    Returns:
        Rescaled, coarse-grained surfaces
    """
    T, L = surfaces.shape
    
    # Spatial block averaging
    L_new = L // block_size
    surfaces_blocked = np.zeros((T, L_new))
    
    for t in range(T):
        surfaces_blocked[t] = block_average_surface(surfaces[t], block_size)
    
    # Height rescaling: h̃ = h / b^α
    surfaces_rescaled = surfaces_blocked / (block_size ** alpha_exponent)
    
    # Time rescaling: select times t_b = t / b^z
    # This means we look at times that are b^z apart in the original
    if z_exponent is not None:
        time_factor = int(block_size ** z_exponent)
        time_indices = np.arange(0, T, max(1, time_factor))
        surfaces_rescaled = surfaces_rescaled[time_indices]
    
    return surfaces_rescaled


def compute_local_observables(surface):
    """
    Compute local observables at each site: (g, s², ∇²h)
    These are the same observables from Exp 15.
    """
    L = len(surface)
    
    # Local gradient magnitude |∇h|
    grad_h = np.gradient(surface)
    
    # Local curvature / Laplacian ∇²h
    lap_h = np.zeros(L)
    for i in range(L):
        ip = (i + 1) % L
        im = (i - 1) % L
        lap_h[i] = surface[ip] + surface[im] - 2*surface[i]
    
    # Local roughness proxy: squared deviation from local mean
    # Using a 3-point window
    local_mean = np.zeros(L)
    for i in range(L):
        ip = (i + 1) % L
        im = (i - 1) % L
        local_mean[i] = (surface[ip] + surface[i] + surface[im]) / 3
    local_var = (surface - local_mean) ** 2
    
    return grad_h, local_var, lap_h


def compute_total_correlation_knn(observables, k=5):
    """
    Compute Total Correlation using k-NN entropy estimator.
    TC(X) = Σ_i H(X_i) - H(X)
    
    Uses Kozachenko-Leonenko estimator for differential entropy.
    """
    N, d = observables.shape
    
    if N < k + 1:
        return np.nan
    
    # Standardize for numerical stability
    obs_std = (observables - np.mean(observables, axis=0)) / (np.std(observables, axis=0) + 1e-10)
    
    # Joint entropy H(X) using k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(obs_std)
    distances, _ = nbrs.kneighbors(obs_std)
    # k-th neighbor distance (excluding self)
    rho = distances[:, k]
    rho = np.maximum(rho, 1e-10)  # Avoid log(0)
    
    # Kozachenko-Leonenko estimator
    H_joint = d * np.mean(np.log(rho)) + np.log(N - 1) - digamma(k) + d * np.log(2)
    
    # Marginal entropies H(X_i)
    H_marginals = 0
    for i in range(d):
        x_i = obs_std[:, i:i+1]
        nbrs_i = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(x_i)
        distances_i, _ = nbrs_i.kneighbors(x_i)
        rho_i = distances_i[:, k]
        rho_i = np.maximum(rho_i, 1e-10)
        H_i = np.mean(np.log(rho_i)) + np.log(N - 1) - digamma(k) + np.log(2)
        H_marginals += H_i
    
    # Total correlation
    TC = H_marginals - H_joint
    
    return TC


def compute_log_det_fisher(observables):
    """
    Compute log-det of Fisher information matrix (as complexity measure).
    For Gaussian approximation: F ≈ Σ^{-1}, so log-det(F) = -log-det(Σ)
    """
    # Covariance matrix
    cov = np.cov(observables, rowvar=False)
    
    # Regularize for numerical stability
    cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
    
    # log-det of Fisher ≈ -log-det of covariance
    sign, logdet = np.linalg.slogdet(cov_reg)
    
    return -logdet  # Fisher log-det


def compute_fisher_trace(observables):
    """
    Compute trace of Fisher information matrix.
    For Gaussian: F = Σ^{-1}, so tr(F) = Σ 1/λ_i where λ_i are eigenvalues of Σ
    """
    cov = np.cov(observables, rowvar=False)
    cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
    
    eigenvalues = np.linalg.eigvalsh(cov_reg)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    return np.sum(1.0 / eigenvalues)


def generate_model_trajectory(model_name, L=512, T=1000, n_runs=5):
    """
    Generate multiple trajectories for a model.
    Returns surfaces averaged across runs for stability.
    """
    all_trajectories = []
    
    for run in range(n_runs):
        sim = GrowthModelSimulator(width=L, height=T, random_state=run*100)
        trajectory = sim.generate_trajectory(model_name)
        all_trajectories.append(trajectory)
    
    return np.array(all_trajectories)


def compute_metrics_at_scale(surfaces, model_name, block_size):
    """
    Compute TC, log-det Fisher, and Fisher trace at a given RG scale.
    
    Args:
        surfaces: array of trajectories (n_runs, T, L)
        model_name: for exponents
        block_size: RG scale b
    
    Returns:
        dict with TC, log_det_F, tr_F
    """
    z = EXPONENTS[model_name]['z']
    alpha = EXPONENTS[model_name]['alpha']
    
    # Collect observables from all runs and times (late-time regime)
    all_observables = []
    
    for run_idx in range(surfaces.shape[0]):
        trajectory = surfaces[run_idx]
        
        # Apply block RG
        if block_size > 1:
            rg_trajectory = apply_block_rg(trajectory, block_size, z, alpha)
        else:
            rg_trajectory = trajectory
        
        # Use late-time snapshots (steady-state regime)
        T_rg = rg_trajectory.shape[0]
        late_start = max(0, T_rg // 2)
        
        for t in range(late_start, T_rg, max(1, T_rg // 20)):
            surface = rg_trajectory[t]
            
            # Get local observables
            grad_h, local_var, lap_h = compute_local_observables(surface)
            
            # Stack into observation vectors
            obs = np.column_stack([grad_h, local_var, lap_h])
            all_observables.append(obs)
    
    # Concatenate all observations
    if len(all_observables) == 0:
        return {'TC': np.nan, 'log_det_F': np.nan, 'tr_F': np.nan}
    
    all_obs = np.vstack(all_observables)
    
    # Subsample if too large (for computational efficiency)
    max_samples = 50000
    if len(all_obs) > max_samples:
        indices = np.random.choice(len(all_obs), max_samples, replace=False)
        all_obs = all_obs[indices]
    
    # Compute metrics
    TC = compute_total_correlation_knn(all_obs, k=5)
    log_det_F = compute_log_det_fisher(all_obs)
    tr_F = compute_fisher_trace(all_obs)
    
    return {'TC': TC, 'log_det_F': log_det_F, 'tr_F': tr_F}


def run_experiment():
    """
    Main experiment: Block RG + Total Correlation Flow
    """
    print("=" * 70)
    print("EXPERIMENT 17: Block RG + Total Correlation Flow")
    print("=" * 70)
    print()
    print("Goals:")
    print("  1. Apply proper block RG with space-time-height rescaling")
    print("  2. Track Total Correlation under RG flow")
    print("  3. Test if KPZ-class flows collapse, EW separates")
    print()
    
    # Parameters
    L = 512  # Large system for multiple RG steps
    T = 500
    n_runs = 3
    block_sizes = [1, 2, 4, 8, 16]  # RG scales
    
    models = ['edwards_wilkinson', 'kpz_equation', 'ballistic_deposition', 
              'eden', 'random_deposition']
    
    # Store results
    results = {model: {'TC': [], 'log_det_F': [], 'tr_F': []} for model in models}
    
    print("-" * 70)
    print("PART 1: Generate trajectories and compute RG flow")
    print("-" * 70)
    
    for model in models:
        short = MODEL_SHORTNAMES[model]
        print(f"\nProcessing {short}...")
        
        # Generate trajectories
        print(f"  Generating {n_runs} trajectories (L={L}, T={T})...", end=" ", flush=True)
        trajectories = generate_model_trajectory(model, L=L, T=T, n_runs=n_runs)
        print("done")
        
        # Compute metrics at each RG scale
        for b in block_sizes:
            if L // b < 8:  # Need minimum system size
                results[model]['TC'].append(np.nan)
                results[model]['log_det_F'].append(np.nan)
                results[model]['tr_F'].append(np.nan)
                continue
            
            print(f"  Scale b={b}...", end=" ", flush=True)
            metrics = compute_metrics_at_scale(trajectories, model, b)
            results[model]['TC'].append(metrics['TC'])
            results[model]['log_det_F'].append(metrics['log_det_F'])
            results[model]['tr_F'].append(metrics['tr_F'])
            print(f"TC={metrics['TC']:.3f}, log-det(F)={metrics['log_det_F']:.2f}")
    
    print()
    print("-" * 70)
    print("PART 2: Total Correlation Flow Results")
    print("-" * 70)
    print()
    
    # Print TC flow table
    print("Total Correlation TC(b) under Block RG:")
    print()
    header = "Model    " + "  ".join([f"b={b:2d}" for b in block_sizes]) + " | Monotone?"
    print(header)
    print("-" * len(header))
    
    for model in models:
        short = MODEL_SHORTNAMES[model]
        tc_values = results[model]['TC']
        
        # Check monotonicity (TC should decrease)
        valid_tc = [v for v in tc_values if not np.isnan(v)]
        if len(valid_tc) >= 2:
            diffs = np.diff(valid_tc)
            monotone = np.all(diffs <= 0.1)  # Allow small fluctuations
            mono_str = "YES" if monotone else "NO"
        else:
            mono_str = "N/A"
        
        tc_str = "  ".join([f"{v:5.2f}" if not np.isnan(v) else "  N/A" for v in tc_values])
        print(f"{short:8s} {tc_str} | {mono_str}")
    
    print()
    print("-" * 70)
    print("PART 3: log-det(Fisher) Flow Results")
    print("-" * 70)
    print()
    
    header = "Model    " + "  ".join([f"b={b:2d}" for b in block_sizes]) + " | Monotone?"
    print(header)
    print("-" * len(header))
    
    for model in models:
        short = MODEL_SHORTNAMES[model]
        logdet_values = results[model]['log_det_F']
        
        valid_ld = [v for v in logdet_values if not np.isnan(v)]
        if len(valid_ld) >= 2:
            diffs = np.diff(valid_ld)
            monotone = np.all(diffs <= 1.0)  # Allow some fluctuation
            mono_str = "YES" if monotone else "NO"
        else:
            mono_str = "N/A"
        
        ld_str = "  ".join([f"{v:5.1f}" if not np.isnan(v) else "  N/A" for v in logdet_values])
        print(f"{short:8s} {ld_str} | {mono_str}")
    
    print()
    print("-" * 70)
    print("PART 4: Universality Basin Analysis")
    print("-" * 70)
    print()
    
    # Test: Do KPZ-class flows converge?
    print("Test 1: Do KPZ-class models (BD, EDEN, KPZ) have similar TC flows?")
    print()
    
    kpz_models = ['kpz_equation', 'ballistic_deposition', 'eden']
    
    # Compute pairwise distances between flows
    def flow_distance(model1, model2):
        tc1 = np.array(results[model1]['TC'])
        tc2 = np.array(results[model2]['TC'])
        valid = ~(np.isnan(tc1) | np.isnan(tc2))
        if np.sum(valid) < 2:
            return np.nan
        return np.mean(np.abs(tc1[valid] - tc2[valid]))
    
    print("TC Flow Distances:")
    print()
    all_models_short = [MODEL_SHORTNAMES[m] for m in models]
    print("        " + "  ".join([f"{s:6s}" for s in all_models_short]))
    
    for i, model1 in enumerate(models):
        row = f"{MODEL_SHORTNAMES[model1]:6s}  "
        for j, model2 in enumerate(models):
            if j <= i:
                row += "       "
            else:
                d = flow_distance(model1, model2)
                row += f"{d:6.2f} " if not np.isnan(d) else "   N/A "
        print(row)
    
    print()
    
    # Compute average within-class vs between-class distances
    kpz_pairs = [('kpz_equation', 'ballistic_deposition'), 
                 ('kpz_equation', 'eden'),
                 ('ballistic_deposition', 'eden')]
    
    cross_pairs = [('edwards_wilkinson', m) for m in kpz_models]
    
    within_kpz = np.nanmean([flow_distance(m1, m2) for m1, m2 in kpz_pairs])
    ew_to_kpz = np.nanmean([flow_distance(m1, m2) for m1, m2 in cross_pairs])
    
    print(f"Average distance within KPZ class: {within_kpz:.3f}")
    print(f"Average distance EW to KPZ class:  {ew_to_kpz:.3f}")
    print()
    
    if ew_to_kpz > 1.5 * within_kpz:
        print("✓ PASS: EW is significantly farther from KPZ-class than KPZ models are from each other")
    else:
        print("? INCONCLUSIVE: Separation not clear")
    
    print()
    print("-" * 70)
    print("PART 5: Fixed-Point Fingerprint Analysis")
    print("-" * 70)
    print()
    
    # Look at large-scale (b=16 or largest valid) TC values
    print("Large-scale (b=16) TC values (fixed-point fingerprints):")
    print()
    
    for model in models:
        short = MODEL_SHORTNAMES[model]
        tc_final = results[model]['TC'][-1] if len(results[model]['TC']) > 0 else np.nan
        if not np.isnan(tc_final):
            print(f"  {short}: TC_∞ ≈ {tc_final:.3f}")
        else:
            print(f"  {short}: TC_∞ = N/A (system too small)")
    
    print()
    
    # Check if KPZ-class have similar fixed points
    kpz_fixed = [results[m]['TC'][-1] for m in kpz_models if not np.isnan(results[m]['TC'][-1])]
    ew_fixed = results['edwards_wilkinson']['TC'][-1]
    
    if len(kpz_fixed) >= 2:
        kpz_spread = np.std(kpz_fixed)
        kpz_mean = np.mean(kpz_fixed)
        print(f"KPZ-class fixed-point spread: {kpz_spread:.3f} (mean={kpz_mean:.3f})")
        
        if not np.isnan(ew_fixed):
            separation = abs(ew_fixed - kpz_mean)
            print(f"EW separation from KPZ mean: {separation:.3f}")
            
            if separation > 2 * kpz_spread:
                print("\n✓ PASS: Fixed points separate by universality class!")
            else:
                print("\n? INCONCLUSIVE: Fixed-point separation not clear")
    
    print()
    print("=" * 70)
    print("EXPERIMENT 17 SUMMARY")
    print("=" * 70)
    print()
    print("Key Findings:")
    print()
    
    # Summarize monotonicity
    mono_count = 0
    for model in models:
        tc = results[model]['TC']
        valid = [v for v in tc if not np.isnan(v)]
        if len(valid) >= 2 and np.all(np.diff(valid) <= 0.1):
            mono_count += 1
    
    print(f"1. TC Monotonicity: {mono_count}/{len(models)} models show decreasing TC under RG")
    
    # Summarize basin structure
    print(f"2. Within-KPZ distance: {within_kpz:.3f}")
    print(f"3. EW-to-KPZ distance: {ew_to_kpz:.3f}")
    print(f"4. Ratio (should be > 1.5 for separation): {ew_to_kpz/within_kpz:.2f}")
    
    print()
    if mono_count >= 3 and ew_to_kpz > 1.5 * within_kpz:
        print("CONCLUSION: Results support 'universality as basin structure'")
        print("  - RG flow reduces statistical coupling (TC decreases)")
        print("  - Same-class flows converge (small within-class distance)")
        print("  - Different-class flows separate (large between-class distance)")
    else:
        print("CONCLUSION: Partial support - further investigation needed")
    
    print()
    print("Defensible statement (if results hold):")
    print('  "RG flow reduces statistical coupling between local operators.')
    print('   Different microscopics exhibit convergent trajectories in')
    print('   information space when in the same universality class."')
    
    return results


if __name__ == "__main__":
    results = run_experiment()
