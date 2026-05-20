"""
Experiment 64: Multi-Scale Coarse-Graining + Hierarchical Peel
================================================================

PURPOSE: Diagnose and attack the TWO structural barriers to ARI > 0.5
identified in Exp 63's post-hoc analysis:

  Barrier 1: EW merged with KPZ+Eden (not separated as density clusters)
  Barrier 2: BD split from KPZ+Eden (discrete lattice effects create a
             separate density peak in feature space)

Exp 63 proved the temporal features are discriminative (kNN=97.7%) but
HDBSCAN gives the SAME partition in both 6D and 10D — literally identical
labels (ARI=1.0 between them). The partition is:

    {EW + KPZ + Eden} | {BD} | {KS} | {RD + 1 noise}

Even perfect EW separation would only give ARI~0.73 because BD is split.
Even perfect BD merger would only give ARI~0.67 because EW is absorbed.
Both problems need to be addressed together.

APPROACH — Three diagnostic sub-experiments:

  64A: MULTI-SCALE FEATURES
    Block-average surfaces at b=1,2,4,8, extract 10D features at each scale.
    Concatenate into 40D multi-scale vector (or use features from the
    coarsest scale where BD≈KPZ). From Exp 23: block CG drops KPZ-BD
    distance by 90% (2.34→0.26). From Exp 24: gradient moments are
    RG-relevant operators — they GROW under CG, potentially widening
    the EW-KPZ gap. So CG may HELP Barrier 2 while HURTING Barrier 1.
    We need to measure both effects quantitatively.

  64B: HIERARCHICAL PEEL
    Remove RD and KS (trivially separated by beta_eff and vel_skew),
    then cluster the hard 4-system {EW, KPZ, BD, Eden} subproblem.
    This eliminates ARI pollution from easy structure and focuses
    entirely on the two barriers.

  64C: PEEL + COARSE-GRAINED FEATURES
    Combine: remove RD/KS, then apply multi-scale features to the
    4-system subproblem. This is the full attack on both barriers.

KEY DIAGNOSTICS (beyond ARI):
  - kNN graph connectivity: does KPZ class form 1 or 2 connected
    components? If 2, no clustering algorithm can merge BD with KPZ.
  - BD-KPZ centroid distance vs coarse-graining scale
  - EW-KPZ centroid distance vs coarse-graining scale (watch for expansion!)
  - Per-scale feature analysis: which scale best separates EW from KPZ
    while merging BD toward KPZ?

PRIOR EVIDENCE (from this project):
  Exp 5:  Gradient+blur → within-KPZ ratio drops 14.5x→2x at sigma=2
  Exp 14: Gaussian CG DESTROYS slope-growth signal (b decreases). Not
          a proper RG. Block averaging is better (Exp 23).
  Exp 15: Fisher curvature flows down under CG for discrete models
          (BD: 6.18→0.18). Discreteness = curvature.
  Exp 23: Block CG drops KPZ-BD distance 90% (2.34→0.26). KEY RESULT.
  Exp 24: But gradient moments are RG-relevant, so EW-KPZ EXPANDS 45%
          under CG. This is the trade-off we must navigate.

PREDICTIONS:
  P1: BD-KPZ centroid distance decreases monotonically with block size
  P2: EW-KPZ centroid distance INCREASES with block size (RG-relevance)
  P3: There exists an optimal scale where combined ARI is maximized
      (balancing BD merger vs EW-KPZ expansion)
  P4: Peel (64B) improves ARI by removing trivial structure
  P5: kNN graph shows KPZ class as 2 components at b=1 but 1 at b>=4
  P6: Best ARI achievable: >0.6 (from peel + optimal scale)
  P7: Multi-scale concatenation outperforms any single scale

SUCCESS CRITERIA:
  - Any sub-experiment achieves ARI > 0.55 (above Exp 63 ceiling)
  - Clear quantitative measurement of the CG trade-off (P1 vs P2)
  - kNN connectivity diagnostic answers: is BD geometrically mergeable?

Usage:
  python 64_multiscale_peel.py --pilot    # Quick (~8 min)
  python 64_multiscale_peel.py            # Full (~30 min)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
import json
import argparse
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

np.random.seed(42)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / 'results_exp64'
FIGURES_DIR = PROJECT_DIR / 'figures'

# ============================================================================
# Simulators (from Exp 62/63, all 1D periodic)
# ============================================================================

from numba import jit


@jit(nopython=True)
def simulate_ew(L=128, T=500, nu=1.0, D=1.0, dt=0.1, seed=None):
    """Edwards-Wilkinson: dh/dt = nu * lap(h) + noise."""
    if seed is not None:
        np.random.seed(seed)
    h = np.random.randn(L) * 0.01
    trajectory = np.zeros((T, L))
    for t in range(T):
        new_h = h.copy()
        for x in range(L):
            lap = h[(x-1) % L] - 2*h[x] + h[(x+1) % L]
            noise = np.sqrt(2 * D * dt) * np.random.randn()
            new_h[x] = h[x] + dt * nu * lap + noise
        h = new_h - np.mean(new_h)
        trajectory[t] = h
    return trajectory


@jit(nopython=True)
def simulate_kpz(L=128, T=500, nu=1.0, lam=1.0, D=1.0, dt=0.01, seed=None):
    """KPZ equation: dh/dt = nu*lap(h) + (lam/2)*(grad h)^2 + noise."""
    if seed is not None:
        np.random.seed(seed)
    h = np.random.randn(L) * 0.01
    trajectory = np.zeros((T, L))
    substeps = max(1, int(0.05 / dt))
    for t in range(T):
        for _ in range(substeps):
            new_h = h.copy()
            for x in range(L):
                lap = h[(x-1) % L] - 2*h[x] + h[(x+1) % L]
                grad = (h[(x+1) % L] - h[(x-1) % L]) / 2.0
                noise = np.sqrt(2 * D * dt) * np.random.randn()
                new_h[x] = h[x] + dt * (nu * lap + (lam/2) * grad**2) + noise
            h = new_h - np.mean(new_h)
        trajectory[t] = h
        if np.any(np.isnan(h)) or np.max(np.abs(h)) > 1e6:
            return None
    return trajectory


@jit(nopython=True)
def simulate_bd(L=128, T=500, depositions_per_step=None, seed=None):
    """Ballistic Deposition (KPZ class, discrete)."""
    if seed is not None:
        np.random.seed(seed)
    if depositions_per_step is None:
        depositions_per_step = L
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for _ in range(depositions_per_step):
            x = np.random.randint(0, L)
            landing = max(h[(x-1) % L], h[x], h[(x+1) % L]) + 1.0
            h[x] = landing
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


@jit(nopython=True)
def simulate_eden(L=128, T=500, seed=None):
    """Eden model (KPZ class, discrete)."""
    if seed is not None:
        np.random.seed(seed)
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for _ in range(L):
            x = np.random.randint(0, L)
            curvature = h[(x-1) % L] + h[(x+1) % L] - 2*h[x]
            prob = 0.5 + 0.3 * np.tanh(curvature)
            if np.random.rand() < prob:
                h[x] += 1.0
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


@jit(nopython=True)
def simulate_rd(L=128, T=500, seed=None):
    """Random Deposition (trivial class)."""
    if seed is not None:
        np.random.seed(seed)
    h = np.zeros(L)
    trajectory = np.zeros((T, L))
    for t in range(T):
        for x in range(L):
            h[x] += np.abs(np.random.randn())
        h = h - np.mean(h)
        trajectory[t] = h.copy()
    return trajectory


def simulate_ks(L=128, T=500, nu=1.0, kappa=2.0, lam=1.0, noise=0.05,
                dt=0.01, record_interval=1, seed=None):
    """Kuramoto-Sivashinsky (KS class). Fourier-space ETD."""
    if seed is not None:
        np.random.seed(seed)
    dx = 2 * np.pi / L
    k = np.fft.fftfreq(L, d=dx / (2 * np.pi))
    k4 = k**4
    k2 = k**2
    h = 0.01 * np.random.randn(L)
    h_hat = np.fft.fft(h)
    Lin = -nu * k4 + kappa * k2
    exp_L = np.exp(Lin * dt)
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(np.abs(Lin) > 1e-12, (exp_L - 1) / Lin, dt)
    n_steps = T * record_interval
    trajectory = np.zeros((T, L))
    snap = 0
    for step in range(n_steps):
        dh_dx = np.real(np.fft.ifft(1j * k * h_hat))
        nl_hat = np.fft.fft(-lam / 2 * dh_dx**2)
        eta_hat = np.fft.fft(noise * np.random.randn(L))
        h_hat = exp_L * h_hat + M * (nl_hat + eta_hat)
        if (step + 1) % record_interval == 0:
            h = np.real(np.fft.ifft(h_hat))
            h = h - np.mean(h)
            if np.any(np.isnan(h)) or np.max(np.abs(h)) > 1e6:
                return None
            trajectory[snap] = h
            snap += 1
    return trajectory


# ============================================================================
# Coarse-graining: block averaging (proper RG, from Exp 23)
# ============================================================================

def coarse_grain_surface(h, block_size):
    """Block-average a 1D surface: h_CG[i] = mean(h[i*b : (i+1)*b]).

    This is the real-space RG transformation used in Exp 23 that
    dropped KPZ-BD distance by 90%.
    """
    L = len(h)
    L_new = L // block_size
    if L_new < 8:  # need at least 8 sites for meaningful features
        return h  # don't coarse-grain if surface is too short
    coarse = np.zeros(L_new)
    for i in range(L_new):
        coarse[i] = np.mean(h[i * block_size:(i + 1) * block_size])
    return coarse


def coarse_grain_trajectory(trajectory, block_size):
    """Apply block CG to every snapshot in a trajectory."""
    if block_size == 1:
        return trajectory
    T, L = trajectory.shape
    L_new = L // block_size
    if L_new < 8:
        return trajectory
    cg_traj = np.zeros((T, L_new))
    for t in range(T):
        cg_traj[t] = coarse_grain_surface(trajectory[t], block_size)
    return cg_traj


# ============================================================================
# Feature extraction (spatial + temporal, from Exp 63)
# ============================================================================

@jit(nopython=True)
def compute_features_single(h):
    """6D spatial feature vector from a single 1D height profile."""
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


SPATIAL_NAMES = ['grad_var', 'grad_skew', 'grad_kurt',
                 'lap_var', 'grad_lap_cov', 'h_var']


def extract_spatial_features(trajectory, late_frac=0.3):
    """Extract 6D spatial features from the late-time portion."""
    T = trajectory.shape[0]
    start = int(T * (1 - late_frac))
    features = []
    for t in range(start, T):
        features.append(compute_features_single(trajectory[t]))
    return np.mean(features, axis=0)


TEMPORAL_NAMES = ['beta_eff', 'vel_skew', 'vel_kurt', 'slope_growth']
ALL_FEATURE_NAMES = SPATIAL_NAMES + TEMPORAL_NAMES


def compute_temporal_features(trajectory):
    """4D temporal feature vector from full (T, L) trajectory.

    Features:
      [0] beta_eff     = Growth exponent from W(t) ~ t^beta, clipped [-1, 2]
      [1] vel_skew     = Skew[dh/dt] (multi-step, stride~10)
      [2] vel_kurt     = Kurt[dh/dt] - 3 (multi-step)
      [3] slope_growth = Pearson r(v, (grad h)^2) — nonlinear coupling
    """
    T, L = trajectory.shape

    # === 1. Growth exponent beta ===
    widths = np.std(trajectory, axis=1)
    t_start = max(2, int(0.05 * T))
    t_end = max(t_start + 10, int(0.4 * T))
    t_end = min(t_end, T)

    t_range = np.arange(t_start, t_end, dtype=np.float64)
    w_range = widths[t_start:t_end]
    valid = w_range > 1e-10
    if valid.sum() >= 5:
        try:
            coeffs = np.polyfit(np.log(t_range[valid]),
                                np.log(w_range[valid]), 1)
            beta_eff = float(np.clip(coeffs[0], -1.0, 2.0))
        except (np.linalg.LinAlgError, ValueError):
            beta_eff = 0.0
    else:
        beta_eff = 0.0

    # === 2-4. Late-time velocity and slope-growth ===
    late_start = max(1, int(T * 0.7))
    if late_start >= T - 2:
        late_start = max(1, T - 10)
    n_avail = T - late_start

    stride = max(1, min(10, n_avail // 20))
    h_early = trajectory[late_start: T - stride]
    h_later = trajectory[late_start + stride: T]
    n_steps = h_early.shape[0]

    if n_steps < 2:
        return np.array([beta_eff, 0.0, 0.0, 0.0])

    vel = h_later - h_early
    vel_flat = vel.flatten()
    vel_mean = np.mean(vel_flat)
    vel_var = np.var(vel_flat)
    vel_std = max(np.sqrt(vel_var), 1e-15)
    vel_c = vel_flat - vel_mean
    vel_skew = float(np.clip(np.mean((vel_c / vel_std)**3), -50.0, 50.0))
    vel_kurt = float(np.clip(np.mean((vel_c / vel_std)**4) - 3.0, -50.0, 200.0))

    # Slope-growth: Pearson r(v, g^2)
    grad = (np.roll(h_early, -1, axis=1) -
            np.roll(h_early, 1, axis=1)) / 2.0
    grad_sq = grad**2

    corrs = []
    for i in range(n_steps):
        v_i = vel[i]
        g2_i = grad_sq[i]
        v_std_i = np.std(v_i)
        g2_std_i = np.std(g2_i)
        if v_std_i > 1e-15 and g2_std_i > 1e-15:
            r_i = np.corrcoef(v_i, g2_i)[0, 1]
            if np.isfinite(r_i):
                corrs.append(r_i)
    slope_growth = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

    return np.array([beta_eff, vel_skew, vel_kurt, slope_growth])


def extract_all_features(trajectory, late_frac=0.3):
    """Extract combined 10D feature vector (6 spatial + 4 temporal)."""
    spatial = extract_spatial_features(trajectory, late_frac)
    temporal = compute_temporal_features(trajectory)
    return np.concatenate([spatial, temporal])


# ============================================================================
# Multi-scale feature extraction (NEW for Exp 64)
# ============================================================================

def extract_multiscale_features(trajectory, block_sizes=(1, 2, 4, 8),
                                late_frac=0.3):
    """Extract 10D features at each coarse-graining scale, concatenate.

    For b=1: standard 10D features on the raw surface.
    For b>1: block-average the surface, then extract 10D features.

    Returns a (10 * len(block_sizes),) vector.

    The idea: fine-scale features preserve EW-KPZ temporal separation
    (from Exp 63), while coarse-scale features collapse BD toward
    KPZ/Eden (from Exp 23). The concatenation lets the clustering
    algorithm see both effects simultaneously.
    """
    all_feats = []
    for b in block_sizes:
        cg_traj = coarse_grain_trajectory(trajectory, b)
        feats = extract_all_features(cg_traj, late_frac)
        all_feats.append(feats)
    return np.concatenate(all_feats)


# ============================================================================
# Dataset generation
# ============================================================================

def generate_dataset(n_samples, L, T, system, seed_offset=0,
                     late_frac=0.3, block_sizes=(1, 2, 4, 8),
                     **sim_kwargs):
    """Generate trajectories and extract features at multiple scales.

    Returns:
        trajectories: list of (T, L) arrays (raw, for diagnostics)
        features_per_scale: dict {b: (n_valid, 10) array}
        multiscale_features: (n_valid, 10*n_scales) array
        params_used: list of dicts
    """
    trajectories = []
    features_per_scale = {b: [] for b in block_sizes}
    multiscale_list = []
    params_used = []

    for i in range(n_samples):
        seed = seed_offset + i

        if system == 'ew':
            nu = np.random.uniform(0.5, 2.0)
            D = np.random.uniform(0.5, 2.0)
            traj = simulate_ew(L=L, T=T, nu=nu, D=D, seed=seed)
            params_used.append({'nu': nu, 'D': D})

        elif system == 'kpz':
            nu = np.random.uniform(0.5, 2.0)
            lam = np.random.uniform(0.5, 3.0)
            D = np.random.uniform(0.5, 2.0)
            traj = simulate_kpz(L=L, T=T, nu=nu, lam=lam, D=D, seed=seed)
            if traj is None:
                lam = np.random.uniform(0.3, 1.5)
                traj = simulate_kpz(L=L, T=T, nu=nu, lam=lam, D=D,
                                    seed=seed + 10000)
            if traj is None:
                traj = simulate_kpz(L=L, T=T, nu=1.0, lam=0.5, D=1.0,
                                    seed=seed + 20000)
            params_used.append({'nu': nu, 'lam': lam, 'D': D})

        elif system == 'bd':
            traj = simulate_bd(L=L, T=T, seed=seed)
            params_used.append({})

        elif system == 'eden':
            traj = simulate_eden(L=L, T=T, seed=seed)
            params_used.append({})

        elif system == 'rd':
            traj = simulate_rd(L=L, T=T, seed=seed)
            params_used.append({})

        elif system == 'ks':
            ri = sim_kwargs.get('record_interval', 10)
            ks_T = min(T, 500)
            traj = simulate_ks(L=L, T=ks_T, seed=seed, record_interval=ri)
            if traj is None:
                traj = simulate_ks(L=L, T=ks_T, seed=seed + 10000,
                                   record_interval=ri, noise=0.02)
            if traj is None:
                traj = simulate_ks(L=L, T=300, seed=seed + 20000,
                                   record_interval=5, noise=0.01)
            params_used.append({'record_interval': ri, 'T_actual': ks_T})

        else:
            raise ValueError(f"Unknown system: {system}")

        if traj is None:
            continue

        # Extract features at every scale
        ms_feats = extract_multiscale_features(traj, block_sizes, late_frac)

        if np.any(np.isnan(ms_feats)) or np.any(np.isinf(ms_feats)):
            continue

        trajectories.append(traj)
        multiscale_list.append(ms_feats)

        # Also store per-scale features for scale-by-scale analysis
        offset = 0
        for b in block_sizes:
            features_per_scale[b].append(ms_feats[offset:offset + 10])
            offset += 10

    # Convert to arrays
    for b in block_sizes:
        if features_per_scale[b]:
            features_per_scale[b] = np.array(features_per_scale[b])
        else:
            features_per_scale[b] = np.zeros((0, 10))

    if multiscale_list:
        multiscale_features = np.array(multiscale_list)
    else:
        multiscale_features = np.zeros((0, 10 * len(block_sizes)))

    return trajectories, features_per_scale, multiscale_features, params_used


# ============================================================================
# Clustering analysis (from Exp 63, adapted)
# ============================================================================

def run_clustering(features, true_labels, class_map, label="",
                   print_output=True):
    """Run full clustering analysis on feature array.

    Returns results dict with ARI, NMI, kNN accuracy, centroid distances,
    and raw cluster labels.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans

    N, D = features.shape
    true_classes = [class_map[l] for l in true_labels]
    unique_classes = sorted(set(true_classes))
    n_classes = len(unique_classes)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pfx = f"  [{label}] " if label else "  "
    results = {'n_samples': N, 'n_features': D, 'n_classes': n_classes}

    # --- HDBSCAN ---
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, N // 20),
            min_samples=3,
            metric='euclidean'
        )
        hdb_labels = clusterer.fit_predict(X)
        n_hdb = len(set(hdb_labels) - {-1})
        n_noise = int((hdb_labels == -1).sum())

        if n_hdb >= 2:
            mask = hdb_labels != -1
            hdb_ari = adjusted_rand_score(
                np.array(true_classes)[mask], hdb_labels[mask])
            hdb_nmi = normalized_mutual_info_score(
                np.array(true_classes)[mask], hdb_labels[mask])
        else:
            hdb_ari = 0.0
            hdb_nmi = 0.0

        results['hdbscan'] = {
            'n_clusters': n_hdb, 'n_noise': n_noise,
            'ari': float(hdb_ari), 'nmi': float(hdb_nmi),
        }
        if print_output:
            print(f"{pfx}HDBSCAN: {n_hdb} clusters, {n_noise} noise, "
                  f"ARI={hdb_ari:.4f}")
    except ImportError:
        results['hdbscan'] = {'ari': 0.0}
        if print_output:
            print(f"{pfx}HDBSCAN not available")

    # --- KMeans ---
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20)
    km_labels = km.fit_predict(X)
    km_ari = adjusted_rand_score(true_classes, km_labels)
    results['kmeans'] = {'k': n_classes, 'ari': float(km_ari)}
    if print_output:
        print(f"{pfx}KMeans(K={n_classes}): ARI={km_ari:.4f}")

    # --- KMeans sweep ---
    from sklearn.metrics import silhouette_score
    km_sweep = {}
    for k in range(2, min(9, N // 5)):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs_k = km_k.fit_predict(X)
        sil = silhouette_score(X, labs_k) if k < N else 0
        ari_k = adjusted_rand_score(true_classes, labs_k)
        km_sweep[k] = {'silhouette': float(sil), 'ari': float(ari_k)}
    best_k = max(km_sweep, key=lambda k: km_sweep[k]['ari'])
    results['kmeans_sweep'] = km_sweep
    results['kmeans_best_k'] = int(best_k)
    if print_output:
        print(f"{pfx}KMeans best K={best_k} (ARI={km_sweep[best_k]['ari']:.4f})")

    # --- kNN (cross-validated) ---
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, true_classes, cv=5, scoring='accuracy')
        results[f'knn_{k}'] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
        }
        if print_output and k == 3:
            print(f"{pfx}3-NN accuracy: {np.mean(scores):.4f}")

    # --- Centroid distances ---
    centroids = {}
    for cls in unique_classes:
        mask = np.array([c == cls for c in true_classes])
        centroids[cls] = X[mask].mean(axis=0)

    centroid_dists = {}
    for i, c1 in enumerate(unique_classes):
        for c2 in unique_classes[i+1:]:
            d = np.linalg.norm(centroids[c1] - centroids[c2])
            centroid_dists[f"{c1}_vs_{c2}"] = float(d)
    results['centroid_distances'] = centroid_dists

    return results, X


# ============================================================================
# kNN graph connectivity diagnostic (NEW for Exp 64)
# ============================================================================

def knn_graph_connectivity(X, true_labels, class_map, k=5):
    """Check if each universality class forms a connected component in
    the k-NN graph.

    Returns per-class: n_components, sizes of components.

    This answers the key question: is the KPZ class geometrically
    connected (so clustering CAN merge it) or split into disconnected
    subclusters (so no clustering algorithm can help)?
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components

    true_classes = [class_map[l] for l in true_labels]
    unique_classes = sorted(set(true_classes))

    # Build full kNN graph
    graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity',
                             include_self=False)
    # Make symmetric (undirected)
    graph = graph + graph.T
    graph[graph > 0] = 1

    results = {}
    for cls in unique_classes:
        mask = np.array([c == cls for c in true_classes])
        indices = np.where(mask)[0]

        if len(indices) < 2:
            results[cls] = {'n_components': 1, 'sizes': [len(indices)]}
            continue

        # Extract subgraph for this class
        subgraph = graph[np.ix_(indices, indices)]
        n_comp, comp_labels = connected_components(subgraph, directed=False)
        sizes = [int((comp_labels == i).sum()) for i in range(n_comp)]
        sizes.sort(reverse=True)

        results[cls] = {
            'n_components': int(n_comp),
            'sizes': sizes,
        }

    return results


# ============================================================================
# Per-system feature analysis at each scale
# ============================================================================

def analyze_scale_effects(features_per_scale, labels, class_map, block_sizes):
    """Track BD-KPZ and EW-KPZ centroid distances across CG scales.

    This is the key diagnostic: does CG close BD-KPZ while opening EW-KPZ?
    """
    from sklearn.preprocessing import StandardScaler

    results = {}
    for b in block_sizes:
        feats = features_per_scale[b]
        if feats.shape[0] == 0:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(feats)

        true_classes = [class_map[l] for l in labels]
        unique_classes = sorted(set(true_classes))

        # Compute centroids
        centroids = {}
        for cls in unique_classes:
            mask = np.array([c == cls for c in true_classes])
            centroids[cls] = X[mask].mean(axis=0)

        # Per-system centroids (not just per-class)
        sys_centroids = {}
        for sys_name in sorted(set(labels)):
            mask = np.array([l == sys_name for l in labels])
            sys_centroids[sys_name] = X[mask].mean(axis=0)

        # Key distances
        scale_data = {}
        if 'EW' in centroids and 'KPZ' in centroids:
            scale_data['EW_vs_KPZ_class'] = float(
                np.linalg.norm(centroids['EW'] - centroids['KPZ']))
        if 'bd' in sys_centroids and 'kpz' in sys_centroids:
            scale_data['BD_vs_KPZ_system'] = float(
                np.linalg.norm(sys_centroids['bd'] - sys_centroids['kpz']))
        if 'bd' in sys_centroids and 'eden' in sys_centroids:
            scale_data['BD_vs_Eden_system'] = float(
                np.linalg.norm(sys_centroids['bd'] - sys_centroids['eden']))
        if 'kpz' in sys_centroids and 'eden' in sys_centroids:
            scale_data['KPZ_vs_Eden_system'] = float(
                np.linalg.norm(sys_centroids['kpz'] - sys_centroids['eden']))
        if 'ew' in sys_centroids and 'kpz' in sys_centroids:
            scale_data['EW_vs_KPZ_system'] = float(
                np.linalg.norm(sys_centroids['ew'] - sys_centroids['kpz']))

        # Per-system feature means for this scale
        sys_means = {}
        for sys_name in sorted(set(labels)):
            mask = np.array([l == sys_name for l in labels])
            sys_means[sys_name] = feats[mask].mean(axis=0).tolist()
        scale_data['per_system_means'] = sys_means

        results[int(b)] = scale_data

    return results


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(pilot=False):
    t_start = time.time()
    print("=" * 70)
    print("Experiment 64: Multi-Scale Coarse-Graining + Hierarchical Peel")
    print("=" * 70)
    print(f"Mode: {'PILOT' if pilot else 'FULL'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # ---- Configuration ----
    if pilot:
        L = 128
        T = 500
        N = 30
    else:
        L = 256
        T = 2000
        N = 80

    block_sizes = [1, 2, 4, 8]
    systems = ['ew', 'kpz', 'bd', 'eden', 'rd', 'ks']
    class_map = {
        'ew': 'EW', 'kpz': 'KPZ', 'bd': 'KPZ', 'eden': 'KPZ',
        'rd': 'trivial', 'ks': 'KS'
    }
    # For the peel subproblem: only EW/KPZ family
    peel_systems = ['ew', 'kpz', 'bd', 'eden']
    peel_class_map = {
        'ew': 'EW', 'kpz': 'KPZ', 'bd': 'KPZ', 'eden': 'KPZ',
    }

    n_features_per_scale = 10
    n_scales = len(block_sizes)

    config = {
        'L': L, 'T': T, 'N': N, 'systems': systems,
        'mode': 'pilot' if pilot else 'full',
        'block_sizes': block_sizes,
        'n_features_per_scale': n_features_per_scale,
    }

    print(f"Config: L={L}, T={T}, N={N}/system ({N*len(systems)} total)")
    print(f"Block sizes: {block_sizes}")
    print(f"Features per scale: {n_features_per_scale} "
          f"(6 spatial + 4 temporal)")
    print(f"Multi-scale: {n_features_per_scale * n_scales}D "
          f"({n_features_per_scale}D x {n_scales} scales)")
    print()

    # ================================================================
    # STEP 1: Generate all trajectories and extract multi-scale features
    # ================================================================
    print("STEP 1: Generating data + multi-scale features")
    print("-" * 60)

    all_features_per_scale = {b: [] for b in block_sizes}
    all_multiscale = []
    all_labels = []

    for sys_name in systems:
        t1 = time.time()
        print(f"  {sys_name:6s}: generating {N} samples...", end='', flush=True)

        sim_kwargs = {}
        if sys_name == 'ks':
            sim_kwargs['record_interval'] = 10 if pilot else 20

        trajs, feat_per_scale, ms_feats, params = generate_dataset(
            N, L, T, sys_name,
            seed_offset=hash(sys_name) % 10000,
            late_frac=0.3,
            block_sizes=block_sizes,
            **sim_kwargs
        )

        n_valid = ms_feats.shape[0]
        all_multiscale.append(ms_feats)
        all_labels.extend([sys_name] * n_valid)

        for b in block_sizes:
            all_features_per_scale[b].append(feat_per_scale[b])

        dt = time.time() - t1
        if n_valid > 0:
            b1_means = feat_per_scale[1].mean(axis=0) if feat_per_scale[1].shape[0] > 0 else np.zeros(10)
            print(f" {n_valid}/{N} valid ({dt:.1f}s)  "
                  f"gvar={b1_means[0]:.3f}  beta={b1_means[6]:.3f}  "
                  f"sg={b1_means[9]:+.4f}")
        else:
            print(f" ALL FAILED ({dt:.1f}s)")

    # Stack
    multiscale_features = np.vstack(all_multiscale)
    for b in block_sizes:
        all_features_per_scale[b] = np.vstack(all_features_per_scale[b])

    total_n = len(all_labels)
    print(f"\n  Total: {total_n} samples")
    print(f"  Per-scale features: {all_features_per_scale[1].shape}")
    print(f"  Multi-scale features: {multiscale_features.shape}")

    # ================================================================
    # STEP 2: Scale-by-scale analysis (key diagnostic)
    # ================================================================
    print("\n\nSTEP 2: Scale-by-scale centroid analysis")
    print("-" * 60)

    scale_analysis = analyze_scale_effects(
        all_features_per_scale, all_labels, class_map, block_sizes)

    print(f"\n  {'Scale':>6s}  {'BD-KPZ':>8s}  {'BD-Eden':>8s}  "
          f"{'KPZ-Eden':>8s}  {'EW-KPZ(cls)':>11s}  {'EW-KPZ(sys)':>11s}")
    print(f"  {'-'*60}")
    for b in block_sizes:
        if b not in scale_analysis:
            continue
        sa = scale_analysis[b]
        bd_kpz = sa.get('BD_vs_KPZ_system', 0)
        bd_eden = sa.get('BD_vs_Eden_system', 0)
        kpz_eden = sa.get('KPZ_vs_Eden_system', 0)
        ew_kpz_cls = sa.get('EW_vs_KPZ_class', 0)
        ew_kpz_sys = sa.get('EW_vs_KPZ_system', 0)
        print(f"  b={b:4d}  {bd_kpz:8.3f}  {bd_eden:8.3f}  "
              f"{kpz_eden:8.3f}  {ew_kpz_cls:11.3f}  {ew_kpz_sys:11.3f}")

    # Check predictions P1 (BD-KPZ decreases) and P2 (EW-KPZ increases)
    if len(scale_analysis) >= 2:
        b1 = block_sizes[0]
        b_max = block_sizes[-1]
        bd_kpz_b1 = scale_analysis[b1].get('BD_vs_KPZ_system', 0)
        bd_kpz_bmax = scale_analysis[b_max].get('BD_vs_KPZ_system', 0)
        ew_kpz_b1 = scale_analysis[b1].get('EW_vs_KPZ_class', 0)
        ew_kpz_bmax = scale_analysis[b_max].get('EW_vs_KPZ_class', 0)

        print(f"\n  P1 (BD-KPZ decreases): b=1 -> b={b_max}: "
              f"{bd_kpz_b1:.3f} -> {bd_kpz_bmax:.3f} "
              f"({'PASS' if bd_kpz_bmax < bd_kpz_b1 else 'FAIL'})")
        print(f"  P2 (EW-KPZ increases): b=1 -> b={b_max}: "
              f"{ew_kpz_b1:.3f} -> {ew_kpz_bmax:.3f} "
              f"({'PASS' if ew_kpz_bmax > ew_kpz_b1 else 'FAIL'})")

    # ================================================================
    # STEP 3: kNN graph connectivity diagnostic
    # ================================================================
    print("\n\nSTEP 3: kNN graph connectivity per class")
    print("-" * 60)

    connectivity_results = {}
    from sklearn.preprocessing import StandardScaler

    for b in block_sizes:
        feats = all_features_per_scale[b]
        scaler = StandardScaler()
        X = scaler.fit_transform(feats)
        conn = knn_graph_connectivity(X, all_labels, class_map, k=5)
        connectivity_results[int(b)] = conn

        print(f"\n  Scale b={b}:")
        for cls, info in sorted(conn.items()):
            n_comp = info['n_components']
            sizes = info['sizes']
            marker = " <-- SPLIT" if n_comp > 1 else ""
            print(f"    {cls:8s}: {n_comp} component(s), "
                  f"sizes={sizes}{marker}")

    # Check P5: KPZ class 2 components at b=1, 1 at b>=4
    kpz_comp_b1 = connectivity_results.get(1, {}).get('KPZ', {}).get('n_components', 0)
    kpz_comp_b4 = connectivity_results.get(4, {}).get('KPZ', {}).get('n_components', 0)
    print(f"\n  P5 (KPZ: 2 comp at b=1, 1 at b>=4): "
          f"b=1: {kpz_comp_b1}, b=4: {kpz_comp_b4}")

    # ================================================================
    # STEP 4A: Clustering — single-scale comparison
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 4A: Clustering at each CG scale (all 6 systems, 4 classes)")
    print("=" * 70)

    scale_clustering = {}
    for b in block_sizes:
        feats = all_features_per_scale[b]
        print(f"\n  --- Scale b={b} ({feats.shape[1]}D) ---")
        res, X = run_clustering(feats, all_labels, class_map,
                                label=f"b={b}")
        scale_clustering[int(b)] = res

    # ================================================================
    # STEP 4B: Clustering — multi-scale concatenated
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 4B: Multi-scale concatenated features "
          f"({multiscale_features.shape[1]}D)")
    print("=" * 70)

    res_ms, X_ms = run_clustering(
        multiscale_features, all_labels, class_map, label="multi-scale")

    # ================================================================
    # STEP 5A: Hierarchical peel — remove RD + KS
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 5: Hierarchical Peel (remove RD + KS)")
    print("=" * 70)

    peel_mask = np.array([l in peel_systems for l in all_labels])
    peel_labels = [l for l in all_labels if l in peel_systems]
    n_peel = sum(peel_mask)

    print(f"\n  Peeled dataset: {n_peel} samples "
          f"({sorted(set(peel_labels))})")
    print(f"  Classes: {sorted(set(peel_class_map[l] for l in peel_labels))}")

    # 5A: Peel + b=1 features (10D)
    print(f"\n  --- 5A: Peel + raw features (b=1, 10D) ---")
    peel_feats_b1 = all_features_per_scale[1][peel_mask]
    res_peel_b1, X_peel_b1 = run_clustering(
        peel_feats_b1, peel_labels, peel_class_map, label="peel+b=1")

    # 5B: Peel + per-scale
    print()
    peel_scale_results = {}
    for b in block_sizes:
        peel_feats = all_features_per_scale[b][peel_mask]
        print(f"  --- 5B: Peel + b={b} ({peel_feats.shape[1]}D) ---")
        res, X = run_clustering(
            peel_feats, peel_labels, peel_class_map,
            label=f"peel+b={b}", print_output=(b == 1 or b == block_sizes[-1]))
        peel_scale_results[int(b)] = res
        if not (b == 1 or b == block_sizes[-1]):
            hdb_ari = res.get('hdbscan', {}).get('ari', 0)
            km_ari = res['kmeans']['ari']
            knn3 = res['knn_3']['mean_accuracy']
            print(f"    [peel+b={b}] ARI(H)={hdb_ari:.4f}, "
                  f"ARI(K)={km_ari:.4f}, 3-NN={knn3:.4f}")

    # 5C: Peel + multiscale
    print(f"\n  --- 5C: Peel + multi-scale ({multiscale_features[peel_mask].shape[1]}D) ---")
    peel_ms_feats = multiscale_features[peel_mask]
    res_peel_ms, X_peel_ms = run_clustering(
        peel_ms_feats, peel_labels, peel_class_map,
        label="peel+multi-scale")

    # Peel connectivity
    print(f"\n  --- 5D: kNN graph connectivity (peeled, b=1) ---")
    peel_conn = knn_graph_connectivity(
        X_peel_b1, peel_labels, peel_class_map, k=5)
    for cls, info in sorted(peel_conn.items()):
        n_comp = info['n_components']
        sizes = info['sizes']
        marker = " <-- STILL SPLIT" if n_comp > 1 else " <-- CONNECTED"
        print(f"    {cls:8s}: {n_comp} component(s), "
              f"sizes={sizes}{marker}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Best ARI across all conditions
    all_results = {}

    # 4A: per-scale, all systems
    for b in block_sizes:
        r = scale_clustering[int(b)]
        hdb = r.get('hdbscan', {}).get('ari', 0)
        km = r['kmeans']['ari']
        knn3 = r['knn_3']['mean_accuracy']
        all_results[f"all_b={b}"] = {
            'hdbscan': hdb, 'kmeans': km, 'knn_3': knn3}

    # 4B: multiscale, all systems
    hdb_ms = res_ms.get('hdbscan', {}).get('ari', 0)
    km_ms = res_ms['kmeans']['ari']
    knn3_ms = res_ms['knn_3']['mean_accuracy']
    all_results["all_multiscale"] = {
        'hdbscan': hdb_ms, 'kmeans': km_ms, 'knn_3': knn3_ms}

    # 5A: peel + b=1
    hdb_p1 = res_peel_b1.get('hdbscan', {}).get('ari', 0)
    km_p1 = res_peel_b1['kmeans']['ari']
    knn3_p1 = res_peel_b1['knn_3']['mean_accuracy']
    all_results["peel_b=1"] = {
        'hdbscan': hdb_p1, 'kmeans': km_p1, 'knn_3': knn3_p1}

    # 5B: peel + per-scale
    for b in block_sizes:
        r = peel_scale_results[int(b)]
        hdb = r.get('hdbscan', {}).get('ari', 0)
        km = r['kmeans']['ari']
        knn3 = r['knn_3']['mean_accuracy']
        all_results[f"peel_b={b}"] = {
            'hdbscan': hdb, 'kmeans': km, 'knn_3': knn3}

    # 5C: peel + multiscale
    hdb_pm = res_peel_ms.get('hdbscan', {}).get('ari', 0)
    km_pm = res_peel_ms['kmeans']['ari']
    knn3_pm = res_peel_ms['knn_3']['mean_accuracy']
    all_results["peel_multiscale"] = {
        'hdbscan': hdb_pm, 'kmeans': km_pm, 'knn_3': knn3_pm}

    # Print summary table
    print(f"\n  {'Condition':<22s}  {'HDBSCAN':>8s}  {'KMeans':>8s}  "
          f"{'3-NN':>8s}  {'Best ARI':>8s}")
    print(f"  {'-'*62}")

    for name, metrics in sorted(all_results.items()):
        best = max(metrics['hdbscan'], metrics['kmeans'])
        marker = " ***" if best > 0.55 else ""
        print(f"  {name:<22s}  {metrics['hdbscan']:8.4f}  "
              f"{metrics['kmeans']:8.4f}  {metrics['knn_3']:8.4f}  "
              f"{best:8.4f}{marker}")

    # Best overall
    best_name = max(all_results, key=lambda n: max(
        all_results[n]['hdbscan'], all_results[n]['kmeans']))
    best_metrics = all_results[best_name]
    best_ari = max(best_metrics['hdbscan'], best_metrics['kmeans'])

    print(f"\n  BEST: {best_name} -> ARI={best_ari:.4f}")
    print(f"  Exp 63 baseline (10D, b=1, all systems): "
          f"HDBSCAN={all_results.get('all_b=1', {}).get('hdbscan', 0):.4f}")

    # ---- Predictions ----
    print(f"\n  Predictions:")
    predictions = {}

    # P1: BD-KPZ decreases with CG
    if len(scale_analysis) >= 2:
        b1 = block_sizes[0]
        bmax = block_sizes[-1]
        bd_kpz_1 = scale_analysis[b1].get('BD_vs_KPZ_system', 0)
        bd_kpz_max = scale_analysis[bmax].get('BD_vs_KPZ_system', 0)
        p1 = bd_kpz_max < bd_kpz_1
        predictions['P1_BD_KPZ_decreases'] = p1
        print(f"    P1 BD-KPZ decreases: {'PASS' if p1 else 'FAIL'} "
              f"({bd_kpz_1:.3f} -> {bd_kpz_max:.3f})")

    # P2: EW-KPZ increases with CG
    if len(scale_analysis) >= 2:
        ew_kpz_1 = scale_analysis[b1].get('EW_vs_KPZ_class', 0)
        ew_kpz_max = scale_analysis[bmax].get('EW_vs_KPZ_class', 0)
        p2 = ew_kpz_max > ew_kpz_1
        predictions['P2_EW_KPZ_increases'] = p2
        print(f"    P2 EW-KPZ increases: {'PASS' if p2 else 'FAIL'} "
              f"({ew_kpz_1:.3f} -> {ew_kpz_max:.3f})")

    # P3: Optimal scale exists
    best_scale_k = max(block_sizes,
                       key=lambda b: scale_clustering.get(int(b), {}).get(
                           'kmeans', {}).get('ari', 0))
    p3 = best_scale_k not in [block_sizes[0], block_sizes[-1]]
    predictions['P3_optimal_scale'] = p3
    print(f"    P3 Optimal scale exists: {'PASS' if p3 else 'FAIL'} "
          f"(best k-means at b={best_scale_k})")

    # P4: Peel improves ARI
    peel_best = max(
        res_peel_b1.get('hdbscan', {}).get('ari', 0),
        res_peel_b1['kmeans']['ari']
    )
    all_best = max(
        scale_clustering.get(1, {}).get('hdbscan', {}).get('ari', 0),
        scale_clustering.get(1, {}).get('kmeans', {}).get('ari', 0)
    )
    p4 = peel_best > all_best
    predictions['P4_peel_improves'] = p4
    print(f"    P4 Peel improves ARI: {'PASS' if p4 else 'FAIL'} "
          f"(all={all_best:.4f}, peel={peel_best:.4f})")

    # P5: KPZ connectivity
    kpz_b1_comp = connectivity_results.get(1, {}).get(
        'KPZ', {}).get('n_components', 0)
    kpz_b4_comp = connectivity_results.get(4, {}).get(
        'KPZ', {}).get('n_components', 0)
    p5 = kpz_b1_comp > 1 and kpz_b4_comp == 1
    predictions['P5_KPZ_merges_under_CG'] = p5
    print(f"    P5 KPZ merges under CG: {'PASS' if p5 else 'FAIL'} "
          f"(b=1: {kpz_b1_comp} comp, b=4: {kpz_b4_comp} comp)")

    # P6: Best ARI > 0.6
    p6 = best_ari > 0.6
    predictions['P6_ARI_above_06'] = p6
    print(f"    P6 ARI > 0.6: {'PASS' if p6 else 'FAIL'} "
          f"(best={best_ari:.4f})")

    # P7: Multi-scale > any single scale
    ms_best_ari = max(hdb_ms, km_ms)
    single_best_ari = max(
        max(scale_clustering.get(int(b), {}).get('hdbscan', {}).get('ari', 0),
            scale_clustering.get(int(b), {}).get('kmeans', {}).get('ari', 0))
        for b in block_sizes
    )
    p7 = ms_best_ari > single_best_ari
    predictions['P7_multiscale_beats_single'] = p7
    print(f"    P7 Multi-scale > single: {'PASS' if p7 else 'FAIL'} "
          f"(ms={ms_best_ari:.4f}, best single={single_best_ari:.4f})")

    n_pass = sum(1 for v in predictions.values() if v)
    print(f"\n    Score: {n_pass}/{len(predictions)} predictions passed")

    # ---- Verdict ----
    print(f"\n  {'='*60}")
    if best_ari > 0.6:
        print(f"  VERDICT: Substantial improvement! Best ARI = {best_ari:.4f}")
        print(f"  achieved at {best_name}.")
    elif best_ari > 0.55:
        print(f"  VERDICT: Modest improvement. Best ARI = {best_ari:.4f}")
        print(f"  at {best_name}. CG trade-off partially resolved.")
    else:
        print(f"  VERDICT: No clear clustering improvement. Best ARI = {best_ari:.4f}")
        print(f"  The CG trade-off (BD closure vs EW-KPZ expansion)")
        print(f"  prevents net improvement in unsupervised ARI.")
        if kpz_b1_comp > 1:
            print(f"  KPZ class has {kpz_b1_comp} components in kNN graph —")
            print(f"  consistent with geometric multimodality. Algorithm")
            print(f"  swaps alone are unlikely to resolve this.")

    # ---- Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        'config': config,
        'scale_analysis': scale_analysis,
        'connectivity': {str(b): {k: v for k, v in conn.items()}
                         for b, conn in connectivity_results.items()},
        'clustering_per_scale': {str(b): r for b, r in scale_clustering.items()},
        'clustering_multiscale': res_ms,
        'peel_b1': res_peel_b1,
        'peel_per_scale': {str(b): r for b, r in peel_scale_results.items()},
        'peel_multiscale': res_peel_ms,
        'peel_connectivity': peel_conn,
        'summary': all_results,
        'predictions': predictions,
        'best': {'condition': best_name, 'ari': best_ari},
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t_start,
    }

    results_path = RESULTS_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return out


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exp 64: Multi-Scale CG + Hierarchical Peel')
    parser.add_argument('--pilot', action='store_true',
                        help='Quick test (~8 min)')
    args = parser.parse_args()

    results = run_experiment(pilot=args.pilot)
