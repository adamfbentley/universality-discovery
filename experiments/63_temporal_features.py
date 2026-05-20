"""
Experiment 63: Temporal Growth Features for Universality Clustering
=====================================================================

PURPOSE: Break the ARI~0.5 ceiling from Exp 62 by adding temporal features
that target the EW-KPZ degeneracy in the 6D gradient feature space.

DIAGNOSIS (from Exp 62):
  The 6 gradient features (grad_var, grad_skew, grad_kurt, lap_var,
  grad_lap_cov, h_var) achieve ARI=0.495 and kNN=82%, but EW and KPZ
  are nearly indistinguishable (centroid distance = 0.87 in standardised
  space, vs 1.85 for KPZ-KS and 4.58 for KPZ-trivial).

  WHY: At stationarity, the 1D KPZ gradient distribution is exactly
  Gaussian with Var[g] = D/nu, INDEPENDENT of lambda (Exp 54 theorem).
  So gradient-moment features cannot see the nonlinearity. EW and KPZ
  have the same stationary gradient statistics.

SOLUTION: Add temporal features that DO see the nonlinearity:

  1. beta_eff        — Growth exponent from W(t) ~ t^beta
                       (beta_EW=0.25, beta_KPZ=0.33, beta_RD=0.50)
  2. vel_skew        — Skew[dh/dt]: nonlinearity signature
                       (EW~0, KPZ>0 from +(lam/2)(grad h)^2, KS<0 from -(lam/2)(grad h)^2)
  3. vel_kurt        — Kurt[dh/dt]-3: dynamics tail weight
  4. slope_growth    — Cov[dh/dt, (dh/dx)^2]: direct nonlinear coupling
                       Approximately 0 for EW under the Gaussian-field argument
                       Tracks the KPZ nonlinearity in the Exp 13 diagnostics

PHYSICAL ARGUMENT:
  KPZ velocity v(x) = nu*lap(h) + (lam/2)*(grad h)^2 + noise.
  For EW, the (grad h)^2 term is absent => velocity is symmetric.
  The slope-growth covariance Cov[v, g^2] vanishes exactly for a Gaussian
  field (odd moment of centered Gaussian = 0), so any nonzero value
  is a direct signature of the KPZ nonlinearity.

EXPECTED SIGNATURES:
  System   | beta  | vel_skew | slope_growth
  ---------|-------|----------|--------------
  EW       | ~0.25 |  ~0      |  ~0
  KPZ      | ~0.33 |  >0      |  >0
  BD       | ~0.33 |  >0      |  >0
  Eden     | ~0.33 |  >0      |  >0
  RD       | ~0.50 |  >0*     |  ~0
  KS       | varies|  <0      |  <0
  * RD vel_skew > 0 from half-normal deposit distribution, not from nonlinearity

FEATURES (10D = 6 spatial + 4 temporal):
  Spatial (Exp 62, validated):  grad_var, grad_skew, grad_kurt,
                                lap_var, grad_lap_cov, h_var
  Temporal (NEW):               beta_eff, vel_skew, vel_kurt, slope_growth

PREDICTIONS:
  P1: ARI > 0.5 (breaks Exp 62 ceiling)
  P2: vel_skew separates EW (~0) from KPZ (>0)
  P3: beta separates EW (0.25) from KPZ (0.33) from RD (0.50)
  P4: kNN accuracy > 85% (up from 82%)
  P5: EW-KPZ centroid distance > 1.5 (up from 0.87)
  P6: slope_growth < 0 for KS (negative nonlinearity)
  P7: slope_growth ~ 0 for EW (Gaussian field argument)

SUCCESS CRITERIA:
  - ARI > 0.5 (HDBSCAN or KMeans, on 10D features)
  - kNN accuracy > 0.85
  - EW-KPZ centroid distance increase > 50% vs 6D baseline
  - Direct comparison shows improvement over Exp 62

Usage:
  python 63_temporal_features.py --pilot    # Quick (~5 min)
  python 63_temporal_features.py            # Full (~25 min)
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

# Unicode safety for Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Reproducibility
np.random.seed(42)

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / 'results_exp63'
FIGURES_DIR = PROJECT_DIR / 'figures'


# ============================================================================
# Simulators (from Exp 62, all 1D periodic)
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
    """KPZ equation: dh/dt = nu*lap(h) + (lam/2)*(grad h)^2 + noise.

    Uses smaller dt (0.01) for stability with large lam/nu ratios.
    Steps_per_unit = 1/dt, so we substep within each recorded time step.
    """
    if seed is not None:
        np.random.seed(seed)
    h = np.random.randn(L) * 0.01
    trajectory = np.zeros((T, L))
    substeps = max(1, int(0.05 / dt))  # keep same physical time per recorded step
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
        # Check for blowup
        if np.any(np.isnan(h)) or np.max(np.abs(h)) > 1e6:
            return None  # signal failure
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
                return None  # blowup
            trajectory[snap] = h
            snap += 1
    return trajectory


# ============================================================================
# Spatial feature extraction (from Exp 62 and related diagnostics)
# ============================================================================

@jit(nopython=True)
def compute_features_single(h):
    """Compute 6D spatial feature vector from a single 1D height profile.

    Features:
      [0] grad_var       = Var[dh/dx]
      [1] grad_skew      = Skew[dh/dx]
      [2] grad_kurt      = Kurt[dh/dx] - 3
      [3] lap_var        = Var[d^2h/dx^2]
      [4] grad_lap_cov   = Cov[|dh/dx|, d^2h/dx^2]
      [5] h_var          = Var[h]
    """
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
    """Extract 6D spatial features from the late-time portion of a trajectory."""
    T = trajectory.shape[0]
    start = int(T * (1 - late_frac))
    features = []
    for t in range(start, T):
        features.append(compute_features_single(trajectory[t]))
    return np.mean(features, axis=0)


# ============================================================================
# NEW: Temporal feature extraction
# ============================================================================

TEMPORAL_NAMES = ['beta_eff', 'vel_skew', 'vel_kurt', 'slope_growth']
ALL_FEATURE_NAMES = SPATIAL_NAMES + TEMPORAL_NAMES


def compute_temporal_features(trajectory):
    """Compute 4D temporal feature vector from full (T, L) trajectory.

    Features:
      [0] beta_eff     = Effective growth exponent from W(t) ~ t^beta
      [1] vel_skew     = Skew[dh/dt] — nonlinearity signature (multi-step)
      [2] vel_kurt     = Kurt[dh/dt] - 3 — dynamics tail weight (multi-step)
      [3] slope_growth = Corr[dh/dt, (dh/dx)^2] — nonlinear coupling (Pearson r)

    Physical justification:
      - beta: EW=0.25, KPZ=0.33, RD=0.50 — directly distinguishes dynamics
      - vel_skew: 0 for EW (symmetric), >0 for KPZ (positive nonlinearity),
        <0 for KS (negative nonlinearity). Uses multi-step velocity to
        accumulate nonlinear signal above noise floor.
      - slope_growth: Pearson correlation between velocity and (grad h)^2.
        Exactly 0 for Gaussian fields (EW), positive for KPZ.
        Using correlation (not covariance) makes it dimensionless and
        bounded in [-1,1], preventing amplitude outliers from discrete
        models (BD) distorting the clustering.

    Implementation notes:
      - Velocity uses stride > 1 (multi-step) to improve SNR for skewness.
        Single-step Δh is noise-dominated at small dt; stride ~10 accumulates
        enough nonlinear signal for the (∇h)² term to emerge above noise.
      - Slope-growth uses Pearson r (dimensionless) instead of raw Cov,
        because BD has ~600x larger amplitude than KPZ continuum (pilot).
    """
    T, L = trajectory.shape

    # === 1. Growth exponent beta from W(t) ~ t^beta ===
    widths = np.std(trajectory, axis=1)  # W(t) = std_x[h(x,t)] for each t

    # Fit in growth regime: [5% T, 40% T] — before saturation
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

    # === 2-4. Late-time velocity and slope-growth features ===
    late_start = max(1, int(T * 0.7))
    if late_start >= T - 2:
        late_start = max(1, T - 10)

    n_avail = T - late_start

    # Multi-step velocity: use stride > 1 to accumulate nonlinear signal
    # above the single-step noise floor. stride ~10 gives ~10x noise
    # averaging while preserving nonlinear (∇h)² contribution.
    stride = max(1, min(10, n_avail // 20))

    # Height arrays for multi-step velocity
    h_early = trajectory[late_start: T - stride]        # (n_steps, L)
    h_later = trajectory[late_start + stride: T]         # (n_steps, L)
    n_steps = h_early.shape[0]

    if n_steps < 2:
        return np.array([beta_eff, 0.0, 0.0, 0.0])

    # Velocity field: v(x,t) = h(x, t+stride) - h(x, t)
    vel = h_later - h_early  # (n_steps, L)

    # --- Velocity statistics (pool over x and t) ---
    vel_flat = vel.flatten()
    vel_mean = np.mean(vel_flat)
    vel_var = np.var(vel_flat)
    vel_std = max(np.sqrt(vel_var), 1e-15)
    vel_c = vel_flat - vel_mean
    vel_skew = float(np.clip(np.mean((vel_c / vel_std)**3), -50.0, 50.0))
    vel_kurt = float(np.clip(np.mean((vel_c / vel_std)**4) - 3.0, -50.0, 200.0))

    # --- Slope-growth: Pearson correlation r(v, g^2) ---
    # Use gradient at the earlier time step (same as velocity reference)
    grad = (np.roll(h_early, -1, axis=1) -
            np.roll(h_early, 1, axis=1)) / 2.0  # (n_steps, L)
    grad_sq = grad**2

    # Per-snapshot Pearson correlation, then average
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


# ============================================================================
# Dataset generation (returns both spatial and temporal features)
# ============================================================================

def generate_feature_dataset(n_samples, L, T, system, seed_offset=0,
                             late_frac=0.3, **sim_kwargs):
    """Generate spatial and temporal features for n_samples from a system.

    Returns:
        spatial_features: (n_valid, 6) array
        temporal_features: (n_valid, 4) array
        params_used: list of dicts
    """
    spatial_list = []
    temporal_list = []
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

        # Extract both feature sets from the same trajectory
        spatial = extract_spatial_features(traj, late_frac)
        temporal = compute_temporal_features(traj)
        combined = np.concatenate([spatial, temporal])

        if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
            continue

        spatial_list.append(spatial)
        temporal_list.append(temporal)

    if len(spatial_list) == 0:
        return np.zeros((0, 6)), np.zeros((0, 4)), params_used
    return np.array(spatial_list), np.array(temporal_list), params_used


# ============================================================================
# Clustering analysis (from Exp 62, generalised for arbitrary feature dim)
# ============================================================================

def run_clustering_analysis(features, true_labels, class_map, label=""):
    """Run full clustering analysis on feature array.

    Args:
        features: (N, D) array
        true_labels: list of system names
        class_map: dict mapping system -> universality class
        label: string label for printing (e.g. "6D" or "10D")

    Returns:
        results dict, X_scaled, hdb_labels, km_labels
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans

    N, D = features.shape
    true_classes = [class_map[l] for l in true_labels]
    unique_classes = sorted(set(true_classes))
    n_true_classes = len(unique_classes)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pfx = f"  [{label}] " if label else "  "

    results = {
        'n_samples': N,
        'n_features': D,
        'n_true_classes': n_true_classes,
        'true_class_counts': {c: true_classes.count(c) for c in unique_classes},
    }

    # --- Per-system feature statistics ---
    per_system = {}
    for sys_name in sorted(set(true_labels)):
        mask = np.array([l == sys_name for l in true_labels])
        per_system[sys_name] = {
            'n': int(mask.sum()),
            'feature_means': features[mask].mean(axis=0).tolist(),
            'feature_stds': features[mask].std(axis=0).tolist(),
        }
    results['per_system'] = per_system

    # --- HDBSCAN ---
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(5, N // 20),
            min_samples=3,
            metric='euclidean'
        )
        hdb_labels = clusterer.fit_predict(X)
        n_hdb_clusters = len(set(hdb_labels) - {-1})
        n_noise = int((hdb_labels == -1).sum())

        if n_hdb_clusters >= 2:
            mask = hdb_labels != -1
            hdb_ari = adjusted_rand_score(
                np.array(true_classes)[mask], hdb_labels[mask])
            hdb_nmi = normalized_mutual_info_score(
                np.array(true_classes)[mask], hdb_labels[mask])
        else:
            hdb_ari = 0.0
            hdb_nmi = 0.0

        results['hdbscan'] = {
            'n_clusters': n_hdb_clusters,
            'n_noise': n_noise,
            'ari': float(hdb_ari),
            'nmi': float(hdb_nmi),
            'labels': hdb_labels.tolist(),
        }
        print(f"{pfx}HDBSCAN: {n_hdb_clusters} clusters, {n_noise} noise, "
              f"ARI={hdb_ari:.3f}, NMI={hdb_nmi:.3f}")
    except ImportError:
        print(f"{pfx}HDBSCAN not available")
        results['hdbscan'] = {'error': 'not installed'}
        hdb_labels = None

    # --- KMeans (K = n_true_classes) ---
    km = KMeans(n_clusters=n_true_classes, random_state=42, n_init=20)
    km_labels = km.fit_predict(X)
    km_ari = adjusted_rand_score(true_classes, km_labels)
    km_nmi = normalized_mutual_info_score(true_classes, km_labels)
    results['kmeans'] = {
        'k': n_true_classes,
        'ari': float(km_ari),
        'nmi': float(km_nmi),
        'labels': km_labels.tolist(),
    }
    print(f"{pfx}KMeans(K={n_true_classes}): ARI={km_ari:.3f}, NMI={km_nmi:.3f}")

    # --- KMeans sweep (K=2..8) ---
    from sklearn.metrics import silhouette_score
    km_sweep = {}
    for k in range(2, 9):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km_k.fit_predict(X)
        sil = silhouette_score(X, labels_k)
        ari_k = adjusted_rand_score(true_classes, labels_k)
        km_sweep[k] = {'silhouette': float(sil), 'ari': float(ari_k)}
    best_k = max(km_sweep, key=lambda k: km_sweep[k]['silhouette'])
    results['kmeans_sweep'] = km_sweep
    results['kmeans_best_k'] = best_k
    print(f"{pfx}KMeans best K={best_k} "
          f"(sil={km_sweep[best_k]['silhouette']:.3f}, "
          f"ARI={km_sweep[best_k]['ari']:.3f})")

    # --- kNN classification (cross-validated) ---
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, true_classes, cv=5, scoring='accuracy')
        results[f'knn_{k}'] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
        }
        print(f"{pfx}{k}-NN CV accuracy: {np.mean(scores):.3f} "
              f"+/- {np.std(scores):.3f}")

    # --- Pairwise centroid distances ---
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
    print(f"{pfx}Centroid distances: {centroid_dists}")

    return results, X, hdb_labels, km_labels


# ============================================================================
# Visualization
# ============================================================================

def make_figures(spatial_feats, temporal_feats, combined_feats,
                 X6_scaled, X10_scaled,
                 true_labels, true_classes, class_map,
                 hdb6, hdb10, km6, km10,
                 results_6d, results_10d, save_dir):
    """Generate comprehensive analysis figures with 6D vs 10D comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sys_colors = {
        'ew': '#2196F3', 'kpz': '#4CAF50', 'bd': '#8BC34A',
        'eden': '#CDDC39', 'rd': '#FF9800', 'ks': '#F44336'
    }
    class_colors = {
        'EW': '#2196F3', 'KPZ': '#4CAF50',
        'trivial': '#FF9800', 'KS': '#F44336'
    }

    unique_systems = sorted(set(true_labels))
    from sklearn.decomposition import PCA

    # ========== Figure 1: 6D vs 10D PCA comparison ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel 1: 6D PCA (baseline)
    ax = axes[0, 0]
    pca6 = PCA(n_components=2)
    Z6 = pca6.fit_transform(X6_scaled)
    for sys_name in unique_systems:
        mask = np.array([l == sys_name for l in true_labels])
        ax.scatter(Z6[mask, 0], Z6[mask, 1],
                   c=sys_colors.get(sys_name, '#999'), label=sys_name,
                   alpha=0.7, s=25, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca6.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca6.explained_variance_ratio_[1]:.1%})')
    ari6 = results_6d.get('hdbscan', {}).get('ari', 0)
    ax.set_title(f'6D Spatial Only (ARI={ari6:.3f})')
    ax.legend(fontsize=7)

    # Panel 2: 10D PCA (enhanced)
    ax = axes[0, 1]
    pca10 = PCA(n_components=2)
    Z10 = pca10.fit_transform(X10_scaled)
    for sys_name in unique_systems:
        mask = np.array([l == sys_name for l in true_labels])
        ax.scatter(Z10[mask, 0], Z10[mask, 1],
                   c=sys_colors.get(sys_name, '#999'), label=sys_name,
                   alpha=0.7, s=25, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca10.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca10.explained_variance_ratio_[1]:.1%})')
    ari10 = results_10d.get('hdbscan', {}).get('ari', 0)
    ax.set_title(f'10D Spatial+Temporal (ARI={ari10:.3f})')
    ax.legend(fontsize=7)

    # Panel 3: 10D UMAP
    ax = axes[0, 2]
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42)
        Z_umap = reducer.fit_transform(X10_scaled)
        for sys_name in unique_systems:
            mask = np.array([l == sys_name for l in true_labels])
            ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                       c=sys_colors.get(sys_name, '#999'), label=sys_name,
                       alpha=0.7, s=25, edgecolors='none')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('10D Feature Space (UMAP)')
        ax.legend(fontsize=7)
    except ImportError:
        ax.text(0.5, 0.5, 'umap-learn not installed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP (unavailable)')

    # Panel 4: HDBSCAN clusters for 10D (PCA view)
    ax = axes[1, 0]
    if hdb10 is not None:
        cmap = plt.cm.tab10
        for cl in sorted(set(hdb10)):
            mask = hdb10 == cl
            color = 'lightgray' if cl == -1 else cmap(cl % 10)
            label = 'Noise' if cl == -1 else f'Cluster {cl}'
            ax.scatter(Z10[mask, 0], Z10[mask, 1],
                       c=[color], label=label, alpha=0.7, s=25)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'10D HDBSCAN (ARI={ari10:.3f})')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'HDBSCAN not available',
                ha='center', va='center', transform=ax.transAxes)

    # Panel 5: KMeans confusion matrix for 10D
    ax = axes[1, 1]
    unique_true = sorted(set(true_classes))
    unique_km = sorted(set(km10))
    confusion = np.zeros((len(unique_true), len(unique_km)))
    for tc, kl in zip(true_classes, km10):
        confusion[unique_true.index(tc), unique_km.index(kl)] += 1
    im = ax.imshow(confusion, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(unique_km)))
    ax.set_xticklabels([str(c) for c in unique_km], fontsize=9)
    ax.set_yticks(range(len(unique_true)))
    ax.set_yticklabels(unique_true, fontsize=9)
    ax.set_xlabel('KMeans Cluster')
    ax.set_ylabel('True Class')
    km10_ari = results_10d['kmeans']['ari']
    ax.set_title(f'10D KMeans Confusion (ARI={km10_ari:.3f})')
    for i in range(len(unique_true)):
        for j in range(len(unique_km)):
            val = int(confusion[i, j])
            color = 'white' if val > confusion.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    # Panel 6: Comparison bar chart
    ax = axes[1, 2]
    metrics = ['HDBSCAN\nARI', 'KMeans\nARI', '3-NN\nAcc']
    vals_6d = [
        results_6d.get('hdbscan', {}).get('ari', 0),
        results_6d['kmeans']['ari'],
        results_6d['knn_3']['mean_accuracy'],
    ]
    vals_10d = [
        results_10d.get('hdbscan', {}).get('ari', 0),
        results_10d['kmeans']['ari'],
        results_10d['knn_3']['mean_accuracy'],
    ]
    x_pos = np.arange(len(metrics))
    w = 0.35
    ax.bar(x_pos - w/2, vals_6d, w, label='6D spatial', color='#90CAF9',
           edgecolor='black', linewidth=0.5)
    ax.bar(x_pos + w/2, vals_10d, w, label='10D spatial+temporal',
           color='#66BB6A', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.set_title('6D vs 10D Comparison')
    ax.legend(fontsize=9)
    ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='ARI=0.5')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels
    for i, (v6, v10) in enumerate(zip(vals_6d, vals_10d)):
        ax.text(i - w/2, v6 + 0.02, f'{v6:.3f}', ha='center', fontsize=8)
        ax.text(i + w/2, v10 + 0.02, f'{v10:.3f}', ha='center', fontsize=8)

    plt.suptitle('Experiment 63: Temporal Features Break EW-KPZ Degeneracy\n'
                 '10D (6 spatial + 4 temporal) vs 6D (spatial only)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path1 = save_dir / 'exp63_main_comparison.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Figure 1 saved: {path1}")
    plt.close()
    import shutil
    shutil.copy2(path1, FIGURES_DIR / 'exp63_main_comparison.png')

    # ========== Figure 2: Temporal feature distributions ==========
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (ax, fname) in enumerate(zip(axes2.flat, TEMPORAL_NAMES)):
        for sys_name in unique_systems:
            mask = np.array([l == sys_name for l in true_labels])
            vals = temporal_feats[mask, idx]
            ax.hist(vals, bins=20, alpha=0.5, label=sys_name,
                    color=sys_colors.get(sys_name, '#999'), density=True,
                    edgecolor='black', linewidth=0.3)
        ax.set_xlabel(fname, fontsize=11)
        ax.set_ylabel('Density')
        ax.set_title(f'{fname} distribution by system')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    plt.suptitle('Temporal Feature Distributions\n'
                 'Key: vel_skew and slope_growth should separate EW from KPZ',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path2 = save_dir / 'exp63_temporal_distributions.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Figure 2 saved: {path2}")
    plt.close()
    shutil.copy2(path2, FIGURES_DIR / 'exp63_temporal_distributions.png')

    # ========== Figure 3: Key feature pairs (new features) ==========
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    pairs = [
        (0, 7, 'grad_var', 'vel_skew'),          # spatial amplitude vs dynamics
        (0, 9, 'grad_var', 'slope_growth'),       # spatial vs nonlinear coupling
        (6, 7, 'beta_eff', 'vel_skew'),           # growth rate vs nonlinearity
        (6, 9, 'beta_eff', 'slope_growth'),       # growth vs coupling
        (7, 9, 'vel_skew', 'slope_growth'),       # two nonlinearity probes
        (7, 8, 'vel_skew', 'vel_kurt'),           # velocity moments
    ]

    for ax, (fi, fj, fname_i, fname_j) in zip(axes3.flat, pairs):
        for sys_name in unique_systems:
            mask = np.array([l == sys_name for l in true_labels])
            ax.scatter(combined_feats[mask, fi], combined_feats[mask, fj],
                       c=sys_colors.get(sys_name, '#999'), label=sys_name,
                       alpha=0.6, s=15, edgecolors='none')
        ax.set_xlabel(fname_i, fontsize=9)
        ax.set_ylabel(fname_j, fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.2)

    plt.suptitle('Feature Pair Plots (spatial + temporal, colored by system)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path3 = save_dir / 'exp63_feature_pairs.png'
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"  Figure 3 saved: {path3}")
    plt.close()
    shutil.copy2(path3, FIGURES_DIR / 'exp63_feature_pairs.png')


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(pilot=False):
    t0 = time.time()
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 63: TEMPORAL GROWTH FEATURES")
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

    systems = ['ew', 'kpz', 'bd', 'eden', 'rd', 'ks']
    class_map = {
        'ew': 'EW', 'kpz': 'KPZ', 'bd': 'KPZ', 'eden': 'KPZ',
        'rd': 'trivial', 'ks': 'KS'
    }

    config = {'L': L, 'T': T, 'N': N, 'systems': systems,
              'mode': 'pilot' if pilot else 'full',
              'spatial_features': SPATIAL_NAMES,
              'temporal_features': TEMPORAL_NAMES}
    print(f"Config: L={L}, T={T}, N={N} per system ({N*len(systems)} total)")
    print(f"Systems: {systems}")
    print(f"Classes: {sorted(set(class_map.values()))}")
    print(f"Features: {len(ALL_FEATURE_NAMES)}D "
          f"(6 spatial + 4 temporal)")
    print()

    # ---- Step 1: Generate features ----
    print("STEP 1: Generating features (spatial + temporal)")
    print("-" * 50)

    all_spatial = []
    all_temporal = []
    all_labels = []

    for sys_name in systems:
        t1 = time.time()
        print(f"  {sys_name:6s}: generating {N} samples...", end='', flush=True)

        sim_kwargs = {}
        if sys_name == 'ks':
            sim_kwargs['record_interval'] = 10 if pilot else 20

        s_feats, t_feats, params = generate_feature_dataset(
            N, L, T, sys_name,
            seed_offset=hash(sys_name) % 10000,
            late_frac=0.3,
            **sim_kwargs
        )

        n_valid = s_feats.shape[0]
        all_spatial.append(s_feats)
        all_temporal.append(t_feats)
        all_labels.extend([sys_name] * n_valid)

        if n_valid < N:
            print(f" ({N - n_valid} failed)", end='')

        if n_valid > 0:
            ms = s_feats.mean(axis=0)
            mt = t_feats.mean(axis=0)
            dt_elapsed = time.time() - t1
            print(f" done ({dt_elapsed:.1f}s)  "
                  f"grad_var={ms[0]:.4f}  "
                  f"beta={mt[0]:.3f}  "
                  f"vel_skew={mt[1]:+.4f}  "
                  f"slope_growth={mt[3]:+.6f}")
        else:
            print(f" done ({time.time()-t1:.1f}s)  ALL FAILED")

    spatial_features = np.vstack(all_spatial)
    temporal_features = np.vstack(all_temporal)
    combined_features = np.hstack([spatial_features, temporal_features])

    print(f"\n  Total: {combined_features.shape[0]} samples x "
          f"{combined_features.shape[1]} features "
          f"(6 spatial + 4 temporal)")

    # ---- Step 2: Baseline clustering (6D spatial only) ----
    print("\nSTEP 2: Clustering — 6D BASELINE (spatial only)")
    print("-" * 50)

    true_classes = [class_map[l] for l in all_labels]
    results_6d, X6, hdb6, km6 = run_clustering_analysis(
        spatial_features, all_labels, class_map, label="6D"
    )
    results_6d['config'] = config

    # ---- Step 3: Enhanced clustering (10D spatial+temporal) ----
    print("\nSTEP 3: Clustering — 10D ENHANCED (spatial + temporal)")
    print("-" * 50)

    results_10d, X10, hdb10, km10 = run_clustering_analysis(
        combined_features, all_labels, class_map, label="10D"
    )
    results_10d['config'] = config

    # ---- Step 4: Comparison ----
    print("\nSTEP 4: Comparison (6D vs 10D)")
    print("-" * 50)

    hdb_ari_6 = results_6d.get('hdbscan', {}).get('ari', 0)
    hdb_ari_10 = results_10d.get('hdbscan', {}).get('ari', 0)
    km_ari_6 = results_6d['kmeans']['ari']
    km_ari_10 = results_10d['kmeans']['ari']
    knn3_6 = results_6d['knn_3']['mean_accuracy']
    knn3_10 = results_10d['knn_3']['mean_accuracy']
    ew_kpz_6 = results_6d['centroid_distances'].get('EW_vs_KPZ', 0)
    ew_kpz_10 = results_10d['centroid_distances'].get('EW_vs_KPZ', 0)

    print(f"  {'Metric':<22s} {'6D':>8s} {'10D':>8s} {'Change':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'HDBSCAN ARI':<22s} {hdb_ari_6:8.3f} {hdb_ari_10:8.3f} "
          f"{hdb_ari_10 - hdb_ari_6:+10.3f}")
    print(f"  {'KMeans ARI':<22s} {km_ari_6:8.3f} {km_ari_10:8.3f} "
          f"{km_ari_10 - km_ari_6:+10.3f}")
    print(f"  {'3-NN accuracy':<22s} {knn3_6:8.3f} {knn3_10:8.3f} "
          f"{knn3_10 - knn3_6:+10.3f}")
    print(f"  {'EW-KPZ distance':<22s} {ew_kpz_6:8.3f} {ew_kpz_10:8.3f} "
          f"{ew_kpz_10 - ew_kpz_6:+10.3f}")

    # ---- Step 5: Check predictions ----
    print("\nSTEP 5: Checking predictions")
    print("-" * 50)

    best_ari_10 = max(hdb_ari_10, km_ari_10)
    p1_pass = best_ari_10 > 0.5
    p2_pass = True  # checked below
    p3_pass = True  # checked below
    p4_pass = knn3_10 > 0.85
    p5_pass = ew_kpz_10 > 1.5
    p6_pass = True  # checked below
    p7_pass = True  # checked below

    # P2: vel_skew separates EW from KPZ
    ew_mask = np.array([l == 'ew' for l in all_labels])
    kpz_mask = np.array([l == 'kpz' for l in all_labels])
    ew_vel_skew = temporal_features[ew_mask, 1].mean()
    kpz_vel_skew = temporal_features[kpz_mask, 1].mean()
    p2_pass = abs(kpz_vel_skew - ew_vel_skew) > 0.1

    # P3: beta separates EW from KPZ from RD
    rd_mask = np.array([l == 'rd' for l in all_labels])
    ew_beta = temporal_features[ew_mask, 0].mean()
    kpz_beta = temporal_features[kpz_mask, 0].mean()
    rd_beta = temporal_features[rd_mask, 0].mean()
    p3_pass = (kpz_beta > ew_beta + 0.02) and (rd_beta > kpz_beta + 0.02)

    # P6: slope_growth < 0 for KS
    ks_mask = np.array([l == 'ks' for l in all_labels])
    ks_sg = temporal_features[ks_mask, 3].mean()
    p6_pass = ks_sg < 0

    # P7: slope_growth ~ 0 for EW
    ew_sg = temporal_features[ew_mask, 3].mean()
    kpz_sg = temporal_features[kpz_mask, 3].mean()
    p7_pass = abs(ew_sg) < abs(kpz_sg) * 0.3  # EW much closer to 0 than KPZ

    print(f"  P1 (ARI > 0.5):           {'PASS' if p1_pass else 'FAIL'}  "
          f"(best ARI={best_ari_10:.3f})")
    print(f"  P2 (vel_skew EW!=KPZ):    {'PASS' if p2_pass else 'FAIL'}  "
          f"(EW={ew_vel_skew:+.4f}, KPZ={kpz_vel_skew:+.4f})")
    print(f"  P3 (beta ordering):       {'PASS' if p3_pass else 'FAIL'}  "
          f"(EW={ew_beta:.3f}, KPZ={kpz_beta:.3f}, RD={rd_beta:.3f})")
    print(f"  P4 (kNN > 85%):           {'PASS' if p4_pass else 'FAIL'}  "
          f"(3-NN={knn3_10:.3f})")
    print(f"  P5 (EW-KPZ dist > 1.5):   {'PASS' if p5_pass else 'FAIL'}  "
          f"(d={ew_kpz_10:.3f})")
    print(f"  P6 (KS slope_growth < 0): {'PASS' if p6_pass else 'FAIL'}  "
          f"(KS={ks_sg:+.6f})")
    print(f"  P7 (EW slope_growth ~ 0): {'PASS' if p7_pass else 'FAIL'}  "
          f"(EW={ew_sg:+.6f}, KPZ={kpz_sg:+.6f})")

    preds = [p1_pass, p2_pass, p3_pass, p4_pass, p5_pass, p6_pass, p7_pass]
    n_pass = sum(preds)
    print(f"\n  Score: {n_pass}/7 predictions passed")

    results_10d['predictions'] = {
        'P1_ari_above_05': bool(p1_pass),
        'P2_vel_skew_separates': bool(p2_pass),
        'P3_beta_ordering': bool(p3_pass),
        'P4_knn_above_85': bool(p4_pass),
        'P5_ew_kpz_dist_above_15': bool(p5_pass),
        'P6_ks_negative_sg': bool(p6_pass),
        'P7_ew_zero_sg': bool(p7_pass),
        'n_pass': int(n_pass),
    }

    # ---- Step 6: Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Comparison summary
    comparison = {
        'metric': ['hdbscan_ari', 'kmeans_ari', 'knn_3', 'ew_kpz_dist'],
        '6D': [hdb_ari_6, km_ari_6, knn3_6, ew_kpz_6],
        '10D': [hdb_ari_10, km_ari_10, knn3_10, ew_kpz_10],
        'delta': [hdb_ari_10 - hdb_ari_6, km_ari_10 - km_ari_6,
                  knn3_10 - knn3_6, ew_kpz_10 - ew_kpz_6],
    }

    # Strip non-serializable items for JSON
    def clean_for_json(d):
        out = {}
        for k, v in d.items():
            if k in ('hdbscan', 'kmeans') and isinstance(v, dict) and 'labels' in v:
                v_copy = dict(v)
                del v_copy['labels']
                out[k] = v_copy
            else:
                out[k] = v
        return out

    save_data = {
        'comparison': comparison,
        'results_6d': clean_for_json(results_6d),
        'results_10d': clean_for_json(results_10d),
        'temporal_feature_means': {
            sys_name: temporal_features[
                np.array([l == sys_name for l in all_labels])
            ].mean(axis=0).tolist()
            for sys_name in systems
        },
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t0,
    }

    rpath = RESULTS_DIR / 'results.json'
    with open(rpath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {rpath}")

    # Save raw features + labels
    np.savez(RESULTS_DIR / 'features.npz',
             spatial=spatial_features,
             temporal=temporal_features,
             combined=combined_features,
             labels=all_labels,
             classes=true_classes,
             spatial_names=SPATIAL_NAMES,
             temporal_names=TEMPORAL_NAMES,
             all_names=ALL_FEATURE_NAMES)
    print(f"  Features saved: {RESULTS_DIR / 'features.npz'}")

    # ---- Step 7: Figures ----
    print("\nSTEP 6: Generating figures")
    print("-" * 50)

    make_figures(spatial_features, temporal_features, combined_features,
                 X6, X10,
                 all_labels, true_classes, class_map,
                 hdb6, hdb10, km6, km10,
                 results_6d, results_10d, RESULTS_DIR)

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Samples: {combined_features.shape[0]}  "
          f"Features: {combined_features.shape[1]}D "
          f"(6 spatial + 4 temporal)")
    print()
    print(f"  {'Metric':<22s} {'6D':>8s} {'10D':>8s} {'Change':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'HDBSCAN ARI':<22s} {hdb_ari_6:8.3f} {hdb_ari_10:8.3f} "
          f"{hdb_ari_10 - hdb_ari_6:+10.3f}")
    print(f"  {'KMeans ARI':<22s} {km_ari_6:8.3f} {km_ari_10:8.3f} "
          f"{km_ari_10 - km_ari_6:+10.3f}")
    print(f"  {'3-NN accuracy':<22s} {knn3_6:8.3f} {knn3_10:8.3f} "
          f"{knn3_10 - knn3_6:+10.3f}")
    print(f"  {'EW-KPZ distance':<22s} {ew_kpz_6:8.3f} {ew_kpz_10:8.3f} "
          f"{ew_kpz_10 - ew_kpz_6:+10.3f}")
    print()
    print(f"  Predictions: {n_pass}/7 passed")
    print()

    if hdb_ari_10 > 0.5 and knn3_10 > 0.85:
        print("  RESULT: Temporal features BREAK the Exp 62 ceiling.")
        print("  The EW-KPZ degeneracy is resolved by dynamics-sensitive")
        print("  observables (velocity skewness, slope-growth coupling).")
    elif knn3_10 > 0.9 and km_ari_10 > km_ari_6 + 0.1:
        print("  RESULT: Temporal features add discriminative signal.")
        print(f"  kNN {knn3_6:.1%} -> {knn3_10:.1%}, KMeans ARI "
              f"{km_ari_6:.3f} -> {km_ari_10:.3f}.")
        print("  HDBSCAN unchanged — density estimation doesn't benefit")
        print("  from extra dimensions (curse of dimensionality).")
        print("  kNN suggests local class information is present; HDBSCAN")
        print("  does not convert it into stable density clusters here.")
    elif hdb_ari_10 > hdb_ari_6 + 0.05:
        print("  RESULT: Temporal features IMPROVE clustering, but don't")
        print("  fully break the ceiling. Marginal improvement.")
    elif hdb_ari_10 > hdb_ari_6:
        print("  RESULT: Slight improvement. Temporal features add some")
        print("  discriminative power but the ceiling is structural.")
    else:
        print("  RESULT: No improvement in HDBSCAN, but check kNN and")
        print("  KMeans for supervised/semi-supervised improvement.")

    print("=" * 70)
    return save_data


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exp 63: Temporal Growth Features for Universality Clustering')
    parser.add_argument('--pilot', action='store_true',
                        help='Quick test (~5 min)')
    args = parser.parse_args()

    results = run_experiment(pilot=args.pilot)
