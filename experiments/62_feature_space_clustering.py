"""
Experiment 62: Feature-Space Anomaly Clustering
=================================================

PURPOSE: The honest synthesis of 61 experiments. Instead of training an
autoencoder to discover features (which fails — Exp 61), use the features
that prior experiments suggest carry useful universality-related signal:

  grad_var, grad_skew, grad_kurt, lap_var, grad_lap_cov, h_var

Then run unsupervised clustering (HDBSCAN, KMeans, UMAP) directly on those
features across 6+ systems. This tests the core discovery question:

  "Do unknown systems cluster by universality class without labels?"

If supervised classification with similar features performs well
(ml-universality-classification project), there may be unsupervised structure.
The question is whether HDBSCAN/UMAP can find it without knowing the labels.

SYSTEMS:
  Continuous PDE:    EW (Edwards-Wilkinson), KPZ equation, KS (Kuramoto-Sivashinsky)
  Discrete growth:   BD (Ballistic Deposition), Eden model, RD (Random Deposition)

  Expected classes:
    EW class:       EW
    KPZ class:      KPZ, BD, Eden
    KS class:       KS
    Trivial class:  RD

FEATURES (per snapshot, 6D):
  grad_var      = Var[dh/dx]         -- amplitude of fluctuations
  grad_skew     = Skew[dh/dx]        -- asymmetry (0 for EW, nonzero for KPZ)
  grad_kurt     = Kurt[dh/dx] - 3    -- tail weight
  lap_var       = Var[d^2h/dx^2]     -- roughness at short scales
  grad_lap_cov  = Cov[|dh/dx|, d^2h/dx^2]  -- nonlinear coupling signature
  h_var         = Var[h]             -- overall roughness / width

PREDICTIONS:
  P1: UMAP shows >= 3 visually distinct clusters
  P2: KPZ, BD, Eden overlap in feature space (same universality class)
  P3: KS separates clearly from KPZ cluster
  P4: RD separates from all others (no spatial correlations)
  P5: HDBSCAN ARI > 0.5 against true universality classes
  P6: EW separates from KPZ (different exponents, zero skewness)

SUCCESS CRITERIA:
  - ARI > 0.5 (HDBSCAN or KMeans)
  - kNN accuracy > 0.7
  - UMAP shows visually separable clusters

Usage:
  python 62_feature_space_clustering.py --pilot    # Quick (~3 min)
  python 62_feature_space_clustering.py            # Full (~20 min)
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

RESULTS_DIR = PROJECT_DIR / 'results_exp62'
FIGURES_DIR = PROJECT_DIR / 'figures'


# ============================================================================
# Simulators (reused from Exp 61, all 1D periodic)
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
# Feature extraction (from Exp 60, validated)
# ============================================================================

@jit(nopython=True)
def compute_features_single(h):
    """Compute 6D feature vector from a single 1D height profile.

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


FEATURE_NAMES = ['grad_var', 'grad_skew', 'grad_kurt',
                 'lap_var', 'grad_lap_cov', 'h_var']


def extract_features_from_trajectory(trajectory, late_frac=0.3):
    """Extract features from the late-time portion of a trajectory.

    Args:
        trajectory: (T, L) array
        late_frac: fraction of trajectory to use (from the end)

    Returns:
        (n_features,) array — mean over late-time snapshots
    """
    T = trajectory.shape[0]
    start = int(T * (1 - late_frac))
    features = []
    for t in range(start, T):
        features.append(compute_features_single(trajectory[t]))
    return np.mean(features, axis=0)


# ============================================================================
# Dataset generation
# ============================================================================

def generate_feature_dataset(n_samples, L, T, system, seed_offset=0,
                             late_frac=0.3, **sim_kwargs):
    """Generate features for n_samples from a system.

    Returns:
        features: (n_samples, 6) array
        params_used: list of dicts (for record keeping)
    """
    features = []
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
                # KPZ blew up — retry with milder parameters
                lam = np.random.uniform(0.3, 1.5)
                traj = simulate_kpz(L=L, T=T, nu=nu, lam=lam, D=D, seed=seed+10000)
            if traj is None:
                # Still failed — use safe defaults
                traj = simulate_kpz(L=L, T=T, nu=1.0, lam=0.5, D=1.0, seed=seed+20000)
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
            ks_T = min(T, 500)  # KS reaches steady state fast, cap to avoid blowup
            traj = simulate_ks(L=L, T=ks_T, seed=seed, record_interval=ri)
            if traj is None:
                # Retry with smaller noise and fewer steps
                traj = simulate_ks(L=L, T=ks_T, seed=seed+10000,
                                   record_interval=ri, noise=0.02)
            if traj is None:
                # Last resort: very conservative
                traj = simulate_ks(L=L, T=300, seed=seed+20000,
                                   record_interval=5, noise=0.01)
            params_used.append({'record_interval': ri, 'T_actual': ks_T})

        else:
            raise ValueError(f"Unknown system: {system}")

        feat = extract_features_from_trajectory(traj, late_frac=late_frac)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            continue  # skip failed samples
        features.append(feat)

    if len(features) == 0:
        return np.zeros((0, 6)), params_used
    return np.array(features), params_used


# ============================================================================
# Analysis
# ============================================================================

def run_clustering_analysis(features, true_labels, class_map):
    """Run full clustering analysis.

    Args:
        features: (N, 6) array
        true_labels: list of system names (e.g. 'ew', 'kpz', ...)
        class_map: dict mapping system -> universality class

    Returns:
        results dict
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans

    N = len(features)
    true_classes = [class_map[l] for l in true_labels]
    unique_classes = sorted(set(true_classes))
    n_true_classes = len(unique_classes)

    # Standardize features (z-score)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    results = {
        'n_samples': N,
        'n_true_classes': n_true_classes,
        'true_class_counts': {c: true_classes.count(c) for c in unique_classes},
    }

    # --- Feature statistics per system ---
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
        print(f"  HDBSCAN: {n_hdb_clusters} clusters, {n_noise} noise, "
              f"ARI={hdb_ari:.3f}, NMI={hdb_nmi:.3f}")
    except ImportError:
        print("  HDBSCAN not available")
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
    print(f"  KMeans(K={n_true_classes}): ARI={km_ari:.3f}, NMI={km_nmi:.3f}")

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
    print(f"  KMeans best K={best_k} (silhouette={km_sweep[best_k]['silhouette']:.3f}, "
          f"ARI={km_sweep[best_k]['ari']:.3f})")

    # --- kNN classification (cross-validated) ---
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, true_classes, cv=5, scoring='accuracy')
        results[f'knn_{k}'] = {
            'mean_accuracy': float(np.mean(scores)),
            'std_accuracy': float(np.std(scores)),
        }
        print(f"  {k}-NN CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    # --- Pairwise distances between class centroids ---
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
    print(f"  Centroid distances: {centroid_dists}")

    return results, X, hdb_labels, km_labels


# ============================================================================
# Visualization
# ============================================================================

def make_figures(features, X_scaled, true_labels, true_classes, class_map,
                 hdb_labels, km_labels, results, save_dir):
    """Generate comprehensive analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Colors per system
    sys_colors = {
        'ew': '#2196F3', 'kpz': '#4CAF50', 'bd': '#8BC34A',
        'eden': '#CDDC39', 'rd': '#FF9800', 'ks': '#F44336'
    }
    # Colors per universality class
    class_colors = {
        'EW': '#2196F3', 'KPZ': '#4CAF50',
        'trivial': '#FF9800', 'KS': '#F44336'
    }

    unique_systems = sorted(set(true_labels))

    # ========== Figure 1: Main 6-panel analysis ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # --- Panel 1: Feature means per system (bar chart) ---
    ax = axes[0, 0]
    n_feat = len(FEATURE_NAMES)
    x_pos = np.arange(n_feat)
    bar_width = 0.12
    for i, sys_name in enumerate(unique_systems):
        means = results['per_system'][sys_name]['feature_means']
        # Use log-scale-safe: plot absolute values
        ax.bar(x_pos + i * bar_width, means, bar_width,
               label=sys_name, color=sys_colors.get(sys_name, '#999'),
               alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x_pos + bar_width * len(unique_systems) / 2)
    ax.set_xticklabels(FEATURE_NAMES, fontsize=8, rotation=30)
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Means by System')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: PCA of feature space ---
    ax = axes[0, 1]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X_scaled)

    for sys_name in unique_systems:
        mask = np.array([l == sys_name for l in true_labels])
        ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1],
                   c=sys_colors.get(sys_name, '#999'), label=sys_name,
                   alpha=0.7, s=25, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Feature Space (PCA)')
    ax.legend(fontsize=7)

    # --- Panel 3: UMAP of feature space ---
    ax = axes[0, 2]
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42)
        Z_umap = reducer.fit_transform(X_scaled)
        for sys_name in unique_systems:
            mask = np.array([l == sys_name for l in true_labels])
            ax.scatter(Z_umap[mask, 0], Z_umap[mask, 1],
                       c=sys_colors.get(sys_name, '#999'), label=sys_name,
                       alpha=0.7, s=25, edgecolors='none')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Feature Space (UMAP)')
        ax.legend(fontsize=7)
        umap_available = True
    except ImportError:
        ax.text(0.5, 0.5, 'umap-learn not installed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('UMAP (unavailable)')
        umap_available = False

    # --- Panel 4: HDBSCAN clusters (PCA view) ---
    ax = axes[1, 0]
    if hdb_labels is not None:
        cmap = plt.cm.tab10
        unique_cl = sorted(set(hdb_labels))
        for cl in unique_cl:
            mask = hdb_labels == cl
            color = 'lightgray' if cl == -1 else cmap(cl % 10)
            label = 'Noise' if cl == -1 else f'Cluster {cl}'
            ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1],
                       c=[color], label=label, alpha=0.7, s=25)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        hdb_ari = results.get('hdbscan', {}).get('ari', 0)
        ax.set_title(f'HDBSCAN Clusters (ARI={hdb_ari:.3f})')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'HDBSCAN not available',
                ha='center', va='center', transform=ax.transAxes)

    # --- Panel 5: KMeans clusters (PCA view) ---
    ax = axes[1, 1]
    cmap = plt.cm.tab10
    for cl in sorted(set(km_labels)):
        mask = km_labels == cl
        ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1],
                   c=[cmap(cl % 10)], label=f'Cluster {cl}', alpha=0.7, s=25)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    km_ari = results.get('kmeans', {}).get('ari', 0)
    ax.set_title(f'KMeans (K={results["n_true_classes"]}) ARI={km_ari:.3f}')
    ax.legend(fontsize=7)

    # --- Panel 6: Confusion matrix (KMeans vs true class) ---
    ax = axes[1, 2]
    unique_true = sorted(set(true_classes))
    unique_km = sorted(set(km_labels))
    confusion = np.zeros((len(unique_true), len(unique_km)))
    for tc, kl in zip(true_classes, km_labels):
        confusion[unique_true.index(tc), unique_km.index(kl)] += 1
    im = ax.imshow(confusion, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(unique_km)))
    ax.set_xticklabels([str(c) for c in unique_km], fontsize=9)
    ax.set_yticks(range(len(unique_true)))
    ax.set_yticklabels(unique_true, fontsize=9)
    ax.set_xlabel('KMeans Cluster')
    ax.set_ylabel('True Universality Class')
    ax.set_title(f'KMeans Confusion (ARI={km_ari:.3f})')
    for i in range(len(unique_true)):
        for j in range(len(unique_km)):
            val = int(confusion[i, j])
            color = 'white' if val > confusion.max()/2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    plt.suptitle('Experiment 62: Feature-Space Anomaly Clustering\n'
                 '6D gradient/spectral features across 6 surface growth systems',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    path1 = save_dir / 'exp62_feature_clustering.png'
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Figure 1 saved: {path1}")
    plt.close()

    # Copy to figures dir
    import shutil
    shutil.copy2(path1, FIGURES_DIR / 'exp62_feature_clustering.png')

    # ========== Figure 2: Feature pair plots (key pairs) ==========
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
    pairs = [
        (0, 1, 'grad_var', 'grad_skew'),       # amplitude vs asymmetry
        (0, 2, 'grad_var', 'grad_kurt'),        # amplitude vs tails
        (0, 5, 'grad_var', 'h_var'),            # gradient vs total roughness
        (1, 2, 'grad_skew', 'grad_kurt'),       # higher moments
        (0, 3, 'grad_var', 'lap_var'),          # gradient vs laplacian
        (0, 4, 'grad_var', 'grad_lap_cov'),     # nonlinear coupling
    ]

    for ax, (fi, fj, fname_i, fname_j) in zip(axes2.flat, pairs):
        for sys_name in unique_systems:
            mask = np.array([l == sys_name for l in true_labels])
            ax.scatter(features[mask, fi], features[mask, fj],
                       c=sys_colors.get(sys_name, '#999'), label=sys_name,
                       alpha=0.6, s=15, edgecolors='none')
        ax.set_xlabel(fname_i, fontsize=9)
        ax.set_ylabel(fname_j, fontsize=9)
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.2)

    plt.suptitle('Feature Pair Plots (raw values, colored by system)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path2 = save_dir / 'exp62_feature_pairs.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Figure 2 saved: {path2}")
    plt.close()
    shutil.copy2(path2, FIGURES_DIR / 'exp62_feature_pairs.png')

    # ========== Figure 3: KMeans silhouette sweep ==========
    if 'kmeans_sweep' in results:
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
        ks = sorted(results['kmeans_sweep'].keys())
        sils = [results['kmeans_sweep'][k]['silhouette'] for k in ks]
        aris = [results['kmeans_sweep'][k]['ari'] for k in ks]
        ax3.plot(ks, sils, 'o-', color='#2196F3', label='Silhouette')
        ax3.set_xlabel('K (number of clusters)')
        ax3.set_ylabel('Silhouette Score', color='#2196F3')
        ax3.tick_params(axis='y', labelcolor='#2196F3')

        ax3b = ax3.twinx()
        ax3b.plot(ks, aris, 's--', color='#F44336', label='ARI')
        ax3b.set_ylabel('ARI vs true classes', color='#F44336')
        ax3b.tick_params(axis='y', labelcolor='#F44336')

        ax3.axvline(results['n_true_classes'], color='gray', linestyle=':',
                     label=f'True K={results["n_true_classes"]}')
        ax3.set_title('KMeans: Optimal K Search')
        ax3.legend(loc='upper left', fontsize=9)
        ax3b.legend(loc='upper right', fontsize=9)
        ax3.grid(alpha=0.3)

        path3 = save_dir / 'exp62_kmeans_sweep.png'
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close()
        shutil.copy2(path3, FIGURES_DIR / 'exp62_kmeans_sweep.png')
        print(f"  Figure 3 saved: {path3}")


# ============================================================================
# Main experiment
# ============================================================================

def run_experiment(pilot=False):
    t0 = time.time()
    np.random.seed(42)

    print("=" * 70)
    print("EXPERIMENT 62: FEATURE-SPACE ANOMALY CLUSTERING")
    print("=" * 70)
    print(f"Mode: {'PILOT' if pilot else 'FULL'}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # ---- Configuration ----
    if pilot:
        L = 128
        T = 500
        N = 30        # samples per system
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
              'mode': 'pilot' if pilot else 'full'}
    print(f"Config: L={L}, T={T}, N={N} per system ({N*len(systems)} total)")
    print(f"Systems: {systems}")
    print(f"Classes: {sorted(set(class_map.values()))}")
    print()

    # ---- Step 1: Generate features ----
    print("STEP 1: Generating features")
    print("-" * 50)

    all_features = []
    all_labels = []

    for sys_name in systems:
        t1 = time.time()
        print(f"  {sys_name:6s}: generating {N} samples...", end='', flush=True)

        sim_kwargs = {}
        if sys_name == 'ks':
            sim_kwargs['record_interval'] = 10 if pilot else 20

        feats, params = generate_feature_dataset(
            N, L, T, sys_name,
            seed_offset=hash(sys_name) % 10000,
            late_frac=0.3,
            **sim_kwargs
        )

        all_features.append(feats)
        n_valid = feats.shape[0]
        all_labels.extend([sys_name] * n_valid)

        if n_valid < N:
            print(f" ({N - n_valid} failed)", end='')

        mean_feat = feats.mean(axis=0) if n_valid > 0 else np.zeros(6)
        dt = time.time() - t1
        print(f" done ({dt:.1f}s)  "
              f"grad_var={mean_feat[0]:.4f}  skew={mean_feat[1]:+.3f}  "
              f"kurt={mean_feat[2]:+.3f}  h_var={mean_feat[5]:.4f}")

    features = np.vstack(all_features)
    print(f"\n  Total: {features.shape[0]} samples x {features.shape[1]} features")

    # ---- Step 2: Clustering analysis ----
    print("\nSTEP 2: Clustering analysis")
    print("-" * 50)

    true_classes = [class_map[l] for l in all_labels]
    results, X_scaled, hdb_labels, km_labels = run_clustering_analysis(
        features, all_labels, class_map
    )
    results['config'] = config

    # ---- Step 3: Check predictions ----
    print("\nSTEP 3: Checking predictions")
    print("-" * 50)

    hdb_n = results.get('hdbscan', {}).get('n_clusters', 0)
    hdb_ari = results.get('hdbscan', {}).get('ari', 0)
    km_ari = results['kmeans']['ari']
    knn3_acc = results['knn_3']['mean_accuracy']

    p1_pass = True  # We'll check UMAP visually
    p2_kpz_overlap = True  # Check: BD, Eden centroids close to KPZ
    bd_kpz_dist = results['centroid_distances'].get('KPZ_vs_KPZ', 0)  # same class

    p3_ks_sep = results['centroid_distances'].get('KPZ_vs_KS', 0) > 1.0
    p4_rd_sep = results['centroid_distances'].get('KPZ_vs_trivial', 0) > 1.0
    p5_ari = max(hdb_ari, km_ari) > 0.5
    p6_ew_sep = results['centroid_distances'].get('EW_vs_KPZ', 0) > 0.5

    print(f"  P1 (UMAP shows clusters):     [check figure]")
    print(f"  P3 (KS separates from KPZ):   {'PASS' if p3_ks_sep else 'FAIL'}  "
          f"(d={results['centroid_distances'].get('KPZ_vs_KS', 0):.2f})")
    print(f"  P4 (RD separates):             {'PASS' if p4_rd_sep else 'FAIL'}  "
          f"(d={results['centroid_distances'].get('KPZ_vs_trivial', 0):.2f})")
    print(f"  P5 (ARI > 0.5):               {'PASS' if p5_ari else 'FAIL'}  "
          f"(HDBSCAN={hdb_ari:.3f}, KMeans={km_ari:.3f})")
    print(f"  P6 (EW separates from KPZ):   {'PASS' if p6_ew_sep else 'FAIL'}  "
          f"(d={results['centroid_distances'].get('EW_vs_KPZ', 0):.2f})")
    print(f"  kNN-3 accuracy:               {knn3_acc:.3f}")

    n_pass = sum([p3_ks_sep, p4_rd_sep, p5_ari, p6_ew_sep, knn3_acc > 0.7])
    print(f"\n  Score: {n_pass}/5 quantitative predictions passed")

    results['predictions'] = {
        'P3_ks_separates': bool(p3_ks_sep),
        'P4_rd_separates': bool(p4_rd_sep),
        'P5_ari_above_05': bool(p5_ari),
        'P6_ew_separates': bool(p6_ew_sep),
        'knn3_above_07': bool(knn3_acc > 0.7),
        'n_pass': int(n_pass),
    }

    # ---- Step 4: Save results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results['timestamp'] = datetime.now().isoformat()
    results['elapsed_seconds'] = time.time() - t0

    # Remove non-serializable items
    results_save = {}
    for k, v in results.items():
        if k in ('hdbscan',) and 'labels' in v:
            v_copy = dict(v)
            del v_copy['labels']
            results_save[k] = v_copy
        elif k == 'kmeans' and 'labels' in v:
            v_copy = dict(v)
            del v_copy['labels']
            results_save[k] = v_copy
        else:
            results_save[k] = v

    rpath = RESULTS_DIR / 'results.json'
    with open(rpath, 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"\n  Results saved: {rpath}")

    # Save raw features + labels for follow-up
    np.savez(RESULTS_DIR / 'features.npz',
             features=features, labels=all_labels,
             classes=true_classes, feature_names=FEATURE_NAMES)
    print(f"  Features saved: {RESULTS_DIR / 'features.npz'}")

    # ---- Step 5: Figures ----
    print("\nSTEP 4: Generating figures")
    print("-" * 50)

    make_figures(features, X_scaled, all_labels, true_classes, class_map,
                 hdb_labels, km_labels, results, RESULTS_DIR)

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Samples: {features.shape[0]}  Features: {features.shape[1]}")
    print(f"  HDBSCAN: {hdb_n} clusters, ARI={hdb_ari:.3f}")
    print(f"  KMeans(K={results['n_true_classes']}): ARI={km_ari:.3f}")
    print(f"  Best K={results['kmeans_best_k']} "
          f"(sil={results['kmeans_sweep'][results['kmeans_best_k']]['silhouette']:.3f})")
    print(f"  3-NN accuracy: {knn3_acc:.3f}")
    print(f"  Predictions: {n_pass}/5 passed")
    print()

    if n_pass >= 4:
        print("  RESULT: Feature-space clustering shows useful structure.")
        print("  Physics features carry signal, but inspect class-level")
        print("  errors before interpreting this as discovery.")
    elif n_pass >= 2:
        print("  RESULT: Partial success. Some classes separate, others overlap.")
        print("  Check which specific separations work/fail.")
    else:
        print("  RESULT: Feature-space clustering fails.")
        print("  The 6D gradient features may not be sufficient for all classes.")

    print("=" * 70)
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exp 62: Feature-Space Clustering')
    parser.add_argument('--pilot', action='store_true', help='Quick test (~3 min)')
    args = parser.parse_args()

    results = run_experiment(pilot=args.pilot)
