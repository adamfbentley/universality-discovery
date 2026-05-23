"""
Experiment 67: Tracy-Widom Height-Distribution Features
=======================================================

Decisive test #2 from the manuscript limitations: the gradient and temporal
feature families omit the one-point height distribution, which is the observable
KPZ theory most directly favours for separating EW from KPZ. This experiment adds
it and asks two questions:

  (A) Physics validation. In the GROWTH regime, does the standardized one-point
      height distribution skew away from Gaussian for KPZ-class growth
      (kpz/bd/eden), toward the Tracy-Widom family, while EW stays Gaussian?
  (B) Discrimination. Do height-distribution moments break the EW/KPZ degeneracy
      and improve unsupervised recovery beyond gradient + temporal features?

Why the GROWTH regime
---------------------
Tracy-Widom one-point statistics are a property of the *growing* interface. In a
finite periodic system the saturated (stationary) height field is a Brownian
bridge and is Gaussian even for KPZ. The late-time gradient features of Exp 62
are therefore degenerate, but height moments measured while w(t) is still rising
should carry the nonlinear signature. We extract moments in a per-sample growth
window defined from the width curve w(t).

Reference values (one-point, flat initial condition)
----------------------------------------------------
  Gaussian (EW):    skew = 0,       excess kurtosis = 0
  Tracy-Widom GOE:  |skew| ~ 0.2935, excess kurtosis ~ 0.1652
  Tracy-Widom GUE:  |skew| ~ 0.2241, excess kurtosis ~ 0.0934
The sign of the skewness follows the sign of lambda; we report signed values and
|skew|, and compare |skew| magnitude to the TW references. Convergence to the
exact TW values is slow at accessible (L, T), so the test is the EW-vs-KPZ
contrast and the direction of departure, not exact TW recovery.

Caveat
------
Pooling over x at fixed t estimates the one-point marginal under spatial
homogeneity (flat IC, periodic BC), the standard numerical proxy; it is not
identical to an ensemble one-point distribution at a fixed site.

Usage
-----
  python 67_height_distribution_features.py            # pilot multi-seed
  python 67_height_distribution_features.py --quick    # tiny smoke run
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_exp67_height_dist"
EXP63_PATH = SCRIPT_DIR / "63_temporal_features.py"

SYSTEMS = ["ew", "kpz", "bd", "eden", "rd", "ks"]
CLASS_MAP = {
    "ew": "EW", "kpz": "KPZ", "bd": "KPZ", "eden": "KPZ",
    "rd": "trivial", "ks": "KS",
}
SYSTEM_SEED_OFFSETS = {
    "ew": 0, "kpz": 100_000, "bd": 200_000,
    "eden": 300_000, "rd": 400_000, "ks": 500_000,
}

# One-point reference values for the standardized height fluctuation.
TW_REFERENCE = {
    "gaussian": {"skew": 0.0, "exkurt": 0.0},
    "tw_goe": {"skew": 0.2935, "exkurt": 0.1652},
    "tw_gue": {"skew": 0.2241, "exkurt": 0.0934},
}
HEIGHT_NAMES = ["h_skew_growth", "h_exkurt_growth", "h_absskew_growth"]


def load_exp63_module():
    spec = importlib.util.spec_from_file_location("exp63_temporal_features", EXP63_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXP63_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_height_moments(trajectory):
    """One-point height-distribution moments in the growth regime.

    Returns [h_skew, h_excess_kurtosis, |h_skew|], pooled over a per-sample
    growth window defined from the width curve w(t) = std_x h(x, t).
    """
    T, L = trajectory.shape
    widths = np.std(trajectory, axis=1)
    if T < 6:
        return np.array([0.0, 0.0, 0.0])

    w_sat = np.median(widths[int(0.8 * T):])
    if w_sat > 1e-12:
        in_window = (widths > 0.2 * w_sat) & (widths < 0.8 * w_sat)
        idx = np.where(in_window)[0]
        idx = idx[idx > max(1, int(0.02 * T))]
    else:
        idx = np.array([], dtype=int)

    if len(idx) < 3:  # fallback: fixed early-mid window
        lo = max(2, int(0.05 * T))
        hi = max(lo + 3, int(0.4 * T))
        idx = np.arange(lo, min(hi, T))

    skews, kurts = [], []
    for t in idx:
        h = trajectory[t]
        s = np.std(h)
        if s < 1e-12:
            continue
        hc = (h - np.mean(h)) / s
        skews.append(float(np.mean(hc ** 3)))
        kurts.append(float(np.mean(hc ** 4) - 3.0))
    if not skews:
        return np.array([0.0, 0.0, 0.0])
    h_skew = float(np.mean(skews))
    h_kurt = float(np.mean(kurts))
    return np.array([h_skew, h_kurt, abs(h_skew)])


def simulate_one(exp63, system, L, T, seed):
    """Simulate one trajectory using the Exp 63 simulators + parameter draws."""
    if system == "ew":
        nu = np.random.uniform(0.5, 2.0)
        D = np.random.uniform(0.5, 2.0)
        return exp63.simulate_ew(L=L, T=T, nu=nu, D=D, seed=seed)
    if system == "kpz":
        nu = np.random.uniform(0.5, 2.0)
        lam = np.random.uniform(0.5, 3.0)
        D = np.random.uniform(0.5, 2.0)
        traj = exp63.simulate_kpz(L=L, T=T, nu=nu, lam=lam, D=D, seed=seed)
        if traj is None:
            traj = exp63.simulate_kpz(L=L, T=T, nu=nu, lam=np.random.uniform(0.3, 1.5),
                                      D=D, seed=seed + 10000)
        if traj is None:
            traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=0.5, D=1.0, seed=seed + 20000)
        return traj
    if system == "bd":
        return exp63.simulate_bd(L=L, T=T, seed=seed)
    if system == "eden":
        return exp63.simulate_eden(L=L, T=T, seed=seed)
    if system == "rd":
        return exp63.simulate_rd(L=L, T=T, seed=seed)
    if system == "ks":
        ks_T = min(T, 500)
        traj = exp63.simulate_ks(L=L, T=ks_T, seed=seed, record_interval=10)
        if traj is None:
            traj = exp63.simulate_ks(L=L, T=ks_T, seed=seed + 10000,
                                     record_interval=10, noise=0.02)
        if traj is None:
            traj = exp63.simulate_ks(L=L, T=300, seed=seed + 20000,
                                     record_interval=5, noise=0.01)
        return traj
    raise ValueError(system)


def generate_seed_dataset(exp63, *, seed_base, n_samples, L, T):
    """One trajectory per sample -> spatial(6) + temporal(4) + height(3)."""
    spatial_all, temporal_all, height_all, labels = [], [], [], []
    for system in SYSTEMS:
        np.random.seed(seed_base + SYSTEM_SEED_OFFSETS[system])
        for i in range(n_samples):
            seed = seed_base + SYSTEM_SEED_OFFSETS[system] + i
            traj = simulate_one(exp63, system, L, T, seed)
            if traj is None:
                continue
            spatial = exp63.extract_spatial_features(traj, 0.3)
            temporal = exp63.compute_temporal_features(traj)
            height = compute_height_moments(traj)
            row = np.concatenate([spatial, temporal, height])
            if np.any(np.isnan(row)) or np.any(np.isinf(row)):
                continue
            spatial_all.append(spatial)
            temporal_all.append(temporal)
            height_all.append(height)
            labels.append(system)
    spatial = np.array(spatial_all)
    temporal = np.array(temporal_all)
    height = np.array(height_all)
    classes = np.array([CLASS_MAP[l] for l in labels])
    return spatial, temporal, height, np.array(labels), classes


def class_centroid_distance(X_scaled, classes, a, b):
    classes = np.asarray(classes)
    ma, mb = classes == a, classes == b
    if not np.any(ma) or not np.any(mb):
        return float("nan")
    return float(np.linalg.norm(X_scaled[ma].mean(0) - X_scaled[mb].mean(0)))


def evaluate_matrix(X, y):
    y = np.asarray(y)
    X_scaled = StandardScaler().fit_transform(X)
    n_classes = len(set(y))
    n_samples = len(y)
    min_cluster_size = max(5, n_samples // 20)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                                min_samples=3, metric="euclidean")
    hdb_labels = clusterer.fit_predict(X_scaled)
    n_clusters = len(set(hdb_labels) - {-1})
    hdb_all_ari = adjusted_rand_score(y, hdb_labels)
    mask = hdb_labels != -1
    if n_clusters >= 2 and np.any(mask):
        hdb_core_ari = adjusted_rand_score(y[mask], hdb_labels[mask])
    else:
        hdb_core_ari = 0.0

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20)
    km_ari = adjusted_rand_score(y, km.fit_predict(X_scaled))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn = cross_val_score(
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        X, y, cv=cv, scoring="accuracy")

    return {
        "hdbscan_clusters": int(n_clusters),
        "hdbscan_all_ari": float(hdb_all_ari),
        "hdbscan_core_ari": float(hdb_core_ari),
        "kmeans_ari": float(km_ari),
        "knn3_accuracy": float(np.mean(knn)),
        "ew_kpz_dist": class_centroid_distance(X_scaled, y, "EW", "KPZ"),
    }


def summarize(rows, metric_names):
    out = {}
    for m in metric_names:
        vals = np.array([r[m] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[m] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)), "max": float(np.max(vals)),
            }
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp 67 height-distribution features")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30, help="samples per system per seed")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=67_000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.seeds, args.samples, args.L, args.T = 2, 12, 96, 300

    RESULTS_DIR.mkdir(exist_ok=True)
    exp63 = load_exp63_module()

    print("Experiment 67: Tracy-Widom height-distribution features")
    print(f"Config: seeds={args.seeds}, N={args.samples}/system, L={args.L}, T={args.T}")
    start = time.time()

    # Representations to compare. Indices into [spatial(6) | temporal(4) | height(3)].
    representations = {
        "6d_spatial": lambda s, t, h: s,
        "8d_spatial_height": lambda s, t, h: np.hstack([s, h[:, :2]]),
        "2d_height_only": lambda s, t, h: h[:, :2],
        "10d_spatial_temporal": lambda s, t, h: np.hstack([s, t]),
        "12d_all": lambda s, t, h: np.hstack([s, t, h[:, :2]]),
    }
    metric_names = ["hdbscan_all_ari", "hdbscan_core_ari", "kmeans_ari",
                    "knn3_accuracy", "ew_kpz_dist"]

    rep_rows = {r: [] for r in representations}
    height_by_system = {s: [] for s in SYSTEMS}
    all_rows = []

    for seed_idx in range(args.seeds):
        seed_base = args.seed_start + seed_idx * 10_000
        spatial, temporal, height, labels, classes = generate_seed_dataset(
            exp63, seed_base=seed_base, n_samples=args.samples, L=args.L, T=args.T)
        print(f"\nseed block {seed_idx + 1}/{args.seeds}: {len(classes)} samples")

        for s in SYSTEMS:
            m = labels == s
            if np.any(m):
                height_by_system[s].append(height[m])

        for rep, fn in representations.items():
            X = fn(spatial, temporal, height)
            metrics = evaluate_matrix(X, classes)
            rep_rows[rep].append(metrics)
            all_rows.append({"seed_index": seed_idx, "representation": rep, **metrics})
        print(
            f"  6D EWKPZ={rep_rows['6d_spatial'][-1]['ew_kpz_dist']:.3f} | "
            f"+height EWKPZ={rep_rows['8d_spatial_height'][-1]['ew_kpz_dist']:.3f} "
            f"core={rep_rows['8d_spatial_height'][-1]['hdbscan_core_ari']:.3f} "
            f"KM={rep_rows['8d_spatial_height'][-1]['kmeans_ari']:.3f} "
            f"kNN={rep_rows['8d_spatial_height'][-1]['knn3_accuracy']:.3f}"
        )

    rep_summary = {r: summarize(rows, metric_names) for r, rows in rep_rows.items()}

    # Per-system growth-regime height moments vs references.
    system_moments = {}
    print("\n=== growth-regime one-point height moments by system ===")
    print(f"{'system':>6} {'class':>8} {'skew':>8} {'|skew|':>8} {'exkurt':>8}")
    for s in SYSTEMS:
        if not height_by_system[s]:
            continue
        H = np.vstack(height_by_system[s])
        skew_m, skew_s = float(np.mean(H[:, 0])), float(np.std(H[:, 0]))
        absskew_m = float(np.mean(H[:, 2]))
        kurt_m, kurt_s = float(np.mean(H[:, 1])), float(np.std(H[:, 1]))
        system_moments[s] = {
            "class": CLASS_MAP[s], "n": int(H.shape[0]),
            "skew_mean": skew_m, "skew_std": skew_s,
            "absskew_mean": absskew_m,
            "exkurt_mean": kurt_m, "exkurt_std": kurt_s,
        }
        print(f"{s:>6} {CLASS_MAP[s]:>8} {skew_m:>8.3f} {absskew_m:>8.3f} {kurt_m:>8.3f}")
    print(f"  reference TW-GOE: |skew|~{TW_REFERENCE['tw_goe']['skew']:.3f} "
          f"exkurt~{TW_REFERENCE['tw_goe']['exkurt']:.3f}; Gaussian: 0, 0")

    print("\n=== representation comparison (mean over seeds) ===")
    print(f"{'representation':>22} {'HDBcore':>8} {'KM':>7} {'kNN':>7} {'EWKPZ':>7}")
    for rep in representations:
        s = rep_summary[rep]
        print(f"{rep:>22} {s['hdbscan_core_ari']['mean']:>8.3f} "
              f"{s['kmeans_ari']['mean']:>7.3f} {s['knn3_accuracy']['mean']:>7.3f} "
              f"{s['ew_kpz_dist']['mean']:>7.3f}")

    write_csv(RESULTS_DIR / "representation_rows.csv", all_rows)
    summary = {
        "config": vars(args),
        "elapsed_seconds": time.time() - start,
        "tw_reference": TW_REFERENCE,
        "system_height_moments": system_moments,
        "representation_summary": rep_summary,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['elapsed_seconds']:.1f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
