"""
Experiment 66: Finite-Size (L-T) Scaling Sweep of the Clustering Ceiling
========================================================================

Decisive test #1 from the manuscript limitations: does the ARI ~ 0.5 density-
clustering ceiling, and the EW-KPZ feature-space separation, move toward
universality recovery as system size L and observation time T grow, or does it
persist?

Physics motivation
------------------
  - The EW/KPZ gradient degeneracy is a STATIONARY-LIMIT statement: the 1D KPZ
    stationary slope measure is Gaussian and lambda-independent (Barabasi &
    Stanley, Ch. 4). So as T grows toward saturation, EW and KPZ stationary
    gradient features should converge, and the EW-KPZ centroid distance is
    predicted to DECREASE with size. This is a sharp, falsifiable prediction.
  - The discrete-to-continuum corrections of BD/Eden close only slowly with L,
    so the intended KPZ class (KPZ + BD + Eden) may or may not become a single
    density cluster as L grows.

Method
------
Reuses the Exp 63 generator (simulators + 6D spatial + 4D temporal features) and
evaluates HDBSCAN / KMeans / leakage-safe kNN, plus the EW-vs-KPZ class centroid
distance, over a ladder of (L, T) sizes and several seeds.

Outputs
-------
  - per-(size, seed) metrics for 6D and 10D representations (CSV)
  - mean +/- std summary across seeds at each size (summary.json)

Note
----
The KS generator caps its recorded length at T = 500 inside the Exp 63 generator,
so KS features do not scale with T; the scaling interpretation therefore focuses
on EW vs KPZ and on the all-system ceiling.

Usage
-----
  python 66_finite_size_sweep.py            # default ladder (~minutes)
  python 66_finite_size_sweep.py --quick    # tiny smoke ladder
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
RESULTS_DIR = PROJECT_DIR / "results_exp66_finite_size"
EXP63_PATH = SCRIPT_DIR / "63_temporal_features.py"

SYSTEMS = ["ew", "kpz", "bd", "eden", "rd", "ks"]
CLASS_MAP = {
    "ew": "EW",
    "kpz": "KPZ",
    "bd": "KPZ",
    "eden": "KPZ",
    "rd": "trivial",
    "ks": "KS",
}
SYSTEM_SEED_OFFSETS = {
    "ew": 0,
    "kpz": 100_000,
    "bd": 200_000,
    "eden": 300_000,
    "rd": 400_000,
    "ks": 500_000,
}

# (L, T) ladder. T grows with L so each system advances along its Family-Vicsek
# curve as the system grows. The largest point matches the Exp 62/63 full config.
DEFAULT_SIZES = [(64, 500), (96, 750), (128, 1000), (192, 1500), (256, 2000)]
QUICK_SIZES = [(48, 200), (64, 300), (96, 400)]


def load_exp63_module():
    spec = importlib.util.spec_from_file_location("exp63_temporal_features", EXP63_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXP63_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_centroid_distance(X_scaled, classes, a, b):
    """Euclidean distance between two class centroids in standardized space."""
    classes = np.asarray(classes)
    ma, mb = classes == a, classes == b
    if not np.any(ma) or not np.any(mb):
        return float("nan")
    return float(np.linalg.norm(X_scaled[ma].mean(0) - X_scaled[mb].mean(0)))


def evaluate_matrix(X, y, *, min_cluster_size=None, min_samples=3):
    """Evaluate one feature matrix against the intended labels."""
    y = np.asarray(y)
    X_scaled = StandardScaler().fit_transform(X)
    n_classes = len(set(y))
    n_samples = len(y)
    if min_cluster_size is None:
        min_cluster_size = max(5, n_samples // 20)

    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError(
            "hdbscan is required for Exp. 66. Install with `pip install hdbscan`."
        ) from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
    )
    hdb_labels = clusterer.fit_predict(X_scaled)
    n_clusters = len(set(hdb_labels) - {-1})
    n_noise = int(np.sum(hdb_labels == -1))
    hdb_all_ari = adjusted_rand_score(y, hdb_labels)
    mask = hdb_labels != -1
    if n_clusters >= 2 and np.any(mask):
        hdb_core_ari = adjusted_rand_score(y[mask], hdb_labels[mask])
        hdb_core_nmi = normalized_mutual_info_score(y[mask], hdb_labels[mask])
    else:
        hdb_core_ari = 0.0
        hdb_core_nmi = 0.0

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20)
    km_labels = km.fit_predict(X_scaled)
    km_ari = adjusted_rand_score(y, km_labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_scores = cross_val_score(
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        X, y, cv=cv, scoring="accuracy",
    )

    return {
        "hdbscan_clusters": int(n_clusters),
        "hdbscan_noise": int(n_noise),
        "hdbscan_all_ari": float(hdb_all_ari),
        "hdbscan_core_ari": float(hdb_core_ari),
        "hdbscan_core_nmi": float(hdb_core_nmi),
        "kmeans_ari": float(km_ari),
        "knn3_accuracy": float(np.mean(knn_scores)),
        "knn3_std": float(np.std(knn_scores)),
        "ew_kpz_dist": class_centroid_distance(X_scaled, y, "EW", "KPZ"),
        "kpz_ks_dist": class_centroid_distance(X_scaled, y, "KPZ", "KS"),
    }


def generate_seed_dataset(exp63, *, seed_base, n_samples, L, T):
    spatial_all, temporal_all, labels = [], [], []
    for system in SYSTEMS:
        sim_kwargs = {}
        if system == "ks":
            sim_kwargs["record_interval"] = 10
        exp63.np.random.seed(seed_base + SYSTEM_SEED_OFFSETS[system])
        spatial, temporal, _ = exp63.generate_feature_dataset(
            n_samples, L, T, system,
            seed_offset=seed_base + SYSTEM_SEED_OFFSETS[system],
            late_frac=0.3, **sim_kwargs,
        )
        spatial_all.append(spatial)
        temporal_all.append(temporal)
        labels.extend([system] * spatial.shape[0])
    spatial = np.vstack(spatial_all)
    temporal = np.vstack(temporal_all)
    combined = np.hstack([spatial, temporal])
    classes = np.array([CLASS_MAP[l] for l in labels])
    return spatial, combined, np.array(labels), classes


def summarize(rows, metric_names):
    out = {}
    for metric in metric_names:
        vals = np.array([r[metric] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[metric] = {"mean": float("nan"), "std": float("nan")}
            continue
        out[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp 66 finite-size L-T sweep")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=25, help="samples per system per seed")
    parser.add_argument("--seed-start", type=int, default=66_000)
    parser.add_argument("--quick", action="store_true", help="tiny smoke ladder")
    args = parser.parse_args()

    sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    if args.quick:
        args.samples = min(args.samples, 12)
        args.seeds = min(args.seeds, 2)

    RESULTS_DIR.mkdir(exist_ok=True)
    exp63 = load_exp63_module()

    print("Experiment 66: finite-size (L-T) scaling sweep")
    print(f"Sizes: {sizes}")
    print(f"Config: seeds={args.seeds}, N={args.samples}/system/seed")
    start = time.time()

    metric_names = [
        "hdbscan_all_ari", "hdbscan_core_ari", "kmeans_ari",
        "knn3_accuracy", "ew_kpz_dist", "kpz_ks_dist",
    ]

    all_rows = []
    per_size_summary = {}

    for (L, T) in sizes:
        print(f"\n=== size L={L}, T={T} ===")
        size_rows = {"6d_spatial": [], "10d_spatial_temporal": []}
        for seed_idx in range(args.seeds):
            seed_base = args.seed_start + seed_idx * 10_000 + L
            spatial, combined, labels, classes = generate_seed_dataset(
                exp63, seed_base=seed_base, n_samples=args.samples, L=L, T=T,
            )
            for rep, X in [("6d_spatial", spatial), ("10d_spatial_temporal", combined)]:
                m = evaluate_matrix(X, classes)
                row = {"L": L, "T": T, "seed_index": seed_idx,
                       "representation": rep, "n_samples": int(len(classes)), **m}
                all_rows.append(row)
                size_rows[rep].append(m)
            print(
                f"  seed {seed_idx}: "
                f"6D HDB_core={size_rows['6d_spatial'][-1]['hdbscan_core_ari']:.3f} "
                f"EWKPZ={size_rows['6d_spatial'][-1]['ew_kpz_dist']:.3f} | "
                f"10D HDB_core={size_rows['10d_spatial_temporal'][-1]['hdbscan_core_ari']:.3f} "
                f"KM={size_rows['10d_spatial_temporal'][-1]['kmeans_ari']:.3f} "
                f"kNN={size_rows['10d_spatial_temporal'][-1]['knn3_accuracy']:.3f} "
                f"EWKPZ={size_rows['10d_spatial_temporal'][-1]['ew_kpz_dist']:.3f}"
            )
        per_size_summary[f"L{L}_T{T}"] = {
            rep: summarize(rows, metric_names) for rep, rows in size_rows.items()
        }

    write_csv(RESULTS_DIR / "sweep_rows.csv", all_rows)

    summary = {
        "config": {"sizes": sizes, "seeds": args.seeds, "samples": args.samples},
        "elapsed_seconds": time.time() - start,
        "per_size": per_size_summary,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compact trend table
    print("\n=== TREND: EW-KPZ centroid distance and ceiling vs size ===")
    print(f"{'size':>12} | {'6D HDBcore':>10} {'6D EWKPZ':>9} | "
          f"{'10D HDBcore':>11} {'10D KM':>7} {'10D kNN':>8} {'10D EWKPZ':>10}")
    for (L, T) in sizes:
        s = per_size_summary[f"L{L}_T{T}"]
        a, b = s["6d_spatial"], s["10d_spatial_temporal"]
        print(
            f"  L={L:<3} T={T:<5}|"
            f"{a['hdbscan_core_ari']['mean']:>10.3f}"
            f"{a['ew_kpz_dist']['mean']:>9.3f} |"
            f"{b['hdbscan_core_ari']['mean']:>11.3f}"
            f"{b['kmeans_ari']['mean']:>7.3f}"
            f"{b['knn3_accuracy']['mean']:>8.3f}"
            f"{b['ew_kpz_dist']['mean']:>10.3f}"
        )

    print(f"\nDone in {summary['elapsed_seconds']:.1f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
