"""
Experiment 65: Robustness Checks for Feature-Geometry Diagnostics
=================================================================

This script adds a compact robustness layer around Experiments 62 and 63.
It uses the Exp. 63 generator because that produces both:

  - 6D spatial features used in Exp. 62
  - 10D spatial + temporal features used in Exp. 63

Outputs:
  - multi-seed pilot metrics for 6D and 10D representations
  - HDBSCAN sensitivity over min_cluster_size and min_samples
  - all-point ARI and non-noise ARI for HDBSCAN
  - leakage-safe kNN cross-validation using a scaler inside the CV pipeline

The defaults are intentionally modest so this can be rerun quickly before
updating the manuscript.
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
RESULTS_DIR = PROJECT_DIR / "results_exp65_robustness"
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


def load_exp63_module():
    spec = importlib.util.spec_from_file_location("exp63_temporal_features", EXP63_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EXP63_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            "hdbscan is required for Exp. 65 robustness checks. "
            "Install it with `python -m pip install hdbscan`."
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
    hdb_all_nmi = normalized_mutual_info_score(y, hdb_labels)
    mask = hdb_labels != -1
    if n_clusters >= 2 and np.any(mask):
        hdb_non_noise_ari = adjusted_rand_score(y[mask], hdb_labels[mask])
        hdb_non_noise_nmi = normalized_mutual_info_score(y[mask], hdb_labels[mask])
    else:
        hdb_non_noise_ari = 0.0
        hdb_non_noise_nmi = 0.0

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20)
    km_labels = km.fit_predict(X_scaled)
    km_ari = adjusted_rand_score(y, km_labels)
    km_nmi = normalized_mutual_info_score(y, km_labels)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_scores = cross_val_score(
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    return {
        "hdbscan_clusters": int(n_clusters),
        "hdbscan_noise": int(n_noise),
        "hdbscan_all_ari": float(hdb_all_ari),
        "hdbscan_all_nmi": float(hdb_all_nmi),
        "hdbscan_non_noise_ari": float(hdb_non_noise_ari),
        "hdbscan_non_noise_nmi": float(hdb_non_noise_nmi),
        "kmeans_ari": float(km_ari),
        "kmeans_nmi": float(km_nmi),
        "knn3_accuracy": float(np.mean(knn_scores)),
        "knn3_std": float(np.std(knn_scores)),
    }


def summarize_rows(rows, metric_names):
    out = {}
    for metric in metric_names:
        vals = np.array([row[metric] for row in rows], dtype=float)
        out[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def generate_seed_dataset(exp63, *, seed_base, n_samples, L, T):
    spatial_all = []
    temporal_all = []
    labels = []

    for system in SYSTEMS:
        sim_kwargs = {}
        if system == "ks":
            sim_kwargs["record_interval"] = 10

        # Keep both the stochastic trajectory seeds and Python-side parameter
        # draws deterministic for each seed block.
        exp63.np.random.seed(seed_base + SYSTEM_SEED_OFFSETS[system])
        spatial, temporal, _ = exp63.generate_feature_dataset(
            n_samples,
            L,
            T,
            system,
            seed_offset=seed_base + SYSTEM_SEED_OFFSETS[system],
            late_frac=0.3,
            **sim_kwargs,
        )

        spatial_all.append(spatial)
        temporal_all.append(temporal)
        labels.extend([system] * spatial.shape[0])

    spatial = np.vstack(spatial_all)
    temporal = np.vstack(temporal_all)
    combined = np.hstack([spatial, temporal])
    classes = np.array([CLASS_MAP[label] for label in labels])
    return spatial, combined, np.array(labels), classes


def load_saved_full_run():
    exp62 = np.load(PROJECT_DIR / "results_exp62" / "features.npz", allow_pickle=True)
    exp63 = np.load(PROJECT_DIR / "results_exp63" / "features.npz", allow_pickle=True)
    return {
        "exp62_spatial_saved": (exp62["features"], exp62["classes"]),
        "exp63_spatial_saved": (exp63["spatial"], exp63["classes"]),
        "exp63_combined_saved": (exp63["combined"], exp63["classes"]),
    }


def hdbscan_sensitivity(X, y):
    rows = []
    n = len(y)
    size_grid = sorted(set([5, 10, max(5, n // 40), max(5, n // 20), max(5, n // 10)]))
    sample_grid = [1, 3, 5, 10]
    for min_cluster_size in size_grid:
        for min_samples in sample_grid:
            metrics = evaluate_matrix(
                X,
                y,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
            rows.append(
                {
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                    **metrics,
                }
            )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Exp 65 robustness checks")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30, help="samples per system per seed")
    parser.add_argument("--L", type=int, default=128)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=65_000)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    exp63 = load_exp63_module()

    print("Experiment 65: robustness checks")
    print(f"Config: seeds={args.seeds}, N={args.samples}/system, L={args.L}, T={args.T}")
    start = time.time()

    seed_rows = []
    for seed_idx in range(args.seeds):
        seed_base = args.seed_start + seed_idx * 10_000
        print(f"\nSeed block {seed_idx + 1}/{args.seeds}: base={seed_base}")
        spatial, combined, labels, classes = generate_seed_dataset(
            exp63,
            seed_base=seed_base,
            n_samples=args.samples,
            L=args.L,
            T=args.T,
        )
        print(f"  generated {len(classes)} samples")

        for representation, X in [("6d_spatial", spatial), ("10d_spatial_temporal", combined)]:
            metrics = evaluate_matrix(X, classes)
            row = {
                "seed_index": seed_idx,
                "seed_base": seed_base,
                "representation": representation,
                "n_samples": int(len(classes)),
                **metrics,
            }
            seed_rows.append(row)
            print(
                f"  {representation}: "
                f"HDBSCAN all/non-noise ARI="
                f"{metrics['hdbscan_all_ari']:.3f}/"
                f"{metrics['hdbscan_non_noise_ari']:.3f}, "
                f"KMeans ARI={metrics['kmeans_ari']:.3f}, "
                f"3-NN={metrics['knn3_accuracy']:.3f}"
            )

    metric_names = [
        "hdbscan_all_ari",
        "hdbscan_non_noise_ari",
        "kmeans_ari",
        "knn3_accuracy",
    ]
    multi_seed_summary = {}
    for representation in sorted({row["representation"] for row in seed_rows}):
        rep_rows = [row for row in seed_rows if row["representation"] == representation]
        multi_seed_summary[representation] = summarize_rows(rep_rows, metric_names)

    print("\nSaved full-run sensitivity")
    saved_summaries = {}
    sensitivity_tables = {}
    for name, (X, y) in load_saved_full_run().items():
        baseline = evaluate_matrix(X, y)
        sensitivity = hdbscan_sensitivity(X, y)
        sensitivity_tables[name] = sensitivity
        saved_summaries[name] = {
            "baseline": baseline,
            "sensitivity": summarize_rows(
                sensitivity,
                ["hdbscan_all_ari", "hdbscan_non_noise_ari", "hdbscan_noise"],
            ),
        }
        write_csv(RESULTS_DIR / f"{name}_hdbscan_sensitivity.csv", sensitivity)
        print(
            f"  {name}: baseline all/non-noise ARI="
            f"{baseline['hdbscan_all_ari']:.3f}/"
            f"{baseline['hdbscan_non_noise_ari']:.3f}; "
            f"sensitivity non-noise ARI range="
            f"{saved_summaries[name]['sensitivity']['hdbscan_non_noise_ari']['min']:.3f}-"
            f"{saved_summaries[name]['sensitivity']['hdbscan_non_noise_ari']['max']:.3f}"
        )

    write_csv(RESULTS_DIR / "multi_seed_pilot_metrics.csv", seed_rows)

    summary = {
        "config": vars(args),
        "elapsed_seconds": time.time() - start,
        "multi_seed_pilot": {
            "rows": seed_rows,
            "summary": multi_seed_summary,
        },
        "saved_full_run": saved_summaries,
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['elapsed_seconds']:.1f}s")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
