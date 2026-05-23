"""
MLP-05: Local-vs-global geometry diagnostics.

This derived analysis supports the ML-paper claim that finite-size feature
representations can be locally class-informative while their global cluster
geometry does not match the physical universality quotient.

Outputs:
  ../tables/local_global_gap.csv
  ../tables/exp62_neighborhood_purity.csv
  ../tables/exp62_feature_quotient_geometry.csv
  ../figures/local_vs_global.png
  ../results/mlp05_summary.json
"""

from __future__ import annotations

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
TABLES_DIR = ML_DIR / "tables"
FIGURES_DIR = ML_DIR / "figures"
RESULTS_DIR = ML_DIR / "results"

KPZ_SYSTEMS = {"kpz", "bd", "eden"}
NON_KPZ_REFERENCE = {"ew", "rd", "ks"}


def ensure_dirs() -> None:
    for path in (TABLES_DIR, FIGURES_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fnum(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def fmt(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def collect_local_global_gap() -> list[dict]:
    rows = read_csv(ML_DIR / "tables" / "representation_performance.csv")
    out = []
    for row in rows:
        knn = fnum(row.get("knn3_accuracy_mean"))
        hdb = fnum(row.get("hdbscan_core_ari_mean"))
        km = fnum(row.get("kmeans_ari_mean"))
        if knn is None:
            continue
        hdb_gap = knn - hdb if hdb is not None else None
        km_gap = knn - km if km is not None else None
        out.append({
            "source": row["source"],
            "representation": row["representation"],
            "representation_family": row["representation_family"],
            "protocol": row["protocol"],
            "n_runs": row["n_runs"],
            "hdbscan_core_ari_mean": fmt(hdb),
            "kmeans_ari_mean": fmt(km),
            "knn3_accuracy_mean": fmt(knn),
            "local_minus_hdbscan": fmt(hdb_gap),
            "local_minus_kmeans": fmt(km_gap),
            "notes": row["notes"],
        })
    return out


def load_exp62():
    data = np.load(PROJECT_DIR / "results_exp62" / "features.npz", allow_pickle=True)
    X = np.asarray(data["features"], dtype=float)
    systems = np.asarray(data["labels"], dtype=str)
    classes = np.asarray(data["classes"], dtype=str)
    names = [str(x) for x in data["feature_names"].tolist()]
    return X, systems, classes, names


def expected_same_label(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = labels.size
    vals = []
    for label in labels:
        vals.append((np.sum(labels == label) - 1) / max(1, n - 1))
    return float(np.mean(vals))


def neighborhood_rows(X: np.ndarray, systems: np.ndarray, classes: np.ndarray) -> list[dict]:
    Xs = StandardScaler().fit_transform(X)
    dist = pairwise_distances(Xs)
    order = np.argsort(dist, axis=1)[:, 1:]
    k_values = [1, 3, 5, 10, 20, 40, 80, 120, 160, 240]
    k_values = [k for k in k_values if k < X.shape[0]]
    class_baseline = expected_same_label(classes)
    system_baseline = expected_same_label(systems)

    out = []
    kpz_mask = np.isin(systems, list(KPZ_SYSTEMS))
    for k in k_values:
        neigh = order[:, :k]
        neigh_classes = classes[neigh]
        neigh_systems = systems[neigh]
        class_purity = np.mean(neigh_classes == classes[:, None])
        system_purity = np.mean(neigh_systems == systems[:, None])

        kpz_neigh = neigh[kpz_mask]
        kpz_system_labels = systems[kpz_mask]
        kpz_neigh_systems = systems[kpz_neigh]
        kpz_neigh_classes = classes[kpz_neigh]
        kpz_class_purity = np.mean(kpz_neigh_classes == "KPZ")
        kpz_same_system = np.mean(kpz_neigh_systems == kpz_system_labels[:, None])
        kpz_cross_system_same_class = np.mean(
            (kpz_neigh_classes == "KPZ") & (kpz_neigh_systems != kpz_system_labels[:, None])
        )

        out.append({
            "k": k,
            "universality_class_purity": fmt(class_purity),
            "microscopic_system_purity": fmt(system_purity),
            "class_random_baseline": fmt(class_baseline),
            "system_random_baseline": fmt(system_baseline),
            "kpz_class_purity": fmt(kpz_class_purity),
            "kpz_same_system_fraction": fmt(kpz_same_system),
            "kpz_cross_system_same_class_fraction": fmt(kpz_cross_system_same_class),
            "interpretation": "KPZ cross-system fraction tests whether KPZ/BD/Eden are locally mixed as a quotient.",
        })
    return out


def exp62_cluster_summary(X: np.ndarray, classes: np.ndarray) -> dict:
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=len(set(classes.tolist())), random_state=42, n_init=20).fit_predict(Xs)
    counts = [np.sum(classes == c) for c in sorted(set(classes.tolist()))]
    n_splits = int(min(5, min(counts)))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X, classes, cv=cv)
    out = {
        "kmeans_ari": float(adjusted_rand_score(classes, km)),
        "kmeans_nmi": float(normalized_mutual_info_score(classes, km)),
        "knn3_accuracy_mean": float(scores.mean()),
        "knn3_accuracy_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
    }
    try:
        import hdbscan

        hl = hdbscan.HDBSCAN(min_cluster_size=max(5, X.shape[0] // 20), min_samples=3).fit_predict(Xs)
        mask = hl != -1
        clusters = set(int(v) for v in hl.tolist()) - {-1}
        out["hdbscan_clusters"] = int(len(clusters))
        out["hdbscan_noise"] = int(np.sum(~mask))
        out["hdbscan_ari_all"] = float(adjusted_rand_score(classes, hl))
        out["hdbscan_ari_core"] = (
            float(adjusted_rand_score(classes[mask], hl[mask]))
            if len(clusters) >= 2 and np.sum(mask) > 1
            else None
        )
    except Exception:
        out["hdbscan_clusters"] = None
        out["hdbscan_noise"] = None
        out["hdbscan_ari_all"] = None
        out["hdbscan_ari_core"] = None
    return out


def feature_quotient_geometry(X: np.ndarray, systems: np.ndarray) -> tuple[list[dict], dict]:
    Xs = StandardScaler().fit_transform(X)
    centroids = {s: Xs[systems == s].mean(axis=0) for s in sorted(set(systems.tolist()))}
    rows = []
    distances = {}
    for a, b in combinations(sorted(centroids), 2):
        dist = float(np.linalg.norm(centroids[a] - centroids[b]))
        distances[(a, b)] = dist
        rows.append({
            "system_a": a,
            "system_b": b,
            "standardized_centroid_distance": fmt(dist),
            "same_universality_class": str(a in KPZ_SYSTEMS and b in KPZ_SYSTEMS),
        })

    within_kpz = [
        distances[tuple(sorted((a, b)))]
        for a, b in combinations(sorted(KPZ_SYSTEMS), 2)
        if tuple(sorted((a, b))) in distances
    ]
    cross = []
    for a in KPZ_SYSTEMS:
        for b in NON_KPZ_REFERENCE:
            key = tuple(sorted((a, b)))
            if key in distances:
                cross.append(distances[key])
    max_within = max(within_kpz)
    min_cross = min(cross)
    diagnostic = {
        "max_within_kpz_centroid_distance": float(max_within),
        "min_kpz_to_non_kpz_centroid_distance": float(min_cross),
        "kpz_quotient_separation_ratio": float(max_within / min_cross),
        "interpretation": "good_if_below_1; above 1 means KPZ subclasses are farther apart than at least one non-KPZ class.",
    }
    return rows, diagnostic


def plot_local_vs_global(gap_rows: list[dict], neighborhood: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    ax1, ax2, ax3, ax4 = axes.ravel()

    selected = [
        r for r in gap_rows
        if r["representation_family"] not in {"positive_control", "comparison"}
        and fnum(r["hdbscan_core_ari_mean"]) is not None
        and fnum(r["kmeans_ari_mean"]) is not None
    ][:10]
    labels = [r["representation"].replace("_", " ") for r in selected]
    x = np.arange(len(selected))
    width = 0.25
    knn = [fnum(r["knn3_accuracy_mean"]) for r in selected]
    hdb = [fnum(r["hdbscan_core_ari_mean"]) for r in selected]
    km = [fnum(r["kmeans_ari_mean"]) for r in selected]
    ax1.bar(x - width, knn, width, label="kNN3 accuracy", color="#4c78a8")
    ax1.bar(x, hdb, width, label="HDBSCAN core ARI", color="#f58518")
    ax1.bar(x + width, km, width, label="KMeans ARI", color="#54a24b")
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel("Score")
    ax1.set_title("Local probes can succeed while global clustering fails")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=0.18)

    gaps = [fnum(r["local_minus_hdbscan"]) for r in selected]
    colors = ["#b279a2" if g is not None and g > 0.25 else "#bab0ac" for g in gaps]
    ax2.bar(x, gaps, color=colors)
    ax2.axhline(0, color="0.2", linewidth=0.8)
    ax2.set_ylim(0, max(0.55, max(gaps) + 0.06))
    ax2.set_ylabel("kNN3 - HDBSCAN core ARI")
    ax2.set_title("Local-global gap")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.grid(axis="y", alpha=0.18)

    k = np.asarray([int(r["k"]) for r in neighborhood])
    class_purity = np.asarray([float(r["universality_class_purity"]) for r in neighborhood])
    system_purity = np.asarray([float(r["microscopic_system_purity"]) for r in neighborhood])
    class_base = float(neighborhood[0]["class_random_baseline"])
    system_base = float(neighborhood[0]["system_random_baseline"])
    ax3.plot(k, class_purity, marker="o", label="universality class purity", color="#4c78a8")
    ax3.plot(k, system_purity, marker="s", label="microscopic system purity", color="#e45756")
    ax3.axhline(class_base, color="#4c78a8", linestyle=":", linewidth=1.2, label="class random baseline")
    ax3.axhline(system_base, color="#e45756", linestyle=":", linewidth=1.2, label="system random baseline")
    ax3.set_xscale("log")
    ax3.set_ylim(0, 1.04)
    ax3.set_xlabel("number of nearest neighbors k")
    ax3.set_ylabel("mean same-label fraction")
    ax3.set_title("Exp62 local neighborhoods are label-enriched")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(alpha=0.18)

    kpz_class = np.asarray([float(r["kpz_class_purity"]) for r in neighborhood])
    kpz_same = np.asarray([float(r["kpz_same_system_fraction"]) for r in neighborhood])
    kpz_cross = np.asarray([float(r["kpz_cross_system_same_class_fraction"]) for r in neighborhood])
    ax4.plot(k, kpz_class, marker="o", label="KPZ-class neighbors", color="#4c78a8")
    ax4.plot(k, kpz_same, marker="s", label="same microscopic system", color="#e45756")
    ax4.plot(k, kpz_cross, marker="^", label="other KPZ-class system", color="#72b7b2")
    ax4.set_xscale("log")
    ax4.set_ylim(0, 1.04)
    ax4.set_xlabel("number of nearest neighbors k")
    ax4.set_ylabel("fraction among KPZ-labeled points")
    ax4.set_title("KPZ local purity is not automatically quotient mixing")
    ax4.legend(frameon=False, fontsize=8)
    ax4.grid(alpha=0.18)

    fig.suptitle(
        "Local separability does not imply global physical quotient recovery",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIGURES_DIR / "local_vs_global.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    gap_rows = collect_local_global_gap()
    X, systems, classes, feature_names = load_exp62()
    neighborhood = neighborhood_rows(X, systems, classes)
    cluster_summary = exp62_cluster_summary(X, classes)
    quotient_rows, quotient_summary = feature_quotient_geometry(X, systems)

    write_csv(TABLES_DIR / "local_global_gap.csv", gap_rows)
    write_csv(TABLES_DIR / "exp62_neighborhood_purity.csv", neighborhood)
    write_csv(TABLES_DIR / "exp62_feature_quotient_geometry.csv", quotient_rows)

    local_minus_hdb = [fnum(r["local_minus_hdbscan"]) for r in gap_rows]
    local_minus_hdb = [x for x in local_minus_hdb if x is not None]
    key_k = {int(r["k"]): r for r in neighborhood}
    summary = {
        "n_gap_rows": len(gap_rows),
        "local_minus_hdbscan_mean": float(np.mean(local_minus_hdb)) if local_minus_hdb else None,
        "local_minus_hdbscan_min": float(np.min(local_minus_hdb)) if local_minus_hdb else None,
        "local_minus_hdbscan_max": float(np.max(local_minus_hdb)) if local_minus_hdb else None,
        "exp62_cluster_summary": cluster_summary,
        "exp62_feature_quotient_geometry": quotient_summary,
        "exp62_neighborhood_key_points": {
            str(k): key_k[k] for k in [1, 3, 10, 40, 80] if k in key_k
        },
        "feature_names": feature_names,
        "outputs": [
            "ml_paper/tables/local_global_gap.csv",
            "ml_paper/tables/exp62_neighborhood_purity.csv",
            "ml_paper/tables/exp62_feature_quotient_geometry.csv",
            "ml_paper/figures/local_vs_global.png",
            "ml_paper/results/mlp05_summary.json",
        ],
    }
    plot_local_vs_global(gap_rows, neighborhood, summary)
    (RESULTS_DIR / "mlp05_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
