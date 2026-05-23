"""
MLP-08: Clusterer and hierarchy controls.

This derived analysis attacks two simple alternative explanations:

1. The observed quotient failure is just a KMeans/HDBSCAN artifact.
2. The representation over-splits microscopic systems, but a simple unsupervised
   hierarchy would merge the KPZ subclasses before mixing in other classes.

It reads only stored vectors:
  - results_exp62/features.npz
  - results_exp70_matched_codex_sweep_full_{equal,exp69}_seed*/summary.json

Exp70 feature matrices are not stored, so this control cannot refit those
feature baselines; it only tests exp62 features and matched exponent vectors.

Outputs:
  ../tables/clusterer_control_rows.csv
  ../tables/clusterer_control_summary.csv
  ../tables/hierarchical_merge_audit.csv
  ../tables/hierarchical_pairwise_distances.csv
  ../figures/clusterer_control_best_ari.png
  ../figures/system_vs_universality_alignment.png
  ../figures/hierarchical_kpz_merge_audit.png
  ../results/mlp08_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
TABLES_DIR = ML_DIR / "tables"
FIGURES_DIR = ML_DIR / "figures"
RESULTS_DIR = ML_DIR / "results"

CLASS_MAP = {
    "ew": "EW",
    "kpz": "KPZ",
    "bd": "KPZ",
    "eden": "KPZ",
    "rd": "trivial",
    "ks": "KS",
}

SUBSETS = [
    ("all_six", ["ew", "kpz", "bd", "eden", "rd", "ks"], "EW/KPZ/BD/Eden/RD/KS"),
    ("no_ks", ["ew", "kpz", "bd", "eden", "rd"], "EW/KPZ/BD/Eden/RD"),
    ("no_rd", ["ew", "kpz", "bd", "eden", "ks"], "EW/KPZ/BD/Eden/KS"),
    ("no_rd_no_ks", ["ew", "kpz", "bd", "eden"], "EW/KPZ/BD/Eden"),
    ("ew_kpz_only", ["ew", "kpz"], "EW/KPZ only"),
]

KPZ_SYSTEMS = {"kpz", "bd", "eden"}


def ensure_dirs() -> None:
    for path in (TABLES_DIR, FIGURES_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


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
    value = fnum(value)
    return "" if value is None else f"{value:.6g}"


def class_labels(system_labels: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_MAP[str(s)] for s in system_labels], dtype=object)


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def load_datasets() -> list[dict]:
    datasets = []
    exp62 = np.load(PROJECT_DIR / "results_exp62" / "features.npz", allow_pickle=True)
    datasets.append({
        "dataset_id": "exp62_spatial",
        "source": "results_exp62/features.npz",
        "representation": "6D spatial morphology",
        "protocol": "stored exp62 feature matrix",
        "dataset_family": "feature",
        "seed_start": "",
        "X": np.asarray(exp62["features"], dtype=float),
        "systems": np.asarray(exp62["labels"], dtype=str),
    })

    pattern = re.compile(r"results_exp70_matched_codex_sweep_full_(equal|exp69)_seed(\d+)$")
    for path in PROJECT_DIR.glob("results_exp70_matched_codex_sweep_full_*"):
        match = pattern.match(path.name)
        if not match:
            continue
        summary = path / "summary.json"
        if not summary.exists():
            continue
        protocol = match.group(1)
        seed_start = int(match.group(2))
        data = json.loads(summary.read_text(encoding="utf-8"))
        datasets.append({
            "dataset_id": f"exponent_{protocol}_{seed_start}",
            "source": str(summary.relative_to(PROJECT_DIR)),
            "representation": "matched effective exponent geometry",
            "protocol": protocol,
            "dataset_family": "effective_exponent",
            "seed_start": seed_start,
            "X": np.asarray(data["matched_exponent_vectors"], dtype=float),
            "systems": np.asarray(data["matched_exponent_system_labels"], dtype=str),
        })
    datasets.sort(key=lambda d: (d["dataset_family"], str(d["protocol"]), str(d["seed_start"])))
    return datasets


def agglomerative_predict(Xs: np.ndarray, n_clusters: int, linkage: str):
    kwargs = {"n_clusters": n_clusters, "linkage": linkage}
    if linkage != "ward":
        try:
            return AgglomerativeClustering(metric="euclidean", **kwargs).fit_predict(Xs)
        except TypeError:
            return AgglomerativeClustering(affinity="euclidean", **kwargs).fit_predict(Xs)
    return AgglomerativeClustering(**kwargs).fit_predict(Xs)


def fixed_cluster_predictions(X: np.ndarray, n_clusters: int) -> list[tuple[str, np.ndarray | None, str]]:
    if n_clusters < 2 or n_clusters >= X.shape[0]:
        return []
    Xs = standardize(X)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        methods = [
            ("kmeans", lambda: KMeans(n_clusters=n_clusters, random_state=42, n_init=50).fit_predict(Xs)),
            ("gmm_full", lambda: GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42, n_init=10).fit_predict(Xs)),
            ("gmm_diag", lambda: GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=42, n_init=10).fit_predict(Xs)),
            ("agglomerative_ward", lambda: agglomerative_predict(Xs, n_clusters, "ward")),
            ("agglomerative_average", lambda: agglomerative_predict(Xs, n_clusters, "average")),
            ("agglomerative_complete", lambda: agglomerative_predict(Xs, n_clusters, "complete")),
            ("spectral_nearest", lambda: SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                assign_labels="kmeans",
                affinity="nearest_neighbors",
                n_neighbors=max(3, min(15, X.shape[0] // 10)),
            ).fit_predict(Xs)),
            ("spectral_rbf", lambda: SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                assign_labels="kmeans",
                affinity="rbf",
                gamma=1.0,
            ).fit_predict(Xs)),
        ]
        for name, fn in methods:
            try:
                out.append((name, np.asarray(fn(), dtype=int), "ok"))
            except Exception as exc:
                out.append((name, None, f"failed: {type(exc).__name__}: {exc}"))
    return out


def hdbscan_prediction(X: np.ndarray) -> tuple[np.ndarray | None, dict, str]:
    try:
        import hdbscan

        Xs = standardize(X)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=max(5, X.shape[0] // 20),
            min_samples=3,
        ).fit_predict(Xs)
        clusters = set(int(x) for x in labels.tolist()) - {-1}
        return labels, {
            "n_clusters_found": int(len(clusters)),
            "n_noise": int(np.sum(labels == -1)),
        }, "ok"
    except Exception as exc:
        return None, {"n_clusters_found": "", "n_noise": ""}, f"failed: {type(exc).__name__}: {exc}"


def subset_data(dataset: dict, systems_keep: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    systems = dataset["systems"]
    mask = np.isin(systems, np.asarray(systems_keep, dtype=str))
    sys = systems[mask]
    return dataset["X"][mask], sys, class_labels(sys)


def collect_clusterer_rows(datasets: list[dict]) -> list[dict]:
    rows = []
    for dataset in datasets:
        for subset_name, systems_keep, systems_display in SUBSETS:
            X, sys, cls = subset_data(dataset, systems_keep)
            if X.shape[0] == 0:
                continue
            targets = {
                "universality": cls,
                "microscopic_system": sys,
            }
            for target_name, target in targets.items():
                n_clusters = len(set(target.tolist()))
                if n_clusters < 2:
                    continue
                for method, pred, status in fixed_cluster_predictions(X, n_clusters):
                    row = base_cluster_row(dataset, subset_name, systems_display, X, cls, sys)
                    row.update({
                        "target_label_set": target_name,
                        "method": method,
                        "requested_n_clusters": n_clusters,
                        "status": status,
                    })
                    if pred is not None:
                        row.update(score_partition(pred, cls, sys))
                    rows.append(row)

            pred, extra, status = hdbscan_prediction(X)
            row = base_cluster_row(dataset, subset_name, systems_display, X, cls, sys)
            row.update({
                "target_label_set": "variable_density",
                "method": "hdbscan",
                "requested_n_clusters": "",
                "status": status,
            })
            row.update(extra)
            if pred is not None:
                row.update(score_partition(pred, cls, sys))
            rows.append(row)
    return rows


def base_cluster_row(dataset, subset_name, systems_display, X, cls, sys) -> dict:
    return {
        "dataset_id": dataset["dataset_id"],
        "source": dataset["source"],
        "dataset_family": dataset["dataset_family"],
        "representation": dataset["representation"],
        "protocol": dataset["protocol"],
        "seed_start": dataset["seed_start"],
        "subset": subset_name,
        "systems": systems_display,
        "n_samples": int(X.shape[0]),
        "n_systems": int(len(set(sys.tolist()))),
        "n_universality_classes": int(len(set(cls.tolist()))),
    }


def score_partition(pred: np.ndarray, cls: np.ndarray, sys: np.ndarray) -> dict:
    mask = pred != -1
    clusters = set(int(x) for x in pred.tolist()) - {-1}
    out = {
        "n_clusters_found": int(len(clusters)),
        "n_noise": int(np.sum(~mask)),
        "ari_universality_all": fmt(adjusted_rand_score(cls, pred)),
        "ari_system_all": fmt(adjusted_rand_score(sys, pred)),
        "nmi_universality_all": fmt(normalized_mutual_info_score(cls, pred)),
        "nmi_system_all": fmt(normalized_mutual_info_score(sys, pred)),
        "system_minus_universality_ari_all": fmt(adjusted_rand_score(sys, pred) - adjusted_rand_score(cls, pred)),
    }
    if len(clusters) >= 2 and np.sum(mask) > 1:
        out.update({
            "ari_universality_core": fmt(adjusted_rand_score(cls[mask], pred[mask])),
            "ari_system_core": fmt(adjusted_rand_score(sys[mask], pred[mask])),
            "system_minus_universality_ari_core": fmt(
                adjusted_rand_score(sys[mask], pred[mask]) - adjusted_rand_score(cls[mask], pred[mask])
            ),
        })
    else:
        out.update({
            "ari_universality_core": "",
            "ari_system_core": "",
            "system_minus_universality_ari_core": "",
        })
    return out


def summarize_clusterer_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (
            row["dataset_family"],
            row["representation"],
            row["protocol"],
            row["subset"],
            row["target_label_set"],
            row["method"],
        )
        groups[key].append(row)
    summary = []
    for key, vals in sorted(groups.items()):
        dataset_family, representation, protocol, subset, target_label_set, method = key
        uni = [fnum(v.get("ari_universality_all")) for v in vals]
        sys = [fnum(v.get("ari_system_all")) for v in vals]
        delta = [fnum(v.get("system_minus_universality_ari_all")) for v in vals]
        summary.append({
            "dataset_family": dataset_family,
            "representation": representation,
            "protocol": protocol,
            "subset": subset,
            "target_label_set": target_label_set,
            "method": method,
            "n_runs": len(vals),
            "ari_universality_mean": fmt(np.mean([x for x in uni if x is not None])),
            "ari_universality_min": fmt(np.min([x for x in uni if x is not None])),
            "ari_universality_max": fmt(np.max([x for x in uni if x is not None])),
            "ari_system_mean": fmt(np.mean([x for x in sys if x is not None])),
            "ari_system_min": fmt(np.min([x for x in sys if x is not None])),
            "ari_system_max": fmt(np.max([x for x in sys if x is not None])),
            "system_minus_universality_ari_mean": fmt(np.mean([x for x in delta if x is not None])),
        })
    return summary


def pairwise_centroid_distances(X: np.ndarray, systems: np.ndarray) -> tuple[dict[tuple[str, str], float], list[dict]]:
    Xs = standardize(X)
    centroids = {s: Xs[systems == s].mean(axis=0) for s in sorted(set(systems.tolist()))}
    distances = {}
    rows = []
    for i, a in enumerate(sorted(centroids)):
        for b in sorted(centroids)[i + 1:]:
            dist = float(np.linalg.norm(centroids[a] - centroids[b]))
            key = tuple(sorted((a, b)))
            distances[key] = dist
            rows.append({
                "system_a": a,
                "system_b": b,
                "standardized_centroid_distance": fmt(dist),
                "same_universality_class": str(a in KPZ_SYSTEMS and b in KPZ_SYSTEMS),
            })
    return distances, rows


def full_hierarchy(X: np.ndarray, systems: np.ndarray, linkage: str = "average") -> tuple[list[dict], dict]:
    unique = sorted(set(systems.tolist()))
    Xs = standardize(X)
    centroids = np.vstack([Xs[systems == s].mean(axis=0) for s in unique])
    if len(unique) < 2:
        return [], {}
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage=linkage,
            metric="euclidean",
            compute_distances=True,
        ).fit(centroids)
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            linkage=linkage,
            affinity="euclidean",
            compute_distances=True,
        ).fit(centroids)

    clusters = {i: frozenset([unique[i]]) for i in range(len(unique))}
    next_id = len(unique)
    merge_rows = []
    kpz_unified_step = None
    first_kpz_non_kpz_mix_step = None
    first_non_kpz_system = ""
    for step, (a, b) in enumerate(model.children_, start=1):
        left = clusters[int(a)]
        right = clusters[int(b)]
        merged = frozenset(set(left) | set(right))
        clusters[next_id] = merged
        next_id += 1
        contains_kpz = bool(set(merged) & KPZ_SYSTEMS)
        kpz_inside = set(merged) & KPZ_SYSTEMS
        non_kpz_inside = set(merged) - KPZ_SYSTEMS
        if kpz_unified_step is None and KPZ_SYSTEMS.issubset(merged):
            kpz_unified_step = step
        if first_kpz_non_kpz_mix_step is None and contains_kpz and non_kpz_inside:
            first_kpz_non_kpz_mix_step = step
            first_non_kpz_system = "/".join(sorted(non_kpz_inside))
        merge_rows.append({
            "merge_step": step,
            "distance": fmt(model.distances_[step - 1]),
            "left_cluster": "/".join(sorted(left)),
            "right_cluster": "/".join(sorted(right)),
            "merged_cluster": "/".join(sorted(merged)),
            "kpz_systems_inside": "/".join(sorted(kpz_inside)),
            "non_kpz_systems_inside": "/".join(sorted(non_kpz_inside)),
        })
    clean = (
        kpz_unified_step is not None
        and (first_kpz_non_kpz_mix_step is None or kpz_unified_step < first_kpz_non_kpz_mix_step)
    )
    diagnostic = {
        "hierarchy_linkage": linkage,
        "kpz_trio_unified_step": kpz_unified_step,
        "first_kpz_non_kpz_mix_step": first_kpz_non_kpz_mix_step,
        "first_non_kpz_system_mixed_with_kpz": first_non_kpz_system,
        "kpz_trio_cleanly_unified_before_non_kpz": bool(clean),
    }
    return merge_rows, diagnostic


def collect_hierarchy_rows(datasets: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    audit_rows = []
    pairwise_rows = []
    merge_rows_all = []
    for dataset in datasets:
        for subset_name, systems_keep, systems_display in SUBSETS:
            if not KPZ_SYSTEMS.issubset(set(systems_keep)):
                continue
            X, sys, cls = subset_data(dataset, systems_keep)
            if len(set(sys.tolist())) < 3:
                continue
            distances, distance_rows = pairwise_centroid_distances(X, sys)
            within = [
                distances[tuple(sorted((a, b)))]
                for i, a in enumerate(sorted(KPZ_SYSTEMS))
                for b in sorted(KPZ_SYSTEMS)[i + 1:]
                if tuple(sorted((a, b))) in distances
            ]
            cross = [
                dist for (a, b), dist in distances.items()
                if ((a in KPZ_SYSTEMS) ^ (b in KPZ_SYSTEMS))
            ]
            max_within = max(within) if within else float("nan")
            min_cross = min(cross) if cross else float("nan")
            merge_rows, diagnostic = full_hierarchy(X, sys)
            common = {
                "dataset_id": dataset["dataset_id"],
                "source": dataset["source"],
                "dataset_family": dataset["dataset_family"],
                "representation": dataset["representation"],
                "protocol": dataset["protocol"],
                "seed_start": dataset["seed_start"],
                "subset": subset_name,
                "systems": systems_display,
                "n_samples": int(X.shape[0]),
                "n_systems": int(len(set(sys.tolist()))),
            }
            audit = dict(common)
            audit.update({
                "max_within_kpz_centroid_distance": fmt(max_within),
                "min_kpz_to_non_kpz_centroid_distance": fmt(min_cross),
                "kpz_quotient_separation_ratio": fmt(max_within / min_cross if min_cross > 0 else None),
            })
            audit.update(diagnostic)
            audit_rows.append(audit)
            for row in distance_rows:
                out = dict(common)
                out.update(row)
                pairwise_rows.append(out)
            for row in merge_rows:
                out = dict(common)
                out.update(row)
                merge_rows_all.append(out)
    return audit_rows, pairwise_rows, merge_rows_all


def best_by_subset(summary_rows: list[dict], dataset_family: str, protocol: str, target: str) -> dict:
    out = {}
    for subset, _, _ in SUBSETS:
        vals = [
            fnum(r["ari_universality_mean"])
            for r in summary_rows
            if r["dataset_family"] == dataset_family
            and r["protocol"] == protocol
            and r["subset"] == subset
            and r["target_label_set"] == target
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            out[subset] = float(max(vals))
    return out


def plot_clusterer_best(summary_rows: list[dict]) -> None:
    subset_order = [s[0] for s in SUBSETS]
    labels = {
        "all_six": "all",
        "no_ks": "no KS",
        "no_rd": "no RD",
        "no_rd_no_ks": "EW/KPZ+\nBD/Eden",
        "ew_kpz_only": "EW/KPZ",
    }
    specs = [
        ("feature", "stored exp62 feature matrix", "Exp62 features", "#4c78a8"),
        ("effective_exponent", "equal", "Exponent equal", "#f58518"),
        ("effective_exponent", "exp69", "Exponent exp69", "#e45756"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(subset_order))
    width = 0.24
    for offset, (family, protocol, label, color) in zip([-width, 0, width], specs):
        vals = []
        for subset in subset_order:
            subset_vals = [
                fnum(r["ari_universality_mean"])
                for r in summary_rows
                if r["dataset_family"] == family
                and r["protocol"] == protocol
                and r["subset"] == subset
                and r["target_label_set"] == "universality"
                and r["method"] != "hdbscan"
            ]
            subset_vals = [v for v in subset_vals if v is not None]
            vals.append(max(subset_vals) if subset_vals else np.nan)
        ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.88)
    ax.set_ylim(-0.05, 1.03)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in subset_order])
    ax.set_ylabel("Best mean universality ARI across fixed-k clusterers")
    ax.set_title("Clusterer stress test: best standard method by subset")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "clusterer_control_best_ari.png", dpi=220)
    plt.close(fig)


def plot_system_vs_universality(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    styles = {
        "feature": ("#4c78a8", "o"),
        "effective_exponent": ("#f58518", "^"),
    }
    for family, (color, marker) in styles.items():
        xs, ys = [], []
        for row in rows:
            if row["status"] != "ok" or row["dataset_family"] != family:
                continue
            if row["subset"] not in {"all_six", "no_rd_no_ks"}:
                continue
            if row["target_label_set"] not in {"universality", "microscopic_system"}:
                continue
            x = fnum(row.get("ari_universality_all"))
            y = fnum(row.get("ari_system_all"))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=32, alpha=0.55, c=color, marker=marker, label=family)
    ax.plot([-0.1, 1], [-0.1, 1], color="0.25", linestyle=":", linewidth=1)
    ax.set_xlim(-0.08, 1.02)
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlabel("ARI to universality labels")
    ax.set_ylabel("ARI to microscopic-system labels")
    ax.set_title("Cluster partitions often align with microscopic identity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "system_vs_universality_alignment.png", dpi=220)
    plt.close(fig)


def plot_hierarchy(audit_rows: list[dict]) -> None:
    subset_order = ["all_six", "no_ks", "no_rd", "no_rd_no_ks"]
    labels = {
        "all_six": "all",
        "no_ks": "no KS",
        "no_rd": "no RD",
        "no_rd_no_ks": "EW/KPZ+\nBD/Eden",
    }
    specs = [
        ("feature", "stored exp62 feature matrix", "Exp62 features", "#4c78a8", "o"),
        ("effective_exponent", "equal", "Exponent equal", "#f58518", "^"),
        ("effective_exponent", "exp69", "Exponent exp69", "#e45756", "D"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(subset_order))
    for family, protocol, label, color, marker in specs:
        means, lows, highs = [], [], []
        for subset in subset_order:
            vals = [
                fnum(r["kpz_quotient_separation_ratio"])
                for r in audit_rows
                if r["dataset_family"] == family and r["protocol"] == protocol and r["subset"] == subset
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                arr = np.asarray(vals, dtype=float)
                means.append(float(arr.mean()))
                lows.append(float(arr.mean() - arr.min()))
                highs.append(float(arr.max() - arr.mean()))
            else:
                means.append(np.nan)
                lows.append(0.0)
                highs.append(0.0)
        ax.errorbar(x, means, yerr=np.asarray([lows, highs]), marker=marker, color=color, label=label, capsize=3)
    ax.axhline(1.0, color="0.25", linestyle=":", linewidth=1.2, label="quotient-friendly threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in subset_order])
    ax.set_ylabel("KPZ quotient separation ratio")
    ax.set_title("Hierarchy control: are KPZ subclasses closer than non-KPZ?")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hierarchical_kpz_merge_audit.png", dpi=220)
    plt.close(fig)


def summarize_hierarchy(audit_rows: list[dict]) -> dict:
    out = {}
    for family in sorted(set(r["dataset_family"] for r in audit_rows)):
        out[family] = {}
        protocols = sorted(set(str(r["protocol"]) for r in audit_rows if r["dataset_family"] == family))
        for protocol in protocols:
            out[family][protocol] = {}
            for subset, _, _ in SUBSETS:
                vals = [
                    fnum(r["kpz_quotient_separation_ratio"])
                    for r in audit_rows
                    if r["dataset_family"] == family and str(r["protocol"]) == protocol and r["subset"] == subset
                ]
                vals = [v for v in vals if v is not None]
                clean = [
                    str(r["kpz_trio_cleanly_unified_before_non_kpz"]).lower() == "true"
                    for r in audit_rows
                    if r["dataset_family"] == family and str(r["protocol"]) == protocol and r["subset"] == subset
                ]
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    out[family][protocol][subset] = {
                        "n": int(arr.size),
                        "ratio_mean": float(arr.mean()),
                        "ratio_min": float(arr.min()),
                        "ratio_max": float(arr.max()),
                        "clean_hierarchy_fraction": float(np.mean(clean)) if clean else None,
                    }
    return out


def main() -> None:
    ensure_dirs()
    datasets = load_datasets()
    cluster_rows = collect_clusterer_rows(datasets)
    cluster_summary = summarize_clusterer_rows(cluster_rows)
    hierarchy_rows, pairwise_rows, merge_rows = collect_hierarchy_rows(datasets)

    write_csv(TABLES_DIR / "clusterer_control_rows.csv", cluster_rows)
    write_csv(TABLES_DIR / "clusterer_control_summary.csv", cluster_summary)
    write_csv(TABLES_DIR / "hierarchical_merge_audit.csv", hierarchy_rows)
    write_csv(TABLES_DIR / "hierarchical_pairwise_distances.csv", pairwise_rows)
    write_csv(TABLES_DIR / "hierarchical_merge_steps.csv", merge_rows)
    plot_clusterer_best(cluster_summary)
    plot_system_vs_universality(cluster_rows)
    plot_hierarchy(hierarchy_rows)

    result = {
        "n_datasets": len(datasets),
        "n_clusterer_rows": len(cluster_rows),
        "n_clusterer_summary_rows": len(cluster_summary),
        "n_hierarchy_rows": len(hierarchy_rows),
        "best_fixed_clusterer_universality_ari": {
            "exp62_features": best_by_subset(cluster_summary, "feature", "stored exp62 feature matrix", "universality"),
            "exponents_equal": best_by_subset(cluster_summary, "effective_exponent", "equal", "universality"),
            "exponents_exp69": best_by_subset(cluster_summary, "effective_exponent", "exp69", "universality"),
        },
        "hierarchy_summary": summarize_hierarchy(hierarchy_rows),
        "limitations": [
            "Exp70 feature matrices were not stored, so true subset refits for those feature baselines still require a tagged rerun.",
            "These are standard off-the-shelf clusterers; a sufficiently tailored representation learner remains a separate control.",
        ],
        "outputs": [
            "ml_paper/tables/clusterer_control_rows.csv",
            "ml_paper/tables/clusterer_control_summary.csv",
            "ml_paper/tables/hierarchical_merge_audit.csv",
            "ml_paper/tables/hierarchical_pairwise_distances.csv",
            "ml_paper/tables/hierarchical_merge_steps.csv",
            "ml_paper/figures/clusterer_control_best_ari.png",
            "ml_paper/figures/system_vs_universality_alignment.png",
            "ml_paper/figures/hierarchical_kpz_merge_audit.png",
            "ml_paper/results/mlp08_summary.json",
        ],
    }
    (RESULTS_DIR / "mlp08_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
