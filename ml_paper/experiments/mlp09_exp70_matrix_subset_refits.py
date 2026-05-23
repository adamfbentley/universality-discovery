"""
MLP-09: True subset refits for Exp70 feature matrices.

MLP-02 could only post-hoc restrict stored Exp70 feature cluster labels because
the original artifacts did not archive feature matrices. Exp70 now writes
`matched_matrices.npz`; this script reads the new tagged matrix runs and refits
KMeans/HDBSCAN on hard subsets for each representation.

Inputs:
  results_exp70_matched_codex_matrix_full_{equal,exp69}_seed*/matched_matrices.npz
  corresponding summary.json

Outputs:
  ../tables/exp70_matrix_subset_refit_rows.csv
  ../tables/exp70_matrix_subset_refit_summary.csv
  ../figures/exp70_matrix_subset_refits.png
  ../results/mlp09_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
TABLES_DIR = ML_DIR / "tables"
FIGURES_DIR = ML_DIR / "figures"
RESULTS_DIR = ML_DIR / "results"

SUBSETS = [
    ("all_six", ["ew", "kpz", "bd", "eden", "rd", "ks"], "EW/KPZ/BD/Eden/RD/KS"),
    ("no_ks", ["ew", "kpz", "bd", "eden", "rd"], "EW/KPZ/BD/Eden/RD"),
    ("no_rd", ["ew", "kpz", "bd", "eden", "ks"], "EW/KPZ/BD/Eden/KS"),
    ("no_rd_no_ks", ["ew", "kpz", "bd", "eden"], "EW/KPZ/BD/Eden"),
    ("ew_kpz_only", ["ew", "kpz"], "EW/KPZ only"),
]

FEATURE_REPS = [
    ("single_L_features", "single-L features", "feature"),
    ("multi_L_features", "raw multi-L features", "feature"),
    ("cross_L_engineered_features", "cross-L engineered features", "feature"),
    ("pca_whitened_multi_L_features", "PCA-whitened multi-L features", "feature"),
]
ALL_REPS = FEATURE_REPS + [
    ("matched_exponent_geometry", "matched effective exponent geometry", "effective_exponent"),
]


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


def discover_runs() -> list[dict]:
    pattern = re.compile(r"results_exp70_matched_codex_matrix_full_(equal|exp69)_seed(\d+)$")
    runs = []
    for path in PROJECT_DIR.glob("results_exp70_matched_codex_matrix_full_*_seed*"):
        match = pattern.match(path.name)
        if not match:
            continue
        matrix = path / "matched_matrices.npz"
        summary = path / "summary.json"
        if not matrix.exists() or not summary.exists():
            continue
        runs.append({
            "protocol": match.group(1),
            "seed_start": int(match.group(2)),
            "dir": path,
            "matrix": matrix,
            "summary": summary,
        })
    runs.sort(key=lambda r: (r["protocol"], r["seed_start"]))
    return runs


def subset_mask(systems: np.ndarray, subset_systems: list[str]) -> np.ndarray:
    return np.isin(systems.astype(str), np.asarray(subset_systems, dtype=str))


def cluster_refit(X: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n_classes = len(set(labels.tolist()))
    out = {
        "kmeans_ari": "",
        "kmeans_nmi": "",
        "hdbscan_ari_all": "",
        "hdbscan_ari_core": "",
        "hdbscan_clusters": "",
        "hdbscan_noise": "",
    }
    if X.shape[0] < 2 or n_classes < 2:
        return out
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=50).fit_predict(Xs)
    out["kmeans_ari"] = fmt(adjusted_rand_score(labels, km))
    out["kmeans_nmi"] = fmt(normalized_mutual_info_score(labels, km))

    try:
        import hdbscan

        hl = hdbscan.HDBSCAN(
            min_cluster_size=max(5, X.shape[0] // 20),
            min_samples=3,
        ).fit_predict(Xs)
        mask = hl != -1
        clusters = set(int(x) for x in hl.tolist()) - {-1}
        out["hdbscan_clusters"] = int(len(clusters))
        out["hdbscan_noise"] = int(np.sum(~mask))
        out["hdbscan_ari_all"] = fmt(adjusted_rand_score(labels, hl))
        if len(clusters) >= 2 and np.sum(mask) > 1 and len(set(labels[mask].tolist())) >= 2:
            out["hdbscan_ari_core"] = fmt(adjusted_rand_score(labels[mask], hl[mask]))
    except Exception:
        pass
    return out


def collect_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        data = np.load(run["matrix"], allow_pickle=True)
        for key, label, family in ALL_REPS:
            X = np.asarray(data[key], dtype=float)
            if key == "matched_exponent_geometry":
                systems = np.asarray(data["matched_exponent_system_labels"], dtype=str)
                classes = np.asarray(data["matched_exponent_class_labels"], dtype=str)
            else:
                systems = np.asarray(data["sample_system_labels"], dtype=str)
                classes = np.asarray(data["sample_class_labels"], dtype=str)
            for subset_name, subset_systems, systems_display in SUBSETS:
                mask = subset_mask(systems, subset_systems)
                metrics = cluster_refit(X[mask], classes[mask])
                rows.append({
                    "source": str(run["matrix"].relative_to(PROJECT_DIR)),
                    "protocol": run["protocol"],
                    "seed_start": run["seed_start"],
                    "representation": label,
                    "representation_key": key,
                    "representation_family": family,
                    "subset": subset_name,
                    "systems": systems_display,
                    "n_samples": int(np.sum(mask)),
                    "n_features": int(X.shape[1]) if X.ndim == 2 else "",
                    "n_true_classes": int(len(set(classes[mask].tolist()))),
                    **metrics,
                })
    return rows


def summarize(values: list) -> dict:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": fmt(arr.mean()),
        "std": fmt(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": fmt(arr.min()),
        "max": fmt(arr.max()),
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["protocol"],
            row["representation"],
            row["representation_key"],
            row["representation_family"],
            row["subset"],
            row["systems"],
        )
        groups[key].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        protocol, representation, representation_key, family, subset, systems = key
        km = summarize([v["kmeans_ari"] for v in vals])
        hdb = summarize([v["hdbscan_ari_core"] for v in vals])
        out.append({
            "protocol": protocol,
            "representation": representation,
            "representation_key": representation_key,
            "representation_family": family,
            "subset": subset,
            "systems": systems,
            "n_runs": len(vals),
            "n_samples_mean": fmt(np.mean([int(v["n_samples"]) for v in vals])),
            "kmeans_ari_mean": km["mean"],
            "kmeans_ari_std": km["std"],
            "kmeans_ari_min": km["min"],
            "kmeans_ari_max": km["max"],
            "hdbscan_core_ari_mean": hdb["mean"],
            "hdbscan_core_ari_std": hdb["std"],
            "hdbscan_core_ari_min": hdb["min"],
            "hdbscan_core_ari_max": hdb["max"],
        })
    return out


def best_feature_summary(summary_rows: list[dict]) -> dict:
    out = {}
    for protocol in sorted(set(r["protocol"] for r in summary_rows)):
        out[protocol] = {}
        for subset, _, _ in SUBSETS:
            features = [
                r for r in summary_rows
                if r["protocol"] == protocol
                and r["subset"] == subset
                and r["representation_family"] == "feature"
            ]
            exponent = [
                r for r in summary_rows
                if r["protocol"] == protocol
                and r["subset"] == subset
                and r["representation_key"] == "matched_exponent_geometry"
            ]
            feature_vals = [(fnum(r["kmeans_ari_mean"]), r["representation"]) for r in features]
            feature_vals = [(v, name) for v, name in feature_vals if v is not None]
            exp_val = fnum(exponent[0]["kmeans_ari_mean"]) if exponent else None
            if feature_vals and exp_val is not None:
                best_val, best_name = max(feature_vals, key=lambda x: x[0])
                out[protocol][subset] = {
                    "best_feature_representation": best_name,
                    "best_feature_kmeans_ari_mean": best_val,
                    "matched_exponent_kmeans_ari_mean": exp_val,
                    "exponent_minus_best_feature": exp_val - best_val,
                }
    return out


def plot_summary(summary_rows: list[dict]) -> None:
    subset_order = [s[0] for s in SUBSETS]
    subset_labels = {
        "all_six": "all",
        "no_ks": "no KS",
        "no_rd": "no RD",
        "no_rd_no_ks": "EW/KPZ+\nBD/Eden",
        "ew_kpz_only": "EW/KPZ",
    }
    reps = [
        ("single_L_features", "single-L", "#4c78a8"),
        ("multi_L_features", "multi-L", "#72b7b2"),
        ("cross_L_engineered_features", "cross-L", "#b279a2"),
        ("matched_exponent_geometry", "exponents", "#f58518"),
    ]
    protocols = ["equal", "exp69"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    by_key = {
        (r["protocol"], r["representation_key"], r["subset"]): r
        for r in summary_rows
    }
    x = np.arange(len(subset_order))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(reps))
    for ax, protocol in zip(axes, protocols):
        for offset, (rep, label, color) in zip(offsets, reps):
            means, lows, highs = [], [], []
            for subset in subset_order:
                row = by_key.get((protocol, rep, subset))
                if not row:
                    means.append(np.nan)
                    lows.append(0.0)
                    highs.append(0.0)
                    continue
                mean = fnum(row["kmeans_ari_mean"])
                lo = fnum(row["kmeans_ari_min"])
                hi = fnum(row["kmeans_ari_max"])
                means.append(mean)
                lows.append(max(0.0, mean - lo) if mean is not None and lo is not None else 0.0)
                highs.append(max(0.0, hi - mean) if mean is not None and hi is not None else 0.0)
            ax.bar(x + offset, means, width=width, label=label, color=color, alpha=0.9)
            ax.errorbar(x + offset, means, yerr=np.asarray([lows, highs]), fmt="none", color="0.2", linewidth=0.8, capsize=2)
        ax.set_title(f"protocol={protocol}")
        ax.set_xticks(x)
        ax.set_xticklabels([subset_labels[s] for s in subset_order])
        ax.set_ylim(-0.08, 1.03)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("KMeans ARI after true subset refit")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Exp70 matrix archive: true feature/exponent subset refits", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "exp70_matrix_subset_refits.png", dpi=220)
    plt.close(fig)


def write_markdown_summary(path: Path, comparison: dict) -> None:
    columns = [
        "protocol",
        "subset",
        "best_feature_representation",
        "best_feature_kmeans_ari_mean",
        "matched_exponent_kmeans_ari_mean",
        "exponent_minus_best_feature",
    ]
    lines = [
        "# Exp70 Matrix Subset Refit Summary",
        "",
        "These rows use archived feature matrices and refit KMeans after selecting each subset.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for protocol in sorted(comparison):
        for subset in [s[0] for s in SUBSETS]:
            row = comparison[protocol].get(subset)
            if not row:
                continue
            vals = {
                "protocol": protocol,
                "subset": subset,
                **row,
            }
            lines.append("| " + " | ".join(str(vals.get(c, "")) for c in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    runs = discover_runs()
    rows = collect_rows(runs)
    summary_rows = summarize_rows(rows)
    comparison = best_feature_summary(summary_rows)
    write_csv(TABLES_DIR / "exp70_matrix_subset_refit_rows.csv", rows)
    write_csv(TABLES_DIR / "exp70_matrix_subset_refit_summary.csv", summary_rows)
    write_markdown_summary(TABLES_DIR / "exp70_matrix_subset_refit_summary.md", comparison)
    plot_summary(summary_rows)
    result = {
        "n_runs": len(runs),
        "run_tags": [r["dir"].name for r in runs],
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "best_feature_vs_exponent": comparison,
        "outputs": [
            "ml_paper/tables/exp70_matrix_subset_refit_rows.csv",
            "ml_paper/tables/exp70_matrix_subset_refit_summary.csv",
            "ml_paper/tables/exp70_matrix_subset_refit_summary.md",
            "ml_paper/figures/exp70_matrix_subset_refits.png",
            "ml_paper/results/mlp09_summary.json",
        ],
    }
    (RESULTS_DIR / "mlp09_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
