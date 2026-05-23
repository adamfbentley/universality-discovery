"""
MLP-02: Hard-subset audit for quotient-compatible clustering.

This derived analysis reads existing artifacts and asks whether the ML-paper
conclusions survive after removing easy or ambiguous systems. It does not run
surface-growth simulations and does not modify original results_exp* folders.

Outputs:
  ../tables/subset_ari_audit.csv
  ../tables/subset_ari_summary.csv
  ../tables/subset_ari_summary.md
  ../figures/subset_ari_audit.png
  ../results/mlp02_summary.json

Two evaluation modes are intentionally separated:
  refit_subset:
    The representation vectors are stored, so clustering is refit on the subset.
  restrict_full_partition:
    Only all-system cluster labels are stored, so ARI is computed after selecting
    the subset. This tests whether the original all-system partition respects the
    subset labels, but it is not a full subset reclustering control.
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

EXP70_REPRESENTATIONS = [
    ("single_L_features", "single-L features", "feature"),
    ("multi_L_features", "raw multi-L features", "feature"),
    ("cross_L_engineered_features", "cross-L engineered features", "feature"),
    ("pca_whitened_multi_L_features", "PCA-whitened multi-L features", "feature"),
    ("matched_exponent_geometry", "matched effective exponent geometry", "effective_exponent"),
]


def ensure_dirs() -> None:
    for path in (TABLES_DIR, FIGURES_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite_values(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def summarize(values) -> dict:
    vals = finite_values(values)
    if not vals:
        return {
            "n": 0,
            "mean": "",
            "std": "",
            "min": "",
            "max": "",
        }
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def maybe_fmt(value) -> str:
    if value == "" or value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6g}"


def json_float(value):
    value = safe_float(value)
    return value if math.isfinite(value) else None


def subset_mask(system_labels: np.ndarray, systems: list[str]) -> np.ndarray:
    return np.isin(system_labels.astype(str), np.asarray(systems, dtype=str))


def true_class_labels(system_labels: np.ndarray) -> np.ndarray:
    return np.asarray([CLASS_MAP[str(s)] for s in system_labels], dtype=object)


def cluster_refit(X: np.ndarray, y: np.ndarray) -> dict:
    n_classes = len(set(y.tolist()))
    out = {
        "kmeans_ari": float("nan"),
        "kmeans_nmi": float("nan"),
        "hdbscan_ari_all": float("nan"),
        "hdbscan_ari_core": float("nan"),
        "hdbscan_clusters": "",
        "hdbscan_noise": "",
        "knn3_accuracy": float("nan"),
        "knn3_accuracy_std": float("nan"),
    }
    if X.shape[0] == 0 or n_classes < 2:
        return out

    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20).fit_predict(Xs)
    out["kmeans_ari"] = float(adjusted_rand_score(y, km))
    out["kmeans_nmi"] = float(normalized_mutual_info_score(y, km))

    try:
        import hdbscan

        hl = hdbscan.HDBSCAN(
            min_cluster_size=max(5, X.shape[0] // 20),
            min_samples=3,
        ).fit_predict(Xs)
        mask = hl != -1
        clusters = set(int(v) for v in hl.tolist()) - {-1}
        out["hdbscan_clusters"] = int(len(clusters))
        out["hdbscan_noise"] = int(np.sum(~mask))
        out["hdbscan_ari_all"] = float(adjusted_rand_score(y, hl))
        if len(clusters) >= 2 and np.sum(mask) > 1 and len(set(y[mask].tolist())) >= 2:
            out["hdbscan_ari_core"] = float(adjusted_rand_score(y[mask], hl[mask]))
    except Exception:
        pass

    counts = np.asarray([np.sum(y == c) for c in sorted(set(y.tolist()))])
    n_splits = int(min(5, counts.min())) if counts.size else 0
    if n_splits >= 2:
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(knn, X, y, cv=cv)
        out["knn3_accuracy"] = float(scores.mean())
        out["knn3_accuracy_std"] = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    return out


def cluster_restrict(y: np.ndarray, kmeans_labels: np.ndarray, hdbscan_labels: np.ndarray) -> dict:
    out = {
        "kmeans_ari": float(adjusted_rand_score(y, kmeans_labels)) if len(set(y.tolist())) >= 2 else float("nan"),
        "kmeans_nmi": float(normalized_mutual_info_score(y, kmeans_labels)) if len(set(y.tolist())) >= 2 else float("nan"),
        "hdbscan_ari_all": float("nan"),
        "hdbscan_ari_core": float("nan"),
        "hdbscan_clusters": "",
        "hdbscan_noise": "",
        "knn3_accuracy": float("nan"),
        "knn3_accuracy_std": float("nan"),
    }
    if hdbscan_labels.size:
        mask = hdbscan_labels != -1
        clusters = set(int(v) for v in hdbscan_labels.tolist()) - {-1}
        out["hdbscan_clusters"] = int(len(clusters))
        out["hdbscan_noise"] = int(np.sum(~mask))
        out["hdbscan_ari_all"] = float(adjusted_rand_score(y, hdbscan_labels))
        if len(clusters) >= 2 and np.sum(mask) > 1 and len(set(y[mask].tolist())) >= 2:
            out["hdbscan_ari_core"] = float(adjusted_rand_score(y[mask], hdbscan_labels[mask]))
    return out


def add_row(rows: list[dict], **kwargs) -> None:
    base = {
        "source": "",
        "source_family": "",
        "protocol": "",
        "seed_start": "",
        "representation": "",
        "representation_family": "",
        "evaluation_mode": "",
        "subset": "",
        "systems": "",
        "n_samples": "",
        "n_true_classes": "",
        "kmeans_ari": "",
        "kmeans_nmi": "",
        "hdbscan_ari_all": "",
        "hdbscan_ari_core": "",
        "hdbscan_clusters": "",
        "hdbscan_noise": "",
        "knn3_accuracy": "",
        "knn3_accuracy_std": "",
        "notes": "",
    }
    base.update(kwargs)
    rows.append(base)


def rows_from_refit(
    rows: list[dict],
    *,
    source: Path,
    source_family: str,
    protocol: str,
    seed_start,
    representation: str,
    representation_family: str,
    X: np.ndarray,
    system_labels: np.ndarray,
    notes: str,
) -> None:
    for subset_name, systems, systems_display in SUBSETS:
        mask = subset_mask(system_labels, systems)
        y = true_class_labels(system_labels[mask])
        metrics = cluster_refit(X[mask], y)
        add_row(
            rows,
            source=str(source.relative_to(PROJECT_DIR)),
            source_family=source_family,
            protocol=protocol,
            seed_start=seed_start,
            representation=representation,
            representation_family=representation_family,
            evaluation_mode="refit_subset",
            subset=subset_name,
            systems=systems_display,
            n_samples=int(mask.sum()),
            n_true_classes=int(len(set(y.tolist()))),
            notes=notes,
            **metrics,
        )


def rows_from_restrict(
    rows: list[dict],
    *,
    source: Path,
    source_family: str,
    protocol: str,
    seed_start,
    representation: str,
    representation_family: str,
    system_labels: np.ndarray,
    class_labels: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    notes: str,
) -> None:
    for subset_name, systems, systems_display in SUBSETS:
        mask = subset_mask(system_labels, systems)
        y = class_labels[mask]
        metrics = cluster_restrict(y, kmeans_labels[mask], hdbscan_labels[mask])
        add_row(
            rows,
            source=str(source.relative_to(PROJECT_DIR)),
            source_family=source_family,
            protocol=protocol,
            seed_start=seed_start,
            representation=representation,
            representation_family=representation_family,
            evaluation_mode="restrict_full_partition",
            subset=subset_name,
            systems=systems_display,
            n_samples=int(mask.sum()),
            n_true_classes=int(len(set(y.tolist()))),
            notes=notes,
            **metrics,
        )


def collect_exp62(rows: list[dict]) -> None:
    path = PROJECT_DIR / "results_exp62" / "features.npz"
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["features"], dtype=float)
    systems = np.asarray(data["labels"], dtype=str)
    rows_from_refit(
        rows,
        source=path,
        source_family="exp62",
        protocol="stored feature matrix, L=128, T=1000, N=80/system",
        seed_start="",
        representation="6D spatial morphology",
        representation_family="feature",
        X=X,
        system_labels=systems,
        notes="Full subset refit from stored exp62 feature vectors.",
    )


def discover_exp70_summaries() -> list[tuple[str, int, Path]]:
    pattern = re.compile(r"results_exp70_matched_codex_sweep_full_(equal|exp69)_seed(\d+)$")
    found = []
    for path in PROJECT_DIR.glob("results_exp70_matched_codex_sweep_full_*"):
        match = pattern.match(path.name)
        if not match:
            continue
        summary = path / "summary.json"
        if summary.exists():
            found.append((match.group(1), int(match.group(2)), summary))
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def collect_exp70_sweep(rows: list[dict]) -> None:
    for protocol, seed_start, path in discover_exp70_summaries():
        data = json.loads(path.read_text(encoding="utf-8"))
        feature_systems = np.asarray(data["sample_system_labels"], dtype=str)
        feature_classes = np.asarray(data["sample_class_labels"], dtype=object)
        exponent_systems = np.asarray(data["matched_exponent_system_labels"], dtype=str)
        exponent_classes = np.asarray(data["matched_exponent_class_labels"], dtype=object)

        for key, label, family in EXP70_REPRESENTATIONS:
            block = data[key]
            if key == "matched_exponent_geometry":
                systems = exponent_systems
                classes = exponent_classes
                hdbscan_labels = np.asarray(block.get("hdbscan_labels", []), dtype=int)
                kmeans_labels = np.asarray(block["kmeans_labels"], dtype=int)
                X = np.asarray(data["matched_exponent_vectors"], dtype=float)
                rows_from_refit(
                    rows,
                    source=path,
                    source_family="exp70_sweep",
                    protocol=protocol,
                    seed_start=seed_start,
                    representation=label,
                    representation_family=family,
                    X=X,
                    system_labels=systems,
                    notes="Subset reclustering from stored matched exponent vectors.",
                )
                rows_from_restrict(
                    rows,
                    source=path,
                    source_family="exp70_sweep",
                    protocol=protocol,
                    seed_start=seed_start,
                    representation=label,
                    representation_family=family,
                    system_labels=systems,
                    class_labels=classes,
                    kmeans_labels=kmeans_labels,
                    hdbscan_labels=hdbscan_labels,
                    notes="Post-hoc restriction of all-system exponent partition; refit_subset rows are preferred.",
                )
            else:
                hdbscan_labels = np.asarray(block.get("hdbscan_labels", []), dtype=int)
                kmeans_labels = np.asarray(block["kmeans_labels"], dtype=int)
                rows_from_restrict(
                    rows,
                    source=path,
                    source_family="exp70_sweep",
                    protocol=protocol,
                    seed_start=seed_start,
                    representation=label,
                    representation_family=family,
                    system_labels=feature_systems,
                    class_labels=feature_classes,
                    kmeans_labels=kmeans_labels,
                    hdbscan_labels=hdbscan_labels,
                    notes="Only cluster labels are stored for this feature representation; this is not a subset refit.",
                )


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


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["source_family"],
            row["protocol"],
            row["representation"],
            row["representation_family"],
            row["evaluation_mode"],
            row["subset"],
            row["systems"],
        )
        groups[key].append(row)

    summary = []
    for key, vals in sorted(groups.items()):
        source_family, protocol, representation, representation_family, mode, subset, systems = key
        km = summarize([safe_float(v["kmeans_ari"]) for v in vals])
        hdb = summarize([safe_float(v["hdbscan_ari_core"]) for v in vals])
        knn = summarize([safe_float(v["knn3_accuracy"]) for v in vals])
        summary.append({
            "source_family": source_family,
            "protocol": protocol,
            "representation": representation,
            "representation_family": representation_family,
            "evaluation_mode": mode,
            "subset": subset,
            "systems": systems,
            "n_rows": len(vals),
            "n_samples_mean": maybe_fmt(np.mean([safe_float(v["n_samples"]) for v in vals])),
            "n_true_classes_mean": maybe_fmt(np.mean([safe_float(v["n_true_classes"]) for v in vals])),
            "kmeans_ari_mean": maybe_fmt(km["mean"]),
            "kmeans_ari_std": maybe_fmt(km["std"]),
            "kmeans_ari_min": maybe_fmt(km["min"]),
            "kmeans_ari_max": maybe_fmt(km["max"]),
            "hdbscan_core_ari_mean": maybe_fmt(hdb["mean"]),
            "hdbscan_core_ari_std": maybe_fmt(hdb["std"]),
            "hdbscan_core_ari_min": maybe_fmt(hdb["min"]),
            "hdbscan_core_ari_max": maybe_fmt(hdb["max"]),
            "knn3_accuracy_mean": maybe_fmt(knn["mean"]),
            "knn3_accuracy_std": maybe_fmt(knn["std"]),
            "notes": vals[0]["notes"],
        })
    return summary


def write_markdown(path: Path, rows: list[dict]) -> None:
    preferred = []
    keep = {
        ("exp70_sweep", "equal", "raw multi-L features", "restrict_full_partition"),
        ("exp70_sweep", "equal", "matched effective exponent geometry", "refit_subset"),
        ("exp70_sweep", "exp69", "raw multi-L features", "restrict_full_partition"),
        ("exp70_sweep", "exp69", "matched effective exponent geometry", "refit_subset"),
    }
    for row in rows:
        marker = (
            row["source_family"],
            row["protocol"],
            row["representation"],
            row["evaluation_mode"],
        )
        if row["source_family"] == "exp62" and row["representation"] == "6D spatial morphology":
            preferred.append(row)
        elif marker in keep:
            preferred.append(row)

    columns = [
        "source_family",
        "protocol",
        "representation",
        "evaluation_mode",
        "subset",
        "kmeans_ari_mean",
        "kmeans_ari_min",
        "kmeans_ari_max",
        "hdbscan_core_ari_mean",
        "knn3_accuracy_mean",
    ]
    lines = [
        "# MLP-02 Hard-Subset ARI Summary",
        "",
        "Rows marked `restrict_full_partition` are post-hoc restrictions of stored all-system cluster labels, not subset refits.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in preferred:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_subset_audit(summary_rows: list[dict]) -> None:
    subset_order = [s[0] for s in SUBSETS]
    x = np.arange(len(subset_order))
    labels = {
        "all_six": "all",
        "no_ks": "no KS",
        "no_rd": "no RD",
        "no_rd_no_ks": "EW/KPZ+\nBD/Eden",
        "ew_kpz_only": "EW/KPZ",
    }
    specs = [
        ("exp62", "", "6D spatial morphology", "refit_subset", "Exp62 spatial refit", "#4c78a8", "o"),
        ("exp70_sweep", "equal", "raw multi-L features", "restrict_full_partition", "Raw multi-L equal", "#72b7b2", "s"),
        ("exp70_sweep", "equal", "matched effective exponent geometry", "refit_subset", "Exponent equal refit", "#f58518", "^"),
        ("exp70_sweep", "exp69", "matched effective exponent geometry", "refit_subset", "Exponent exp69 refit", "#e45756", "D"),
    ]
    by_key = {
        (
            r["source_family"],
            r["protocol"],
            r["representation"],
            r["evaluation_mode"],
            r["subset"],
        ): r
        for r in summary_rows
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for source_family, protocol, representation, mode, label, color, marker in specs:
        means, lows, highs = [], [], []
        for subset in subset_order:
            row = by_key.get((source_family, protocol, representation, mode, subset))
            if not row:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            mean = safe_float(row["kmeans_ari_mean"])
            low = safe_float(row["kmeans_ari_min"])
            high = safe_float(row["kmeans_ari_max"])
            means.append(mean)
            lows.append(max(0.0, mean - low) if math.isfinite(low) else 0.0)
            highs.append(max(0.0, high - mean) if math.isfinite(high) else 0.0)
        ax.errorbar(
            x,
            means,
            yerr=np.asarray([lows, highs], dtype=float),
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in subset_order])
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("KMeans ARI")
    ax.set_title("Hard-subset audit: removing easy/pathological systems")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "subset_ari_audit.png", dpi=220)
    plt.close(fig)


def key_findings(summary_rows: list[dict]) -> dict:
    def lookup(source_family, protocol, representation, mode, subset, metric):
        for row in summary_rows:
            if (
                row["source_family"] == source_family
                and (protocol is None or row["protocol"] == protocol)
                and row["representation"] == representation
                and row["evaluation_mode"] == mode
                and row["subset"] == subset
            ):
                return safe_float(row[metric])
        return float("nan")

    out = {}
    for subset, _, _ in SUBSETS:
        raw_equal = lookup(
            "exp70_sweep",
            "equal",
            "raw multi-L features",
            "restrict_full_partition",
            subset,
            "kmeans_ari_mean",
        )
        exp_equal = lookup(
            "exp70_sweep",
            "equal",
            "matched effective exponent geometry",
            "refit_subset",
            subset,
            "kmeans_ari_mean",
        )
        exp69 = lookup(
            "exp70_sweep",
            "exp69",
            "matched effective exponent geometry",
            "refit_subset",
            subset,
            "kmeans_ari_mean",
        )
        spatial = lookup(
            "exp62",
            None,
            "6D spatial morphology",
            "refit_subset",
            subset,
            "kmeans_ari_mean",
        )
        delta = exp_equal - raw_equal if math.isfinite(exp_equal) and math.isfinite(raw_equal) else float("nan")
        out[subset] = {
            "exp62_spatial_refit_kmeans_ari": json_float(spatial),
            "raw_multi_L_equal_restricted_kmeans_ari": json_float(raw_equal),
            "exponent_equal_refit_kmeans_ari": json_float(exp_equal),
            "exponent_exp69_refit_kmeans_ari": json_float(exp69),
            "exponent_equal_minus_raw_multi_L_equal": json_float(delta),
        }
    return out


def main() -> None:
    ensure_dirs()
    rows: list[dict] = []
    collect_exp62(rows)
    collect_exp70_sweep(rows)
    summary_rows = aggregate_rows(rows)

    audit_csv = TABLES_DIR / "subset_ari_audit.csv"
    summary_csv = TABLES_DIR / "subset_ari_summary.csv"
    summary_md = TABLES_DIR / "subset_ari_summary.md"
    write_csv(audit_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_markdown(summary_md, summary_rows)
    plot_subset_audit(summary_rows)

    summary = {
        "n_rows": len(rows),
        "n_summary_rows": len(summary_rows),
        "subsets": [
            {"name": name, "systems": systems, "display": display}
            for name, systems, display in SUBSETS
        ],
        "evaluation_modes": {
            "refit_subset": "clustering refit after selecting subset; preferred when vectors are stored",
            "restrict_full_partition": "ARI after selecting rows from an all-system partition; not a full reclustering baseline",
        },
        "key_findings": key_findings(summary_rows),
        "outputs": [
            str(audit_csv.relative_to(PROJECT_DIR)),
            str(summary_csv.relative_to(PROJECT_DIR)),
            str(summary_md.relative_to(PROJECT_DIR)),
            str((FIGURES_DIR / "subset_ari_audit.png").relative_to(PROJECT_DIR)),
            str((RESULTS_DIR / "mlp02_summary.json").relative_to(PROJECT_DIR)),
        ],
    }
    (RESULTS_DIR / "mlp02_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
