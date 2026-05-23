"""
MLP-03: Exponent-cloud instability analysis.

Reads tagged Exp70/Exp71 matched exponent summaries and generates paper-focused
tables/figures showing that effective exponent geometry is protocol-sensitive.

Outputs:
  ../tables/exponent_cloud_means.csv
  ../tables/exponent_cloud_pairwise.csv
  ../tables/exponent_quotient_metrics.csv
  ../figures/exponent_clouds_alpha_beta_by_seed.png
  ../figures/exponent_clouds_alpha_z_by_seed.png
  ../figures/exponent_ari_by_seed.png
  ../results/mlp03_summary.json
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
TABLES_DIR = ML_DIR / "tables"
FIGURES_DIR = ML_DIR / "figures"
RESULTS_DIR = ML_DIR / "results"

SYSTEMS = ["ew", "kpz", "bd", "eden", "rd", "ks"]
DISPLAY = {"ew": "EW", "kpz": "KPZ", "bd": "BD", "eden": "Eden", "rd": "RD", "ks": "KS"}
COLORS = {
    "ew": "#1f77b4",
    "kpz": "#d62728",
    "bd": "#ff7f0e",
    "eden": "#9467bd",
    "rd": "#2ca02c",
    "ks": "#111111",
}
MARKERS = {"ew": "o", "kpz": "s", "bd": "^", "eden": "D", "rd": "P", "ks": "X"}
HARD_NO_KS = ["ew", "kpz", "bd", "eden", "rd"]
KPZ_GROUP = ["kpz", "bd", "eden"]
NON_KPZ_REFERENCE = ["ew", "rd"]


def discover_runs() -> list[dict]:
    pattern = re.compile(r"results_exp70_matched_codex_sweep_full_(equal|exp69)_seed(\d+)$")
    runs = []
    for path in PROJECT_DIR.glob("results_exp70_matched_codex_sweep_full_*"):
        m = pattern.match(path.name)
        if not m:
            continue
        summary = path / "summary.json"
        if not summary.exists():
            continue
        protocol, seed_text = m.group(1), m.group(2)
        data = json.loads(summary.read_text(encoding="utf-8"))
        X = np.asarray(data["matched_exponent_vectors"], dtype=float)
        labels = np.asarray(data["matched_exponent_system_labels"])
        runs.append({
            "protocol": protocol,
            "seed_start": int(seed_text),
            "path": summary,
            "data": data,
            "X": X,
            "labels": labels,
            "kmeans_ari": float(data["matched_exponent_geometry"]["kmeans_ari"]),
            "hdbscan_core_ari": float(data["matched_exponent_geometry"]["hdbscan_ari_core"]),
            "ks_excluded_kmeans_ari": float(data["matched_exponent_geometry_KS_excluded"]["kmeans_ari"]),
        })
    runs.sort(key=lambda r: (r["protocol"], r["seed_start"]))
    return runs


def stats_for(vals: np.ndarray) -> dict:
    return {
        "n": int(vals.shape[0]),
        "alpha_mean": float(vals[:, 0].mean()),
        "beta_mean": float(vals[:, 1].mean()),
        "z_mean": float(vals[:, 2].mean()),
        "alpha_std": float(vals[:, 0].std(ddof=1)) if vals.shape[0] > 1 else 0.0,
        "beta_std": float(vals[:, 1].std(ddof=1)) if vals.shape[0] > 1 else 0.0,
        "z_std": float(vals[:, 2].std(ddof=1)) if vals.shape[0] > 1 else 0.0,
    }


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd, mu, sd


def run_centroids(run: dict, systems: list[str]) -> dict[str, np.ndarray]:
    mask = np.isin(run["labels"], systems)
    Xs, _, _ = standardize(run["X"][mask])
    labs = run["labels"][mask]
    return {s: Xs[labs == s].mean(axis=0) for s in systems if np.any(labs == s)}


def collect_tables(runs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    means, pairwise, quotient = [], [], []
    for run in runs:
        for system in SYSTEMS:
            m = run["labels"] == system
            if not np.any(m):
                continue
            row = {
                "protocol": run["protocol"],
                "seed_start": run["seed_start"],
                "system": system,
                "source": str(run["path"].relative_to(PROJECT_DIR)),
                "kmeans_ari": run["kmeans_ari"],
                "hdbscan_core_ari": run["hdbscan_core_ari"],
                "ks_excluded_kmeans_ari": run["ks_excluded_kmeans_ari"],
            }
            row.update(stats_for(run["X"][m]))
            means.append(row)

        cents = run_centroids(run, HARD_NO_KS)
        distances = {}
        for i, a in enumerate(HARD_NO_KS):
            for b in HARD_NO_KS[i + 1:]:
                if a not in cents or b not in cents:
                    continue
                dist = float(np.linalg.norm(cents[a] - cents[b]))
                distances[(a, b)] = dist
                pairwise.append({
                    "protocol": run["protocol"],
                    "seed_start": run["seed_start"],
                    "system_a": a,
                    "system_b": b,
                    "standardized_centroid_distance_no_ks": dist,
                    "source": str(run["path"].relative_to(PROJECT_DIR)),
                })

        within = [
            distances[tuple(sorted((a, b), key=HARD_NO_KS.index))]
            for i, a in enumerate(KPZ_GROUP)
            for b in KPZ_GROUP[i + 1:]
            if tuple(sorted((a, b), key=HARD_NO_KS.index)) in distances
        ]
        cross = []
        for a in KPZ_GROUP:
            for b in NON_KPZ_REFERENCE:
                key = tuple(sorted((a, b), key=HARD_NO_KS.index))
                if key in distances:
                    cross.append(distances[key])
        max_within = max(within) if within else float("nan")
        min_cross = min(cross) if cross else float("nan")
        quotient.append({
            "protocol": run["protocol"],
            "seed_start": run["seed_start"],
            "kmeans_ari": run["kmeans_ari"],
            "hdbscan_core_ari": run["hdbscan_core_ari"],
            "ks_excluded_kmeans_ari": run["ks_excluded_kmeans_ari"],
            "max_within_kpz_centroid_distance": max_within,
            "min_kpz_to_ew_or_rd_centroid_distance": min_cross,
            "kpz_quotient_separation_ratio": max_within / min_cross if min_cross > 0 else float("nan"),
            "interpretation": "good_if_below_1; lower means KPZ subclasses are tighter than separation from EW/RD",
        })
    return means, pairwise, quotient


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cloud_grid(runs: list[dict], x_idx: int, y_idx: int, filename: str,
                    x_label: str, y_label: str, include_systems: list[str]) -> None:
    protocols = ["equal", "exp69"]
    seeds = sorted({r["seed_start"] for r in runs})
    by_key = {(r["protocol"], r["seed_start"]): r for r in runs}
    fig, axes = plt.subplots(len(protocols), len(seeds), figsize=(3.2 * len(seeds), 6.4), sharex=True, sharey=True)
    if len(protocols) == 1:
        axes = np.asarray([axes])
    for row, protocol in enumerate(protocols):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            run = by_key.get((protocol, seed))
            if run is None:
                ax.axis("off")
                continue
            for system in include_systems:
                m = run["labels"] == system
                if not np.any(m):
                    continue
                ax.scatter(
                    run["X"][m, x_idx],
                    run["X"][m, y_idx],
                    s=18,
                    alpha=0.58,
                    c=COLORS[system],
                    marker=MARKERS[system],
                    linewidths=0,
                    label=DISPLAY[system],
                )
            ax.set_title(f"{protocol}, seed {seed}\nKMeans ARI={run['kmeans_ari']:.3f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{protocol}\n{y_label}")
            if row == len(protocols) - 1:
                ax.set_xlabel(x_label)
            ax.grid(alpha=0.18, linewidth=0.6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(include_systems), frameon=False)
    fig.suptitle("Effective exponent clouds by seed-start and sampling protocol", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / filename, dpi=220)
    plt.close(fig)


def plot_ari(runs: list[dict]) -> None:
    protocols = ["equal", "exp69"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for protocol in protocols:
        rs = [r for r in runs if r["protocol"] == protocol]
        seeds = [r["seed_start"] for r in rs]
        km = [r["kmeans_ari"] for r in rs]
        hdb = [r["hdbscan_core_ari"] for r in rs]
        ax.plot(seeds, km, marker="o", label=f"{protocol}: KMeans")
        ax.plot(seeds, hdb, marker="s", linestyle="--", label=f"{protocol}: HDBSCAN core")
    ax.axhline(0.6049586776859505, color="0.25", linestyle=":", label="raw multi-L KMeans baseline")
    ax.set_xlabel("seed-start")
    ax.set_ylabel("ARI")
    ax.set_title("Protocol sensitivity of matched effective-exponent geometry")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exponent_ari_by_seed.png", dpi=220)
    plt.close(fig)


def summarize_quotient(quotient_rows: list[dict]) -> dict:
    out = {}
    for protocol in sorted({r["protocol"] for r in quotient_rows}):
        vals = np.asarray([float(r["kpz_quotient_separation_ratio"]) for r in quotient_rows if r["protocol"] == protocol])
        ari = np.asarray([float(r["kmeans_ari"]) for r in quotient_rows if r["protocol"] == protocol])
        out[protocol] = {
            "n": int(vals.size),
            "ratio_mean": float(vals.mean()),
            "ratio_min": float(vals.min()),
            "ratio_max": float(vals.max()),
            "kmeans_ari_mean": float(ari.mean()),
            "kmeans_ari_min": float(ari.min()),
            "kmeans_ari_max": float(ari.max()),
        }
    return out


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    runs = discover_runs()
    if not runs:
        raise SystemExit("No exp70 codex sweep summaries found.")

    means, pairwise, quotient = collect_tables(runs)
    write_csv(TABLES_DIR / "exponent_cloud_means.csv", means)
    write_csv(TABLES_DIR / "exponent_cloud_pairwise.csv", pairwise)
    write_csv(TABLES_DIR / "exponent_quotient_metrics.csv", quotient)

    plot_cloud_grid(
        runs,
        x_idx=0,
        y_idx=1,
        filename="exponent_clouds_alpha_beta_by_seed.png",
        x_label="alpha_eff",
        y_label="beta_eff",
        include_systems=HARD_NO_KS,
    )
    plot_cloud_grid(
        runs,
        x_idx=0,
        y_idx=2,
        filename="exponent_clouds_alpha_z_by_seed.png",
        x_label="alpha_eff",
        y_label="z_eff",
        include_systems=HARD_NO_KS,
    )
    plot_ari(runs)

    summary = {
        "n_runs": len(runs),
        "protocols": sorted({r["protocol"] for r in runs}),
        "seed_starts": sorted({r["seed_start"] for r in runs}),
        "outputs": [
            "ml_paper/tables/exponent_cloud_means.csv",
            "ml_paper/tables/exponent_cloud_pairwise.csv",
            "ml_paper/tables/exponent_quotient_metrics.csv",
            "ml_paper/figures/exponent_clouds_alpha_beta_by_seed.png",
            "ml_paper/figures/exponent_clouds_alpha_z_by_seed.png",
            "ml_paper/figures/exponent_ari_by_seed.png",
        ],
        "quotient_metric_summary": summarize_quotient(quotient),
        "note": "Distances are standardized within each run after excluding KS; ratio < 1 means KPZ subclasses are closer to each other than to EW/RD.",
    }
    (RESULTS_DIR / "mlp03_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
