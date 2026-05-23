"""
MLP-01: Consolidate representation evidence for the ML-focused paper.

This script reads existing result artifacts and writes a single table comparing
local class-informativeness, cluster compatibility, and quotient-compatibility
evidence. It does not run simulations or modify original results_exp* artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
PROJECT_DIR = ML_DIR.parent
TABLES_DIR = ML_DIR / "tables"
RESULTS_DIR = ML_DIR / "results"


def read_json(path: str) -> dict:
    return json.loads((PROJECT_DIR / path).read_text(encoding="utf-8"))


def read_csv(path: str) -> list[dict]:
    with (PROJECT_DIR / path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value):
    if value is None or value == "":
        return None
    return float(value)


def knn_accuracy(block):
    if block is None:
        return None
    if isinstance(block, dict):
        for key in ("accuracy", "mean_accuracy", "knn3_accuracy"):
            if key in block:
                return block[key]
        return None
    return block


def summarize(values: list[float]) -> dict:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "std": stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def add_row(rows: list[dict], **kwargs) -> None:
    defaults = {
        "source": "",
        "representation": "",
        "systems": "EW/KPZ/BD/Eden/RD/KS",
        "protocol": "",
        "representation_family": "",
        "n_runs": "",
        "hdbscan_core_ari_mean": "",
        "hdbscan_core_ari_std": "",
        "kmeans_ari_mean": "",
        "kmeans_ari_std": "",
        "knn3_accuracy_mean": "",
        "knn3_accuracy_std": "",
        "local_global_gap": "",
        "quotient_compatibility": "",
        "notes": "",
    }
    defaults.update(kwargs)
    rows.append(defaults)


def fmt(x):
    if x is None or x == "":
        return ""
    return f"{float(x):.6g}"


def row_from_summary(rows, *, source, representation, protocol, family, hdb=None,
                     km=None, knn=None, systems="EW/KPZ/BD/Eden/RD/KS",
                     quotient="", notes=""):
    gap = None
    if knn and hdb and knn["mean"] is not None and hdb["mean"] is not None:
        gap = knn["mean"] - hdb["mean"]
    add_row(
        rows,
        source=source,
        representation=representation,
        systems=systems,
        protocol=protocol,
        representation_family=family,
        n_runs=max((hdb or {}).get("n", 0), (km or {}).get("n", 0), (knn or {}).get("n", 0)),
        hdbscan_core_ari_mean=fmt((hdb or {}).get("mean")),
        hdbscan_core_ari_std=fmt((hdb or {}).get("std")),
        kmeans_ari_mean=fmt((km or {}).get("mean")),
        kmeans_ari_std=fmt((km or {}).get("std")),
        knn3_accuracy_mean=fmt((knn or {}).get("mean")),
        knn3_accuracy_std=fmt((knn or {}).get("std")),
        local_global_gap=fmt(gap),
        quotient_compatibility=quotient,
        notes=notes,
    )


def collect_exp62(rows):
    path = "results_exp62/results.json"
    d = read_json(path)
    row_from_summary(
        rows,
        source=path,
        representation="6D spatial morphology",
        protocol=f"L={d['config']['L']}, T={d['config']['T']}, N={d['config']['N']}",
        family="spatial",
        hdb=summarize([d["hdbscan"]["ari"]]),
        km=summarize([d["kmeans"]["ari"]]),
        knn=summarize([knn_accuracy(d.get("knn_3"))]),
        quotient="fails",
        notes="Baseline feature-space HDBSCAN ceiling; KMeans much worse.",
    )


def collect_exp65(rows):
    path = "results_exp65_robustness/summary.json"
    d = read_json(path)
    by_rep = {}
    for row in d["multi_seed_pilot"]["rows"]:
        by_rep.setdefault(row["representation"], []).append(row)
    for rep, rep_rows in sorted(by_rep.items()):
        row_from_summary(
            rows,
            source=path,
            representation=rep,
            protocol="5 seed blocks, L=128, T=500",
            family="robustness",
            hdb=summarize([r["hdbscan_non_noise_ari"] for r in rep_rows]),
            km=summarize([r["kmeans_ari"] for r in rep_rows]),
            knn=summarize([r["knn3_accuracy"] for r in rep_rows]),
            quotient="fails",
            notes="Seed robustness check; local kNN can be high while HDBSCAN remains near ceiling.",
        )


def collect_exp66(rows):
    path = "results_exp66_finite_size/sweep_rows.csv"
    data = read_csv(path)
    by_rep = {}
    for row in data:
        by_rep.setdefault(row["representation"], []).append(row)
    for rep, rep_rows in sorted(by_rep.items()):
        row_from_summary(
            rows,
            source=path,
            representation=rep,
            protocol="finite-size sweep L=64..256, 3 seeds/size",
            family="finite-size",
            hdb=summarize([fnum(r["hdbscan_core_ari"]) for r in rep_rows]),
            km=summarize([fnum(r["kmeans_ari"]) for r in rep_rows]),
            knn=summarize([fnum(r["knn3_accuracy"]) for r in rep_rows]),
            quotient="fails",
            notes="Accessible-size sweep preserves local/global separation.",
        )


def collect_exp67(rows):
    path = "results_exp67_height_dist/representation_rows.csv"
    data = read_csv(path)
    by_rep = {}
    for row in data:
        by_rep.setdefault(row["representation"], []).append(row)
    for rep, rep_rows in sorted(by_rep.items()):
        row_from_summary(
            rows,
            source=path,
            representation=rep,
            protocol="height/TW feature audit, 5 seed blocks",
            family="height_distribution",
            hdb=summarize([fnum(r["hdbscan_core_ari"]) for r in rep_rows]),
            km=summarize([fnum(r["kmeans_ari"]) for r in rep_rows]),
            knn=summarize([fnum(r["knn3_accuracy"]) for r in rep_rows]),
            quotient="fails",
            notes="Height moments improve some separations but do not restore global quotient clustering.",
        )


def collect_exp69(rows):
    path = "results_exp69_collapse_full/summary.json"
    d = read_json(path)
    for name, label, family, quotient, notes in [
        ("feature_geometry", "single-L Exp63 feature geometry", "feature", "fails",
         "Information-asymmetric exp69 feature baseline."),
        ("exponent_geometry", "effective exponent geometry", "effective_exponent", "single_protocol_success",
         "High-ARI single-protocol result; later shown protocol-sensitive by exp71."),
    ]:
        block = d["head_to_head_all6_4class"][name]
        row_from_summary(
            rows,
            source=path,
            representation=label,
            protocol="exp69 all-six 4-class head-to-head",
            family=family,
            hdb=summarize([block["hdbscan_ari_core"]]),
            km=summarize([block["kmeans_ari"]]),
            quotient=quotient,
            notes=notes,
        )


def collect_exp71(rows):
    path = "results_exp71_protocol_sweep/summary.json"
    d = read_json(path)
    mapping = {
        "single_L_features": ("single-L features", "feature"),
        "multi_L_features": ("raw multi-L features", "multi_L_feature"),
        "cross_L_engineered_features": ("cross-L engineered features", "multi_L_feature"),
        "pca_whitened_multi_L_features": ("PCA-whitened multi-L features", "multi_L_feature"),
        "matched_exponent_geometry": ("matched effective exponent geometry", "effective_exponent"),
    }
    for protocol, block in d["by_protocol"].items():
        for key, (label, family) in mapping.items():
            row_from_summary(
                rows,
                source=path,
                representation=label,
                protocol=f"exp71 five seed-start sweep, protocol={protocol}",
                family=family,
                hdb=block[f"{key}.hdbscan_ari_core"],
                km=block[f"{key}.kmeans_ari"],
                quotient="not_robust",
                notes="Repeated matched protocol sweep; exponent geometry ties best feature baseline on average.",
            )
        delta = block["matched_exponent_minus_best_feature.kmeans_ari"]
        add_row(
            rows,
            source=path,
            representation="exponent minus best feature",
            protocol=f"exp71 five seed-start sweep, protocol={protocol}",
            representation_family="comparison",
            n_runs=delta["n"],
            kmeans_ari_mean=fmt(delta["mean"]),
            kmeans_ari_std=fmt(delta["std"]),
            quotient_compatibility="no_advantage",
            notes="Mean KMeans advantage of matched exponent geometry over best feature baseline.",
        )


def collect_positive_controls(rows):
    ising_path = "results_exp52d_full/results.json"
    potts_path = "results_exp57c_pilot/summary.json"
    ising = read_json(ising_path)
    potts = read_json(potts_path)
    add_row(
        rows,
        source=ising_path,
        representation="Ising PCA/FSS positive control",
        systems="2D Ising",
        protocol="L=32..96 finite-size scaling",
        representation_family="positive_control",
        quotient_compatibility="works_on_clean_control",
        notes=f"nu={ising['results']['nu_optimal']:.3f}, deviation={ising['results']['deviation_percent']:.2f}%.",
    )
    add_row(
        rows,
        source=potts_path,
        representation="Potts Binder positive control",
        systems="3-state Potts",
        protocol="pilot Binder derivative",
        representation_family="positive_control",
        quotient_compatibility="works_on_clean_control",
        notes=f"nu={potts['nu_fit']:.3f}, error={potts['error_pct']:.2f}%.",
    )


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    cols = [
        "representation",
        "protocol",
        "hdbscan_core_ari_mean",
        "hdbscan_core_ari_std",
        "kmeans_ari_mean",
        "kmeans_ari_std",
        "knn3_accuracy_mean",
        "knn3_accuracy_std",
        "local_global_gap",
        "quotient_compatibility",
    ]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    rows: list[dict] = []
    collect_exp62(rows)
    collect_exp65(rows)
    collect_exp66(rows)
    collect_exp67(rows)
    collect_exp69(rows)
    collect_exp71(rows)
    collect_positive_controls(rows)

    csv_path = TABLES_DIR / "representation_performance.csv"
    md_path = TABLES_DIR / "representation_performance.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    summary = {
        "n_rows": len(rows),
        "outputs": [str(csv_path.relative_to(PROJECT_DIR)), str(md_path.relative_to(PROJECT_DIR))],
        "message": "Consolidated evidence table for quotient-label ML paper.",
    }
    (RESULTS_DIR / "mlp01_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
