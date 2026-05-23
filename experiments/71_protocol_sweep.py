"""
Experiment 71: Protocol Sweep for Exp70 Matched Baselines
=========================================================

Runs or aggregates repeated Exp70 tagged runs across seed starts and sampling
protocols, then writes a compact CI-style summary. This is the robustness check
that follows from the referee/Codex observation that matched exponent ARI is
protocol-sensitive.

Usage:
  python 71_protocol_sweep.py --full --run
  python 71_protocol_sweep.py --full --run --seed-starts 69000,70000,71000
  python 71_protocol_sweep.py --full --seed-starts 69000,70000,71000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
EXP70_PATH = SCRIPT_DIR / "70_matched_baseline.py"
RESULTS_DIR = PROJECT_DIR / "results_exp71_protocol_sweep"

REPRESENTATIONS = [
    "single_L_features",
    "multi_L_features",
    "cross_L_engineered_features",
    "pca_whitened_multi_L_features",
    "matched_exponent_geometry",
]

REPRESENTATIONS_KS_EXCLUDED = [
    "single_L_features_KS_excluded",
    "multi_L_features_KS_excluded",
    "cross_L_engineered_features_KS_excluded",
    "pca_whitened_multi_L_features_KS_excluded",
    "matched_exponent_geometry_KS_excluded",
]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_text(text: str) -> list[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    bad = sorted(set(vals) - {"equal", "exp69"})
    if bad:
        raise ValueError(f"Unknown protocol(s): {bad}")
    return vals


def mode_name(args) -> str:
    if args.quick:
        return "quick"
    if args.full:
        return "full"
    return "pilot"


def tag_for(prefix: str, mode: str, protocol: str, seed_start: int) -> str:
    return f"{prefix}_{mode}_{protocol}_seed{seed_start}"


def result_dir_for(tag: str) -> Path:
    return PROJECT_DIR / f"results_exp70_matched_{tag}"


def run_exp70(args, protocol: str, seed_start: int, tag: str) -> None:
    summary = result_dir_for(tag) / "summary.json"
    if summary.exists():
        print(f"skip existing {summary}")
        return

    cmd = [sys.executable, str(EXP70_PATH), "--seed-start", str(seed_start), "--out-tag", tag]
    if args.full:
        cmd.append("--full")
    if args.quick:
        cmd.append("--quick")
    if protocol == "exp69":
        cmd.append("--exp69-sampling")

    print("running", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def load_run(protocol: str, seed_start: int, tag: str) -> dict | None:
    path = result_dir_for(tag) / "summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "protocol": protocol,
        "seed_start": seed_start,
        "tag": tag,
        "path": str(path.relative_to(PROJECT_DIR)),
        "summary": data,
    }


def value(data: dict, rep: str, metric: str) -> float:
    block = data["summary"].get(rep)
    if not block:
        return float("nan")
    return float(block.get(metric, float("nan")))


def tcrit_95(n: int) -> float:
    table = {
        1: float("nan"),
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    return table.get(n, 1.96)


def summarize(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    half = float(tcrit_95(n) * sem) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half if n > 1 else None,
        "ci95_high": mean + half if n > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(x) for x in arr.tolist()],
    }


def aggregate(runs: list[dict], protocols: list[str]) -> dict:
    out: dict = {"by_protocol": {}, "runs": []}
    for run in runs:
        row = {
            "protocol": run["protocol"],
            "seed_start": run["seed_start"],
            "tag": run["tag"],
            "path": run["path"],
            "config": run["summary"].get("config", {}),
        }
        for rep in REPRESENTATIONS:
            row[f"{rep}.kmeans_ari"] = value(run, rep, "kmeans_ari")
            row[f"{rep}.hdbscan_ari_core"] = value(run, rep, "hdbscan_ari_core")
        for rep in REPRESENTATIONS_KS_EXCLUDED:
            row[f"{rep}.kmeans_ari"] = value(run, rep, "kmeans_ari")
            row[f"{rep}.hdbscan_ari_core"] = value(run, rep, "hdbscan_ari_core")
        best_feature = max(row[f"{rep}.kmeans_ari"] for rep in REPRESENTATIONS if rep != "matched_exponent_geometry")
        row["best_feature.kmeans_ari"] = best_feature
        row["matched_exponent_minus_best_feature.kmeans_ari"] = (
            row["matched_exponent_geometry.kmeans_ari"] - best_feature
        )
        out["runs"].append(row)

    for protocol in protocols:
        subset = [r for r in out["runs"] if r["protocol"] == protocol]
        block = {}
        metric_keys = [
            "best_feature.kmeans_ari",
            "matched_exponent_minus_best_feature.kmeans_ari",
        ]
        for rep in REPRESENTATIONS:
            metric_keys += [f"{rep}.kmeans_ari", f"{rep}.hdbscan_ari_core"]
        for rep in REPRESENTATIONS_KS_EXCLUDED:
            metric_keys += [f"{rep}.kmeans_ari", f"{rep}.hdbscan_ari_core"]
        for key in metric_keys:
            block[key] = summarize([r[key] for r in subset])
        out["by_protocol"][protocol] = block
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames and key != "config":
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            flat = {k: v for k, v in row.items() if k != "config"}
            w.writerow(flat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run", action="store_true", help="Run missing Exp70 tagged jobs before aggregation")
    ap.add_argument("--seed-starts", default="69000,70000,71000")
    ap.add_argument("--protocols", default="equal,exp69")
    ap.add_argument("--tag-prefix", default="codex_sweep")
    args = ap.parse_args()

    seeds = parse_csv_ints(args.seed_starts)
    protocols = parse_csv_text(args.protocols)
    mode = mode_name(args)
    start = time.time()

    RESULTS_DIR.mkdir(exist_ok=True)

    for protocol in protocols:
        for seed_start in seeds:
            tag = tag_for(args.tag_prefix, mode, protocol, seed_start)
            if args.run:
                run_exp70(args, protocol, seed_start, tag)

    runs = []
    missing = []
    for protocol in protocols:
        for seed_start in seeds:
            tag = tag_for(args.tag_prefix, mode, protocol, seed_start)
            run = load_run(protocol, seed_start, tag)
            if run is None:
                missing.append(tag)
            else:
                runs.append(run)

    summary = {
        "config": {
            "mode": mode,
            "seed_starts": seeds,
            "protocols": protocols,
            "tag_prefix": args.tag_prefix,
            "ran_missing": bool(args.run),
        },
        "elapsed_seconds": time.time() - start,
        "missing_tags": missing,
        **aggregate(runs, protocols),
    }

    out_json = RESULTS_DIR / "summary.json"
    out_csv = RESULTS_DIR / "runs.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_csv, summary["runs"])

    print(f"aggregated {len(runs)} runs; missing={len(missing)}")
    for protocol in protocols:
        block = summary["by_protocol"][protocol]
        exp = block["matched_exponent_geometry.kmeans_ari"]
        best = block["best_feature.kmeans_ari"]
        delta = block["matched_exponent_minus_best_feature.kmeans_ari"]
        print(
            f"{protocol:>5}: exponent KMeans {exp.get('mean', float('nan')):.3f} "
            f"(n={exp.get('n', 0)}), best feature {best.get('mean', float('nan')):.3f}, "
            f"delta {delta.get('mean', float('nan')):.3f}"
        )
    print(f"wrote {out_json.relative_to(PROJECT_DIR)} and {out_csv.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
