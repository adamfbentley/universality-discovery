"""Exp 76b: regenerate per-seed W_sat(L) ladders with more seeds.

The exp75 ladders are 6-seed means; their seed noise dominates the real-data
sanity gate for the exp76 amortized estimator (EW effective exponents swing
0.19-1.07). This script reproduces the exp75 protocol exactly (same simulators,
same T_schedule = 30 L^1.5, same late-20% W_sat estimator) but with 24 seeds,
writing one row per (system, L, seed) so downstream code can form seed-mean
ladders, SEMs, and bootstrap resamples.

Incremental + resumable: appends to the output csv, skipping rows already
present, so it can be run in bounded chunks or restarted freely. Use --out to
shard per system for parallel runs; exp76 eval merges all shard_*.csv plus
wsat_perseed.csv.

Run:  python experiments/76b_regenerate_ladders.py [--seeds 24]
          [--systems ew,kpz,bd,eden] [--out shard_ew.csv]
"""

import argparse
import csv
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(__file__)
EXP63_PATH = os.path.join(HERE, "63_temporal_features.py")
OUT_DIR = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")
L_LADDER = [32, 48, 64, 96, 128, 192, 256]
T_SCHEDULE = lambda L: int(30 * L ** 1.5)


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63", EXP63_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def w_sat(traj):
    T = traj.shape[0]
    return float(np.mean(np.std(traj[int(0.8 * T):], axis=1)))


def existing_rows(path):
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                done.add((r["system"], int(r["L"]), int(r["seed"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--systems", default="ew,kpz,bd,eden")
    ap.add_argument("--out", default="wsat_perseed.csv",
                    help="output csv name (per-system shard for parallel runs)")
    args = ap.parse_args()
    systems = args.systems.split(",")
    out_csv = os.path.join(OUT_DIR, args.out)

    os.makedirs(OUT_DIR, exist_ok=True)
    done = existing_rows(out_csv)
    new_file = not os.path.exists(out_csv)
    exp63 = load_exp63()
    sims = {s: getattr(exp63, f"simulate_{s}") for s in systems}

    fh = open(out_csv, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["system", "L", "seed", "W_sat"])
        fh.flush()

    t0 = time.time()
    for sysname in systems:
        for L in L_LADDER:
            T = T_SCHEDULE(L)
            for s in range(args.seeds):
                if (sysname, L, s) in done:
                    continue
                seed = 5000 + 31 * s
                traj = None
                for attempt in range(6):
                    tr = sims[sysname](L=L, T=T, seed=seed + attempt * 100003)
                    if tr is not None and np.all(np.isfinite(tr)):
                        traj = tr
                        break
                if traj is None:
                    print(f"FAILED {sysname} L={L} seed={s}", flush=True)
                    continue
                writer.writerow([sysname, L, s, f"{w_sat(traj):.6f}"])
                fh.flush()
            print(f"[{time.time()-t0:7.1f}s] {sysname} L={L} done", flush=True)
    fh.close()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
