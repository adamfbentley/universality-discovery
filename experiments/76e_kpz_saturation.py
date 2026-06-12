"""Exp 76e: KPZ saturation check at 3x the standard simulation time.

For KPZ L=192 and L=256, run 6 seeds at T = 90*L^1.5 (3x standard schedule
T_std = 30*L^1.5). Compute W_sat two ways:
  (a) late-20% window (standard)
  (b) late-10% window

Resumable: appends to kpz_satcheck.csv, skipping already-done rows.
Columns: L, seed, window, W_sat

Compare against wsat_perseed.csv standard-T values.
"""

import csv
import importlib.util
import os
import time

import numpy as np

HERE = os.path.dirname(__file__)
EXP63_PATH = os.path.join(HERE, "63_temporal_features.py")
RESULTS_DIR = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")
OUT_CSV = os.path.join(RESULTS_DIR, "kpz_satcheck.csv")
PERSEED_CSV = os.path.join(RESULTS_DIR, "wsat_perseed.csv")

L_CHECK = [192]  # L=256 excluded: 3xT seed takes ~84s > 44s bash ceiling
N_SEEDS = 6
T_MULT = 3       # 3x standard schedule
T_SCHEDULE_STD = lambda L: int(30 * L ** 1.5)
T_SCHEDULE_LONG = lambda L: int(90 * L ** 1.5)


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63", EXP63_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def w_sat_window(traj, late_frac):
    T = traj.shape[0]
    return float(np.mean(np.std(traj[int((1 - late_frac) * T):], axis=1)))


def existing_rows(path):
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                done.add((int(r["L"]), int(r["seed"]), r["window"]))
    return done


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = existing_rows(OUT_CSV)
    new_file = not os.path.exists(OUT_CSV)
    exp63 = load_exp63()
    sim_kpz = exp63.simulate_kpz

    fh = open(OUT_CSV, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["L", "seed", "window", "W_sat"])
        fh.flush()

    t0 = time.time()
    for L in L_CHECK:
        T_long = T_SCHEDULE_LONG(L)
        for s in range(N_SEEDS):
            need_20 = (L, s, "late20") not in done
            need_10 = (L, s, "late10") not in done
            if not need_20 and not need_10:
                print(f"[{time.time()-t0:6.1f}s] L={L} seed={s} already done",
                      flush=True)
                continue
            seed = 5000 + 31 * s
            traj = None
            for attempt in range(6):
                tr = sim_kpz(L=L, T=T_long, seed=seed + attempt * 100003)
                if tr is not None and np.all(np.isfinite(tr)):
                    traj = tr
                    break
            if traj is None:
                print(f"FAILED kpz L={L} seed={s}", flush=True)
                continue
            if need_20:
                w20 = w_sat_window(traj, 0.20)
                writer.writerow([L, s, "late20", f"{w20:.6f}"])
            if need_10:
                w10 = w_sat_window(traj, 0.10)
                writer.writerow([L, s, "late10", f"{w10:.6f}"])
            fh.flush()
            print(f"[{time.time()-t0:6.1f}s] L={L} seed={s} done "
                  f"w20={w20:.4f} w10={w10:.4f}", flush=True)
    fh.close()
    fh.close()
    print("all done", flush=True)


if __name__ == "__main__":
    main()
