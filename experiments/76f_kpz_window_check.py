"""Exp 76f: KPZ window check at standard T = 30*L^1.5.

Run one seed per invocation (--L, --seed CLI args). Records W_sat from three
windows (late-40%, late-20%, late-10%) and late_beta = slope of log W(t) vs
log t over the last 20% of frames, into kpz_window.csv (append, resumable).

late_beta near 0 + window-stable W_sat => saturated
late_beta > 0.02 or late-10%/late-40% W_sat ratio differs > 1 SEM => not saturated

Usage:
    python experiments/76f_kpz_window_check.py --L 192 --seed 0
"""

import argparse
import csv
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(__file__)
EXP63_PATH = os.path.join(HERE, "63_temporal_features.py")
RESULTS_DIR = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")
OUT_CSV = os.path.join(RESULTS_DIR, "kpz_window.csv")

T_SCHEDULE = lambda L: int(30 * L ** 1.5)


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63", EXP63_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def w_sat_window(traj, late_frac):
    """Mean std over the last late_frac fraction of trajectory frames."""
    T = traj.shape[0]
    return float(np.mean(np.std(traj[int((1 - late_frac) * T):], axis=1)))


def late_beta(traj, late_frac=0.20):
    """Slope of log W(t) vs log t over the last late_frac of frames.
    W(t) = std over lattice at time t. Positive => still growing."""
    T = traj.shape[0]
    t0 = int((1 - late_frac) * T)
    ts = np.arange(t0, T)
    Wt = np.std(traj[t0:], axis=1)          # (n_frames,)
    mask = (ts > 0) & (Wt > 0)
    if mask.sum() < 4:
        return float("nan")
    log_t = np.log(ts[mask].astype(float))
    log_W = np.log(Wt[mask])
    slope = np.polyfit(log_t, log_W, 1)[0]
    return float(slope)


def already_done(L, seed):
    if not os.path.exists(OUT_CSV):
        return False
    with open(OUT_CSV) as fh:
        for r in csv.DictReader(fh):
            if int(r["L"]) == L and int(r["seed"]) == seed:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    L, seed_idx = args.L, args.seed

    if already_done(L, seed_idx):
        print(f"L={L} seed={seed_idx} already in {OUT_CSV}, skipping.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    exp63 = load_exp63()
    sim_kpz = exp63.simulate_kpz

    T = T_SCHEDULE(L)
    rng_seed = 5000 + 31 * seed_idx

    traj = None
    for attempt in range(6):
        tr = sim_kpz(L=L, T=T, seed=rng_seed + attempt * 100003)
        if tr is not None and np.all(np.isfinite(tr)):
            traj = tr
            break

    if traj is None:
        print(f"FAILED kpz L={L} seed={seed_idx}")
        return

    w40 = w_sat_window(traj, 0.40)
    w20 = w_sat_window(traj, 0.20)
    w10 = w_sat_window(traj, 0.10)
    lb   = late_beta(traj, 0.20)

    new_file = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["system", "L", "seed",
                             "w_late40", "w_late20", "w_late10", "late_beta"])
        writer.writerow(["kpz", L, seed_idx,
                         f"{w40:.6f}", f"{w20:.6f}", f"{w10:.6f}", f"{lb:.6f}"])

    print(f"kpz L={L} seed={seed_idx}: "
          f"w40={w40:.4f} w20={w20:.4f} w10={w10:.4f} late_beta={lb:.4f}")


if __name__ == "__main__":
    main()
