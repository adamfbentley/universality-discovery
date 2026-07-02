"""Exp 80: does the identifiability floor transfer beyond W_sat(L) ladders?

Two demonstrations, answering the obvious referee objection to exp77 ("this
is about one observable in one domain"):

A. Growth exponent beta from W(t) ~ t^beta in the growth regime (temporal
   scaling, same systems). The time window spans more decades than the
   accessible L window, so the floor for beta should be far lower than for
   alpha at comparable noise -- which would retrodict the exp74 observation
   that BD's beta converged at sizes where its alpha stayed anomalous.

B. Correlation-length exponent nu for the 2D Ising model on the exp52d
   design (L in {32,48,64,96}), via the standard route: a pseudo-critical
   log-derivative observable scales as L^{1/nu} (1 + corrections), so
   estimating 1/nu is again a slope-under-corrections problem. exp52d
   recovered nu = 1.073 (7.3% off exact); the floor says how much of that
   was inevitable for the design.

Both reuse the exp77 confusion-gap machinery; only the design (x-grid),
noise, and declared correction class change.

Output: results_exp80_second_observable_floors/floors.json
"""

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results_exp80_second_observable_floors")
E77 = os.path.join(HERE, "77_minimax_floor.py")
E63 = os.path.join(HERE, "63_temporal_features.py")

CLASS_NOTE = ("two power corrections, omega in [0.3, 2.5], "
              "|u| <= 1 at the window start")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def floor_on_design(m77, xs, sigma, m, da_hi=0.6, tol=4e-3):
    """exp77 floor with U_BOUNDS tightened to +-1 on an arbitrary x-design."""
    m77.U_BOUNDS = (-0.75, 1.0)
    Ls = np.exp(xs)                      # m77 works in L; x = log L
    thresh = sigma ** 2 / m
    lo, hi = 0.0, da_hi
    if m77.confusion_gap(da_hi, Ls) <= thresh:
        return da_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if m77.confusion_gap(mid, Ls) <= thresh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def part_a_beta(m77, rep, n_seeds=10, L=1024, t_max=5000):
    """Growth-regime ladders: 7 log-spaced times in [50, t_max]."""
    e63 = load(E63, "e63")
    ts = np.unique(np.geomspace(50, t_max, 7).astype(int))
    logW = {s: [] for s in ("ew", "kpz")}
    for s in logW:
        fn = getattr(e63, f"simulate_{s}")
        for seed in range(n_seeds):
            tr = fn(L=L, T=t_max, seed=9000 + 17 * seed)
            if tr is None:
                continue
            W = np.std(tr, axis=1)
            logW[s].append(np.log(W[ts - 1]))
    out = {}
    for s, rows in logW.items():
        A = np.vstack(rows)
        sigma = float(np.median(A.std(axis=0, ddof=1)))
        naive = np.polyfit(np.log(ts), A.mean(axis=0), 1)[0]
        fl = {mm: floor_on_design(m77, np.log(ts), sigma, mm)
              for mm in (10, 24)}
        out[s] = {"t_points": ts.tolist(), "L": L, "n_seeds": len(rows),
                  "sigma_logW": sigma, "naive_beta": float(naive),
                  "floor_m10": fl[10], "floor_m24": fl[24]}
        print(f"[A] {s}: sigma={sigma:.3f} naive_beta={naive:.3f} "
              f"floor(m=10)={fl[10]:.3f} floor(m=24)={fl[24]:.3f}")
    rep["A_beta_growth"] = out
    rep["A_window_decades"] = float(np.log10(ts[-1] / ts[0]))


def part_b_ising_nu(m77, rep):
    """Floor for 1/nu on the exp52d design, as a function of per-point
    noise; locate the noise level at which the floor equals the observed
    exp52d deviation (0.073)."""
    xs = np.log(np.array([32., 48., 64., 96.]))
    grid = {}
    for sig in (0.005, 0.01, 0.02, 0.04):
        f = floor_on_design(m77, xs, sig, 1)   # sig = per-point sem
        grid[str(sig)] = f
        print(f"[B] ising design, sem={sig}: floor on 1/nu = {f:.3f}")
    rep["B_ising_nu"] = {
        "design_L": [32, 48, 64, 96],
        "window_decades": float(np.log10(96 / 32)),
        "floor_vs_sem": grid,
        "exp52d_observed_deviation": 0.073,
        "note": ("nu=1 so d(nu) ~ d(1/nu); exp52d recovered nu to 7.3%. "
                 "The floor at plausible sem shows how much of that "
                 "deviation was forced by the 0.48-decade design."),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    m77 = load(E77, "m77")
    m77.N_STARTS = 7
    rep = {"correction_class": CLASS_NOTE}
    part_b_ising_nu(m77, rep)      # cheap, run first
    part_a_beta(m77, rep)
    with open(os.path.join(OUT, "floors.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", os.path.join(OUT, "floors.json"))


if __name__ == "__main__":
    main()
