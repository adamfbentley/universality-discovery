"""Exp 76d: Classical estimators on the real 24-seed ladders.

Loads seed-mean W_sat(L) ladders from wsat_perseed.csv, runs the classical
estimators (naive, fit_w1, fit_w0p5, fit_free) from exp76, and prints:
  - per-system per-L mean W_sat ± SEM
  - 6 adjacent effective exponents of the seed-mean ladder
  - classical alpha estimates for each system x method

Output: results_exp76_amortized_extrapolation/classical_on_real.json
"""

import csv
import importlib.util
import json
import os
import warnings

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
EXP76_PATH = os.path.join(HERE, "76_amortized_extrapolation.py")
RESULTS_DIR = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")
PERSEED_CSV = os.path.join(RESULTS_DIR, "wsat_perseed.csv")
L_LADDER = np.array([32, 48, 64, 96, 128, 192, 256], dtype=float)
LOG_L = np.log(L_LADDER)


def load_exp76():
    spec = importlib.util.spec_from_file_location("exp76", EXP76_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_perseed():
    """Return {system: (n_seeds, 7) array} from wsat_perseed.csv."""
    rows = {}
    with open(PERSEED_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["system"], {}).setdefault(
                int(r["seed"]), {})[float(r["L"])] = float(r["W_sat"])
    out = {}
    for sysname, seeds in rows.items():
        mats = [np.array([m[L] for L in L_LADDER])
                for m in seeds.values() if set(m) == set(L_LADDER)]
        if mats:
            out[sysname] = np.vstack(mats)
    return out


def main():
    m = load_exp76()
    perseed = load_perseed()

    results = {}
    for sysname in ["ew", "kpz", "bd", "eden"]:
        if sysname not in perseed:
            print(f"WARNING: {sysname} not in perseed CSV", flush=True)
            continue
        Wseeds = perseed[sysname]          # (n_seeds, 7)
        n = len(Wseeds)
        Wmean = Wseeds.mean(axis=0)        # (7,)
        Wsem = Wseeds.std(axis=0, ddof=1) / np.sqrt(n)  # (7,)

        # Effective exponents of seed-mean ladder
        aeff = np.diff(np.log(Wmean)) / np.diff(LOG_L)

        # Classical fits on seed-mean ladder (single point, 1×7 array)
        cb = m.classical_fits(Wmean[None, :])
        alpha_hat = {k: float(v[0]) if np.isfinite(v[0]) else None
                     for k, v in cb.items()}

        results[sysname] = {
            "n_seeds": n,
            "L_ladder": L_LADDER.tolist(),
            "W_sat_mean": Wmean.tolist(),
            "W_sat_sem": Wsem.tolist(),
            "aeff_adjacent": aeff.tolist(),
            "classical_alpha": alpha_hat,
        }

        print(f"\n--- {sysname} (n={n} seeds) ---", flush=True)
        print("  L     W_sat_mean   W_sat_SEM", flush=True)
        for i, L in enumerate(L_LADDER):
            print(f"  {int(L):3d}   {Wmean[i]:.4f}       {Wsem[i]:.4f}", flush=True)
        print("  Adjacent eff. exponents:", flush=True)
        pairs = [(int(L_LADDER[i]), int(L_LADDER[i+1])) for i in range(6)]
        for (La, Lb), ae in zip(pairs, aeff):
            print(f"    alpha_eff({La},{Lb}) = {ae:.4f}", flush=True)
        print("  Classical alpha estimates:", flush=True)
        for k, v in alpha_hat.items():
            vstr = f"{v:.4f}" if v is not None else "FAILED"
            print(f"    {k:12s} = {vstr}", flush=True)

    out_path = os.path.join(RESULTS_DIR, "classical_on_real.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
