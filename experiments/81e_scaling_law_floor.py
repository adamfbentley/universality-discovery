"""Exp 81 Part C (stretch): the reflexive corollary -- a minimax floor for
neural-scaling-law exponent fits, using the exp77 machinery UNCHANGED.

Target: L(N) = a * N^{-alpha_s} * (1 + b*N^{-omega}), a 2-decade compute
window (7 log-spaced N), noise sigma_logL in {0.01, 0.03, 0.1} (no external
data needed -- these bracket typical published-scale scatter). Reports the
resolvable Delta_alpha_s and the seed(=training-run) requirement, exactly
mirroring exp77's per-system floor table.
"""

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp81_hierarchy")
E77_PATH = os.path.join(HERE, "77_minimax_floor.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m77 = load_module(E77_PATH, "m77_scaling_law")
    m77.N_STARTS = 7  # unchanged from exp77's own default

    # 2-decade compute window, 7 log-spaced N (design is x=log(N); exp77's
    # ladder()/confusion_gap() only ever use log-values, so any N range
    # spanning 2 decades is equivalent -- 1e6..1e8 chosen as representative
    # of published compute-scan windows).
    Ns = m77.ladder(1e8, n=7, L_min=1e6)

    sigmas = (0.01, 0.03, 0.1)
    ms = (6, 24)
    table = {}
    for sig in sigmas:
        table[str(sig)] = {}
        for m in ms:
            f = m77.floor(Ns, sig, m)
            table[str(sig)][f"m{m}"] = f
            print(f"[Scaling-law floor] sigma_logL={sig} m={m}: "
                  f"resolvable Delta_alpha_s = {f:.4f}")

    report = {
        "design": {"N_min": 1e6, "N_max": 1e8, "n_points": 7,
                    "window_decades": 2.0},
        "adversary_class_unchanged_from_exp77": {
            "omega_bounds": list(m77.W_BOUNDS), "u_bounds": list(m77.U_BOUNDS)},
        "floor_vs_sigma_and_m": table,
        "note": ("Machinery identical to experiments/77_minimax_floor.py; "
                 "only the design (x=log N replacing x=log L) and the "
                 "noise grid (sigma_logL, matching typical published "
                 "compute-scan scatter) changed. Interpretation: this is a "
                 "floor for the ML community's OWN scaling-law exponent "
                 "fits, class-conditional exactly like every other floor "
                 "number in this project -- it is not a claim about any "
                 "specific published result."),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scaling_law_floor.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
