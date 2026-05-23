"""
Experiment 68b: Archived calibration for the collapse-metric methodology
========================================================================

Reproduces and SAVES the calibration numbers cited in
docs/path1_collapse_metric_plan.md (previously only observed in console runs),
so every claim there has a retained result artifact.

Covers:
  * EW growth exponent beta (seed-averaged variance, W-gating) -> ~0.25
  * KPZ effective beta vs lambda (crossover; lambda-dependent)
  * BD/Eden effective beta (intrinsic-width corrected) + saturation diagnostics

Read-only w.r.t. other experiments; writes only results_exp68_calibration/.
"""

from __future__ import annotations
import importlib.util
import json
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_exp68_calibration"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ens_var(simfn, T, nseeds):
    var = np.zeros(T); n = 0
    for s in range(nseeds):
        tr = simfn(s)
        if tr is not None and np.all(np.isfinite(tr)):
            var += tr.var(axis=1); n += 1
    return (var / n if n else None), n


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    e63 = load(SCRIPT_DIR / "63_temporal_features.py", "e63")
    e69 = load(SCRIPT_DIR / "69_collapse_metric_full.py", "e69")
    t0 = time.time()
    out = {"note": "Archived calibration for docs/path1_collapse_metric_plan.md",
           "method": "W = sqrt(<var_x h>_seeds); beta by W-gating (continuum) or "
                     "intrinsic-width fit (discrete).", "results": {}}

    # EW beta at L=128
    L, T, ns = 128, 10000, 16
    v, n = ens_var(lambda s: e63.simulate_ew(L=L, T=T, nu=1.0, D=1.0, seed=s), T, ns)
    b, _ = e69.beta_wgated(v, 1.0)
    out["results"]["EW_L128"] = {"beta": float(b), "n_seeds": n, "theory_beta": 0.25}
    print(f"EW   L128: beta={b:.3f} (theory 0.25, n={n})")

    # KPZ effective beta vs lambda at L=128
    out["results"]["KPZ_lambda_sweep_L128"] = {}
    for lam in [2.0, 4.0, 8.0]:
        v, n = ens_var(lambda s: e63.simulate_kpz(L=L, T=12000, nu=1.0, lam=lam, D=1.0, seed=s), 12000, 12)
        if v is None:
            out["results"]["KPZ_lambda_sweep_L128"][str(lam)] = {"beta": None, "note": "all blew up"}
            print(f"KPZ  lam={lam}: blew up"); continue
        b, _ = e69.beta_wgated(v, 1.0)
        out["results"]["KPZ_lambda_sweep_L128"][str(lam)] = {"beta": float(b), "n_seeds": n}
        print(f"KPZ  lam={lam} L128: beta={b:.3f} (effective; theory asymptotic 0.333, n={n})")

    # BD / Eden effective beta + saturation diagnostics
    for name, fn in [("BD", e69.bd_var_curve), ("Eden", e69.eden_var_curve)]:
        Ld = 96
        arr, rpm = e69.discrete_var_curves(fn, Ld, T_mono=2200, n_seed=8, seed_base=1)
        v = arr.mean(axis=0)
        b, wi2 = e69.beta_intrinsic(v, rpm)
        W = np.sqrt(v); wsat = float(np.median(W[int(0.85 * len(W)):]))
        # saturation time in monolayers (first time W >= 0.9 Wsat)
        above = np.where(W >= 0.9 * wsat)[0]
        t_sat_ml = float(above[0] / rpm) if len(above) else None
        out["results"][f"{name}_L96"] = {
            "beta_intrinsic": float(b), "intrinsic_width": float(np.sqrt(wi2)),
            "Wsat": wsat, "t_sat_monolayers": t_sat_ml, "n_seeds": arr.shape[0],
            "theory_beta": 1 / 3,
            "note": "effective; far from KPZ 1/3 at accessible L (strong corrections to scaling)",
        }
        print(f"{name:>4} L96: beta_intrinsic={b:.3f} (theory 0.333), Wi={np.sqrt(wi2):.2f}, "
              f"Wsat={wsat:.2f}, t_sat~{t_sat_ml} monolayers")

    out["elapsed_seconds"] = time.time() - t0
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR}/summary.json ({out['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
