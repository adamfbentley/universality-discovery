"""Exp 77: Le Cam minimax resolution floor for FSS exponent estimation.

Companion to ml_paper/THEORY_minimax_floor.md. Computes, by direct adversarial
optimization, the confusion gap

    D^2(da) = min_{c, u1, w1, u2, w2} sum_i [ da*x_i + c
                + log(1 + u1*(L_i/L_1)^{-w1}) - log(1 + u2*(L_i/L_1)^{-w2}) ]^2

(u = correction size at L_1, so positivity is controlled), and the resolution
floor  da*(design, sigma, m) = max{ da : D^2 <= sigma^2/m }  via bisection.

Outputs (results_exp77_minimax_floor/floor.json). Parts are mergeable:
run --part sys / law / m in separate calls; results accumulate.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp77_minimax_floor")
PERSEED_CSV = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation",
                           "wsat_perseed.csv")

W_BOUNDS = (0.3, 2.5)
U_BOUNDS = (-0.75, 4.0)
N_STARTS = 7
RNG = np.random.default_rng(77)


def ladder(L_max, n=7, L_min=32):
    return np.exp(np.linspace(np.log(L_min), np.log(L_max), n))


def corr(L, u, w, L1):
    return np.log(1.0 + u * (L / L1) ** (-w))


def confusion_gap(da, Ls):
    x = np.log(Ls)
    L1 = Ls[0]

    def obj(p):
        c, u1, w1, u2, w2 = p
        diff = da * x + c + corr(Ls, u1, w1, L1) - corr(Ls, u2, w2, L1)
        return np.sum(diff ** 2)

    best = np.inf
    bounds = [(None, None), U_BOUNDS, W_BOUNDS, U_BOUNDS, W_BOUNDS]
    for k in range(N_STARTS):
        p0 = np.array([
            -da * x.mean(),
            RNG.uniform(*U_BOUNDS), RNG.uniform(*W_BOUNDS),
            RNG.uniform(*U_BOUNDS), RNG.uniform(*W_BOUNDS),
        ])
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds)
        if res.fun < best:
            best = float(res.fun)
    return best


def floor(Ls, sigma, m, da_hi=0.6, tol=2.5e-3):
    thresh = sigma ** 2 / m
    lo, hi = 0.0, da_hi
    if confusion_gap(da_hi, Ls) <= thresh:
        return da_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if confusion_gap(mid, Ls) <= thresh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def linearized_floor(Ls, sigma, m, n_w_grid=40):
    x = np.log(Ls)
    L1 = Ls[0]
    ws = np.linspace(W_BOUNDS[0], W_BOUNDS[1], n_w_grid)
    basis = [np.ones_like(x)] + [(Ls / L1) ** (-w) for w in ws]
    A = np.vstack(basis).T
    Q, _ = np.linalg.qr(A)
    x_perp = x - Q @ (Q.T @ x)
    nrm = np.linalg.norm(x_perp)
    return float(sigma / (np.sqrt(m) * nrm)), float(nrm)


def measured_sigma():
    import csv
    by = {}
    with open(PERSEED_CSV) as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["system"], float(r["L"])), []).append(
                float(r["W_sat"]))
    out = {}
    for (s, L), ws in sorted(by.items()):
        out.setdefault(s, []).append(np.std(np.log(ws), ddof=1))
    return {s: float(np.median(v)) for s, v in out.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["sys", "law", "m", "all"])
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "floor.json")
    report = {"omega_bounds": W_BOUNDS, "u_bounds": U_BOUNDS,
              "criterion": "KL<=1/2 i.e. D^2 <= sigma^2/m; floor err>=da/4"}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            report.update(json.load(fh))

    real_design = np.array([32., 48., 64., 96., 128., 192., 256.])
    sig = measured_sigma()
    report["sigma_measured"] = sig
    print("measured per-seed sigma(logW):",
          {k: round(v, 4) for k, v in sig.items()})

    per_sys = {}
    for s, sg in (sig.items() if args.part in ("sys", "all") else []):
        per_sys[s] = {}
        for m in (6, 24):
            f = floor(real_design, sg, m)
            lf, nrm = linearized_floor(real_design, sg, m)
            per_sys[s][f"m{m}"] = {"floor_exact": f, "floor_linearized": lf}
        per_sys[s]["x_perp_norm"] = nrm
        print(f"  {s:5s} sigma={sg:.3f}  floor(m=6)="
              f"{per_sys[s]['m6']['floor_exact']:.3f}  floor(m=24)="
              f"{per_sys[s]['m24']['floor_exact']:.3f}  (lin "
              f"{per_sys[s]['m24']['floor_linearized']:.3f})")
    if per_sys:
        report["per_system_real_design"] = per_sys

    law = {}
    for L_max in ((128, 256, 512, 1024, 4096, 16384)
                  if args.part in ("law", "all") else ()):
        Ls = ladder(L_max)
        f = floor(Ls, 0.15, 24)
        lf, nrm = linearized_floor(Ls, 0.15, 24)
        law[str(L_max)] = {"floor_exact": f, "floor_linearized": lf,
                           "x_perp_norm": nrm}
        print(f"  L_max={L_max:6d}  floor={f:.4f}  lin={lf:.4f} "
              f"||P_perp x||={nrm:.4f}")
    if law:
        report["resolution_law_vs_Lmax"] = law

    if args.part in ("m", "all"):
        in_m = {str(mm): floor(real_design, 0.15, mm)
                for mm in (3, 6, 12, 24, 96)}
        report["floor_vs_m_sigma0.15"] = in_m
        print("  floor vs m:", {k: round(v, 4) for k, v in in_m.items()})

    cmp_path = os.path.join(HERE, "..",
                            "results_exp76_amortized_extrapolation",
                            "summary_full24seed.json")
    if os.path.exists(cmp_path):
        with open(cmp_path) as fh:
            s76 = json.load(fh)
        report["exp76_amortized_rmse_synthetic"] = (
            s76.get("benchmark_vs_classical", {})
               .get("amortized_mix", {}).get("rmse"))
        report["exp76_real_intervals"] = {
            k: v.get("seed_bootstrap90_mix")
            for k, v in s76.get("real_systems", {}).items()}
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
