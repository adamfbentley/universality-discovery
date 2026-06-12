"""Exp 78: three referee-proofing checks for the exp76/77 claims.

A. Exact-measure check: in 1D (periodic), stationary EW and KPZ interfaces are
   Brownian, so W_sat = sqrt(L/12) exactly. Compare both systems' 24-seed
   ladders to the exact value -> validates EW data, diagnoses the KPZ
   integrator (suspected fluctuation-dissipation violation, cf. Lam-Shin).
B. Expert baseline: BD's textbook correction is an additive intrinsic width,
   W^2 = b + a L^{2 alpha}. Fit (a, b, alpha) directly and with alpha fixed
   at 0.5 -> does the known ansatz recover alpha = 0.5 classically?
C. Discriminability control: generate BD-like ladders (additive-width form,
   BD-matched amplitude/correction/noise) with true alpha in
   {0.40, 0.45, 0.50, 0.55}; predict with the exp76 mix model. If the output
   distributions separate, the BD result is measurement, not prior-mean
   shrinkage (training alpha prior is centered at 0.5).

Output: results_exp76_amortized_extrapolation/referee_checks.json
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
R76 = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")
L_LADDER = np.array([32., 48., 64., 96., 128., 192., 256.])


def load_perseed():
    by = {}
    with open(os.path.join(R76, "wsat_perseed.csv")) as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["system"], {}).setdefault(
                int(r["seed"]), {})[float(r["L"])] = float(r["W_sat"])
    out = {}
    for s, seeds in by.items():
        out[s] = np.vstack([[m[L] for L in L_LADDER]
                            for m in seeds.values() if len(m) == 7])
    return out


def check_exact_measure(data, rep):
    exact = np.sqrt(L_LADDER / 12.0)
    out = {}
    for s in ("ew", "kpz"):
        W = data[s]
        ratio = W.mean(axis=0) / exact
        sem = W.std(axis=0, ddof=1) / np.sqrt(len(W)) / exact
        dev_sigma = (ratio - 1.0) / sem
        out[s] = {"L": L_LADDER.tolist(), "ratio": ratio.round(4).tolist(),
                  "sem": sem.round(4).tolist(),
                  "deviation_in_sigma": dev_sigma.round(2).tolist()}
        print(f"[A] {s}: W/sqrt(L/12) =",
              " ".join(f"{r:.3f}({d:+.1f}s)" for r, d in zip(ratio, dev_sigma)))
    rep["A_exact_measure"] = out


def check_additive_fit(data, rep):
    from scipy.optimize import curve_fit
    W2 = (data["bd"] ** 2).mean(axis=0)
    sem2 = (data["bd"] ** 2).std(axis=0, ddof=1) / np.sqrt(len(data["bd"]))
    # free alpha
    f3 = lambda L, a, b, al: b + a * L ** (2 * al)
    p3, c3 = curve_fit(f3, L_LADDER, W2, p0=[0.5, 16., 0.5], sigma=sem2,
                       absolute_sigma=True, maxfev=10000)
    # fixed alpha = 1/2 (linear)
    A = np.vstack([L_LADDER, np.ones_like(L_LADDER)]).T
    coef, res2, *_ = np.linalg.lstsq(A, W2, rcond=None)
    pred = A @ coef
    chi2 = float(np.sum(((W2 - pred) / sem2) ** 2))
    rep["B_additive_fit_bd"] = {
        "free_alpha": float(p3[2]), "free_alpha_err": float(np.sqrt(c3[2, 2])),
        "free_a": float(p3[0]), "free_b": float(p3[1]),
        "fixed_half_a": float(coef[0]), "fixed_half_b": float(coef[1]),
        "fixed_half_chi2_per_dof": chi2 / (len(L_LADDER) - 2),
    }
    print(f"[B] BD W^2 = b + a L^(2a): alpha = {p3[2]:.4f} +- "
          f"{np.sqrt(c3[2,2]):.4f}; fixed-0.5 chi2/dof = "
          f"{chi2/(len(L_LADDER)-2):.2f} (a={coef[0]:.3f}, b={coef[1]:.2f})")


def check_discriminability(data, rep, n_rep=300):
    import importlib.util
    import joblib
    spec = importlib.util.spec_from_file_location(
        "e76", os.path.join(HERE, "76_amortized_extrapolation.py"))
    e76 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e76)
    models = joblib.load(os.path.join(R76, "model_mix.joblib"))
    rng = np.random.default_rng(78)
    sigma_seed = 0.019                      # measured BD per-seed noise
    m_seeds = 24
    out = {}
    for alpha in (0.40, 0.45, 0.50, 0.55):
        A = 4.13 / 32.0 ** alpha            # BD-matched amplitude at L=32
        ratio = rng.uniform(0.8, 1.2, n_rep)  # BD-matched intrinsic width
        base = A * L_LADDER[None, :] ** alpha
        Wi = ratio * base[0, 0] / 1.0
        W0 = np.sqrt(Wi[:, None] ** 2 + base ** 2)
        noise = (sigma_seed / np.sqrt(m_seeds)
                 * rng.standard_normal((n_rep, 7)))
        W = W0 * np.exp(noise)
        pred = models["point"].predict(e76.featurize(W))
        out[str(alpha)] = {"mean": float(pred.mean()),
                           "std": float(pred.std(ddof=1)),
                           "q05": float(np.percentile(pred, 5)),
                           "q95": float(np.percentile(pred, 95))}
        print(f"[C] true alpha={alpha:.2f}: pred = {pred.mean():.4f} "
              f"+- {pred.std(ddof=1):.4f}  [{np.percentile(pred,5):.3f},"
              f"{np.percentile(pred,95):.3f}]")
    rep["C_discriminability_bdlike"] = out


def main():
    rep = {}
    data = load_perseed()
    check_exact_measure(data, rep)
    check_additive_fit(data, rep)
    check_discriminability(data, rep)
    with open(os.path.join(R76, "referee_checks.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved referee_checks.json")


if __name__ == "__main__":
    main()
