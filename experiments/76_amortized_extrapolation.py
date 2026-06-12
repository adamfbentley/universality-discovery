"""Exp 76: Amortized finite-size extrapolation of the roughness exponent.

Context (exp75): direct correction-to-scaling fits W_sat(L) = A L^a (1 + B L^-w)
cannot recover the *known* EW/KPZ alpha = 0.5 from L <= 256 ladders -- the
result swings with the assumed correction exponent w, so the systematic
(correction-form) uncertainty dominates. BD's alpha therefore cannot be pinned.

Idea: replace the parametric fit with an *amortized estimator*: train a
regressor on synthetic W_sat(L) ladders drawn from a PRIOR OVER CORRECTION
FAMILIES (not a single assumed form), with realistic seed noise, and ask it for
alpha_inf given the 7-point ladder. The estimator implicitly marginalizes over
correction forms -- precisely what the fixed-w fit cannot do. Trained per-family
AND on the mixture, the spread across training priors is an honest systematic.

Inputs are scale-invariant by construction: the 6 adjacent effective exponents
alpha_eff(L_i, L_i+1) = dlogW/dlogL (amplitude A drops out exactly), plus the
naive global slope and a curvature/noise summary.

Correction families (generative prior):
  F0  pure:        W = A L^a                                  (B = 0)
  F1  power:       W = A L^a (1 + B L^-w),   w ~ U[0.3, 2.5]
  F2  two-term:    W = A L^a (1 + B1 L^-w + B2 L^-2w)
  F3  intrinsic:   W^2 = Wi^2 + (A L^a)^2    (Krug-Meakin form, BD's textbook
                   correction: an additive lattice-scale intrinsic width)
  F4  log:         W = A L^a (1 + B / ln L)

Noise: multiplicative log-normal per ladder point, sigma ~ U[0.0, 0.10],
covering the observed exp75 seed-mean scatter (EW alpha_eff swings 0.19-1.07).

Sanity gate (same as exp75): the estimator must recover alpha ~ 0.5 on the real
EW / KPZ / Eden ladders (known class values) before any BD number is meaningful.

Stages (each fast; rerun is cheap, all artifacts cached):
  --stage gen      generate synthetic train/test sets        -> npz
  --stage train    fit point + quantile GBMs per prior       -> joblib
  --stage eval     synthetic benchmark vs classical fits, transfer matrix,
                   real-data gate, BD answer                  -> json/csv/md

Run:  python experiments/76_amortized_extrapolation.py --stage all [--pilot]
"""

import argparse
import json
import os
import warnings

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "results_exp76_amortized_extrapolation")
EXP75_CSV = os.path.join(os.path.dirname(__file__), "..",
                         "results_exp75_correction_to_scaling", "wsat_vs_L.csv")

L_LADDER = np.array([32, 48, 64, 96, 128, 192, 256], dtype=float)
LOG_L = np.log(L_LADDER)
ALPHA_LO, ALPHA_HI = 0.05, 0.95
FAMILIES = ["F0_pure", "F1_power", "F2_twoterm", "F3_intrinsic", "F4_log"]
TRAIN_PRIORS = FAMILIES + ["mix"]
QUANTILES = [0.05, 0.5, 0.95]
RNG_SEED = 76


# ----------------------------------------------------------------------------
# Synthetic ladder generation
# ----------------------------------------------------------------------------

def _sample_family(fam, n, rng):
    """Return (n, 7) noiseless W ladders and (n,) true alpha."""
    a = rng.uniform(ALPHA_LO, ALPHA_HI, n)
    A = np.exp(rng.uniform(np.log(0.05), np.log(5.0), n))
    base = A[:, None] * L_LADDER[None, :] ** a[:, None]

    if fam == "F0_pure":
        W = base
    elif fam == "F1_power":
        w = rng.uniform(0.3, 2.5, n)
        corr_at_Lmin = rng.uniform(-0.75, 4.0, n)        # (1+B L^-w) at L=32
        B = corr_at_Lmin * L_LADDER[0] ** w
        W = base * (1.0 + B[:, None] * L_LADDER[None, :] ** -w[:, None])
    elif fam == "F2_twoterm":
        w = rng.uniform(0.3, 2.0, n)
        c1 = rng.uniform(-0.6, 3.0, n)
        c2 = rng.uniform(-0.5, 2.0, n)
        B1 = c1 * L_LADDER[0] ** w
        B2 = c2 * L_LADDER[0] ** (2 * w)
        corr = (1.0 + B1[:, None] * L_LADDER[None, :] ** -w[:, None]
                + B2[:, None] * L_LADDER[None, :] ** (-2 * w[:, None]))
        corr = np.clip(corr, 0.05, None)
        W = base * corr
    elif fam == "F3_intrinsic":
        # W^2 = Wi^2 + base^2 ; Wi up to ~3x base at L_min (BD regime: large)
        ratio = rng.uniform(0.0, 3.0, n)                 # Wi / base(L_min)
        Wi = ratio * base[:, 0]
        W = np.sqrt(Wi[:, None] ** 2 + base ** 2)
    elif fam == "F4_log":
        c = rng.uniform(-0.6, 4.0, n)                    # (1+B/lnL) at L=32
        B = c * np.log(L_LADDER[0])
        W = base * (1.0 + B[:, None] / LOG_L[None, :])
    else:
        raise ValueError(fam)
    return W, a


def make_dataset(fam, n, rng, sigma_max=0.10):
    if fam == "mix":
        per = n // len(FAMILIES)
        Ws, alphas = zip(*[_sample_family(f, per, rng) for f in FAMILIES])
        W, a = np.vstack(Ws), np.concatenate(alphas)
    else:
        W, a = _sample_family(fam, n, rng)
    # multiplicative log-normal noise, per-ladder noise level
    sig = rng.uniform(0.0, sigma_max, len(a))
    W = W * np.exp(sig[:, None] * rng.standard_normal(W.shape))
    keep = np.all(np.isfinite(W) & (W > 1e-8), axis=1)
    return W[keep], a[keep]


def featurize(W):
    """Scale-invariant features from a (n, 7) ladder."""
    logW = np.log(W)
    aeff = np.diff(logW, axis=1) / np.diff(LOG_L)[None, :]      # 6 eff. exps
    naive = ((logW[:, -1] - logW[:, 0]) / (LOG_L[-1] - LOG_L[0]))[:, None]
    # curvature + roughness of the log-log curve (noise/correction proxy)
    x = LOG_L - LOG_L.mean()
    coef = np.polynomial.polynomial.polyfit(x, logW.T, 2)        # (3, n)
    fitted = np.polynomial.polynomial.polyval(x, coef)           # (n, 7)
    resid = logW - fitted
    feats = np.hstack([aeff, naive, coef[2][:, None],
                       resid.std(axis=1)[:, None],
                       np.diff(aeff, axis=1)])                   # 6+1+1+1+5=14
    return feats


# ----------------------------------------------------------------------------
# Classical baselines (the exp75 estimators, reimplemented)
# ----------------------------------------------------------------------------

def classical_fits(W):
    """Return dict of per-ladder alpha estimates for classical methods."""
    from scipy.optimize import curve_fit
    logW, out = np.log(W), {}
    out["naive"] = (logW[:, -1] - logW[:, 0]) / (LOG_L[-1] - LOG_L[0])

    def fit_fixed_omega(omega):
        est = np.full(len(W), np.nan)
        def f(L, lnA, a, B):
            return lnA + a * np.log(L) + np.log(np.clip(1 + B * L ** -omega,
                                                        1e-6, None))
        for i in range(len(W)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(f, L_LADDER, logW[i],
                                     p0=[logW[i, 0] - 0.5 * LOG_L[0], 0.5, 0.0],
                                     maxfev=4000)
                est[i] = p[1]
            except Exception:
                pass
        return est

    out["fit_w1"] = fit_fixed_omega(1.0)
    out["fit_w0p5"] = fit_fixed_omega(0.5)

    est = np.full(len(W), np.nan)                                 # free omega
    def g(L, lnA, a, B, w):
        return lnA + a * np.log(L) + np.log(np.clip(1 + B * L ** -np.abs(w),
                                                    1e-6, None))
    for i in range(len(W)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(g, L_LADDER, logW[i],
                                 p0=[logW[i, 0] - 0.5 * LOG_L[0], 0.5, 0.0, 1.0],
                                 maxfev=6000)
            est[i] = p[1]
        except Exception:
            pass
    out["fit_free"] = est
    return out


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------

def stage_gen(n_train, n_test, pilot):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    data = {}
    for fam in TRAIN_PRIORS:
        W, a = make_dataset(fam, n_train, rng)
        data[f"trainW_{fam}"], data[f"traina_{fam}"] = W, a
    for fam in FAMILIES:                       # test sets are per-family only
        W, a = make_dataset(fam, n_test, rng)
        data[f"testW_{fam}"], data[f"testa_{fam}"] = W, a
    np.savez_compressed(os.path.join(RESULTS_DIR, "datasets.npz"), **data)
    print(f"gen: {len(TRAIN_PRIORS)} train priors x {n_train}, "
          f"{len(FAMILIES)} test families x {n_test}")


def stage_train(prior):
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    d = np.load(os.path.join(RESULTS_DIR, "datasets.npz"))
    X = featurize(d[f"trainW_{prior}"])
    y = d[f"traina_{prior}"]
    models = {}
    m = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    m.fit(X, y)
    models["point"] = m
    for q in QUANTILES:
        mq = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                           max_iter=300, random_state=0)
        mq.fit(X, y)
        models[f"q{q}"] = mq
    joblib.dump(models, os.path.join(RESULTS_DIR, f"model_{prior}.joblib"))
    print(f"train[{prior}]: {X.shape[0]} samples, {X.shape[1]} features")


def _load_real_ladders():
    import csv
    rows = {}
    with open(EXP75_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["system"], {})[float(r["L"])] = float(r["W_sat"])
    out = {}
    for sysname, m in rows.items():
        if set(m) == set(L_LADDER):
            out[sysname] = np.array([m[L] for L in L_LADDER])
    return out


PERSEED_CSV = os.path.join(RESULTS_DIR, "wsat_perseed.csv")


def _load_perseed_ladders():
    """Return {system: (n_seeds, 7) W_sat array} from exp76b, if present."""
    import csv
    if not os.path.exists(PERSEED_CSV):
        return {}
    rows = {}
    with open(PERSEED_CSV) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["system"], {}).setdefault(
                int(r["seed"]), {})[float(r["L"])] = float(r["W_sat"])
    out = {}
    for sysname, seeds in rows.items():
        mats = [np.array([m[L] for L in L_LADDER])
                for m in seeds.values() if set(m) == set(L_LADDER)]
        if len(mats) >= 8:
            out[sysname] = np.vstack(mats)
    return out


def _predict_with_bootstrap(models, Wseeds, n_boot=400, rng=None):
    """Seed-mean point prediction + seed-bootstrap 90% interval (data unc.)."""
    rng = rng or np.random.default_rng(7)
    n = len(Wseeds)
    Wmean = Wseeds.mean(axis=0)
    point = float(models["point"].predict(featurize(Wmean[None, :]))[0])
    idx = rng.integers(0, n, size=(n_boot, n))
    Wboot = Wseeds[idx].mean(axis=1)                       # (n_boot, 7)
    preds = models["point"].predict(featurize(Wboot))
    return point, [float(np.percentile(preds, 5)),
                   float(np.percentile(preds, 95))]


def stage_eval(n_classical):
    import joblib
    d = np.load(os.path.join(RESULTS_DIR, "datasets.npz"))
    models = {p: joblib.load(os.path.join(RESULTS_DIR, f"model_{p}.joblib"))
              for p in TRAIN_PRIORS}
    report = {"L_ladder": L_LADDER.tolist(), "quantiles": QUANTILES}

    # 1) transfer matrix: train prior x test family, RMSE on alpha
    rmse = {}
    cover = {}
    for p in TRAIN_PRIORS:
        for f in FAMILIES:
            X = featurize(d[f"testW_{f}"])
            y = d[f"testa_{f}"]
            pred = models[p]["point"].predict(X)
            lo = models[p]["q0.05"].predict(X)
            hi = models[p]["q0.95"].predict(X)
            rmse[f"{p}|{f}"] = float(np.sqrt(np.mean((pred - y) ** 2)))
            cover[f"{p}|{f}"] = float(np.mean((y >= lo) & (y <= hi)))
    report["rmse_train_x_test"] = rmse
    report["coverage90_train_x_test"] = cover

    # 2) classical baselines on a subsample of the MIXTURE of test families
    rng = np.random.default_rng(1)
    Wm = np.vstack([d[f"testW_{f}"] for f in FAMILIES])
    ym = np.concatenate([d[f"testa_{f}"] for f in FAMILIES])
    idx = rng.choice(len(ym), size=min(n_classical, len(ym)), replace=False)
    Wc, yc = Wm[idx], ym[idx]
    cb = classical_fits(Wc)
    base = {}
    for k, v in cb.items():
        ok = np.isfinite(v)
        base[k] = {"rmse": float(np.sqrt(np.mean((v[ok] - yc[ok]) ** 2))),
                   "bias": float(np.mean(v[ok] - yc[ok])),
                   "frac_ok": float(ok.mean())}
    predc = models["mix"]["point"].predict(featurize(Wc))
    base["amortized_mix"] = {"rmse": float(np.sqrt(np.mean((predc - yc) ** 2))),
                             "bias": float(np.mean(predc - yc)), "frac_ok": 1.0}
    report["benchmark_vs_classical"] = base

    # 3) real data: sanity gate + BD
    # Prefer high-seed exp76b ladders; fall back to exp75 6-seed means.
    perseed = _load_perseed_ladders()
    real = {s: m.mean(axis=0) for s, m in perseed.items()} or _load_real_ladders()
    report["real_data_source"] = ("exp76b_perseed" if perseed
                                  else "exp75_seedmean")
    if perseed:
        report["n_seeds_real"] = {s: int(len(m)) for s, m in perseed.items()}
    real_out = {}
    for sysname, W in real.items():
        X = featurize(W[None, :])
        per_prior = {}
        for p in TRAIN_PRIORS:
            per_prior[p] = {
                "point": float(models[p]["point"].predict(X)[0]),
                "q05": float(models[p]["q0.05"].predict(X)[0]),
                "q50": float(models[p]["q0.5"].predict(X)[0]),
                "q95": float(models[p]["q0.95"].predict(X)[0]),
            }
        pts = [per_prior[p]["point"] for p in TRAIN_PRIORS]
        real_out[sysname] = {
            "per_prior": per_prior,
            "alpha_hat_mix": per_prior["mix"]["point"],
            "interval90_mix": [per_prior["mix"]["q05"], per_prior["mix"]["q95"]],
            "prior_spread": [float(np.min(pts)), float(np.max(pts))],
        }
        if sysname in perseed:                      # seed-bootstrap (data unc.)
            _, boot90 = _predict_with_bootstrap(models["mix"], perseed[sysname])
            real_out[sysname]["seed_bootstrap90_mix"] = boot90
    report["real_systems"] = real_out

    theory = {"ew": 0.5, "kpz": 0.5, "bd": 0.5, "eden": 0.5}
    gate = {s: {"alpha_hat": real_out[s]["alpha_hat_mix"],
                "abs_err": abs(real_out[s]["alpha_hat_mix"] - theory[s])}
            for s in ("ew", "kpz", "eden") if s in real_out}
    gate_pass = all(v["abs_err"] <= 0.10 for v in gate.values())
    report["sanity_gate"] = {"systems": gate, "tolerance": 0.10,
                             "pass": bool(gate_pass)}

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    # console digest
    print("=== benchmark (mixture test set) ===")
    for k, v in base.items():
        print(f"  {k:14s} rmse={v['rmse']:.3f} bias={v['bias']:+.3f}")
    print(f"=== real systems (source: {report['real_data_source']}) ===")
    for s, r in real_out.items():
        boot = r.get("seed_bootstrap90_mix")
        bs = (f" boot90=[{boot[0]:.3f},{boot[1]:.3f}]" if boot else "")
        print(f"  {s:5s} alpha_hat={r['alpha_hat_mix']:.3f} "
              f"90%=[{r['interval90_mix'][0]:.3f},{r['interval90_mix'][1]:.3f}] "
              f"prior-spread=[{r['prior_spread'][0]:.3f},{r['prior_spread'][1]:.3f}]"
              + bs)
    print(f"=== sanity gate (EW/KPZ/Eden within 0.10 of 0.5): "
          f"{'PASS' if gate_pass else 'FAIL'} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gen", "train", "eval", "all"])
    ap.add_argument("--prior", default=None, help="train a single prior")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    n_train = 40_000 if args.pilot else 200_000
    n_test = 4_000 if args.pilot else 20_000
    n_classical = 400 if args.pilot else 2_000

    if args.stage in ("gen", "all"):
        stage_gen(n_train, n_test, args.pilot)
    if args.stage in ("train", "all"):
        priors = [args.prior] if args.prior else TRAIN_PRIORS
        for p in priors:
            stage_train(p)
    if args.stage in ("eval", "all"):
        stage_eval(n_classical)


if __name__ == "__main__":
    main()
