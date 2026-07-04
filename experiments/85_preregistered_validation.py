"""Exp 85: preregistered predictive validation.

Stages are intentionally separated because the commit order is part of the
experiment:

  1. tasks12       real-data local floors and class-adequacy bootstrap
  2. phase1        blind fractional-EW predictions + prediction hash
  3. phase2        score only after predictions are committed in HEAD
  4. report        write the findings-only report

No phase-2 truth fields are written to predictions.json.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, norm


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results_exp85_preregistered")
WSAT_CSV = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "wsat_perseed.csv")
SUMMARY76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "summary_full24seed.json")
CLASSICAL_REAL76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "classical_on_real.json")
FLOOR77 = os.path.join(ROOT, "results_exp77_minimax_floor", "floor.json")
HIER81 = os.path.join(ROOT, "results_exp81_hierarchy", "floors_hierarchy.json")
ISING52D = os.path.join(ROOT, "results_exp52d_full", "results.json")

E77_PATH = os.path.join(HERE, "77_minimax_floor.py")
E83_PATH = os.path.join(HERE, "83_audit_response.py")
E81_PATH = os.path.join(HERE, "81_fractional_ew_testbed.py")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
HALF_DESIGN_BD = np.array([32., 48., 64., 96.])
M_REAL = 24
OMEGA_BOUNDS = (0.3, 2.5)
MASTER_SEED = 85000


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def write_json(path: str, obj) -> None:
    ensure_results_dir()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_exists(relpath: str) -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{relpath.replace(os.sep, '/')}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def require_phase1_committed() -> None:
    rels = [
        os.path.join("results_exp85_preregistered", "predictions.json"),
        os.path.join("results_exp85_preregistered", "predictions.sha256"),
    ]
    missing = [p for p in rels if not git_blob_exists(p)]
    if missing:
        raise SystemExit(
            "Refusing to unblind: phase-1 prediction artifacts are not "
            f"committed in HEAD: {missing}")


def load_wsat_perseed() -> Dict[str, np.ndarray]:
    rows: Dict[str, Dict[int, Dict[float, float]]] = {}
    with open(WSAT_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["system"], {}).setdefault(
                int(r["seed"]), {})[float(r["L"])] = float(r["W_sat"])
    out = {}
    for system, seeds in rows.items():
        mats = []
        for m in seeds.values():
            if set(m) == set(DESIGN):
                mats.append([m[L] for L in DESIGN])
        out[system] = np.asarray(mats, dtype=float)
    return out


def y_sigma_from_wseeds(W: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
    logw = np.log(W)
    y = logw.mean(axis=0)
    per_l = logw.std(axis=0, ddof=1)
    return y, float(np.median(per_l)), per_l.tolist()


def corr_log(Ls: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    L1 = Ls[0]
    return np.log(1.0 + u * (Ls / L1) ** (-w))


def corr_log_scalar(Ls: np.ndarray, u: float, w: float) -> np.ndarray:
    L1 = Ls[0]
    val = 1.0 + u * (Ls / L1) ** (-w)
    if np.any(val <= 0):
        return np.full_like(Ls, np.nan, dtype=float)
    return np.log(val)


def u_bounds_for_class(U: float, legacy_positive: bool = False) -> Tuple[float, float]:
    if legacy_positive:
        return (-0.75, float(U))
    return (-min(float(U), 0.95), float(U))


def logform_mu(Ls: np.ndarray, c: float, alpha: float, u: float, w: float) -> np.ndarray:
    g = corr_log_scalar(Ls, u, w)
    if np.any(~np.isfinite(g)):
        return np.full_like(Ls, np.nan, dtype=float)
    return c + alpha * np.log(Ls) + g


def fit_logform(
    y: np.ndarray,
    Ls: np.ndarray,
    U: float,
    seed: int,
    n_starts: int = 12,
    alpha_bounds: Tuple[float, float] = (-0.2, 1.2),
    legacy_positive: bool = False,
) -> Dict:
    x = np.log(Ls)
    rng = np.random.default_rng(seed)
    slope, intercept = np.polyfit(x, y, 1)
    ub = u_bounds_for_class(U, legacy_positive)
    bounds = [(None, None), alpha_bounds, ub, OMEGA_BOUNDS]

    def obj(p):
        c, alpha, u, w = p
        mu = logform_mu(Ls, c, alpha, u, w)
        if np.any(~np.isfinite(mu)):
            return 1e30
        r = y - mu
        return float(np.sum(r * r))

    starts = [
        np.array([intercept, slope, np.clip(0.0, *ub), 1.0]),
        np.array([intercept, 0.5, np.clip(0.0, *ub), 1.0]),
    ]
    for _ in range(n_starts):
        starts.append(np.array([
            intercept + rng.normal(0, 0.3),
            rng.uniform(*alpha_bounds),
            rng.uniform(*ub),
            rng.uniform(*OMEGA_BOUNDS),
        ]))

    best = None
    for p0 in starts:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "maxfun": 1500})
        if best is None or res.fun < best.fun:
            best = res
    c, alpha, u, w = best.x
    mu = logform_mu(Ls, c, alpha, u, w)
    return {
        "c": float(c),
        "alpha": float(alpha),
        "u": float(u),
        "omega": float(w),
        "sse": float(best.fun),
        "mu": mu.tolist(),
        "success": bool(best.success),
        "u_bounds": list(ub),
    }


def confusion_gap_pair(
    da: float,
    Ls: np.ndarray,
    U: float,
    seed: int,
    n_starts: int = 10,
    legacy_positive: bool = False,
) -> float:
    x = np.log(Ls)
    rng = np.random.default_rng(seed)
    ub = u_bounds_for_class(U, legacy_positive)
    bounds = [(None, None), ub, OMEGA_BOUNDS, ub, OMEGA_BOUNDS]

    def obj(p):
        c, u1, w1, u2, w2 = p
        g1 = corr_log_scalar(Ls, u1, w1)
        g2 = corr_log_scalar(Ls, u2, w2)
        if np.any(~np.isfinite(g1)) or np.any(~np.isfinite(g2)):
            return 1e30
        diff = da * x + c + g1 - g2
        return float(np.sum(diff * diff))

    starts = [np.array([-da * x.mean(), 0.0, 1.0, 0.0, 1.0])]
    for _ in range(n_starts):
        starts.append(np.array([
            -da * x.mean() + rng.normal(0, 0.5),
            rng.uniform(*ub), rng.uniform(*OMEGA_BOUNDS),
            rng.uniform(*ub), rng.uniform(*OMEGA_BOUNDS),
        ]))
    best = np.inf
    for p0 in starts:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 250, "maxfun": 1200})
        best = min(best, float(res.fun))
    return best


def confusion_gap_local(
    da: float,
    Ls: np.ndarray,
    fit: Dict,
    U: float,
    seed: int,
    n_starts: int = 10,
) -> Dict:
    x = np.log(Ls)
    mu0 = np.asarray(fit["mu"], dtype=float)
    rng = np.random.default_rng(seed)
    ub = u_bounds_for_class(U, legacy_positive=False)
    bounds = [(None, None), ub, OMEGA_BOUNDS]

    def one_sign(sign):
        alpha2 = fit["alpha"] + sign * da

        def obj(p):
            c2, u2, w2 = p
            mu2 = logform_mu(Ls, c2, alpha2, u2, w2)
            if np.any(~np.isfinite(mu2)):
                return 1e30
            r = mu2 - mu0
            return float(np.sum(r * r))

        starts = [np.array([mu0.mean() - alpha2 * x.mean(), fit["u"], fit["omega"]])]
        for _ in range(n_starts):
            starts.append(np.array([
                mu0.mean() - alpha2 * x.mean() + rng.normal(0, 0.5),
                rng.uniform(*ub),
                rng.uniform(*OMEGA_BOUNDS),
            ]))
        best = np.inf
        for p0 in starts:
            p0[1] = np.clip(p0[1], *ub)
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 250, "maxfun": 1000})
            best = min(best, float(res.fun))
        return best

    plus = one_sign(+1.0)
    minus = one_sign(-1.0)
    return {"gap": float(min(plus, minus)), "plus": float(plus), "minus": float(minus)}


def floor_from_gap_grid(
    da_grid: np.ndarray,
    gaps: np.ndarray,
    sigma: float,
    m: int,
) -> Tuple[float, np.ndarray, bool]:
    running = np.maximum.accumulate(gaps)
    thresh = sigma * sigma / m
    raw_diffs = np.diff(gaps)
    tol = 0.05 * np.maximum(gaps[:-1], 1e-12)
    monotone = bool(np.all(raw_diffs >= -tol))
    idx = np.searchsorted(running, thresh)
    if idx == 0:
        return float(da_grid[0]), running, monotone
    if idx >= len(da_grid):
        return float(da_grid[-1]), running, monotone
    x0, x1 = da_grid[idx - 1], da_grid[idx]
    y0, y1 = running[idx - 1], running[idx]
    if y1 <= y0:
        return float(x1), running, monotone
    frac = (thresh - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0)), running, monotone


def monotone_bad_indices(gaps: np.ndarray) -> np.ndarray:
    diffs = np.diff(gaps)
    tol = 0.05 * np.maximum(gaps[:-1], 1e-12)
    return np.where(diffs < -tol)[0]


def repair_pair_gap_monotonicity(
    gaps: np.ndarray,
    da_grid: np.ndarray,
    Ls: np.ndarray,
    U: float,
    base_seed: int,
    starts: int,
    legacy_positive: bool,
) -> np.ndarray:
    """Retry isolated raw-monotonicity dips with a larger optimizer budget.

    A dip y[i] > y[i+1] means the smaller-delta point is the suspect one:
    the adversary failed to find a confusion at least as good as the one
    available at the larger delta. This keeps the raw grid honest before the
    running-max floor path is applied.
    """
    out = np.array(gaps, dtype=float)
    for attempt in range(3):
        bad = monotone_bad_indices(out)
        if len(bad) == 0:
            break
        for idx in bad:
            better = confusion_gap_pair(
                float(da_grid[idx]), Ls, U,
                seed=base_seed + 100000 * (attempt + 1) + int(idx),
                n_starts=starts * (3 + attempt),
                legacy_positive=legacy_positive)
            out[idx] = min(out[idx], better)
    return out


def task1_instance_local(args) -> Dict:
    data = load_wsat_perseed()
    summary = read_json(SUMMARY76)
    classical = read_json(CLASSICAL_REAL76)
    floor77 = read_json(FLOOR77)

    da_grid = np.linspace(0.0, args.task1_da_hi, args.task1_grid)
    systems = sorted(data)
    out = {
        "five_tuple": (
            "Level-0, class N=1/log-form/U as tabulated/omega_min=0.3, "
            "design={32,48,64,96,128,192,256}, sigma=measured per-system "
            "median sd(log W), m=24"),
        "systems": {},
    }

    for si, s in enumerate(systems):
        y, sigma, per_l = y_sigma_from_wseeds(data[s])
        fit = fit_logform(y, DESIGN, U=4.0, seed=850100 + si,
                          n_starts=args.fit_starts, legacy_positive=True)
        U_honest = max(0.03, 2.0 * abs(fit["u"]))

        legacy_floor = floor77["per_system_real_design"][s]["m24"]["floor_exact"]

        h_gaps = np.array([
            confusion_gap_pair(da, DESIGN, U_honest, seed=851000 + si * 1000 + j,
                               n_starts=args.gap_starts)
            for j, da in enumerate(da_grid)
        ])
        h_gaps = repair_pair_gap_monotonicity(
            h_gaps, da_grid, DESIGN, U_honest, base_seed=851000 + si * 1000,
            starts=args.gap_starts, legacy_positive=False)
        honest_floor, h_running, h_mono = floor_from_gap_grid(
            da_grid, h_gaps, sigma, M_REAL)

        local_rows = [
            confusion_gap_local(da, DESIGN, fit, U_honest,
                                seed=852000 + si * 1000 + j,
                                n_starts=args.gap_starts)
            for j, da in enumerate(da_grid)
        ]
        l_gaps = np.array([r["gap"] for r in local_rows])
        local_floor, l_running, l_mono = floor_from_gap_grid(
            da_grid, l_gaps, sigma, M_REAL)

        nesting_ok = bool(local_floor <= honest_floor + args.nesting_tol)
        if not nesting_ok:
            raise RuntimeError(
                f"G-85a nesting violation for {s}: local {local_floor} > "
                f"global honest {honest_floor}")

        true_alpha = 0.5
        exp76_alpha = summary["real_systems"][s]["alpha_hat_mix"]
        class_alpha = classical[s]["classical_alpha"]
        observed = {
            "truth_reference_alpha": true_alpha,
            "exp76_mix_abs_error": float(abs(exp76_alpha - true_alpha)),
            "exp76_mix_signed_error": float(exp76_alpha - true_alpha),
            "exp75_or_classical_abs_errors": {
                k: (None if v is None else float(abs(v - true_alpha)))
                for k, v in class_alpha.items()
            },
            "exp75_or_classical_signed_errors": {
                k: (None if v is None else float(v - true_alpha))
                for k, v in class_alpha.items()
            },
        }

        out["systems"][s] = {
            "sigma": sigma,
            "per_L_sigma_logW": per_l,
            "fit_declared_N1_U4": {
                k: v for k, v in fit.items() if k != "mu"
            },
            "honest_U_2x_abs_fit_u": U_honest,
            "global_U4_floor_exp77": legacy_floor,
            "honest_class_global_floor": honest_floor,
            "instance_local_floor": local_floor,
            "gates": {
                "local_floor_le_global_honest_floor": nesting_ok,
                "global_honest_D2_monotone": h_mono,
                "local_D2_monotone": l_mono,
            },
            "observed_errors": observed,
            "diagnostic_grids": {
                "da": da_grid.tolist(),
                "honest_global_D2": h_gaps.tolist(),
                "honest_global_D2_running_max": h_running.tolist(),
                "local_D2": l_gaps.tolist(),
                "local_D2_running_max": l_running.tolist(),
                "local_plus_D2": [r["plus"] for r in local_rows],
                "local_minus_D2": [r["minus"] for r in local_rows],
            },
        }
        print(f"[G-85a] {s}: U_honest={U_honest:.4f} "
              f"global_U4={legacy_floor:.4f} honest_global={honest_floor:.4f} "
              f"local={local_floor:.4f} monotone=({h_mono},{l_mono})")

    return out


def additive_fixed_half_fit(W: np.ndarray, Ls: np.ndarray) -> Dict:
    W2 = W.mean(axis=0) ** 2
    A = np.vstack([Ls, np.ones_like(Ls)]).T
    coef, *_ = np.linalg.lstsq(A, W2, rcond=None)
    a, b = coef
    pred = a * Ls + b
    return {"a": float(a), "b": float(b), "pred_W2": pred.tolist()}


def additive_chi2_fixed_half(Wseeds: np.ndarray, Ls: np.ndarray) -> Dict:
    W2_seeds = Wseeds ** 2
    W2_mean = W2_seeds.mean(axis=0)
    sem2 = W2_seeds.std(axis=0, ddof=1) / math.sqrt(Wseeds.shape[0])
    A = np.vstack([Ls, np.ones_like(Ls)]).T
    coef, *_ = np.linalg.lstsq(A / sem2[:, None], W2_mean / sem2, rcond=None)
    pred = A @ coef
    chi2_obs = float(np.sum(((W2_mean - pred) / sem2) ** 2))
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "pred_W2": pred.tolist(),
        "chi2": chi2_obs,
        "dof": int(len(Ls) - 2),
        "sem_W2": sem2.tolist(),
        "W2_mean": W2_mean.tolist(),
    }


def fit_logform_chi2(y: np.ndarray, Ls: np.ndarray, sigma: float, m: int,
                     U: float, seed: int, starts: int) -> Dict:
    fit = fit_logform(y, Ls, U, seed=seed, n_starts=starts,
                      legacy_positive=(U >= 4.0))
    mu = np.asarray(fit["mu"], dtype=float)
    sigma_m = sigma / math.sqrt(m)
    stat = float(np.sum(((y - mu) / sigma_m) ** 2))
    fit2 = {k: v for k, v in fit.items() if k != "mu"}
    fit2.update({"chi2": stat, "dof": int(len(Ls) - 4)})
    fit2["mu"] = mu.tolist()
    return fit2


def bootstrap_gof_logform(
    y: np.ndarray,
    Ls: np.ndarray,
    sigma: float,
    m: int,
    U: float,
    seed: int,
    n_boot: int,
    starts: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    fit = fit_logform_chi2(y, Ls, sigma, m, U, seed + 1, starts)
    mu0 = np.asarray(fit["mu"], dtype=float)
    sigma_m = sigma / math.sqrt(m)
    boot = []
    for b in range(n_boot):
        yb = mu0 + sigma_m * rng.standard_normal(len(Ls))
        fb = fit_logform_chi2(yb, Ls, sigma, m, U, seed + 1000 + b, starts)
        boot.append(fb["chi2"])
    boot = np.asarray(boot)
    p = float((1 + np.sum(boot >= fit["chi2"])) / (len(boot) + 1))
    return {
        "fit": {k: v for k, v in fit.items() if k != "mu"},
        "observed_chi2": fit["chi2"],
        "dof": fit["dof"],
        "bootstrap_p": p,
        "n_boot": int(n_boot),
        "bootstrap_chi2_quantiles": {
            "q05": float(np.percentile(boot, 5)),
            "q50": float(np.percentile(boot, 50)),
            "q95": float(np.percentile(boot, 95)),
        },
    }


def bootstrap_gof_additive(Wseeds: np.ndarray, Ls: np.ndarray,
                           seed: int, n_boot: int) -> Dict:
    rng = np.random.default_rng(seed)
    obs = additive_chi2_fixed_half(Wseeds, Ls)
    pred = np.asarray(obs["pred_W2"], dtype=float)
    sem = np.asarray(obs["sem_W2"], dtype=float)
    boot = []
    for _ in range(n_boot):
        W2b = pred + sem * rng.standard_normal(len(Ls))
        A = np.vstack([Ls, np.ones_like(Ls)]).T
        coef, *_ = np.linalg.lstsq(A / sem[:, None], W2b / sem, rcond=None)
        rb = W2b - A @ coef
        boot.append(float(np.sum((rb / sem) ** 2)))
    boot = np.asarray(boot)
    p = float((1 + np.sum(boot >= obs["chi2"])) / (len(boot) + 1))
    return {
        "fit": {k: v for k, v in obs.items() if k not in ("pred_W2", "sem_W2", "W2_mean")},
        "observed_chi2": obs["chi2"],
        "dof": obs["dof"],
        "bootstrap_p": p,
        "n_boot": int(n_boot),
        "bootstrap_chi2_quantiles": {
            "q05": float(np.percentile(boot, 5)),
            "q50": float(np.percentile(boot, 50)),
            "q95": float(np.percentile(boot, 95)),
        },
        "scale": "W^2 seed-mean with per-L SEM, alpha fixed at 0.5",
    }


def task2_class_adequacy(args) -> Dict:
    data = load_wsat_perseed()
    out = {
        "five_tuple": (
            "Level-0, classes N=1/U=0.5/omega_min=0.3, "
            "N=1/U=4/omega_min=0.3, and additive-width alpha=0.5; "
            "design={32,48,64,96,128,192,256}; sigma=measured per-system; "
            "m=24"),
        "n_boot": args.gof_boot,
        "systems": {},
    }
    for si, s in enumerate(sorted(data)):
        y, sigma, per_l = y_sigma_from_wseeds(data[s])
        row = {"sigma": sigma, "per_L_sigma_logW": per_l, "classes": {}}
        for U in (0.5, 4.0):
            res = bootstrap_gof_logform(
                y, DESIGN, sigma, M_REAL, U,
                seed=853000 + si * 10000 + int(U * 100),
                n_boot=args.gof_boot,
                starts=args.gof_starts,
            )
            row["classes"][f"N1_U{U:g}"] = res
            print(f"[G-85b] {s} N1_U{U:g}: chi2={res['observed_chi2']:.3f} "
                  f"p={res['bootstrap_p']:.3f}")
        add = bootstrap_gof_additive(data[s], DESIGN,
                                     seed=854000 + si * 10000,
                                     n_boot=args.gof_boot)
        row["classes"]["additive_width_alpha_fixed_0p5"] = add
        print(f"[G-85b] {s} additive_fixed_alpha0.5: "
              f"chi2/dof={add['observed_chi2']/add['dof']:.3f} "
              f"p={add['bootstrap_p']:.3f}")
        out["systems"][s] = row

    bd_p = out["systems"]["bd"]["classes"][
        "additive_width_alpha_fixed_0p5"]["bootstrap_p"]
    power_anchor = bool(bd_p < 0.05)
    out["gates"] = {
        "BD_additive_width_alpha_fixed_0p5_p_lt_0p05": power_anchor,
        "pass": power_anchor,
    }
    if not power_anchor:
        write_json(os.path.join(RESULTS_DIR, "task2_gof.json"), out)
        raise RuntimeError(
            "G-85b power anchor failed: BD additive-width p >= 0.05")
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


def hull_corr_vec(Ls: np.ndarray, lam: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = np.log(Ls)
    x1 = x[0]
    atoms = np.log(1.0 + u[None, :] * np.exp(-w[None, :] * (x[:, None] - x1)))
    return atoms @ lam


@dataclass
class ModulusCurve:
    U: float
    design: List[float]
    da: np.ndarray
    delta: np.ndarray
    running_delta: np.ndarray
    omega_prime: np.ndarray
    mixes: List[Optional[Tuple]]

    def omega_at(self, delta_value: float) -> float:
        d = self.running_delta
        keep = np.r_[True, np.diff(d) > 1e-10]
        if delta_value <= d[keep][0]:
            return float(self.da[keep][0])
        if delta_value >= d[keep][-1]:
            return float(self.da[keep][-1])
        return float(np.interp(delta_value, d[keep], self.da[keep]))

    def to_json(self) -> Dict:
        return {
            "U": self.U,
            "design": self.design,
            "da": self.da.tolist(),
            "delta_raw": self.delta.tolist(),
            "delta_running_max": self.running_delta.tolist(),
            "omega_prime": self.omega_prime.tolist(),
        }


def compute_modulus_curve(
    U: float,
    Ls: np.ndarray,
    da_hi: float,
    n_grid: int,
    n_starts: int,
    seed: int,
) -> ModulusCurve:
    e83 = load_module(E83_PATH, f"e83_curve_U{str(U).replace('.', '_')}_{len(Ls)}")
    da_grid = np.linspace(0.0, da_hi, n_grid)
    deltas = np.zeros_like(da_grid)
    mixes: List[Optional[Tuple]] = [None]
    for j, da in enumerate(da_grid[1:], start=1):
        gap, mix = e83.hull_confusion_gap(
            float(da), Ls, J=4, U=float(U), n_starts=n_starts,
            seed=seed + j, maxiter=120, maxfun=900)
        deltas[j] = math.sqrt(max(float(gap), 0.0))
        mixes.append(mix)
        print(f"[curve] design_n={len(Ls)} U={U:g} da={da:.4f} "
              f"delta={deltas[j]:.6g}")
    running = np.maximum.accumulate(deltas)
    keep = np.r_[True, np.diff(running) > 1e-10]
    prime = np.zeros_like(da_grid)
    if np.sum(keep) >= 3:
        prime_keep = np.gradient(da_grid[keep], running[keep], edge_order=1)
        prime[keep] = prime_keep
        prime = np.interp(np.arange(len(prime)), np.where(keep)[0], prime[keep])
    else:
        prime[:] = 1.0
    prime = np.clip(prime, 1e-6, 1e6)
    return ModulusCurve(float(U), Ls.tolist(), da_grid, deltas, running, prime, mixes)


def cv_abs_normal_shift(t: float, alpha: float = 0.05) -> float:
    t = abs(float(t))

    def cdf_abs(c):
        return norm.cdf(c - t) - norm.cdf(-c - t)

    lo, hi = 0.0, max(2.0, t + norm.ppf(1.0 - alpha / 2.0) + 2.0)
    while cdf_abs(hi) < 1.0 - alpha:
        hi *= 1.5
    return float(brentq(lambda c: cdf_abs(c) - (1.0 - alpha), lo, hi))


def ak_half_length(curve: ModulusCurve, sigma: float, m: int,
                   alpha: float = 0.05) -> Dict:
    sigma_m = sigma / math.sqrt(m)
    vals = []
    for i in range(2, len(curve.da) - 2):
        delta = curve.running_delta[i]
        omega = curve.da[i]
        op = curve.omega_prime[i]
        if delta <= 0 or op <= 0 or not np.isfinite(op):
            continue
        t = omega / (2.0 * sigma_m * op) - delta / (2.0 * sigma_m)
        cv = cv_abs_normal_shift(t, alpha)
        half = cv * sigma_m * op
        if np.isfinite(half):
            vals.append((half, i, delta, omega, op, t, cv))
    if not vals:
        raise RuntimeError("No valid A-K half-length grid point")
    half, idx, delta, omega, op, t, cv = min(vals, key=lambda z: z[0])
    return {
        "half_length": float(half),
        "grid_index": int(idx),
        "delta_star": float(delta),
        "omega_delta_star": float(omega),
        "omega_prime_delta_star": float(op),
        "bias_over_sd_arg": float(t),
        "cv": float(cv),
        "sigma_m": float(sigma_m),
        "minimizer_interior": bool(2 < idx < len(curve.da) - 3),
    }


def ak_center(y: np.ndarray, curve: ModulusCurve, info: Dict) -> Dict:
    idx = int(info["grid_index"])
    mix = curve.mixes[idx]
    if mix is None:
        slope = np.polyfit(np.log(np.asarray(curve.design)), y, 1)[0]
        return {"center": float(slope), "fallback": "linear_slope_no_pair"}
    c, m1, m0 = mix
    lam1, u1, w1 = [np.asarray(v, dtype=float) for v in m1]
    lam0, u0, w0 = [np.asarray(v, dtype=float) for v in m0]
    Ls = np.asarray(curve.design, dtype=float)
    x = np.log(Ls)
    mu0 = hull_corr_vec(Ls, lam0, u0, w0)
    mu1 = curve.da[idx] * x + float(c) + hull_corr_vec(Ls, lam1, u1, w1)
    d = mu1 - mu0
    dist = float(np.linalg.norm(d))
    if dist <= 1e-12:
        slope = np.polyfit(x, y, 1)[0]
        return {"center": float(slope), "fallback": "zero_distance_pair"}
    op = float(info["omega_prime_delta_star"])
    wvec_raw = op * d / dist
    # Numerical extremal pairs from the hull optimizer do not always produce
    # weights with exact zero response to constants and unit response to the
    # target slope direction. Those constraints define the alpha functional:
    # a pure line c + alpha log(L) must map to alpha before correction
    # nuisances are considered. Project the raw pair direction onto that
    # affine-normalized subspace by the minimum Euclidean adjustment.
    C = np.vstack([np.ones_like(x), x]).T
    target = np.array([0.0, 1.0])
    gram = C.T @ C
    correction = C @ (np.linalg.solve(gram, target - C.T @ wvec_raw))
    wvec = wvec_raw + correction
    omega = float(curve.da[idx])
    bias = (omega - dist * op) / 2.0
    intercept_raw = bias - float(np.dot(wvec_raw, mu0))
    # With the normalized weights the natural pure-power intercept is zero.
    # Keep the raw A-K intercept for diagnostics, but center the reported
    # estimator on the alpha functional itself.
    intercept = 0.0
    center = intercept + float(np.dot(wvec, y))
    return {
        "center": float(center),
        "affine_intercept": float(intercept),
        "raw_affine_intercept": float(intercept_raw),
        "affine_weights": wvec.tolist(),
        "raw_affine_weights": wvec_raw.tolist(),
        "weight_norm": float(np.linalg.norm(wvec)),
        "raw_weight_norm": float(np.linalg.norm(wvec_raw)),
        "constraint_sum_weights": float(np.sum(wvec)),
        "constraint_logL_response": float(np.dot(wvec, x)),
        "pair_distance": dist,
        "pair_da": omega,
        "pair_bias": float(bias),
    }


def honest_ci_for_y(y: np.ndarray, sigma: float, m: int, curve: ModulusCurve) -> Dict:
    info = ak_half_length(curve, sigma, m)
    cen = ak_center(y, curve, info)
    center = cen["center"]
    half = info["half_length"]
    if "weight_norm" in cen:
        raw_norm = max(cen.get("raw_weight_norm", info["omega_prime_delta_star"]), 1e-12)
        half *= cen["weight_norm"] / raw_norm
    floor = curve.omega_at(sigma / math.sqrt(m))
    out = {
        "center": float(center),
        "lo": float(center - half),
        "hi": float(center + half),
        "half_length": float(half),
        "floor": float(floor),
        "half_length_over_floor": float(half / floor) if floor > 0 else None,
        "ak": info,
        "center_details": cen,
    }
    return out


def choose_omega_tilde_from_table(rng: np.random.Generator) -> float:
    if os.path.exists(HIER81):
        tab = read_json(HIER81).get("omega_eff_measurements", {})
        candidates = sorted({
            float(v["omega_tilde"]) for v in tab.values()
            if 0.3 <= float(v["omega_eff_measured"]) <= 2.5
        })
        if candidates:
            return float(rng.choice(candidates))
    return float(rng.uniform(0.3, 2.0))


def draw_truth_configs(seed: int = MASTER_SEED) -> List[Dict]:
    rng = np.random.default_rng(seed)
    cfgs = []
    for i in range(200):
        alpha = float(rng.uniform(0.3, 0.7))
        base = {
            "id": f"cfg_{i:03d}",
            "alpha": alpha,
            "D": 1.0,
            "nu": 1.0,
        }
        if i < 120:
            om = choose_omega_tilde_from_table(rng)
            u = float(rng.uniform(-0.6, 1.0))
            base.update({
                "kind": "in_class_single_power",
                "u": u,
                "omega_tilde": om,
                "declared_in_class_U": 1.0,
            })
        elif i < 160:
            w1 = float(rng.uniform(0.35, 1.2))
            w2 = float(rng.uniform(1.3, 2.5))
            u1 = float(rng.uniform(-0.35, 0.7))
            u2 = float(rng.uniform(-0.25, 0.45))
            base.update({
                "kind": "out_of_class_mild_twoterm",
                "u1": u1,
                "u2": u2,
                "omega1": w1,
                "omega2": w2,
            })
        else:
            if i % 2 == 0:
                base.update({
                    "kind": "out_of_class_hard_log",
                    "b_log": float(rng.uniform(-0.8, 2.5)),
                })
            else:
                base.update({
                    "kind": "out_of_class_hard_near_marginal",
                    "u": float(rng.uniform(0.4, 1.0)),
                    "omega_tilde": float(rng.uniform(0.10, 0.18)),
                })
        cfgs.append(base)
    return cfgs


def sample_fractional_W_matrix(cfg: Dict, seed: int, m: int = 24,
                               design: np.ndarray = DESIGN) -> np.ndarray:
    e81 = load_module(E81_PATH, "e81_sampler")
    rng = np.random.default_rng(seed)
    alpha = cfg["alpha"]
    z = e81.z_of_alpha(alpha)
    D = cfg.get("D", 1.0)
    nu = cfg.get("nu", 1.0)
    W = np.zeros((m, len(design)))

    if cfg["kind"] in ("in_class_single_power", "out_of_class_hard_near_marginal"):
        om = cfg["omega_tilde"]
        nu2 = e81.nu2_from_u(cfg["u"], z, D, nu, om, design=design)
        for j, L in enumerate(design):
            for s in range(m):
                h, _, _, _ = e81.sample_field(int(L), z, D, nu, nu2, om, rng)
                W[s, j] = math.sqrt(float(np.mean(h * h)))
        return W

    for j, L in enumerate(design):
        for s in range(m):
            h, _, _, _ = e81.sample_field(int(L), z, D, nu, 0.0, 1.0, rng)
            W[s, j] = math.sqrt(float(np.mean(h * h)))

    r = design / design[0]
    if cfg["kind"] == "out_of_class_mild_twoterm":
        corr = 1.0 + cfg["u1"] * r ** (-cfg["omega1"]) + cfg["u2"] * r ** (-cfg["omega2"])
        corr = np.clip(corr, 0.05, None)
        W *= corr[None, :]
    elif cfg["kind"] == "out_of_class_hard_log":
        corr = 1.0 + cfg["b_log"] / np.log(design)
        corr = np.clip(corr, 0.05, None)
        W *= corr[None, :]
    else:
        raise ValueError(cfg["kind"])
    return W


def make_blind_dataset(configs: List[Dict], seed: int = MASTER_SEED + 1) -> Dict:
    rows = []
    for i, cfg in enumerate(configs):
        W = sample_fractional_W_matrix(cfg, seed + i)
        rows.append({"id": cfg["id"], "W": W.tolist()})
        if (i + 1) % 25 == 0:
            print(f"[phase1] generated blind ladders for {i + 1}/200 configs")
    return {
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "master_seed": MASTER_SEED,
        "configs": rows,
        "truth_fields_present": False,
    }


def load_blind_data() -> Dict:
    return read_json(os.path.join(RESULTS_DIR, "blind_ladders.json"))


def analyze_blind_rows(rows: List[Dict], curves: Dict[str, ModulusCurve],
                       args, phase_label: str) -> List[Dict]:
    preds = []
    for i, row in enumerate(rows):
        W = np.asarray(row["W"], dtype=float)
        y, sigma, per_l = y_sigma_from_wseeds(W)
        pr = {
            "id": row["id"],
            "sigma": sigma,
            "per_L_sigma_logW": per_l,
            "classes": {},
        }
        for key, curve in curves.items():
            ci = honest_ci_for_y(y, sigma, W.shape[0], curve)
            U = float(key.replace("U", ""))
            # Smaller bootstrap budget in the 200-config blind loop; Task 2
            # carries the full 500-draw real-data adequacy gate.
            gof = bootstrap_gof_logform(
                y, np.asarray(curve.design, dtype=float), sigma, W.shape[0],
                U, seed=860000 + i * 1000 + int(U * 100),
                n_boot=args.main_gof_boot, starts=args.main_gof_starts)
            ci["adequacy_p"] = gof["bootstrap_p"]
            ci["adequacy_n_boot"] = gof["n_boot"]
            ci["adequacy_chi2"] = gof["observed_chi2"]
            pr["classes"][key] = ci
        preds.append(pr)
        if (i + 1) % 25 == 0:
            print(f"[{phase_label}] analyzed {i + 1}/{len(rows)} blind ladders")
    return preds


def calibration_pregate(curves: Dict[str, ModulusCurve], args) -> Dict:
    rng = np.random.default_rng(MASTER_SEED - 1)
    rows = []
    alphas = []
    for i in range(args.calibration_n):
        cfg = {
            "id": f"cal_{i:03d}",
            "kind": "pure_nu2_zero",
            "alpha": float(rng.uniform(0.3, 0.7)),
            "D": 1.0,
            "nu": 1.0,
        }
        # Pure sampling is a special case of the out-of-class path with nu2=0.
        e81 = load_module(E81_PATH, f"e81_cal_{i}")
        z = e81.z_of_alpha(cfg["alpha"])
        W = np.zeros((M_REAL, len(DESIGN)))
        srng = np.random.default_rng(MASTER_SEED - 10000 + i)
        for j, L in enumerate(DESIGN):
            for s in range(M_REAL):
                h, _, _, _ = e81.sample_field(int(L), z, 1.0, 1.0, 0.0, 1.0, srng)
                W[s, j] = math.sqrt(float(np.mean(h * h)))
        rows.append({"id": cfg["id"], "W": W.tolist()})
        alphas.append(cfg["alpha"])

    # Calibration checks coverage only; no adequacy bootstrap needed.
    out_by_class = {}
    for key, curve in curves.items():
        covered = []
        halfs = []
        for row, alpha in zip(rows, alphas):
            W = np.asarray(row["W"], dtype=float)
            y, sigma, _ = y_sigma_from_wseeds(W)
            ci = honest_ci_for_y(y, sigma, W.shape[0], curve)
            covered.append(ci["lo"] <= alpha <= ci["hi"])
            halfs.append(ci["half_length"])
        cov = float(np.mean(covered))
        n = len(covered)
        two_sigma = 2.0 * math.sqrt(0.95 * 0.05 / n)
        out_by_class[key] = {
            "n": n,
            "coverage": cov,
            "binomial_2sigma_window_around_0p95": [0.95 - two_sigma, 0.95 + two_sigma],
            "within_binomial_2sigma": bool(0.95 - two_sigma <= cov <= 0.95 + two_sigma),
            "median_half_length": float(np.median(halfs)),
        }
        print(f"[G-85c pre] {key}: pure coverage={cov:.3f} "
              f"window=[{0.95-two_sigma:.3f},{0.95+two_sigma:.3f}]")
    gate = all(v["within_binomial_2sigma"] for v in out_by_class.values())
    return {"classes": out_by_class, "gate_precalibration": gate}


def build_curves(args, design: np.ndarray = DESIGN,
                 classes: Tuple[float, ...] = (0.5, 1.0),
                 tag: str = "full") -> Dict[str, ModulusCurve]:
    curves = {}
    for U in classes:
        curve = compute_modulus_curve(
            U, design, args.curve_da_hi, args.curve_grid, args.curve_starts,
            seed=855000 + int(U * 1000) + len(design))
        curves[f"U{U:g}"] = curve
        write_json(os.path.join(RESULTS_DIR, f"modulus_{tag}_U{U:g}.json"),
                   curve.to_json())
    return curves


def task4_blind_predictions(args) -> Dict:
    bd_curve = build_curves(args, HALF_DESIGN_BD, classes=(0.5,), tag="bd_half")["U0.5"]
    data = load_wsat_perseed()
    Wbd = data["bd"][:, :len(HALF_DESIGN_BD)]
    y, sigma, per_l = y_sigma_from_wseeds(Wbd)
    bd_ci = honest_ci_for_y(y, sigma, Wbd.shape[0], bd_curve)
    bd_ci["sigma"] = sigma
    bd_ci["per_L_sigma_logW"] = per_l
    bd_ci["fit_window_L"] = HALF_DESIGN_BD.tolist()
    bd_ci["class"] = "N=1/U=0.5/omega_min=0.3"

    ising = {
        "status": "not_run_missing_persisted_half_window_ladder",
        "fit_window_L": [32, 48],
        "class": "Ising-honest |u|<=0.3, omega>=1",
        "note": (
            "results_exp52d_full stores only the full-window nu summary and "
            "PNG, not the per-L collapse observable needed for a blind "
            "half-window fit. The script records this before unblinding "
            "comparisons.")
    }
    return {
        "bd_half_window": bd_ci,
        "ising_half_window": ising,
        "truth_fields_present": False,
    }


def phase1(args) -> Dict:
    ensure_results_dir()
    curves = build_curves(args, DESIGN, classes=(0.5, 1.0), tag="fractional")
    pregate = calibration_pregate(curves, args)
    write_json(os.path.join(RESULTS_DIR, "task3_pregate.json"), pregate)
    if not pregate["gate_precalibration"]:
        write_json(os.path.join(RESULTS_DIR, "phase1_blocked.json"), {
            "blocked_at": "G-85c pre-gate",
            "reason": "pure nu2=0 calibration coverage outside binomial 2sigma window",
            "pregate": pregate,
            "predictions_written": False,
            "truth_fields_written": False,
        })
        raise RuntimeError("G-85c pre-gate failed: pure coverage outside binomial 2sigma")

    configs = draw_truth_configs(MASTER_SEED)
    blind = make_blind_dataset(configs)
    write_json(os.path.join(RESULTS_DIR, "blind_ladders.json"), blind)
    preds = analyze_blind_rows(blind["configs"], curves, args, "phase1")
    task4 = task4_blind_predictions(args)

    pred_obj = {
        "experiment": "85_preregistered_validation",
        "phase": "phase1_blind_predictions",
        "master_seed": MASTER_SEED,
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "classes": list(curves.keys()),
        "ci_machinery_source": (
            "local A-K fixed-length CI implementation; "
            "results_exp84_presubmission was absent"),
        "task3_pregate_summary": pregate,
        "task3_predictions": preds,
        "task4_blind_predictions": task4,
        "truth_fields_present": False,
    }
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    write_json(pred_path, pred_obj)
    digest = sha256_file(pred_path)
    with open(os.path.join(RESULTS_DIR, "predictions.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  predictions.json\n")
    print(f"[phase1] wrote predictions.json sha256={digest}")
    return pred_obj


def phase2(args) -> Dict:
    require_phase1_committed()
    pred = read_json(os.path.join(RESULTS_DIR, "predictions.json"))
    actual_digest = sha256_file(os.path.join(RESULTS_DIR, "predictions.json"))
    sha_text = open(os.path.join(RESULTS_DIR, "predictions.sha256"), encoding="utf-8").read()
    recorded_digest = sha_text.split()[0]
    if actual_digest != recorded_digest:
        raise SystemExit("Refusing to unblind: predictions.sha256 does not match predictions.json")

    cfgs = {c["id"]: c for c in draw_truth_configs(MASTER_SEED)}
    scores = {"phase1_commit_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "predictions_sha256": actual_digest,
        "task3": {},
        "task4": {},
    }

    for key in pred["classes"]:
        rows = []
        for pr in pred["task3_predictions"]:
            cfg = cfgs[pr["id"]]
            ci = pr["classes"][key]
            covered = bool(ci["lo"] <= cfg["alpha"] <= ci["hi"])
            rows.append({
                "id": pr["id"],
                "kind": cfg["kind"],
                "truth_alpha": cfg["alpha"],
                "lo": ci["lo"],
                "hi": ci["hi"],
                "center": ci["center"],
                "covered": covered,
                "adequacy_p": ci["adequacy_p"],
                "adequacy_pass_0p05": bool(ci["adequacy_p"] >= 0.05),
                "half_length_over_floor": ci["half_length_over_floor"],
            })
        in_rows = [r for r in rows if r["kind"] == "in_class_single_power"]
        out_rows = [r for r in rows if r["kind"] != "in_class_single_power"]
        in_cov = float(np.mean([r["covered"] for r in in_rows]))
        out_cov = float(np.mean([r["covered"] for r in out_rows]))
        n_in = len(in_rows)
        two_sigma = 2.0 * math.sqrt(0.95 * 0.05 / n_in)
        fals = {}
        for ap in (False, True):
            for cov in (False, True):
                fals[f"adequacy_{'pass' if ap else 'fail'}__{'cover' if cov else 'miss'}"] = int(
                    sum((r["adequacy_pass_0p05"] == ap) and (r["covered"] == cov)
                        for r in out_rows))
        scores["task3"][key] = {
            "in_class_n": n_in,
            "in_class_coverage": in_cov,
            "in_class_binomial_2sigma_window_around_0p95": [0.95 - two_sigma, 0.95 + two_sigma],
            "in_class_coverage_gate_within_2sigma": bool(0.95 - two_sigma <= in_cov <= 0.95 + two_sigma),
            "out_of_class_n": len(out_rows),
            "out_of_class_coverage": out_cov,
            "falsifiability_2x2_out_of_class": fals,
            "half_length_over_floor_quantiles": {
                "q10": float(np.percentile([r["half_length_over_floor"] for r in rows], 10)),
                "q50": float(np.percentile([r["half_length_over_floor"] for r in rows], 50)),
                "q90": float(np.percentile([r["half_length_over_floor"] for r in rows], 90)),
            },
            "per_config_scores": rows,
        }
        print(f"[phase2] {key}: in_cov={in_cov:.3f} out_cov={out_cov:.3f} "
              f"gate={scores['task3'][key]['in_class_coverage_gate_within_2sigma']}")

    t4 = pred["task4_blind_predictions"]
    bd = t4["bd_half_window"]
    summary = read_json(SUMMARY76)
    bd_full = summary["real_systems"]["bd"]["alpha_hat_mix"]
    scores["task4"]["bd_half_window"] = {
        "prediction_lo": bd["lo"],
        "prediction_hi": bd["hi"],
        "prediction_center": bd["center"],
        "exact_alpha_reference": 0.5,
        "full_window_exp76_alpha_hat_mix": bd_full,
        "covers_exact_0p5": bool(bd["lo"] <= 0.5 <= bd["hi"]),
        "covers_full_window_exp76": bool(bd["lo"] <= bd_full <= bd["hi"]),
    }
    if os.path.exists(ISING52D):
        is52 = read_json(ISING52D)
        nu = is52["results"]["nu_optimal"]
        scores["task4"]["ising_half_window"] = {
            "prediction_status": t4["ising_half_window"]["status"],
            "exact_1_over_nu": 1.0,
            "full_window_1_over_nu_fit": float(1.0 / nu),
            "source": "results_exp52d_full/results.json",
        }
    write_json(os.path.join(RESULTS_DIR, "score.json"), scores)
    return scores


def run_tasks12(args) -> Dict:
    ensure_results_dir()
    t1 = task1_instance_local(args)
    write_json(os.path.join(RESULTS_DIR, "task1_local_floor.json"), t1)
    t2 = task2_class_adequacy(args)
    write_json(os.path.join(RESULTS_DIR, "task2_gof.json"), t2)
    return {"task1": t1, "task2": t2}


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{float(x):.{nd}f}"


def write_report() -> None:
    t1 = read_json(os.path.join(RESULTS_DIR, "task1_local_floor.json"))
    t2 = read_json(os.path.join(RESULTS_DIR, "task2_gof.json"))
    pregate = read_json(os.path.join(RESULTS_DIR, "task3_pregate.json"))
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    pred = read_json(pred_path) if os.path.exists(pred_path) else None
    score_path = os.path.join(RESULTS_DIR, "score.json")
    score = read_json(score_path) if os.path.exists(score_path) else None

    lines = []
    lines.append("# Exp 85 Report -- Preregistered Validation")
    lines.append("")
    lines.append("Findings only. `CLAIMS_REGISTER.md` was not edited.")
    lines.append("")
    lines.append("## Gate Ledger")
    lines.append("")
    lines.append("| Gate | Check | Result | Proof path |")
    lines.append("|---|---|---|---|")
    for s, row in t1["systems"].items():
        g = row["gates"]
        ok = g["local_floor_le_global_honest_floor"] and g["global_honest_D2_monotone"] and g["local_D2_monotone"]
        lines.append(f"| G-85a {s} | local floor <= honest global floor; D2 monotone | {'met' if ok else 'not met'} | `results_exp85_preregistered/task1_local_floor.json` |")
    g2 = t2["gates"]["BD_additive_width_alpha_fixed_0p5_p_lt_0p05"]
    lines.append(f"| G-85b | BD additive-width alpha=0.5 p < 0.05 power anchor | {'met' if g2 else 'not met'} | `results_exp85_preregistered/task2_gof.json` |")
    for key, row in pregate["classes"].items():
        lines.append(f"| G-85c pre {key} | pure nu2=0 95% CI coverage within binomial 2sigma | {'met' if row['within_binomial_2sigma'] else 'not met'} | `results_exp85_preregistered/task3_pregate.json` |")
    if score:
        for key, row in score["task3"].items():
            lines.append(f"| G-85c main {key} | in-class coverage within binomial 2sigma of 0.95 | {'met' if row['in_class_coverage_gate_within_2sigma'] else 'not met'} | `results_exp85_preregistered/score.json` |")
    if pred:
        lines.append(f"| Blinding | predictions hash committed before scoring | recorded | `results_exp85_preregistered/predictions.sha256`; phase-2 HEAD `{score['phase1_commit_head'] if score else 'pending'}` |")
    else:
        lines.append("| Blinding | predictions hash committed before scoring | not reached | G-85c pre-gate blocked phase 1 before `predictions.json` was written |")
    lines.append("")

    lines.append("## Task 1 Decision Table")
    lines.append("")
    lines.append(t1["five_tuple"])
    lines.append("")
    lines.append("| System | sigma | fitted u | honest U | global U=4 floor | honest global floor | instance-local floor | exp76 abs err | fit_w1 abs err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, row in t1["systems"].items():
        obs = row["observed_errors"]
        fit_w1 = obs["exp75_or_classical_abs_errors"].get("fit_w1")
        lines.append(
            f"| {s} | {fmt(row['sigma'])} | {fmt(row['fit_declared_N1_U4']['u'])} | "
            f"{fmt(row['honest_U_2x_abs_fit_u'])} | {fmt(row['global_U4_floor_exp77'])} | "
            f"{fmt(row['honest_class_global_floor'])} | {fmt(row['instance_local_floor'])} | "
            f"{fmt(obs['exp76_mix_abs_error'])} | {fmt(fit_w1)} |")
    lines.append("")

    lines.append("## Task 2 Adequacy")
    lines.append("")
    lines.append(t2["five_tuple"])
    lines.append("")
    lines.append("| System | N1 U=0.5 p | N1 U=4 p | additive alpha=0.5 p | additive chi2/dof |")
    lines.append("|---|---:|---:|---:|---:|")
    for s, row in t2["systems"].items():
        c = row["classes"]
        add = c["additive_width_alpha_fixed_0p5"]
        lines.append(
            f"| {s} | {fmt(c['N1_U0.5']['bootstrap_p'], 3)} | "
            f"{fmt(c['N1_U4']['bootstrap_p'], 3)} | "
            f"{fmt(add['bootstrap_p'], 3)} | {fmt(add['observed_chi2']/add['dof'], 2)} |")
    lines.append("")

    lines.append("## Task 3 Coverage")
    lines.append("")
    lines.append("Five-tuple: Level-0 fractional-EW seed log-width ladder, class N=1 hull J=4 with U in {0.5,1}, omega_min=0.3, design={32,48,64,96,128,192,256}, sigma=config-measured median sd(log W), m=24.")
    lines.append("")
    lines.append("| Class | Pre-gate coverage | Pre-gate window | Main in-class coverage | Main window | Out-of-class coverage | median half/floor |")
    lines.append("|---|---:|---|---:|---|---:|---:|")
    for key, pre in pregate["classes"].items():
        if score:
            sc = score["task3"][key]
            lines.append(
                f"| {key} | {fmt(pre['coverage'],3)} | "
                f"[{fmt(pre['binomial_2sigma_window_around_0p95'][0],3)}, {fmt(pre['binomial_2sigma_window_around_0p95'][1],3)}] | "
                f"{fmt(sc['in_class_coverage'],3)} | "
                f"[{fmt(sc['in_class_binomial_2sigma_window_around_0p95'][0],3)}, {fmt(sc['in_class_binomial_2sigma_window_around_0p95'][1],3)}] | "
                f"{fmt(sc['out_of_class_coverage'],3)} | {fmt(sc['half_length_over_floor_quantiles']['q50'],2)} |")
        else:
            lines.append(
                f"| {key} | {fmt(pre['coverage'],3)} | "
                f"[{fmt(pre['binomial_2sigma_window_around_0p95'][0],3)}, {fmt(pre['binomial_2sigma_window_around_0p95'][1],3)}] | "
                "pending | pending | pending | pending |")
    lines.append("")
    if score:
        lines.append("### Falsifiability Table")
        lines.append("")
        for key, sc in score["task3"].items():
            f = sc["falsifiability_2x2_out_of_class"]
            lines.append(f"{key}: adequacy pass/cover={f['adequacy_pass__cover']}, pass/miss={f['adequacy_pass__miss']}, fail/cover={f['adequacy_fail__cover']}, fail/miss={f['adequacy_fail__miss']}.")
        lines.append("")
    else:
        lines.append("Main blind coverage and falsifiability tables were not generated because the pre-gate stopped the run before `predictions.json`.")
        lines.append("")

    lines.append("## Task 4 Real-Data Half-Window")
    lines.append("")
    lines.append("Five-tuple BD: Level-0 real BD log-width ladder, class N=1/U=0.5/omega_min=0.3, design={32,48,64,96}, sigma=BD half-window measured median sd(log W), m=24.")
    lines.append("")
    if pred:
        bd = pred["task4_blind_predictions"]["bd_half_window"]
        if score:
            bds = score["task4"]["bd_half_window"]
            lines.append(
                f"BD blind CI before unblinding: [{fmt(bd['lo'])}, {fmt(bd['hi'])}], center {fmt(bd['center'])}, half-length {fmt(bd['half_length'])}. "
                f"After unblinding: exact reference 0.5 covered={fmt(bds['covers_exact_0p5'])}; full-window exp76 value {fmt(bds['full_window_exp76_alpha_hat_mix'])} covered={fmt(bds['covers_full_window_exp76'])}.")
        else:
            lines.append(f"BD blind CI before unblinding: [{fmt(bd['lo'])}, {fmt(bd['hi'])}], center {fmt(bd['center'])}, half-length {fmt(bd['half_length'])}.")
    else:
        lines.append("BD half-window blind prediction was not generated because G-85c pre-gate blocked phase 1 before Task 4 prediction output.")
    lines.append("")
    if pred:
        ising = pred["task4_blind_predictions"]["ising_half_window"]
        lines.append(f"Ising half-window status: {ising['status']}.")
    else:
        lines.append("Ising half-window status: not reached.")
    if score and "ising_half_window" in score["task4"]:
        isg = score["task4"]["ising_half_window"]
        lines.append(f"Stored full-window comparison only: exact 1/nu={fmt(isg['exact_1_over_nu'])}; full-window 1/nu fit={fmt(isg['full_window_1_over_nu_fit'])}.")
    lines.append("")

    lines.append("## What We Did Not Do")
    lines.append("")
    lines.append("- No entry was added to `CLAIMS_REGISTER.md`.")
    lines.append("- `results_exp84_presubmission/` was absent in this checkout, so the A-K fixed-length CI machinery was implemented locally in `experiments/85_preregistered_validation.py`.")
    if pred:
        lines.append("- The Ising half-window blind fit was not run because the exp52d artifact stores only the final nu summary and figure, not the per-L collapse observable required for fitting on L={32,48}.")
        lines.append("- Task 3 adequacy p-values used a smaller bootstrap budget than Task 2; the per-config budget is recorded in `predictions.json`.")
    else:
        lines.append("- `predictions.json`, `predictions.sha256`, blind main coverage, falsifiability scoring, and real-data half-window predictions were not produced because G-85c pre-gate failed.")
    lines.append("")
    lines.append("## Anomalies And Bugs")
    lines.append("")
    lines.append("- G-85c pre-gate failed by overcoverage after the affine center was corrected to have zero response to constants and unit response to `log L`: U=0.5 coverage 1.000 and U=1 coverage 1.000, outside the preregistered [0.919, 0.981] window.")
    lines.append("- Before the affine-normalization fix, the U=1 calibration undercovered (0.830), revealing that raw extremal-pair weights did not define a properly normalized alpha estimator. The script now records normalized and raw weights.")
    if pred:
        lines.append("- The blinding procedure writes no truth fields to `predictions.json`; phase 2 reconstructs the deterministic truth configs only after checking that `predictions.json` and `predictions.sha256` exist in `HEAD`.")
        lines.append("- Out-of-class two-term and log-form fractional-EW ladders use exact pure fractional-EW sampling for the stochastic field and then apply deterministic correction factors; only the in-class and near-marginal single-power cases use `nu2_from_u` inside the fractional-EW spectrum.")
        lines.append("- The A-K affine center uses the nearest computed modulus-grid extremal pair at the selected delta; no continuous interpolation of the pair itself was attempted.")

    report_path = os.path.join(ROOT, "ml_paper", "EXP85_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["task1", "task2", "tasks12", "phase1", "phase2", "report", "all"],
                    default="all")
    ap.add_argument("--fit-starts", type=int, default=14)
    ap.add_argument("--gap-starts", type=int, default=8)
    ap.add_argument("--task1-grid", type=int, default=31)
    ap.add_argument("--task1-da-hi", type=float, default=0.8)
    ap.add_argument("--nesting-tol", type=float, default=0.02)
    ap.add_argument("--gof-boot", type=int, default=500)
    ap.add_argument("--gof-starts", type=int, default=8)
    ap.add_argument("--curve-grid", type=int, default=35)
    ap.add_argument("--curve-da-hi", type=float, default=1.2)
    ap.add_argument("--curve-starts", type=int, default=6)
    ap.add_argument("--calibration-n", type=int, default=200)
    ap.add_argument("--main-gof-boot", type=int, default=80)
    ap.add_argument("--main-gof-starts", type=int, default=4)
    args = ap.parse_args()

    if args.stage == "task1":
        ensure_results_dir()
        write_json(os.path.join(RESULTS_DIR, "task1_local_floor.json"),
                   task1_instance_local(args))
    if args.stage == "task2":
        ensure_results_dir()
        write_json(os.path.join(RESULTS_DIR, "task2_gof.json"),
                   task2_class_adequacy(args))
    if args.stage in ("tasks12", "all"):
        run_tasks12(args)
    if args.stage in ("phase1", "all"):
        phase1(args)
    if args.stage in ("phase2", "all"):
        phase2(args)
    if args.stage in ("report", "all"):
        write_report()


if __name__ == "__main__":
    main()
