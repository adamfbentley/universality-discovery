"""Exp 85b: corrected A-K validation and blind scoring.

The phase boundary is part of the experiment:

  1. tasks_abc: A-K construction gates, corrected calibration pregates,
     identified-set profiles, floor curves, and truth-pinned local floors.
  2. phase1: blind predictions only, plus SHA256.
  3. phase2: only after predictions are committed in HEAD, reconstruct truth
     configs and score.
  4. report: findings-only ledger.

No truth fields are written to predictions.json.
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
RESULTS_DIR = os.path.join(ROOT, "results_exp85b_validation")

WSAT_CSV = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "wsat_perseed.csv")
SUMMARY76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "summary_full24seed.json")
ISING52D = os.path.join(ROOT, "results_exp52d_full", "results.json")
HIER81 = os.path.join(ROOT, "results_exp81_hierarchy", "floors_hierarchy.json")

E85_PATH = os.path.join(HERE, "85_preregistered_validation.py")
E81_PATH = os.path.join(HERE, "81_fractional_ew_testbed.py")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
BD_HALF_DESIGN = np.array([32., 48., 64., 96.])
M_REAL = 24
OMEGA_BOUNDS = (0.3, 2.5)
MASTER_SEED = 85200
TARGET_MID_ALPHA = 0.5


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_e85():
    return load_module(E85_PATH, "e85b_reuse_85")


def get_e81():
    return load_module(E81_PATH, "e85b_reuse_81")


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
        os.path.join("results_exp85b_validation", "predictions.json"),
        os.path.join("results_exp85b_validation", "predictions.sha256"),
    ]
    missing = [p for p in rels if not git_blob_exists(p)]
    if missing:
        raise SystemExit(
            "Refusing to unblind: phase-1 prediction artifacts are not "
            f"committed in HEAD: {missing}")


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{float(x):.{nd}f}"


def u_bounds_for_class(U: float) -> Tuple[float, float]:
    return (-min(float(U), 0.95), float(U))


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


def hull_corr_vec(Ls: np.ndarray, lam: np.ndarray, u: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
    x = np.log(Ls)
    x1 = x[0]
    atoms = np.log(1.0 + u[None, :] * np.exp(-w[None, :] * (x[:, None] - x1)))
    return atoms @ lam


def corr_log_scalar(Ls: np.ndarray, u: float, w: float) -> np.ndarray:
    val = 1.0 + u * (Ls / Ls[0]) ** (-w)
    if np.any(val <= 0):
        return np.full_like(Ls, np.nan, dtype=float)
    return np.log(val)


def logform_mu(Ls: np.ndarray, c: float, alpha: float, u: float,
               w: float) -> np.ndarray:
    g = corr_log_scalar(Ls, u, w)
    if np.any(~np.isfinite(g)):
        return np.full_like(Ls, np.nan, dtype=float)
    return c + alpha * np.log(Ls) + g


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


def sigma_by_real_system() -> Dict[str, float]:
    return {s: y_sigma_from_wseeds(W)[1] for s, W in load_wsat_perseed().items()}


def _duplicate_mixture(lam, u, w, J_new):
    while len(lam) < J_new:
        lam = np.concatenate([lam, lam]) / 2.0
        u = np.concatenate([u, u])
        w = np.concatenate([w, w])
    return lam, u, w


def hull_confusion_gap(
    da: float,
    Ls: np.ndarray,
    J: int,
    U: float,
    n_starts: int,
    seed: int,
    warm_mixture=None,
    maxiter: int = 160,
    maxfun: int = 1200,
) -> Tuple[float, Tuple]:
    """Convex-hull modulus gap with exp85b's U bounds.

    The objective is ||da*x + c + g_1 - g_0||^2. The returned mixture is
    (c, high_correction, low_correction), matching that sign convention.
    """
    if warm_mixture is None and J > 1:
        _, warm_mixture = hull_confusion_gap(
            da, Ls, J // 2, U, n_starts=max(1, n_starts // 2), seed=seed,
            maxiter=maxiter, maxfun=maxfun)

    x = np.log(Ls)
    x1 = x[0]
    rng = np.random.default_rng(seed)
    u_bounds = u_bounds_for_class(U)

    def unpack(p):
        idx = 0
        c = p[idx]
        idx += 1
        lam1 = softmax(p[idx:idx + J])
        idx += J
        u1 = p[idx:idx + J]
        idx += J
        w1 = p[idx:idx + J]
        idx += J
        lam0 = softmax(p[idx:idx + J])
        idx += J
        u0 = p[idx:idx + J]
        idx += J
        w0 = p[idx:idx + J]
        return c, lam1, u1, w1, lam0, u0, w0

    def hull_corr_x(xi, lam, u, w):
        atom_arg = 1.0 + u * np.exp(-w * (xi - x1))
        if np.any(atom_arg <= 0):
            return np.nan
        return float(np.dot(lam, np.log(atom_arg)))

    def obj(p):
        c, lam1, u1, w1, lam0, u0, w0 = unpack(p)
        total = 0.0
        for xi in x:
            g1 = hull_corr_x(xi, lam1, u1, w1)
            g0 = hull_corr_x(xi, lam0, u0, w0)
            if not np.isfinite(g1) or not np.isfinite(g0):
                return 1e30
            diff = da * xi + c + g1 - g0
            total += diff * diff
        return float(total)

    bounds = ([(None, None)] + [(None, None)] * J + [u_bounds] * J
              + [OMEGA_BOUNDS] * J + [(None, None)] * J
              + [u_bounds] * J + [OMEGA_BOUNDS] * J)
    starts = []
    if warm_mixture is not None:
        c_w, (lam1w, u1w, w1w), (lam0w, u0w, w0w) = warm_mixture
        lam1w, u1w, w1w = _duplicate_mixture(
            np.asarray(lam1w), np.asarray(u1w), np.asarray(w1w), J)
        lam0w, u0w, w0w = _duplicate_mixture(
            np.asarray(lam0w), np.asarray(u0w), np.asarray(w0w), J)
        starts.append(np.concatenate([
            [c_w],
            np.log(np.clip(lam1w, 1e-12, None)),
            np.clip(u1w, *u_bounds),
            np.clip(w1w, *OMEGA_BOUNDS),
            np.log(np.clip(lam0w, 1e-12, None)),
            np.clip(u0w, *u_bounds),
            np.clip(w0w, *OMEGA_BOUNDS),
        ]))

    starts.append(np.concatenate([
        [-da * x.mean()],
        np.zeros(J), np.zeros(J), np.ones(J),
        np.zeros(J), np.zeros(J), np.ones(J),
    ]))
    for _ in range(n_starts):
        starts.append(np.concatenate([
            [-da * x.mean() + rng.normal(0, 0.4)],
            rng.normal(0, 1, J),
            rng.uniform(*u_bounds, J),
            rng.uniform(*OMEGA_BOUNDS, J),
            rng.normal(0, 1, J),
            rng.uniform(*u_bounds, J),
            rng.uniform(*OMEGA_BOUNDS, J),
        ]))

    best = np.inf
    best_p = None
    for p0 in starts:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxfun": maxfun})
        if float(res.fun) < best:
            best = float(res.fun)
            best_p = res.x

    c, lam1, u1, w1, lam0, u0, w0 = unpack(best_p)
    mix = (
        float(c),
        (np.asarray(lam1), np.asarray(u1), np.asarray(w1)),
        (np.asarray(lam0), np.asarray(u0), np.asarray(w0)),
    )
    return best, mix


def mixture_to_json(mix):
    if mix is None:
        return None
    c, high, low = mix
    return {
        "c": float(c),
        "high": {
            "lambda": np.asarray(high[0], dtype=float).tolist(),
            "u": np.asarray(high[1], dtype=float).tolist(),
            "omega": np.asarray(high[2], dtype=float).tolist(),
        },
        "low": {
            "lambda": np.asarray(low[0], dtype=float).tolist(),
            "u": np.asarray(low[1], dtype=float).tolist(),
            "omega": np.asarray(low[2], dtype=float).tolist(),
        },
    }


def mixture_from_json(obj):
    if obj is None:
        return None
    return (
        float(obj["c"]),
        (
            np.asarray(obj["high"]["lambda"], dtype=float),
            np.asarray(obj["high"]["u"], dtype=float),
            np.asarray(obj["high"]["omega"], dtype=float),
        ),
        (
            np.asarray(obj["low"]["lambda"], dtype=float),
            np.asarray(obj["low"]["u"], dtype=float),
            np.asarray(obj["low"]["omega"], dtype=float),
        ),
    )


@dataclass
class ModulusCurve:
    U: float
    design: List[float]
    da: np.ndarray
    delta: np.ndarray
    running_delta: np.ndarray
    omega_prime: np.ndarray
    mixes: List[Optional[Tuple]]
    J: int

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
            "J": self.J,
            "design": self.design,
            "da": self.da.tolist(),
            "delta_raw": self.delta.tolist(),
            "delta_running_max": self.running_delta.tolist(),
            "omega_prime": self.omega_prime.tolist(),
            "mixes": [mixture_to_json(m) for m in self.mixes],
        }

    @staticmethod
    def from_json(obj: Dict):
        mixes_obj = obj.get("mixes")
        if mixes_obj is None:
            mixes = [None for _ in obj["da"]]
        else:
            mixes = [mixture_from_json(m) for m in mixes_obj]
        return ModulusCurve(
            U=float(obj["U"]),
            J=int(obj.get("J", 4)),
            design=[float(v) for v in obj["design"]],
            da=np.asarray(obj["da"], dtype=float),
            delta=np.asarray(obj["delta_raw"], dtype=float),
            running_delta=np.asarray(obj["delta_running_max"], dtype=float),
            omega_prime=np.asarray(obj["omega_prime"], dtype=float),
            mixes=mixes,
        )


def compute_modulus_curve(
    U: float,
    Ls: np.ndarray,
    da_hi: float,
    n_grid: int,
    n_starts: int,
    seed: int,
    J: int = 4,
) -> ModulusCurve:
    da_grid = np.linspace(0.0, da_hi, n_grid)
    deltas = np.zeros_like(da_grid)
    mixes: List[Optional[Tuple]] = [None]
    warm = None
    for j, da in enumerate(da_grid[1:], start=1):
        gap, mix = hull_confusion_gap(
            float(da), Ls, J=J, U=float(U), n_starts=n_starts,
            seed=seed + j, warm_mixture=warm)
        deltas[j] = math.sqrt(max(float(gap), 0.0))
        mixes.append(mix)
        warm = mix
        print(f"[modulus] n={len(Ls)} U={U:g} da={da:.4f} "
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
    return ModulusCurve(float(U), Ls.tolist(), da_grid, deltas, running,
                        prime, mixes, J)


def curve_path(tag: str, U: float) -> str:
    return os.path.join(RESULTS_DIR, f"modulus_{tag}_U{U:g}.json")


def build_or_load_curve(args, U: float, Ls: np.ndarray, tag: str) -> ModulusCurve:
    path = curve_path(tag, U)
    if args.reuse_curves and os.path.exists(path):
        return ModulusCurve.from_json(read_json(path))
    legacy = os.path.join(ROOT, "results_exp85_preregistered",
                          f"modulus_{tag}_U{U:g}.json")
    if args.reuse_curves and tag == "fractional" and np.array_equal(Ls, DESIGN) and os.path.exists(legacy):
        curve = ModulusCurve.from_json(read_json(legacy))
        curve.J = args.hull_J
        write_json(path, curve.to_json())
        print(f"[modulus] reused exp85 grid for {tag} U={U:g}; extremal pairs are lazy")
        return curve
    curve = compute_modulus_curve(
        U, Ls, args.curve_da_hi, args.curve_grid, args.curve_starts,
        seed=865000 + int(1000 * U) + len(Ls), J=args.hull_J)
    write_json(path, curve.to_json())
    return curve


def cv_abs_normal_shift(t: float, alpha: float = 0.05) -> float:
    t = abs(float(t))

    def cdf_abs(c):
        return norm.cdf(c - t) - norm.cdf(-c - t)

    lo = 0.0
    hi = max(2.0, t + norm.ppf(1.0 - alpha / 2.0) + 2.0)
    while cdf_abs(hi) < 1.0 - alpha:
        hi *= 1.5
    return float(brentq(lambda c: cdf_abs(c) - (1.0 - alpha), lo, hi))


def ak_half_length(curve: ModulusCurve, sigma: float, m: int,
                   alpha: float = 0.05) -> Dict:
    sigma_m = sigma / math.sqrt(m)
    vals = []
    for i in range(2, len(curve.da) - 2):
        delta = float(curve.running_delta[i])
        omega = float(curve.da[i])
        op = float(curve.omega_prime[i])
        if delta <= 0 or op <= 0 or not np.isfinite(op):
            continue
        bias = (omega - delta * op) / 2.0
        if bias < -1e-10:
            continue
        t = bias / max(sigma_m * op, 1e-15)
        cv = cv_abs_normal_shift(t, alpha)
        half = cv * sigma_m * op
        if np.isfinite(half):
            vals.append((half, i, delta, omega, op, bias, t, cv))
    if not vals:
        raise RuntimeError("No valid A-K half-length grid point")
    half, idx, delta, omega, op, bias, t, cv = min(vals, key=lambda z: z[0])
    return {
        "half_length": float(half),
        "grid_index": int(idx),
        "delta_star": float(delta),
        "omega_delta_star": float(omega),
        "omega_prime_delta_star": float(op),
        "pair_bias_bound": float(bias),
        "bias_over_sd_arg": float(t),
        "cv": float(cv),
        "sigma_m": float(sigma_m),
        "minimizer_interior": bool(2 < idx < len(curve.da) - 3),
    }


def extremal_pair_means(curve: ModulusCurve, info: Dict,
                        target_mid: float = TARGET_MID_ALPHA) -> Dict:
    idx = int(info["grid_index"])
    mix = curve.mixes[idx]
    if mix is None:
        da = float(curve.da[idx])
        print(f"[modulus] lazy extremal pair U={curve.U:g} idx={idx} da={da:.4f}")
        _, mix = hull_confusion_gap(
            da, np.asarray(curve.design, dtype=float), J=curve.J, U=curve.U,
            n_starts=6, seed=890000 + int(curve.U * 1000) + idx,
            maxiter=140, maxfun=1000)
        curve.mixes[idx] = mix
    c, high, low = mix
    Ls = np.asarray(curve.design, dtype=float)
    x = np.log(Ls)
    omega = float(curve.da[idx])
    target_low = target_mid - 0.5 * omega
    target_high = target_mid + 0.5 * omega
    lam_hi, u_hi, w_hi = [np.asarray(v, dtype=float) for v in high]
    lam_lo, u_lo, w_lo = [np.asarray(v, dtype=float) for v in low]
    corr_low = hull_corr_vec(Ls, lam_lo, u_lo, w_lo)
    corr_high = hull_corr_vec(Ls, lam_hi, u_hi, w_hi)
    mu_low = target_low * x + corr_low
    mu_high = target_high * x + float(c) + corr_high
    return {
        "mu_low": mu_low,
        "mu_high": mu_high,
        "target_low": float(target_low),
        "target_high": float(target_high),
        "pair_da": float(omega),
        "c_high_minus_low": float(c),
    }


def affine_estimators(curve: ModulusCurve, info: Dict,
                      target_mid: float = TARGET_MID_ALPHA) -> Dict:
    pair = extremal_pair_means(curve, info, target_mid=target_mid)
    Ls = np.asarray(curve.design, dtype=float)
    x = np.log(Ls)
    d = pair["mu_high"] - pair["mu_low"]
    dist = float(np.linalg.norm(d))
    if dist <= 1e-12:
        raise RuntimeError("Degenerate extremal pair distance")
    op_raw = float(info["omega_prime_delta_star"])
    direction = d / dist
    pure_resp_per_unit = float(np.dot(direction, x))
    op = op_raw
    cap_target = 0.995
    stabilized = False
    if pure_resp_per_unit > 0 and op_raw * pure_resp_per_unit > cap_target:
        op = cap_target / pure_resp_per_unit
        stabilized = True
    info_eff = dict(info)
    if stabilized:
        bias_eff = (float(curve.da[int(info["grid_index"])]) - dist * op) / 2.0
        sigma_m = float(info["sigma_m"])
        t_eff = bias_eff / max(sigma_m * op, 1e-15)
        cv_eff = cv_abs_normal_shift(t_eff)
        info_eff.update({
            "omega_prime_delta_star_raw": op_raw,
            "omega_prime_delta_star": float(op),
            "pair_bias_bound": float(bias_eff),
            "bias_over_sd_arg": float(t_eff),
            "cv": float(cv_eff),
            "half_length": float(cv_eff * sigma_m * op),
            "slope_response_cap_applied": True,
            "slope_response_cap_target": cap_target,
        })
    else:
        info_eff["slope_response_cap_applied"] = False
    w_ak = op * d / dist
    bias = float(info_eff["pair_bias_bound"])
    intercept_ak = pair["target_low"] + bias - float(np.dot(w_ak, pair["mu_low"]))

    C = np.vstack([np.ones_like(x), x]).T
    target = np.array([0.0, 1.0])
    correction = C @ np.linalg.solve(C.T @ C, target - C.T @ w_ak)
    w_gls = w_ak + correction
    intercept_gls = 0.0
    raw_norm = float(np.linalg.norm(w_ak))
    gls_norm = float(np.linalg.norm(w_gls))
    gls_half = float(info_eff["half_length"] * gls_norm / max(raw_norm, 1e-15))

    return {
        "ak": {
            "affine_intercept": float(intercept_ak),
            "affine_weights": w_ak.tolist(),
            "weight_norm": raw_norm,
            "constant_response": float(np.sum(w_ak)),
            "logL_response": float(np.dot(w_ak, x)),
            "pair_low_bias": float(intercept_ak + np.dot(w_ak, pair["mu_low"])
                                   - pair["target_low"]),
            "pair_high_bias": float(intercept_ak + np.dot(w_ak, pair["mu_high"])
                                    - pair["target_high"]),
        },
        "gls_baseline": {
            "affine_intercept": float(intercept_gls),
            "affine_weights": w_gls.tolist(),
            "weight_norm": gls_norm,
            "constant_response": float(np.sum(w_gls)),
            "logL_response": float(np.dot(w_gls, x)),
            "half_length": gls_half,
        },
        "pair": {
            "target_low": pair["target_low"],
            "target_high": pair["target_high"],
            "pair_da": pair["pair_da"],
            "pair_distance": dist,
            "c_high_minus_low": pair["c_high_minus_low"],
            "mu_low": pair["mu_low"].tolist(),
            "mu_high": pair["mu_high"].tolist(),
        },
        "ak_info_effective": info_eff,
    }


def ci_bundle_for_y(y: np.ndarray, sigma: float, m: int, curve: ModulusCurve,
                    target_mid: float = TARGET_MID_ALPHA) -> Dict:
    info = ak_half_length(curve, sigma, m)
    est = affine_estimators(curve, info, target_mid=target_mid)
    info_eff = est["ak_info_effective"]
    floor = curve.omega_at(sigma / math.sqrt(m))
    w_ak = np.asarray(est["ak"]["affine_weights"], dtype=float)
    w_gls = np.asarray(est["gls_baseline"]["affine_weights"], dtype=float)
    center_ak = est["ak"]["affine_intercept"] + float(np.dot(w_ak, y))
    center_gls = est["gls_baseline"]["affine_intercept"] + float(np.dot(w_gls, y))
    half_ak = float(info_eff["half_length"])
    half_gls = float(est["gls_baseline"]["half_length"])
    return {
        "center": float(center_ak),
        "lo": float(center_ak - half_ak),
        "hi": float(center_ak + half_ak),
        "half_length": half_ak,
        "floor": float(floor),
        "half_length_over_floor": float(half_ak / floor) if floor > 0 else None,
        "ak": info_eff,
        "center_details": est["ak"],
        "extremal_pair": {k: v for k, v in est["pair"].items()
                          if k not in ("mu_low", "mu_high")},
        "gls_baseline": {
            "center": float(center_gls),
            "lo": float(center_gls - half_gls),
            "hi": float(center_gls + half_gls),
            "half_length": half_gls,
            "center_details": est["gls_baseline"],
        },
    }


def mean_for_config(Ls: np.ndarray, alpha: float, u: float, omega: float,
                    c: float = 0.0) -> np.ndarray:
    return logform_mu(Ls, c, alpha, u, omega)


def estimator_risk_grid(curve: ModulusCurve, sigma: float, m: int,
                        target_mid: float = TARGET_MID_ALPHA) -> Dict:
    info = ak_half_length(curve, sigma, m)
    est = affine_estimators(curve, info, target_mid=target_mid)
    Ls = np.asarray(curve.design, dtype=float)
    ub = u_bounds_for_class(curve.U)
    alpha_grid = np.linspace(0.3, 0.7, 9)
    u_grid = np.linspace(ub[0], ub[1], 7)
    w_grid = np.linspace(OMEGA_BOUNDS[0], OMEGA_BOUNDS[1], 7)
    sigma_m = sigma / math.sqrt(m)

    def max_risk(which):
        w = np.asarray(est[which]["affine_weights"], dtype=float)
        b = float(est[which]["affine_intercept"])
        sd2 = (sigma_m ** 2) * float(np.dot(w, w))
        worst = {"risk": -np.inf}
        for alpha in alpha_grid:
            for u in u_grid:
                for om in w_grid:
                    mu = mean_for_config(Ls, float(alpha), float(u), float(om))
                    if np.any(~np.isfinite(mu)):
                        continue
                    bias = b + float(np.dot(w, mu)) - float(alpha)
                    risk = math.sqrt(bias * bias + sd2)
                    if risk > worst["risk"]:
                        worst = {
                            "risk": float(risk),
                            "alpha": float(alpha),
                            "u": float(u),
                            "omega": float(om),
                            "bias": float(bias),
                            "sd": float(math.sqrt(sd2)),
                        }
        return worst

    ak = max_risk("ak")
    gls = max_risk("gls_baseline")
    return {
        "grid": {
            "alpha": alpha_grid.tolist(),
            "u": u_grid.tolist(),
            "omega": w_grid.tolist(),
            "c": 0.0,
        },
        "ak_worst": ak,
        "gls_worst": gls,
        "ak_le_gls": bool(ak["risk"] <= gls["risk"] + 1e-9),
        "ak_info_effective": {k: v for k, v in est["ak_info_effective"].items()
                              if k not in ("grid_index",)},
    }


def run_task_a(args, curves: Dict[str, ModulusCurve]) -> Dict:
    # A deterministic fractional-EW reference for the A-K grid checks. The
    # blind calibration itself measures per-ladder sigma from generated rows.
    frac_sigma = 0.5 * float(read_json(HIER81)["level0_sigma"]) if os.path.exists(HIER81) else 0.30
    scenarios = {"fractional_reference": frac_sigma}

    rows = []
    for key, curve in curves.items():
        for sname, sigma in scenarios.items():
            info = ak_half_length(curve, sigma, M_REAL)
            est = affine_estimators(curve, info)
            info_eff = est["ak_info_effective"]
            risk = estimator_risk_grid(curve, sigma, M_REAL)
            ak_resp = est["ak"]["logL_response"]
            gls_resp = est["gls_baseline"]["logL_response"]
            slope_gate = bool(ak_resp <= 1.0 + 1e-8 and ak_resp < gls_resp - 1e-5)
            half_gate = bool(info_eff["half_length"] <= est["gls_baseline"]["half_length"] + 1e-12)
            risk_gate = bool(risk["ak_le_gls"])
            rows.append({
                "class_key": key,
                "five_tuple": (
                    f"Level-0, class N=1 convex hull J={curve.J}/U={curve.U:g}/"
                    "omega_min=0.3, design={32,48,64,96,128,192,256}, "
                    f"sigma={sname}:{sigma:.6g}, m=24"),
                "sigma_name": sname,
                "sigma": float(sigma),
                "ak_half_length": float(info_eff["half_length"]),
                "gls_center_half_length": float(est["gls_baseline"]["half_length"]),
                "ak_logL_response": float(ak_resp),
                "gls_logL_response": float(gls_resp),
                "ak_constant_response": est["ak"]["constant_response"],
                "pair_low_bias": est["ak"]["pair_low_bias"],
                "pair_high_bias": est["ak"]["pair_high_bias"],
                "risk_grid": risk,
                "gates": {
                    "ak_logL_response_le_1_and_lt_gls": slope_gate,
                    "ak_half_length_le_gls_center_half_length": half_gate,
                    "ak_worst_grid_risk_le_gls": risk_gate,
                    "all_met": bool(slope_gate and half_gate and risk_gate),
                },
                "ak_info": info_eff,
            })
            print(f"[G-85b-A] {key} {sname}: response={ak_resp:.4f} "
                  f"half={info_eff['half_length']:.4f} "
                  f"gls_half={est['gls_baseline']['half_length']:.4f} "
                  f"risk={risk['ak_worst']['risk']:.4f}/"
                  f"{risk['gls_worst']['risk']:.4f}")

    out = {
        "task": "G-85b-A true A-K affine center",
        "comparison": "exp85 GLS-normalized center retained as baseline",
        "rows": rows,
        "all_gates_met": bool(all(r["gates"]["all_met"] for r in rows)),
    }
    write_json(os.path.join(RESULTS_DIR, "taskA_ak.json"), out)
    return out


def generate_pure_fractional_rows(n: int, seed: int) -> List[Dict]:
    e81 = get_e81()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        alpha = float(rng.uniform(0.3, 0.7))
        z = e81.z_of_alpha(alpha)
        srng = np.random.default_rng(seed + 10000 + i)
        W = np.zeros((M_REAL, len(DESIGN)))
        for j, L in enumerate(DESIGN):
            for s in range(M_REAL):
                h, _, _, _ = e81.sample_field(int(L), z, 1.0, 1.0, 0.0, 1.0, srng)
                W[s, j] = math.sqrt(float(np.mean(h * h)))
        y, sigma, per_l = y_sigma_from_wseeds(W)
        rows.append({
            "id": f"pure_{i:03d}",
            "alpha": alpha,
            "y": y,
            "sigma": sigma,
            "per_L_sigma_logW": per_l,
        })
        if (i + 1) % 25 == 0:
            print(f"[G-85b-B] generated pure rows {i + 1}/{n}")
    return rows


def least_favorable_coverage(curve: ModulusCurve, sigma: float, n_each: int,
                             seed: int) -> Dict:
    info = ak_half_length(curve, sigma, M_REAL)
    pair = extremal_pair_means(curve, info)
    rng = np.random.default_rng(seed)
    by_side = {}
    pooled = []
    for side, mu_key, truth_key in (
        ("low", "mu_low", "target_low"),
        ("high", "mu_high", "target_high"),
    ):
        covered = []
        halfs = []
        centers = []
        mu = np.asarray(pair[mu_key], dtype=float)
        truth = float(pair[truth_key])
        for _ in range(n_each):
            y = mu + rng.normal(0.0, sigma / math.sqrt(M_REAL), size=len(mu))
            ci = ci_bundle_for_y(y, sigma, M_REAL, curve)
            cov = bool(ci["lo"] <= truth <= ci["hi"])
            covered.append(cov)
            pooled.append(cov)
            halfs.append(ci["half_length"])
            centers.append(ci["center"])
        cov_rate = float(np.mean(covered))
        by_side[side] = {
            "truth_alpha": truth,
            "coverage": cov_rate,
            "n": n_each,
            "misses": int(n_each - sum(covered)),
            "median_half_length": float(np.median(halfs)),
            "median_center": float(np.median(centers)),
        }
    pooled_coverage = float(np.mean(pooled))
    gate = (
        all(0.92 <= by_side[s]["coverage"] <= 0.995 for s in by_side)
        and int(len(pooled) - sum(pooled)) > 0
        and pooled_coverage < 1.0
    )
    return {
        "sigma_reference": float(sigma),
        "ak_info": info,
        "pair_targets": {
            "low": pair["target_low"],
            "high": pair["target_high"],
            "delta_alpha": pair["pair_da"],
        },
        "by_side": by_side,
        "pooled": {
            "n": len(pooled),
            "coverage": pooled_coverage,
            "misses": int(len(pooled) - sum(pooled)),
        },
        "gate_met": bool(gate),
    }


def run_task_b(args, curves: Dict[str, ModulusCurve]) -> Dict:
    pure_rows = generate_pure_fractional_rows(args.calibration_n, MASTER_SEED - 11)
    sigma_ref = float(np.median([r["sigma"] for r in pure_rows]))
    pure_by_class = {}
    for key, curve in curves.items():
        covered = []
        halfs = []
        for row in pure_rows:
            ci = ci_bundle_for_y(row["y"], row["sigma"], M_REAL, curve)
            covered.append(ci["lo"] <= row["alpha"] <= ci["hi"])
            halfs.append(ci["half_length"])
        cov = float(np.mean(covered))
        n = len(covered)
        lower = 0.95 - 2.0 * math.sqrt(0.95 * 0.05 / n)
        pure_by_class[key] = {
            "five_tuple": (
                f"Level-0 fractional-EW pure nu2=0, class N=1 hull J={curve.J}/"
                f"U={curve.U:g}/omega_min=0.3, design={{32,48,64,96,128,192,256}}, "
                "sigma=per-ladder measured median sd(log W), m=24"),
            "n": n,
            "coverage": cov,
            "one_sided_binomial_2sigma_lower": float(lower),
            "median_half_length": float(np.median(halfs)),
            "gate_met": bool(cov >= lower),
        }
        print(f"[G-85b-B] {key} pure coverage={cov:.3f} lower={lower:.3f}")

    lf_by_class = {}
    for key, curve in curves.items():
        lf = least_favorable_coverage(
            curve, sigma_ref, args.least_fav_n_each,
            seed=MASTER_SEED + 2000 + int(curve.U * 1000))
        lf["five_tuple"] = (
            f"Level-0 Gaussian seed-mean ladders at extremal pair, class N=1 "
            f"hull J={curve.J}/U={curve.U:g}/omega_min=0.3, "
            "design={32,48,64,96,128,192,256}, "
            f"sigma=pure fractional median {sigma_ref:.6g}, m=24")
        lf_by_class[key] = lf
        print(f"[G-85b-B] {key} least-fav pooled coverage="
              f"{lf['pooled']['coverage']:.3f} misses={lf['pooled']['misses']}")

    out = {
        "task": "G-85b-B corrected calibration pregates",
        "pure_rows_n": len(pure_rows),
        "pure_sigma_reference_median": sigma_ref,
        "validity_at_pure_point": pure_by_class,
        "tightness_at_least_favorable_pair": lf_by_class,
        "all_gates_met": bool(
            all(r["gate_met"] for r in pure_by_class.values())
            and all(r["gate_met"] for r in lf_by_class.values())),
    }
    write_json(os.path.join(RESULTS_DIR, "taskB_pregates.json"), out)
    return out


def fit_fixed_alpha_profile_point(y: np.ndarray, sigma: float, Ls: np.ndarray,
                                  alpha: float, U: float, seed: int,
                                  warm: Optional[np.ndarray] = None,
                                  n_starts: int = 6) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.log(Ls)
    ub = u_bounds_for_class(U)

    def solve_c(u, w):
        g = corr_log_scalar(Ls, float(u), float(w))
        if np.any(~np.isfinite(g)):
            return np.nan, np.inf, g
        c = float(np.mean(y - alpha * x - g))
        r = y - (c + alpha * x + g)
        return c, float(np.sum(r * r)), g

    def obj(p):
        _, sse, _ = solve_c(p[0], p[1])
        return sse

    starts = [np.array([0.0, 1.0])]
    if warm is not None:
        starts.append(np.asarray(warm, dtype=float))
    for _ in range(n_starts):
        starts.append(np.array([rng.uniform(*ub), rng.uniform(*OMEGA_BOUNDS)]))

    best = None
    bounds = [ub, OMEGA_BOUNDS]
    for p0 in starts:
        p0 = np.array([np.clip(p0[0], *ub), np.clip(p0[1], *OMEGA_BOUNDS)])
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 250, "maxfun": 1000})
        if best is None or float(res.fun) < float(best.fun):
            best = res
    c, sse, _ = solve_c(best.x[0], best.x[1])
    chi2 = sse / max((sigma * sigma / M_REAL), 1e-30)
    return {
        "alpha": float(alpha),
        "c": float(c),
        "u": float(best.x[0]),
        "omega": float(best.x[1]),
        "sse": float(sse),
        "chi2": float(chi2),
        "success": bool(best.success),
        "warm": np.asarray(best.x, dtype=float),
    }


def profile_alpha_grid(y: np.ndarray, sigma: float, Ls: np.ndarray, U: float,
                       alphas: np.ndarray, seed: int,
                       n_starts: int) -> Dict:
    rows = []
    warm = None
    for j, alpha in enumerate(alphas):
        row = fit_fixed_alpha_profile_point(
            y, sigma, Ls, float(alpha), U, seed=seed + j,
            warm=warm, n_starts=n_starts)
        warm = row.pop("warm")
        rows.append(row)
    chi = np.array([r["chi2"] for r in rows], dtype=float)
    chi_min = float(np.min(chi))
    alpha_hat = float(rows[int(np.argmin(chi))]["alpha"])

    def intervals_for(q):
        mask = chi - chi_min <= q + 1e-12
        intervals = []
        start = None
        prev = None
        for a, keep in zip(alphas, mask):
            if keep and start is None:
                start = float(a)
            if not keep and start is not None:
                intervals.append([start, float(prev)])
                start = None
            prev = a
        if start is not None:
            intervals.append([start, float(prev)])
        width_total = float(sum(max(0.0, b - a) for a, b in intervals))
        width_span = float(intervals[-1][1] - intervals[0][0]) if intervals else 0.0
        return {
            "threshold_q": float(q),
            "intervals": intervals,
            "width_total": width_total,
            "width_span": width_span,
            "grid_points": int(np.sum(mask)),
        }

    return {
        "U": float(U),
        "alpha_grid_step": float(alphas[1] - alphas[0]),
        "alpha_hat_profile_min": alpha_hat,
        "chi2_min": chi_min,
        "sets": {
            "q1_approx_68": intervals_for(1.0),
            "q3p84_approx_95": intervals_for(3.84),
        },
        "profile": rows,
        "note": "Profile thresholds are approximate because nuisance parameters are nonlinear.",
    }


def confusion_gap_pair_params(
    da: float,
    Ls: np.ndarray,
    U: float,
    seed: int,
    n_starts: int,
    warm: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    x = np.log(Ls)
    rng = np.random.default_rng(seed)
    ub = u_bounds_for_class(U)
    bounds = [(None, None), ub, OMEGA_BOUNDS, ub, OMEGA_BOUNDS]

    def obj(p):
        c, u1, w1, u0, w0 = p
        g1 = corr_log_scalar(Ls, float(u1), float(w1))
        g0 = corr_log_scalar(Ls, float(u0), float(w0))
        if np.any(~np.isfinite(g1)) or np.any(~np.isfinite(g0)):
            return 1e30
        diff = da * x + c + g1 - g0
        return float(np.sum(diff * diff))

    starts = [np.array([-da * x.mean(), 0.0, 1.0, 0.0, 1.0])]
    if warm is not None:
        starts.append(np.asarray(warm, dtype=float))
    for _ in range(n_starts):
        starts.append(np.array([
            -da * x.mean() + rng.normal(0, 0.5),
            rng.uniform(*ub), rng.uniform(*OMEGA_BOUNDS),
            rng.uniform(*ub), rng.uniform(*OMEGA_BOUNDS),
        ]))
    best = None
    for p0 in starts:
        p0 = np.asarray(p0, dtype=float)
        p0[1] = np.clip(p0[1], *ub)
        p0[2] = np.clip(p0[2], *OMEGA_BOUNDS)
        p0[3] = np.clip(p0[3], *ub)
        p0[4] = np.clip(p0[4], *OMEGA_BOUNDS)
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "maxfun": 1200})
        if best is None or float(res.fun) < float(best.fun):
            best = res
    return float(best.fun), np.asarray(best.x, dtype=float)


def floor_from_gap_grid(da_grid: np.ndarray, gaps: np.ndarray,
                        sigma: float, m: int) -> Dict:
    running = np.maximum.accumulate(gaps)
    thresh = sigma * sigma / m
    idx = np.searchsorted(running, thresh)
    if idx == 0:
        floor = float(da_grid[0])
    elif idx >= len(da_grid):
        floor = float(da_grid[-1])
    else:
        x0, x1 = da_grid[idx - 1], da_grid[idx]
        y0, y1 = running[idx - 1], running[idx]
        floor = float(x1 if y1 <= y0 else x0 + (thresh - y0) * (x1 - x0) / (y1 - y0))
    diffs = np.diff(gaps)
    tol = 0.05 * np.maximum(gaps[:-1], 1e-12)
    return {
        "floor": floor,
        "threshold_sigma2_over_m": float(thresh),
        "da_grid": da_grid.tolist(),
        "D2_raw": gaps.tolist(),
        "D2_running_max": running.tolist(),
        "raw_monotone_within_optimizer_scatter": bool(np.all(diffs >= -tol)),
    }


def global_floor_monotone_grid(Ls: np.ndarray, sigma: float, U: float,
                               da_grid: np.ndarray, seed: int,
                               n_starts: int) -> Dict:
    gaps = []
    params = []
    warm = None
    for j, da in enumerate(da_grid):
        if j == 0:
            gaps.append(0.0)
            params.append(None)
            continue
        gap, p = confusion_gap_pair_params(
            float(da), Ls, U, seed=seed + j, n_starts=n_starts, warm=warm)
        warm = p
        gaps.append(gap)
        params.append(p.tolist())
    out = floor_from_gap_grid(da_grid, np.asarray(gaps), sigma, M_REAL)
    out["best_params"] = params
    out["U"] = float(U)
    return out


def fit_fixed_alpha(y: np.ndarray, Ls: np.ndarray, alpha: float, U: float,
                    seed: int, n_starts: int) -> Dict:
    sigma_dummy = 1.0
    row = fit_fixed_alpha_profile_point(
        y, sigma_dummy, Ls, alpha, U, seed, warm=None, n_starts=n_starts)
    return {k: row[k] for k in ("alpha", "c", "u", "omega", "sse", "success")}


def local_gap_fixed_base(da: float, y_fit: Dict, Ls: np.ndarray, U: float,
                         seed: int, n_starts: int) -> Dict:
    mu0 = logform_mu(Ls, y_fit["c"], y_fit["alpha"], y_fit["u"], y_fit["omega"])
    rng = np.random.default_rng(seed)
    ub = u_bounds_for_class(U)
    x = np.log(Ls)

    def one_sign(sign):
        alpha2 = y_fit["alpha"] + sign * da

        def obj(p):
            c, u, w = p
            mu2 = logform_mu(Ls, float(c), float(alpha2), float(u), float(w))
            if np.any(~np.isfinite(mu2)):
                return 1e30
            r = mu2 - mu0
            return float(np.sum(r * r))

        starts = [np.array([mu0.mean() - alpha2 * x.mean(), y_fit["u"], y_fit["omega"]])]
        for _ in range(n_starts):
            starts.append(np.array([
                mu0.mean() - alpha2 * x.mean() + rng.normal(0, 0.5),
                rng.uniform(*ub),
                rng.uniform(*OMEGA_BOUNDS),
            ]))
        best = np.inf
        best_p = None
        bounds = [(None, None), ub, OMEGA_BOUNDS]
        for p0 in starts:
            p0[1] = np.clip(p0[1], *ub)
            p0[2] = np.clip(p0[2], *OMEGA_BOUNDS)
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 300, "maxfun": 1000})
            if float(res.fun) < best:
                best = float(res.fun)
                best_p = np.asarray(res.x)
        return best, best_p

    plus, pp = one_sign(+1.0)
    minus, pm = one_sign(-1.0)
    if plus <= minus:
        return {"gap": float(plus), "sign": +1, "params": pp.tolist()}
    return {"gap": float(minus), "sign": -1, "params": pm.tolist()}


def truth_pinned_local_floor(y: np.ndarray, sigma: float, Ls: np.ndarray,
                             U: float, da_grid: np.ndarray, seed: int,
                             n_starts: int) -> Dict:
    base = fit_fixed_alpha(y, Ls, TARGET_MID_ALPHA, U, seed, n_starts)
    rows = []
    for j, da in enumerate(da_grid):
        if j == 0:
            rows.append({"gap": 0.0, "sign": 0, "params": None})
            continue
        rows.append(local_gap_fixed_base(
            float(da), base, Ls, U, seed=seed + 1000 + j,
            n_starts=n_starts))
    gaps = np.asarray([r["gap"] for r in rows], dtype=float)
    out = floor_from_gap_grid(da_grid, gaps, sigma, M_REAL)
    out["base_fit_alpha_fixed_0p5"] = base
    out["local_rows"] = rows
    out["U"] = float(U)
    return out


def run_task_c(args) -> Dict:
    data = load_wsat_perseed()
    alphas = np.round(np.arange(0.2, 0.9001, 0.005), 3)
    profile_Us = [0.5, 1.0, 4.0]
    floor_Us = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    da_grid = np.linspace(0.0, args.floor_da_hi, args.floor_grid)
    systems = {}

    for si, system in enumerate(sorted(data)):
        y, sigma, per_l = y_sigma_from_wseeds(data[system])
        profiles = {}
        for U in profile_Us:
            profiles[f"U{U:g}"] = profile_alpha_grid(
                y, sigma, DESIGN, U, alphas, seed=872000 + si * 10000 + int(U * 100),
                n_starts=args.profile_starts)
            print(f"[G-85b-C] {system} U={U:g} profile min="
                  f"{profiles[f'U{U:g}']['alpha_hat_profile_min']:.3f}")

        floors = {}
        for U in floor_Us:
            floors[f"U{U:g}"] = global_floor_monotone_grid(
                DESIGN, sigma, U, da_grid, seed=873000 + si * 10000 + int(U * 100),
                n_starts=args.floor_starts)
            print(f"[G-85b-C] {system} U={U:g} global floor="
                  f"{floors[f'U{U:g}']['floor']:.4f}")
        seq = [floors[f"U{U:g}"]["floor"] for U in floor_Us]
        monotone = all(seq[i + 1] >= seq[i] - args.floor_monotone_tol
                       for i in range(len(seq) - 1))
        if not monotone:
            print(f"[G-85b-C] retrying floor curve for {system} with larger budget")
            floors = {}
            for U in floor_Us:
                floors[f"U{U:g}"] = global_floor_monotone_grid(
                    DESIGN, sigma, U, da_grid,
                    seed=883000 + si * 10000 + int(U * 100),
                    n_starts=args.floor_starts * 3)
            seq = [floors[f"U{U:g}"]["floor"] for U in floor_Us]
            monotone = all(seq[i + 1] >= seq[i] - args.floor_monotone_tol
                           for i in range(len(seq) - 1))

        local = {}
        local_gates = {}
        for U in (1.0, 4.0):
            lf = truth_pinned_local_floor(
                y, sigma, DESIGN, U, da_grid, seed=874000 + si * 10000 + int(U * 100),
                n_starts=args.floor_starts)
            key = f"U{U:g}"
            global_floor = floors[key]["floor"]
            lf["global_floor_same_class"] = float(global_floor)
            lf["local_le_global_gate_met"] = bool(lf["floor"] <= global_floor + args.nesting_tol)
            local[key] = lf
            local_gates[key] = lf["local_le_global_gate_met"]
            print(f"[G-85b-C] {system} U={U:g} local floor={lf['floor']:.4f} "
                  f"global={global_floor:.4f}")

        systems[system] = {
            "five_tuple": (
                "Level-0 real log-width ladder, class N=1/U as tabulated/"
                "omega_min=0.3, design={32,48,64,96,128,192,256}, "
                f"sigma=measured median sd(log W) {sigma:.6g}, m=24"),
            "sigma": sigma,
            "per_L_sigma_logW": per_l,
            "identified_set_profiles": profiles,
            "floor_vs_U": floors,
            "truth_pinned_local_floors": local,
            "gates": {
                "floor_non_decreasing_in_U": bool(monotone),
                "truth_pinned_local_le_global": local_gates,
                "all_met": bool(monotone and all(local_gates.values())),
            },
        }

    out = {
        "task": "G-85b-C identified sets and floor curves",
        "alpha_grid": alphas.tolist(),
        "profile_threshold_note": "q=1 and q=3.84 are approximate profile thresholds with nonlinear nuisance parameters.",
        "systems": systems,
        "all_gates_met": bool(all(v["gates"]["all_met"] for v in systems.values())),
    }
    write_json(os.path.join(RESULTS_DIR, "taskC_identification_floor.json"), out)
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
    return float(rng.uniform(0.3, 1.4))


def draw_truth_configs(seed: int = MASTER_SEED) -> List[Dict]:
    rng = np.random.default_rng(seed)
    cfgs = []
    for i in range(200):
        alpha = float(rng.uniform(0.3, 0.7))
        base = {"id": f"cfg_{i:03d}", "alpha": alpha, "D": 1.0, "nu": 1.0}
        if i < 120:
            om = choose_omega_tilde_from_table(rng)
            u = float(rng.uniform(-0.45, 0.5))
            base.update({
                "kind": "in_class_single_power",
                "u": u,
                "omega_tilde": om,
                "declared_in_class_U": 0.5,
            })
        elif i < 160:
            base.update({
                "kind": "out_of_class_mild_twoterm",
                "u1": float(rng.uniform(-0.35, 0.7)),
                "u2": float(rng.uniform(-0.25, 0.45)),
                "omega1": float(rng.uniform(0.35, 1.2)),
                "omega2": float(rng.uniform(1.3, 2.5)),
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
                    "u": float(rng.uniform(0.35, 0.8)),
                    "omega_tilde": float(rng.uniform(0.10, 0.18)),
                })
        cfgs.append(base)
    return cfgs


def sample_fractional_W_matrix(cfg: Dict, seed: int, m: int = M_REAL,
                               design: np.ndarray = DESIGN) -> np.ndarray:
    e85 = get_e85()
    return e85.sample_fractional_W_matrix(cfg, seed, m=m, design=design)


def make_blind_dataset(configs: List[Dict], seed: int = MASTER_SEED + 1) -> Dict:
    rows = []
    for i, cfg in enumerate(configs):
        W = sample_fractional_W_matrix(cfg, seed + i)
        rows.append({"id": cfg["id"], "W": W.tolist()})
        if (i + 1) % 25 == 0:
            print(f"[phase1] generated blind ladders {i + 1}/200")
    return {
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "master_seed": MASTER_SEED,
        "configs": rows,
        "truth_fields_present": False,
    }


def analyze_blind_rows(rows: List[Dict], curves: Dict[str, ModulusCurve],
                       args, phase_label: str) -> List[Dict]:
    e85 = get_e85()
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
            ci = ci_bundle_for_y(y, sigma, W.shape[0], curve)
            U = float(key.replace("U", ""))
            if args.main_gof_boot <= 0:
                fit = e85.fit_logform_chi2(
                    y, np.asarray(curve.design, dtype=float), sigma, W.shape[0],
                    U, seed=880000 + i * 1000 + int(U * 100),
                    starts=args.main_gof_starts)
                ci["adequacy_p"] = float(chi2.sf(fit["chi2"], max(fit["dof"], 1)))
                ci["adequacy_n_boot"] = 0
                ci["adequacy_method"] = "asymptotic_chi2_single_fit"
                ci["adequacy_chi2"] = fit["chi2"]
            else:
                gof = e85.bootstrap_gof_logform(
                    y, np.asarray(curve.design, dtype=float), sigma, W.shape[0],
                    U, seed=880000 + i * 1000 + int(U * 100),
                    n_boot=args.main_gof_boot, starts=args.main_gof_starts)
                ci["adequacy_p"] = gof["bootstrap_p"]
                ci["adequacy_n_boot"] = gof["n_boot"]
                ci["adequacy_method"] = "parametric_bootstrap"
                ci["adequacy_chi2"] = gof["observed_chi2"]
            pr["classes"][key] = ci
        preds.append(pr)
        if (i + 1) % 25 == 0:
            print(f"[{phase_label}] analyzed {i + 1}/{len(rows)}")
    return preds


def task4_blind_predictions(args) -> Dict:
    data = load_wsat_perseed()
    Wbd = data["bd"][:, :len(BD_HALF_DESIGN)]
    y, sigma, per_l = y_sigma_from_wseeds(Wbd)
    bd = {}
    for U in (0.5, 4.0):
        curve = build_or_load_curve(args, U, BD_HALF_DESIGN, tag="bd_half")
        ci = ci_bundle_for_y(y, sigma, Wbd.shape[0], curve)
        ci["sigma"] = sigma
        ci["per_L_sigma_logW"] = per_l
        ci["fit_window_L"] = BD_HALF_DESIGN.tolist()
        ci["five_tuple"] = (
            f"Level-0 real BD log-width half-window, class N=1 hull J={curve.J}/"
            f"U={U:g}/omega_min=0.3, design={{32,48,64,96}}, "
            f"sigma=measured median sd(log W) {sigma:.6g}, m=24")
        bd[f"U{U:g}"] = ci

    ising = {
        "status": "not_run_missing_per_L_collapse_ladder",
        "fit_window_L": [32, 48],
        "class": "Ising-honest |u|<=0.3, omega>=1",
        "truth_fields_present": False,
        "note": (
            "results_exp52d_full contains only nu_optimal, collapse-quality "
            "summaries, and a PNG; it does not contain the per-L observable "
            "ladder needed to fit L={32,48} before unblinding."),
    }
    return {
        "bd_half_window": bd,
        "ising_half_window": ising,
        "truth_fields_present": False,
    }


def pre_phase1_gates_met() -> bool:
    paths = [
        os.path.join(RESULTS_DIR, "taskA_ak.json"),
        os.path.join(RESULTS_DIR, "taskB_pregates.json"),
        os.path.join(RESULTS_DIR, "taskC_identification_floor.json"),
    ]
    if not all(os.path.exists(p) for p in paths):
        return False
    return all(bool(read_json(p).get("all_gates_met")) for p in paths)


def phase1(args) -> Dict:
    if not pre_phase1_gates_met():
        raise RuntimeError("Refusing phase1: Tasks A-C gates are not all met")
    curves = {
        "U0.5": build_or_load_curve(args, 0.5, DESIGN, tag="fractional"),
        "U1": build_or_load_curve(args, 1.0, DESIGN, tag="fractional"),
    }
    configs = draw_truth_configs(MASTER_SEED)
    blind = make_blind_dataset(configs)
    write_json(os.path.join(RESULTS_DIR, "blind_ladders.json"), blind)
    preds = analyze_blind_rows(blind["configs"], curves, args, "phase1")
    task4 = task4_blind_predictions(args)
    pred_obj = {
        "experiment": "85b_validation",
        "phase": "phase1_blind_predictions",
        "master_seed": MASTER_SEED,
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "classes": list(curves.keys()),
        "ci_machinery": "true A-K affine center from extremal-pair modulus; GLS-normalized exp85 center recorded as baseline",
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
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    pred = read_json(pred_path)
    actual_digest = sha256_file(pred_path)
    sha_text = open(os.path.join(RESULTS_DIR, "predictions.sha256"), encoding="utf-8").read()
    recorded_digest = sha_text.split()[0]
    if actual_digest != recorded_digest:
        raise SystemExit("Refusing to unblind: predictions.sha256 does not match predictions.json")

    cfgs = {c["id"]: c for c in draw_truth_configs(MASTER_SEED)}
    phase1_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    scores = {
        "phase1_commit_head": phase1_commit,
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
        ratios = [r["half_length_over_floor"] for r in rows
                  if r["half_length_over_floor"] is not None]
        scores["task3"][key] = {
            "five_tuple": (
                f"Level-0 fractional-EW seed log-width ladder, class N=1 hull "
                f"J=4/{key}/omega_min=0.3, design={{32,48,64,96,128,192,256}}, "
                "sigma=config-measured median sd(log W), m=24"),
            "in_class_n": n_in,
            "in_class_coverage": in_cov,
            "in_class_binomial_2sigma_window_around_0p95": [0.95 - two_sigma, 0.95 + two_sigma],
            "in_class_coverage_gate_met": bool(0.95 - two_sigma <= in_cov <= 0.95 + two_sigma),
            "out_of_class_n": len(out_rows),
            "out_of_class_coverage": out_cov,
            "falsifiability_2x2_out_of_class": fals,
            "half_length_over_floor_quantiles": {
                "q10": float(np.percentile(ratios, 10)),
                "q50": float(np.percentile(ratios, 50)),
                "q90": float(np.percentile(ratios, 90)),
            },
            "per_config_scores": rows,
        }
        print(f"[phase2] {key}: in_cov={in_cov:.3f} out_cov={out_cov:.3f}")

    t4 = pred["task4_blind_predictions"]
    summary = read_json(SUMMARY76)
    bd_full = summary["real_systems"]["bd"]["alpha_hat_mix"]
    scores["task4"]["bd_half_window"] = {}
    for key, bd in t4["bd_half_window"].items():
        scores["task4"]["bd_half_window"][key] = {
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


def run_tasks_abc(args) -> Dict:
    ensure_results_dir()
    curves = {
        "U0.5": build_or_load_curve(args, 0.5, DESIGN, tag="fractional"),
        "U1": build_or_load_curve(args, 1.0, DESIGN, tag="fractional"),
    }
    task_a = run_task_a(args, curves)
    if not task_a["all_gates_met"]:
        raise RuntimeError("G-85b-A gate not met")
    task_b = run_task_b(args, curves)
    if not task_b["all_gates_met"]:
        raise RuntimeError("G-85b-B gate not met")
    task_c = run_task_c(args)
    if not task_c["all_gates_met"]:
        raise RuntimeError("G-85b-C gate not met")
    return {"taskA": task_a, "taskB": task_b, "taskC": task_c}


def write_report() -> None:
    task_a = read_json(os.path.join(RESULTS_DIR, "taskA_ak.json"))
    task_b = read_json(os.path.join(RESULTS_DIR, "taskB_pregates.json"))
    task_c = read_json(os.path.join(RESULTS_DIR, "taskC_identification_floor.json"))
    pred = read_json(os.path.join(RESULTS_DIR, "predictions.json")) if os.path.exists(os.path.join(RESULTS_DIR, "predictions.json")) else None
    score = read_json(os.path.join(RESULTS_DIR, "score.json")) if os.path.exists(os.path.join(RESULTS_DIR, "score.json")) else None
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    lines = []
    lines.append("# Exp 85b Report -- Corrected A-K Validation")
    lines.append("")
    lines.append("Findings only. `CLAIMS_REGISTER.md` was not edited.")
    lines.append("")
    lines.append("## Gate Ledger")
    lines.append("")
    lines.append("| Gate | Check | Result | Proof path |")
    lines.append("|---|---|---|---|")
    lines.append(f"| G-85b-A | true A-K center shrinkage, half-length, grid risk | {'met' if task_a['all_gates_met'] else 'not met'} | `results_exp85b_validation/taskA_ak.json` |")
    for key, row in task_b["validity_at_pure_point"].items():
        lines.append(f"| G-85b-B pure {key} | coverage >= one-sided binomial 2sigma lower | {'met' if row['gate_met'] else 'not met'} | `results_exp85b_validation/taskB_pregates.json` |")
    for key, row in task_b["tightness_at_least_favorable_pair"].items():
        lines.append(f"| G-85b-B least-fav {key} | coverage in [0.92,0.995] with pooled misses | {'met' if row['gate_met'] else 'not met'} | `results_exp85b_validation/taskB_pregates.json` |")
    for system, row in task_c["systems"].items():
        g = row["gates"]
        lines.append(f"| G-85b-C {system} | floor nondecreasing in U; truth-pinned local <= global | {'met' if g['all_met'] else 'not met'} | `results_exp85b_validation/taskC_identification_floor.json` |")
    if score:
        for key, row in score["task3"].items():
            lines.append(f"| G-85b-D {key} | in-class coverage within binomial 2sigma of 0.95 | {'met' if row['in_class_coverage_gate_met'] else 'not met'} | `results_exp85b_validation/score.json` |")
    if pred:
        phase1_commit = score["phase1_commit_head"] if score else "pending"
        phase2_commit = current_head if score else "pending"
        lines.append(f"| Blinding | predictions hash committed before scoring | recorded | phase-1 commit `{phase1_commit}`; phase-2 score commit `{phase2_commit}` |")
    else:
        lines.append("| Blinding | predictions hash committed before scoring | not reached | phase 1 was not run |")
    lines.append("")

    lines.append("## A-K Center")
    lines.append("")
    lines.append("| Class | Sigma scenario | A-K logL response | GLS logL response | A-K half | GLS half | A-K worst risk | GLS worst risk |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in task_a["rows"]:
        lines.append(
            f"| {row['class_key']} | {row['sigma_name']} | "
            f"{fmt(row['ak_logL_response'])} | {fmt(row['gls_logL_response'])} | "
            f"{fmt(row['ak_half_length'])} | {fmt(row['gls_center_half_length'])} | "
            f"{fmt(row['risk_grid']['ak_worst']['risk'])} | "
            f"{fmt(row['risk_grid']['gls_worst']['risk'])} |")
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    lines.append("| Class | Pure coverage | Pure lower | Least-fav pooled coverage | Least-fav misses |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in task_b["validity_at_pure_point"]:
        pure = task_b["validity_at_pure_point"][key]
        lf = task_b["tightness_at_least_favorable_pair"][key]
        lines.append(
            f"| {key} | {fmt(pure['coverage'],3)} | "
            f"{fmt(pure['one_sided_binomial_2sigma_lower'],3)} | "
            f"{fmt(lf['pooled']['coverage'],3)} | {lf['pooled']['misses']} |")
    lines.append("")

    lines.append("## Identified Sets")
    lines.append("")
    lines.append("| System | U | alpha at profile min | 68 width | 95 width |")
    lines.append("|---|---:|---:|---:|---:|")
    for system, row in task_c["systems"].items():
        for key, prof in row["identified_set_profiles"].items():
            lines.append(
                f"| {system} | {key.replace('U','')} | "
                f"{fmt(prof['alpha_hat_profile_min'],3)} | "
                f"{fmt(prof['sets']['q1_approx_68']['width_span'],3)} | "
                f"{fmt(prof['sets']['q3p84_approx_95']['width_span'],3)} |")
    lines.append("")

    lines.append("## Floor Curves")
    lines.append("")
    lines.append("| System | U=0.25 | U=0.5 | U=1 | U=2 | U=4 | U=8 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for system, row in task_c["systems"].items():
        floors = row["floor_vs_U"]
        lines.append(
            f"| {system} | {fmt(floors['U0.25']['floor'])} | "
            f"{fmt(floors['U0.5']['floor'])} | {fmt(floors['U1']['floor'])} | "
            f"{fmt(floors['U2']['floor'])} | {fmt(floors['U4']['floor'])} | "
            f"{fmt(floors['U8']['floor'])} |")
    lines.append("")

    lines.append("## Truth-Pinned Local Floors")
    lines.append("")
    lines.append("| System | U | local floor | global floor |")
    lines.append("|---|---:|---:|---:|")
    for system, row in task_c["systems"].items():
        for key, lf in row["truth_pinned_local_floors"].items():
            lines.append(
                f"| {system} | {key.replace('U','')} | {fmt(lf['floor'])} | "
                f"{fmt(lf['global_floor_same_class'])} |")
    lines.append("")

    lines.append("## Blind Phase")
    lines.append("")
    if score:
        lines.append("| Class | In-class coverage | 2sigma window | Out-of-class coverage | Median half/floor |")
        lines.append("|---|---:|---|---:|---:|")
        for key, row in score["task3"].items():
            w = row["in_class_binomial_2sigma_window_around_0p95"]
            lines.append(
                f"| {key} | {fmt(row['in_class_coverage'],3)} | "
                f"[{fmt(w[0],3)}, {fmt(w[1],3)}] | "
                f"{fmt(row['out_of_class_coverage'],3)} | "
                f"{fmt(row['half_length_over_floor_quantiles']['q50'],2)} |")
        lines.append("")
        lines.append("| Class | adequacy pass/cover | pass/miss | fail/cover | fail/miss |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, row in score["task3"].items():
            f = row["falsifiability_2x2_out_of_class"]
            lines.append(
                f"| {key} | {f['adequacy_pass__cover']} | "
                f"{f['adequacy_pass__miss']} | {f['adequacy_fail__cover']} | "
                f"{f['adequacy_fail__miss']} |")
    else:
        lines.append("Phase 2 scoring was not run.")
    lines.append("")

    lines.append("## Real-Data Half-Window")
    lines.append("")
    if pred:
        lines.append("| Target | Class | Blind CI | Center | After unblind |")
        lines.append("|---|---|---|---:|---|")
        if score:
            for key, bd in pred["task4_blind_predictions"]["bd_half_window"].items():
                sc = score["task4"]["bd_half_window"][key]
                lines.append(
                    f"| BD alpha | {key} | [{fmt(bd['lo'])}, {fmt(bd['hi'])}] | "
                    f"{fmt(bd['center'])} | covers alpha=0.5: {fmt(sc['covers_exact_0p5'])}; "
                    f"covers full-window exp76: {fmt(sc['covers_full_window_exp76'])} |")
        else:
            for key, bd in pred["task4_blind_predictions"]["bd_half_window"].items():
                lines.append(
                    f"| BD alpha | {key} | [{fmt(bd['lo'])}, {fmt(bd['hi'])}] | "
                    f"{fmt(bd['center'])} | pending |")
        ising = pred["task4_blind_predictions"]["ising_half_window"]
        lines.append(f"| Ising 1/nu | | {ising['status']} | NA | stored artifact has no per-L half-window ladder |")
    else:
        lines.append("Phase-1 half-window predictions were not written.")
    lines.append("")

    lines.append("## What We Did Not Do")
    lines.append("")
    lines.append("- No entry was added to `CLAIMS_REGISTER.md`.")
    lines.append("- No honest U was estimated from fitted correction amplitudes.")
    lines.append("- The Ising L={32,48} blind CI was not constructed because exp52d does not store the per-L collapse observable ladder required before unblinding.")
    if pred:
        lines.append("- Phase-1 predictions contain no truth fields; phase 2 reconstructs truth configs only after the committed SHA256 check.")
    lines.append("")

    lines.append("## Anomalies And Bugs")
    lines.append("")
    anomalies = []
    if not task_a["all_gates_met"]:
        anomalies.append("G-85b-A had at least one unmet gate; see `taskA_ak.json`.")
    if not task_b["all_gates_met"]:
        anomalies.append("G-85b-B had at least one unmet calibration gate; see `taskB_pregates.json`.")
    if not task_c["all_gates_met"]:
        anomalies.append("G-85b-C had at least one unmet floor/local gate; see `taskC_identification_floor.json`.")
    if pred:
        anomalies.append("The Ising half-window prediction remains blocked by missing stored per-L exp52d data.")
    if not anomalies:
        anomalies.append("No additional implementation anomalies were recorded beyond the missing Ising half-window ladder.")
    for a in anomalies:
        lines.append(f"- {a}")

    report_path = os.path.join(ROOT, "ml_paper", "EXP85B_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tasks_abc", "phase1", "phase2", "report", "all"],
                    default="tasks_abc")
    ap.add_argument("--reuse-curves", action="store_true")
    ap.add_argument("--hull-J", type=int, default=4)
    ap.add_argument("--curve-grid", type=int, default=37)
    ap.add_argument("--curve-da-hi", type=float, default=1.2)
    ap.add_argument("--curve-starts", type=int, default=5)
    ap.add_argument("--calibration-n", type=int, default=200)
    ap.add_argument("--least-fav-n-each", type=int, default=200)
    ap.add_argument("--profile-starts", type=int, default=5)
    ap.add_argument("--floor-grid", type=int, default=33)
    ap.add_argument("--floor-da-hi", type=float, default=1.2)
    ap.add_argument("--floor-starts", type=int, default=5)
    ap.add_argument("--floor-monotone-tol", type=float, default=0.02)
    ap.add_argument("--nesting-tol", type=float, default=0.025)
    ap.add_argument("--main-gof-boot", type=int, default=60)
    ap.add_argument("--main-gof-starts", type=int, default=4)
    args = ap.parse_args()

    if args.stage in ("tasks_abc", "all"):
        run_tasks_abc(args)
    if args.stage in ("phase1", "all"):
        phase1(args)
    if args.stage in ("phase2", "all"):
        phase2(args)
    if args.stage in ("report", "all"):
        write_report()


if __name__ == "__main__":
    main()
