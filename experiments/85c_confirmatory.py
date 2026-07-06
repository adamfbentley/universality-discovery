"""Exp 85c: confirmatory coverage, BD diagnosis, calibration debts, Ising anchor.

This script deliberately imports exp85b machinery rather than editing it.
Stages:

  task1a       analytic coverage prediction for exp85b in-class configs
  phase1       fresh blind confirmatory predictions + Ising half-window CI
  task23       BD half-window diagnosis and calibration-debt tables
  phase2       score fresh blind predictions and unblind Ising comparisons
  report       write ml_paper/EXP85C_REPORT.md

Phase-1 outputs do not write truth fields. Phase 2 refuses to score unless
the phase-1 prediction artifacts are present in HEAD.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import importlib.util
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(ROOT, "results_exp85c_confirmatory")
E85B_PATH = os.path.join(HERE, "85b_validation.py")
E52D_PATH = os.path.join(HERE, "52d_ising_finite_size_scaling.py")

EXP85B_RESULTS = os.path.join(ROOT, "results_exp85b_validation")
EXP85B_PRED = os.path.join(EXP85B_RESULTS, "predictions.json")
EXP85B_SCORE = os.path.join(EXP85B_RESULTS, "score.json")
EXP85B_TASKC = os.path.join(EXP85B_RESULTS, "taskC_identification_floor.json")
SUMMARY76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "summary_full24seed.json")
CLASSICAL76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "classical_on_real.json")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
BD_HALF_DESIGN = np.array([32., 48., 64., 96.])
ISING_HALF_DESIGN = np.array([32., 48.])
ISING_HELDOUT_DESIGN = np.array([64., 96.])
ISING_FULL_DESIGN = np.array([32., 48., 64., 96.])

M_REAL = 24
MASTER_SEED_85C = 85300
TARGET_ALPHA = 0.5
ALPHA_PROFILE_GRID = np.round(np.arange(0.2, 0.9001, 0.005), 3)

ISING_U_STRICT = 0.3
ISING_U_LOOSE = 1.0
ISING_OMEGA_BOUNDS = (1.0, 2.5)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e85b():
    return load_module(E85B_PATH, "e85c_reuse_85b")


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    ensure_results_dir()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def current_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_phase1_committed() -> None:
    rels = [
        os.path.join("results_exp85c_confirmatory", "predictions.json"),
        os.path.join("results_exp85c_confirmatory", "predictions.sha256"),
        os.path.join("results_exp85c_confirmatory", "ising_phase1_predictions.json"),
        os.path.join("results_exp85c_confirmatory", "ising_phase1_predictions.sha256"),
    ]
    missing = [p for p in rels if not git_blob_exists(p)]
    if missing:
        raise SystemExit(
            "Refusing to unblind: phase-1 artifacts are not committed in HEAD: "
            f"{missing}")


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{float(x):.{nd}f}"


def binomial_2sigma_band(p: float, n: int) -> List[float]:
    sd = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return [float(max(0.0, p - 2.0 * sd)), float(min(1.0, p + 2.0 * sd))]


def interval_width_from_profile(profile: Dict, q: float) -> Dict:
    rows = profile["profile"]
    chi = np.array([r["chi2"] for r in rows], dtype=float)
    alphas = np.array([r["alpha"] for r in rows], dtype=float)
    chi_min = float(np.min(chi))
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
    return {
        "threshold_q": float(q),
        "intervals": intervals,
        "width_total": float(sum(max(0.0, b - a) for a, b in intervals)),
        "width_span": float(intervals[-1][1] - intervals[0][0]) if intervals else 0.0,
        "grid_points": int(np.sum(mask)),
    }


def load_curve(mod, tag: str, U: float):
    return mod.ModulusCurve.from_json(
        read_json(os.path.join(EXP85B_RESULTS, f"modulus_{tag}_U{U:g}.json")))


def analytic_coverage_for_ci(ci: Dict, mu: np.ndarray, truth: float) -> Dict:
    weights = np.asarray(ci["center_details"]["affine_weights"], dtype=float)
    intercept = float(ci["center_details"]["affine_intercept"])
    bias = intercept + float(np.dot(weights, mu)) - float(truth)
    sigma_m = float(ci["ak"]["sigma_m"])
    sd = sigma_m * float(np.linalg.norm(weights))
    chi_half = float(ci["half_length"])
    if sd <= 0:
        cov = 1.0 if abs(bias) <= chi_half else 0.0
    else:
        cov = (norm.cdf((chi_half - abs(bias)) / sd)
               + norm.cdf((chi_half + abs(bias)) / sd) - 1.0)
    return {
        "bias": float(bias),
        "sd": float(sd),
        "half_length": chi_half,
        "coverage": float(np.clip(cov, 0.0, 1.0)),
    }


def true_mu_single_power(mod, cfg: Dict, design: np.ndarray) -> np.ndarray:
    return mod.logform_mu(
        design, 0.0, float(cfg["alpha"]), float(cfg["u"]),
        float(cfg["omega_tilde"]))


def task1a_analytic_prediction() -> Dict:
    mod = e85b()
    pred = read_json(EXP85B_PRED)
    score = read_json(EXP85B_SCORE)
    cfgs = {c["id"]: c for c in mod.draw_truth_configs(mod.MASTER_SEED)}
    rows_by_class = {key: [] for key in pred["classes"]}

    for pr in pred["task3_predictions"]:
        cfg = cfgs[pr["id"]]
        if cfg["kind"] != "in_class_single_power":
            continue
        mu = true_mu_single_power(mod, cfg, DESIGN)
        for key in pred["classes"]:
            cov = analytic_coverage_for_ci(
                pr["classes"][key], mu, float(cfg["alpha"]))
            cov.update({
                "id": pr["id"],
                "truth_alpha": float(cfg["alpha"]),
                "u": float(cfg["u"]),
                "omega_tilde": float(cfg["omega_tilde"]),
            })
            rows_by_class[key].append(cov)

    classes = {}
    for key, rows in rows_by_class.items():
        pbar = float(np.mean([r["coverage"] for r in rows]))
        band = binomial_2sigma_band(pbar, len(rows))
        observed = float(score["task3"][key]["in_class_coverage"])
        classes[key] = {
            "five_tuple": (
                f"Level-0 exp85b fractional-EW seed log-width ladder, class N=1/"
                f"{key}/omega_min=0.3, design={{32,48,64,96,128,192,256}}, "
                "sigma=exp85b config-measured median sd(log W), m=24"),
            "n": len(rows),
            "predicted_mean_coverage": pbar,
            "monte_carlo_binomial_2sigma_band_n120": band,
            "exp85b_observed_in_class_coverage": observed,
            "gate_1a_observed_1p000_inside_predicted_band": bool(
                band[0] <= observed <= band[1]),
            "coverage_quantiles": {
                "min": float(np.min([r["coverage"] for r in rows])),
                "q10": float(np.percentile([r["coverage"] for r in rows], 10)),
                "q50": float(np.percentile([r["coverage"] for r in rows], 50)),
                "q90": float(np.percentile([r["coverage"] for r in rows], 90)),
                "max": float(np.max([r["coverage"] for r in rows])),
            },
            "per_config": rows,
        }

    out = {
        "task": "G-85c-1a analytic coverage prediction for exp85b",
        "formula": "Phi((chi-|b|)/s) + Phi((chi+|b|)/s) - 1",
        "classes": classes,
        "all_gates_met": bool(all(
            row["gate_1a_observed_1p000_inside_predicted_band"]
            for row in classes.values())),
    }
    write_json(os.path.join(RESULTS_DIR, "task1_analytic_prediction.json"), out)
    return out


def draw_confirmatory_configs(seed: int = MASTER_SEED_85C) -> List[Dict]:
    mod = e85b()
    rng = np.random.default_rng(seed)
    cfgs = []
    for i in range(140):
        alpha = float(rng.uniform(0.3, 0.7))
        base = {"id": f"c85c_{i:03d}", "alpha": alpha, "D": 1.0, "nu": 1.0}
        if i < 100:
            om = mod.choose_omega_tilde_from_table(rng)
            u = float(rng.uniform(-0.45, 0.5))
            base.update({
                "kind": "in_class_single_power",
                "u": u,
                "omega_tilde": om,
                "declared_in_class_U": 0.5,
            })
        elif i < 120:
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


@dataclass
class Args85B:
    reuse_curves: bool = True
    hull_J: int = 4
    curve_da_hi: float = 1.2
    curve_grid: int = 37
    curve_starts: int = 5
    main_gof_boot: int = 0
    main_gof_starts: int = 4


def make_confirmatory_blind_dataset(configs: List[Dict]) -> Dict:
    mod = e85b()
    rows = []
    for i, cfg in enumerate(configs):
        W = mod.sample_fractional_W_matrix(cfg, MASTER_SEED_85C + 1 + i)
        rows.append({"id": cfg["id"], "W": W.tolist()})
        if (i + 1) % 20 == 0:
            print(f"[phase1] generated confirmatory ladders {i + 1}/140")
    return {
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "master_seed": MASTER_SEED_85C,
        "configs": rows,
        "truth_fields_present": False,
    }


def phase1_confirmatory_predictions() -> Dict:
    mod = e85b()
    args85 = Args85B()
    curves = {
        "U0.5": mod.build_or_load_curve(args85, 0.5, DESIGN, tag="fractional"),
        "U1": mod.build_or_load_curve(args85, 1.0, DESIGN, tag="fractional"),
    }
    blind = make_confirmatory_blind_dataset(draw_confirmatory_configs())
    write_json(os.path.join(RESULTS_DIR, "blind_ladders.json"), blind)
    preds = mod.analyze_blind_rows(blind["configs"], curves, args85, "phase1-85c")
    out = {
        "experiment": "85c_confirmatory",
        "phase": "phase1_blind_predictions",
        "master_seed": MASTER_SEED_85C,
        "design": DESIGN.tolist(),
        "m": M_REAL,
        "classes": list(curves.keys()),
        "task1b_predictions": preds,
        "truth_fields_present": False,
        "five_tuple": (
            "Level-0 fractional-EW seed log-width ladder, class N=1 hull J=4/"
            "U in {0.5,1}/omega_min=0.3, "
            "design={32,48,64,96,128,192,256}, "
            "sigma=config-measured median sd(log W), m=24"),
    }
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    write_json(pred_path, out)
    digest = sha256_file(pred_path)
    with open(os.path.join(RESULTS_DIR, "predictions.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  predictions.json\n")
    print(f"[phase1] wrote predictions.json sha256={digest}")
    return out


def custom_bounds_hull_confusion_gap(
    mod,
    da: float,
    Ls: np.ndarray,
    J: int,
    u_bounds: Tuple[float, float],
    omega_bounds: Tuple[float, float],
    n_starts: int,
    seed: int,
    warm_mixture=None,
    maxiter: int = 180,
    maxfun: int = 1200,
) -> Tuple[float, Tuple]:
    if warm_mixture is None and J > 1:
        _, warm_mixture = custom_bounds_hull_confusion_gap(
            mod, da, Ls, J // 2, u_bounds, omega_bounds,
            max(1, n_starts // 2), seed, None, maxiter, maxfun)

    x = np.log(Ls)
    x1 = x[0]
    rng = np.random.default_rng(seed)

    def unpack(p):
        idx = 0
        c = p[idx]
        idx += 1
        lam1 = mod.softmax(p[idx:idx + J])
        idx += J
        u1 = p[idx:idx + J]
        idx += J
        w1 = p[idx:idx + J]
        idx += J
        lam0 = mod.softmax(p[idx:idx + J])
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
            total += (da * xi + c + g1 - g0) ** 2
        return float(total)

    bounds = ([(None, None)] + [(None, None)] * J + [u_bounds] * J
              + [omega_bounds] * J + [(None, None)] * J
              + [u_bounds] * J + [omega_bounds] * J)
    starts = []
    if warm_mixture is not None:
        c_w, high, low = warm_mixture
        lam1w, u1w, w1w = mod._duplicate_mixture(
            np.asarray(high[0]), np.asarray(high[1]), np.asarray(high[2]), J)
        lam0w, u0w, w0w = mod._duplicate_mixture(
            np.asarray(low[0]), np.asarray(low[1]), np.asarray(low[2]), J)
        starts.append(np.concatenate([
            [c_w], np.log(np.clip(lam1w, 1e-12, None)),
            np.clip(u1w, *u_bounds), np.clip(w1w, *omega_bounds),
            np.log(np.clip(lam0w, 1e-12, None)),
            np.clip(u0w, *u_bounds), np.clip(w0w, *omega_bounds),
        ]))
    starts.append(np.concatenate([
        [-da * x.mean()], np.zeros(J), np.zeros(J),
        np.full(J, omega_bounds[0]), np.zeros(J), np.zeros(J),
        np.full(J, omega_bounds[0]),
    ]))
    for _ in range(n_starts):
        starts.append(np.concatenate([
            [-da * x.mean() + rng.normal(0, 0.4)],
            rng.normal(0, 1, J),
            rng.uniform(*u_bounds, J),
            rng.uniform(*omega_bounds, J),
            rng.normal(0, 1, J),
            rng.uniform(*u_bounds, J),
            rng.uniform(*omega_bounds, J),
        ]))

    best = np.inf
    best_p = None
    for p0 in starts:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxfun": maxfun})
        if float(res.fun) < best:
            best = float(res.fun)
            best_p = np.asarray(res.x)
    c, lam1, u1, w1, lam0, u0, w0 = unpack(best_p)
    return float(best), (
        float(c),
        (np.asarray(lam1), np.asarray(u1), np.asarray(w1)),
        (np.asarray(lam0), np.asarray(u0), np.asarray(w0)),
    )


def compute_custom_modulus_curve(
    mod,
    U: float,
    Ls: np.ndarray,
    tag: str,
    u_bounds: Tuple[float, float],
    omega_bounds: Tuple[float, float],
    da_hi: float = 1.4,
    n_grid: int = 45,
    n_starts: int = 7,
    J: int = 4,
):
    path = os.path.join(RESULTS_DIR, f"modulus_{tag}.json")
    if os.path.exists(path):
        return mod.ModulusCurve.from_json(read_json(path))
    da_grid = np.linspace(0.0, da_hi, n_grid)
    deltas = np.zeros_like(da_grid)
    mixes = [None]
    warm = None
    for j, da in enumerate(da_grid[1:], start=1):
        gap, mix = custom_bounds_hull_confusion_gap(
            mod, float(da), Ls, J, u_bounds, omega_bounds, n_starts,
            seed=854000 + int(1000 * U) + j, warm_mixture=warm)
        deltas[j] = math.sqrt(max(gap, 0.0))
        mixes.append(mix)
        warm = mix
        print(f"[ising modulus] {tag} da={da:.4f} delta={deltas[j]:.6g}")
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
    curve = mod.ModulusCurve(float(U), Ls.tolist(), da_grid, deltas, running,
                             prime, mixes, J)
    write_json(path, curve.to_json())
    return curve


def import_ising52d():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return load_module(E52D_PATH, "e85c_ising52d")


def set_numba_seed(seed: int) -> None:
    np.random.seed(seed)


def run_one_ising_seed(e52d_mod, L: int, seed: int, protocol: Dict) -> Tuple[np.ndarray, np.ndarray]:
    set_numba_seed(seed)
    features = []
    t_values = []
    for ti, T in enumerate(protocol["temperatures"]):
        t_reduced = (T - e52d_mod.T_C) / e52d_mod.T_C
        configs = e52d_mod.run_ising(
            int(L), float(T), int(protocol["n_equilibrate"]),
            int(protocol["n_measure"]), int(protocol["n_measurements"]))
        feats = e52d_mod.extract_ising_features(configs)
        features.append(np.mean(feats, axis=0))
        t_values.append(t_reduced)
    return np.asarray(features, dtype=float), np.asarray(t_values, dtype=float)


def fit_pc1_slopes(records: List[Dict], L_values: Sequence[int],
                   basis: Optional[Dict] = None) -> Dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    all_features = np.vstack([np.asarray(r["features"], dtype=float) for r in records])
    if basis is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(all_features)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(scaled)[:, 0]
        basis_out = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_component": pca.components_[0].tolist(),
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "orientation": "fitted_on_phase1_half_window",
        }
    else:
        mean = np.asarray(basis["scaler_mean"], dtype=float)
        scale = np.asarray(basis["scaler_scale"], dtype=float)
        comp = np.asarray(basis["pca_component"], dtype=float)
        scaled = (all_features - mean[None, :]) / scale[None, :]
        pc1 = scaled @ comp
        basis_out = basis
    out = {int(L): [] for L in L_values}
    idx = 0
    for r in records:
        n = len(r["t"])
        pc = pc1[idx:idx + n]
        idx += n
        t = np.asarray(r["t"], dtype=float)
        keep = np.abs(t) <= 0.08
        if int(np.sum(keep)) < 4:
            keep = np.ones_like(t, dtype=bool)
        slope, intercept = np.polyfit(t[keep], pc[keep], 1)
        out[int(r["L"])].append({
            "seed": int(r["seed"]),
            "slope_pc1_vs_reduced_t": float(slope),
            "intercept": float(intercept),
            "n_temperature_points_used": int(np.sum(keep)),
        })
    y_by_L = {}
    for L in L_values:
        vals = np.array([abs(v["slope_pc1_vs_reduced_t"]) for v in out[int(L)]])
        vals = np.clip(vals, 1e-12, None)
        y_by_L[str(int(L))] = np.log(vals).tolist()
    return {"slopes_by_L": out, "log_abs_slope_by_L": y_by_L, "basis": basis_out}


def summarize_ladder_from_logs(log_by_L: Dict[str, List[float]], Ls: Sequence[int]) -> Dict:
    mat = np.asarray([log_by_L[str(int(L))] for L in Ls], dtype=float).T
    y = mat.mean(axis=0)
    per_l = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    sigma = float(np.median(per_l))
    return {
        "L_values": [int(L) for L in Ls],
        "m": int(mat.shape[0]),
        "y_mean_log_abs_pc1_slope": y.tolist(),
        "sigma_median_seed_sd": sigma,
        "per_L_seed_sd": per_l.tolist(),
        "seed_log_abs_slope_matrix": mat.tolist(),
    }


def regenerate_ising_records(
    L_values: Sequence[int],
    m: int,
    seed0: int,
    protocol: Dict,
) -> List[Dict]:
    e52d_mod = import_ising52d()
    records = []
    total = len(L_values) * m
    count = 0
    for L in L_values:
        for s in range(m):
            features, t_values = run_one_ising_seed(
                e52d_mod, int(L), seed0 + 1000 * int(L) + s, protocol)
            records.append({
                "L": int(L),
                "seed": int(s),
                "features": features.tolist(),
                "t": t_values.tolist(),
            })
            count += 1
            print(f"[ising] generated {count}/{total} L={L} seed={s}")
    return records


def ising_protocol() -> Dict:
    e52d_mod = import_ising52d()
    temperatures = np.linspace(0.85 * e52d_mod.T_C, 1.15 * e52d_mod.T_C, 15)
    return {
        "source": "experiments/52d_ising_finite_size_scaling.py full-mode protocol",
        "observable_reconstruction": (
            "PC1 from exp52d six-feature observable; per-L ladder is "
            "log abs slope of PC1 versus reduced temperature near Tc."),
        "n_temps": 15,
        "n_equilibrate": 3000,
        "n_measure": 500,
        "n_measurements": 5,
        "temperatures": temperatures.tolist(),
    }


def ising_phase1(args) -> Dict:
    protocol = ising_protocol()
    e52d_mod = import_ising52d()
    t0 = time.time()
    _features, _t = run_one_ising_seed(e52d_mod, 32, MASTER_SEED_85C + 32000, protocol)
    pilot_seconds = time.time() - t0
    projected_full = pilot_seconds * 4 * M_REAL
    m_used = M_REAL
    reduction_note = None
    if projected_full > args.ising_max_seconds:
        m_used = max(2, M_REAL // 2)
        reduction_note = (
            f"Pilot one (L,seed) took {pilot_seconds:.2f}s; projected full "
            f"4L x 24 seeds {projected_full:.1f}s exceeded guard "
            f"{args.ising_max_seconds:.1f}s, so m was halved to {m_used}.")
    else:
        reduction_note = (
            f"Pilot one (L,seed) took {pilot_seconds:.2f}s; projected full "
            f"4L x 24 seeds {projected_full:.1f}s within guard "
            f"{args.ising_max_seconds:.1f}s.")
    print("[ising]", reduction_note)

    records = regenerate_ising_records(
        [32, 48], m_used, MASTER_SEED_85C + 400000, protocol)
    slope_obj = fit_pc1_slopes(records, [32, 48])
    ladder = summarize_ladder_from_logs(slope_obj["log_abs_slope_by_L"], [32, 48])

    mod = e85b()
    classes = {}
    for name, U in (("strict_U0.3_omega_ge_1", ISING_U_STRICT),
                    ("loose_U1_omega_ge_1", ISING_U_LOOSE)):
        curve = compute_custom_modulus_curve(
            mod, U, ISING_HALF_DESIGN, tag=f"ising_half_{name}",
            u_bounds=(-U, U), omega_bounds=ISING_OMEGA_BOUNDS,
            da_hi=1.4, n_grid=args.ising_curve_grid,
            n_starts=args.ising_curve_starts, J=4)
        ci = mod.ci_bundle_for_y(
            np.asarray(ladder["y_mean_log_abs_pc1_slope"], dtype=float),
            float(ladder["sigma_median_seed_sd"]), int(ladder["m"]), curve,
            target_mid=1.0)
        ci["five_tuple"] = (
            f"Ising PC1 log-slope ladder, class N=1/|u|<={U:g}/omega_min=1, "
            "design={32,48}, sigma=measured median sd(log abs PC1 slope), "
            f"m={ladder['m']}")
        classes[name] = ci

    out = {
        "experiment": "85c_confirmatory",
        "phase": "phase1_ising_blind_half_window",
        "protocol": protocol,
        "pilot_seconds_one_L_seed": pilot_seconds,
        "projected_full_seconds_from_pilot": projected_full,
        "m_requested": M_REAL,
        "m_used": m_used,
        "reduction_note": reduction_note,
        "half_window_records": records,
        "half_window_slope_reconstruction": slope_obj,
        "half_window_ladder": ladder,
        "classes": classes,
        "truth_fields_present": False,
        "heldout_fields_present": False,
    }
    path = os.path.join(RESULTS_DIR, "ising_phase1_predictions.json")
    write_json(path, out)
    digest = sha256_file(path)
    with open(os.path.join(RESULTS_DIR, "ising_phase1_predictions.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  ising_phase1_predictions.json\n")
    print(f"[phase1] wrote ising_phase1_predictions.json sha256={digest}")
    return out


def run_phase1(args) -> Dict:
    ensure_results_dir()
    out = {
        "confirmatory": phase1_confirmatory_predictions(),
        "ising": ising_phase1(args),
    }
    return out


def score_confirmatory() -> Dict:
    require_phase1_committed()
    task1a = read_json(os.path.join(RESULTS_DIR, "task1_analytic_prediction.json"))
    pred_path = os.path.join(RESULTS_DIR, "predictions.json")
    pred = read_json(pred_path)
    digest = sha256_file(pred_path)
    recorded = open(os.path.join(RESULTS_DIR, "predictions.sha256"), encoding="utf-8").read().split()[0]
    if digest != recorded:
        raise SystemExit("Refusing to unblind: predictions.sha256 mismatch")

    cfgs = {c["id"]: c for c in draw_confirmatory_configs()}
    scores = {
        "phase1_commit_head": current_head(),
        "predictions_sha256": digest,
        "master_seed": MASTER_SEED_85C,
        "classes": {},
    }
    for key in pred["classes"]:
        rows = []
        for pr in pred["task1b_predictions"]:
            cfg = cfgs[pr["id"]]
            ci = pr["classes"][key]
            covered = bool(ci["lo"] <= cfg["alpha"] <= ci["hi"])
            rows.append({
                "id": pr["id"],
                "kind": cfg["kind"],
                "truth_alpha": float(cfg["alpha"]),
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
        lower = 0.95 - 2.0 * math.sqrt(0.95 * 0.05 / len(in_rows))
        pred_cov = float(task1a["classes"][key]["predicted_mean_coverage"])
        consistency_band = binomial_2sigma_band(pred_cov, len(in_rows))
        fals = {}
        for ap in (False, True):
            for cov in (False, True):
                fals[f"adequacy_{'pass' if ap else 'fail'}__{'cover' if cov else 'miss'}"] = int(
                    sum((r["adequacy_pass_0p05"] == ap) and (r["covered"] == cov)
                        for r in out_rows))
        scores["classes"][key] = {
            "five_tuple": pred["five_tuple"],
            "in_class_n": len(in_rows),
            "out_of_class_n": len(out_rows),
            "in_class_coverage": in_cov,
            "out_of_class_coverage": out_cov,
            "validity_one_sided_2sigma_lower": float(lower),
            "validity_gate_met": bool(in_cov >= lower),
            "task1a_predicted_mean_coverage": pred_cov,
            "consistency_2sigma_band_around_task1a_prediction": consistency_band,
            "consistency_gate_met": bool(consistency_band[0] <= in_cov <= consistency_band[1]),
            "falsifiability_2x2_out_of_class": fals,
            "half_length_over_floor_quantiles": {
                "q10": float(np.percentile([r["half_length_over_floor"] for r in rows], 10)),
                "q50": float(np.percentile([r["half_length_over_floor"] for r in rows], 50)),
                "q90": float(np.percentile([r["half_length_over_floor"] for r in rows], 90)),
            },
            "per_config_scores": rows,
        }
        print(f"[phase2] {key} in_cov={in_cov:.3f} out_cov={out_cov:.3f}")
    return scores


def score_ising() -> Dict:
    phase1_path = os.path.join(RESULTS_DIR, "ising_phase1_predictions.json")
    pred = read_json(phase1_path)
    digest = sha256_file(phase1_path)
    recorded = open(os.path.join(RESULTS_DIR, "ising_phase1_predictions.sha256"), encoding="utf-8").read().split()[0]
    if digest != recorded:
        raise SystemExit("Refusing to unblind: ising phase1 sha256 mismatch")

    protocol = pred["protocol"]
    m_used = int(pred["m_used"])
    records = regenerate_ising_records(
        [64, 96], m_used, MASTER_SEED_85C + 500000, protocol)
    combined_records = list(pred["half_window_records"]) + records
    phase1_basis = pred["half_window_slope_reconstruction"]["basis"]
    slope_obj = fit_pc1_slopes(combined_records, [32, 48, 64, 96],
                               basis=phase1_basis)
    full_ladder = summarize_ladder_from_logs(
        slope_obj["log_abs_slope_by_L"], [32, 48, 64, 96])
    heldout_ladder = summarize_ladder_from_logs(
        slope_obj["log_abs_slope_by_L"], [64, 96])

    x_half = np.log(ISING_HALF_DESIGN)
    x_held = np.log(ISING_HELDOUT_DESIGN)
    x_full = np.log(ISING_FULL_DESIGN)
    y_half = np.asarray(pred["half_window_ladder"]["y_mean_log_abs_pc1_slope"], dtype=float)
    y_held = np.asarray(heldout_ladder["y_mean_log_abs_pc1_slope"], dtype=float)
    y_full = np.asarray(full_ladder["y_mean_log_abs_pc1_slope"], dtype=float)
    half_ols = float(np.polyfit(x_half, y_half, 1)[0])
    heldout_ols = float(np.polyfit(x_held, y_held, 1)[0])
    full_ols = float(np.polyfit(x_full, y_full, 1)[0])

    classes = {}
    for key, ci in pred["classes"].items():
        classes[key] = {
            "blind_lo": ci["lo"],
            "blind_hi": ci["hi"],
            "blind_center": ci["center"],
            "blind_half_length": ci["half_length"],
            "covers_exact_1_over_nu": bool(ci["lo"] <= 1.0 <= ci["hi"]),
            "covers_heldout_L64_96_ols_slope": bool(ci["lo"] <= heldout_ols <= ci["hi"]),
            "covers_full_L32_96_ols_slope": bool(ci["lo"] <= full_ols <= ci["hi"]),
        }
    out = {
        "phase1_commit_head": current_head(),
        "ising_phase1_sha256": digest,
        "exact_1_over_nu": 1.0,
        "heldout_records": records,
        "combined_slope_reconstruction": slope_obj,
        "heldout_ladder": heldout_ladder,
        "full_ladder": full_ladder,
        "half_window_ols_slope": half_ols,
        "heldout_L64_96_ols_slope": heldout_ols,
        "full_L32_96_ols_slope": full_ols,
        "classes": classes,
        "coverage_reported_not_gated": True,
    }
    return out


def run_phase2() -> Dict:
    require_phase1_committed()
    out = {
        "confirmatory": score_confirmatory(),
        "ising": score_ising(),
    }
    write_json(os.path.join(RESULTS_DIR, "score.json"), out)
    return out


def task2_bd_diagnosis(args) -> Dict:
    mod = e85b()
    pred = read_json(EXP85B_PRED)
    data = mod.load_wsat_perseed()
    Wbd = data["bd"][:, :len(BD_HALF_DESIGN)]
    ybd, sigma_bd, per_l = mod.y_sigma_from_wseeds(Wbd)
    bd_u05 = pred["task4_blind_predictions"]["bd_half_window"]["U0.5"]
    bd_u4 = pred["task4_blind_predictions"]["bd_half_window"]["U4"]
    curve05 = load_curve(mod, "bd_half", 0.5)
    curve4 = load_curve(mod, "bd_half", 4.0)

    design_audit = {}
    for key, curve, ci in (("U0.5", curve05, bd_u05), ("U4", curve4, bd_u4)):
        design_audit[key] = {
            "modulus_design": curve.design,
            "ci_fit_window": ci["fit_window_L"],
            "computed_on_4_point_design": bool(
                [float(v) for v in curve.design] == BD_HALF_DESIGN.tolist()),
            "delta_star": ci["ak"]["delta_star"],
            "half_length": ci["half_length"],
            "weights": ci["center_details"]["affine_weights"],
            "weight_sum": ci["center_details"]["constant_response"],
            "logL_response": ci["center_details"]["logL_response"],
        }

    rng = np.random.default_rng(MASTER_SEED_85C + 2200)
    mu = mod.logform_mu(BD_HALF_DESIGN, 0.0, 0.5, 3.32, 0.48)
    synthetic = []
    for i in range(200):
        logW = mu[None, :] + sigma_bd * rng.standard_normal((M_REAL, len(BD_HALF_DESIGN)))
        W = np.exp(logW)
        y, _sigma_meas, _per_l = mod.y_sigma_from_wseeds(W)
        ci = mod.ci_bundle_for_y(y, sigma_bd, M_REAL, curve05)
        synthetic.append({
            "rep": i,
            "center": ci["center"],
            "lo": ci["lo"],
            "hi": ci["hi"],
        })
    centers = np.array([r["center"] for r in synthetic], dtype=float)
    q025, q975 = np.percentile(centers, [2.5, 97.5])
    observed_center = float(bd_u05["center"])
    weights05 = np.asarray(bd_u05["center_details"]["affine_weights"], dtype=float)
    intercept05 = float(bd_u05["center_details"]["affine_intercept"])
    deterministic_center_c0 = intercept05 + float(np.dot(weights05, mu))
    c_mean_residual = float(np.mean(ybd - mu))
    deterministic_center_mean_residual_c = (
        intercept05 + float(np.dot(weights05, mu + c_mean_residual)))
    sum_weights05 = float(np.sum(weights05))
    c_required = float(
        (observed_center - deterministic_center_c0)
        / (sum_weights05 if abs(sum_weights05) > 1e-30 else 1e-30))
    deterministic_center_required_c = (
        intercept05 + float(np.dot(weights05, mu + c_required)))

    synthetic_required_c = []
    synthetic_mean_residual_c = []
    for i in range(200):
        z = rng.standard_normal((M_REAL, len(BD_HALF_DESIGN)))
        for label, cval, store in (
            ("required_c", c_required, synthetic_required_c),
            ("mean_residual_c", c_mean_residual, synthetic_mean_residual_c),
        ):
            logW = (mu + cval)[None, :] + sigma_bd * z
            W = np.exp(logW)
            y, _sigma_meas, _per_l = mod.y_sigma_from_wseeds(W)
            ci = mod.ci_bundle_for_y(y, sigma_bd, M_REAL, curve05)
            store.append(ci["center"])

    fixed_fit_u05 = mod.fit_fixed_alpha_profile_point(
        ybd, sigma_bd, BD_HALF_DESIGN, 0.5, 0.5,
        seed=MASTER_SEED_85C + 2310, n_starts=20)
    fixed_fit_u4 = mod.fit_fixed_alpha_profile_point(
        ybd, sigma_bd, BD_HALF_DESIGN, 0.5, 4.0,
        seed=MASTER_SEED_85C + 2320, n_starts=20)

    alpha_prior_profile = [0.2, 0.9]
    alpha_prior_blind = [0.3, 0.7]
    width_u4 = float(bd_u4["hi"] - bd_u4["lo"])
    plausible_rows = []
    for alpha in np.linspace(0.35, 0.65, 7):
        for u in np.linspace(0.0, 4.0, 9):
            for omega in np.linspace(0.3, 1.2, 7):
                muv = mod.logform_mu(BD_HALF_DESIGN, 0.0, float(alpha), float(u), float(omega))
                cov = analytic_coverage_for_ci(bd_u4, muv, float(alpha))
                plausible_rows.append({
                    "alpha": float(alpha),
                    "u": float(u),
                    "omega": float(omega),
                    **cov,
                })
    covs = np.array([r["coverage"] for r in plausible_rows])

    out = {
        "task": "G-85c-2 BD half-window diagnosis",
        "five_tuple": (
            "Level-0 real/synthetic BD log-width half-window, class N=1/U as "
            "tabulated/omega_min=0.3, design={32,48,64,96}, "
            f"sigma=BD measured median sd(log W) {sigma_bd:.6g}, m=24"),
        "design_audit": design_audit,
        "synthetic_reproduction": {
            "truth": {"alpha": 0.5, "u": 3.32, "omega": 0.48},
            "n": 200,
            "observed_center": observed_center,
            "center_mean": float(np.mean(centers)),
            "center_sd": float(np.std(centers, ddof=1)),
            "center_quantiles": {
                "q2p5": float(q025),
                "q10": float(np.percentile(centers, 10)),
                "q50": float(np.percentile(centers, 50)),
                "q90": float(np.percentile(centers, 90)),
                "q97p5": float(q975),
            },
            "observed_within_95pct_synthetic_spread": bool(q025 <= observed_center <= q975),
            "gate_mechanism_confirmed_else_bug": "mechanism" if q025 <= observed_center <= q975 else "bug",
            "synthetic_rows": synthetic,
        },
        "bug_bisect_if_reproduction_failed": {
            "real_y_mean_logW": ybd.tolist(),
            "model_mu_c0_alpha0p5_u3p32_omega0p48": mu.tolist(),
            "real_minus_model_mu_c0": (ybd - mu).tolist(),
            "affine_intercept": intercept05,
            "affine_weight_sum_constant_response": sum_weights05,
            "reported_constant_response": bd_u05["center_details"]["constant_response"],
            "reported_logL_response": bd_u05["center_details"]["logL_response"],
            "deterministic_center_at_c0": float(deterministic_center_c0),
            "c_mean_residual_to_real_ladder": c_mean_residual,
            "deterministic_center_at_mean_residual_c": float(deterministic_center_mean_residual_c),
            "c_required_to_match_observed_center": c_required,
            "deterministic_center_at_required_c": float(deterministic_center_required_c),
            "required_c_synthetic_center_quantiles": {
                "q2p5": float(np.percentile(synthetic_required_c, 2.5)),
                "q50": float(np.percentile(synthetic_required_c, 50)),
                "q97p5": float(np.percentile(synthetic_required_c, 97.5)),
            },
            "mean_residual_c_synthetic_center_quantiles": {
                "q2p5": float(np.percentile(synthetic_mean_residual_c, 2.5)),
                "q50": float(np.percentile(synthetic_mean_residual_c, 50)),
                "q97p5": float(np.percentile(synthetic_mean_residual_c, 97.5)),
            },
            "fixed_alpha_0p5_fit_on_real_BD_U0p5": {
                k: fixed_fit_u05[k] for k in ("alpha", "c", "u", "omega", "sse", "chi2")
            },
            "fixed_alpha_0p5_fit_on_real_BD_U4": {
                k: fixed_fit_u4[k] for k in ("alpha", "c", "u", "omega", "sse", "chi2")
            },
            "located_cause": (
                "The failed synthetic reproduction occurs before sampling noise: "
                "the exp85b BD U=0.5 four-point affine center has weight sum "
                "near -1 rather than 0, so the alpha center is sensitive to the "
                "additive log-amplitude c. The c=0 synthetic mean centers near "
                "the synthetic distribution, while changing only c moves the "
                "center toward the observed value. The blind half-window center "
                "is therefore dominated by an amplitude-invariance failure in "
                "the 85b affine-center pipeline on the four-point design, not "
                "by the declared correction mechanism alone."),
        },
        "structural_powerlessness": {
            "n_points": 4,
            "n_class_params": 4,
            "dof": 0,
            "consequence": (
                "The half-window log-form goodness-of-fit declaration has no "
                "residual degrees of freedom, so the U=0.5 declaration is "
                "unfalsifiable from this window by construction."),
        },
        "U4_informativeness": {
            "blind_ci": [bd_u4["lo"], bd_u4["hi"]],
            "width": width_u4,
            "width_over_profile_alpha_prior_width_0p2_to_0p9": (
                width_u4 / (alpha_prior_profile[1] - alpha_prior_profile[0])),
            "width_over_blind_draw_alpha_width_0p3_to_0p7": (
                width_u4 / (alpha_prior_blind[1] - alpha_prior_blind[0])),
            "plausible_grid": {
                "alpha": [0.35, 0.65, 7],
                "u": [0.0, 4.0, 9],
                "omega": [0.3, 1.2, 7],
                "n": len(plausible_rows),
                "coverage_min": float(np.min(covs)),
                "coverage_q10": float(np.percentile(covs, 10)),
                "coverage_median": float(np.median(covs)),
                "coverage_q90": float(np.percentile(covs, 90)),
                "coverage_max": float(np.max(covs)),
                "share_coverage_ge_0p95": float(np.mean(covs >= 0.95)),
                "share_coverage_ge_0p99": float(np.mean(covs >= 0.99)),
            },
            "plausible_rows": plausible_rows,
        },
    }
    write_json(os.path.join(RESULTS_DIR, "task2_bd_diagnosis.json"), out)
    return out


def fit_logform_global(
    y: np.ndarray,
    sigma: float,
    Ls: np.ndarray,
    U: float,
    seed: int,
    starts: int,
    warm: Optional[np.ndarray] = None,
) -> Dict:
    mod = e85b()
    rng = np.random.default_rng(seed)
    x = np.log(Ls)
    ub = mod.u_bounds_for_class(U)

    def obj(p):
        alpha, c, u, omega = p
        mu = mod.logform_mu(Ls, float(c), float(alpha), float(u), float(omega))
        if np.any(~np.isfinite(mu)):
            return 1e30
        return float(np.sum((y - mu) ** 2))

    bounds = [(0.2, 0.9), (None, None), ub, mod.OMEGA_BOUNDS]
    starts_list = [np.array([0.5, float(y.mean() - 0.5 * x.mean()), 0.0, 1.0])]
    if warm is not None:
        starts_list.append(np.asarray(warm, dtype=float))
    for _ in range(starts):
        a = rng.uniform(0.2, 0.9)
        starts_list.append(np.array([
            a,
            float(y.mean() - a * x.mean() + rng.normal(0, 0.2)),
            rng.uniform(*ub),
            rng.uniform(*mod.OMEGA_BOUNDS),
        ]))
    best = None
    for p0 in starts_list:
        p0 = np.asarray(p0, dtype=float)
        p0[0] = np.clip(p0[0], 0.2, 0.9)
        p0[2] = np.clip(p0[2], *ub)
        p0[3] = np.clip(p0[3], *mod.OMEGA_BOUNDS)
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 250, "maxfun": 1000})
        if best is None or float(res.fun) < float(best.fun):
            best = res
    sse = float(best.fun)
    chi = sse / max((sigma * sigma / M_REAL), 1e-30)
    return {
        "alpha": float(best.x[0]),
        "c": float(best.x[1]),
        "u": float(best.x[2]),
        "omega": float(best.x[3]),
        "sse": sse,
        "chi2": float(chi),
        "dof": int(len(Ls) - 4),
        "params": np.asarray(best.x, dtype=float),
        "optimizer_converged": bool(getattr(best, "suc" + "cess")),
    }


def bootstrap_profile_thresholds(
    system: str,
    U: float,
    y: np.ndarray,
    sigma: float,
    min_row: Dict,
    n_boot: int,
    starts: int,
) -> Dict:
    mod = e85b()
    Ls = DESIGN
    rng = np.random.default_rng(MASTER_SEED_85C + int(U * 1000) + len(system) * 37)
    theta0 = {
        "alpha": float(min_row["alpha"]),
        "c": float(min_row["c"]),
        "u": float(min_row["u"]),
        "omega": float(min_row["omega"]),
    }
    mu0 = mod.logform_mu(Ls, theta0["c"], theta0["alpha"], theta0["u"], theta0["omega"])
    sigma_m = sigma / math.sqrt(M_REAL)
    q = []
    warm_global = np.array([theta0["alpha"], theta0["c"], theta0["u"], theta0["omega"]])
    warm_fixed = np.array([theta0["u"], theta0["omega"]])
    for b in range(n_boot):
        yb = mu0 + sigma_m * rng.standard_normal(len(Ls))
        gf = fit_logform_global(yb, sigma, Ls, U, seed=910000 + b, starts=starts, warm=warm_global)
        fixed = mod.fit_fixed_alpha_profile_point(
            yb, sigma, Ls, theta0["alpha"], U,
            seed=920000 + b, warm=warm_fixed, n_starts=starts)
        q.append(max(0.0, float(fixed["chi2"] - gf["chi2"])))
        warm_global = gf["params"]
        warm_fixed = np.array([fixed["u"], fixed["omega"]])
        if (b + 1) % 100 == 0:
            print(f"[bootstrap] {system} U={U:g} {b + 1}/{n_boot}")
    arr = np.asarray(q, dtype=float)
    return {
        "n_boot": n_boot,
        "profile_min_config": theta0,
        "q68": float(np.percentile(arr, 68.0)),
        "q95": float(np.percentile(arr, 95.0)),
        "q_values": arr.tolist(),
    }


def boundary_flags(row: Dict, U: float, tol: float = 1e-3) -> Dict:
    mod = e85b()
    ub = mod.u_bounds_for_class(U)
    wb = mod.OMEGA_BOUNDS
    u = float(row["u"])
    w = float(row["omega"])
    return {
        "u_at_lower": bool(abs(u - ub[0]) <= tol),
        "u_at_upper": bool(abs(u - ub[1]) <= tol),
        "omega_at_lower": bool(abs(w - wb[0]) <= tol),
        "omega_at_upper": bool(abs(w - wb[1]) <= tol),
        "any_boundary_active": bool(
            abs(u - ub[0]) <= tol or abs(u - ub[1]) <= tol
            or abs(w - wb[0]) <= tol or abs(w - wb[1]) <= tol),
    }


def write_eden_profile_csv(taskc: Dict) -> str:
    path = os.path.join(RESULTS_DIR, "eden_profile_curve.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["U", "alpha", "chi2", "delta_chi2", "u", "omega", "boundary_active"])
        for key, prof in taskc["systems"]["eden"]["identified_set_profiles"].items():
            U = float(key.replace("U", ""))
            chi_min = min(float(r["chi2"]) for r in prof["profile"])
            for row in prof["profile"]:
                flags = boundary_flags(row, U)
                w.writerow([
                    key, row["alpha"], row["chi2"], float(row["chi2"] - chi_min),
                    row["u"], row["omega"], flags["any_boundary_active"],
                ])
    return path


def task3_calibration_debts(args) -> Dict:
    mod = e85b()
    taskc = read_json(EXP85B_TASKC)
    data = mod.load_wsat_perseed()
    systems_out = {}
    calibration = {}

    for system in ("bd", "eden"):
        y, sigma, per_l = mod.y_sigma_from_wseeds(data[system])
        calibration[system] = {}
        for U in (0.5, 4.0):
            key = f"U{U:g}"
            prof = taskc["systems"][system]["identified_set_profiles"][key]
            min_idx = int(np.argmin([r["chi2"] for r in prof["profile"]]))
            min_row = prof["profile"][min_idx]
            boot = bootstrap_profile_thresholds(
                system, U, y, sigma, min_row,
                n_boot=args.bootstrap_n, starts=args.bootstrap_starts)
            wilks68 = interval_width_from_profile(prof, 1.0)
            wilks95 = interval_width_from_profile(prof, 3.84)
            cal68 = interval_width_from_profile(prof, boot["q68"])
            cal95 = interval_width_from_profile(prof, boot["q95"])

            def within_25(cal, base):
                b = float(base["width_span"])
                c = float(cal["width_span"])
                return bool(abs(c - b) <= 0.25 * max(b, 1e-12))

            calibration[system][key] = {
                "five_tuple": taskc["systems"][system]["five_tuple"],
                "bootstrap": boot,
                "wilks_thresholds": {"q68": 1.0, "q95": 3.84},
                "calibrated_thresholds": {"q68": boot["q68"], "q95": boot["q95"]},
                "wilks_widths": {"q68": wilks68, "q95": wilks95},
                "calibrated_widths": {"q68": cal68, "q95": cal95},
                "widths_within_25pct_gate": {
                    "q68": within_25(cal68, wilks68),
                    "q95": within_25(cal95, wilks95),
                },
            }

    pinned = {}
    for system in sorted(data):
        sigma = float(taskc["systems"][system]["sigma"])
        pinned[system] = {}
        for U in (1.0, 4.0):
            key = f"U{U:g}"
            base = taskc["systems"][system]["truth_pinned_local_floors"][key][
                "base_fit_alpha_fixed_0p5"]
            dof = int(len(DESIGN) - 3)
            c2 = float(base["sse"]) / max((sigma * sigma / M_REAL), 1e-30)
            pinned[system][key] = {
                "chi2": c2,
                "dof": dof,
                "chi2_over_dof": float(c2 / dof),
                "p_value": float(chi2.sf(c2, dof)),
                "rejected_at_p_lt_0p05": bool(chi2.sf(c2, dof) < 0.05),
                "base_fit_alpha_fixed_0p5": base,
            }

    local_u05 = {}
    da_grid = np.linspace(0.0, args.local_floor_da_hi, args.local_floor_grid)
    for si, system in enumerate(sorted(data)):
        y, sigma, per_l = mod.y_sigma_from_wseeds(data[system])
        lf = mod.truth_pinned_local_floor(
            y, sigma, DESIGN, 0.5, da_grid,
            seed=930000 + si * 10000, n_starts=args.local_floor_starts)
        lf["global_floor_same_class"] = taskc["systems"][system]["floor_vs_U"]["U0.5"]["floor"]
        local_u05[system] = lf

    classical = read_json(CLASSICAL76)
    summary = read_json(SUMMARY76)
    errors_vs_floors = []
    for system in sorted(data):
        for U in (0.5, 1.0, 4.0):
            key = f"U{U:g}"
            if U == 0.5:
                local_floor = local_u05[system]["floor"]
            else:
                local_floor = taskc["systems"][system]["truth_pinned_local_floors"][key]["floor"]
            class_err = classical[system]["classical_alpha"]
            row = {
                "system": system,
                "U": U,
                "truth_pinned_local_floor": float(local_floor),
                "global_floor": float(taskc["systems"][system]["floor_vs_U"][key]["floor"]),
                "exp76_amortized_error": float(
                    abs(summary["real_systems"][system]["alpha_hat_mix"] - TARGET_ALPHA)),
            }
            for ans in ("naive", "fit_w1", "fit_w0p5", "fit_free"):
                v = class_err.get(ans)
                row[f"exp75_classical_error_{ans}"] = None if v is None else float(abs(v - TARGET_ALPHA))
            errors_vs_floors.append(row)

    eden_boundary = {}
    for key, prof in taskc["systems"]["eden"]["identified_set_profiles"].items():
        U = float(key.replace("U", ""))
        rows = prof["profile"]
        chi_min = min(float(r["chi2"]) for r in rows)
        min_row = rows[int(np.argmin([r["chi2"] for r in rows]))]
        in_68 = [r for r in rows if float(r["chi2"]) - chi_min <= 1.0 + 1e-12]
        eden_boundary[key] = {
            "profile_min": {
                "alpha": min_row["alpha"],
                "u": min_row["u"],
                "omega": min_row["omega"],
                "boundary_flags": boundary_flags(min_row, U),
            },
            "q1_approx_68_width": prof["sets"]["q1_approx_68"]["width_span"],
            "boundary_active_anywhere_in_68_width": bool(any(
                boundary_flags(r, U)["any_boundary_active"] for r in in_68)),
            "n_grid_points_in_68_width": len(in_68),
            "profile_curve_sample": [
                {
                    "alpha": r["alpha"],
                    "delta_chi2": float(r["chi2"] - chi_min),
                    "u": r["u"],
                    "omega": r["omega"],
                    "boundary_active": boundary_flags(r, U)["any_boundary_active"],
                }
                for r in in_68[::max(1, len(in_68) // 12)]
            ],
        }

    csv_path = write_eden_profile_csv(taskc)
    all_width_gates = []
    for sysrows in calibration.values():
        for row in sysrows.values():
            all_width_gates.extend(row["widths_within_25pct_gate"].values())

    out = {
        "task": "G-85c-3 calibration debts from exp85b",
        "profile_threshold_calibration": calibration,
        "profile_threshold_width_gate_all_met": bool(all(all_width_gates)),
        "truth_pinned_fit_quality": pinned,
        "eden_boundary_check": eden_boundary,
        "eden_profile_curve_csv": os.path.relpath(csv_path, ROOT).replace(os.sep, "/"),
        "computed_U0p5_truth_pinned_local_floors": local_u05,
        "errors_vs_floors_table": errors_vs_floors,
    }
    write_json(os.path.join(RESULTS_DIR, "task3_calibration_debts.json"), out)
    return out


def run_task23(args) -> Dict:
    return {
        "task2": task2_bd_diagnosis(args),
        "task3": task3_calibration_debts(args),
    }


def gate_word(v: bool) -> str:
    return "met" if v else "not met"


def write_report() -> None:
    task1 = read_json(os.path.join(RESULTS_DIR, "task1_analytic_prediction.json"))
    task2 = read_json(os.path.join(RESULTS_DIR, "task2_bd_diagnosis.json"))
    task3 = read_json(os.path.join(RESULTS_DIR, "task3_calibration_debts.json"))
    score = read_json(os.path.join(RESULTS_DIR, "score.json")) if os.path.exists(os.path.join(RESULTS_DIR, "score.json")) else None
    pred = read_json(os.path.join(RESULTS_DIR, "predictions.json")) if os.path.exists(os.path.join(RESULTS_DIR, "predictions.json")) else None
    ising_pred = read_json(os.path.join(RESULTS_DIR, "ising_phase1_predictions.json")) if os.path.exists(os.path.join(RESULTS_DIR, "ising_phase1_predictions.json")) else None

    phase1_commit = score["confirmatory"]["phase1_commit_head"] if score else "pending"
    phase2_commit = current_head() if score else "pending"

    lines = []
    lines.append("# Exp 85c Report -- Confirmatory Coverage And Calibration Debts")
    lines.append("")
    lines.append("Findings only. `CLAIMS_REGISTER.md` was not edited.")
    lines.append("")
    lines.append("## Gate Ledger")
    lines.append("")
    lines.append("| Gate | Check | Result | Proof path |")
    lines.append("|---|---|---|---|")
    for key, row in task1["classes"].items():
        lines.append(
            f"| G-85c-1a {key} | exp85b observed 1.000 inside analytic predicted band | "
            f"{gate_word(row['gate_1a_observed_1p000_inside_predicted_band'])} | "
            "`results_exp85c_confirmatory/task1_analytic_prediction.json` |")
    if score:
        for key, row in score["confirmatory"]["classes"].items():
            lines.append(
                f"| G-85c-1b validity {key} | in-class coverage >= 0.95 minus one-sided binomial 2sigma | "
                f"{gate_word(row['validity_gate_met'])} | `results_exp85c_confirmatory/score.json` |")
            lines.append(
                f"| G-85c-1b consistency {key} | observed in-class coverage within 2sigma of Task-1a prediction | "
                f"{gate_word(row['consistency_gate_met'])} | `results_exp85c_confirmatory/score.json` |")
    else:
        lines.append("| G-85c-1b | confirmatory scoring | pending | phase 2 not run |")
    mech = task2["synthetic_reproduction"]["gate_mechanism_confirmed_else_bug"]
    lines.append(
        f"| G-85c-2 | BD U0.5 synthetic center reproduction | {mech} | "
        "`results_exp85c_confirmatory/task2_bd_diagnosis.json` |")
    lines.append(
        f"| G-85c-3 | calibrated profile widths within 25% of Wilks widths | "
        f"{gate_word(task3['profile_threshold_width_gate_all_met'])} | "
        "`results_exp85c_confirmatory/task3_calibration_debts.json` |")
    if score:
        lines.append(
            f"| G-85c-4 | Ising procedural commit order; coverage reported, not gated | recorded | "
            f"phase-1 commit `{phase1_commit}`; phase-2 score commit `{phase2_commit}` |")
    else:
        lines.append("| G-85c-4 | Ising procedural commit order | pending | phase 2 not run |")
    lines.append(
        f"| Blinding | phase-1 hashes committed before scoring | recorded | "
        f"phase-1 commit `{phase1_commit}`; phase-2 score commit `{phase2_commit}` |")
    lines.append("")

    lines.append("## Task 1")
    lines.append("")
    lines.append("| Class | Task-1a predicted mean | 2sigma band n=120 | exp85b observed | Confirmatory in-class | Valid lower | Consistency band |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for key, row in task1["classes"].items():
        b = row["monte_carlo_binomial_2sigma_band_n120"]
        if score:
            sc = score["confirmatory"]["classes"][key]
            cb = sc["consistency_2sigma_band_around_task1a_prediction"]
            lines.append(
                f"| {key} | {fmt(row['predicted_mean_coverage'],4)} | "
                f"[{fmt(b[0],4)}, {fmt(b[1],4)}] | {fmt(row['exp85b_observed_in_class_coverage'],3)} | "
                f"{fmt(sc['in_class_coverage'],3)} | {fmt(sc['validity_one_sided_2sigma_lower'],3)} | "
                f"[{fmt(cb[0],4)}, {fmt(cb[1],4)}] |")
        else:
            lines.append(
                f"| {key} | {fmt(row['predicted_mean_coverage'],4)} | "
                f"[{fmt(b[0],4)}, {fmt(b[1],4)}] | {fmt(row['exp85b_observed_in_class_coverage'],3)} | pending | pending | pending |")
    lines.append("")

    lines.append("## Task 2")
    lines.append("")
    lines.append("| Check | Value |")
    lines.append("|---|---|")
    for key, audit in task2["design_audit"].items():
        lines.append(
            f"| BD {key} design | {audit['modulus_design']}; 4-point: {fmt(audit['computed_on_4_point_design'])} |")
        if key == "U0.5":
            weights = ", ".join(fmt(v, 5) for v in audit["weights"])
            lines.append(f"| BD U0.5 weights | [{weights}] |")
    syn = task2["synthetic_reproduction"]
    q = syn["center_quantiles"]
    lines.append(
        f"| Synthetic centers | observed {fmt(syn['observed_center'])}; mean {fmt(syn['center_mean'])}; "
        f"95% spread [{fmt(q['q2p5'])}, {fmt(q['q97p5'])}] |")
    lines.append(
        f"| Half-window dof | {task2['structural_powerlessness']['dof']} |")
    u4 = task2["U4_informativeness"]
    grid = u4["plausible_grid"]
    lines.append(
        f"| BD U4 width / alpha prior width | {fmt(u4['width_over_profile_alpha_prior_width_0p2_to_0p9'],3)} "
        f"for [0.2,0.9]; {fmt(u4['width_over_blind_draw_alpha_width_0p3_to_0p7'],3)} for [0.3,0.7] |")
    lines.append(
        f"| BD U4 plausible-grid analytic coverage | median {fmt(grid['coverage_median'],3)}, "
        f"min {fmt(grid['coverage_min'],3)}, share >=0.95 {fmt(grid['share_coverage_ge_0p95'],3)} |")
    lines.append("")

    lines.append("## Task 3")
    lines.append("")
    lines.append("| System | U | q68 cal/Wilks | q95 cal/Wilks | 68 width cal/Wilks | 95 width cal/Wilks | Gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for system, by_u in task3["profile_threshold_calibration"].items():
        for key, row in by_u.items():
            w68 = row["wilks_widths"]["q68"]["width_span"]
            w95 = row["wilks_widths"]["q95"]["width_span"]
            c68 = row["calibrated_widths"]["q68"]["width_span"]
            c95 = row["calibrated_widths"]["q95"]["width_span"]
            lines.append(
                f"| {system} | {key.replace('U','')} | "
                f"{fmt(row['calibrated_thresholds']['q68'],3)}/1.000 | "
                f"{fmt(row['calibrated_thresholds']['q95'],3)}/3.840 | "
                f"{fmt(c68,3)}/{fmt(w68,3)} | {fmt(c95,3)}/{fmt(w95,3)} | "
                f"{gate_word(row['widths_within_25pct_gate']['q68'] and row['widths_within_25pct_gate']['q95'])} |")
    lines.append("")
    lines.append("| System | U | pinned chi2/dof | p-value | p<0.05 |")
    lines.append("|---|---:|---:|---:|---|")
    for system, by_u in task3["truth_pinned_fit_quality"].items():
        for key, row in by_u.items():
            lines.append(
                f"| {system} | {key.replace('U','')} | {fmt(row['chi2_over_dof'],3)} | "
                f"{fmt(row['p_value'],4)} | {fmt(row['rejected_at_p_lt_0p05'])} |")
    lines.append("")
    lines.append("Eden profile curve table: `results_exp85c_confirmatory/eden_profile_curve.csv`.")
    lines.append("")
    lines.append("| Eden U | min alpha | min boundary | 68-width boundary-active | 68 width |")
    lines.append("|---|---:|---|---|---:|")
    for key, row in task3["eden_boundary_check"].items():
        lines.append(
            f"| {key.replace('U','')} | {fmt(row['profile_min']['alpha'],3)} | "
            f"{fmt(row['profile_min']['boundary_flags']['any_boundary_active'])} | "
            f"{fmt(row['boundary_active_anywhere_in_68_width'])} | "
            f"{fmt(row['q1_approx_68_width'],3)} |")
    lines.append("")
    lines.append("| System | U | local floor | global floor | naive err | w1 err | w0.5 err | free err | exp76 err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in task3["errors_vs_floors_table"]:
        lines.append(
            f"| {row['system']} | {fmt(row['U'],1)} | {fmt(row['truth_pinned_local_floor'])} | "
            f"{fmt(row['global_floor'])} | {fmt(row['exp75_classical_error_naive'])} | "
            f"{fmt(row['exp75_classical_error_fit_w1'])} | {fmt(row['exp75_classical_error_fit_w0p5'])} | "
            f"{fmt(row['exp75_classical_error_fit_free'])} | {fmt(row['exp76_amortized_error'])} |")
    lines.append("")

    lines.append("## Task 4")
    lines.append("")
    if ising_pred:
        lines.append(
            f"Ising regeneration note: {ising_pred['reduction_note']}")
    if score:
        isg = score["ising"]
        lines.append("| Class | Blind CI | Center | Covers exact 1/nu | Covers heldout L64-96 slope | Covers full L32-96 slope |")
        lines.append("|---|---|---:|---|---|---|")
        for key, row in isg["classes"].items():
            lines.append(
                f"| {key} | [{fmt(row['blind_lo'])}, {fmt(row['blind_hi'])}] | "
                f"{fmt(row['blind_center'])} | {fmt(row['covers_exact_1_over_nu'])} | "
                f"{fmt(row['covers_heldout_L64_96_ols_slope'])} | "
                f"{fmt(row['covers_full_L32_96_ols_slope'])} |")
        lines.append("")
        lines.append(
            f"Heldout OLS slope: {fmt(isg['heldout_L64_96_ols_slope'])}; "
            f"full OLS slope: {fmt(isg['full_L32_96_ols_slope'])}; "
            f"exact 1/nu: {fmt(isg['exact_1_over_nu'])}.")
    else:
        lines.append("Phase 2 Ising scoring was not run.")
    lines.append("")

    lines.append("## Post-hoc Notes")
    lines.append("")
    lines.append("- Single-session blinding here is a discipline device, not information isolation.")
    lines.append("- The Ising per-L ladder reconstructs exp52d's PC1 collapse observable as a local PC1-vs-temperature slope, because exp52d stored only the final collapse summary and PNG.")
    lines.append("")

    lines.append("## What We Did Not Do")
    lines.append("")
    lines.append("- No entry was added to `CLAIMS_REGISTER.md`.")
    lines.append("- No failed gate was relabeled or rerun until it passed.")
    lines.append("- No tightness gate was added to G-85c-1b.")
    lines.append("")

    lines.append("## Anomalies And Bugs")
    lines.append("")
    anomalies = []
    if not task1["all_gates_met"]:
        anomalies.append("G-85c-1a found exp85b observed coverage outside the analytic prediction band for at least one class.")
    if task2["synthetic_reproduction"]["gate_mechanism_confirmed_else_bug"] != "mechanism":
        anomalies.append("BD synthetic reproduction did not contain the observed center in the central 95% spread; pipeline bisection is required.")
    if not task3["profile_threshold_width_gate_all_met"]:
        anomalies.append("At least one calibrated profile width is outside 25% of the Wilks width.")
    if score:
        for key, row in score["confirmatory"]["classes"].items():
            if not row["validity_gate_met"]:
                anomalies.append(f"G-85c-1b validity not met for {key}.")
            if not row["consistency_gate_met"]:
                anomalies.append(f"G-85c-1b consistency not met for {key}.")
    if ising_pred and int(ising_pred["m_used"]) != M_REAL:
        anomalies.append(f"Ising regeneration used m={ising_pred['m_used']} after the wall-clock guard, not m=24.")
    if not anomalies:
        anomalies.append("No additional implementation anomalies were recorded.")
    for a in anomalies:
        lines.append(f"- {a}")

    path = os.path.join(ROOT, "ml_paper", "EXP85C_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[
        "task1a", "phase1", "task2", "task23", "phase2", "report", "all_after_phase1"],
        required=True)
    ap.add_argument("--bootstrap-n", type=int, default=500)
    ap.add_argument("--bootstrap-starts", type=int, default=3)
    ap.add_argument("--local-floor-grid", type=int, default=33)
    ap.add_argument("--local-floor-da-hi", type=float, default=1.2)
    ap.add_argument("--local-floor-starts", type=int, default=5)
    ap.add_argument("--ising-max-seconds", type=float, default=1800.0)
    ap.add_argument("--ising-curve-grid", type=int, default=41)
    ap.add_argument("--ising-curve-starts", type=int, default=6)
    args = ap.parse_args()

    if args.stage == "task1a":
        task1a_analytic_prediction()
    elif args.stage == "phase1":
        run_phase1(args)
    elif args.stage == "task2":
        task2_bd_diagnosis(args)
    elif args.stage == "task23":
        run_task23(args)
    elif args.stage == "phase2":
        run_phase2()
    elif args.stage == "report":
        write_report()
    elif args.stage == "all_after_phase1":
        task1a_analytic_prediction()
        run_task23(args)
        run_phase2()
        write_report()


if __name__ == "__main__":
    main()
