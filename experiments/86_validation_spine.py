"""Exp 86: validation spine.

Stages are split to preserve the blind boundary:

  task1_phase1  fixed-estimator predictions only, no truth fields
  task1_phase2  score Task 1 after task1_phase1 artifacts are committed
  report        write ml_paper/EXP86_REPORT.md from available artifacts

Later tasks can extend this file without editing exp85b/85c.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize


HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

E85B_PATH = os.path.join(HERE, "85b_validation.py")
E85C_PATH = os.path.join(HERE, "85c_confirmatory.py")

EXP85B_RESULTS = os.path.join(ROOT, "results_exp85b_validation")
EXP85C_RESULTS = os.path.join(ROOT, "results_exp85c_confirmatory")
EXP86_TASK1 = os.path.join(ROOT, "results_exp86_task1")
EXP86_TASK2 = os.path.join(ROOT, "results_exp86_task2")

SUMMARY76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "summary_full24seed.json")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
BD_HALF_DESIGN = np.array([32., 48., 64., 96.])
M_REAL = 24
MASTER_SEED_85C = 85300
TARGET_ALPHA = 0.5
OMEGA_BOUNDS_DEFAULT = (0.3, 2.5)
ISING_OMEGA_BOUNDS = (1.0, 2.5)
ISING_HALF_DESIGN = np.array([32., 48.])
ISING_HELDOUT_DESIGN = np.array([64., 96.])
ISING_FULL_DESIGN = np.array([32., 48., 64., 96.])
ISING_U_STRICT = 0.3
ISING_U_LOOSE = 1.0
BIAS_BOUND_CACHE: Dict[Tuple, Dict] = {}


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def e85b():
    return load_module(E85B_PATH, "e86_reuse_85b")


def e85c():
    return load_module(E85C_PATH, "e86_reuse_85c")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def latest_commit_for_path(relpath: str) -> str:
    out = subprocess.check_output(
        ["git", "log", "-n", "1", "--format=%H", "--", relpath],
        cwd=ROOT,
        text=True,
    ).strip()
    return out if out else "pending"


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


def require_task1_phase1_committed() -> None:
    rels = [
        os.path.join("results_exp86_task1", "phase1_predictions.json"),
        os.path.join("results_exp86_task1", "phase1_predictions.sha256"),
    ]
    missing = [p for p in rels if not git_blob_exists(p)]
    if missing:
        raise SystemExit(
            "Refusing to score Task 1: phase-1 artifacts are not committed in HEAD: "
            f"{missing}")


def require_task2_phase1_committed() -> None:
    rels = [
        os.path.join("results_exp86_task2", "phase1_predictions.json"),
        os.path.join("results_exp86_task2", "phase1_predictions.sha256"),
    ]
    missing = [p for p in rels if not git_blob_exists(p)]
    if missing:
        raise SystemExit(
            "Refusing to score Task 2: phase-1 artifacts are not committed in HEAD: "
            f"{missing}")


def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{float(x):.{nd}f}"


def load_curve(mod, tag: str, U: float):
    return mod.ModulusCurve.from_json(
        read_json(os.path.join(EXP85B_RESULTS, f"modulus_{tag}_U{U:g}.json")))


def constrained_projection_weights(
    raw_w: np.ndarray,
    Ls: np.ndarray,
    target: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Closest equal-variance weights to raw_w subject to constant/slope constraints."""
    x = np.log(np.asarray(Ls, dtype=float))
    C = np.vstack([np.ones_like(x), x]).T
    target_vec = np.asarray(target, dtype=float)
    correction = C @ np.linalg.solve(C.T @ C, target_vec - C.T @ raw_w)
    return np.asarray(raw_w + correction, dtype=float)


def corr_log_scalar_bounds(
    Ls: np.ndarray,
    u: float,
    omega: float,
) -> np.ndarray:
    val = 1.0 + float(u) * (Ls / Ls[0]) ** (-float(omega))
    if np.any(val <= 0):
        return np.full_like(Ls, np.nan, dtype=float)
    return np.log(val)


def u_bounds_for_class(U: float, mod=None) -> Tuple[float, float]:
    if mod is None:
        mod = e85b()
    return mod.u_bounds_for_class(float(U))


def correction_bias_bound(
    weights: np.ndarray,
    Ls: np.ndarray,
    U: float,
    omega_bounds: Tuple[float, float] = OMEGA_BOUNDS_DEFAULT,
    u_bounds: Optional[Tuple[float, float]] = None,
    seed: int = 0,
    starts: int = 12,
) -> Dict:
    """Maximize |w'g(u,omega)| for one log-form correction atom.

    The correction class used by exp85b is the convex hull of these atoms; for a
    linear functional the supremum is attained at an extreme atom.
    """
    mod = e85b()
    Ls = np.asarray(Ls, dtype=float)
    w = np.asarray(weights, dtype=float)
    ub = tuple(float(v) for v in (u_bounds if u_bounds is not None else u_bounds_for_class(U, mod)))
    bounds = [ub, omega_bounds]
    rng = np.random.default_rng(seed)

    def val(p):
        g = corr_log_scalar_bounds(Ls, float(p[0]), float(p[1]))
        if np.any(~np.isfinite(g)):
            return np.nan
        return float(np.dot(w, g))

    candidates = []
    grid_u = np.linspace(ub[0], ub[1], 31)
    grid_om = np.linspace(omega_bounds[0], omega_bounds[1], 31)
    for u in grid_u:
        for om in grid_om:
            v = val((u, om))
            if np.isfinite(v):
                candidates.append((abs(v), v, np.array([u, om])))
    for _ in range(starts):
        candidates.append((0.0, 0.0, np.array([
            rng.uniform(*ub), rng.uniform(*omega_bounds)])))

    best = max(candidates, key=lambda t: t[0])
    best_signed = None
    for sign in (-1.0, 1.0):
        starts_local = [best[2]]
        starts_local.extend(np.array([rng.uniform(*ub), rng.uniform(*omega_bounds)])
                            for _ in range(starts))
        best_res = None

        def obj(p):
            v = val(p)
            if not np.isfinite(v):
                return 1e30
            return -sign * v

        for p0 in starts_local:
            res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "maxfun": 800})
            if best_res is None or float(res.fun) < float(best_res.fun):
                best_res = res
        v = val(best_res.x)
        cand = {
            "signed_value": float(v),
            "abs_value": float(abs(v)),
            "u": float(best_res.x[0]),
            "omega": float(best_res.x[1]),
            "sign_objective": float(sign),
        }
        if best_signed is None or cand["abs_value"] > best_signed["abs_value"]:
            best_signed = cand

    de_best = None
    for sign in (-1.0, 1.0):
        def obj_de(p):
            v = val(p)
            if not np.isfinite(v):
                return 1e30
            return -sign * v

        res = differential_evolution(
            obj_de, bounds=bounds, seed=seed + int(50 * (sign + 2)),
            maxiter=40, popsize=8, polish=True, tol=1e-7)
        v = val(res.x)
        cand = {
            "signed_value": float(v),
            "abs_value": float(abs(v)),
            "u": float(res.x[0]),
            "omega": float(res.x[1]),
            "sign_objective": float(sign),
        }
        if de_best is None or cand["abs_value"] > de_best["abs_value"]:
            de_best = cand

    best_final = max([best_signed, de_best], key=lambda z: z["abs_value"])
    best_final["u_bounds"] = [float(ub[0]), float(ub[1])]
    best_final["omega_bounds"] = [float(omega_bounds[0]), float(omega_bounds[1])]
    return best_final


def fixed_estimator_for_curve(
    curve,
    sigma: float,
    m: int,
    U: float,
    Ls: Optional[np.ndarray] = None,
    omega_bounds: Tuple[float, float] = OMEGA_BOUNDS_DEFAULT,
    u_bounds: Optional[Tuple[float, float]] = None,
    seed: int = 0,
) -> Dict:
    mod = e85b()
    if Ls is None:
        Ls = np.asarray(curve.design, dtype=float)
    info = mod.ak_half_length(curve, sigma, m)
    est = mod.affine_estimators(curve, info)
    raw_w = np.asarray(est["ak"]["affine_weights"], dtype=float)
    fixed_w = constrained_projection_weights(raw_w, Ls)
    x = np.log(np.asarray(Ls, dtype=float))
    bias_key = (
        tuple(np.round(fixed_w, 12).tolist()),
        tuple(np.round(np.asarray(Ls, dtype=float), 12).tolist()),
        float(U),
        tuple(float(v) for v in (u_bounds if u_bounds is not None else u_bounds_for_class(U, mod))),
        tuple(float(v) for v in omega_bounds),
    )
    if bias_key not in BIAS_BOUND_CACHE:
        BIAS_BOUND_CACHE[bias_key] = correction_bias_bound(
            fixed_w, Ls, U, omega_bounds=omega_bounds,
            u_bounds=u_bounds,
            seed=seed + int(1000 * U) + len(Ls))
    bias = dict(BIAS_BOUND_CACHE[bias_key])
    bias["cache_key_weights_rounded_12"] = list(bias_key[0])
    sigma_m = sigma / math.sqrt(m)
    sd = sigma_m * float(np.linalg.norm(fixed_w))
    t = bias["abs_value"] / max(sd, 1e-30)
    cv = mod.cv_abs_normal_shift(t)
    half = cv * sd
    return {
        "weights": fixed_w.tolist(),
        "raw_weights": raw_w.tolist(),
        "intercept": 0.0,
        "sum_weights": float(np.sum(fixed_w)),
        "logL_response": float(np.dot(fixed_w, x)),
        "raw_sum_weights": float(np.sum(raw_w)),
        "raw_logL_response": float(np.dot(raw_w, x)),
        "sd": float(sd),
        "sigma_m": float(sigma_m),
        "bias_bound": bias,
        "cv": float(cv),
        "half_length": float(half),
        "ak_grid_info": info,
        "projection_source": "equal-variance closest projection of exp85b A-K weights onto sum(w)=0 and w dot logL=1",
    }


def fixed_ci_for_y(
    y: np.ndarray,
    sigma: float,
    m: int,
    curve,
    U: float,
    seed: int,
    omega_bounds: Tuple[float, float] = OMEGA_BOUNDS_DEFAULT,
    u_bounds: Optional[Tuple[float, float]] = None,
) -> Dict:
    Ls = np.asarray(curve.design, dtype=float)
    est = fixed_estimator_for_curve(
        curve, sigma, m, U, Ls=Ls, omega_bounds=omega_bounds,
        u_bounds=u_bounds, seed=seed)
    weights = np.asarray(est["weights"], dtype=float)
    center = float(np.dot(weights, y))
    half = float(est["half_length"])
    c_test = 1.23456789
    center_shifted = float(np.dot(weights, y + c_test))
    return {
        "center": center,
        "lo": center - half,
        "hi": center + half,
        "half_length": half,
        "fixed_estimator": est,
        "amplitude_invariance_test": {
            "added_constant": c_test,
            "center_original": center,
            "center_shifted": center_shifted,
            "abs_difference": float(abs(center_shifted - center)),
        },
    }


@dataclass
class Args85B:
    reuse_curves: bool = True
    hull_J: int = 4
    curve_da_hi: float = 1.2
    curve_grid: int = 37
    curve_starts: int = 5
    main_gof_boot: int = 0
    main_gof_starts: int = 4


def task1_phase1() -> Dict:
    ensure_dir(EXP86_TASK1)
    mod = e85b()
    curves7 = {
        "U0.5": load_curve(mod, "fractional", 0.5),
        "U1": load_curve(mod, "fractional", 1.0),
    }
    curves_bd = {
        "U0.5": load_curve(mod, "bd_half", 0.5),
        "U4": load_curve(mod, "bd_half", 4.0),
    }

    exp85b_pred = read_json(os.path.join(EXP85B_RESULTS, "predictions.json"))
    old_bd = exp85b_pred["task4_blind_predictions"]["bd_half_window"]["U0.5"]
    old_w = np.asarray(old_bd["center_details"]["affine_weights"], dtype=float)
    old_x = np.log(BD_HALF_DESIGN)
    reproduce = {
        "five_tuple": (
            "Level-0 real BD log-width half-window, class N=1/U=0.5/"
            "omega_min=0.3, design={32,48,64,96}, "
            f"noise source=BD measured median sd(log W) {old_bd['sigma']:.6g}, m=24"),
        "exp85b_weights": old_w.tolist(),
        "sum_weights": float(np.sum(old_w)),
        "logL_response": float(np.dot(old_w, old_x)),
        "target_sum_weights": 0.0,
        "target_logL_response": 1.0,
    }

    fixed_weight_checks = {}
    for label, curve, U, sigma in [
        ("7pt_U0.5", curves7["U0.5"], 0.5, 0.30),
        ("7pt_U1", curves7["U1"], 1.0, 0.30),
        ("bd4_U0.5", curves_bd["U0.5"], 0.5, old_bd["sigma"]),
        ("bd4_U4", curves_bd["U4"], 4.0, old_bd["sigma"]),
    ]:
        est = fixed_estimator_for_curve(
            curve, sigma, M_REAL, U, seed=860100 + len(fixed_weight_checks))
        fixed_weight_checks[label] = {
            "sum_weights": est["sum_weights"],
            "logL_response": est["logL_response"],
            "raw_sum_weights": est["raw_sum_weights"],
            "raw_logL_response": est["raw_logL_response"],
            "weights": est["weights"],
            "half_length_at_reference_sigma": est["half_length"],
        }

    blind_path = os.path.join(EXP85C_RESULTS, "blind_ladders.json")
    blind = read_json(blind_path)
    rows = []
    for i, row in enumerate(blind["configs"]):
        W = np.asarray(row["W"], dtype=float)
        y, sigma, per_l = mod.y_sigma_from_wseeds(W)
        out = {
            "id": row["id"],
            "sigma": sigma,
            "per_L_sigma_logW": per_l,
            "classes": {},
        }
        for key, curve in curves7.items():
            U = float(key.replace("U", ""))
            ci = fixed_ci_for_y(
                y, sigma, W.shape[0], curve, U,
                seed=861000 + i * 100 + int(100 * U))
            ci["five_tuple"] = (
                f"Level-0 exp85c blind fractional-EW seed log-width ladder, "
                f"class N=1/{key}/omega_min=0.3, "
                "design={32,48,64,96,128,192,256}, "
                "noise source=config-measured median sd(log W), m=24")
            out["classes"][key] = ci
        rows.append(out)
        if (i + 1) % 25 == 0:
            print(f"[task1 phase1] fixed predictions {i + 1}/{len(blind['configs'])}")

    data = mod.load_wsat_perseed()
    Wbd = data["bd"][:, :len(BD_HALF_DESIGN)]
    ybd, sigma_bd, per_l_bd = mod.y_sigma_from_wseeds(Wbd)
    bd_fixed = {}
    for key, curve in curves_bd.items():
        U = float(key.replace("U", ""))
        ci = fixed_ci_for_y(
            ybd, sigma_bd, Wbd.shape[0], curve, U,
            seed=862000 + int(100 * U))
        ci["sigma"] = sigma_bd
        ci["per_L_sigma_logW"] = per_l_bd
        ci["fit_window_L"] = BD_HALF_DESIGN.tolist()
        ci["five_tuple"] = (
            f"Level-0 real BD log-width half-window, class N=1/{key}/"
            "omega_min=0.3, design={32,48,64,96}, "
            f"noise source=BD measured median sd(log W) {sigma_bd:.6g}, m=24")
        bd_fixed[key] = ci

    out = {
        "experiment": "86_validation_spine",
        "task": "G-86-1 phase1 fixed estimator predictions",
        "phase": "phase1_blind_predictions",
        "source_blind_ladders": "results_exp85c_confirmatory/blind_ladders.json",
        "truth_fields_present": False,
        "bd_truth_fields_present": False,
        "reproduce_exp85b_bug": reproduce,
        "fixed_weight_checks": fixed_weight_checks,
        "main_blind_predictions": rows,
        "bd_half_window_fixed_predictions": bd_fixed,
        "phase1_head_when_written": current_head(),
    }
    path = os.path.join(EXP86_TASK1, "phase1_predictions.json")
    write_json(path, out)
    digest = sha256_file(path)
    with open(os.path.join(EXP86_TASK1, "phase1_predictions.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  phase1_predictions.json\n")
    print(f"[task1 phase1] wrote phase1_predictions.json sha256={digest}")
    return out


def task1_phase2() -> Dict:
    require_task1_phase1_committed()
    phase1_path = os.path.join(EXP86_TASK1, "phase1_predictions.json")
    phase1 = read_json(phase1_path)
    digest = sha256_file(phase1_path)
    recorded = open(os.path.join(EXP86_TASK1, "phase1_predictions.sha256"), encoding="utf-8").read().split()[0]
    if digest != recorded:
        raise SystemExit("Refusing to score Task 1: phase1 sha256 mismatch")

    cfgs = {c["id"]: c for c in e85c().draw_confirmatory_configs(MASTER_SEED_85C)}
    exp85c_score = read_json(os.path.join(EXP85C_RESULTS, "score.json"))
    classes = {}
    for key in ("U0.5", "U1"):
        rows = []
        for pr in phase1["main_blind_predictions"]:
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
            })
        in_rows = [r for r in rows if r["kind"] == "in_class_single_power"]
        in_cov = float(np.mean([r["covered"] for r in in_rows]))
        baseline_cov = float(exp85c_score["confirmatory"]["classes"][key]["in_class_coverage"])
        n = len(in_rows)
        two_sigma = 2.0 * math.sqrt(0.95 * 0.05 / n)
        classes[key] = {
            "five_tuple": phase1["main_blind_predictions"][0]["classes"][key]["five_tuple"],
            "in_class_n": n,
            "fixed_estimator_in_class_coverage": in_cov,
            "exp85c_main_in_class_coverage": baseline_cov,
            "nominal_0p95_binomial_2sigma": two_sigma,
            "no_regression_gate_met": bool(abs(in_cov - baseline_cov) <= two_sigma + 1e-12),
            "per_config_scores": rows,
        }

    summary76 = read_json(SUMMARY76)
    bd_full = float(summary76["real_systems"]["bd"]["alpha_hat_mix"])
    bd = {}
    for key, ci in phase1["bd_half_window_fixed_predictions"].items():
        bd[key] = {
            "five_tuple": ci["five_tuple"],
            "lo": ci["lo"],
            "hi": ci["hi"],
            "center": ci["center"],
            "half_length": ci["half_length"],
            "covers_alpha_0p5": bool(ci["lo"] <= 0.5 <= ci["hi"]),
            "exp76_full_window_alpha_hat_mix": bd_full,
            "covers_exp76_full_window": bool(ci["lo"] <= bd_full <= ci["hi"]),
        }

    fixed_checks = phase1["fixed_weight_checks"]
    invariant_gate = bool(all(
        abs(v["sum_weights"]) < 1e-9 and abs(v["logL_response"] - 1.0) < 1e-9
        for v in fixed_checks.values()))
    amplitude_gate = bool(all(
        ci["amplitude_invariance_test"]["abs_difference"] < 1e-9
        for pr in phase1["main_blind_predictions"][:10]
        for ci in pr["classes"].values()))
    bd_amplitude_gate = bool(all(
        ci["amplitude_invariance_test"]["abs_difference"] < 1e-9
        for ci in phase1["bd_half_window_fixed_predictions"].values()))

    out = {
        "experiment": "86_validation_spine",
        "task": "G-86-1 phase2 fixed estimator scoring",
        "phase1_commit_head": current_head(),
        "phase1_predictions_sha256": digest,
        "gates": {
            "fixed_weight_constraints_met": invariant_gate,
            "amplitude_invariance_main_sample_met": amplitude_gate,
            "amplitude_invariance_bd_met": bd_amplitude_gate,
            "no_regression_7pt_coverage_met": bool(all(
                v["no_regression_gate_met"] for v in classes.values())),
            "G_86_1_all_met": bool(
                invariant_gate and amplitude_gate and bd_amplitude_gate
                and all(v["no_regression_gate_met"] for v in classes.values())),
        },
        "reproduce_exp85b_bug": phase1["reproduce_exp85b_bug"],
        "fixed_weight_checks": fixed_checks,
        "main_coverage": classes,
        "bd_half_window_unblind": bd,
        "truth_fields_loaded_after_phase1_commit": True,
    }
    write_json(os.path.join(EXP86_TASK1, "task1_score.json"), out)
    return out


def exp52d_protocol() -> Dict:
    helper = e85c()
    e52d_mod = helper.import_ising52d()
    temperatures = np.linspace(0.85 * e52d_mod.T_C, 1.15 * e52d_mod.T_C, 15)
    return {
        "source": "experiments/52d_ising_finite_size_scaling.py full-mode protocol",
        "observable": (
            "PC1 curves from the exp52d six-feature Ising representation; "
            "1/nu is extracted by minimizing compute_collapse_quality over nu."),
        "n_temps": 15,
        "n_equilibrate": 3000,
        "n_measure": 500,
        "n_measurements": 5,
        "temperatures": temperatures.tolist(),
    }


def regenerate_exp52d_records(
    L_values: Sequence[int],
    m: int,
    seed0: int,
    protocol: Dict,
) -> List[Dict]:
    helper = e85c()
    return helper.regenerate_ising_records(L_values, m, seed0, protocol)


def exp52d_collapse_summary(records: List[Dict], L_values: Sequence[int],
                            label: str) -> Dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    helper = e85c()
    e52d_mod = helper.import_ising52d()
    L_set = {int(L) for L in L_values}
    recs = sorted(
        [r for r in records if int(r["L"]) in L_set],
        key=lambda r: (int(r["L"]), int(r["seed"])),
    )
    all_features = np.vstack([np.asarray(r["features"], dtype=float) for r in recs])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(all_features)
    pca = PCA(n_components=min(6, all_features.shape[1]))
    coords = pca.fit_transform(scaled)

    pc1_by_L = {int(L): [] for L in L_values}
    t_by_L = {int(L): [] for L in L_values}
    slopes_by_L = {str(int(L)): [] for L in L_values}
    idx = 0
    for r in recs:
        L = int(r["L"])
        t = np.asarray(r["t"], dtype=float)
        n = len(t)
        pc1 = coords[idx:idx + n, 0]
        idx += n
        pc1_by_L[L].extend(pc1.tolist())
        t_by_L[L].extend(t.tolist())
        keep = np.abs(t) <= 0.08
        if int(np.sum(keep)) < 4:
            keep = np.ones_like(t, dtype=bool)
        slope, intercept = np.polyfit(t[keep], pc1[keep], 1)
        slopes_by_L[str(L)].append({
            "seed": int(r["seed"]),
            "slope_pc1_vs_reduced_t": float(slope),
            "intercept": float(intercept),
            "n_temperature_points_used": int(np.sum(keep)),
        })

    pc1_np = {int(L): np.asarray(pc1_by_L[int(L)], dtype=float) for L in L_values}
    t_np = {int(L): np.asarray(t_by_L[int(L)], dtype=float) for L in L_values}
    quality_exact = e52d_mod.compute_collapse_quality(pc1_np, t_np, list(map(int, L_values)), 1.0)
    nu_opt, quality_opt = e52d_mod.find_optimal_nu(pc1_np, t_np, list(map(int, L_values)))
    nu_grid = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    quality_grid = {
        f"{nu:.1f}": float(e52d_mod.compute_collapse_quality(
            pc1_np, t_np, list(map(int, L_values)), float(nu)))
        for nu in nu_grid
    }

    log_by_L = {}
    for L in L_values:
        vals = np.array([
            abs(row["slope_pc1_vs_reduced_t"])
            for row in slopes_by_L[str(int(L))]
        ], dtype=float)
        log_by_L[str(int(L))] = np.log(np.clip(vals, 1e-12, None)).tolist()
    mat = np.asarray([log_by_L[str(int(L))] for L in L_values], dtype=float).T
    y = mat.mean(axis=0)
    per_l = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros(mat.shape[1])
    sigma = float(np.median(per_l))
    leading_slope_ols = None
    if len(L_values) >= 2:
        leading_slope_ols = float(np.polyfit(np.log(np.asarray(L_values, dtype=float)), y, 1)[0])

    return {
        "label": label,
        "L_values": [int(L) for L in L_values],
        "m": int(mat.shape[0]),
        "exp52d_actual_observable": "argmin_nu compute_collapse_quality(PC1, t * L^(1/nu))",
        "nu_optimal": float(nu_opt),
        "one_over_nu_optimal": float(1.0 / nu_opt),
        "collapse_quality_at_nu_1": float(quality_exact),
        "collapse_quality_optimal": float(quality_opt),
        "collapse_quality_grid": quality_grid,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_point_count_by_L": {str(int(L)): int(len(pc1_by_L[int(L)])) for L in L_values},
        "leading_slope_ladder": {
            "observable_note": (
                "Auxiliary derivative ladder from the exp52d PC1 curves; "
                "not the exp52d collapse-argmin estimator."),
            "L_values": [int(L) for L in L_values],
            "m": int(mat.shape[0]),
            "y_mean_log_abs_pc1_slope": y.tolist(),
            "sigma_median_seed_sd": sigma,
            "per_L_seed_sd": per_l.tolist(),
            "seed_log_abs_slope_matrix": mat.tolist(),
            "leading_slope_ols_vs_logL": leading_slope_ols,
            "slopes_by_L": slopes_by_L,
        },
    }


def compute_custom_modulus_curve_exp86(
    U: float,
    Ls: np.ndarray,
    tag: str,
    u_bounds: Tuple[float, float],
    omega_bounds: Tuple[float, float],
    da_hi: float = 1.4,
    n_grid: int = 37,
    n_starts: int = 5,
    J: int = 4,
):
    ensure_dir(EXP86_TASK2)
    path = os.path.join(EXP86_TASK2, f"modulus_{tag}.json")
    mod = e85b()
    if os.path.exists(path):
        return mod.ModulusCurve.from_json(read_json(path))
    helper = e85c()
    da_grid = np.linspace(0.0, da_hi, n_grid)
    deltas = np.zeros_like(da_grid)
    mixes = [None]
    warm = None
    for j, da in enumerate(da_grid[1:], start=1):
        gap, mix = helper.custom_bounds_hull_confusion_gap(
            mod, float(da), Ls, J, u_bounds, omega_bounds, n_starts,
            seed=864000 + int(1000 * U) + j, warm_mixture=warm)
        deltas[j] = math.sqrt(max(gap, 0.0))
        mixes.append(mix)
        warm = mix
        print(f"[task2 modulus] {tag} da={da:.4f} delta={deltas[j]:.6g}")
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


def task2_phase1(args) -> Dict:
    ensure_dir(EXP86_TASK2)
    protocol = exp52d_protocol()
    helper = e85c()
    e52d_mod = helper.import_ising52d()
    t0 = time.time()
    _features, _t = helper.run_one_ising_seed(
        e52d_mod, 32, MASTER_SEED_85C + 860200, protocol)
    pilot_seconds = time.time() - t0
    projected_full = pilot_seconds * len(ISING_FULL_DESIGN) * M_REAL
    m_used = M_REAL
    if projected_full > args.ising_max_seconds:
        m_used = max(2, M_REAL // 2)
    reduction_note = (
        f"Pilot one (L,seed) took {pilot_seconds:.2f}s; projected 4L x 24 seeds "
        f"{projected_full:.1f}s against guard {args.ising_max_seconds:.1f}s; "
        f"m_used={m_used}.")
    print("[task2 phase1]", reduction_note)

    seed0_half = MASTER_SEED_85C + 860300
    records = regenerate_exp52d_records([32, 48], m_used, seed0_half, protocol)
    collapse = exp52d_collapse_summary(records, [32, 48], "half_window_L32_48")

    y = np.asarray(collapse["leading_slope_ladder"]["y_mean_log_abs_pc1_slope"], dtype=float)
    sigma = float(collapse["leading_slope_ladder"]["sigma_median_seed_sd"])
    classes = {}
    for name, U in (("strict_abs_u_le_0.3_omega_ge_1", ISING_U_STRICT),
                    ("loose_abs_u_le_1_omega_ge_1", ISING_U_LOOSE)):
        ub = (-float(U), float(U))
        curve = compute_custom_modulus_curve_exp86(
            U, ISING_HALF_DESIGN, tag=f"task2_half_{name}",
            u_bounds=ub, omega_bounds=ISING_OMEGA_BOUNDS,
            n_grid=args.ising_curve_grid, n_starts=args.ising_curve_starts)
        ci = fixed_ci_for_y(
            y, sigma, int(collapse["leading_slope_ladder"]["m"]),
            curve, U, seed=865000 + int(1000 * U),
            omega_bounds=ISING_OMEGA_BOUNDS, u_bounds=ub)
        ci["five_tuple"] = (
            f"Level-0 exp52d PC1 leading-slope auxiliary log-ladder, "
            f"class N=1/|u|<={U:g}/omega_min=1, design={{32,48}}, "
            "noise source=seed sd(log abs PC1 slope), "
            f"m={collapse['leading_slope_ladder']['m']}")
        ci["interpretation_note"] = (
            "This fixed affine CI is on the derivative ladder. The exp52d actual "
            "1/nu observable is the collapse-quality argmin, which is reported "
            "separately and is not an affine per-L scalar.")
        classes[name] = ci

    fixed_invariants = bool(all(
        abs(ci["fixed_estimator"]["sum_weights"]) < 1e-9
        and abs(ci["fixed_estimator"]["logL_response"] - 1.0) < 1e-9
        and ci["amplitude_invariance_test"]["abs_difference"] < 1e-9
        for ci in classes.values()))
    out = {
        "experiment": "86_validation_spine",
        "task": "G-86-2 phase1 Ising actual exp52d observable",
        "phase": "phase1_blind_predictions",
        "protocol": protocol,
        "pilot_seconds_one_L_seed": pilot_seconds,
        "projected_full_seconds_from_pilot": projected_full,
        "m_requested": M_REAL,
        "m_used": m_used,
        "reduction_note": reduction_note,
        "seed0_half_window": seed0_half,
        "half_window_records": records,
        "actual_collapse_half_window": collapse,
        "fixed_estimator_auxiliary_slope_CIs": classes,
        "fixed_estimator_invariants_hold_on_design": fixed_invariants,
        "truth_fields_present": False,
        "heldout_fields_present": False,
        "observable_anomaly": (
            "Exp52d extracts 1/nu by global PC1 collapse-quality minimization. "
            "It does not define a native affine per-L ladder; the Task-1 fixed "
            "estimator can only be applied to an auxiliary leading-slope ladder."),
        "phase1_head_when_written": current_head(),
    }
    path = os.path.join(EXP86_TASK2, "phase1_predictions.json")
    write_json(path, out)
    digest = sha256_file(path)
    with open(os.path.join(EXP86_TASK2, "phase1_predictions.sha256"), "w", encoding="utf-8") as fh:
        fh.write(f"{digest}  phase1_predictions.json\n")
    print(f"[task2 phase1] wrote phase1_predictions.json sha256={digest}")
    return out


def task2_phase2() -> Dict:
    require_task2_phase1_committed()
    phase1_path = os.path.join(EXP86_TASK2, "phase1_predictions.json")
    phase1 = read_json(phase1_path)
    digest = sha256_file(phase1_path)
    recorded = open(os.path.join(EXP86_TASK2, "phase1_predictions.sha256"), encoding="utf-8").read().split()[0]
    if digest != recorded:
        raise SystemExit("Refusing to score Task 2: phase1 sha256 mismatch")

    protocol = phase1["protocol"]
    m_used = int(phase1["m_used"])
    seed0_heldout = MASTER_SEED_85C + 860400
    heldout_records = regenerate_exp52d_records([64, 96], m_used, seed0_heldout, protocol)
    half_records = list(phase1["half_window_records"])
    heldout = exp52d_collapse_summary(heldout_records, [64, 96], "heldout_L64_96")
    full = exp52d_collapse_summary(
        half_records + heldout_records, [32, 48, 64, 96], "full_L32_96")

    exact_one_over_nu = 1.0
    classes = {}
    for key, ci in phase1["fixed_estimator_auxiliary_slope_CIs"].items():
        lo = float(ci["lo"])
        hi = float(ci["hi"])
        classes[key] = {
            "five_tuple": ci["five_tuple"],
            "blind_lo": lo,
            "blind_hi": hi,
            "blind_center": float(ci["center"]),
            "blind_half_length": float(ci["half_length"]),
            "covers_exact_1_over_nu": bool(lo <= exact_one_over_nu <= hi),
            "covers_heldout_actual_collapse_1_over_nu": bool(
                lo <= heldout["one_over_nu_optimal"] <= hi),
            "covers_full_actual_collapse_1_over_nu": bool(
                lo <= full["one_over_nu_optimal"] <= hi),
            "covers_heldout_auxiliary_slope_ols": bool(
                lo <= heldout["leading_slope_ladder"]["leading_slope_ols_vs_logL"] <= hi),
            "covers_full_auxiliary_slope_ols": bool(
                lo <= full["leading_slope_ladder"]["leading_slope_ols_vs_logL"] <= hi),
        }

    fixed_invariants = bool(phase1["fixed_estimator_invariants_hold_on_design"])
    out = {
        "experiment": "86_validation_spine",
        "task": "G-86-2 phase2 Ising unblind",
        "phase1_commit_head": current_head(),
        "phase1_predictions_sha256": digest,
        "exact_1_over_nu": exact_one_over_nu,
        "seed0_heldout": seed0_heldout,
        "heldout_records": heldout_records,
        "actual_collapse_half_window": phase1["actual_collapse_half_window"],
        "actual_collapse_heldout_L64_96": heldout,
        "actual_collapse_full_L32_96": full,
        "fixed_estimator_auxiliary_slope_unblind": classes,
        "gates": {
            "phase1_predictions_committed_before_heldout_generation": True,
            "phase1_sha256_verified": True,
            "fixed_estimator_invariants_hold_on_design": fixed_invariants,
            "G_86_2_procedural_met": bool(fixed_invariants),
        },
        "coverage_reported_not_gated_n_equals_1": True,
        "observable_anomaly": phase1["observable_anomaly"],
    }
    write_json(os.path.join(EXP86_TASK2, "task2_score.json"), out)
    return out


def write_report() -> None:
    task1_phase1_path = os.path.join(EXP86_TASK1, "phase1_predictions.json")
    task1_score_path = os.path.join(EXP86_TASK1, "task1_score.json")
    task2_phase1_path = os.path.join(EXP86_TASK2, "phase1_predictions.json")
    task2_score_path = os.path.join(EXP86_TASK2, "task2_score.json")
    phase1 = read_json(task1_phase1_path) if os.path.exists(task1_phase1_path) else None
    score1 = read_json(task1_score_path) if os.path.exists(task1_score_path) else None
    phase1_task2 = read_json(task2_phase1_path) if os.path.exists(task2_phase1_path) else None
    score2 = read_json(task2_score_path) if os.path.exists(task2_score_path) else None

    phase1_commit = score1["phase1_commit_head"] if score1 else "pending"
    phase2_commit = latest_commit_for_path("results_exp86_task1/task1_score.json") if score1 else "pending"
    task2_phase1_commit = score2["phase1_commit_head"] if score2 else "pending"
    task2_phase2_commit = latest_commit_for_path("results_exp86_task2/task2_score.json") if score2 else "pending"

    lines = []
    lines.append("# Exp 86 Report -- Validation Spine")
    lines.append("")
    lines.append("Findings only. `CLAIMS_REGISTER.md` was not edited.")
    lines.append("")
    lines.append("## Gate Ledger")
    lines.append("")
    lines.append("| Gate | Check | Result | Proof path |")
    lines.append("|---|---|---|---|")
    if score1:
        g = score1["gates"]
        lines.append(
            f"| G-86-1 | fixed constraints, amplitude invariance, 7-point no-regression | "
            f"{'met' if g['G_86_1_all_met'] else 'not met'} | `results_exp86_task1/task1_score.json` |")
        lines.append(
            f"| Task 1 blinding | fixed phase-1 predictions committed before truth scoring | recorded | "
            f"phase-1 commit `{phase1_commit}`; phase-2 score commit `{phase2_commit}` |")
    elif phase1:
        lines.append("| G-86-1 | fixed constraints, amplitude invariance, 7-point no-regression | pending | phase 2 not run |")
    else:
        lines.append("| G-86-1 | fixed constraints, amplitude invariance, 7-point no-regression | pending | task not run |")
    if score2:
        g2 = score2["gates"]
        lines.append(
            f"| G-86-2 | real Ising external anchor procedural gate | "
            f"{'met' if g2['G_86_2_procedural_met'] else 'not met'} | "
            f"phase-1 commit `{task2_phase1_commit}`; phase-2 score commit `{task2_phase2_commit}` |")
    elif phase1_task2:
        lines.append("| G-86-2 | real Ising external anchor procedural gate | pending | phase 2 not run |")
    else:
        lines.append("| G-86-2 | real Ising external anchor | pending | not run yet |")
    lines.append("| G-86-3 | exact honest CIs, smooth-prior van Trees, data-driven U | pending | not run yet |")
    lines.append("| G-86-4 | modulus prior-art pass | pending | not run yet |")
    lines.append("")

    if phase1:
        lines.append("## Task 1")
        lines.append("")
        rep = phase1["reproduce_exp85b_bug"]
        lines.append("| Quantity | Value |")
        lines.append("|---|---:|")
        lines.append(f"| exp85b BD U0.5 sum(w) | {fmt(rep['sum_weights'],4)} |")
        lines.append(f"| exp85b BD U0.5 w dot logL | {fmt(rep['logL_response'],4)} |")
        if score1:
            for label, row in score1["fixed_weight_checks"].items():
                lines.append(f"| fixed {label} sum(w) | {fmt(row['sum_weights'],10)} |")
                lines.append(f"| fixed {label} w dot logL | {fmt(row['logL_response'],10)} |")
        else:
            for label, row in phase1["fixed_weight_checks"].items():
                lines.append(f"| fixed {label} sum(w) | {fmt(row['sum_weights'],10)} |")
                lines.append(f"| fixed {label} w dot logL | {fmt(row['logL_response'],10)} |")
        lines.append("")
        if score1:
            lines.append("| Class | fixed in-class coverage | exp85c coverage | 2sigma slack | no-regression |")
            lines.append("|---|---:|---:|---:|---|")
            for key, row in score1["main_coverage"].items():
                lines.append(
                    f"| {key} | {fmt(row['fixed_estimator_in_class_coverage'],3)} | "
                    f"{fmt(row['exp85c_main_in_class_coverage'],3)} | "
                    f"{fmt(row['nominal_0p95_binomial_2sigma'],3)} | "
                    f"{fmt(row['no_regression_gate_met'])} |")
            lines.append("")
            lines.append("| BD class | fixed CI | center | covers alpha=0.5 | covers exp76 full-window |")
            lines.append("|---|---|---:|---|---|")
            for key, row in score1["bd_half_window_unblind"].items():
                lines.append(
                    f"| {key} | [{fmt(row['lo'])}, {fmt(row['hi'])}] | "
                    f"{fmt(row['center'])} | {fmt(row['covers_alpha_0p5'])} | "
                    f"{fmt(row['covers_exp76_full_window'])} |")
        else:
            lines.append("Task 1 phase 2 was not run.")
        lines.append("")

    if phase1_task2:
        lines.append("## Task 2")
        lines.append("")
        half = phase1_task2["actual_collapse_half_window"]
        lines.append("| Quantity | Value |")
        lines.append("|---|---:|")
        lines.append(f"| half-window actual collapse nu_opt | {fmt(half['nu_optimal'],4)} |")
        lines.append(f"| half-window actual collapse 1/nu_opt | {fmt(half['one_over_nu_optimal'],4)} |")
        lines.append(f"| half-window auxiliary slope vs logL | {fmt(half['leading_slope_ladder']['leading_slope_ols_vs_logL'],4)} |")
        if score2:
            held = score2["actual_collapse_heldout_L64_96"]
            full = score2["actual_collapse_full_L32_96"]
            lines.append(f"| heldout actual collapse 1/nu_opt | {fmt(held['one_over_nu_optimal'],4)} |")
            lines.append(f"| full actual collapse 1/nu_opt | {fmt(full['one_over_nu_optimal'],4)} |")
            lines.append(f"| exact 1/nu | {fmt(score2['exact_1_over_nu'],4)} |")
        lines.append("")
        lines.append("| Class | auxiliary fixed CI | center | covers exact 1/nu | covers heldout actual 1/nu | covers full actual 1/nu |")
        lines.append("|---|---|---:|---|---|---|")
        source_classes = (
            score2["fixed_estimator_auxiliary_slope_unblind"]
            if score2 else phase1_task2["fixed_estimator_auxiliary_slope_CIs"])
        for key, row in source_classes.items():
            lo = row.get("blind_lo", row.get("lo"))
            hi = row.get("blind_hi", row.get("hi"))
            center = row.get("blind_center", row.get("center"))
            lines.append(
                f"| {key} | [{fmt(lo)}, {fmt(hi)}] | {fmt(center)} | "
                f"{fmt(row.get('covers_exact_1_over_nu'))} | "
                f"{fmt(row.get('covers_heldout_actual_collapse_1_over_nu'))} | "
                f"{fmt(row.get('covers_full_actual_collapse_1_over_nu'))} |")
        lines.append("")
        lines.append(f"Observable note: {phase1_task2['observable_anomaly']}")
        lines.append("")

    lines.append("## Post-hoc Notes")
    lines.append("")
    lines.append("- Single-session blinding is a discipline device, not information isolation.")
    lines.append("- Task 1 uses the already-committed exp85c blind ladders for the 7-point no-regression check; fixed predictions were written and committed before truth configs were loaded for exp86 scoring.")
    if score2:
        lines.append("- Task 2 uses the actual exp52d collapse-quality minimizer for the Ising anchor; the fixed affine CI is reported on the auxiliary leading-slope ladder because the source estimator is global, not a native per-L affine observable.")
    lines.append("")

    lines.append("## What We Did Not Do")
    lines.append("")
    lines.append("- No entry was added to `CLAIMS_REGISTER.md`.")
    if score2:
        lines.append("- Tasks 3 and 4 are pending.")
    else:
        lines.append("- Tasks 2, 3, and 4 are pending in this partial report." if not score1 else "- Tasks 2, 3, and 4 are pending.")
    lines.append("")

    lines.append("## Anomalies And Bugs")
    lines.append("")
    anomalies = []
    if score1 and not score1["gates"]["G_86_1_all_met"]:
        anomalies.append("G-86-1 did not meet all required subgates.")
    if score2 and not score2["gates"]["G_86_2_procedural_met"]:
        anomalies.append("G-86-2 procedural gate did not meet all required subgates.")
    if phase1_task2:
        anomalies.append(phase1_task2["observable_anomaly"])
    if not score1:
        anomalies.append("Task 1 scoring is pending.")
    anomalies.append("Tasks 3-4 are not yet run in this report snapshot." if score2 else "Tasks 2-4 are not yet run in this report snapshot.")
    for a in anomalies:
        lines.append(f"- {a}")

    path = os.path.join(ROOT, "ml_paper", "EXP86_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["task1_phase1", "task1_phase2",
                                        "task2_phase1", "task2_phase2", "report"],
                    required=True)
    ap.add_argument("--ising-max-seconds", type=float, default=1800.0)
    ap.add_argument("--ising-curve-grid", type=int, default=37)
    ap.add_argument("--ising-curve-starts", type=int, default=5)
    args = ap.parse_args()
    if args.stage == "task1_phase1":
        task1_phase1()
    elif args.stage == "task1_phase2":
        task1_phase2()
    elif args.stage == "task2_phase1":
        task2_phase1(args)
    elif args.stage == "task2_phase2":
        task2_phase2()
    elif args.stage == "report":
        write_report()


if __name__ == "__main__":
    main()
