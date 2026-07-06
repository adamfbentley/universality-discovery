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

SUMMARY76 = os.path.join(
    ROOT, "results_exp76_amortized_extrapolation", "summary_full24seed.json")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
BD_HALF_DESIGN = np.array([32., 48., 64., 96.])
M_REAL = 24
MASTER_SEED_85C = 85300
TARGET_ALPHA = 0.5
OMEGA_BOUNDS_DEFAULT = (0.3, 2.5)
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
    ub = u_bounds_for_class(U, mod)
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
        tuple(float(v) for v in omega_bounds),
    )
    if bias_key not in BIAS_BOUND_CACHE:
        BIAS_BOUND_CACHE[bias_key] = correction_bias_bound(
            fixed_w, Ls, U, omega_bounds=omega_bounds,
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
) -> Dict:
    Ls = np.asarray(curve.design, dtype=float)
    est = fixed_estimator_for_curve(
        curve, sigma, m, U, Ls=Ls, omega_bounds=omega_bounds, seed=seed)
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


def write_report() -> None:
    task1_phase1_path = os.path.join(EXP86_TASK1, "phase1_predictions.json")
    task1_score_path = os.path.join(EXP86_TASK1, "task1_score.json")
    phase1 = read_json(task1_phase1_path) if os.path.exists(task1_phase1_path) else None
    score1 = read_json(task1_score_path) if os.path.exists(task1_score_path) else None

    phase1_commit = score1["phase1_commit_head"] if score1 else "pending"
    phase2_commit = current_head() if score1 else "pending"

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

    lines.append("## Post-hoc Notes")
    lines.append("")
    lines.append("- Single-session blinding is a discipline device, not information isolation.")
    lines.append("- Task 1 uses the already-committed exp85c blind ladders for the 7-point no-regression check; fixed predictions were written and committed before truth configs were loaded for exp86 scoring.")
    lines.append("")

    lines.append("## What We Did Not Do")
    lines.append("")
    lines.append("- No entry was added to `CLAIMS_REGISTER.md`.")
    lines.append("- Tasks 2, 3, and 4 are pending in this partial report." if not score1 else "- Tasks 2, 3, and 4 are pending.")
    lines.append("")

    lines.append("## Anomalies And Bugs")
    lines.append("")
    anomalies = []
    if score1 and not score1["gates"]["G_86_1_all_met"]:
        anomalies.append("G-86-1 did not meet all required subgates.")
    if not score1:
        anomalies.append("Task 1 scoring is pending.")
    anomalies.append("Tasks 2-4 are not yet run in this report snapshot.")
    for a in anomalies:
        lines.append(f"- {a}")

    path = os.path.join(ROOT, "ml_paper", "EXP86_REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["task1_phase1", "task1_phase2", "report"],
                    required=True)
    args = ap.parse_args()
    if args.stage == "task1_phase1":
        task1_phase1()
    elif args.stage == "task1_phase2":
        task1_phase2()
    elif args.stage == "report":
        write_report()


if __name__ == "__main__":
    main()
