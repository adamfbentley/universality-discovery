"""
Experiment 68: Collapse Geometry as a Universality Metric (Path 1, v1)
=====================================================================

Premise (docs/path1_collapse_metric_plan.md): the feature-geometry paper showed
Euclidean density clustering of finite-size features does NOT recover universality
classes, because the feature space is dominated by non-universal nuisance
directions. The correct operationalization of universality is equivalence under
the Family-Vicsek rescaling group: two systems share a class iff one rescaling
collapses their width curves onto a common master curve.

This experiment builds that on the Family-Vicsek width curve

    W(L,t) = L^alpha F(t/L^z),    W ~ t^beta (growth),   beta = alpha/z,

which encodes the exponents that SEPARATE EW from KPZ (beta: EW 1/4 vs KPZ 1/3;
z: EW 2 vs KPZ 3/2) -- exactly the degeneracy the static gradient features
(alpha = 1/2 for both) cannot break.

Calibration (see plan doc) fixed the methodology:
  * W(t) = sqrt(<var_x h>_seeds)  (seed-averaged variance; single curves too noisy)
  * beta by W-gating: fit log W vs log t over 0.2 Wsat < W < 0.5 Wsat
  * alpha from Wsat-vs-L scaling; z = alpha/beta (+ a proper 2D collapse z)
  * KPZ beta is an effective, lambda-dependent crossover exponent at accessible L
    (reported as effective, not asymptotic)

v1 scope: continuum systems EW, KPZ, KS, RD (clean time resolution). The discrete
KPZ-class models BD/Eden need sub-monolayer recording + intrinsic-width
subtraction and are the documented next step (v2).

Usage:
  python 68_collapse_metric.py --quick   # tiny smoke run
  python 68_collapse_metric.py           # pilot (default)
  python 68_collapse_metric.py --full    # larger L ladder
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_exp68_collapse"
EXP63_PATH = SCRIPT_DIR / "63_temporal_features.py"

SYSTEMS = ["ew", "kpz", "ks", "rd"]
CLASS_MAP = {"ew": "EW", "kpz": "KPZ", "ks": "KS", "rd": "trivial"}
SYSTEM_SEED_OFFSETS = {"ew": 0, "kpz": 1_000_000, "ks": 2_000_000, "rd": 3_000_000}
THEORY = {  # 1+1 D, validation reporting only
    "ew": {"alpha": 0.5, "beta": 0.25, "z": 2.0},
    "kpz": {"alpha": 0.5, "beta": 1 / 3, "z": 1.5},
    "ks": {"alpha": None, "beta": None, "z": None},
    "rd": {"alpha": 0.0, "beta": 0.5, "z": None},
}
KPZ_LAM = 4.0  # effective-beta crossover; see plan doc


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63", EXP63_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def T_of(system, L, cap):
    if system == "ew":
        return int(np.clip(0.5 * L * L, 1500, cap))
    if system == "kpz":
        return int(np.clip(0.6 * L * L, 1500, cap))
    if system == "ks":
        return min(4000, cap)
    if system == "rd":
        return min(2500, cap)
    raise ValueError(system)


# ----------------------------------------------------------------------------
# Width-curve generation: store per-seed variance curves var_x h(t)
# ----------------------------------------------------------------------------

def simulate_var_curves(exp63, system, L, T, n_seed, seed_base):
    out = []
    for s in range(n_seed):
        seed = seed_base + s
        if system == "ew":
            traj = exp63.simulate_ew(L=L, T=T, nu=1.0, D=1.0, seed=seed)
        elif system == "kpz":
            traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=KPZ_LAM, D=1.0, seed=seed)
            if traj is None:
                traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=2.0, D=0.6, seed=seed + 7)
        elif system == "ks":
            traj = exp63.simulate_ks(L=L, T=T, seed=seed, record_interval=1)
            if traj is None:
                traj = exp63.simulate_ks(L=L, T=T, seed=seed + 7, record_interval=1, noise=0.02)
        elif system == "rd":
            traj = exp63.simulate_rd(L=L, T=T, seed=seed)
        else:
            raise ValueError(system)
        if traj is None:
            continue
        v = traj.var(axis=1)
        if np.all(np.isfinite(v)):
            out.append(v)
    return np.array(out) if out else np.zeros((0, T))


# ----------------------------------------------------------------------------
# Exponent extraction
# ----------------------------------------------------------------------------

def beta_wgated(W, lo=0.2, hi=0.5):
    """Growth exponent: slope of log W vs log t over 0.2 Wsat < W < 0.5 Wsat."""
    T = len(W)
    wsat = float(np.median(W[int(0.85 * T):]))
    if wsat <= 0:
        return np.nan
    lt = np.log(np.arange(1, T + 1, dtype=float))
    lw = np.log(np.clip(W, 1e-12, None))
    m = (W > lo * wsat) & (W < hi * wsat)
    if m.sum() < 8:
        return np.nan
    return float(np.polyfit(lt[m], lw[m], 1)[0])


def exponents_from_varset(var_by_L):
    """var_by_L: dict L -> mean variance curve. Returns alpha, beta, z, Wcurves."""
    Ls = sorted(var_by_L)
    betas, wsat, Wcurves = [], [], {}
    for L in Ls:
        W = np.sqrt(np.clip(var_by_L[L], 0, None))
        Wcurves[L] = W
        b = beta_wgated(W)
        if np.isfinite(b):
            betas.append(b)
        T = len(W)
        wsat.append(float(np.median(W[int(0.85 * T):])))
    beta = float(np.mean(betas)) if betas else np.nan
    La = np.array(Ls, dtype=float)
    wa = np.array(wsat, dtype=float)
    if np.all(wa > 0) and len(Ls) >= 2:
        alpha = float(np.polyfit(np.log(La), np.log(wa), 1)[0])
    else:
        alpha = np.nan
    z = float(np.clip(alpha / beta, 0.0, 5.0)) if (np.isfinite(alpha) and np.isfinite(beta) and beta > 1e-6) else np.nan
    return alpha, beta, z, Wcurves


# ----------------------------------------------------------------------------
# Proper 2D Family-Vicsek collapse (validation cross-check)
# ----------------------------------------------------------------------------

def fv_collapse_quality(Wcurves, alpha, z, n_windows=25, min_L=2):
    us, ys, Lc = [], [], []
    for L, W in Wcurves.items():
        T = len(W)
        t = np.arange(1, T + 1, dtype=float)
        good = W > 0
        us.append(np.log(t[good]) - z * np.log(L))
        ys.append(np.log(W[good]) - alpha * np.log(L))
        Lc.append(np.full(int(good.sum()), L))
    u = np.concatenate(us); y = np.concatenate(ys); Lc = np.concatenate(Lc)
    if len(u) < 10:
        return np.inf
    order = np.argsort(u); u, y, Lc = u[order], y[order], Lc[order]
    edges = np.linspace(u.min(), u.max(), n_windows + 1)
    tot, nv = 0.0, 0
    for i in range(n_windows):
        hi = (u <= edges[i + 1]) if i == n_windows - 1 else (u < edges[i + 1])
        m = (u >= edges[i]) & hi
        if m.sum() >= 3 and len(set(Lc[m].tolist())) >= min_L:
            tot += float(np.var(y[m])); nv += 1
    return tot / nv if nv else np.inf


def fit_fv_collapse(Wcurves):
    best, best_q = (0.5, 1.5), np.inf
    for a in np.linspace(0.2, 1.0, 9):
        for z in np.linspace(0.8, 2.6, 10):
            q = fv_collapse_quality(Wcurves, a, z)
            if q < best_q:
                best, best_q = (a, z), q
    res = minimize(lambda p: fv_collapse_quality(Wcurves, p[0], p[1]),
                   x0=np.array(best), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
    a, z = float(res.x[0]), float(res.x[1])
    return a, z, (a / z if z > 1e-6 else np.nan), float(res.fun)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Exp 68 collapse-metric (continuum v1)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed-start", type=int, default=68_000)
    args = ap.parse_args()

    if args.quick:
        L_values, n_seed, n_boot, cap = [32, 48], 5, 12, 4000
    elif args.full:
        L_values, n_seed, n_boot, cap = [48, 64, 96, 128], 12, 40, 20000
    else:
        L_values, n_seed, n_boot, cap = [48, 64, 96], 10, 30, 12000

    RESULTS_DIR.mkdir(exist_ok=True)
    exp63 = load_exp63()
    rng = np.random.default_rng(args.seed_start)

    print("Experiment 68: collapse geometry as a universality metric (continuum v1)")
    print(f"L = {L_values}, n_seed = {n_seed}, n_boot = {n_boot}, KPZ lambda = {KPZ_LAM}")
    start = time.time()

    # --- Generate per-seed variance curves ---
    raw = {}
    for system in SYSTEMS:
        raw[system] = {}
        for L in L_values:
            T = T_of(system, L, cap)
            sb = args.seed_start + SYSTEM_SEED_OFFSETS[system] + L
            raw[system][L] = simulate_var_curves(exp63, system, L, T, n_seed, sb)
        print(f"  {system:>4}: " + ", ".join(
            f"L{L}(T{T_of(system,L,cap)},n{raw[system][L].shape[0]})" for L in L_values))

    # --- Per-system exponents (mean over all seeds) + proper 2D collapse ---
    system_exponents, system_collapse, mean_W = {}, {}, {}
    print("\n=== per-system exponents (seed-mean curves) ===")
    print(f"{'sys':>4} {'class':>7} | {'alpha':>6} {'beta':>6} {'z':>6} | "
          f"{'a_col':>6} {'z_col':>6} {'qual':>7} | theory a/b/z")
    for system in SYSTEMS:
        var_by_L = {L: raw[system][L].mean(axis=0) for L in L_values if raw[system][L].shape[0] > 0}
        a, b, z, Wc = exponents_from_varset(var_by_L)
        a_col, z_col, b_col, qual = fit_fv_collapse(Wc)
        system_exponents[system] = {"alpha": a, "beta": b, "z": z}
        system_collapse[system] = {"alpha_col": a_col, "z_col": z_col, "beta_col": b_col, "quality": qual}
        mean_W[system] = {L: Wc[L] for L in Wc}
        th = THEORY[system]
        ths = (f"{th['alpha']}/{th['beta']:.2f}/{th['z']}" if th["beta"] is not None else "   -   ")
        print(f"{system:>4} {CLASS_MAP[system]:>7} | {a:>6.3f} {b:>6.3f} {z:>6.3f} | "
              f"{a_col:>6.3f} {z_col:>6.3f} {qual:>7.4f} | {ths}")

    # --- Bootstrap exponent cloud (per-sample) ---
    sub = max(2, n_seed // 2)
    cloud_X, cloud_sys = [], []
    for system in SYSTEMS:
        for _ in range(n_boot):
            var_by_L = {}
            for L in L_values:
                arr = raw[system][L]
                if arr.shape[0] == 0:
                    continue
                idx = rng.choice(arr.shape[0], size=min(sub, arr.shape[0]), replace=False)
                var_by_L[L] = arr[idx].mean(axis=0)
            if len(var_by_L) < 2:
                continue
            a, b, z, _ = exponents_from_varset(var_by_L)
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(z):
                cloud_X.append([a, b, z]); cloud_sys.append(system)
    cloud_X = np.array(cloud_X)
    cloud_sys = np.array(cloud_sys)
    cloud_classes = np.array([CLASS_MAP[s] for s in cloud_sys])

    # --- Cluster the exponent cloud; compare to old ~0.5 feature ceiling ---
    Xs = StandardScaler().fit_transform(cloud_X)
    n_classes = len(set(cloud_classes))
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20).fit_predict(Xs)
    km_ari = adjusted_rand_score(cloud_classes, km)
    km_nmi = normalized_mutual_info_score(cloud_classes, km)
    hdb_ari = hdb_core = float("nan"); n_hdb = 0
    try:
        import hdbscan
        hl = hdbscan.HDBSCAN(min_cluster_size=max(5, len(cloud_classes) // 20), min_samples=3).fit_predict(Xs)
        n_hdb = len(set(hl) - {-1})
        hdb_ari = adjusted_rand_score(cloud_classes, hl)
        mask = hl != -1
        if n_hdb >= 2 and mask.any():
            hdb_core = adjusted_rand_score(cloud_classes[mask], hl[mask])
    except Exception as e:
        print("hdbscan unavailable:", e)

    print("\n=== exponent-cloud clustering (ARI vs intended classes) ===")
    print(f"  cloud: {len(cloud_classes)} samples, {n_classes} classes")
    print(f"  KMeans  ARI = {km_ari:.3f}  NMI = {km_nmi:.3f}")
    print(f"  HDBSCAN ARI(all) = {hdb_ari:.3f}  ARI(core) = {hdb_core:.3f}  ({n_hdb} clusters)")
    print(f"  reference: old feature-geometry HDBSCAN ARI ceiling ~ 0.50")

    # --- System-level universality distance (standardized exponents) ---
    sysv = np.array([[system_exponents[s]["alpha"], system_exponents[s]["beta"],
                      system_exponents[s]["z"]] for s in SYSTEMS])
    sv = StandardScaler().fit_transform(sysv)
    dmat = np.sqrt(((sv[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
    idx = {s: i for i, s in enumerate(SYSTEMS)}
    ew_kpz = float(dmat[idx["ew"], idx["kpz"]])
    print("\n=== system-level universality distance (standardized exponents) ===")
    print("       " + " ".join(f"{s:>6}" for s in SYSTEMS))
    for i, s in enumerate(SYSTEMS):
        print(f"  {s:>4} " + " ".join(f"{dmat[i, j]:6.2f}" for j in range(len(SYSTEMS))))
    print(f"\n  EW-vs-KPZ exponent distance = {ew_kpz:.2f} "
          f"(the degeneracy gradient features could not break)")

    # --- Save ---
    summary = {
        "config": {"L_values": L_values, "n_seed": n_seed, "n_boot": n_boot,
                   "kpz_lambda": KPZ_LAM, "T_per_L": {s: {L: T_of(s, L, cap) for L in L_values} for s in SYSTEMS}},
        "elapsed_seconds": time.time() - start,
        "system_exponents": system_exponents,
        "system_collapse": system_collapse,
        "theory": {k: {kk: vv for kk, vv in v.items()} for k, v in THEORY.items()},
        "cloud_clustering": {
            "n_samples": int(len(cloud_classes)), "n_classes": int(n_classes),
            "kmeans_ari": float(km_ari), "kmeans_nmi": float(km_nmi),
            "hdbscan_ari_all": float(hdb_ari), "hdbscan_ari_core": float(hdb_core),
            "hdbscan_clusters": int(n_hdb), "old_feature_ceiling_ari": 0.50,
        },
        "system_distance_matrix": {SYSTEMS[i]: {SYSTEMS[j]: float(dmat[i, j])
                                                for j in range(len(SYSTEMS))} for i in range(len(SYSTEMS))},
        "ew_vs_kpz_distance": ew_kpz,
        "scope_note": "continuum v1; BD/Eden (discrete KPZ) require sub-monolayer "
                      "recording + intrinsic-width subtraction (v2). KPZ beta is an "
                      "effective lambda-dependent crossover exponent at accessible L.",
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez(RESULTS_DIR / "mean_width_curves.npz",
             **{f"{s}_L{L}": mean_W[s][L] for s in SYSTEMS for L in mean_W[s]})

    print(f"\nDone in {summary['elapsed_seconds']:.1f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
