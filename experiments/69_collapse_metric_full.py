"""
Experiment 69: Collapse Geometry on All Six Systems + Matched Head-to-Head (Path 1, v2)
======================================================================================

Builds on Exp 68 (continuum collapse metric) by:

  (1) Adding the discrete KPZ-class models BD and Eden with sub-monolayer-resolved
      width curves and an intrinsic-width scan (the corrections that masked their
      scaling at one-monolayer-per-step recording).
  (2) Running the collapse-geometry clustering on all six systems, for both the
      4-class label (KPZ = KPZ+BD+Eden) and continuum-only subsets.
  (3) A MATCHED head-to-head: the old gradient+temporal feature pipeline (Exp 63)
      vs collapse geometry, on the identical six-system set.

Scientific question
-------------------
The main paper found two failures conflated into one ARI ceiling:
  * EW vs KPZ degeneracy  -> a REPRESENTATION problem (stationary gradient measure
    is lambda-blind).
  * discrete vs continuum (BD/Eden vs KPZ) -> a FINITE-SIZE / corrections-to-scaling
    problem (discrete models reach KPZ exponents only at very large L).

Collapse geometry should FIX the first (it reads beta, z, which separate EW from
KPZ) but NOT the second (the discrete effective exponents are corrections-corrupted
at accessible L). This experiment tests exactly that, and reports it honestly.

Calibration note (see docs/path1_collapse_metric_plan.md): at L <= 128, BD/Eden do
NOT show clean KPZ scaling. The archived L96 calibration gives BD beta ~0.20 with
t_sat_90pct ~22 monolayers, Eden beta ~0.19 with t_sat_90pct ~1753 monolayers, and
the intrinsic-width scan selected Wi^2 = 0 for both. These are reported as
effective exponents, not asymptotic.

Usage:
  python 69_collapse_metric_full.py --quick
  python 69_collapse_metric_full.py            # pilot
  python 69_collapse_metric_full.py --full
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from numba import jit
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_exp69_collapse_full"
EXP63_PATH = SCRIPT_DIR / "63_temporal_features.py"

SYSTEMS = ["ew", "kpz", "bd", "eden", "rd", "ks"]
CONTINUUM = ["ew", "kpz", "ks", "rd"]
CLASS_MAP = {"ew": "EW", "kpz": "KPZ", "bd": "KPZ", "eden": "KPZ", "rd": "trivial", "ks": "KS"}
SEED_OFF = {"ew": 0, "kpz": 1_000_000, "bd": 2_000_000, "eden": 3_000_000, "rd": 4_000_000, "ks": 5_000_000}
THEORY = {
    "ew": (0.5, 0.25, 2.0), "kpz": (0.5, 1 / 3, 1.5), "bd": (0.5, 1 / 3, 1.5),
    "eden": (0.5, 1 / 3, 1.5), "rd": (0.0, 0.5, None), "ks": (None, None, None),
}
KPZ_LAM = 4.0
RECS_PER_ML = 8  # sub-monolayer recording resolution for discrete models


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63", EXP63_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------
# Discrete simulators: sub-monolayer-resolved variance curves
# ----------------------------------------------------------------------------

@jit(nopython=True)
def bd_var_curve(L, n_rec, dep_per_rec, seed):
    np.random.seed(seed)
    h = np.zeros(L)
    var = np.zeros(n_rec)
    for r in range(n_rec):
        for _ in range(dep_per_rec):
            x = np.random.randint(0, L)
            h[x] = max(h[(x - 1) % L], h[x], h[(x + 1) % L]) + 1.0
        m = h - h.mean()
        var[r] = np.mean(m * m)
    return var


@jit(nopython=True)
def eden_var_curve(L, n_rec, dep_per_rec, seed):
    np.random.seed(seed)
    h = np.zeros(L)
    var = np.zeros(n_rec)
    for r in range(n_rec):
        for _ in range(dep_per_rec):
            x = np.random.randint(0, L)
            curv = h[(x - 1) % L] + h[(x + 1) % L] - 2 * h[x]
            if np.random.rand() < 0.5 + 0.3 * np.tanh(curv):
                h[x] += 1.0
        m = h - h.mean()
        var[r] = np.mean(m * m)
    return var


def discrete_var_curves(fn, L, T_mono, n_seed, seed_base):
    rpm = RECS_PER_ML
    dep = max(1, L // rpm)
    n_rec = int(T_mono * rpm)
    out = []
    for s in range(n_seed):
        v = fn(L, n_rec, dep, seed_base + s)
        if np.all(np.isfinite(v)):
            out.append(v)
    return (np.array(out) if out else np.zeros((0, n_rec))), rpm


def continuum_var_curves(exp63, system, L, T, n_seed, seed_base):
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
    return (np.array(out) if out else np.zeros((0, T))), 1.0  # rpm=1 (one record/step)


# ----------------------------------------------------------------------------
# Intrinsic-width-corrected exponent extraction (works for all systems)
# ----------------------------------------------------------------------------

def beta_wgated(var_curve, rpm, lo=0.2, hi=0.5):
    """Clean continuum growth exponent (Exp 68 method): W-gated, no intrinsic width."""
    W = np.sqrt(np.clip(var_curve, 0, None))
    T = len(W)
    wsat = float(np.median(W[int(0.85 * T):]))
    if wsat <= 0:
        return np.nan, 0.0
    lt = np.log(np.arange(1, T + 1, dtype=float) / rpm)
    lw = np.log(np.clip(W, 1e-12, None))
    m = (W > lo * wsat) & (W < hi * wsat)
    if m.sum() < 8:
        return np.nan, 0.0
    return float(np.polyfit(lt[m], lw[m], 1)[0]), 0.0  # beta, wi2=0


def beta_intrinsic(var_curve, rpm, lo=0.08, hi=0.6):
    """Discrete growth exponent: fit log(W^2 - Wi^2) ~ 2 beta log t, scanning Wi^2."""
    T = len(var_curve)
    t = np.arange(1, T + 1, dtype=float) / rpm
    w2sat = float(np.median(var_curve[int(0.85 * T):]))
    if w2sat <= 0:
        return np.nan, 0.0
    best = (np.nan, -np.inf, 0.0)
    for f in np.linspace(0.0, 0.7, 20):
        wi2 = f * w2sat
        ex = var_curve - wi2
        denom = w2sat - wi2
        m = (ex > lo * denom) & (ex < hi * denom) & (ex > 1e-9)
        if m.sum() < 8:
            continue
        lt, le = np.log(t[m]), np.log(ex[m])
        A = np.polyfit(lt, le, 1)
        pred = np.polyval(A, lt)
        ss = float(np.sum((le - pred) ** 2)); tot = float(np.sum((le - le.mean()) ** 2))
        r2 = 1 - ss / tot if tot > 0 else -np.inf
        if r2 > best[1]:
            best = (float(A[0] / 2), float(r2), float(wi2))
    return best[0], best[2]  # beta, wi2


def exponents(var_by_L, rpm, discrete=False):
    """Return alpha, beta, z. Continuum: W-gated. Discrete: intrinsic-width scan."""
    Ls = sorted(var_by_L)
    betas, wi2s, w2sat = [], [], []
    for L in Ls:
        if discrete:
            b, wi2 = beta_intrinsic(var_by_L[L], rpm)
        else:
            b, wi2 = beta_wgated(var_by_L[L], rpm)
        if np.isfinite(b):
            betas.append(b)
        wi2s.append(wi2)
        T = len(var_by_L[L])
        w2sat.append(float(np.median(var_by_L[L][int(0.85 * T):])))
    beta = float(np.mean(betas)) if betas else np.nan
    wi2_global = float(np.median(wi2s)) if (discrete and wi2s) else 0.0
    La = np.array(Ls, dtype=float)
    wsat_sc = np.sqrt(np.clip(np.array(w2sat) - wi2_global, 1e-9, None))
    if len(Ls) >= 2 and np.all(wsat_sc > 0):
        alpha = float(np.polyfit(np.log(La), np.log(wsat_sc), 1)[0])
    else:
        alpha = np.nan
    z = float(np.clip(alpha / beta, 0.0, 6.0)) if (np.isfinite(alpha) and np.isfinite(beta) and beta > 1e-6) else np.nan
    return alpha, beta, z


# ----------------------------------------------------------------------------
# Clustering helpers
# ----------------------------------------------------------------------------

def _confusion(classes, labels):
    classes = np.asarray(classes); labels = np.asarray(labels)
    out = {}
    for c in sorted(set(classes.tolist())):
        out[str(c)] = {str(int(cl)): int(np.sum((classes == c) & (labels == cl)))
                       for cl in sorted(set(labels.tolist()))}
    return out


def cluster_ari(X, classes):
    Xs = StandardScaler().fit_transform(X)
    nc = len(set(classes))
    km = KMeans(n_clusters=nc, random_state=42, n_init=20).fit_predict(Xs)
    out = {"n_samples": int(len(classes)),
           "kmeans_ari": float(adjusted_rand_score(classes, km)),
           "kmeans_nmi": float(normalized_mutual_info_score(classes, km)),
           "kmeans_confusion": _confusion(classes, km)}
    try:
        import hdbscan
        hl = hdbscan.HDBSCAN(min_cluster_size=max(5, len(classes) // 20), min_samples=3).fit_predict(Xs)
        ncl = len(set(hl) - {-1})
        mask = hl != -1
        out["hdbscan_clusters"] = int(ncl)
        out["hdbscan_noise"] = int(np.sum(hl == -1))
        out["hdbscan_ari_all"] = float(adjusted_rand_score(classes, hl))
        out["hdbscan_ari_core"] = float(adjusted_rand_score(np.asarray(classes)[mask], hl[mask])) if (ncl >= 2 and mask.any()) else 0.0
        out["hdbscan_confusion"] = _confusion(np.asarray(classes)[mask], hl[mask]) if mask.any() else {}
    except Exception:
        out["hdbscan_clusters"] = 0
        out["hdbscan_noise"] = 0
        out["hdbscan_ari_all"] = out["hdbscan_ari_core"] = float("nan")
        out["hdbscan_confusion"] = {}
    return out


def T_cont(system, L, cap):
    if system == "ew":
        return int(np.clip(0.5 * L * L, 1500, cap))
    if system == "kpz":
        return int(np.clip(0.6 * L * L, 1500, cap))
    if system == "ks":
        return min(4000, cap)
    return min(2500, cap)  # rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed-start", type=int, default=69_000)
    ap.add_argument("--out-tag", type=str, default="", help="suffix for results dir (archive pilots separately)")
    args = ap.parse_args()
    if args.quick:
        L_values, n_seed_c, n_seed_d, n_boot, cap = [48, 64], 5, 8, 12, 4000
    elif args.full:
        L_values, n_seed_c, n_seed_d, n_boot, cap = [48, 64, 96, 128], 12, 20, 40, 20000
    else:
        L_values, n_seed_c, n_seed_d, n_boot, cap = [48, 64, 96], 10, 16, 30, 12000

    results_dir = PROJECT_DIR / ("results_exp69_collapse_full" + (f"_{args.out_tag}" if args.out_tag else ""))
    results_dir.mkdir(exist_ok=True)
    exp63 = load_exp63()
    rng = np.random.default_rng(args.seed_start)
    print("Experiment 69: collapse geometry (all 6 systems) + matched head-to-head")
    print(f"L={L_values}, continuum seeds={n_seed_c}, discrete seeds={n_seed_d}, n_boot={n_boot}")
    start = time.time()

    # --- Generate variance curves ---
    raw, rpm = {}, {}
    for system in SYSTEMS:
        raw[system] = {}
        for L in L_values:
            sb = args.seed_start + SEED_OFF[system] + L
            if system in CONTINUUM:
                arr, r = continuum_var_curves(exp63, system, L, T_cont(system, L, cap), n_seed_c, sb)
            elif system == "bd":
                arr, r = discrete_var_curves(bd_var_curve, L, T_mono=max(300, 4 * L), n_seed=n_seed_d, seed_base=sb)
            else:  # eden
                arr, r = discrete_var_curves(eden_var_curve, L, T_mono=max(300, 4 * L), n_seed=n_seed_d, seed_base=sb)
            raw[system][L] = arr
            rpm[system] = r
        print(f"  {system:>4}: " + ", ".join(f"L{L}(n{raw[system][L].shape[0]})" for L in L_values))

    # --- Per-system exponents (validation / honest reporting) ---
    sys_exp = {}
    print("\n=== per-system effective exponents (with intrinsic-width scan) ===")
    print(f"{'sys':>4} {'class':>7} | {'alpha':>6} {'beta':>6} {'z':>6} | theory a/b/z")
    for system in SYSTEMS:
        var_by_L = {L: raw[system][L].mean(axis=0) for L in L_values if raw[system][L].shape[0] > 0}
        a, b, z = exponents(var_by_L, rpm[system], discrete=(system not in CONTINUUM))
        sys_exp[system] = {"alpha": a, "beta": b, "z": z}
        ta, tb, tz = THEORY[system]
        ths = (f"{ta}/{tb:.2f}/{tz}" if tb is not None else "   -   ")
        print(f"{system:>4} {CLASS_MAP[system]:>7} | {a:>6.3f} {b:>6.3f} {z:>6.3f} | {ths}")

    # --- Bootstrap exponent cloud over all systems ---
    sub_c, sub_d = max(2, n_seed_c // 2), max(2, n_seed_d // 2)
    Xc, sysc = [], []
    for system in SYSTEMS:
        sub = sub_c if system in CONTINUUM else sub_d
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
            a, b, z = exponents(var_by_L, rpm[system], discrete=(system not in CONTINUUM))
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(z):
                Xc.append([a, b, z]); sysc.append(system)
    Xc = np.array(Xc); sysc = np.array(sysc)
    classes6 = np.array([CLASS_MAP[s] for s in sysc])

    # collapse-geometry clustering: all-6 (4-class), continuum-only, 6-way system
    collapse_all = cluster_ari(Xc, classes6)
    cont_mask = np.isin(sysc, CONTINUUM)
    collapse_cont = cluster_ari(Xc[cont_mask], classes6[cont_mask])
    collapse_sys6 = cluster_ari(Xc, sysc)  # can collapse recover system identity?

    # --- Matched head-to-head: old feature pipeline on the SAME 6 systems ---
    Lf, Tf = (L_values[-1], 1000)
    n_feat = 25 if not args.quick else 10
    feat_X, feat_lab = [], []
    for system in SYSTEMS:
        sk = {"record_interval": 10} if system == "ks" else {}
        exp63.np.random.seed(args.seed_start + SEED_OFF[system])
        sp, tp, _ = exp63.generate_feature_dataset(n_feat, Lf, Tf, system,
                                                   seed_offset=args.seed_start + SEED_OFF[system],
                                                   late_frac=0.3, **sk)
        comb = np.hstack([sp, tp])
        feat_X.append(comb); feat_lab.extend([system] * comb.shape[0])
    feat_X = np.vstack(feat_X)
    feat_classes = np.array([CLASS_MAP[s] for s in feat_lab])
    feat_lab = np.array(feat_lab)
    feature_all = cluster_ari(feat_X, feat_classes)

    # KS-excluded variants: KS beta is not meaningfully fit, so its separation is
    # partly an artifact of failed exponent extraction. Report with KS removed.
    noks_c = sysc != "ks"
    collapse_noks = cluster_ari(Xc[noks_c], classes6[noks_c])
    noks_f = feat_lab != "ks"
    feature_noks = cluster_ari(feat_X[noks_f], feat_classes[noks_f])

    # --- Report (NOTE: head-to-head is information-ASYMMETRIC; see plan doc) ---
    print("\n=== HEAD-TO-HEAD (all 6 systems, 4-class) -- NOT information-matched ===")
    print("    (collapse uses multi-L curves; feature uses single L. exp70 = matched baseline)")
    print(f"  feature geometry (10D Exp63, single-L):  KMeans={feature_all['kmeans_ari']:.3f}  "
          f"HDBSCAN core={feature_all['hdbscan_ari_core']:.3f} "
          f"(clusters={feature_all['hdbscan_clusters']}, noise={feature_all.get('hdbscan_noise',0)})")
    print(f"  exponent geometry (alpha,beta,z, multi-L): KMeans={collapse_all['kmeans_ari']:.3f}  "
          f"HDBSCAN core={collapse_all['hdbscan_ari_core']:.3f} "
          f"(clusters={collapse_all['hdbscan_clusters']}, noise={collapse_all.get('hdbscan_noise',0)})")
    print(f"  [KS-excluded] feature KMeans={feature_noks['kmeans_ari']:.3f}  "
          f"exponent KMeans={collapse_noks['kmeans_ari']:.3f}")
    print("\n=== exponent geometry, CONTINUUM-only (EW/KPZ/KS/RD, 4-class) ===")
    print(f"  KMeans ARI={collapse_cont['kmeans_ari']:.3f}  HDBSCAN core={collapse_cont['hdbscan_ari_core']:.3f}")
    print("\n=== exponent geometry, 6-way SYSTEM identity ===")
    print(f"  KMeans ARI={collapse_sys6['kmeans_ari']:.3f}  NMI={collapse_sys6['kmeans_nmi']:.3f}")

    # system-level distance matrix
    sv = StandardScaler().fit_transform(np.array([[sys_exp[s]["alpha"], sys_exp[s]["beta"], sys_exp[s]["z"]] for s in SYSTEMS]))
    dmat = np.sqrt(((sv[:, None, :] - sv[None, :, :]) ** 2).sum(-1))
    idx = {s: i for i, s in enumerate(SYSTEMS)}
    print("\n=== system distance matrix (standardized exponents) ===")
    print("       " + " ".join(f"{s:>6}" for s in SYSTEMS))
    for i, s in enumerate(SYSTEMS):
        print(f"  {s:>4} " + " ".join(f"{dmat[i, j]:6.2f}" for j in range(len(SYSTEMS))))
    print(f"\n  EW-vs-KPZ      = {dmat[idx['ew'],idx['kpz']]:.2f}  (the gradient-degenerate pair)")
    print(f"  KPZ-vs-BD      = {dmat[idx['kpz'],idx['bd']]:.2f}")
    print(f"  KPZ-vs-Eden    = {dmat[idx['kpz'],idx['eden']]:.2f}  "
          f"(BD/Eden land close to KPZ, but on EFFECTIVE exponents)")

    summary = {
        "config": {"L_values": L_values, "n_seed_continuum": n_seed_c, "n_seed_discrete": n_seed_d,
                   "n_boot": n_boot, "kpz_lambda": KPZ_LAM, "recs_per_ml": RECS_PER_ML},
        "elapsed_seconds": time.time() - start,
        "system_exponents": sys_exp,
        "theory": {k: {"alpha": v[0], "beta": v[1], "z": v[2]} for k, v in THEORY.items()},
        "head_to_head_all6_4class": {"feature_geometry": feature_all, "exponent_geometry": collapse_all},
        "head_to_head_KS_excluded": {"feature_geometry": feature_noks, "exponent_geometry": collapse_noks},
        "exponent_continuum_4class": collapse_cont,
        "exponent_system6way": collapse_sys6,
        "system_distance_matrix": {SYSTEMS[i]: {SYSTEMS[j]: float(dmat[i, j]) for j in range(len(SYSTEMS))}
                                   for i in range(len(SYSTEMS))},
        "caveats": [
            "Head-to-head is information-ASYMMETRIC: exponent geometry uses multi-L "
            "curves (L=48..128) + bootstrap-of-collapse; feature side is single-L. "
            "A matched multi-L feature baseline (exp70) is required.",
            "Clustering is on exponent VECTORS (alpha,beta,z), not a collapse-residual "
            "metric. 'Exponent geometry', not a true collapse metric.",
            "Exponents are EFFECTIVE not asymptotic (EW z~2.6, KPZ z~1.1, BD/Eden "
            "alpha~0.25-0.35); KPZ beta is lambda-dependent (crossover).",
            "KS beta is not meaningfully fit (~5.6); its separation is partly an "
            "artifact -- see KS-excluded variant.",
            "Headline ARIs are single-run (no confidence intervals).",
        ],
        "interpretation": "On the identical 6-system set, exponent geometry aligns far "
                          "better with the class labels than single-L feature vectors. In "
                          "exponent space BD/Eden land much closer to continuum KPZ "
                          "(0.28-0.38) than to EW/RD/KS (1.6-3.3), so the KPZ class is "
                          "largely -- but not cleanly -- recovered (HDBSCAN over-segments). "
                          "This is not yet an information-matched test nor asymptotic "
                          "universality recovery.",
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['elapsed_seconds']:.1f}s -> {results_dir}")


if __name__ == "__main__":
    main()
