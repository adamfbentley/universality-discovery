"""Experiment 74: does a multi-L scaling-collapse representation recover BD's KPZ class?

Background
----------
Exp 72/73 showed that single-L local feature geometry cannot place ballistic
deposition (BD) with its KPZ universality class: BD shows anomalous local
roughening (alpha_local ~ 0.23) and the resulting clustering split survives
amplitude normalization, so the obstruction lives in the multi-scale (scaling)
structure, not a single amplitude.

The constructive question: if we instead build the representation from the
*scaling structure itself* -- a Family-Vicsek collapse W(L,t) = L^alpha f(t/L^z)
fitted across an L-ladder -- does BD finally group with KPZ/Eden? This is the
"true collapse metric" the physics roadmap repeatedly flagged as missing.

Two representations are tested, both seed-resampled for error bars:
  (1) fitted collapse exponents (alpha, z, beta=alpha/z) per system.
  (2) the amplitude-normalized collapse master-curve *shape*.

Decisive metric (reused from exp73): BD's distance to its KPZ-class partners
{KPZ, Eden} divided by the within-{KPZ,Eden,...} spread. >>1 = BD is still an
outlier (scaling representation also fails at accessible L); ~1 = BD merged
(the scaling representation factors through the quotient). Single-L features
gave a BD-outlier ratio ~9 (exp73), so that is the number to beat.

Sanity anchor: the collapse fit should recover the known dynamic exponents
EW z~2 (beta~1/4) and KPZ z~1.5 (beta~1/3); if it does, BD's result is trusted.

Outputs (read-only w.r.t. existing results):
- results_exp74_collapse_representation/summary.json
- results_exp74_collapse_representation/exponents_by_system.csv
- results_exp74_collapse_representation/collapse_representation.png  (if mpl)

Usage:
  python experiments/74_collapse_representation.py            # full
  python experiments/74_collapse_representation.py --quick    # smoke
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXP63_PATH = PROJECT_DIR / "experiments" / "63_temporal_features.py"
RESULTS_DIR = PROJECT_DIR / "results_exp74_collapse_representation"

# 1+1D references: EW (alpha=1/2,z=2,beta=1/4); KPZ class (alpha=1/2,z=3/2,beta=1/3)
THEORY = {"ew": (0.5, 2.0, 0.25), "kpz": (0.5, 1.5, 1 / 3),
          "bd": (0.5, 1.5, 1 / 3), "eden": (0.5, 1.5, 1 / 3),
          "rd": (None, None, 0.5), "ks": (None, None, None)}
KPZ_PARTNERS = ["kpz", "eden"]          # BD's same-class partners to merge with
N_TIME = 120                            # log-spaced samples of each W(t) curve


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63_temporal_features", EXP63_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate(exp63, system, L, T, seed):
    sims = {
        "ew": lambda s: exp63.simulate_ew(L=L, T=T, seed=s),
        "kpz": lambda s: exp63.simulate_kpz(L=L, T=T, seed=s),
        "bd": lambda s: exp63.simulate_bd(L=L, T=T, seed=s),
        "eden": lambda s: exp63.simulate_eden(L=L, T=T, seed=s),
        "rd": lambda s: exp63.simulate_rd(L=L, T=T, seed=s),
        "ks": lambda s: exp63.simulate_ks(L=L, T=T, seed=s),
    }
    fn = sims[system]
    for attempt in range(6):
        traj = fn(seed + attempt * 100003)
        if traj is not None and np.all(np.isfinite(traj)):
            return traj
    return None


def width_curve(traj):
    """W(t) = std_x[h] subsampled on a log-spaced time grid -> (t, W)."""
    T = traj.shape[0]
    w = np.std(traj, axis=1)
    idx = np.unique(np.geomspace(2, T, N_TIME).astype(int) - 1)
    idx = idx[idx >= 1]
    t = idx.astype(float) + 1.0
    return t, w[idx]


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def measure_alpha_beta(curves):
    """Robust, decoupled FSS exponents from one seed's L-ladder.

    alpha: saturated width W_sat(L) ~ L^alpha   (clean -- separate plateaus).
    beta : growth-regime W(t) ~ t^beta at the largest L (longest growth phase).
    z    : alpha / beta.
    Avoids the alpha-z degeneracy of a joint collapse fit, which is
    underdetermined at accessible L (it cannot even recover EW z=2 / KPZ z=1.5).
    """
    Ls = sorted(curves)
    W_sat = []
    for L in Ls:
        t, w = curves[L]
        W_sat.append(np.median(w[int(0.8 * len(w)):]))   # late plateau
    alpha = loglog_slope(np.array(Ls, float), np.array(W_sat))

    # beta from the largest L's growth window [2%, 20%] of its time axis
    t, w = curves[Ls[-1]]
    lo, hi = int(0.02 * len(t)), int(0.20 * len(t))
    hi = max(hi, lo + 5)
    beta = loglog_slope(t[lo:hi], w[lo:hi])
    z = alpha / beta if beta and np.isfinite(beta) and beta != 0 else float("nan")
    return alpha, beta, z, float(np.mean(W_sat))


def master_curve_shape(curves, alpha, z, x_grid):
    """Amplitude-normalized collapsed master curve on a common log-x grid."""
    xs, ys = [], []
    for L, t, w in curves:
        xs.append(t / (L ** z))
        ys.append(w / (L ** alpha))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    pos = (x > 0) & (y > 0)
    x, y = np.log(x[pos]), np.log(y[pos])
    order = np.argsort(x)
    x, y = x[order], y[order]
    yg = np.interp(x_grid, x, y, left=np.nan, right=np.nan)
    # normalize plateau (late-x median) to 0 in log space (removes amplitude)
    plateau = np.nanmedian(yg[-len(yg) // 4:])
    return yg - plateau


def outlier_ratio(points, labels, target, partners):
    """distance(target, mean(partners)) / mean pairwise spread among partners-group."""
    labels = np.array(labels)
    cen = {s: np.nanmean(points[labels == s], axis=0) for s in set(labels)}
    grp = [s for s in partners if s in cen]
    if target not in cen or len(grp) < 1:
        return float("nan"), float("nan")
    grp_cen = np.nanmean([cen[s] for s in grp], axis=0)
    d_target = float(np.sqrt(np.nansum((cen[target] - grp_cen) ** 2)))
    # spread = mean pairwise distance among partner-group members + target's class
    members = grp
    pair = [np.sqrt(np.nansum((cen[a] - cen[b]) ** 2))
            for i, a in enumerate(members) for b in members[i + 1:]]
    spread = float(np.mean(pair)) if pair else float("nan")
    ratio = d_target / spread if spread and spread > 0 else float("nan")
    return d_target, ratio


def main():
    parser = argparse.ArgumentParser(description="Exp 74: collapse representation")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp63 = load_exp63()
    RESULTS_DIR.mkdir(exist_ok=True)

    systems = ["ew", "kpz", "bd", "eden", "rd", "ks"]
    if args.quick:
        L_ladder = [24, 32, 48, 64]
        n_seeds = 4
        ks_cap = 1500
    else:
        L_ladder = [32, 48, 64, 96, 128]
        n_seeds = 8
        ks_cap = 3000
    T_schedule = lambda L: int(30 * L ** 1.5)

    print("Experiment 74: multi-L scaling-collapse representation")
    print(f"  L-ladder={L_ladder} seeds={n_seeds}")

    # ---- simulate: per (system, seed) collapse curves across the L-ladder ----
    # store W(L,t) per seed so we can bootstrap and seed-average
    data = {s: [] for s in systems}   # data[s] = list over seeds of {L: (t,W)}
    for s in systems:
        Tcap = ks_cap if s == "ks" else None
        for seed in range(n_seeds):
            curves = {}
            ok = True
            for L in L_ladder:
                T = T_schedule(L)
                if Tcap:
                    T = min(T, Tcap)
                traj = simulate(exp63, s, L, T, seed=2000 + 17 * seed)
                if traj is None:
                    ok = False
                    break
                curves[L] = width_curve(traj)
            if ok:
                data[s].append(curves)
        print(f"  {s:5s}: {len(data[s])} usable seeds")

    # ---- per-system decoupled FSS exponents (per seed) + seed-pooled shape ----
    x_grid = np.linspace(-6, 4, 60)   # common log-x grid for shape comparison
    exponents = {}     # system -> summary dict
    shapes = {}        # system -> master-curve shape vector (seed-pooled)
    per_seed_exp = {}  # system -> list of (alpha, beta) for the merge metric
    for s in systems:
        seeds = data[s]
        if not seeds:
            continue
        ab_list = [measure_alpha_beta(cv)[:2] for cv in seeds]  # (alpha, beta)
        per_seed_exp[s] = ab_list
        arr = np.array(ab_list)
        a_mean, b_mean = float(np.nanmean(arr[:, 0])), float(np.nanmean(arr[:, 1]))
        z_mean = a_mean / b_mean if b_mean else float("nan")
        pooled = [(L, t, w) for cv in seeds for L, (t, w) in cv.items()]
        shapes[s] = master_curve_shape(pooled, a_mean, z_mean, x_grid)
        exponents[s] = {
            "alpha": a_mean, "beta": b_mean, "z": z_mean,
            "alpha_seed_std": float(np.nanstd(arr[:, 0])),
            "beta_seed_std": float(np.nanstd(arr[:, 1])),
            "theory": THEORY[s],
        }
        th = THEORY[s]
        print(f"  [{s:5s}] alpha={a_mean:.3f} beta={b_mean:.3f} z={z_mean:.3f}  "
              f"(theory z={th[1]}, beta={th[2]})")

    # ---- sanity anchor: did the estimator recover EW/KPZ dynamics? ----
    anchor_ok = (abs(exponents.get("ew", {}).get("beta", 0) - 0.25) < 0.08 and
                 abs(exponents.get("kpz", {}).get("beta", 0) - 1 / 3) < 0.08)
    print(f"\n  sanity anchor (EW beta~0.25, KPZ beta~0.33 recovered): "
          f"{'PASS' if anchor_ok else 'WEAK -- exponents unreliable at this size'}")

    # ---- representation 1: robust exponent space (alpha, beta) ----
    rep_systems = [s for s in systems if s in exponents]
    pts1, lab1 = [], []
    for s in rep_systems:
        for (ab, bb) in per_seed_exp[s]:
            pts1.append([ab, bb]); lab1.append(s)
    pts1 = np.array(pts1)
    d1, ratio1 = outlier_ratio(pts1, lab1, "bd", KPZ_PARTNERS)

    # ---- representation 2: collapse master-curve shape ----
    shp_systems = [s for s in rep_systems if s in shapes]
    pts2 = np.array([shapes[s] for s in shp_systems])
    # replace nans with column means for distance robustness
    col_mean = np.nanmean(pts2, axis=0)
    inds = np.where(np.isnan(pts2))
    pts2[inds] = np.take(col_mean, inds[1])
    d2, ratio2 = outlier_ratio(pts2, shp_systems, "bd", KPZ_PARTNERS)

    # ---- BD nearest-neighbour class in each representation ----
    def nearest(points, labels, target):
        labels = np.array(labels)
        cen = {s: np.nanmean(points[labels == s], axis=0) for s in set(labels)}
        if target not in cen:
            return None
        others = {s: np.sqrt(np.nansum((cen[target] - cen[s]) ** 2))
                  for s in cen if s != target}
        return min(others, key=others.get), {k: round(v, 3) for k, v in sorted(others.items(), key=lambda kv: kv[1])}

    nn1 = nearest(pts1, lab1, "bd")
    nn2 = nearest(pts2, shp_systems, "bd")

    print("\n" + "=" * 72)
    print("Does the scaling representation pull BD into the KPZ class?")
    print(f"  single-L feature baseline (exp73): BD-outlier ratio ~ 9.2  (fails)")
    print(f"  rep1 exponents (alpha,beta):  BD-outlier ratio = {ratio1:.2f}; "
          f"BD nearest system = {nn1[0] if nn1 else None}")
    print(f"  rep2 collapse shape:          BD-outlier ratio = {ratio2:.2f}; "
          f"BD nearest system = {nn2[0] if nn2 else None}")
    print(f"  (ratio ~1 => BD merged with KPZ class; >>1 => still an outlier)")

    out = {
        "config": {"L_ladder": L_ladder, "n_seeds": n_seeds, "ks_cap": ks_cap,
                   "T_schedule": {str(L): T_schedule(L) for L in L_ladder},
                   "quick": args.quick},
        "exponents": exponents,
        "sanity_anchor_passed": bool(anchor_ok),
        "single_L_baseline_bd_outlier_ratio": 9.2,
        "rep1_exponents_alpha_beta": {"bd_outlier_ratio": ratio1,
                                      "bd_nearest_system": nn1[0] if nn1 else None,
                                      "bd_distances": nn1[1] if nn1 else None},
        "rep2_collapse_shape": {"bd_outlier_ratio": ratio2,
                                "bd_nearest_system": nn2[0] if nn2 else None,
                                "bd_distances": nn2[1] if nn2 else None},
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RESULTS_DIR / "exponents_by_system.csv", "w") as f:
        f.write("system,alpha,beta,z,alpha_seed_std,beta_seed_std,"
                "theory_z,theory_beta\n")
        for s in rep_systems:
            e = exponents[s]
            tz = e["theory"][1] if e["theory"][1] is not None else ""
            tb = e["theory"][2] if e["theory"][2] is not None else ""
            f.write(f"{s},{e['alpha']:.4f},{e['beta']:.4f},{e['z']:.4f},"
                    f"{e['alpha_seed_std']:.4f},{e['beta_seed_std']:.4f},"
                    f"{tz},{tb}\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"ew": "#1f77b4", "kpz": "#ff7f0e", "bd": "#d62728",
                  "eden": "#9467bd", "rd": "#7f7f7f", "ks": "#2ca02c"}
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for s in rep_systems:
            a = [p[0] for p, l in zip(pts1, lab1) if l == s]
            b = [p[1] for p, l in zip(pts1, lab1) if l == s]
            ax[0].scatter(b, a, color=colors[s], label=s, s=40,
                          edgecolor="k" if s == "bd" else "none")
        ax[0].axvline(1 / 3, ls="--", c="orange", lw=0.8)   # KPZ beta
        ax[0].axvline(0.25, ls="--", c="blue", lw=0.8)      # EW beta
        ax[0].set_xlabel("beta (growth fit)"); ax[0].set_ylabel("alpha (W_sat vs L)")
        ax[0].set_title("Rep 1: decoupled FSS exponents")
        ax[0].legend(fontsize=8)
        for s in shp_systems:
            ax[1].plot(x_grid, shapes[s], color=colors[s], label=s,
                       lw=2.5 if s == "bd" else 1.3)
        ax[1].set_xlabel("log(t/L^z)"); ax[1].set_ylabel("log(W/L^a) - plateau")
        ax[1].set_title("Rep 2: collapse master-curve shape")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "collapse_representation.png", dpi=130)
        plt.close(fig)
        print(f"  figure -> {RESULTS_DIR/'collapse_representation.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
