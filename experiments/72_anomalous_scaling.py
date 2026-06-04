"""Experiment 72: local vs global roughness exponents (anomalous-scaling diagnostic).

Motivation
----------
The repository's central negative result is that finite-size feature geometry
does not recover the KPZ universality quotient: the discrete KPZ-class members
(ballistic deposition, Eden) repeatedly split off from the continuum members
(EW, KPZ) in clustering (exp62-64), and that split is the dominant cause of the
ARI ~ 0.5 ceiling.

The features the project clusters on are *local* slope statistics
(`grad_var = Var[dh/dx]`, etc.). Universality, however, lives in the *global*
roughness exponent. Standard (Family-Vicsek) kinetic roughening has
alpha_local == alpha_global; under *intrinsic anomalous roughening* they differ,
and local observables then track a non-universal, lattice-scale quantity that
need not respect the universality class.

This script tests, directly and without any cloud compute, whether the
discrete KPZ-class systems exhibit a local-vs-global exponent mismatch that the
continuum systems do not -- i.e. whether anomalous local roughening is the
mechanism behind the clustering split.

What is measured (per system)
-----------------------------
- alpha_global: from the saturated global width W(L) ~ L^alpha across an L-ladder.
- alpha_local : from the height-difference correlation
                G(r) = < [h(x+r) - h(x)]^2 >  ~  r^(2 alpha_local), small-r slope.
- delta_alpha = alpha_global - alpha_local  (anomaly indicator; ~0 for Family-Vicsek).
- intrinsic_width = sqrt(G(2)): the lattice-scale local roughness amplitude that
  the gradient features (Var[dh/dx] ~ G(2)/4) actually respond to.
- late_beta: late-time d log W / d log t, a saturation check (~0 = saturated).

Reference values (1+1D): EW, KPZ, BD, Eden all have alpha = 1/2 asymptotically.
RD is uncorrelated (alpha_local = 0, never saturates -> alpha_global ~ 0).
KS is a crossover/chaotic system and is reported for context only.

Outputs (read-only w.r.t. existing results):
- results_exp72_anomalous_scaling/summary.json
- results_exp72_anomalous_scaling/width_vs_L.csv
- results_exp72_anomalous_scaling/Gr_curves.csv
- results_exp72_anomalous_scaling/anomalous_scaling.png  (if matplotlib available)

Usage:
  python experiments/72_anomalous_scaling.py            # full diagnostic
  python experiments/72_anomalous_scaling.py --quick    # fast smoke run
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXP63_PATH = PROJECT_DIR / "experiments" / "63_temporal_features.py"
RESULTS_DIR = PROJECT_DIR / "results_exp72_anomalous_scaling"

# Asymptotic roughness exponent for reference. KPZ-class = 1/2 in 1+1D.
ALPHA_THEORY = {"ew": 0.5, "kpz": 0.5, "bd": 0.5, "eden": 0.5,
                "rd": None, "ks": None}
KPZ_CLASS = {"ew", "kpz", "bd", "eden"}          # share alpha = 1/2
DISCRETE_KPZ = {"bd", "eden"}                    # the clustering-split systems
CONTINUUM_KPZ = {"ew", "kpz"}


def load_exp63():
    spec = importlib.util.spec_from_file_location("exp63_temporal_features", EXP63_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate(exp63, system, L, T, seed):
    """Run one simulation, retrying blowups (KPZ/KS return None) with new seeds."""
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


def height_difference_correlation(frames):
    """G(r) = < [h(x+r) - h(x)]^2 > averaged over x (periodic), frames.

    frames: (n_frames, L). Returns r (1..L//2) and G(r).
    """
    n_frames, L = frames.shape
    r_vals = np.arange(1, L // 2 + 1)
    G = np.zeros(len(r_vals))
    for i, r in enumerate(r_vals):
        diff = np.roll(frames, -r, axis=1) - frames      # (n_frames, L)
        G[i] = np.mean(diff ** 2)
    return r_vals, G


def fit_loglog(x, y):
    """Least-squares slope of log y vs log x. Returns (slope, intercept, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def late_time_beta(width_traj):
    """d log W / d log t over the late half -- ~0 if saturated."""
    T = len(width_traj)
    t0 = max(2, T // 2)
    t = np.arange(t0, T, dtype=float)
    w = width_traj[t0:]
    slope, _, _ = fit_loglog(t, w)
    return slope


def analyze_system(exp63, system, L_ladder, T_schedule, n_seeds, ks_cap, verbose=True):
    """Return per-system measurements across the L-ladder."""
    per_L = []
    for L in L_ladder:
        T = T_schedule(L)
        if system == "ks":
            T = min(T, ks_cap)
        widths_sat = []          # saturated global width per seed
        G_accum = None           # G(r) averaged over seeds
        late_betas = []
        n_ok = 0
        for s in range(n_seeds):
            traj = simulate(exp63, system, L, T, seed=1000 + 13 * s)
            if traj is None:
                continue
            n_ok += 1
            # global width trajectory W(t) = std_x[h]
            w_t = np.std(traj, axis=1)
            late_betas.append(late_time_beta(w_t))
            # late-time (saturated) window: last 25% of frames, subsampled
            t_lo = int(0.75 * T)
            late = traj[t_lo:]
            if len(late) > 80:
                idx = np.linspace(0, len(late) - 1, 80).astype(int)
                late = late[idx]
            widths_sat.append(float(np.mean(np.std(late, axis=1))))
            r_vals, G = height_difference_correlation(late)
            G_accum = G if G_accum is None else G_accum + G
        if n_ok == 0:
            per_L.append({"L": L, "T": T, "n_ok": 0})
            continue
        G_mean = G_accum / n_ok
        # alpha_local: small-r slope of G(r) ~ r^(2 alpha_local).
        # Fit window: r in [2, L/8] (above lattice cutoff, well below xi~L).
        r_hi = max(4, L // 8)
        fit_mask = (r_vals >= 2) & (r_vals <= r_hi)
        slope_G, _, r2_G = fit_loglog(r_vals[fit_mask], G_mean[fit_mask])
        alpha_local = slope_G / 2.0
        # intrinsic / lattice-scale roughness amplitude (gradient-feature scale)
        g2 = float(G_mean[1]) if len(G_mean) > 1 else float(G_mean[0])
        per_L.append({
            "L": L, "T": T, "n_ok": n_ok,
            "W_sat": float(np.mean(widths_sat)),
            "W_sat_std": float(np.std(widths_sat)),
            "alpha_local": alpha_local,
            "alpha_local_r2": r2_G,
            "intrinsic_width": float(np.sqrt(max(g2, 0.0))),
            "late_beta": float(np.mean(late_betas)),
            "r_fit_hi": int(r_hi),
            "Gr": G_mean.tolist(),
            "r_vals": r_vals.tolist(),
        })
        if verbose:
            print(f"  L={L:4d} T={T:6d} n={n_ok:2d}  "
                  f"W_sat={np.mean(widths_sat):8.3f}  "
                  f"alpha_local={alpha_local:+.3f} (R2={r2_G:.3f})  "
                  f"late_beta={np.mean(late_betas):+.3f}")

    # alpha_global from W_sat(L) ~ L^alpha across the ladder
    Ls = [d["L"] for d in per_L if d.get("n_ok", 0) > 0 and "W_sat" in d]
    Ws = [d["W_sat"] for d in per_L if d.get("n_ok", 0) > 0 and "W_sat" in d]
    alpha_global, _, r2_global = fit_loglog(Ls, Ws)

    # pooled alpha_local (mean over L, weighted by fit quality / n)
    al = [(d["alpha_local"], d["alpha_local_r2"]) for d in per_L
          if "alpha_local" in d and np.isfinite(d["alpha_local"])]
    if al:
        alpha_local_pooled = float(np.mean([a for a, _ in al]))
    else:
        alpha_local_pooled = float("nan")

    return {
        "system": system,
        "alpha_global": alpha_global,
        "alpha_global_r2": r2_global,
        "alpha_local_pooled": alpha_local_pooled,
        "delta_alpha": (alpha_global - alpha_local_pooled
                        if np.isfinite(alpha_global) and np.isfinite(alpha_local_pooled)
                        else float("nan")),
        "alpha_theory": ALPHA_THEORY[system],
        "per_L": per_L,
    }


def make_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"  (figure skipped: {e})")
        return
    order = ["ew", "kpz", "bd", "eden", "rd", "ks"]
    colors = {"ew": "#1f77b4", "kpz": "#ff7f0e", "bd": "#d62728",
              "eden": "#9467bd", "rd": "#7f7f7f", "ks": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: G(r) at largest L
    ax = axes[0]
    for sysn in order:
        rec = results.get(sysn)
        if not rec:
            continue
        last = [d for d in rec["per_L"] if "Gr" in d]
        if not last:
            continue
        d = last[-1]
        ax.loglog(d["r_vals"], d["Gr"], "o-", ms=3, color=colors[sysn],
                  label=f"{sysn} (a_loc={rec['alpha_local_pooled']:.2f})")
    ax.set_xlabel("r"); ax.set_ylabel("G(r) = <[h(x+r)-h(x)]^2>")
    ax.set_title("Height-difference correlation (local)")
    ax.legend(fontsize=7)

    # Panel 2: W_sat vs L
    ax = axes[1]
    for sysn in order:
        rec = results.get(sysn)
        if not rec:
            continue
        Ls = [d["L"] for d in rec["per_L"] if "W_sat" in d]
        Ws = [d["W_sat"] for d in rec["per_L"] if "W_sat" in d]
        if len(Ls) >= 2:
            ax.loglog(Ls, Ws, "s-", color=colors[sysn],
                      label=f"{sysn} (a_glob={rec['alpha_global']:.2f})")
    ax.set_xlabel("L"); ax.set_ylabel("W_sat(L)")
    ax.set_title("Saturated global width (global)")
    ax.legend(fontsize=7)

    # Panel 3: delta_alpha bar chart, discrete KPZ highlighted
    ax = axes[2]
    names, deltas, bar_colors = [], [], []
    for sysn in order:
        rec = results.get(sysn)
        if not rec or not np.isfinite(rec["delta_alpha"]):
            continue
        names.append(sysn)
        deltas.append(rec["delta_alpha"])
        bar_colors.append("#d62728" if sysn in DISCRETE_KPZ else "#bbbbbb")
    ax.bar(names, deltas, color=bar_colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("delta_alpha = alpha_global - alpha_local")
    ax.set_title("Anomaly gap (red = discrete KPZ class)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  figure -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Exp 72: local vs global roughness exponents")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--systems", nargs="*", default=None,
                        help="subset of systems to run")
    args = parser.parse_args()

    exp63 = load_exp63()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.quick:
        L_ladder = [16, 32, 64]
        n_seeds = 4
        # T sized (roughly) to saturate KPZ (the binding constraint): ~20 * L^1.5
        T_schedule = lambda L: int(30 * L ** 1.5)
        ks_cap = 1500
    else:
        L_ladder = [32, 64, 128]
        n_seeds = 10
        T_schedule = lambda L: int(30 * L ** 1.5)   # 32->5400, 64->15360, 128->43400
        ks_cap = 3000

    systems = args.systems or ["ew", "kpz", "bd", "eden", "rd", "ks"]
    print(f"Experiment 72: local vs global roughness exponents")
    print(f"  L-ladder={L_ladder}  seeds={n_seeds}  systems={systems}")
    print(f"  T(L)={[T_schedule(L) for L in L_ladder]} (KS capped at {ks_cap})")

    results = {}
    for sysn in systems:
        print(f"\n[{sysn}]")
        results[sysn] = analyze_system(
            exp63, sysn, L_ladder, T_schedule, n_seeds, ks_cap)

    # ---- console summary ----
    print("\n" + "=" * 72)
    print(f"{'system':6s} {'a_global':>9s} {'a_local':>9s} {'delta':>8s} "
          f"{'a_theory':>9s} {'glob_R2':>8s}")
    print("-" * 72)
    for sysn in systems:
        r = results[sysn]
        th = r["alpha_theory"]
        th_s = f"{th:.2f}" if th is not None else "  -- "
        print(f"{sysn:6s} {r['alpha_global']:>9.3f} {r['alpha_local_pooled']:>9.3f} "
              f"{r['delta_alpha']:>8.3f} {th_s:>9s} {r['alpha_global_r2']:>8.3f}")

    # ---- mechanism test ----
    disc = [results[s]["delta_alpha"] for s in DISCRETE_KPZ
            if s in results and np.isfinite(results[s]["delta_alpha"])]
    cont = [results[s]["delta_alpha"] for s in CONTINUUM_KPZ
            if s in results and np.isfinite(results[s]["delta_alpha"])]
    verdict = {}
    if disc and cont:
        verdict = {
            "discrete_kpz_mean_delta_alpha": float(np.mean(disc)),
            "continuum_kpz_mean_delta_alpha": float(np.mean(cont)),
            "separation": float(np.mean(disc) - np.mean(cont)),
        }
        print("\nMechanism test (anomalous local roughening):")
        print(f"  discrete KPZ (bd, eden) mean delta_alpha  = {verdict['discrete_kpz_mean_delta_alpha']:+.3f}")
        print(f"  continuum KPZ (ew, kpz) mean delta_alpha  = {verdict['continuum_kpz_mean_delta_alpha']:+.3f}")
        print(f"  separation                                = {verdict['separation']:+.3f}")

    # ---- write artifacts ----
    summary = {
        "config": {"L_ladder": L_ladder, "n_seeds": n_seeds,
                   "T_schedule": {str(L): T_schedule(L) for L in L_ladder},
                   "ks_cap": ks_cap, "quick": args.quick},
        "definitions": {
            "alpha_global": "W_sat(L) ~ L^alpha_global (saturated global width vs L)",
            "alpha_local": "G(r)=<[h(x+r)-h(x)]^2> ~ r^(2 alpha_local), small-r slope",
            "delta_alpha": "alpha_global - alpha_local (>0 suggests anomalous local roughening)",
            "intrinsic_width": "sqrt(G(2)); lattice-scale amplitude probed by Var[dh/dx]",
        },
        "results": results,
        "mechanism_test": verdict,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # width_vs_L.csv
    with open(RESULTS_DIR / "width_vs_L.csv", "w") as f:
        f.write("system,L,T,n_ok,W_sat,W_sat_std,alpha_local,alpha_local_r2,"
                "intrinsic_width,late_beta\n")
        for sysn in systems:
            for d in results[sysn]["per_L"]:
                if d.get("n_ok", 0) == 0:
                    continue
                f.write(f"{sysn},{d['L']},{d['T']},{d['n_ok']},{d['W_sat']:.6f},"
                        f"{d['W_sat_std']:.6f},{d['alpha_local']:.6f},"
                        f"{d['alpha_local_r2']:.6f},{d['intrinsic_width']:.6f},"
                        f"{d['late_beta']:.6f}\n")

    # Gr_curves.csv (largest L per system)
    with open(RESULTS_DIR / "Gr_curves.csv", "w") as f:
        f.write("system,L,r,Gr\n")
        for sysn in systems:
            curves = [d for d in results[sysn]["per_L"] if "Gr" in d]
            if not curves:
                continue
            d = curves[-1]
            for r, g in zip(d["r_vals"], d["Gr"]):
                f.write(f"{sysn},{d['L']},{r},{g:.6e}\n")

    make_figure(results, RESULTS_DIR / "anomalous_scaling.png")
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
