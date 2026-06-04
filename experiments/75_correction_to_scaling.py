"""Experiment 75: correction-to-scaling extrapolation -- does BD's alpha reach 0.5?

Background
----------
Exp 74 pinned the residual obstruction to BD's *saturated* roughness exponent:
alpha_eff(L<=128) ~ 0.34, well below the KPZ-class value 1/2, while BD's growth
exponent beta had already converged. BD is famous for strong corrections to
scaling: its large lattice-scale intrinsic width adds a sub-leading term that
depresses the effective exponent at accessible L. The decisive question is
whether *correcting* for that term recovers alpha_infinity = 1/2 -- i.e. whether
BD's roughness exponent is anomalous only because of finite size, or genuinely
unrecoverable from accessible data.

Model: W_sat(L) = A * L^alpha * (1 + B * L^(-omega)).
The effective exponent alpha_eff(L) = d log W_sat / d log L drifts to alpha as
L -> infinity:  alpha_eff(Lbar) = alpha_inf + c * Lbar^(-omega).

Two extrapolations (per system, seed-bootstrapped):
  (1) effective-exponent intercept: linear fit of alpha_eff vs Lbar^(-omega)
      at fixed omega (primary; omega in {1.0, 0.5} + free-omega curve_fit).
  (2) direct nonlinear fit of W_sat(L) to A L^alpha (1 + B L^(-omega)).

Decisive: does BD's extrapolated alpha_inf reach ~0.5 (completing the exp74
merge), and with what uncertainty at accessible L? Sanity anchor: EW and KPZ
should extrapolate to ~0.5 (their true 1+1D roughness exponent).

Outputs (read-only):
- results_exp75_correction_to_scaling/summary.json
- results_exp75_correction_to_scaling/wsat_vs_L.csv
- results_exp75_correction_to_scaling/correction_to_scaling.png  (if mpl)

Usage:
  python experiments/75_correction_to_scaling.py            # full
  python experiments/75_correction_to_scaling.py --quick    # smoke
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXP63_PATH = PROJECT_DIR / "experiments" / "63_temporal_features.py"
RESULTS_DIR = PROJECT_DIR / "results_exp75_correction_to_scaling"

ALPHA_THEORY = {"ew": 0.5, "kpz": 0.5, "bd": 0.5, "eden": 0.5, "rd": None}

try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


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
    }
    fn = sims[system]
    for attempt in range(6):
        traj = fn(seed + attempt * 100003)
        if traj is not None and np.all(np.isfinite(traj)):
            return traj
    return None


def w_sat(traj):
    """Saturated global width: mean over the late 20% of frames of std_x[h]."""
    T = traj.shape[0]
    return float(np.mean(np.std(traj[int(0.8 * T):], axis=1)))


def alpha_eff(Ls, Ws):
    """Successive-pair effective exponents at geometric-mean L."""
    Ls, Ws = np.asarray(Ls, float), np.asarray(Ws, float)
    Lbar, aeff = [], []
    for i in range(len(Ls) - 1):
        if Ws[i] > 0 and Ws[i + 1] > 0:
            Lbar.append(np.sqrt(Ls[i] * Ls[i + 1]))
            aeff.append((np.log(Ws[i + 1]) - np.log(Ws[i])) /
                        (np.log(Ls[i + 1]) - np.log(Ls[i])))
    return np.array(Lbar), np.array(aeff)


def extrapolate_intercept(Lbar, aeff, omega):
    """Linear fit alpha_eff = alpha_inf + c*Lbar^(-omega); return intercept."""
    if len(Lbar) < 2:
        return float("nan")
    x = Lbar ** (-omega)
    slope, intercept = np.polyfit(x, aeff, 1)
    return float(intercept)


def fit_correction_fixed_omega(Ls, Ws, omega):
    """Fit W = A L^alpha (1 + B L^(-omega)) with omega FIXED (3 params: A,alpha,B).

    Fixing omega is far more stable than a free 4-param fit on noisy finite-L data.
    """
    if not HAVE_SCIPY or len(Ls) < 3:
        return float("nan")
    Ls, Ws = np.asarray(Ls, float), np.asarray(Ws, float)

    def model(L, A, alpha, B):
        return A * L ** alpha * (1.0 + B * L ** (-omega))
    try:
        p0 = [Ws[0] / Ls[0] ** 0.5, 0.5, 0.0]
        bounds = ([1e-6, 0.0, -50], [1e6, 1.2, 50])
        popt, _ = curve_fit(model, Ls, Ws, p0=p0, bounds=bounds, maxfev=20000)
        return float(popt[1])
    except Exception:
        return float("nan")


def fit_correction_free(Ls, Ws):
    """Free 4-param fit (diagnostic only; unstable on noisy data)."""
    if not HAVE_SCIPY or len(Ls) < 4:
        return float("nan"), float("nan")
    Ls, Ws = np.asarray(Ls, float), np.asarray(Ws, float)

    def model(L, A, alpha, B, omega):
        return A * L ** alpha * (1.0 + B * L ** (-omega))
    try:
        p0 = [Ws[0] / Ls[0] ** 0.5, 0.5, 0.0, 1.0]
        bounds = ([1e-6, 0.0, -50, 0.1], [1e6, 1.2, 50, 4.0])
        popt, _ = curve_fit(model, Ls, Ws, p0=p0, bounds=bounds, maxfev=20000)
        return float(popt[1]), float(popt[3])
    except Exception:
        return float("nan"), float("nan")


def analyze(exp63, system, L_ladder, T_schedule, n_seeds):
    """Per-(L,seed) W_sat; then naive alpha and correction-to-scaling alpha_inf."""
    Wsat = {L: [] for L in L_ladder}
    for L in L_ladder:
        T = T_schedule(L)
        for s in range(n_seeds):
            traj = simulate(exp63, system, L, T, seed=5000 + 31 * s)
            if traj is not None:
                Wsat[L].append(w_sat(traj))
    Ls = [L for L in L_ladder if Wsat[L]]
    W_mean = [float(np.mean(Wsat[L])) for L in Ls]

    # naive single power law over full ladder
    naive_alpha = float(np.polyfit(np.log(Ls), np.log(W_mean), 1)[0]) if len(Ls) >= 2 else float("nan")

    # primary: direct correction-to-scaling fit at fixed omega (uses all L jointly)
    a_dir_w1 = fit_correction_fixed_omega(Ls, W_mean, 1.0)
    a_dir_w05 = fit_correction_fixed_omega(Ls, W_mean, 0.5)
    a_free, omega_free = fit_correction_free(Ls, W_mean)
    # secondary: effective-exponent intercept (noisier finite-difference method)
    Lbar, aeff = alpha_eff(Ls, W_mean)
    a_int_w1 = extrapolate_intercept(Lbar, aeff, 1.0)

    # bootstrap the primary estimator (direct fit, omega=1) over seeds
    rng = np.random.default_rng(75)
    boot = []
    for _ in range(200):
        Wb = []
        for L in Ls:
            vals = np.array(Wsat[L])
            Wb.append(float(np.mean(vals[rng.integers(0, len(vals), len(vals))])))
        boot.append(fit_correction_fixed_omega(Ls, Wb, 1.0))
    boot = np.array([b for b in boot if np.isfinite(b)])

    return {
        "system": system,
        "Ls": Ls,
        "W_sat": W_mean,
        "alpha_eff_Lbar": Lbar.tolist(),
        "alpha_eff": aeff.tolist(),
        "naive_alpha": naive_alpha,
        "alpha_inf_directfit_omega1": a_dir_w1,
        "alpha_inf_directfit_omega1_std": float(np.std(boot)) if len(boot) else float("nan"),
        "alpha_inf_directfit_omega0p5": a_dir_w05,
        "alpha_inf_intercept_omega1": a_int_w1,
        "alpha_inf_freefit": a_free,
        "omega_freefit": omega_free,
        "alpha_theory": ALPHA_THEORY[system],
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 75: correction-to-scaling")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp63 = load_exp63()
    RESULTS_DIR.mkdir(exist_ok=True)

    if args.quick:
        L_ladder = [24, 32, 48, 64, 96]
        n_seeds = 4
    else:
        L_ladder = [32, 48, 64, 96, 128, 192, 256]
        n_seeds = 6
    T_schedule = lambda L: int(30 * L ** 1.5)
    systems = ["ew", "kpz", "bd", "eden", "rd"]

    print("Experiment 75: correction-to-scaling extrapolation of alpha")
    print(f"  L-ladder={L_ladder}  seeds={n_seeds}  scipy={HAVE_SCIPY}")

    results = {}
    for s in systems:
        print(f"\n[{s}] simulating to saturation...")
        results[s] = analyze(exp63, s, L_ladder, T_schedule, n_seeds)
        r = results[s]
        print("  alpha_eff(Lbar): " +
              ", ".join(f"{lb:.0f}:{ae:+.3f}" for lb, ae in
                        zip(r["alpha_eff_Lbar"], r["alpha_eff"])))
        print(f"  naive alpha = {r['naive_alpha']:.3f} | "
              f"alpha_inf direct(w=1) = {r['alpha_inf_directfit_omega1']:.3f} +/- {r['alpha_inf_directfit_omega1_std']:.3f} | "
              f"direct(w=.5) = {r['alpha_inf_directfit_omega0p5']:.3f} | "
              f"free-fit = {r['alpha_inf_freefit']:.3f} (omega={r['omega_freefit']:.2f})")

    # ---- summary table ----
    print("\n" + "=" * 80)
    print(f"{'system':6s} {'naive_a':>8s} {'a_inf direct(w1)':>17s} {'direct(w.5)':>12s} "
          f"{'freefit':>8s} {'theory':>7s}")
    print("-" * 80)
    for s in systems:
        r = results[s]
        th = r["alpha_theory"]
        th_s = f"{th:.2f}" if th is not None else "  -- "
        print(f"{s:6s} {r['naive_alpha']:>8.3f} "
              f"{r['alpha_inf_directfit_omega1']:>10.3f}+-{r['alpha_inf_directfit_omega1_std']:.2f} "
              f"{r['alpha_inf_directfit_omega0p5']:>12.3f} {r['alpha_inf_freefit']:>8.3f} {th_s:>7s}")

    # ---- sanity gate: do EW/KPZ extrapolate to ~0.5? ----
    ew_kpz_ok = all(abs(results[s]["alpha_inf_directfit_omega1"] - 0.5) < 0.12
                    for s in ("ew", "kpz")
                    if np.isfinite(results[s]["alpha_inf_directfit_omega1"]))
    print(f"\nsanity gate (EW & KPZ extrapolate to ~0.5): "
          f"{'PASS -- extrapolation trustworthy' if ew_kpz_ok else 'FAIL -- extrapolation unreliable at L<=max'}")

    # ---- decisive BD verdict ----
    bd = results["bd"]
    print("\nBD verdict (does correction-to-scaling complete the merge?):")
    print(f"  naive alpha at L<=max : {bd['naive_alpha']:.3f}  (exp74 saw ~0.34)")
    print(f"  extrapolated alpha_inf: {bd['alpha_inf_directfit_omega1']:.3f} +/- "
          f"{bd['alpha_inf_directfit_omega1_std']:.3f}  (KPZ-class target 0.5)")
    reaches = abs(bd["alpha_inf_directfit_omega1"] - 0.5) < (2 * bd["alpha_inf_directfit_omega1_std"] + 0.05)
    print(f"  consistent with 0.5 within ~2 sigma: {reaches}")
    print(f"  (only meaningful if the sanity gate PASSES)")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({"config": {"L_ladder": L_ladder, "n_seeds": n_seeds,
                              "quick": args.quick, "scipy": HAVE_SCIPY},
                   "results": results}, f, indent=2)
    with open(RESULTS_DIR / "wsat_vs_L.csv", "w") as f:
        f.write("system,L,W_sat\n")
        for s in systems:
            for L, w in zip(results[s]["Ls"], results[s]["W_sat"]):
                f.write(f"{s},{L},{w:.6f}\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"ew": "#1f77b4", "kpz": "#ff7f0e", "bd": "#d62728",
                  "eden": "#9467bd", "rd": "#7f7f7f"}
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for s in systems:
            r = results[s]
            ax[0].loglog(r["Ls"], r["W_sat"], "o-", color=colors[s], label=s)
        ax[0].set_xlabel("L"); ax[0].set_ylabel("W_sat(L)")
        ax[0].set_title("Saturated width vs L")
        ax[0].legend(fontsize=8)
        for s in systems:
            r = results[s]
            if not r["alpha_eff_Lbar"]:
                continue
            x = np.array(r["alpha_eff_Lbar"]) ** (-1.0)
            ax[1].plot(x, r["alpha_eff"], "o-", color=colors[s], label=s,
                       lw=2.5 if s == "bd" else 1.3)
            ax[1].scatter([0], [r["alpha_inf_directfit_omega1"]], color=colors[s],
                          marker="*", s=120, zorder=5)
        ax[1].axhline(0.5, ls="--", c="k", lw=0.8, label="KPZ-class alpha=1/2")
        ax[1].set_xlabel("1/Lbar  (extrapolation to L->inf at x=0)")
        ax[1].set_ylabel("effective alpha")
        ax[1].set_title("Correction-to-scaling extrapolation (stars = alpha_inf)")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "correction_to_scaling.png", dpi=130)
        plt.close(fig)
        print(f"\n  figure -> {RESULTS_DIR/'correction_to_scaling.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
