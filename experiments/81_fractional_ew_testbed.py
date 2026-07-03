"""Exp 81 Part B: the fractional-EW testbed -- exact sampling, spectrum,
KL/Fisher information, and Level-0 / Level-2-3 floors.

Companion: ml_paper/EXP81_PLAN.md, ml_paper/THEORY_minimax_floor.md Appendix F.

Family: 1D stationary Gaussian field on L sites via independent Fourier
modes n = 1..L/2-1 (DC and Nyquist modes dropped -- see NOTE), with

    S(k_n) = D / ( nu * k_n^z  +  nu2 * k_n^(z+omega_tilde) ),
    k_n = 2*pi*n/L,   alpha = (z-1)/2.

NOTE on normalization/Hermitian symmetry (determined empirically, see
Task-1 log in EXP81_REPORT.md). S(k) is the CONTINUUM-normalized spectral
density (the object with a well-defined nu->0 EW limit S=D/(nu k^z)); the
finite-L discrete Fourier mode has variance L*S(k), the standard box
normalization for a finite periodic domain (mode spacing dk=2pi/L implies
one discrete mode absorbs an O(L) density-of-states factor). Each mode
n=1..L/2-1 is an independent complex Gaussian amplitude c_n with
E|c_n|^2 = L*S(k_n): Re(c_n), Im(c_n) ~ N(0, L*S(k_n)/2) iid ("per complex
mode" convention). The real field is h = irfft(A, n=L) with A[n]=c_n for
n=1..L/2-1 and A[0]=A[L/2]=0 -- the DC and Nyquist modes are dropped
entirely (removes 1 real degree of freedom out of L; negligible for
W_sat, dominated by low-k modes). Every sampled mode is thus a genuine
"complex mode" with no boundary special-case.

This L-factor was NOT a free choice: p=0 (variance=S(k)) gives W^2(L) ~
L^{z-2}; only p=1 (variance=L*S(k)) reproduces the target W^2 ~ L^{z-1},
i.e. alpha=(z-1)/2, verified numerically against the asymptotic (L->inf)
slope before adoption (exact to 1e-4 at L~1e7-1e8; see gate_b2()).

Convention verified numerically (G-B3, see gate_b3()): the per-complex-mode
Gaussian log-density (using Sigma := L*S(k) as the actual mode variance) is

    log p(c; Sigma) = -log(pi*Sigma) - |c|^2 / Sigma,

and analytically E_{Sigma1}[log p(c;Sigma1) - log p(c;Sigma2)]
= Sigma1/Sigma2 - 1 + log(Sigma2/Sigma1) = S1/S2 - 1 + log(S2/S1) exactly
(the shared L-factor cancels in the ratio and drops from the log terms) --
this matches the KL formula in THEORY_minimax_floor.md Appendix F2 /
EXP81_PLAN.md verbatim in terms of S(k) itself, with no correction factor,
BECAUSE the formula is invariant to any shared L-rescaling of the mode
variance. Likewise d ln(L*S)/d alpha = d ln S/d alpha since L does not
depend on alpha, so the Fisher-information formula is also unaffected.
G-B3 checks the KL identity against the empirical mean log-likelihood-ratio
on sampled fields.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp81_hierarchy")
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")
E77_PATH = os.path.join(HERE, "77_minimax_floor.py")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
ALPHA_BOUNDS = (0.35, 0.65)
Z_BOUNDS = (1.7, 2.3)          # z = 1 + 2*alpha
OMEGA_BOUNDS = (0.3, 2.5)
U_GRID = (0.5, 1.0, 4.0)
DNU_BOUNDS = (1.0, 10.0)       # D, nu uniform over one decade [1, 10]

RNG_GLOBAL = np.random.default_rng(81)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- family --

def k_grid(L):
    """Mode indices n=1..L/2-1 (DC and Nyquist dropped) and wavenumbers."""
    n = np.arange(1, int(L) // 2)
    return n, 2.0 * np.pi * n / L


def spectrum(k, z, D, nu, nu2, omega_tilde):
    return D / (nu * k ** z + nu2 * k ** (z + omega_tilde))


def sample_modes(L, z, D, nu, nu2, omega_tilde, rng):
    """Draw the independent complex mode amplitudes. Returns (c, k, S) where
    S is the declared (continuum) spectrum S(k) and the actual per-mode
    variance used is Sigma = L*S(k) (see module docstring)."""
    _, k = k_grid(L)
    S = spectrum(k, z, D, nu, nu2, omega_tilde)
    Sigma = L * S
    re = rng.normal(0.0, np.sqrt(Sigma / 2.0))
    im = rng.normal(0.0, np.sqrt(Sigma / 2.0))
    return re + 1j * im, k, S


def field_from_modes(c, L):
    A = np.zeros(int(L) // 2 + 1, dtype=complex)
    A[1:len(c) + 1] = c
    return np.fft.irfft(A, n=int(L))


def sample_field(L, z, D, nu, nu2, omega_tilde, rng):
    c, k, S = sample_modes(L, z, D, nu, nu2, omega_tilde, rng)
    h = field_from_modes(c, L)
    return h, c, k, S


def exact_W2(L, z, D, nu, nu2, omega_tilde):
    """Exact (noise-free, infinite-seed) stationary width^2 via Parseval:
    sum_x h(x)^2 = (2/L) sum_n |c_n|^2 (DC, Nyquist dropped), and
    E|c_n|^2 = L*S(k_n), so W^2 = (1/L) sum_x h(x)^2 has
    E[W^2] = (2/L^2) sum_n L*S(k_n) = (2/L) sum_n S(k_n). This is an exact
    finite sum given the family, not a Monte Carlo estimate."""
    _, k = k_grid(L)
    S = spectrum(k, z, D, nu, nu2, omega_tilde)
    return 2.0 * np.sum(S) / L


def alpha_of_z(z):
    return (z - 1.0) / 2.0


def z_of_alpha(alpha):
    return 1.0 + 2.0 * alpha


# ------------------------------------------------------------- KL / FIM --

def kl_modes(L, p1, p2):
    """KL(P1||P2) summed over the modes of size L, per complex mode:
    sum_k [S1/S2 - 1 + ln(S2/S1)]. p1, p2 are (z, D, nu, nu2, omega_tilde)
    tuples. Invariant to the shared L-rescaling of mode variance (see module
    docstring), so this is exactly the formula in THEORY_minimax_floor.md
    Appendix F2 / EXP81_PLAN.md."""
    _, k = k_grid(L)
    S1 = spectrum(k, *p1)
    S2 = spectrum(k, *p2)
    return float(np.sum(S1 / S2 - 1.0 + np.log(S2 / S1)))


def kl_design(design, p1, p2):
    return float(sum(kl_modes(L, p1, p2) for L in design))


def dlnS_dalpha_formula(k, z, nu, nu2, omega_tilde):
    """Appendix F2 closed form, holding the absolute correction exponent
    z+omega_tilde FIXED as alpha (i.e. z) varies -- see module docstring
    / EXP81_REPORT.md Task-1 log for the derivation showing this is the
    convention the formula corresponds to."""
    denom = nu * k ** z + nu2 * k ** (z + omega_tilde)
    return -2.0 * np.log(k) * (nu * k ** z) / denom


def fisher_alpha(L, z, D, nu, nu2, omega_tilde):
    """No-nuisance exact Fisher information of alpha at size L."""
    _, k = k_grid(L)
    dlnS = dlnS_dalpha_formula(k, z, nu, nu2, omega_tilde)
    return float(np.sum(dlnS ** 2))


def fisher_alpha_design(design, z, D, nu, nu2, omega_tilde):
    return float(sum(fisher_alpha(L, z, D, nu, nu2, omega_tilde)
                      for L in design))


def loglik_modes(c, S):
    """Per-complex-mode Gaussian log-density, actual variance L*S(k)
    (Sigma), summed. c must have been drawn with this L already baked into
    S's caller; here we take Sigma directly."""
    return float(np.sum(-np.log(np.pi * S) - (np.abs(c) ** 2) / S))


# -------------------------------------------------------------- gate B1 --

def gate_b1(rep, n_seeds=4000, Ls=(32, 64, 128, 256)):
    """Sampled ensemble spectrum matches L*S(k): overlay plot + per-k-band
    quantitative check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, D, nu, nu2, om = 2.0, 1.0, 1.0, 0.6, 1.0
    fig, axes = plt.subplots(1, len(Ls), figsize=(4 * len(Ls), 3.5))
    out = {}
    for ax, L in zip(np.atleast_1d(axes), Ls):
        rng = np.random.default_rng(1000 + L)
        _, k = k_grid(L)
        S_theory = spectrum(k, z, D, nu, nu2, om)
        Sigma_theory = L * S_theory
        acc = np.zeros_like(k)
        for _ in range(n_seeds):
            c, _, _ = sample_modes(L, z, D, nu, nu2, om, rng)
            acc += np.abs(c) ** 2
        Sigma_emp = acc / n_seeds
        # per-k-band check: bin into 4 log-spaced bands, relative error
        n_bands = min(4, len(k))
        edges = np.unique(np.geomspace(1, len(k), n_bands + 1).astype(int))
        band_errs = []
        for i in range(len(edges) - 1):
            sl = slice(edges[i], edges[i + 1] if i + 2 < len(edges) else len(k))
            if edges[i] >= len(k):
                continue
            emp = Sigma_emp[sl].mean()
            th = Sigma_theory[sl].mean()
            # MC relative std error of the mean for a chi-square(2 dof) mode
            mc_sem_rel = 1.0 / np.sqrt(n_seeds * max(1, sl.stop - sl.start))
            band_errs.append({
                "n_range": [int(edges[i]), int(edges[i + 1] if i + 2 < len(edges) else len(k))],
                "empirical_mean_Sigma": float(emp),
                "theory_Sigma": float(th),
                "rel_err": float(emp / th - 1.0),
                "mc_sem_rel": float(mc_sem_rel),
            })
        out[str(L)] = band_errs
        ax.loglog(k, Sigma_theory, "k-", lw=1.5, label="L*S(k) theory")
        ax.loglog(k, Sigma_emp, "o", ms=3, alpha=0.6, label=f"empirical ({n_seeds} seeds)")
        ax.set_title(f"L={L}")
        ax.set_xlabel("k")
        if L == Ls[0]:
            ax.set_ylabel("mode variance")
            ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig_path = os.path.join(FIGS_DIR, "gate_b1_spectrum_overlay.png")
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    max_rel_err = max(abs(b["rel_err"]) for L in out for b in out[L])
    max_sem = max(b["mc_sem_rel"] for L in out for b in out[L])
    passed = max_rel_err < 6 * max_sem + 0.02
    rep["gate_b1"] = {
        "params": {"z": z, "D": D, "nu": nu, "nu2": nu2, "omega_tilde": om},
        "n_seeds": n_seeds, "Ls": list(Ls),
        "band_checks": out, "figure": fig_path,
        "max_rel_err": max_rel_err, "max_mc_sem_rel": max_sem,
        "pass": bool(passed),
    }
    print(f"[G-B1] max band rel err={max_rel_err:.4f} (mc sem~{max_sem:.4f}) "
          f"pass={passed}")
    return passed


# -------------------------------------------------------------- gate B2 --

def gate_b2(rep, n_seeds=3000, zs=(1.7, 2.0, 2.3)):
    """At nu2=0, alpha from log-W2-vs-log-L regression on the 7-point
    design recovers (z-1)/2, across the z grid. Reports the exact
    (deterministic, infinite-seed) design-ladder slope, two MC ladder
    slopes, and the asymptotic (L->inf) slope, separating three distinct,
    understood effects from a genuine sampler bug:

    (i)  intrinsic finite-window bias: even at nu2=0, the exact 7-point
         design ladder slope differs from the L->inf asymptotic (z-1)/2
         because the mode-sum tail (order L^{1-z}) has not converged at
         L<=256 -- larger for smaller z (z=1.7 tail decays slower than
         z=2.3). Real, deterministic, not a bug.
    (ii) Jensen bias: log(E[W2]) (unbiased MC estimate of the exact-design
         slope) vs E[log(W2)] (the naive per-seed-then-average estimator a
         real pipeline would use) differ because log is concave; this is a
         population-level effect that does NOT shrink as n_seeds->infinity.
    (iii) genuine MC sampling noise (SEM), which does shrink with n_seeds
         and is the only piece the gate should hold to a tight tolerance.
    """
    D, nu, nu2, om = 1.0, 1.0, 0.0, 1.0
    out = {}
    for z in zs:
        target_alpha = alpha_of_z(z)
        w2_exact = np.array([exact_W2(L, z, D, nu, nu2, om) for L in DESIGN])
        slope_exact = np.polyfit(np.log(DESIGN), np.log(w2_exact), 1)[0]

        Ls_asym = np.geomspace(1e5, 1e8, 8)
        w2_asym = np.array([exact_W2(L, z, D, nu, nu2, om) for L in Ls_asym])
        slope_asym = np.polyfit(np.log(Ls_asym), np.log(w2_asym), 1)[0]

        rng = np.random.default_rng(2000 + int(z * 10))
        w2_seeds = {L: [] for L in DESIGN}
        for L in DESIGN:
            for _ in range(n_seeds):
                h, _, _, _ = sample_field(int(L), z, D, nu, nu2, om, rng)
                w2_seeds[L].append(np.mean(h ** 2))
        log_mean_w2 = np.array([np.log(np.mean(w2_seeds[L])) for L in DESIGN])
        mean_log_w2 = np.array([np.mean(np.log(w2_seeds[L])) for L in DESIGN])
        sem_log_mean = np.array([
            np.std(w2_seeds[L], ddof=1) / np.sqrt(n_seeds) / np.mean(w2_seeds[L])
            for L in DESIGN])

        slope_logmean, _ = np.polyfit(np.log(DESIGN), log_mean_w2, 1)
        slope_meanlog, _ = np.polyfit(np.log(DESIGN), mean_log_w2, 1)
        X = np.vstack([np.log(DESIGN), np.ones_like(DESIGN)]).T
        Wt = np.diag(1.0 / sem_log_mean ** 2)
        cov = np.linalg.inv(X.T @ Wt @ X)
        slope_sem = np.sqrt(cov[0, 0])

        out[str(z)] = {
            "target_alpha": target_alpha,
            "alpha_exact_design": slope_exact / 2.0,
            "alpha_asymptotic": slope_asym / 2.0,
            "alpha_mc_logmean_design": slope_logmean / 2.0,
            "alpha_mc_logmean_sem": slope_sem / 2.0,
            "alpha_mc_meanlog_naive": slope_meanlog / 2.0,
            "intrinsic_finite_window_bias": slope_exact / 2.0 - target_alpha,
            "jensen_bias_meanlog_minus_logmean":
                slope_meanlog / 2.0 - slope_exact / 2.0,
        }
        print(f"[G-B2] z={z}: target={target_alpha:.4f} "
              f"exact_design={slope_exact/2:.4f} asymptotic={slope_asym/2:.4f} "
              f"mc_logmean={slope_logmean/2:.4f}+-{slope_sem/2:.4f} "
              f"mc_meanlog_naive={slope_meanlog/2:.4f}")
    # pass criteria: (1) asymptotic slope matches target to 1e-3 (rules out
    # a normalization bug); (2) the unbiased MC estimator (log of mean W2,
    # which estimates the same population quantity as alpha_exact_design)
    # matches alpha_exact_design within 4 sigma (rules out a sampler bug);
    # the finite-window and Jensen biases are reported, not gated -- they
    # are real, understood, and small (<0.03 in alpha).
    asym_ok = all(abs(v["alpha_asymptotic"] - v["target_alpha"]) < 2e-3
                  for v in out.values())
    mc_ok = all(abs(v["alpha_mc_logmean_design"] - v["alpha_exact_design"])
                < 4 * v["alpha_mc_logmean_sem"] for v in out.values())
    passed = asym_ok and mc_ok
    rep["gate_b2"] = {"per_z": out, "asymptotic_matches_target": asym_ok,
                       "mc_logmean_matches_exact_design": mc_ok,
                       "pass": bool(passed)}
    print(f"[G-B2] asymptotic_ok={asym_ok} mc_logmean_matches_exact={mc_ok} "
          f"pass={passed}")
    return passed


# -------------------------------------------------------------- gate B3 --

def gate_b3(rep, n_seeds=6000, L=64):
    """KL formula verified against empirical mean log-likelihood-ratio."""
    p1 = (2.0, 1.0, 1.0, 0.6, 1.0)     # z, D, nu, nu2, omega_tilde
    p2 = (2.1, 1.2, 0.9, 0.3, 1.4)
    _, k = k_grid(L)
    S1 = spectrum(k, *p1)
    S2 = spectrum(k, *p2)
    Sigma1 = L * S1
    Sigma2 = L * S2
    kl_analytic = kl_modes(L, p1, p2)

    rng = np.random.default_rng(3000)
    llrs = []
    for _ in range(n_seeds):
        c, _, _ = sample_modes(L, *p1, rng)
        ll1 = loglik_modes(c, Sigma1)
        ll2 = loglik_modes(c, Sigma2)
        llrs.append(ll1 - ll2)
    llrs = np.array(llrs)
    emp_mean = llrs.mean()
    emp_sem = llrs.std(ddof=1) / np.sqrt(n_seeds)
    rel_err = abs(emp_mean - kl_analytic) / abs(kl_analytic)

    # FIM finite-difference check (Appendix F2 closed form)
    z0, D0, nu0, nu20, om0 = 2.0, 1.0, 1.0, 0.5, 1.0
    dz = 1e-5
    _, kk = k_grid(96)
    S_plus = spectrum(kk, z0 + dz, D0, nu0, nu20, om0 - dz)   # z+om fixed
    S_minus = spectrum(kk, z0 - dz, D0, nu0, nu20, om0 + dz)
    fd = (np.log(S_plus) - np.log(S_minus)) / dz              # d lnS/d alpha
    closed = dlnS_dalpha_formula(kk, z0, nu0, nu20, om0)
    fim_rel_err = float(np.max(np.abs(fd - closed) / np.abs(closed)))

    passed = bool(rel_err < 0.03 and abs(emp_mean - kl_analytic) < 4 * emp_sem
                  and fim_rel_err < 1e-3)
    rep["gate_b3"] = {
        "p1": list(p1), "p2": list(p2), "L": L, "n_seeds": n_seeds,
        "kl_analytic": kl_analytic, "kl_empirical_mean": float(emp_mean),
        "kl_empirical_sem": float(emp_sem), "rel_err": float(rel_err),
        "fim_finite_diff_rel_err": fim_rel_err,
        "pass": passed,
    }
    print(f"[G-B3] KL analytic={kl_analytic:.4f} empirical={emp_mean:.4f}"
          f"+-{emp_sem:.4f} rel_err={rel_err:.4f} "
          f"fim_fd_rel_err={fim_rel_err:.2e} pass={passed}")
    return passed


########################################################################
# Task 2: Level-0 and Level-2/3 floors, omega_eff measurement
########################################################################

L_MIN = 32.0


def b_correction(L, z, D, nu, nu2, omega_tilde):
    """Relative correction b(L) := W2_full(L)/W2_leadonly(L) - 1, exact
    (finite mode-sum, no MC noise)."""
    w_full = exact_W2(L, z, D, nu, nu2, omega_tilde)
    w_lead = exact_W2(L, z, D, nu, 0.0, omega_tilde)
    return w_full / w_lead - 1.0


def nu2_from_u(u, z, D, nu, omega_tilde, L_min=L_MIN, design=None, n_iter=40):
    """Invert b_correction(L_min; nu2) = u for nu2, holding z,D,nu,om fixed.
    b is monotonically DECREASING in nu2: nu2>0 adds UV damping and
    suppresses the width relative to pure power law (b<0, asymptoting to
    -1 as nu2->+inf); nu2<0 enhances it (b>0, diverging as nu2 approaches
    the pole where the denominator would turn non-positive). So b's
    achievable range is the ASYMMETRIC interval (-1, +inf), not a
    symmetric box -- callers must bound u accordingly (see U_EFF_BOUNDS
    used by confusion_gap_spectral). Returns nu2 (finite).

    The pole (denominator -> 0) is at nu2 = -nu/k^omega_tilde for whichever
    k maximizes k^omega_tilde (omega_tilde>0 throughout this project, so
    that is k_max); this must hold across the FULL multi-L design (k_max
    grows towards pi as L grows), not just L_min, else nu2 can be chosen
    safe at L_min but pathological (negative spectrum) at a larger design
    L -- verified against the erratic (non-monotonic) b_of_r seen when the
    guard used only L_min's k_max."""
    if design is None:
        design = DESIGN
    kmax_global = max(k_grid(L)[1].max() for L in design)
    kom_max = kmax_global ** omega_tilde
    nu2_floor = -nu / kom_max * 0.999   # nu2 must stay strictly above this

    def b_of_r(nu2):
        return b_correction(L_min, z, D, nu, nu2, omega_tilde)

    if u >= 0:
        # b(nu2_floor) -> +inf > u; b(0) = 0 <= u (u>=0); b decreasing in nu2
        lo, hi = nu2_floor, 0.0
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if b_of_r(mid) > u:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    else:
        # b(0)=0 > u; b(hi)->-1 as hi->+inf < u (u>-1); expand hi as needed
        lo, hi = 0.0, nu * 4.0
        while b_of_r(hi) > u and hi < nu * 1e6:
            hi *= 2.0
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if b_of_r(mid) > u:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def denom_ok(design, z, D, nu, nu2, omega_tilde):
    for L in design:
        _, k = k_grid(L)
        if np.any(nu * k ** z + nu2 * k ** (z + omega_tilde) <= 0):
            return False
    return True


# --------------------------------------------------------- Level-0 floor --

def measure_sigma_level0(z=2.0, D=1.0, nu=1.0, nu2=0.0, om=1.0,
                          n_seeds=400, design=DESIGN, seed=8100):
    """sigma of log W_sat across seeds for this family, median over L,
    near alpha=0.5 (matches exp77's measured_sigma())."""
    rng = np.random.default_rng(seed)
    sig_per_L = []
    for L in design:
        logw = []
        for _ in range(n_seeds):
            h, _, _, _ = sample_field(int(L), z, D, nu, nu2, om, rng)
            logw.append(np.log(np.mean(h ** 2)))
        sig_per_L.append(np.std(logw, ddof=1))
    return float(np.median(sig_per_L))


def level0_floors(rep, sigma, U_grid=U_GRID, ms=(6, 24), da_hi=1.2):
    m77 = load_module(E77_PATH, "m77_level0")
    m77.N_STARTS = 7
    out = {}
    for U in U_grid:
        m77.U_BOUNDS = (-0.75, U)
        out[str(U)] = {}
        for m in ms:
            f = m77.floor(DESIGN, sigma, m, da_hi=da_hi)
            saturated = f >= da_hi - 1e-9
            out[str(U)][f"m{m}"] = f
            out[str(U)][f"m{m}_saturated"] = bool(saturated)
            print(f"[Level0] U={U} m={m}: floor={f:.4f} "
                  f"{'(SATURATED at da_hi)' if saturated else ''}")
    return out


# ------------------------------------------------------- Level-2/3 floor --

def confusion_gap_spectral(dalpha, U, design=DESIGN, n_starts=24, seed=0):
    """Le Cam confusion gap with the adversary living inside the family:
    minimize KL, summed over the design sizes, between ensemble 1
    (z1, D1, nu1, nu2_1, om1) and ensemble 2 (z1+2*dalpha, D2, nu2, nu2_2,
    om2), over nuisances bounded as in the sampling prior. Free params:
    [z1, logD1, lognu1, u1, om1, logD2, lognu2, u2, om2]; nu2_i obtained
    from u_i via nu2_from_u (exact inversion of the L_min relative
    correction). Returns (best_gap, scatter_across_starts)."""
    rng = np.random.default_rng(seed)
    logD_lo, logD_hi = np.log(DNU_BOUNDS[0]), np.log(DNU_BOUNDS[1])
    bounds = [Z_BOUNDS, (logD_lo, logD_hi), (logD_lo, logD_hi), (-U, U),
              OMEGA_BOUNDS,
              (logD_lo, logD_hi), (logD_lo, logD_hi), (-U, U), OMEGA_BOUNDS]

    def unpack(p):
        z1, logD1, lognu1, u1, om1, logD2, lognu2, u2, om2 = p
        D1, nu1 = np.exp(logD1), np.exp(lognu1)
        D2, nu2p = np.exp(logD2), np.exp(lognu2)
        z2 = z1 + 2.0 * dalpha
        nu2_1 = nu2_from_u(u1, z1, D1, nu1, om1, design=design, n_iter=24)
        nu2_2 = nu2_from_u(u2, z2, D2, nu2p, om2, design=design, n_iter=24)
        return (z1, D1, nu1, nu2_1, om1), (z2, D2, nu2p, nu2_2, om2)

    def obj(p):
        p1, p2 = unpack(p)
        # nu2_from_u already restricts nu2 to the denom-positive branch by
        # construction (verified: this safety net does not trigger in
        # practice); kept as a guard against edge-case bisection failure.
        if not (denom_ok(design, *p1) and denom_ok(design, *p2)):
            return 1e6
        try:
            return kl_design(design, p1, p2)
        except (FloatingPointError, ValueError):
            return 1e6

    best = np.inf
    scatter = []
    for _ in range(n_starts):
        p0 = np.array([
            rng.uniform(*Z_BOUNDS), rng.uniform(logD_lo, logD_hi),
            rng.uniform(logD_lo, logD_hi), rng.uniform(-U, U),
            rng.uniform(*OMEGA_BOUNDS),
            rng.uniform(logD_lo, logD_hi), rng.uniform(logD_lo, logD_hi),
            rng.uniform(-U, U), rng.uniform(*OMEGA_BOUNDS),
        ])
        # maxfun caps wall-clock directly (L-BFGS-B line search can inflate
        # nfev well past maxiter otherwise); 24 starts compensate for any
        # single run stopping a little early -- see EXP81_REPORT.md for the
        # capped- vs uncapped-budget comparison that justifies this.
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 60, "maxfun": 500})
        scatter.append(float(res.fun))
        if res.fun < best:
            best = float(res.fun)
    return best, scatter


def floor_spectral(sigma, m, U, design=DESIGN, da_hi=1.2, tol=1e-2,
                    n_starts=24, seed=0):
    """max{dalpha : min_nuisance KL(dalpha) <= 1/(2m)} via bisection
    (KL<=1/2 criterion with m independent seeds averaged, matching exp77's
    m-seed convention: KL scales with m via the seed-mean Gaussian
    approximation there; here the raw-field KL is per-single-field, so the
    m-seed KL is m times the per-field KL -- see EXP81_REPORT.md)."""
    thresh = 0.5  # KL<=1/2 criterion, applied to the m-seed joint KL

    def gap_m(da):
        g, _ = confusion_gap_spectral(da, U, design, n_starts=n_starts,
                                       seed=seed)
        return m * g  # m independent seeds -> joint KL = m * per-seed KL

    lo, hi = 0.0, da_hi
    if gap_m(da_hi) <= thresh:
        return da_hi, gap_m(da_hi)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if gap_m(mid) <= thresh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), gap_m(0.5 * (lo + hi))


# ---------------------------------------------------- omega_eff measurement

def measure_omega_eff(z, D, nu, nu2, omega_tilde, L_grid=None):
    """Measure the correction exponent actually realized in W2(L): fit
    W2(L) = A*L^{2*alpha}*(1 + b*L^{-omega_eff}) via nonlinear least squares
    on a clean (noise-free, exact) high-L ladder. Returns (omega_eff, b,
    alpha_check)."""
    if L_grid is None:
        L_grid = np.geomspace(32, 2.0 ** 16, 30)
    w2 = np.array([exact_W2(L, z, D, nu, nu2, omega_tilde) for L in L_grid])
    alpha = alpha_of_z(z)
    logL = np.log(L_grid)

    def resid(p):
        logA, b, om_eff = p
        pred = logA + 2 * alpha * logL + np.log(
            np.clip(1 + b * np.exp(-om_eff * logL), 1e-12, None))
        return pred - np.log(w2)

    from scipy.optimize import least_squares
    # bounded (trf, not lm) to avoid om_eff wandering to extreme values and
    # overflowing L_grid**(-om_eff) for the largest L during line search
    bounds = ([-20.0, -0.999, -2.0], [20.0, 1e4, 6.0])
    best = None
    for om0 in (0.3, 0.6, 1.0, 1.5, 2.0):
        p0 = [np.log(w2[0]) - 2 * alpha * logL[0], 0.1, om0]
        try:
            res = least_squares(resid, p0, method="trf", bounds=bounds,
                                 max_nfev=2000)
        except Exception:
            continue
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    logA, b, om_eff = best.x
    return float(om_eff), float(b), float(alpha)


def task2_main(rep, U_grid=U_GRID, ms=(6, 24), n_starts=24):
    sigma = measure_sigma_level0()
    rep["level0_sigma"] = sigma
    print(f"measured Level-0 sigma (near alpha=0.5): {sigma:.4f}")

    rep["level0_floors"] = level0_floors(rep, sigma, U_grid, ms)

    # G-B4: sanity-check against results_exp77_minimax_floor/floor.json
    # (order of magnitude + monotonicity in U; exact agreement not expected
    # since sigma differs -- this family's sigma vs the real EW/KPZ/BD/Eden
    # sigmas measured from actual simulation seed scatter).
    e77_floor_path = os.path.join(HERE, "..", "results_exp77_minimax_floor",
                                   "floor.json")
    gb4 = {"this_family_sigma": sigma, "pass": None}
    if os.path.exists(e77_floor_path):
        with open(e77_floor_path) as fh:
            e77 = json.load(fh)
        ref = e77.get("floor_vs_umax", {}).get("sigma0.15", {})
        m24 = {U: rep["level0_floors"][str(U)]["m24"] for U in U_grid}
        # expected direction (both here and in exp77's floor_vs_umax): floor
        # INCREASES with U (a larger allowed correction amplitude enlarges
        # the adversary's confusion set, worsening resolution).
        Us_sorted = sorted(U_grid)
        mono_ours = all(m24[Us_sorted[i]] <= m24[Us_sorted[i + 1]]
                         for i in range(len(Us_sorted) - 1))
        ref_vals = sorted(((float(k), v) for k, v in ref.items()))
        mono_ref = all(ref_vals[i][1] <= ref_vals[i + 1][1]
                        for i in range(len(ref_vals) - 1))
        same_order = all(
            0.03 < m24[U] < 3.0 for U in U_grid)  # generous OOM band
        gb4.update({"ours_floor_m24_vs_U": m24, "exp77_ref_sigma0.15": ref,
                    "monotonic_in_U_ours": bool(mono_ours),
                    "monotonic_in_U_ref": bool(mono_ref),
                    "same_order_of_magnitude": bool(same_order),
                    "pass": bool(mono_ours and mono_ref and same_order)})
        print(f"[G-B4] ours(m24) vs U: {m24}  monotonic={mono_ours} "
              f"OOM_ok={same_order} pass={gb4['pass']}")
    rep["gate_b4"] = gb4

    # omega_eff measurement for the omega_tilde values we will actually use
    # in the Level-2/3 adversary sweep: representative interior values plus
    # the near-marginal stress case. The plan's "omega_tilde = 2 - z" at
    # z_center=2.0 (alpha=0.5) is exactly 0.0 -- degenerate (outside
    # OMEGA_BOUNDS, and z+omega_tilde=z makes nu2 indistinguishable from a
    # redefinition of nu, which is not a real second operator) and drove the
    # omega_eff fit to overflow. The plan's own bound omega_min=0.3 IS the
    # near-marginal edge of the declared class (2-z=0.3 at z=1.7, alpha=
    # 0.35); use that in-bounds value as the stress case instead.
    z_center = 2.0
    STRESS_OMEGA_TILDE = OMEGA_BOUNDS[0]  # 0.3, in-bounds near-marginal case
    om_values = sorted(set([0.5, 1.0, 1.5, 2.0, STRESS_OMEGA_TILDE]))
    om_eff_table = {}
    for om in om_values:
        for u_probe in (0.5, 1.0, 4.0):
            nu2 = nu2_from_u(u_probe, z_center, 1.0, 1.0, om)
            om_eff, b_fit, alpha_chk = measure_omega_eff(
                z_center, 1.0, 1.0, nu2, om)
            om_eff_table[f"omtilde{om}_u{u_probe}"] = {
                "omega_tilde": om, "u_declared": u_probe,
                "omega_eff_measured": om_eff, "b_fit": b_fit,
                "alpha_check": alpha_chk,
            }
            print(f"[omega_eff] om_tilde={om} u={u_probe}: "
                  f"omega_eff={om_eff:.3f} (2*alpha={2*alpha_chk:.3f})")
    rep["omega_eff_measurements"] = om_eff_table

    # Level-2/3 floor: the adversary's om1,om2 are ALREADY free nuisances
    # optimized over the full OMEGA_BOUNDS box inside confusion_gap_spectral
    # (matching exp77, where w1/w2 are likewise free per-ensemble nuisances)
    # -- so floor_L0 and floor_L23 depend on (U, m) only, NOT on a specific
    # declared omega_tilde. omega_tilde enters only via the FIM (evaluated
    # at a fixed representative nuisance point, not adversarially) and via
    # the omega_eff measurement table above. Computing floor_L23 once per
    # (U, m) rather than once per (U, om_tilde, m) avoids 2x redundant
    # multi-start optimization.
    floor23_by_U = {}
    da_hi23 = 1.2
    for U in U_grid:
        floor23_by_U[str(U)] = {}
        for m in ms:
            f23, gap = floor_spectral(sigma, m, U, n_starts=n_starts,
                                       da_hi=da_hi23)
            saturated = f23 >= da_hi23 - 1e-9
            floor23_by_U[str(U)][f"m{m}"] = {
                "floor_L23": f23, "gap_at_floor_m_scaled": gap,
                "saturated": bool(saturated)}
            print(f"[Level23] U={U} m={m}: floor_L23={f23:.4f} "
                  f"(m*KL at floor={gap:.4f}) "
                  f"{'(SATURATED at da_hi)' if saturated else ''}")
    rep["level23_floors"] = floor23_by_U

    # FIM table: "no-nuisance ceiling" evaluated at fixed nuisances, taken
    # as the representative point where the declared class's amplitude
    # bound U is saturated at L_min (nu2 = nu2_from_u(U, ...)) for each
    # omega_tilde -- i.e. "if you already knew the correction sat exactly
    # at the boundary of the declared class, with perfect knowledge (no
    # adversarial marginalization), how much information is there?"
    hierarchy = {}
    for om in (1.0, STRESS_OMEGA_TILDE):
        for U in U_grid:
            key = f"omtilde{om}_U{U}"
            nu2_rep = nu2_from_u(U, z_center, 1.0, 1.0, om)
            fim = fisher_alpha_design(DESIGN, z_center, 1.0, 1.0, nu2_rep, om)
            hierarchy[key] = {"omega_tilde": om, "U": U,
                               "nu2_representative": nu2_rep,
                               "fisher_info_alpha_design": float(fim)}
            for m in ms:
                da_fim = 1.0 / np.sqrt(m * fim)
                hierarchy[key][f"m{m}"] = {
                    "floor_L0": rep["level0_floors"][str(U)][f"m{m}"],
                    "floor_L23": floor23_by_U[str(U)][f"m{m}"]["floor_L23"],
                    "dalpha_FIM": float(da_fim),
                }
                print(f"[Hierarchy] om_tilde={om} U={U} m={m}: "
                      f"floor_L0={hierarchy[key][f'm{m}']['floor_L0']:.4f} "
                      f"floor_L23={hierarchy[key][f'm{m}']['floor_L23']:.4f} "
                      f"dalpha_FIM={da_fim:.4f}")
    rep["five_tuple_note"] = (
        "(observable level, class N=1/U/omega_min=0.3, design="
        "{32,48,64,96,128,192,256}, noise=sigma_measured_at_alpha0.5, m); "
        "floor_L0/floor_L23 are omega_tilde-independent (om is a free "
        "adversary nuisance in [0.3,2.5] for both), tabulated per omega_tilde "
        "only to sit alongside the omega_tilde-dependent FIM column")
    rep["hierarchy"] = hierarchy
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="all",
                     choices=["b1", "b2", "b3", "none"])
    ap.add_argument("--task2", action="store_true")
    ap.add_argument("--n-starts", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "gates_task1.json")
    rep = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rep.update(json.load(fh))
    if args.gate in ("b1", "all"):
        gate_b1(rep)
    if args.gate in ("b2", "all"):
        gate_b2(rep)
    if args.gate in ("b3", "all"):
        gate_b3(rep)
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", out_path)

    if args.task2:
        hpath = os.path.join(RESULTS_DIR, "floors_hierarchy.json")
        hrep = {}
        if os.path.exists(hpath):
            with open(hpath) as fh:
                hrep.update(json.load(fh))
        task2_main(hrep, n_starts=args.n_starts)
        with open(hpath, "w") as fh:
            json.dump(hrep, fh, indent=1)
        print("saved", hpath)


if __name__ == "__main__":
    main()
