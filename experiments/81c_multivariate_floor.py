"""Exp 81 Part A, Task 4: multivariate (Level-1) Le Cam confusion gap.

Companion: ml_paper/EXP81_PLAN.md Part A, THEORY_minimax_floor.md Appendix F1.

Estimator observes K channels y_k(L_i) = c_k + theta_k(alpha)*x_i +
g(L_i; u_{k}, omega) + noise, noise ~ N(0,Sigma) across channels,
independent across L and seed. Per-channel correction uses the SAME
log-form as exp77 (g(L;u,w) = log(1+u*(L/L1)^-w), u = correction size at
L1) rather than the appendix F1 schematic's bare linear exponential --
this is required for the K=1 reduction to literally coincide with exp77's
confusion_gap (G-A1's anchor check), and keeps the bounds identical
(U_BOUNDS=(-0.75,U), OMEGA_BOUNDS=(0.3,2.5), matching exp77/Level-0
exactly rather than a symmetric (-U,U) box).

For theta_k(alpha) LINEAR in alpha (theta_k = a_k + b_k*alpha, true for
both channel maps used here: theta=alpha has b=1, theta=-(1+2*alpha) has
b=-2, theta=0 has b=0), the mean difference at design point x_i, after
allowing each of the two competing configurations ("A","B") its own free
per-channel correction amplitude and (shared or private) correction rate,
is

    Delta_mu_k(x_i) = b_k*Delta_alpha*x_i + Delta_c_k
                      + g(x_i; u_{k,A}, omega_A,k) - g(x_i; u_{k,B}, omega_B,k)

(private case: omega_A,k, omega_B,k free per channel; shared case: a single
omega_A shared across all K channels of config A, and likewise omega_B for
config B -- "the same irrelevant operators feed every observable"). The
confusion gap is

    D^2(Delta_alpha) = min_{Delta_c, {corrections}} sum_i
                        Delta_mu(x_i)^T Sigma^{-1} Delta_mu(x_i),

whitened by Sigma so the floor criterion is exactly D^2 <= 1/m (matching
exp77's sigma^2/m after whitening by a single sigma at K=1 -- see G-A1).
"""

import argparse
import importlib.util
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp81_hierarchy")
E77_PATH = os.path.join(HERE, "77_minimax_floor.py")

DESIGN = np.array([32., 48., 64., 96., 128., 192., 256.])
OMEGA_BOUNDS = (0.3, 2.5)
U_BOUNDS = (-0.75, None)   # upper set per-call to the declared U; matches exp77
N_STARTS = 24


def corr(x, u, w, x1):
    """exp77's log-form correction, g(L;u,w)=log(1+u*(L/L1)^-w), in log-L
    coordinates: x=log(L), x1=log(L1)."""
    return np.log(1.0 + u * np.exp(-w * (x - x1)))

# channel exponent maps used to build the K=1..5 progression (plan: "theta_k
# grid including theta=alpha, theta=-(1+2*alpha), and two pure-nuisance
# channels theta=0"); 5th channel repeats the width map (another observable
# sharing the same exponent, independent noise/correction realization).
CHANNEL_B = [1.0, -2.0, 0.0, 0.0, 1.0]   # d(theta_k)/d(alpha)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_sigma(K, sigma, rho=0.0):
    return (sigma ** 2) * ((1 - rho) * np.eye(K) + rho * np.ones((K, K)))


def confusion_gap_multivariate(dalpha, b, Sigma_inv, U, design=DESIGN,
                                shared=True, n_starts=N_STARTS, seed=0,
                                extra_starts=None):
    """b: (K,) array of d(theta_k)/d(alpha). Returns (best_D2, scatter,
    best_p). extra_starts: optional list of additional p0 vectors (already
    in this call's parametrization) to seed the multi-start with -- used to
    warm-start the 'private' (more free parameters) search from a converged
    'shared' solution, since private strictly nests shared (replicating a
    shared omega across channels is a valid private point) and MUST attain
    D^2 <= the shared optimum; without warm-starting, the higher-dimensional
    private search can under-converge with a fixed budget and spuriously
    report a WORSE (higher) D^2 than shared, inverting the true ordering."""
    K = len(b)
    x = np.log(design)
    rng = np.random.default_rng(seed)

    if shared:
        n_om = 2  # omega_A, omega_B (each shared across channels)
    else:
        n_om = 2 * K  # omega_A_k, omega_B_k per channel

    def unpack(p):
        dc = p[:K]
        om = p[K:K + n_om]
        uA = p[K + n_om:K + n_om + K]
        uB = p[K + n_om + K:K + n_om + 2 * K]
        if shared:
            omA = np.full(K, om[0])
            omB = np.full(K, om[1])
        else:
            omA = om[:K]
            omB = om[K:]
        return dc, omA, omB, uA, uB

    x1 = x[0]

    def obj(p):
        dc, omA, omB, uA, uB = unpack(p)
        total = 0.0
        for xi in x:
            corrA = corr(xi, uA, omA, x1)
            corrB = corr(xi, uB, omB, x1)
            dmu = b * dalpha * xi + dc + corrA - corrB
            total += dmu @ Sigma_inv @ dmu
        return total

    u_bounds = (U_BOUNDS[0], U)
    bounds = ([(None, None)] * K + [OMEGA_BOUNDS] * n_om
              + [u_bounds] * K + [u_bounds] * K)
    best = np.inf
    best_p = None
    scatter = []
    starts = list(extra_starts or [])
    starts += [None] * n_starts  # None => draw a random p0 below
    for p0 in starts:
        if p0 is None:
            p0 = np.concatenate([
                rng.normal(0, 1.0, size=K),
                rng.uniform(*OMEGA_BOUNDS, size=n_om),
                rng.uniform(*u_bounds, size=K),
                rng.uniform(*u_bounds, size=K),
            ])
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 150, "maxfun": 800})
        scatter.append(float(res.fun))
        if res.fun < best:
            best = float(res.fun)
            best_p = res.x
    return best, scatter, best_p


def shared_solution_as_private_start(p_shared, K):
    """Expand a converged shared-parametrization optimum (dc(K), omA, omB,
    uA(K), uB(K)) into a private-parametrization p0 (dc(K), omA_k(K),
    omB_k(K), uA(K), uB(K)) by replicating the shared omegas across
    channels -- a private point with IDENTICAL objective value."""
    dc = p_shared[:K]
    omA, omB = p_shared[K], p_shared[K + 1]
    uA = p_shared[K + 2:K + 2 + K]
    uB = p_shared[K + 2 + K:K + 2 + 2 * K]
    return np.concatenate([dc, np.full(K, omA), np.full(K, omB), uA, uB])


def floor_multivariate(m, b, Sigma_inv, U, design=DESIGN, shared=True,
                        da_hi=1.2, tol=1e-2, n_starts=N_STARTS, seed=0,
                        warm_start_from_shared=False):
    """max{Delta_alpha : min D^2 <= 1/m} via bisection. If shared=False and
    warm_start_from_shared=True, every evaluation is additionally seeded
    with the converged shared-case optimum at that dalpha (expanded into
    the private parametrization), guaranteeing D2_private <= D2_shared
    always holds (private strictly nests shared)."""
    thresh = 1.0 / m
    K = len(b)

    def gap(da):
        extra = None
        if warm_start_from_shared and not shared:
            g_shared, _, p_shared = confusion_gap_multivariate(
                da, b, Sigma_inv, U, design, shared=True,
                n_starts=n_starts, seed=seed)
            extra = [shared_solution_as_private_start(p_shared, K)]
        g, _, _ = confusion_gap_multivariate(da, b, Sigma_inv, U, design,
                                              shared, n_starts, seed,
                                              extra_starts=extra)
        return g

    lo, hi = 0.0, da_hi
    if gap(da_hi) <= thresh:
        return da_hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if gap(mid) <= thresh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ G-A1 --

def gate_a1(rep):
    """K=1 reduction reproduces the exp77 exact floor. Anchor: BD, m=24,
    floor 0.27 (results_exp77_minimax_floor/floor.json)."""
    m77 = load_module(E77_PATH, "m77_ga1")
    sigma_bd = m77.measured_sigma()["bd"]
    Sigma_inv = np.array([[1.0 / sigma_bd ** 2]])
    b = np.array([1.0])
    f_shared = floor_multivariate(24, b, Sigma_inv, U=4.0, shared=True)
    f_private = floor_multivariate(24, b, Sigma_inv, U=4.0, shared=False,
                                    warm_start_from_shared=True)
    anchor = 0.27
    tol_achieved = max(abs(f_shared - anchor), abs(f_private - anchor))
    passed = tol_achieved < 0.03
    rep["gate_a1"] = {
        "sigma_bd": sigma_bd, "K": 1, "m": 24, "U": 4.0,
        "floor_K1_shared_param": f_shared, "floor_K1_private_param": f_private,
        "anchor_exp77_bd_m24": anchor, "abs_diff": tol_achieved,
        "tolerance_achieved": tol_achieved, "pass": bool(passed),
    }
    print(f"[G-A1] K=1 floor(shared-param)={f_shared:.4f} "
          f"floor(private-param)={f_private:.4f} anchor={anchor} "
          f"diff={tol_achieved:.4f} pass={passed}")
    return passed


# ------------------------------------------------------------------ G-A2 --

def gate_a2(rep, sigma=0.15, U=1.0, ms=(24,), Ks=(1, 2, 3, 4, 5)):
    """floor-vs-K curves: (i) shared correction spectrum, (ii) private
    per-channel corrections, (iii) shared spectrum + correlated noise
    (Sigma off-diagonal 0.8)."""
    out = {"sigma": sigma, "U": U, "channel_b": CHANNEL_B}
    for m in ms:
        curves = {"shared": {}, "private": {}, "shared_correlated": {}}
        for K in Ks:
            b = np.array(CHANNEL_B[:K])
            Sigma0 = make_sigma(K, sigma, rho=0.0)
            Sigma_corr = make_sigma(K, sigma, rho=0.8)
            f_shared = floor_multivariate(m, b, np.linalg.inv(Sigma0), U,
                                           shared=True)
            f_private = floor_multivariate(m, b, np.linalg.inv(Sigma0), U,
                                            shared=False,
                                            warm_start_from_shared=True)
            f_shared_corr = floor_multivariate(m, b, np.linalg.inv(Sigma_corr),
                                                U, shared=True)
            curves["shared"][str(K)] = f_shared
            curves["private"][str(K)] = f_private
            curves["shared_correlated"][str(K)] = f_shared_corr
            print(f"[G-A2] m={m} K={K}: shared={f_shared:.4f} "
                  f"private={f_private:.4f} shared_corr(rho=0.8)={f_shared_corr:.4f}")
        out[f"m{m}"] = curves
    # H-A2 check: shared should out-perform (lower floor / better
    # resolution) private as K grows, beyond quadrature (~1/sqrt(K))
    m0 = ms[0]
    shared_K1 = out[f"m{m0}"]["shared"]["1"]
    shared_K5 = out[f"m{m0}"]["shared"][str(Ks[-1])]
    private_K1 = out[f"m{m0}"]["private"]["1"]
    private_K5 = out[f"m{m0}"]["private"][str(Ks[-1])]
    quad_pred = shared_K1 / np.sqrt(Ks[-1])
    rep["gate_a2"] = {
        "curves": out,
        "H_A2_shared_gain_K1_to_K5": shared_K1 / shared_K5 if shared_K5 else None,
        "H_A1_private_gain_K1_to_K5": private_K1 / private_K5 if private_K5 else None,
        "quadrature_prediction_K5": quad_pred,
        "pass": True,
    }
    print(f"[G-A2] shared gain K1->K5: {shared_K1:.4f}->{shared_K5:.4f} "
          f"private gain: {private_K1:.4f}->{private_K5:.4f} "
          f"quadrature_pred(K5)={quad_pred:.4f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="all", choices=["a1", "a2", "all"])
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "multivariate_floor.json")
    rep = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rep.update(json.load(fh))
    if args.gate in ("a1", "all"):
        gate_a1(rep)
    if args.gate in ("a2", "all"):
        gate_a2(rep)
    rep["note_g_a3"] = ("Measured-Sigma pilot (adapting 76b_regenerate_"
                         "ladders.py, EW+KPZ) NOT run -- optional per plan; "
                         "synthetic-channel gates above are the primary "
                         "deliverable. See EXP81_REPORT.md 'what we did not do'.")
    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
