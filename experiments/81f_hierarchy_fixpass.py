"""Exp 81f: fix-pass responding to ml_paper/EXP81_AUDIT.md.

Blockers being fixed (audit's own numbering):
1. Bisection resolution: adaptive relative-tolerance (<=5%) bisection,
   replacing the original's fixed absolute tol=1e-2 (25-90% relative error
   on the small floor_L23 cells the audit flagged).
2. Monotonicity as a gate: warm-start each (U,m) computation from the
   converged solution at the next-smaller U (box-nesting: (-U',U') subset
   (-U,U) for U>U' means a smaller-U optimum is automatically feasible at
   larger U, no reparametrization needed -- simpler than the exp81c hull-J
   case, which needed atom duplication). Assert floor nondecreasing in U as
   a hard gate (G-B6); escalate optimizer budget and retry on violation,
   never report a violating number.
3. Aligned adversary classes: a NEW Level-0 floor (floor_L0_aligned) with
   the adversary restricted to the PUSHFORWARD of the physical (nu, nu2,
   omega_tilde) family -- the SAME nuisance pair generating the Level-2/3
   spectral ensembles -- observed only through the 7-point log(W_sat)
   summary (not the full spectrum). This makes floor_L23 <= floor_L0_aligned
   a genuine data-processing-inequality validation gate (G-B7), and the
   L23/L0_aligned ratio the citable hierarchy measurement the plan asked
   for (as opposed to the original floor_L0, computed under exp77's
   UNRELATED abstract log-ladder adversary, kept only as a secondary,
   clearly-labeled comparison).
4. H-A3 (shared correction spectrum + correlated noise) K-curve re-run with
   items 1-2 in place; corrected K=5 gain reported with a convergence
   statement (G-A4).

Reuses experiments/81_fractional_ew_testbed.py and
experiments/81c_multivariate_floor.py via importlib WITHOUT modifying them
-- this file only adds new, independently-gated computations, so the
already-verified gates in those files (G-B1-B5, G-A1-A2) are untouched.
"""

import argparse
import importlib.util
import json
import os

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp81f_fixpass")
E81_PATH = os.path.join(HERE, "81_fractional_ew_testbed.py")
E81C_PATH = os.path.join(HERE, "81c_multivariate_floor.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e81 = load_module(E81_PATH, "e81_fixpass")
e81c = load_module(E81C_PATH, "e81c_fixpass")

DESIGN = e81.DESIGN
Z_BOUNDS = e81.Z_BOUNDS
OMEGA_BOUNDS = e81.OMEGA_BOUNDS
DNU_BOUNDS = e81.DNU_BOUNDS
U_GRID_SORTED = sorted(e81.U_GRID)          # (0.5, 1.0, 4.0)


########################################################################
# Part A: Level-0/Level-2-3 fix (items 1-3)
########################################################################

def log_w2_diff_gap(design, p1, p2):
    """Aligned Level-0 objective: squared log-W2 difference summed over
    the design, using the SAME physical (z,D,nu,nu2,omega_tilde) pair for
    both ensembles as Level 2/3 -- the pushforward of the physical family
    through the summary map L -> log(exact_W2(L))."""
    total = 0.0
    for L in design:
        w1 = e81.exact_W2(L, *p1)
        w2 = e81.exact_W2(L, *p2)
        d = np.log(w1) - np.log(w2)
        total += d * d
    return total


def _optimize_two_config(dalpha, U, design, objective_fn, n_starts=10, seed=0,
                          extra_starts=None, maxiter=50, maxfun=400):
    """Shared scaffolding for both the KL (Level 2/3) and log-W2-diff
    (aligned Level 0) confusion gaps: same 9-parameter adversary
    [z1,logD1,lognu1,u1,om1,logD2,lognu2,u2,om2], same nu2_from_u
    inversion, same box bounds; only `objective_fn(design,p1,p2)` differs.
    Returns (best, best_p)."""
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
        nu2_1 = e81.nu2_from_u(u1, z1, D1, nu1, om1, design=design, n_iter=24)
        nu2_2 = e81.nu2_from_u(u2, z2, D2, nu2p, om2, design=design, n_iter=24)
        return (z1, D1, nu1, nu2_1, om1), (z2, D2, nu2p, nu2_2, om2)

    def obj(p):
        p1, p2 = unpack(p)
        if not (e81.denom_ok(design, *p1) and e81.denom_ok(design, *p2)):
            return 1e6
        try:
            return objective_fn(design, p1, p2)
        except (FloatingPointError, ValueError):
            return 1e6

    starts = list(extra_starts or [])
    starts += [None] * n_starts
    best = np.inf
    best_p = None
    for p0 in starts:
        if p0 is None:
            p0 = np.array([
                rng.uniform(*Z_BOUNDS), rng.uniform(logD_lo, logD_hi),
                rng.uniform(logD_lo, logD_hi), rng.uniform(-U, U),
                rng.uniform(*OMEGA_BOUNDS),
                rng.uniform(logD_lo, logD_hi), rng.uniform(logD_lo, logD_hi),
                rng.uniform(-U, U), rng.uniform(*OMEGA_BOUNDS),
            ])
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": maxiter, "maxfun": maxfun})
        if res.fun < best:
            best = float(res.fun)
            best_p = res.x
    return best, best_p


def gap_chained_over_U(objective_fn, dalpha, U, design=DESIGN, n_starts=8,
                       seed=0, maxiter=50, maxfun=400):
    """Recursively warm-starts from the next-smaller U in U_GRID_SORTED.
    Box-nesting ((-U',U') subset (-U,U) for U>U') means the smaller-U
    optimum is automatically a feasible, guaranteed-no-worse point at
    larger U -- the warm start needs no reparametrization (unlike the
    hull-J atom-duplication case in exp81c). NOTE: an earlier version used
    a much smaller random-restart budget for chained (non-base) calls,
    reasoning the warm start alone would carry most of the convergence --
    this was WRONG: verified directly (da=0.6, U=4.0: gap=21.9 at
    n_starts_chained=3 vs gap=1.1e-6 at n_starts_chained=8, a >10^7 ratio).
    The warm-started point is feasible but often far from the U=4.0-
    specific optimum, which needs genuine exploration of the EXPANDED
    (-U,U) range that a smaller-U warm start never visits -- the local
    L-BFGS-B optimizer alone cannot reliably escape to it. So EVERY call
    (base and chained) uses the full n_starts budget; only maxiter/maxfun
    per individual optimization were reduced (50/400 vs the original
    80/700) to control wall-clock, which independently verified as NOT the
    source of the degradation. Returns (gap, best_p)."""
    idx = U_GRID_SORTED.index(U)
    extra = None
    if idx > 0:
        U_prev = U_GRID_SORTED[idx - 1]
        _, p_prev = gap_chained_over_U(objective_fn, dalpha, U_prev, design,
                                       n_starts, seed, maxiter, maxfun)
        extra = [p_prev]
    return _optimize_two_config(dalpha, U, design, objective_fn,
                                n_starts=n_starts, seed=seed,
                                extra_starts=extra, maxiter=maxiter,
                                maxfun=maxfun)


def adaptive_floor(gap_fn, thresh, da_hi=1.2, rel_tol=0.05, abs_tol_min=1e-3,
                    max_iter=40):
    """Bisection with an ADAPTIVE (relative) stopping tolerance: stops once
    the search interval width is <=rel_tol*max(lo,abs_tol_min), giving
    uniform ~5% relative precision across floors spanning 0.02-1.1 (a fixed
    absolute tol=1e-2 -- the original code's choice -- is 25-90% relative
    error at the small end, exactly what the audit flagged)."""
    lo, hi = 0.0, da_hi
    ghi = gap_fn(da_hi)
    if ghi <= thresh:
        return da_hi, ghi, True
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g = gap_fn(mid)
        if g <= thresh:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= max(rel_tol * max(lo, abs_tol_min), abs_tol_min):
            break
    return 0.5 * (lo + hi), gap_fn(0.5 * (lo + hi)), False


def floor_l23_fixed(sigma, m, U, design=DESIGN, da_hi=1.2, rel_tol=0.05,
                    n_starts=10, seed=0):
    def gap_m(da):
        g, _ = gap_chained_over_U(e81.kl_design, da, U, design, n_starts, seed)
        return m * g  # m independent seeds -> joint KL = m * per-seed KL
    return adaptive_floor(gap_m, thresh=0.5, da_hi=da_hi, rel_tol=rel_tol)


def floor_l0_aligned_fixed(sigma, m, U, design=DESIGN, da_hi=1.2, rel_tol=0.05,
                           n_starts=10, seed=0):
    def gap(da):
        g, _ = gap_chained_over_U(log_w2_diff_gap, da, U, design, n_starts, seed)
        return g
    return adaptive_floor(gap, thresh=sigma ** 2 / m, da_hi=da_hi, rel_tol=rel_tol)


def compute_hierarchy_fixed(sigma, ms=(6, 24), n_starts=10, escalation=(10, 20, 32)):
    """Compute floor_L23 and floor_L0_aligned for every U in U_GRID_SORTED
    (increasing order, so each benefits from the chain), for every m.
    G-B6: assert nondecreasing in U for BOTH floor types; on violation,
    escalate n_starts and retry (never report a violating number)."""
    results = {"escalation_used": {}}
    for n_try in escalation:
        l23 = {}
        l0a = {}
        for m in ms:
            l23[m] = {}
            l0a[m] = {}
            for U in U_GRID_SORTED:
                f23, g23, sat23 = floor_l23_fixed(sigma, m, U, n_starts=n_try)
                f0a, g0a, sat0a = floor_l0_aligned_fixed(sigma, m, U, n_starts=n_try)
                l23[m][U] = {"floor": f23, "gap_at_floor": g23, "saturated": sat23}
                l0a[m][U] = {"floor": f0a, "gap_at_floor": g0a, "saturated": sat0a}
                print(f"[fixpass n_starts={n_try}] U={U} m={m}: "
                      f"floor_L23={f23:.4f} floor_L0_aligned={f0a:.4f}")
        mono_l23 = all(
            l23[m][U_GRID_SORTED[i]]["floor"] <= l23[m][U_GRID_SORTED[i + 1]]["floor"] + 1e-9
            for m in ms for i in range(len(U_GRID_SORTED) - 1))
        mono_l0a = all(
            l0a[m][U_GRID_SORTED[i]]["floor"] <= l0a[m][U_GRID_SORTED[i + 1]]["floor"] + 1e-9
            for m in ms for i in range(len(U_GRID_SORTED) - 1))
        dp_ok = all(
            l23[m][U]["floor"] <= l0a[m][U]["floor"] + 1e-6
            for m in ms for U in U_GRID_SORTED)
        print(f"[fixpass n_starts={n_try}] G-B6(L23)={mono_l23} "
              f"G-B6(L0_aligned)={mono_l0a} G-B7(dp)={dp_ok}")
        if mono_l23 and mono_l0a and dp_ok:
            results["n_starts_final"] = n_try
            results["floor_L23"] = {str(m): {str(U): v for U, v in row.items()}
                                    for m, row in l23.items()}
            results["floor_L0_aligned"] = {str(m): {str(U): v for U, v in row.items()}
                                           for m, row in l0a.items()}
            results["gate_B6_monotone_in_U_L23"] = mono_l23
            results["gate_B6_monotone_in_U_L0_aligned"] = mono_l0a
            results["gate_B7_data_processing_L23_le_L0_aligned"] = dp_ok
            results["pass"] = True
            return results
    # exhausted escalation budget without satisfying all gates -- report
    # honestly, do not silently accept a violating number
    results["n_starts_final"] = escalation[-1]
    results["floor_L23"] = {str(m): {str(U): v for U, v in row.items()}
                            for m, row in l23.items()}
    results["floor_L0_aligned"] = {str(m): {str(U): v for U, v in row.items()}
                                   for m, row in l0a.items()}
    results["gate_B6_monotone_in_U_L23"] = mono_l23
    results["gate_B6_monotone_in_U_L0_aligned"] = mono_l0a
    results["gate_B7_data_processing_L23_le_L0_aligned"] = dp_ok
    results["pass"] = False
    return results


def task_hierarchy(rep, ms=(6, 24), escalation=(10, 20, 32)):
    sigma = e81.measure_sigma_level0()
    print(f"[fixpass] Level-0 sigma (same family/point as original exp81): {sigma:.4f}")
    res = compute_hierarchy_fixed(sigma, ms=ms, escalation=escalation)
    res["sigma"] = sigma
    res["U_grid"] = U_GRID_SORTED
    res["five_tuple_note"] = ("(observable level, class N=1/U/omega_min=0.3, "
        "design={32,48,64,96,128,192,256}, noise=sigma_measured_at_alpha0.5, m); "
        "floor_L0_aligned uses the SAME physical (z,D,nu,nu2,omega_tilde) "
        "adversary as floor_L23, observed only through the summary -- this is "
        "the citable hierarchy pair (G-B7 validated). The ORIGINAL floor_L0 "
        "(exp77's abstract log-ladder adversary, results_exp81_hierarchy/"
        "floors_hierarchy.json) is a DIFFERENT, non-nested adversary class and "
        "is not comparable to floor_L23 by data-processing; kept only as a "
        "secondary reference, not re-litigated here.")
    print(f"[fixpass] hierarchy pass={res['pass']} "
          f"(n_starts_final={res['n_starts_final']})")
    rep["hierarchy_fixed"] = res
    return res["pass"]


########################################################################
# Part B: multivariate H-A3 fix (item 4) -- K-monotone chaining (G-A4)
########################################################################

CHANNEL_B = e81c.CHANNEL_B
MV_DESIGN = e81c.DESIGN


def channel_alone_gap(da, b_k, sigma, U, design=MV_DESIGN, n_starts=8, seed=0):
    """PRIVATE mode decomposes EXACTLY under diagonal noise: with each
    channel's omega fully independent and Sigma diagonal, the K-channel
    Mahalanobis sum separates into K independent single-channel problems
    (each identical in structure to exp77's K=1 case, just with the
    channel's own b_k). A pure-nuisance channel (b_k=0) has NO forced tilt,
    so the adversary can set Delta_c_k=0 and both corrections to zero,
    giving gap=0 EXACTLY -- verified analytically, not just numerically."""
    if b_k == 0.0:
        return 0.0
    Sigma_inv = np.array([[1.0 / sigma ** 2]])
    g, _, _ = e81c.confusion_gap_multivariate(da, np.array([b_k]), Sigma_inv,
                                              U, design, shared=True,
                                              n_starts=n_starts, seed=seed)
    return g


def private_gap_exact(da, K, sigma, U, design=MV_DESIGN, n_starts=8, seed=0):
    """D^2(K) for PRIVATE mode = sum of independent per-channel gaps (exact
    decomposition -- see channel_alone_gap). Automatically, exactly
    monotonic nondecreasing in K (each added term is >=0), no chaining or
    convergence gate needed for this case."""
    return sum(channel_alone_gap(da, CHANNEL_B[k], sigma, U, design,
                                  n_starts, seed=seed + k) for k in range(K))


def extend_channel_solution(p_prev, K_prev, shared):
    """Build a valid K_prev+1-channel warm start from a converged K_prev
    solution: append a new channel with zero amplitude/offset (a valid,
    though not necessarily optimal, point -- local optimization from here
    can only improve). For 'private' mode the new channel also needs its
    own 2 new omega parameters (defaulted to the mean of the existing
    per-channel omegas); for 'shared' mode the 2 shared omegas are already
    valid unchanged."""
    if shared:
        n_om_prev = 2
    else:
        n_om_prev = 2 * K_prev
    dc_prev = p_prev[:K_prev]
    om_prev = p_prev[K_prev:K_prev + n_om_prev]
    uA_prev = p_prev[K_prev + n_om_prev:K_prev + n_om_prev + K_prev]
    uB_prev = p_prev[K_prev + n_om_prev + K_prev:K_prev + n_om_prev + 2 * K_prev]
    dc_new = np.concatenate([dc_prev, [0.0]])
    if shared:
        om_new = om_prev
    else:
        omA_prev, omB_prev = om_prev[:K_prev], om_prev[K_prev:]
        om_new = np.concatenate([omA_prev, [omA_prev.mean()],
                                 omB_prev, [omB_prev.mean()]])
    uA_new = np.concatenate([uA_prev, [0.0]])
    uB_new = np.concatenate([uB_prev, [0.0]])
    return np.concatenate([dc_new, om_new, uA_new, uB_new])


def eval_multivariate_obj(dalpha, b, Sigma_inv, design, shared, p):
    """Standalone re-evaluation of e81c.confusion_gap_multivariate's inner
    objective at a GIVEN parameter vector p (no optimization) -- used to
    guard against L-BFGS-B occasionally returning a point WORSE than its
    own starting point when cut off by maxiter/maxfun mid-descent (observed
    directly: a warm start that is analytically a stationary point -- the
    new channel's own gradient is exactly zero there, since its contribution
    to the Mahalanobis sum is exactly zero at zero amplitude -- still came
    back worse than K-1's own floor after 'optimization', which is only
    possible if the local solver overshot and was cut off before returning)."""
    K = len(b)
    x = np.log(design)
    x1 = x[0]
    n_om = 2 if shared else 2 * K
    dc = p[:K]
    om = p[K:K + n_om]
    uA = p[K + n_om:K + n_om + K]
    uB = p[K + n_om + K:K + n_om + 2 * K]
    if shared:
        omA, omB = np.full(K, om[0]), np.full(K, om[1])
    else:
        omA, omB = om[:K], om[K:]
    total = 0.0
    for xi in x:
        corrA = e81c.corr(xi, uA, omA, x1)
        corrB = e81c.corr(xi, uB, omB, x1)
        dmu = b * dalpha * xi + dc + corrA - corrB
        total += dmu @ Sigma_inv @ dmu
    return float(total)


def per_channel_min(da, b_k, omegaA, omegaB, U, design=MV_DESIGN, n_starts=2,
                    seed=0):
    """min over (dc,uA,uB) of sum_xi (b_k*da*xi+dc+corr(uA,omegaA)-
    corr(uB,omegaB))^2 -- a 3-parameter, 7-point nonlinear least squares,
    cheap and reliable with a modest multistart."""
    x = np.log(design)
    x1 = x[0]
    u_bounds = (-0.75, U)
    rng = np.random.default_rng(seed)

    def obj(p):
        dc, uA, uB = p
        corrA = e81c.corr(x, uA, omegaA, x1)
        corrB = e81c.corr(x, uB, omegaB, x1)
        dmu = b_k * da * x + dc + corrA - corrB
        return float(np.sum(dmu ** 2))

    bounds = [(None, None), u_bounds, u_bounds]
    best = np.inf
    best_x = None
    for _ in range(n_starts):
        p0 = np.array([rng.normal(0, 1), rng.uniform(*u_bounds),
                       rng.uniform(*u_bounds)])
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 50, "maxfun": 300})
        if res.fun < best:
            best = float(res.fun)
            best_x = res.x
    return best, best_x


def shared_gap_outer2d(da, K, sigma, U, design=MV_DESIGN, n_om_grid=9, seed=0):
    """EXACT (to grid + per-channel-multistart precision) shared-mode gap
    for DIAGONAL Sigma: for FIXED (omegaA,omegaB) the per-channel
    minimization separates exactly (no cross-channel coupling once the
    shared rate is fixed), so this reduces to a reliable outer 2D grid
    search over the only genuinely-coupled parameters, rather than a
    fragile (3K+2)-dimensional joint multistart -- which was verified
    directly to under-converge for K=2 (chained multistart gave floor
    0.1055 at one da, but a later K=3 call at the SAME da found gap 0.0437
    < K=2's own reported 0.0974 there, an impossible ordering given K=3
    nests K=2 exactly for a nuisance 3rd channel -- proof the K=2 solve
    itself, not the chain, was the weak link). Monotonicity in K is
    AUTOMATIC here: for any fixed omega pair, the K-channel sum is the
    (K-1)-channel sum plus one more nonnegative term, evaluated over the
    SAME omega grid for both, so min-over-grid(K) >= min-over-grid(K-1)
    always -- no chaining needed for this (uncorrelated) case at all.
    Returns (best, best_omA, best_omB, full_param_vector) where
    full_param_vector is in confusion_gap_multivariate's 'shared' p-layout,
    reusable as an exact/near-exact warm start for the correlated case."""
    oms = np.linspace(OMEGA_BOUNDS[0], OMEGA_BOUNDS[1], n_om_grid)
    best = np.inf
    best_omA = best_omB = None
    best_channel_solutions = None
    for omA in oms:
        for omB in oms:
            vals_and_x = [per_channel_min(da, CHANNEL_B[k], omA, omB, U,
                                          design, seed=seed + k)
                         for k in range(K)]
            total = sum(v for v, _ in vals_and_x) / sigma ** 2
            if total < best:
                best = total
                best_omA, best_omB = omA, omB
                best_channel_solutions = [x for _, x in vals_and_x]
    dc = np.array([s[0] for s in best_channel_solutions])
    uA = np.array([s[1] for s in best_channel_solutions])
    uB = np.array([s[2] for s in best_channel_solutions])
    full_p = np.concatenate([dc, [best_omA, best_omB], uA, uB])
    return best, best_omA, best_omB, full_p


def gap_chained_over_K(dalpha, K, Ks_sorted, sigma, rho, U, shared,
                       design=MV_DESIGN, n_starts=8, seed=0):
    """K-monotone chaining for SHARED and SHARED_CORRELATED modes (which do
    NOT decompose like private, due to the shared-omega constraint and/or
    off-diagonal noise coupling channels together). Data-processing
    argument: observing MORE channels can only help an observer distinguish
    two hypotheses (or leave it unchanged), so the TRUE D^2(K) is
    nondecreasing in K; warm-starting from a valid K-1 extension (new
    channel at zero) gives the optimizer a running start, and any observed
    violation after this fix is treated as under-convergence (escalate),
    never reported as a genuine finding."""
    idx = Ks_sorted.index(K)
    b = np.array(CHANNEL_B[:K])
    Sigma_inv = np.linalg.inv(e81c.make_sigma(K, sigma, rho=rho))
    extras = []
    if idx > 0:
        K_prev = Ks_sorted[idx - 1]
        _, p_prev = gap_chained_over_K(dalpha, K_prev, Ks_sorted, sigma, rho,
                                       U, shared, design, n_starts, seed)
        extras.append(extend_channel_solution(p_prev, K_prev, shared))
    if rho != 0.0:
        # for the CORRELATED case, also seed with the reliable outer-2D
        # solution of the UNCORRELATED ("shared", rho=0) problem at the
        # same (da,K) -- not exact here (Sigma isn't diagonal), but a much
        # better-informed starting point than blind random restarts, since
        # the underlying signal structure is identical, only the noise
        # correlation differs.
        _, _, _, p_outer2d = shared_gap_outer2d(dalpha, K, sigma, U, design,
                                                seed=seed)
        extras.append(p_outer2d)
    g, _, p = e81c.confusion_gap_multivariate(dalpha, b, Sigma_inv, U, design,
                                              shared=shared, n_starts=n_starts,
                                              seed=seed,
                                              extra_starts=extras or None)
    for e in extras:
        # guard: the optimizer must never return worse than any of its own
        # warm starts (an under-converged L-BFGS-B run can, in principle,
        # return a point worse than where it began if cut off mid-descent)
        g_e = eval_multivariate_obj(dalpha, b, Sigma_inv, design, shared, e)
        if g_e < g:
            g, p = g_e, e
    return g, p


def floor_shared_fixed(m, K, Ks_sorted, sigma, rho, U, design=MV_DESIGN,
                       da_hi=1.2, rel_tol=0.05, n_starts=8, seed=0):
    if rho == 0.0:
        # exact (to grid+multistart precision) and automatically monotone
        # in K -- no chaining needed (see shared_gap_outer2d docstring)
        def gap(da):
            g, _, _, _ = shared_gap_outer2d(da, K, sigma, U, design, seed=seed)
            return g
    else:
        def gap(da):
            g, _ = gap_chained_over_K(da, K, Ks_sorted, sigma, rho, U, True,
                                      design, n_starts, seed)
            return g
    return adaptive_floor(gap, thresh=1.0 / m, da_hi=da_hi, rel_tol=rel_tol)


def floor_private_fixed(m, K, sigma, U, design=MV_DESIGN, da_hi=1.2,
                        rel_tol=0.05, n_starts=8, seed=0):
    def gap(da):
        return private_gap_exact(da, K, sigma, U, design, n_starts, seed)
    return adaptive_floor(gap, thresh=1.0 / m, da_hi=da_hi, rel_tol=rel_tol)


def task_multivariate(rep, sigma=0.15, U=1.0, m=24, Ks=(1, 2, 3, 4, 5),
                      escalation=(8, 16, 24)):
    for n_try in escalation:
        curves = {"shared": {}, "private": {}, "shared_correlated": {}}
        for K in Ks:
            f_priv, g_priv, _ = floor_private_fixed(m, K, sigma, U,
                                                    n_starts=n_try)
            f_shared, g_shared, _ = floor_shared_fixed(m, K, list(Ks), sigma,
                                                       0.0, U, n_starts=n_try)
            f_corr, g_corr, _ = floor_shared_fixed(m, K, list(Ks), sigma,
                                                   0.8, U, n_starts=n_try)
            curves["private"][K] = f_priv
            curves["shared"][K] = f_shared
            curves["shared_correlated"][K] = f_corr
            print(f"[fixpass n_starts={n_try}] K={K}: private={f_priv:.4f} "
                  f"shared={f_shared:.4f} shared_corr={f_corr:.4f}")

        def nonincreasing(d):
            vals = [d[K] for K in Ks]
            return all(vals[i + 1] <= vals[i] + 1e-6 for i in range(len(vals) - 1))

        mono_priv = nonincreasing(curves["private"])
        mono_shared = nonincreasing(curves["shared"])
        mono_corr = nonincreasing(curves["shared_correlated"])
        print(f"[fixpass n_starts={n_try}] G-A4: private={mono_priv} "
              f"shared={mono_shared} shared_corr={mono_corr}")
        if mono_priv and mono_shared and mono_corr:
            gain_shared = curves["shared"][Ks[0]] / curves["shared"][Ks[-1]]
            gain_private = curves["private"][Ks[0]] / curves["private"][Ks[-1]]
            gain_corr = (curves["shared_correlated"][Ks[0]]
                        / curves["shared_correlated"][Ks[-1]])
            out = {
                "sigma": sigma, "U": U, "m": m, "Ks": list(Ks),
                "curves": curves, "n_starts_final": n_try,
                "gate_A4_private_nonincreasing": mono_priv,
                "gate_A4_shared_nonincreasing": mono_shared,
                "gate_A4_shared_correlated_nonincreasing": mono_corr,
                "H_A1_private_gain_K1_to_K5": gain_private,
                "H_A2_shared_gain_K1_to_K5": gain_shared,
                "H_A3_shared_correlated_gain_K1_to_K5": gain_corr,
                "convergence_statement": (
                    f"K-monotonicity (G-A4) satisfied at n_starts={n_try} "
                    "for all three curves (private: exact decomposition, "
                    "provably monotonic; shared/shared_correlated: "
                    "K-chained warm start + multi-start, empirically "
                    "monotonic after this budget)."),
                "pass": True,
            }
            rep["multivariate_fixed"] = out
            print(f"[fixpass] H-A3 corrected K=5 gain: {gain_corr:.3f}x "
                  f"(n_starts={n_try})")
            return True
    # escalation exhausted without satisfying G-A4 for all three curves
    out = {"sigma": sigma, "U": U, "m": m, "Ks": list(Ks), "curves": curves,
           "n_starts_final": escalation[-1],
           "gate_A4_private_nonincreasing": mono_priv,
           "gate_A4_shared_nonincreasing": mono_shared,
           "gate_A4_shared_correlated_nonincreasing": mono_corr,
           "convergence_statement": (
               f"G-A4 NOT satisfied after escalating to n_starts="
               f"{escalation[-1]} -- reporting curves as-is with violation "
               "flagged; per the audit's own instruction, this means the "
               "numbers below are NOT citable as converged, only as a "
               "record of the attempt."),
           "pass": False}
    rep["multivariate_fixed"] = out
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["hierarchy", "multivariate", "all"])
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fixpass.json")
    rep = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rep.update(json.load(fh))

    if args.task in ("hierarchy", "all"):
        task_hierarchy(rep)
    if args.task in ("multivariate", "all"):
        task_multivariate(rep)

    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
