"""
Independent numerical replication for the simulated human audit (2026-07-04).
Reproduces the floor theorem's headline numbers FROM THE THEORY-NOTE FORMULAS
ALONE (no exp77 code imported), as an independent cross-check.

Companion: ml_paper/HUMAN_AUDIT_SIM_2026-07-04.md
Run: python3 ml_paper/experiments/audit_2026-07-04_numeric_checks.py

Checks:
  (A) confusion gap D^2(da) on the real 7-point design + near-non-identifiability
  (A2) per-system resolution floors (D^2 <= sigma^2/m, m=24)
  (E) monotonicity of D^2 in da (validates the running-max floor definition)
  (B) closed-form modulus constant c_1 -> 0.0375 and exponent N+1 = 2
  (C) information-destruction mechanism: unbounded correction span saturates
      the 7-dim design (=> the 59x / 98.3% figure is an AMPLITUDE-BOUND effect)
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

L = np.array([32, 48, 64, 96, 128, 192, 256.])
x = np.log(L); n = len(L); L1 = L[0]
UB = (-0.75, 4.0); OB = (0.3, 2.5)          # u = B*L1^{-om} bound; omega bound
rng = np.random.default_rng(1)


def g(u, om):
    """correction on the log axis: g = log(1 + B L^{-om}), u := B L1^{-om}."""
    return np.log1p(u * L1**om * L**(-om))


def D2(da, nstart=12):
    """confusion gap: min over (c, eta1, eta2) of || da*x + c + g1 - g2 ||^2."""
    best = np.inf
    for _ in range(nstart):
        p0 = [rng.uniform(-.5, .5), rng.uniform(*UB), rng.uniform(*OB),
              rng.uniform(*UB), rng.uniform(*OB)]
        def obj(p):
            c, u1, o1, u2, o2 = p
            r = da*x + c + g(u1, o1) - g(u2, o2)
            return r @ r
        res = minimize(obj, p0, method='L-BFGS-B',
                       bounds=[(-5, 5), UB, OB, UB, OB])
        best = min(best, res.fun)
    return best


def E1(da, T, U, ngrid=4000):
    """closed-form modulus, N=1: min_{c,|b|<=U,w>0} ||da*t - c - b e^{-w t}||_L2[0,T]."""
    t = np.linspace(0, T, ngrid); f = da * t
    def resid(w):
        e = np.exp(-w * t)
        A = np.column_stack([np.ones_like(t), e])
        c, b = np.linalg.lstsq(A, f, rcond=None)[0]
        if abs(b) > U:                       # clamp amplitude, refit intercept
            b = np.sign(b) * U; c = np.mean(f - b * e)
        r = f - c - b * e
        return np.sqrt(np.trapezoid(r * r, t))
    return minimize_scalar(resid, bounds=(1e-3, 50), method='bounded').fun


if __name__ == "__main__":
    grid = [0.05, 0.10, 0.14, 0.20, 0.27, 0.35, 0.44, 0.55]
    D = {d: D2(d) for d in grid}
    print("(A) confusion gap on real 7-pt design (note: ~1.3e-7 @ 0.10):")
    for d in grid:
        print(f"    D2({d:.2f}) = {D[d]:.3e}")
    print("(E) D2 monotone increasing:",
          all(D[grid[i+1]] >= D[grid[i]] - 1e-12 for i in range(len(grid)-1)))

    print("(A2) floors (largest grid da with D2 <= sigma^2/m, m=24):")
    for nm, s in [("BD", 0.019), ("Eden", 0.138), ("EW", 0.145), ("KPZ", 0.154)]:
        thr = s**2 / 24
        below = [d for d in grid if D[d] <= thr]
        print(f"    {nm:4s} sigma={s} thr={thr:.2e} floor>= {max(below) if below else float('nan')}")

    print("(B) closed-form modulus constant, T=2 U=4, da*T/U -> 0:")
    T, U = 2.0, 4.0
    for da in [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625]:
        e = E1(da, T, U); c1 = e / (np.sqrt(T) * U * (da*T/U)**2)
        print(f"    daT/U={da*T/U:.4f}  E1={e:.3e}  c1={c1:.4f}")
    das = [0.1, 0.05, 0.025, 0.0125]
    p = np.polyfit(np.log(das), np.log([E1(d, T, U) for d in das]), 1)[0]
    print(f"    fitted exponent dlogE1/dlog(da) = {p:.3f}  (theory N+1 = 2)")

    print("(C) unbounded-correction span rank on 7 pts (=> 100% loss without U bound):")
    oms = np.linspace(0.3, 2.5, 60)
    V = np.column_stack([np.ones(n)] + [L**(-o) for o in oms])
    r = int((np.linalg.svd(V, compute_uv=False) > 1e-10).sum())
    print(f"    rank(span{{1, L^-omega}}) = {r} of {n}  -> the 59x/98.3% figure is set by U, not projection")
