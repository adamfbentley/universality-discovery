"""
Independent replication for REVIEW_2026-07-06.md (senior-researcher adversarial
pass). Reproduces every load-bearing number FROM THE STATED FORMULAS ONLY
(THEORY_minimax_floor.md, EXP86_REPORT.md, CLAIMS_REGISTER.md). Imports no
exp77/79/86 pipeline code; uses a different optimizer config / seed / more starts
than audit_2026-07-04_numeric_checks.py.

Checks:
  1. confusion gap D^2(da) on the real 7-point design; monotonicity; seed counts
  2. per-system floors, m=24 (largest da with D2 <= sigma^2/m)
  3. closed-form modulus constant c1 -> 0.0375 and exponent N+1 = 2
  4. N1 affine-center estimator bug (Sum w = -0.9986, w.logL = -2.85) + the
     minimum-norm constrained fix (Sum w = 0, Sum w*logL = 1; amplitude-invariant)
  5. BD half-window amplitude-contamination arithmetic (exp85c Task 2 diagnosis)

Run: python3 ml_paper/experiments/review_2026-07-06_checks.py
NOTE: check 1+2 are optimizer-heavy; run sections separately if time-limited.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar

rng = np.random.default_rng(20260706)

# --- design + measured noise (floor.json sigma_measured) -------------------
L7 = np.array([32, 48, 64, 96, 128, 192, 256.]); x7 = np.log(L7); L1 = L7[0]
UB = (-0.75, 4.0); OB = (0.3, 2.5)
SIG = dict(bd=0.018977, eden=0.137817, ew=0.145393, kpz=0.154140)


def g_corr(u, om, L):
    """log-axis correction g = log(1 + B L^{-om}), u := B L1^{-om}."""
    return np.log1p(u * (L1**om) * L**(-om))


def D2(da, L=L7, nstart=40):
    """confusion gap: min_{c,u1,o1,u2,o2} || da*logL + c + g1 - g2 ||^2."""
    xx = np.log(L); best = np.inf
    for _ in range(nstart):
        p0 = [rng.uniform(-.5, .5), rng.uniform(*UB), rng.uniform(*OB),
              rng.uniform(*UB), rng.uniform(*OB)]
        def obj(p):
            c, u1, o1, u2, o2 = p
            r = da*xx + c + g_corr(u1, o1, L) - g_corr(u2, o2, L)
            return r @ r
        r = minimize(obj, p0, method='L-BFGS-B', bounds=[(-5, 5), UB, OB, UB, OB],
                     options=dict(maxiter=500, ftol=1e-15, gtol=1e-12))
        best = min(best, r.fun)
    return best


def E1(da, T, U, ngrid=8000):
    """closed-form modulus, N=1: ||da*t - c - b e^{-w t}||_L2[0,T], |b|<=U."""
    t = np.linspace(0, T, ngrid); f = da*t
    def resid(w):
        e = np.exp(-w*t)
        A = np.column_stack([np.ones_like(t), e])
        c, b = np.linalg.lstsq(A, f, rcond=None)[0]
        if abs(b) > U:
            b = np.sign(b)*U; c = np.mean(f - b*e)
        r = f - c - b*e
        return np.sqrt(np.trapezoid(r*r, t))
    return minimize_scalar(resid, bounds=(1e-4, 80), method='bounded',
                           options=dict(xatol=1e-10)).fun


def check_1_2():
    grid = [0.02, 0.05, 0.10, 0.14, 0.20, 0.25, 0.27, 0.35, 0.43, 0.44, 0.45, 0.50]
    D = {d: D2(d) for d in grid}
    print("CHECK 1: confusion gap, real 7-point design")
    for d in grid:
        print(f"   D2({d:.2f}) = {D[d]:.3e}")
    print(f"   D2(0.10) = {D[0.10]:.3e}   [note 1.3e-7]")
    print("   monotone:", all(D[grid[i+1]] >= D[grid[i]]-1e-13
                              for i in range(len(grid)-1)))
    print(f"   seeds da=0.1: EW {SIG['ew']**2/D[0.10]:.2e}  BD {SIG['bd']**2/D[0.10]:.2e}")
    print("CHECK 2: per-system floors, m=24")
    for nm in ['bd', 'eden', 'ew', 'kpz']:
        thr = SIG[nm]**2/24
        below = [d for d in grid if D[d] <= thr]
        print(f"   {nm:5s} thr={thr:.2e} floor>= {max(below) if below else 'nan'}")
    print("   [claim BD 0.27, EW/KPZ/Eden 0.44]")


def check_3():
    print("CHECK 3: closed-form modulus constant c1 + exponent (T=2,U=4)")
    T, U = 2.0, 4.0
    for da in [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]:
        e = E1(da, T, U)
        print(f"   daT/U={da*T/U:.5f} E1={e:.4e} c1={e/(np.sqrt(T)*U*(da*T/U)**2):.5f}")
    das = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    sl = np.polyfit(np.log(das), np.log([E1(d, T, U) for d in das]), 1)[0]
    print(f"   fitted exponent={sl:.4f} [theory 2];  [claim c1->0.0375]")


def check_4_5():
    print("CHECK 4: affine-center estimator bug + fix, design {32,48,64,96}")
    L4 = np.array([32, 48, 64, 96.]); x4 = np.log(L4)
    w_bug = np.array([2.33076, -3.64820, -4.28174, 4.60060])
    print(f"   BUG sum(w)={w_bug.sum():.4f} [-0.9986]  w.logL={w_bug@x4:.4f} [-2.8537]")
    A = np.vstack([np.ones_like(x4), x4]); b = np.array([0., 1.])
    w_fix = A.T @ np.linalg.solve(A @ A.T, b)      # minimum-norm constrained soln
    print(f"   FIX sum(w)={w_fix.sum():.10f} [0]  w.logL={w_fix@x4:.10f} [1]")
    print(f"   FIX resp to +7.3 const = {w_fix@np.full(4, 7.3):.2e} (need 0)")
    print(f"   FIX resp to 0.5*logL   = {w_fix@(0.5*x4):.6f} (need 0.5)")
    print("CHECK 5: BD half-window amplitude-contamination arithmetic")
    sumw, center0, c_req = -0.9986, -2.2743, -1.5609
    print(f"   center0 + sum(w)*c_req = {center0 + sumw*c_req:.4f} [observed -0.7156]")


if __name__ == "__main__":
    check_4_5()   # instant
    check_3()     # moderate
    check_1_2()   # optimizer-heavy; run last / separately if time-limited
