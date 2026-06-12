import numpy as np
from scipy.optimize import minimize, lsq_linear

def E_N(N, da, T, U, n_grid=240, n_starts=14, seed=0):
    """min || da*x - c - sum a_i e^{-w_i x} ||_L2[0,T], |a_i|<=U, w_i>=0."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, T, n_grid)
    wgt = np.sqrt(T / n_grid)
    target = da * x

    def inner(ws):                       # bounded least squares given exponents
        cols = [np.ones_like(x)] + [np.exp(-w * x) for w in ws]
        A = np.vstack(cols).T * wgt
        b = target * wgt
        lo = np.r_[-np.inf, -U * np.ones(len(ws))]
        hi = np.r_[ np.inf,  U * np.ones(len(ws))]
        res = lsq_linear(A, b, bounds=(lo, hi))
        return np.sqrt(2 * res.cost)     # lsq_linear cost = 0.5*||r||^2

    best = np.inf
    for _ in range(n_starts):
        w0 = np.sort(rng.uniform(0.01, 3.0 / T, N))   # near-confluent region
        r = minimize(lambda lw: inner(np.exp(lw)), np.log(w0 + 1e-9),
                     method="Nelder-Mead",
                     options={"maxiter": 500, "fatol": 1e-14, "xatol": 1e-6})
        best = min(best, r.fun)
    return best

T0, U0, da0 = 2.0, 4.0, 0.1
print("== N-scaling (da=0.1, T=2, U=4; pred ratio per N: ~ da*T/U = 0.05) ==")
prev = None
for N in (1, 2, 3):
    e = E_N(N, da0, T0, U0)
    print(f"  N={N}: E={e:.3e}" + (f"  ratio={e/prev:.3f}" if prev else ""))
    prev = e
print("== U-scaling at N=2 (pred E ~ U^-2) ==")
for U in (1.0, 2.0, 4.0, 8.0):
    print(f"  U={U}: E={E_N(2, da0, T0, U):.3e}")
print("== da-scaling at N=2 (pred E ~ da^3) ==")
for da in (0.05, 0.1, 0.2, 0.4):
    print(f"  da={da}: E={E_N(2, da, T0, U0):.3e}")
print("== T-scaling at N=2 (pred E ~ T^3.5 incl sqrt(T)) ==")
for T in (1.0, 2.0, 4.0):
    print(f"  T={T}: E={E_N(2, da0, T, U0):.3e}")
