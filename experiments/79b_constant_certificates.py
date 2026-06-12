import numpy as np
from scipy.optimize import minimize
from scipy.special import comb, factorial

def construction(N, da, T, U, n_grid=400):
    """Explicit Richardson construction; returns (E, ws, amps, c)."""
    om = N * da / U                       # amplitude bound binds at i=1
    x = np.linspace(0, T, n_grid); wgt = np.sqrt(T / n_grid)
    beta = np.array([(-1)**(i-1) * comb(N, i, exact=True) for i in range(1, N+1)])
    ws = om * np.arange(1, N+1)
    amps = da * beta / ws                 # a_i = da*beta_i/(i*om)
    # model: da*x ~ c + sum a_i e^{-w_i x}; construction: sum beta_i f_i
    g = sum(a * np.exp(-w * x) for a, w in zip(amps, ws))
    # f_i = (1-e^{-iwx})/(iw): sum beta_i f_i = sum (da beta_i/(i om))(1 - e^-..)
    # = const - g  with const = sum a_i. So approximant = const_term - ... care:
    approx = sum(da * b / w * (1 - np.exp(-w * x)) for b, w in zip(beta, ws))
    r = da * x - approx
    # free constant + refit amplitudes (linear lsq, unconstrained — check bound after)
    cols = [np.ones_like(x)] + [np.exp(-w * x) for w in ws]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A * wgt, da * x * wgt, rcond=None)
    r2 = da * x - A @ coef
    ok = np.all(np.abs(coef[1:]) <= U * 1.0001)
    E = np.sqrt(np.sum(r2**2) * (T / n_grid))
    return E, ok

def optimum(N, da, T, U, n_grid=400, extra_starts=6, seed=1):
    """True optimum, seeded with the construction's exponents."""
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(seed)
    x = np.linspace(0, T, n_grid); wgt = np.sqrt(T / n_grid)
    def inner(ws):
        cols = [np.ones_like(x)] + [np.exp(-w * x) for w in ws]
        A = np.vstack(cols).T * wgt
        res = lsq_linear(A, da * x * wgt,
                         bounds=(np.r_[-np.inf, -U*np.ones(N)],
                                 np.r_[np.inf, U*np.ones(N)]))
        return np.sqrt(2 * res.cost)
    om = N * da / U
    starts = [om * np.arange(1, N+1)]
    for _ in range(extra_starts):
        starts.append(np.sort(rng.uniform(0.2, 3, N)) * om)
    best = np.inf
    for w0 in starts:
        r = minimize(lambda lw: inner(np.exp(lw)), np.log(w0),
                     method="Nelder-Mead",
                     options={"maxiter": 800, "fatol": 1e-16, "xatol": 1e-7})
        best = min(best, r.fun)
    return best

T, U = 2.0, 4.0
c_pred = {1: 0.03727, 2: 0.02519, 3: 0.03215}
print(f"{'N':>2} {'da*T/U':>8} {'E_constr':>11} {'E_opt':>11} {'c_constr':>9} {'c_opt':>9} {'c_pred':>8}")
for N in (1, 2, 3):
    for ratio in (0.05, 0.02, 0.01):
        da = ratio * U / T
        Ec, ok = construction(N, da, T, U)
        Eo = optimum(N, da, T, U)
        norm = np.sqrt(T) * U * ratio**(N+1)
        print(f"{N:>2} {ratio:>8} {Ec:>11.3e} {Eo:>11.3e} "
              f"{Ec/norm:>9.4f} {Eo/norm:>9.4f} {c_pred[N]:>8.4f}  bound_ok={ok}")
