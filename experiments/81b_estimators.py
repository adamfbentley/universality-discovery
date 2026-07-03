"""Exp 81 Part B, Task 3: trained estimators (summary MLP + raw-field CNN)
vs the Level-0 / Level-2-3 floors from experiments/81_fractional_ew_testbed.py.

Sampling prior (EXP81_PLAN.md "Sampling prior"): alpha ~ U[0.35,0.65],
log D, log nu ~ U(decade) INDEPENDENTLY of alpha, correction amplitude
u := b(L_min) ~ U[-U,U], omega_tilde ~ U[0.3,2.5]. Independence of alpha
from the nuisances is asserted in code (batch correlation check) -- this is
the defense against the discriminative-vs-universal failure mode from
exp62-71 (a network estimating alpha from amplitude cues rather than shape).

Everything here is vectorized across the batch (including the u->nu2
bisection inversion) so that ~1e6 fresh synthetic samples can be streamed
through training in a couple of minutes on CPU.
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(HERE, "..", "results_exp81_hierarchy")
E81_PATH = os.path.join(HERE, "81_fractional_ew_testbed.py")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e81 = load_module(E81_PATH, "e81_core")

DESIGN = e81.DESIGN
ALPHA_BOUNDS = e81.ALPHA_BOUNDS
OMEGA_BOUNDS = e81.OMEGA_BOUNDS
DNU_BOUNDS = e81.DNU_BOUNDS
L_MIN = e81.L_MIN


# ------------------------------------------------------ vectorized prior --

def batch_nu2_from_u(u, z, D, nu, om, design=DESIGN, n_iter=24):
    """Vectorized version of e81.nu2_from_u: all inputs are (B,) arrays."""
    kmax_global = max(e81.k_grid(L)[1].max() for L in design)
    kom_max = kmax_global ** om
    nu2_floor = -nu / kom_max * 0.999
    _, k = e81.k_grid(L_MIN)  # (M,)

    def b_of_r(nu2):
        denom_full = (nu[:, None] * k[None, :] ** z[:, None]
                      + nu2[:, None] * k[None, :] ** (z[:, None] + om[:, None]))
        S_full = D[:, None] / denom_full
        w_full = 2.0 * np.sum(S_full, axis=1) / L_MIN
        S_lead = D[:, None] / (nu[:, None] * k[None, :] ** z[:, None])
        w_lead = 2.0 * np.sum(S_lead, axis=1) / L_MIN
        return w_full / w_lead - 1.0

    pos = u >= 0
    lo = np.where(pos, nu2_floor, 0.0)
    hi = np.where(pos, 0.0, nu * 4.0)
    for _ in range(20):
        b_hi = b_of_r(hi)
        mask = (~pos) & (b_hi > u) & (hi < nu * 1e6)
        if not np.any(mask):
            break
        hi = np.where(mask, hi * 2.0, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        go_lo = b_of_r(mid) > u
        lo = np.where(go_lo, mid, lo)
        hi = np.where(go_lo, hi, mid)
    return 0.5 * (lo + hi)


def batch_sample_field(L, z, D, nu, nu2, om, rng):
    """All params (B,); returns h (B,L) and w2 (B,)."""
    _, k = e81.k_grid(L)
    B = len(z)
    denom = (nu[:, None] * k[None, :] ** z[:, None]
             + nu2[:, None] * k[None, :] ** (z[:, None] + om[:, None]))
    S = D[:, None] / denom
    Sigma = L * S
    re = rng.normal(0.0, 1.0, size=Sigma.shape) * np.sqrt(Sigma / 2.0)
    im = rng.normal(0.0, 1.0, size=Sigma.shape) * np.sqrt(Sigma / 2.0)
    c = re + 1j * im
    A = np.zeros((B, int(L) // 2 + 1), dtype=complex)
    A[:, 1:c.shape[1] + 1] = c
    h = np.fft.irfft(A, n=int(L), axis=-1)
    w2 = np.mean(h ** 2, axis=-1)
    return h, w2


def sample_prior_batch(B, U, rng, nu2_zero=False):
    """Returns dict with alpha, z, D, nu, u, om, nu2 (B,) arrays, all iid,
    nuisances independent of alpha by construction."""
    alpha = rng.uniform(*ALPHA_BOUNDS, size=B)
    z = 1.0 + 2.0 * alpha
    logD = rng.uniform(np.log(DNU_BOUNDS[0]), np.log(DNU_BOUNDS[1]), size=B)
    lognu = rng.uniform(np.log(DNU_BOUNDS[0]), np.log(DNU_BOUNDS[1]), size=B)
    D = np.exp(logD)
    nu = np.exp(lognu)
    om = rng.uniform(*OMEGA_BOUNDS, size=B)
    if nu2_zero:
        u = np.zeros(B)
        nu2 = np.zeros(B)
    else:
        u = rng.uniform(-U, U, size=B)
        nu2 = batch_nu2_from_u(u, z, D, nu, om)
    return {"alpha": alpha, "z": z, "D": D, "nu": nu, "u": u, "om": om,
            "nu2": nu2}


def assert_prior_independence(prior, tol=0.08):
    """NOTE: only reliable for large samples (tol should be several times
    1/sqrt(B) under the true null); called once on a dedicated large batch
    by verify_prior_independence_once(), not on small per-step batches."""
    alpha = prior["alpha"]
    for key in ("u", "om", "D", "nu"):
        v = prior[key]
        if np.std(v) < 1e-12:
            continue
        corr = float(np.corrcoef(alpha, v)[0, 1])
        assert abs(corr) < tol, f"prior leakage: corr(alpha,{key})={corr:.3f}"
    return True


def verify_prior_independence_once(U, n=20000, seed=12345):
    """One dedicated large-sample check (independent of training batch
    size) that the sampling prior's nuisances are drawn independently of
    alpha -- the defense against the discriminative-vs-universal leakage
    failure mode (exp62-71)."""
    rng = np.random.default_rng(seed)
    prior = sample_prior_batch(n, U, rng, nu2_zero=False)
    assert_prior_independence(prior, tol=0.03)
    print(f"[prior independence] verified on n={n} samples at U={U}: "
          + ", ".join(f"corr(alpha,{k})="
                       f"{np.corrcoef(prior['alpha'], prior[k])[0, 1]:.4f}"
                       for k in ("u", "om", "D", "nu")))


def build_batch(B, U, rng, design=DESIGN, nu2_zero=False, want_fields=False):
    prior = sample_prior_batch(B, U, rng, nu2_zero=nu2_zero)
    log_w2 = np.zeros((B, len(design)), dtype=np.float32)
    fields = [] if want_fields else None
    for i, L in enumerate(design):
        h, w2 = batch_sample_field(int(L), prior["z"], prior["D"], prior["nu"],
                                    prior["nu2"], prior["om"], rng)
        log_w2[:, i] = np.log(w2).astype(np.float32)
        if want_fields:
            fields.append(h.astype(np.float32))
    return prior, log_w2, fields


########################################################################
# Models
########################################################################

def get_torch():
    import torch
    return torch


def train_summary_mlp(U, n_steps, batch_size, seed, nu2_zero=False,
                       lr=1e-3, hidden=64):
    torch = get_torch()
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = nn.Sequential(
        nn.Linear(len(DESIGN), hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_hist = []
    for step in range(n_steps):
        prior, log_w2, _ = build_batch(batch_size, U, rng, nu2_zero=nu2_zero)
        x = torch.from_numpy(log_w2)
        y = torch.from_numpy(prior["alpha"].astype(np.float32)).unsqueeze(1)
        pred = net(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.item()))
    return net, loss_hist


def eval_summary_mlp(net, U, n_test, seed, nu2_zero=False):
    torch = get_torch()
    rng = np.random.default_rng(seed + 999999)
    prior, log_w2, _ = build_batch(n_test, U, rng, nu2_zero=nu2_zero)
    with torch.no_grad():
        pred = net(torch.from_numpy(log_w2)).numpy().ravel()
    rmse = float(np.sqrt(np.mean((pred - prior["alpha"]) ** 2)))
    return rmse, pred, prior


class PerLEncoder:
    """Builds one small 1D-conv encoder per design size L, global-average
    pooled, concatenated into a shared MLP head -- 'per-L encoder, pooled
    head' as specified in the plan."""

    def __init__(self, design, feat=12, seed=0):
        torch = get_torch()
        import torch.nn as nn
        torch.manual_seed(seed)
        self.design = design
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(8, feat, kernel_size=5, padding=2), nn.ReLU(),
            ) for _ in design
        ])
        self.head = nn.Sequential(
            nn.Linear(feat * len(design), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.params = list(self.encoders.parameters()) + list(self.head.parameters())

    def forward(self, fields):
        torch = get_torch()
        pooled = []
        for enc, h in zip(self.encoders, fields):
            x = torch.from_numpy(h).unsqueeze(1)  # (B,1,L)
            f = enc(x)                             # (B,feat,L)
            pooled.append(f.mean(dim=-1))           # (B,feat)
        cat = torch.cat(pooled, dim=1)
        return self.head(cat)


def train_cnn(U, n_steps, batch_size, seed, nu2_zero=False, lr=1e-3):
    torch = get_torch()
    rng = np.random.default_rng(seed + 5)
    model = PerLEncoder(DESIGN, seed=seed)
    opt = torch.optim.Adam(model.params, lr=lr)
    loss_hist = []
    for step in range(n_steps):
        prior, _, fields = build_batch(batch_size, U, rng, nu2_zero=nu2_zero,
                                        want_fields=True)
        y = torch.from_numpy(prior["alpha"].astype(np.float32)).unsqueeze(1)
        pred = model.forward(fields)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.item()))
    return model, loss_hist


def eval_cnn(model, U, n_test, seed, nu2_zero=False):
    torch = get_torch()
    rng = np.random.default_rng(seed + 999998)
    prior, _, fields = build_batch(n_test, U, rng, nu2_zero=nu2_zero,
                                    want_fields=True)
    with torch.no_grad():
        pred = model.forward(fields).numpy().ravel()
    rmse = float(np.sqrt(np.mean((pred - prior["alpha"]) ** 2)))
    return rmse, pred, prior


########################################################################
# Driver
########################################################################

def fim_rmse_bound(U, om=1.0, z_center=2.0, ms=None):
    """Raw-field FIM ceiling (fair reference for the CNN, which consumes
    the raw fields): RMSE-scale reference ~ dalpha_FIM at m=1 (single field
    per L, no seed averaging -- matches the single-realization training
    regime), using the representative nu2 at the declared class boundary."""
    nu2_rep = e81.nu2_from_u(U, z_center, 1.0, 1.0, om)
    fim = e81.fisher_alpha_design(DESIGN, z_center, 1.0, 1.0, nu2_rep, om)
    return 1.0 / np.sqrt(fim), fim


def summary_fim_bound(U, om=1.0, z_center=2.0, D=1.0, nu=1.0, n_seeds=600,
                       seed=777):
    """Fair reference for the SUMMARY MLP: the 7-point log-W_sat ladder is
    a LOSSY function of the raw field even at nu2=0 (it discards mode-by-
    mode shape, keeping only each L's aggregate width), so the raw-field
    FIM is not achievable by any estimator that only sees this summary.
    Uses the same independent-per-L Gaussian ansatz as
    THEORY_minimax_floor.md/exp77 (delta-method: Fisher info of alpha from
    a vector of independent approximately-Gaussian log-W2(L) with measured
    per-L sigma and analytic per-L d(log W2)/d(alpha))."""
    nu2_rep = e81.nu2_from_u(U, z_center, D, nu, om)
    dz = 1e-4
    rng = np.random.default_rng(seed)
    I_summary = 0.0
    for L in DESIGN:
        wp = e81.exact_W2(L, z_center + dz, D, nu, nu2_rep, om)
        wm = e81.exact_W2(L, z_center - dz, D, nu, nu2_rep, om)
        dlogw_dalpha = (np.log(wp) - np.log(wm)) / dz
        logw = [np.log(np.mean(
            e81.sample_field(int(L), z_center, D, nu, nu2_rep, om, rng)[0] ** 2))
            for _ in range(n_seeds)]
        sigma_L = np.std(logw, ddof=1)
        I_summary += (dlogw_dalpha / sigma_L) ** 2
    return 1.0 / np.sqrt(I_summary), I_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "estimators.json")
    rep = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rep.update(json.load(fh))
    rep["training_config"] = {
        "n_steps": args.n_steps, "batch_size": args.batch_size,
        "n_test": args.n_test,
        "prior": "alpha~U[0.35,0.65] indep of log D,log nu~U(decade), "
                 "u:=b(L_min)~U[-U,U], omega_tilde~U[0.3,2.5]",
    }
    verify_prior_independence_once(1.0)

    # ---------------- G-B5: nu2=0 easy regime ----------------
    # Two DIFFERENT ceilings apply: the raw-field FIM is the fair reference
    # for the CNN (which consumes raw fields, so can in principle approach
    # it); the 7-point log-W_sat summary is a LOSSY function of the raw
    # field even at nu2=0 (it discards mode-by-mode shape), so the fair
    # reference for the summary MLP is the summary-level Fisher ceiling,
    # which is intrinsically worse (larger) than the raw-field FIM -- using
    # the raw FIM for the MLP would be comparing it to an unreachable bound.
    t0 = time.time()
    net0, hist0 = train_summary_mlp(1.0, args.n_steps, args.batch_size,
                                     args.seed, nu2_zero=True)
    rmse_mlp0, _, _ = eval_summary_mlp(net0, 1.0, args.n_test, args.seed,
                                        nu2_zero=True)
    fim0 = e81.fisher_alpha_design(DESIGN, 2.0, 1.0, 1.0, 0.0, 1.0)
    fim0_da = 1.0 / np.sqrt(fim0)
    summary_fim0_da, summary_I0 = summary_fim_bound(0.0, om=1.0)
    print(f"[G-B5] summary MLP (nu2=0): rmse={rmse_mlp0:.4f} "
          f"summary_FIM_bound={summary_fim0_da:.4f} "
          f"(raw_FIM_bound={fim0_da:.4f}) "
          f"ratio_to_summary_ceiling={rmse_mlp0/summary_fim0_da:.2f} "
          f"time={time.time()-t0:.1f}s")

    result = {"nu2_zero_gate": {
        "raw_fim_dalpha_m1": fim0_da,
        "summary_fim_dalpha_m1": summary_fim0_da,
        "summary_mlp_rmse": rmse_mlp0,
        "summary_mlp_ratio_to_summary_ceiling": rmse_mlp0 / summary_fim0_da,
        "summary_mlp_ratio_to_raw_fim": rmse_mlp0 / fim0_da,
        "note": ("summary_ceiling accounts for the 7-point log-W_sat "
                 "summary being lossy relative to the raw field even at "
                 "nu2=0; the MLP is compared against IT (its own reachable "
                 "ceiling), while the CNN is compared against the raw FIM"),
    }}

    if not args.skip_cnn:
        t0 = time.time()
        cnn0, histc0 = train_cnn(1.0, args.n_steps, args.batch_size, args.seed,
                                  nu2_zero=True)
        rmse_cnn0, _, _ = eval_cnn(cnn0, 1.0, args.n_test, args.seed,
                                    nu2_zero=True)
        print(f"[G-B5] raw CNN (nu2=0): rmse={rmse_cnn0:.4f} "
              f"FIM_bound={fim0_da:.4f} ratio={rmse_cnn0/fim0_da:.2f} "
              f"time={time.time()-t0:.1f}s")
        result["nu2_zero_gate"]["cnn_rmse"] = rmse_cnn0
        result["nu2_zero_gate"]["cnn_ratio_to_raw_fim"] = rmse_cnn0 / fim0_da

    gate_b5_pass = (rmse_mlp0 / summary_fim0_da < 2.0) and (
        args.skip_cnn or rmse_cnn0 / fim0_da < 2.0)
    result["nu2_zero_gate"]["pass"] = bool(gate_b5_pass)
    print(f"[G-B5] pass={gate_b5_pass}")
    rep["gate_b5"] = result["nu2_zero_gate"]

    # ---------------- Task-2 floors at m=1 (single-shot, matching the
    # trained estimators' no-seed-averaging regime) ----------------
    sigma_l0 = e81.measure_sigma_level0()
    floors_m1 = {}
    for U in (0.5, 1.0, 4.0):
        f0 = e81.level0_floors({}, sigma_l0, U_grid=(U,), ms=(1,))
        f23, _ = e81.floor_spectral(sigma_l0, 1, U, n_starts=24)
        floors_m1[str(U)] = {"floor_L0_m1": f0[str(U)]["m1"],
                              "floor_L23_m1": f23}
        print(f"[Task2 floors @ m=1] U={U}: floor_L0={f0[str(U)]['m1']:.4f} "
              f"floor_L23={f23:.4f}")
    rep["task2_floors_m1_reference"] = floors_m1

    # ---------------- hard regimes: U in {0.5, 1, 4} ----------------
    hard = {}
    for U in (0.5, 1.0, 4.0):
        t0 = time.time()
        net, _ = train_summary_mlp(U, args.n_steps, args.batch_size, args.seed)
        rmse_mlp, pred_mlp, prior_mlp = eval_summary_mlp(net, U, args.n_test,
                                                          args.seed)
        fim_da_U, fim_val_U = fim_rmse_bound(U)
        summary_fim_da_U, _ = summary_fim_bound(U)
        floor_L0_m1 = floors_m1[str(U)]["floor_L0_m1"]
        floor_L23_m1 = floors_m1[str(U)]["floor_L23_m1"]
        entry = {"summary_mlp_rmse": rmse_mlp,
                  "dalpha_raw_FIM_m1": fim_da_U,
                  "dalpha_summary_FIM_m1": summary_fim_da_U,
                  "floor_L0_m1": floor_L0_m1,
                  "summary_mlp_beats_raw_fim": bool(rmse_mlp < fim_da_U),
                  "summary_mlp_beats_own_summary_ceiling":
                      bool(rmse_mlp < summary_fim_da_U),
                  "summary_mlp_beats_floor_L0": bool(rmse_mlp < floor_L0_m1)}
        print(f"[Hard U={U}] summary_mlp rmse={rmse_mlp:.4f} "
              f"floor_L0(m=1)={floor_L0_m1:.4f} "
              f"summary_FIM_m1={summary_fim_da_U:.4f} "
              f"(raw_FIM_m1={fim_da_U:.4f}) time={time.time()-t0:.1f}s")
        if not args.skip_cnn:
            t0 = time.time()
            cnn, _ = train_cnn(U, args.n_steps, args.batch_size, args.seed)
            rmse_cnn, pred_cnn, prior_cnn = eval_cnn(cnn, U, args.n_test,
                                                      args.seed)
            entry["cnn_rmse"] = rmse_cnn
            entry["floor_L23_m1"] = floor_L23_m1
            entry["cnn_beats_raw_fim"] = bool(rmse_cnn < fim_da_U)
            entry["cnn_beats_floor_L23"] = bool(rmse_cnn < floor_L23_m1)
            print(f"[Hard U={U}] cnn rmse={rmse_cnn:.4f} "
                  f"floor_L23(m=1)={floor_L23_m1:.4f} "
                  f"time={time.time()-t0:.1f}s")
        hard[str(U)] = entry
    rep["hard_regimes"] = hard

    with open(out_path, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
