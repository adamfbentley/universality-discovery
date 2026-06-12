"""Exp 76c: Leave-one-family-out (LOFO) control — exclude F3_intrinsic.

Robustness check before claiming any result: train a mixture model on
F0_pure + F1_power + F2_twoterm + F4_log (no F3_intrinsic) and evaluate
ONLY the real systems. If BD's estimate is similar to the full-mix result,
it is not an artifact of including the Krug-Meakin correction form in the
training prior.

Imports functions from 76_amortized_extrapolation.py via importlib so this
script stays thin and doesn't duplicate logic.

Output: results_exp76_amortized_extrapolation/lofo_control.json
"""

import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
EXP76_PATH = os.path.join(HERE, "76_amortized_extrapolation.py")
RESULTS_DIR = os.path.join(HERE, "..", "results_exp76_amortized_extrapolation")

FAMILIES_LOFO = ["F0_pure", "F1_power", "F2_twoterm", "F4_log"]   # F3 excluded
QUANTILES = [0.05, 0.5, 0.95]


def load_exp76():
    spec = importlib.util.spec_from_file_location("exp76", EXP76_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    m = load_exp76()
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Build F3-excluded mixture training set from existing datasets.npz.
    # Subsample to 200k total (50k per family) to match the original mix
    # model size (stage_train trains on 200k samples per prior).
    d = np.load(os.path.join(RESULTS_DIR, "datasets.npz"))
    N_TOTAL = 200_000
    per_family = N_TOTAL // len(FAMILIES_LOFO)   # 50k each
    rng = np.random.default_rng(76)
    Ws_list, a_list = [], []
    for f in FAMILIES_LOFO:
        Wf = d[f"trainW_{f}"]; af = d[f"traina_{f}"]
        idx = rng.choice(len(Wf), size=per_family, replace=False)
        Ws_list.append(Wf[idx]); a_list.append(af[idx])
    Ws = np.vstack(Ws_list)
    alphas = np.concatenate(a_list)
    X = m.featurize(Ws)
    y = alphas
    print(f"LOFO train set: {X.shape[0]} samples ({per_family}/family), "
          f"{X.shape[1]} features (families: {', '.join(FAMILIES_LOFO)})",
          flush=True)

    # Fit 4 HistGradientBoosting models (point + q0.05 + q0.50 + q0.95)
    # Same settings as stage_train in exp76
    models = {}
    m_pt = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    m_pt.fit(X, y)
    models["point"] = m_pt
    for q in QUANTILES:
        mq = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                           max_iter=300, random_state=0)
        mq.fit(X, y)
        models[f"q{q}"] = mq
    print("LOFO model trained.", flush=True)

    # Evaluate real systems only
    perseed = m._load_perseed_ladders()
    real = {s: mm.mean(axis=0) for s, mm in perseed.items()} or m._load_real_ladders()
    source = "exp76b_perseed" if perseed else "exp75_seedmean"

    results = {}
    for sysname, W in real.items():
        Xr = m.featurize(W[None, :])
        point = float(models["point"].predict(Xr)[0])
        q05 = float(models["q0.05"].predict(Xr)[0])
        q95 = float(models["q0.95"].predict(Xr)[0])
        entry = {"point": point, "q05": q05, "q95": q95}
        if sysname in perseed:
            _, boot90 = m._predict_with_bootstrap(models, perseed[sysname])
            entry["seed_bootstrap90"] = boot90
        results[sysname] = entry
        boot = entry.get("seed_bootstrap90")
        bs = (f" boot90=[{boot[0]:.3f},{boot[1]:.3f}]" if boot else "")
        print(f"  {sysname:5s} point={point:.3f} q05={q05:.3f} q95={q95:.3f}" + bs,
              flush=True)

    out = {
        "description": "LOFO control: mix trained on F0+F1+F2+F4 (F3_intrinsic excluded)",
        "excluded_family": "F3_intrinsic",
        "train_families": FAMILIES_LOFO,
        "real_data_source": source,
        "real_systems": results,
    }
    out_path = os.path.join(RESULTS_DIR, "lofo_control.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
