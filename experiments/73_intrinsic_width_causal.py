"""Experiment 73: causal test -- does removing intrinsic width merge BD into the KPZ class?

Background
----------
Exp 72 showed that, at accessible finite sizes, ballistic deposition (BD) alone
exhibits intrinsic anomalous roughening: an anomalous local roughness exponent
(alpha_local ~ 0.23 vs the universal 0.5) and a lattice-scale intrinsic width
~5x the continuum value. Eden, EW, and KPZ are Family-Vicsek (alpha_local ~ 0.5).
This *correlates* with the clustering behaviour (exp62-64): BD is exactly the
system that breaks out of the KPZ universality class, while Eden stays merged
with EW/KPZ.

This experiment turns that correlation into a causal test. The 6D spatial
features the project clusters on include `grad_var = Var[dh/dx] ~ G(2)/4`, so the
intrinsic width (sqrt G(2)) IS the gradient-amplitude coordinate of the feature
space. If BD's split is *caused* by that non-universal amplitude, then removing
it should pull BD into the KPZ class -- while removing an unrelated axis should
not, and the easy classes (RD, KS) should stay separable either way.

Manipulations (each re-extracts the same 6D features on rescaled surfaces)
-------------------------------------------------------------------------
- baseline   : raw surfaces (reproduces the documented ARI ~ 0.5 ceiling).
- iw_norm    : h -> h / intrinsic_width  (removes the local roughness amplitude).
- gw_norm    : h -> h / global_width     (control: removes the GLOBAL amplitude).
- shuf_iw    : h -> h / shuffled(intrinsic_width)  (control: wrong per-surface
               scale; rules out "any rescaling merges things").

Decisive metric: BD's distance to the {EW,KPZ,Eden} blob centroid, normalised by
the spread within that blob (>>1 = BD is an outlier; ~1 = BD merged), plus
whether HDBSCAN places BD in the same cluster as the continuum-KPZ majority.

Outputs (read-only w.r.t. existing results):
- results_exp73_intrinsic_width_causal/summary.json
- results_exp73_intrinsic_width_causal/condition_metrics.csv

Usage:
  python experiments/73_intrinsic_width_causal.py            # full test
  python experiments/73_intrinsic_width_causal.py --quick    # fast smoke run
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
EXP63_PATH = PROJECT_DIR / "experiments" / "63_temporal_features.py"
EXP62_NPZ = PROJECT_DIR / "results_exp62" / "features.npz"
RESULTS_DIR = PROJECT_DIR / "results_exp73_intrinsic_width_causal"

CLASS_MAP = {"ew": "EW", "kpz": "KPZ", "bd": "KPZ", "eden": "KPZ",
             "rd": "RD", "ks": "KS"}
# BD should join these (its KPZ-class continuum/FV partners + the degenerate EW blob)
KPZ_BLOB = ["ew", "kpz", "eden"]
LATE_FRAC = 0.3


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
        "ks": lambda s: exp63.simulate_ks(L=L, T=T, seed=s),
    }
    fn = sims[system]
    for attempt in range(6):
        traj = fn(seed + attempt * 100003)
        if traj is not None and np.all(np.isfinite(traj)):
            return traj
    return None


def intrinsic_width(traj):
    """sqrt G(2) on the late-time window: G(2) = <[h(x+2)-h(x)]^2>."""
    start = int(traj.shape[0] * (1 - LATE_FRAC))
    late = traj[start:]
    diff = np.roll(late, -2, axis=1) - late
    return float(np.sqrt(max(np.mean(diff ** 2), 1e-12)))


def global_width(traj):
    start = int(traj.shape[0] * (1 - LATE_FRAC))
    return float(np.mean(np.std(traj[start:], axis=1)))


def generate(exp63, systems, L, T, n, ks_cap, seed_base=4000):
    """Return per-surface feature variants, labels, and per-surface scales."""
    feats_raw, feats_iw, feats_gw, labels = [], [], [], []
    iw_list, gw_list = [], []
    for si, system in enumerate(systems):
        Tsys = min(T, ks_cap) if system == "ks" else T
        got = 0
        attempt = 0
        while got < n and attempt < n * 4:
            seed = seed_base + si * 100000 + attempt
            attempt += 1
            traj = simulate(exp63, system, L, Tsys, seed)
            if traj is None:
                continue
            iw = intrinsic_width(traj)
            gw = global_width(traj)
            feats_raw.append(exp63.extract_spatial_features(traj, LATE_FRAC))
            feats_iw.append(exp63.extract_spatial_features(traj / max(iw, 1e-9), LATE_FRAC))
            feats_gw.append(exp63.extract_spatial_features(traj / max(gw, 1e-9), LATE_FRAC))
            iw_list.append(iw)
            gw_list.append(gw)
            labels.append(system)
            got += 1
        print(f"  {system:5s}: {got} surfaces (iw_mean={np.mean(iw_list[-got:]):.3f})")
    return (np.array(feats_raw), np.array(feats_iw), np.array(feats_gw),
            labels, np.array(iw_list), np.array(gw_list))


def bd_merge_metrics(X, labels, hdb_labels):
    """Quantify whether BD has merged into the {EW,KPZ,Eden} blob in space X."""
    labels = np.array(labels)
    systems = sorted(set(labels))
    cen = {s: X[labels == s].mean(axis=0) for s in systems}

    blob_systems = [s for s in KPZ_BLOB if s in cen]
    blob_centroid = np.mean([cen[s] for s in blob_systems], axis=0)

    # within-blob spread: mean pairwise centroid distance among the blob systems
    pair = [np.linalg.norm(cen[a] - cen[b])
            for i, a in enumerate(blob_systems) for b in blob_systems[i + 1:]]
    spread = float(np.mean(pair)) if pair else float("nan")

    d_bd_blob = float(np.linalg.norm(cen["bd"] - blob_centroid)) if "bd" in cen else float("nan")
    ratio = d_bd_blob / spread if spread and spread > 0 else float("nan")

    # HDBSCAN co-membership: does BD sit in the continuum-KPZ majority cluster?
    bd_in_blob_cluster = float("nan")
    blob_cluster = None
    if hdb_labels is not None:
        hdb_labels = np.array(hdb_labels)
        blob_mask = np.isin(labels, blob_systems) & (hdb_labels != -1)
        if blob_mask.sum() > 0:
            vals, counts = np.unique(hdb_labels[blob_mask], return_counts=True)
            blob_cluster = int(vals[np.argmax(counts)])
            bd_mask = (labels == "bd")
            if bd_mask.sum() > 0:
                bd_in_blob_cluster = float(np.mean(hdb_labels[bd_mask] == blob_cluster))

    # control: are RD/KS still separable from the blob? (structure not destroyed)
    easy_sep = {}
    for s in ("rd", "ks"):
        if s in cen:
            easy_sep[s] = float(np.linalg.norm(cen[s] - blob_centroid) / spread) \
                if spread and spread > 0 else float("nan")

    return {
        "d_bd_to_blob": d_bd_blob,
        "blob_internal_spread": spread,
        "bd_outlier_ratio": ratio,
        "bd_fraction_in_blob_cluster": bd_in_blob_cluster,
        "blob_hdbscan_cluster": blob_cluster,
        "easy_class_outlier_ratio": easy_sep,
    }


def run_condition(exp63, name, feats, labels):
    print(f"\n=== condition: {name} ===")
    results, X, hdb_labels, km_labels = exp63.run_clustering_analysis(
        feats, labels, CLASS_MAP, label=name)
    merge = bd_merge_metrics(X, labels, hdb_labels)
    print(f"  BD outlier ratio (d_BD/blob_spread) = {merge['bd_outlier_ratio']:.2f}  "
          f"(>>1 outlier, ~1 merged)")
    print(f"  BD fraction in continuum-KPZ cluster = {merge['bd_fraction_in_blob_cluster']}")
    print(f"  RD/KS outlier ratios (should stay >>1) = {merge['easy_class_outlier_ratio']}")
    return {
        "hdbscan_clusters": results.get("hdbscan", {}).get("n_clusters"),
        "hdbscan_ari": results.get("hdbscan", {}).get("ari"),
        "kmeans_ari": results.get("kmeans", {}).get("ari"),
        "knn3_accuracy": results.get("knn_3", {}).get("mean_accuracy"),
        "merge": merge,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 73: intrinsic-width causal test")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    exp63 = load_exp63()
    RESULTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(73)

    systems = ["ew", "kpz", "bd", "eden", "rd", "ks"]
    if args.quick:
        L, T, n, ks_cap = 128, 600, 30, 600
    else:
        L, T, n, ks_cap = 128, 1000, 80, 1000

    out = {"config": {"L": L, "T": T, "n_per_system": n, "ks_cap": ks_cap,
                      "late_frac": LATE_FRAC, "quick": args.quick}}

    # --- anchor: documented stored feature matrix reproduces the ceiling ---
    if EXP62_NPZ.exists():
        npz = np.load(EXP62_NPZ, allow_pickle=True)
        stored_feats = npz["features"]
        stored_labels = [str(x) for x in npz["labels"]]
        print("=== anchor: stored exp62 feature matrix ===")
        anchor = run_condition(exp63, "stored_exp62", stored_feats, stored_labels)
        out["anchor_stored_exp62"] = anchor

    # --- regenerate surfaces with per-surface scales ---
    print("\n=== regenerating surfaces (paired features + intrinsic width) ===")
    feats_raw, feats_iw, feats_gw, labels, iw, gw = generate(
        exp63, systems, L, T, n, ks_cap)

    # validate the proxy relation intrinsic_width ~ 2*sqrt(grad_var)
    grad_var = feats_raw[:, 0]
    proxy = 2.0 * np.sqrt(np.clip(grad_var, 0, None))
    corr = float(np.corrcoef(iw, proxy)[0, 1])
    print(f"\ncheck: corr(intrinsic_width, 2*sqrt(grad_var)) = {corr:.4f} "
          f"(confirms iw is the gradient-amplitude coordinate)")
    out["iw_vs_gradvar_corr"] = corr
    out["intrinsic_width_by_system"] = {
        s: float(np.mean(iw[np.array(labels) == s])) for s in systems}

    # --- control: shuffled (mismatched) per-surface intrinsic width ---
    # Normalising surface i by iw[perm(i)] is exact in feature space: dividing a
    # surface by a scalar c divides the dimensionful features by c^2 and leaves
    # the scale-free moments (skew, kurt) unchanged. Using a *wrong* per-surface
    # scale should NOT merge BD -- it rules out "any rescaling merges things".
    iw_shuf = iw.copy()
    rng.shuffle(iw_shuf)
    feats_shuf = feats_raw.copy()
    for col in (0, 3, 4, 5):  # grad_var, lap_var, grad_lap_cov, h_var
        feats_shuf[:, col] = feats_raw[:, col] / (iw_shuf ** 2)

    conditions = {
        "baseline": feats_raw,
        "iw_norm": feats_iw,
        "gw_norm": feats_gw,
        "shuf_iw_control": feats_shuf,
    }
    out["conditions"] = {}
    for name, feats in conditions.items():
        out["conditions"][name] = run_condition(exp63, name, feats, labels)

    # --- write artifacts ---
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(RESULTS_DIR / "condition_metrics.csv", "w") as f:
        f.write("condition,hdbscan_clusters,hdbscan_ari,kmeans_ari,knn3_accuracy,"
                "bd_outlier_ratio,bd_fraction_in_blob_cluster\n")
        for name, c in out["conditions"].items():
            m = c["merge"]
            f.write(f"{name},{c['hdbscan_clusters']},{c['hdbscan_ari']},"
                    f"{c['kmeans_ari']},{c['knn3_accuracy']},"
                    f"{m['bd_outlier_ratio']:.4f},{m['bd_fraction_in_blob_cluster']}\n")

    # --- verdict line ---
    base = out["conditions"]["baseline"]["merge"]
    iwn = out["conditions"]["iw_norm"]["merge"]
    print("\n" + "=" * 72)
    print("CAUSAL TEST SUMMARY (BD outlier ratio: >>1 = split, ~1 = merged):")
    print(f"  baseline        : {base['bd_outlier_ratio']:.2f}  "
          f"(BD in blob cluster: {base['bd_fraction_in_blob_cluster']})")
    print(f"  iw_norm         : {iwn['bd_outlier_ratio']:.2f}  "
          f"(BD in blob cluster: {iwn['bd_fraction_in_blob_cluster']})")
    print(f"  gw_norm (ctrl)  : {out['conditions']['gw_norm']['merge']['bd_outlier_ratio']:.2f}")
    print(f"  shuf_iw (ctrl)  : {out['conditions']['shuf_iw_control']['merge']['bd_outlier_ratio']:.2f}")
    print(f"\nArtifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
