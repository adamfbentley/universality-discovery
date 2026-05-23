"""
Experiment 70: Matched Multi-L Feature Baseline (Tranche 2 / referee-required)
=============================================================================

The exp69 head-to-head was information-asymmetric: exponent geometry used MULTI-L
width curves while the feature pipeline used a SINGLE L. So its ~2x ARI gain could
be due to the better representation (exponents) OR simply to using more
information (multiple L). This experiment narrows that objection by giving the
feature pipeline a raw multi-L concatenation baseline, with matched systems, seed
budget, and bootstrap sample construction. It does not rule out stronger
cross-L-engineered feature baselines.

Three representations, matched sample construction (per system: n_boot bootstraps
of seed-subsets at each L):
  1. single-L features  : 10D Exp63 features at the largest L only        (baseline)
  2. multi-L features    : 10D features at each L, concatenated (10*nL dims)
  3. cross-L features    : Lmax values + raw/log slopes + normalized contrasts
  4. exponent geometry   : (alpha, beta, z), recomputed from matched seed-subsets

Logic:
  * If raw multi-L concatenation stays near the single-L baseline, simple access
    to more L is not sufficient.
  * If engineered cross-L/exponent coordinates jump higher, the useful
    information is in scaling relations across L, not raw feature proximity.

Fixed parameters match exp69 (EW nu=1,D=1; KPZ lambda=4,nu=1,D=1; discrete defaults)
so both methods see the same physical systems. Features use exp63's 6 spatial +
4 temporal extractors.

Usage:
  python 70_matched_baseline.py --quick
  python 70_matched_baseline.py            # pilot
  python 70_matched_baseline.py --full
  python 70_matched_baseline.py --full --exp69-sampling
  python 70_matched_baseline.py --full --out-tag full_rerun_YYYYMMDD
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results_exp70_matched"
EXP63_PATH = SCRIPT_DIR / "63_temporal_features.py"
EXP69_PATH = SCRIPT_DIR / "69_collapse_metric_full.py"

SYSTEMS = ["ew", "kpz", "bd", "eden", "rd", "ks"]
CONTINUUM_SYSTEMS = {"ew", "kpz", "ks", "rd"}
CLASS_MAP = {"ew": "EW", "kpz": "KPZ", "bd": "KPZ", "eden": "KPZ", "rd": "trivial", "ks": "KS"}
SEED_OFF = {"ew": 0, "kpz": 1_000_000, "bd": 2_000_000, "eden": 3_000_000, "rd": 4_000_000, "ks": 5_000_000}
KPZ_LAM = 4.0
T_FEAT = 1000  # trajectory length for feature extraction (matches single-L baseline)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def feature_vectors(exp63, system, L, T, n_seed, seed_base):
    """Return seed-aligned 10D feature vectors (6 spatial + 4 temporal)."""
    out = {}
    for s in range(n_seed):
        seed = seed_base + s
        if system == "ew":
            traj = exp63.simulate_ew(L=L, T=T, nu=1.0, D=1.0, seed=seed)
        elif system == "kpz":
            traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=KPZ_LAM, D=1.0, seed=seed)
            if traj is None:
                traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=2.0, D=0.6, seed=seed + 7)
        elif system == "bd":
            traj = exp63.simulate_bd(L=L, T=T, seed=seed)
        elif system == "eden":
            traj = exp63.simulate_eden(L=L, T=T, seed=seed)
        elif system == "rd":
            traj = exp63.simulate_rd(L=L, T=T, seed=seed)
        elif system == "ks":
            traj = exp63.simulate_ks(L=L, T=min(T, 500), seed=seed, record_interval=10)
            if traj is None:
                traj = exp63.simulate_ks(L=L, T=min(T, 500), seed=seed + 7, record_interval=10, noise=0.02)
        else:
            raise ValueError(system)
        if traj is None:
            continue
        sp = exp63.extract_spatial_features(traj, 0.3)
        tp = exp63.compute_temporal_features(traj)
        v = np.concatenate([sp, tp])
        if np.all(np.isfinite(v)):
            out[int(seed)] = v
    return out


def variance_curves(exp63, exp69, system, L, n_seed, seed_base, cap):
    """Return seed-aligned variance curves for exponent extraction."""
    out = {}
    if system in exp69.CONTINUUM:
        T = exp69.T_cont(system, L, cap)
        rpm = 1.0
    else:
        T = None
        rpm = exp69.RECS_PER_ML

    for s in range(n_seed):
        seed = seed_base + s
        if system == "ew":
            traj = exp63.simulate_ew(L=L, T=T, nu=1.0, D=1.0, seed=seed)
            v = None if traj is None else traj.var(axis=1)
        elif system == "kpz":
            traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=KPZ_LAM, D=1.0, seed=seed)
            if traj is None:
                traj = exp63.simulate_kpz(L=L, T=T, nu=1.0, lam=2.0, D=0.6, seed=seed + 7)
            v = None if traj is None else traj.var(axis=1)
        elif system == "ks":
            traj = exp63.simulate_ks(L=L, T=T, seed=seed, record_interval=1)
            if traj is None:
                traj = exp63.simulate_ks(L=L, T=T, seed=seed + 7, record_interval=1, noise=0.02)
            v = None if traj is None else traj.var(axis=1)
        elif system == "rd":
            traj = exp63.simulate_rd(L=L, T=T, seed=seed)
            v = None if traj is None else traj.var(axis=1)
        elif system == "bd":
            n_rec = int(max(300, 4 * L) * rpm)
            dep = max(1, L // rpm)
            v = exp69.bd_var_curve(L, n_rec, dep, seed)
        elif system == "eden":
            n_rec = int(max(300, 4 * L) * rpm)
            dep = max(1, L // rpm)
            v = exp69.eden_var_curve(L, n_rec, dep, seed)
        else:
            raise ValueError(system)

        if v is not None and np.all(np.isfinite(v)):
            out[int(seed)] = v
    return out, rpm


def _confusion(classes, labels):
    classes = np.asarray(classes); labels = np.asarray(labels)
    return {str(c): {str(int(cl)): int(np.sum((classes == c) & (labels == cl)))
                      for cl in sorted(set(labels.tolist()))} for c in sorted(set(classes.tolist()))}


def _as_jsonable_labels(labels):
    return [int(x) for x in np.asarray(labels).tolist()]


def cluster_ari(X, classes):
    Xs = StandardScaler().fit_transform(X)
    nc = len(set(classes))
    km = KMeans(n_clusters=nc, random_state=42, n_init=20).fit_predict(Xs)
    out = {"n_samples": int(len(classes)), "n_features": int(X.shape[1]),
           "kmeans_ari": float(adjusted_rand_score(classes, km)),
           "kmeans_nmi": float(normalized_mutual_info_score(classes, km)),
           "kmeans_confusion": _confusion(classes, km),
           "kmeans_labels": _as_jsonable_labels(km)}
    try:
        import hdbscan
        hl = hdbscan.HDBSCAN(min_cluster_size=max(5, len(classes) // 20), min_samples=3).fit_predict(Xs)
        ncl = len(set(hl) - {-1}); mask = hl != -1
        out["hdbscan_clusters"] = int(ncl)
        out["hdbscan_noise"] = int(np.sum(hl == -1))
        out["hdbscan_ari_all"] = float(adjusted_rand_score(classes, hl))
        out["hdbscan_ari_core"] = float(adjusted_rand_score(np.asarray(classes)[mask], hl[mask])) if (ncl >= 2 and mask.any()) else 0.0
        out["hdbscan_labels"] = _as_jsonable_labels(hl)
    except Exception:
        out["hdbscan_clusters"] = 0; out["hdbscan_noise"] = 0
        out["hdbscan_ari_all"] = out["hdbscan_ari_core"] = float("nan")
        out["hdbscan_labels"] = []
    return out


def column_diagnostics(X, L_values, block_dim=10):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return {}
    std = X.std(axis=0)
    out = {
        "n_features": int(X.shape[1]),
        "raw_std_min": float(np.min(std)),
        "raw_std_mean": float(np.mean(std)),
        "raw_std_max": float(np.max(std)),
        "zero_std_columns": int(np.sum(std <= 1e-12)),
    }
    if X.shape[1] == block_dim * len(L_values):
        per_block = {}
        for i, L in enumerate(L_values):
            sl = slice(i * block_dim, (i + 1) * block_dim)
            per_block[str(L)] = {
                "raw_std_min": float(np.min(std[sl])),
                "raw_std_mean": float(np.mean(std[sl])),
                "raw_std_max": float(np.max(std[sl])),
            }
        out["per_L_block"] = per_block
        extra = std[: block_dim * (len(L_values) - 1)]
        if extra.size:
            out["non_Lmax_extra_columns"] = {
                "n_features": int(extra.size),
                "raw_std_min": float(np.min(extra)),
                "raw_std_mean": float(np.mean(extra)),
                "raw_std_max": float(np.max(extra)),
                "zero_std_columns": int(np.sum(extra <= 1e-12)),
            }
        else:
            out["non_Lmax_extra_columns"] = {"n_features": 0}
    return out


def cross_L_engineered_features(per_L, L_values):
    """Generic feature-side finite-size coordinates, without using class labels."""
    M = np.vstack([per_L[L] for L in L_values]).astype(float)
    logL = np.log(np.asarray(L_values, dtype=float))
    x = logL - logL.mean()
    denom = float(np.sum(x * x))

    if denom > 0:
        raw_slope = np.sum(x[:, None] * (M - M.mean(axis=0)), axis=0) / denom
        eps = np.maximum(1e-9, 1e-6 * np.nanmedian(np.abs(M), axis=0))
        logmag = np.log(np.abs(M) + eps)
        logmag_slope = np.sum(x[:, None] * (logmag - logmag.mean(axis=0)), axis=0) / denom
    else:
        raw_slope = np.zeros(M.shape[1])
        logmag_slope = np.zeros(M.shape[1])

    first, last = M[0], M[-1]
    rel_delta = (last - first) / (np.abs(last) + np.abs(first) + 1e-9)
    return np.concatenate([last, raw_slope, rel_delta, logmag_slope])


def pca_whiten(X, variance=0.95):
    """Return a PCA-whitened multi-L block and diagnostics for the artifact."""
    Xs = StandardScaler().fit_transform(X)
    max_comp = min(Xs.shape[0] - 1, Xs.shape[1])
    if max_comp < 1:
        return Xs, {"n_components": int(Xs.shape[1]), "explained_variance": 1.0}
    pca = PCA(n_components=max_comp, whiten=True, svd_solver="full", random_state=42)
    Z = pca.fit_transform(Xs)
    csum = np.cumsum(pca.explained_variance_ratio_)
    keep = int(np.searchsorted(csum, variance) + 1)
    keep = max(1, min(keep, Z.shape[1]))
    return Z[:, :keep], {
        "n_components": int(keep),
        "explained_variance": float(csum[keep - 1]),
        "total_components": int(Z.shape[1]),
    }


def partition_agreement(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return float(adjusted_rand_score(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed-start", type=int, default=70_000)
    ap.add_argument("--out-tag", default="", help="Optional suffix: results_exp70_matched_<tag>")
    ap.add_argument("--exp69-sampling", action="store_true",
                    help="Use exp69's continuum/discrete seed counts for exact exponent-cloud comparison")
    args = ap.parse_args()
    if args.quick:
        L_values, n_seed, n_boot = [48, 64], 6, 12
        n_seed_exp69 = {"continuum": 5, "discrete": 8}
    elif args.full:
        L_values, n_seed, n_boot = [48, 64, 96, 128], 16, 40
        n_seed_exp69 = {"continuum": 12, "discrete": 20}
    else:
        L_values, n_seed, n_boot = [48, 64, 96], 14, 30
        n_seed_exp69 = {"continuum": 10, "discrete": 16}

    if args.exp69_sampling:
        n_seed_by_system = {
            system: n_seed_exp69["continuum"] if system in CONTINUUM_SYSTEMS else n_seed_exp69["discrete"]
            for system in SYSTEMS
        }
    else:
        n_seed_by_system = {system: n_seed for system in SYSTEMS}

    results_dir = PROJECT_DIR / f"results_exp70_matched_{args.out_tag}" if args.out_tag else RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    exp63 = load_module(EXP63_PATH, "exp63")
    exp69 = load_module(EXP69_PATH, "exp69")
    rng = np.random.default_rng(args.seed_start)
    print("Experiment 70: matched multi-L feature baseline")
    print(f"L={L_values}, n_seed_by_system={n_seed_by_system}, n_boot={n_boot}, T_feat={T_FEAT}, KPZ lambda={KPZ_LAM}")
    start = time.time()

    cap = 4000 if args.quick else (20000 if args.full else 12000)

    # --- Generate feature matrices and variance curves per (system, L) ---
    raw, raw_var, rpm = {}, {}, {}
    for system in SYSTEMS:
        raw[system] = {}; raw_var[system] = {}
        for L in L_values:
            sb = args.seed_start + SEED_OFF[system] + L
            n_here = n_seed_by_system[system]
            raw[system][L] = feature_vectors(exp63, system, L, T_FEAT, n_here, sb)
            raw_var[system][L], rpm[system] = variance_curves(exp63, exp69, system, L, n_here, sb, cap)
        print(f"  {system:>4}: " + ", ".join(
            f"L{L}(feat{len(raw[system][L])}/var{len(raw_var[system][L])})" for L in L_values))

    Lmax = L_values[-1]

    # --- Bootstrap clouds: single-L, multi-L, cross-L, and matched exponent geometry ---
    X_single, X_multi, X_cross, X_exp, syslab, syslab_exp = [], [], [], [], [], []
    bootstrap_seed_sets = []
    for system in SYSTEMS:
        sub = max(2, n_seed_by_system[system] // 2)
        for _ in range(n_boot):
            per_L, var_by_L, seed_sets = {}, {}, {}
            ok = True
            for L in L_values:
                seeds = sorted(set(raw[system][L]) & set(raw_var[system][L]))
                if not seeds:
                    ok = False; break
                chosen = rng.choice(np.array(seeds, dtype=int), size=min(sub, len(seeds)), replace=False)
                seed_sets[str(L)] = [int(x) for x in chosen.tolist()]
                per_L[L] = np.vstack([raw[system][L][int(seed)] for seed in chosen]).mean(axis=0)
                var_by_L[L] = np.vstack([raw_var[system][L][int(seed)] for seed in chosen]).mean(axis=0)
            if not ok:
                continue
            X_single.append(per_L[Lmax])
            X_multi.append(np.concatenate([per_L[L] for L in L_values]))
            X_cross.append(cross_L_engineered_features(per_L, L_values))
            syslab.append(system)
            bootstrap_seed_sets.append({"system": system, "seeds_by_L": seed_sets})
            a, b, z = exp69.exponents(var_by_L, rpm[system], discrete=(system not in exp69.CONTINUUM))
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(z):
                X_exp.append([a, b, z])
                syslab_exp.append(system)
    X_single = np.array(X_single); X_multi = np.array(X_multi); X_cross = np.array(X_cross)
    X_exp = np.array(X_exp)
    syslab = np.array(syslab)
    syslab_exp = np.array(syslab_exp)
    classes = np.array([CLASS_MAP[s] for s in syslab])
    classes_exp = np.array([CLASS_MAP[s] for s in syslab_exp])

    res_single = cluster_ari(X_single, classes)
    res_multi = cluster_ari(X_multi, classes)
    res_cross = cluster_ari(X_cross, classes)
    X_multi_pca, pca_diag = pca_whiten(X_multi)
    res_multi_pca = cluster_ari(X_multi_pca, classes)
    noks = syslab != "ks"
    res_single_noks = cluster_ari(X_single[noks], classes[noks])
    res_multi_noks = cluster_ari(X_multi[noks], classes[noks])
    res_cross_noks = cluster_ari(X_cross[noks], classes[noks])
    X_multi_pca_noks, pca_diag_noks = pca_whiten(X_multi[noks])
    res_multi_pca_noks = cluster_ari(X_multi_pca_noks, classes[noks])
    res_exp = cluster_ari(X_exp, classes_exp) if len(classes_exp) else None
    noks_exp = syslab_exp != "ks"
    res_exp_noks = cluster_ari(X_exp[noks_exp], classes_exp[noks_exp]) if len(classes_exp) else None

    matrix_artifact = results_dir / "matched_matrices.npz"
    np.savez_compressed(
        matrix_artifact,
        single_L_features=X_single,
        multi_L_features=X_multi,
        cross_L_engineered_features=X_cross,
        pca_whitened_multi_L_features=X_multi_pca,
        matched_exponent_geometry=X_exp,
        sample_system_labels=syslab,
        sample_class_labels=classes,
        matched_exponent_system_labels=syslab_exp,
        matched_exponent_class_labels=classes_exp,
        L_values=np.asarray(L_values, dtype=int),
    )

    # --- Load exponent geometry for reference (exp69 full) ---
    exp_ref = None
    p = PROJECT_DIR / "results_exp69_collapse_full" / "summary.json"
    if p.exists():
        d = json.load(open(p))
        eg = d["head_to_head_all6_4class"]["exponent_geometry"]
        egk = d["head_to_head_KS_excluded"]["exponent_geometry"]
        exp_ref = {"all6_kmeans": eg["kmeans_ari"], "all6_hdbscan_core": eg["hdbscan_ari_core"],
                   "ks_excluded_kmeans": egk["kmeans_ari"]}

    print("\n=== representation vs information (all 6 systems, 4-class, matched seed subsets) ===")
    print(f"  single-L features ({X_single.shape[1]}D): KMeans={res_single['kmeans_ari']:.3f}  HDBcore={res_single['hdbscan_ari_core']:.3f}")
    print(f"  multi-L  features ({X_multi.shape[1]}D): KMeans={res_multi['kmeans_ari']:.3f}  HDBcore={res_multi['hdbscan_ari_core']:.3f}")
    print(f"  cross-L engineered ({X_cross.shape[1]}D): KMeans={res_cross['kmeans_ari']:.3f}  HDBcore={res_cross['hdbscan_ari_core']:.3f}")
    print(f"  PCA-whitened multi-L ({X_multi_pca.shape[1]}D): KMeans={res_multi_pca['kmeans_ari']:.3f}  HDBcore={res_multi_pca['hdbscan_ari_core']:.3f}")
    if res_exp:
        print(f"  exponent geometry ({X_exp.shape[1]}D, matched): KMeans={res_exp['kmeans_ari']:.3f}  HDBcore={res_exp['hdbscan_ari_core']:.3f}")
    if exp_ref:
        print(f"  exponent geometry (3D, exp69 ref):  KMeans={exp_ref['all6_kmeans']:.3f}  HDBcore={exp_ref['all6_hdbscan_core']:.3f}")
    print("  [KS-excluded]")
    print(f"  single-L features: KMeans={res_single_noks['kmeans_ari']:.3f}   multi-L features: KMeans={res_multi_noks['kmeans_ari']:.3f}"
          + f"   cross-L: KMeans={res_cross_noks['kmeans_ari']:.3f}"
          + f"   PCA multi-L: KMeans={res_multi_pca_noks['kmeans_ari']:.3f}"
          + (f"   exponent matched: KMeans={res_exp_noks['kmeans_ari']:.3f}" if res_exp_noks else "")
          + (f"   exponent ref: KMeans={exp_ref['ks_excluded_kmeans']:.3f}" if exp_ref else ""))

    exp_for_hint = res_exp if res_exp else exp_ref
    exp_km = exp_for_hint["kmeans_ari"] if res_exp else (exp_ref["all6_kmeans"] if exp_ref else None)
    best_feature_km = max(res_multi["kmeans_ari"], res_cross["kmeans_ari"], res_multi_pca["kmeans_ari"])
    verdict = ("FEATURE_BASELINES_STILL_BELOW: raw, engineered, and PCA-whitened features "
               "remain well below exponent geometry"
               if (exp_km is not None and best_feature_km < exp_km - 0.15)
               else "MIXED: at least one feature baseline closes much of the exponent gap or no exponent baseline")

    summary = {
        "config": {"L_values": L_values, "n_seed": n_seed, "n_boot": n_boot, "T_feat": T_FEAT,
                   "n_seed_by_system": n_seed_by_system, "exp69_sampling": args.exp69_sampling,
                   "kpz_lambda": KPZ_LAM, "Lmax": Lmax, "out_tag": args.out_tag},
        "elapsed_seconds": time.time() - start,
        "single_L_features": res_single,
        "multi_L_features": res_multi,
        "cross_L_engineered_features": res_cross,
        "pca_whitened_multi_L_features": res_multi_pca,
        "matched_exponent_geometry": res_exp,
        "single_L_features_KS_excluded": res_single_noks,
        "multi_L_features_KS_excluded": res_multi_noks,
        "cross_L_engineered_features_KS_excluded": res_cross_noks,
        "pca_whitened_multi_L_features_KS_excluded": res_multi_pca_noks,
        "matched_exponent_geometry_KS_excluded": res_exp_noks,
        "partition_agreement": {
            "single_vs_multi_kmeans_label_ari": partition_agreement(
                res_single.get("kmeans_labels"), res_multi.get("kmeans_labels")),
            "single_vs_multi_hdbscan_label_ari": partition_agreement(
                res_single.get("hdbscan_labels"), res_multi.get("hdbscan_labels")),
            "single_vs_cross_L_kmeans_label_ari": partition_agreement(
                res_single.get("kmeans_labels"), res_cross.get("kmeans_labels")),
            "multi_vs_cross_L_kmeans_label_ari": partition_agreement(
                res_multi.get("kmeans_labels"), res_cross.get("kmeans_labels")),
        },
        "column_diagnostics": {
            "single_L_features": column_diagnostics(X_single, [Lmax]),
            "multi_L_features": column_diagnostics(X_multi, L_values),
            "cross_L_engineered_features": column_diagnostics(
                X_cross, ["Lmax", "raw_slope", "rel_delta", "logmag_slope"]),
            "pca_whitened_multi_L_features": column_diagnostics(
                X_multi_pca, [f"PC{i + 1}" for i in range(X_multi_pca.shape[1])], block_dim=1),
            "matched_exponent_geometry": column_diagnostics(X_exp, ["exponent"], block_dim=3) if len(X_exp) else {},
        },
        "pca_whiten_diagnostics": {
            "all6": pca_diag,
            "KS_excluded": pca_diag_noks,
        },
        "sample_system_labels": syslab.tolist(),
        "sample_class_labels": classes.tolist(),
        "matched_exponent_system_labels": syslab_exp.tolist(),
        "matched_exponent_class_labels": classes_exp.tolist(),
        "matched_exponent_vectors": X_exp.tolist(),
        "matrix_artifact": str(matrix_artifact.relative_to(PROJECT_DIR)),
        "bootstrap_seed_sets": bootstrap_seed_sets,
        "exponent_geometry_reference_exp69": exp_ref,
        "verdict_hint": verdict,
        "note": "Single-L, raw multi-L, cross-L engineered, and PCA-whitened feature "
                "baselines use identical seed subsets. Matched exponent geometry is "
                "recomputed from the same seed subsets, but it is still a hand-built "
                "scaling representation. These baselines test specific feature-side "
                "uses of multi-L information, not every possible learned representation.",
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['elapsed_seconds']:.1f}s -> {results_dir}")


if __name__ == "__main__":
    main()
