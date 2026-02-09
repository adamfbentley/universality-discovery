"""
Experiment 29: RG Operator Sweep (Rescuing Differential Contraction)

Motivation
----------
Exp24 (RG differential contraction) failed in the saved outputs:
- BD→KPZ contracted only modestly
- EW↔KPZ drifted substantially

A key reason is that Exp24 uses:
1) spatial block averaging only (no height/time rescaling), AND
2) per-block whitening (re-defining the metric at every b).

But RG comparisons are only meaningful when:
- the RG map is scaling-consistent (space + height + time rescaling), and
- distances are computed in a *fixed* metric (same whitening / same feature
  normalization) so changes in distance reflect geometry, not metric drift.

This script sweeps a small set of RG-map / metric choices and reports whether
"differential contraction" emerges under any physically-faithful setup.

Design
------
For each block size b and each model:
- pick a time index using either raw time (baseline) or a fixed rescaled time t'
  with t = floor(t' * b^z) (RG-consistent comparison)
- apply spatial block averaging (size b)
- optionally rescale heights by b^alpha
- extract the same 6D gradient-moment features as Exp21/24

Then compute three distances:
- d(BD,KPZ), d(EW,KPZ), d(BD,EW)

Distance metric variants:
- fixed whitening derived at b=1 (recommended)
- per-b whitening (Exp24-style; included as a control)

Success criterion (practical):
- strong contraction of within-class BD→KPZ, while EW↔KPZ stays roughly stable
  under the *fixed* metric.

Date: January 21, 2026
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.physics_simulation import GrowthModelSimulator


def block_average_1d(h: np.ndarray, block_size: int) -> np.ndarray:
    if block_size == 1:
        return h
    L = len(h)
    L_new = L // block_size
    return np.array([np.mean(h[i * block_size : (i + 1) * block_size]) for i in range(L_new)])


def extract_gradient_moments_from_surface(h: np.ndarray) -> np.ndarray:
    """Same 6D feature philosophy as Exp21/24, but for a single 1D surface."""
    grad = (np.roll(h, -1) - np.roll(h, 1)) / 2.0
    lap = np.roll(h, -1) + np.roll(h, 1) - 2.0 * h

    grad_mean = np.mean(grad)
    grad_std = np.std(grad)
    if grad_std > 1e-10:
        grad_skew = np.mean((grad - grad_mean) ** 3) / (grad_std**3)
        grad_kurt = np.mean((grad - grad_mean) ** 4) / (grad_std**4) - 3.0
    else:
        grad_skew = 0.0
        grad_kurt = 0.0

    grad_lap_cov = np.mean((grad - np.mean(grad)) * (lap - np.mean(lap)))
    h_var = np.var(h - np.mean(h))

    return np.array(
        [
            np.var(grad),
            grad_skew,
            grad_kurt,
            np.var(lap),
            grad_lap_cov,
            h_var,
        ]
    )


def mean_pairwise_distance(A: np.ndarray, B: np.ndarray) -> float:
    dists = [euclidean(a, b) for a in A for b in B]
    return float(np.mean(dists))


@dataclass(frozen=True)
class RGMode:
    name: str
    use_time_rescaling: bool
    z_strategy: str  # 'none'|'common'|'class'
    common_z: float | None
    alpha: float
    height_rescale: bool
    whitening: str  # 'fixed_b1'|'per_b'|'none'


MODEL_SPECS = {
    "EW": ("edwards_wilkinson", {"diffusion": 1.0, "noise_strength": 1.0, "dt": 0.1}),
    "KPZ": (
        "kpz_equation",
        {"diffusion": 1.0, "nonlinearity": 1.0, "noise_strength": 1.0, "dt": 0.05},
    ),
    "BD": ("ballistic_deposition", {"noise_strength": 0.2}),
}

CLASS_Z = {
    "EW": 2.0,
    "KPZ": 1.5,
    "BD": 1.5,
}


def pick_time_index(
    *,
    T: int,
    block_size: int,
    model_short: str,
    tprime_target: float,
    mode: RGMode,
) -> int:
    if not mode.use_time_rescaling:
        return T - 1

    if mode.z_strategy == "class":
        z = CLASS_Z[model_short]
    elif mode.z_strategy == "common":
        if mode.common_z is None:
            raise ValueError("common_z must be set when z_strategy='common'")
        z = mode.common_z
    else:
        raise ValueError(f"Unexpected z_strategy: {mode.z_strategy}")

    t_raw = int(np.floor(tprime_target * (block_size**z)))
    return max(0, min(T - 1, t_raw))


def compute_whitening_params(features_by_model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([features_by_model[k] for k in ["EW", "KPZ", "BD"]])
    return np.mean(X, axis=0), np.std(X, axis=0)


def whiten(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-10)


def run():
    # Core parameters
    L = 512
    T = 3000
    n_samples = 30
    block_sizes = [1, 2, 4, 8, 16, 32]

    # RG-consistent comparison time (in rescaled units). Must satisfy t' * b^z <= T.
    # For b=32, z=1.5 => 32^1.5 ~ 181; with T=3000 this allows t' up to ~16.
    tprime_target = 12.0

    # RG modes to compare (minimal but informative)
    modes: list[RGMode] = [
        RGMode(
            name="baseline_spatial_only__per_b_whiten (Exp24-style)",
            use_time_rescaling=False,
            z_strategy="none",
            common_z=None,
            alpha=0.5,
            height_rescale=False,
            whitening="per_b",
        ),
        RGMode(
            name="spatial+height__fixed_b1_whiten",
            use_time_rescaling=False,
            z_strategy="none",
            common_z=None,
            alpha=0.5,
            height_rescale=True,
            whitening="fixed_b1",
        ),
        RGMode(
            name="space+time+height (common z=1.75)__fixed_b1_whiten",
            use_time_rescaling=True,
            z_strategy="common",
            common_z=1.75,
            alpha=0.5,
            height_rescale=True,
            whitening="fixed_b1",
        ),
        RGMode(
            name="space+time+height (class z)__fixed_b1_whiten",
            use_time_rescaling=True,
            z_strategy="class",
            common_z=None,
            alpha=0.5,
            height_rescale=True,
            whitening="fixed_b1",
        ),
    ]

    out_dir = project_root / "results" / "exp29_rg_operator_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 29: RG Operator Sweep")
    print("=" * 78)
    print(f"L={L}, T={T}, n_samples={n_samples}, block_sizes={block_sizes}")
    print(f"t'_target={tprime_target}")

    # Pre-generate trajectories once so all modes compare apples-to-apples
    print("\nPhase 1: generating trajectories...")
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    trajectories: dict[str, list[np.ndarray]] = {"EW": [], "KPZ": [], "BD": []}

    for short in ["EW", "KPZ", "BD"]:
        model_type, kwargs = MODEL_SPECS[short]
        print(f"  {short}: {model_type}")
        for i in range(n_samples):
            traj = simulator.generate_trajectory(model_type, **kwargs)
            trajectories[short].append(traj)

    # Run each mode
    for mode in modes:
        print("\n" + "-" * 78)
        print(f"Mode: {mode.name}")

        # If using fixed whitening at b=1, compute it now using b=1 features
        fixed_whiten = None
        if mode.whitening == "fixed_b1":
            features_b1: dict[str, np.ndarray] = {}
            for short in ["EW", "KPZ", "BD"]:
                feats = []
                for traj in trajectories[short]:
                    t_idx = pick_time_index(T=T, block_size=1, model_short=short, tprime_target=tprime_target, mode=mode)
                    h = traj[t_idx]
                    h_c = block_average_1d(h, 1)
                    if mode.height_rescale:
                        h_c = h_c / (1 ** mode.alpha)
                    feats.append(extract_gradient_moments_from_surface(h_c))
                features_b1[short] = np.array(feats)
            fixed_whiten = compute_whitening_params(features_b1)

        distances = {
            "b": [],
            "d_bd_kpz": [],
            "d_ew_kpz": [],
            "d_bd_ew": [],
        }

        for b in block_sizes:
            features_by_model: dict[str, np.ndarray] = {}
            for short in ["EW", "KPZ", "BD"]:
                feats = []
                for traj in trajectories[short]:
                    t_idx = pick_time_index(
                        T=T,
                        block_size=b,
                        model_short=short,
                        tprime_target=tprime_target,
                        mode=mode,
                    )
                    h = traj[t_idx]
                    h_c = block_average_1d(h, b)
                    if mode.height_rescale:
                        h_c = h_c / (b ** mode.alpha)
                    feats.append(extract_gradient_moments_from_surface(h_c))
                features_by_model[short] = np.array(feats)

            # Whitening strategy
            if mode.whitening == "per_b":
                m, s = compute_whitening_params(features_by_model)
                features_by_model = {k: whiten(v, m, s) for k, v in features_by_model.items()}
            elif mode.whitening == "fixed_b1":
                if fixed_whiten is None:
                    raise RuntimeError("fixed_b1 whitening requested but not computed")
                m, s = fixed_whiten
                features_by_model = {k: whiten(v, m, s) for k, v in features_by_model.items()}
            elif mode.whitening == "none":
                pass
            else:
                raise ValueError(f"Unknown whitening strategy: {mode.whitening}")

            d_bd_kpz = mean_pairwise_distance(features_by_model["BD"], features_by_model["KPZ"])
            d_ew_kpz = mean_pairwise_distance(features_by_model["EW"], features_by_model["KPZ"])
            d_bd_ew = mean_pairwise_distance(features_by_model["BD"], features_by_model["EW"])

            distances["b"].append(b)
            distances["d_bd_kpz"].append(d_bd_kpz)
            distances["d_ew_kpz"].append(d_ew_kpz)
            distances["d_bd_ew"].append(d_bd_ew)

            print(f"  b={b:>2}: d(BD,KPZ)={d_bd_kpz:6.3f} | d(EW,KPZ)={d_ew_kpz:6.3f} | d(BD,EW)={d_bd_ew:6.3f}")

        # Save and plot
        b = np.array(distances["b"], dtype=float)
        d_bd_kpz = np.array(distances["d_bd_kpz"], dtype=float)
        d_ew_kpz = np.array(distances["d_ew_kpz"], dtype=float)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(b, d_bd_kpz, "o-", label="d(BD,KPZ) (within-class)")
        plt.plot(b, d_ew_kpz, "o-", label="d(EW,KPZ) (between-class)")
        plt.xscale("log", base=2)
        plt.xlabel("Block size b")
        plt.ylabel("Mean pairwise distance")
        plt.title(mode.name)
        plt.legend()
        plt.tight_layout()

        safe_name = (
            mode.name.replace(" ", "_")
            .replace("/", "-")
            .replace("=", "")
            .replace("(", "")
            .replace(")", "")
        )
        fig_path = out_dir / f"{safe_name}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        # Write a simple text summary
        txt_path = out_dir / f"{safe_name}.txt"
        contraction = 1.0 - (d_bd_kpz[-1] / (d_bd_kpz[0] + 1e-12))
        drift = (d_ew_kpz[-1] / (d_ew_kpz[0] + 1e-12)) - 1.0
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Mode: {mode.name}\n")
            f.write(f"t'_target = {tprime_target}\n")
            f.write("b\td(BD,KPZ)\td(EW,KPZ)\td(BD,EW)\n")
            for i in range(len(b)):
                f.write(
                    f"{int(b[i])}\t{distances['d_bd_kpz'][i]:.6f}\t{distances['d_ew_kpz'][i]:.6f}\t{distances['d_bd_ew'][i]:.6f}\n"
                )
            f.write("\n")
            f.write(f"BD→KPZ contraction: {contraction*100:.1f}%\n")
            f.write(f"EW↔KPZ drift: {drift*100:.1f}%\n")

        print(f"  Saved: {fig_path.name}, {txt_path.name}")

    print("\nDone. Results written to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    run()
