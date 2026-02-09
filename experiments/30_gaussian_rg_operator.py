"""
Experiment 30: Gaussian-Smoothing RG Operator

Motivation
----------
Exp29 showed that block averaging + Family-Vicsek rescaling FAILS to produce
differential contraction. All variants (with/without height rescaling, with/without
time rescaling) made distances worse or caused divergence.

Alternative hypothesis: Block averaging is too discrete/harsh for these observables.
Try GAUSSIAN SMOOTHING as the RG operator instead (inspired by Exp14).

Theory
------
Gaussian smoothing σ acts as a continuous coarse-graining scale:
- h_σ(x) = ∫ G_σ(x-x') h(x') dx'  where G_σ is Gaussian kernel
- σ → 0: microscopic (raw) features
- σ → ∞: maximally coarse-grained features

Unlike block averaging:
- No discretization jumps
- Preserves spatial continuity
- More similar to field-theoretic RG (momentum-space cutoff)

Prediction
----------
If the geometric framework is RG-compatible:
- d(BD, KPZ)(σ) should CONTRACT as σ increases (within-class convergence)
- d(EW, KPZ)(σ) should remain roughly CONSTANT (between-class separation preserved)

This would validate the RG picture using a physically faithful coarse-graining operator.

Date: January 21, 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.physics_simulation import GrowthModelSimulator


def gaussian_smooth_1d(h: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing with scale sigma."""
    if sigma < 0.01:
        return h
    return gaussian_filter1d(h, sigma, mode='wrap')


def extract_gradient_moments_from_surface(h: np.ndarray) -> np.ndarray:
    """Same 6D feature extraction as Exp21/24/29."""
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


def compute_whitening_params(features_by_model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([features_by_model[k] for k in ["EW", "KPZ", "BD"]])
    return np.mean(X, axis=0), np.std(X, axis=0)


def whiten(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-10)


MODEL_SPECS = {
    "EW": ("edwards_wilkinson", {"diffusion": 1.0, "noise_strength": 1.0, "dt": 0.1}),
    "KPZ": (
        "kpz_equation",
        {"diffusion": 1.0, "nonlinearity": 1.0, "noise_strength": 1.0, "dt": 0.05},
    ),
    "BD": ("ballistic_deposition", {"noise_strength": 0.2}),
}


def run():
    # Core parameters
    L = 512
    T = 3000
    n_samples = 30
    sigmas = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # Gaussian smoothing scales

    out_dir = project_root / "results" / "exp30_gaussian_rg_operator"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("EXPERIMENT 30: Gaussian-Smoothing RG Operator")
    print("=" * 78)
    print(f"L={L}, T={T}, n_samples={n_samples}")
    print(f"Gaussian smoothing scales σ: {sigmas}")
    print()
    print("Testing whether smooth coarse-graining succeeds where block averaging failed")
    print()

    # Pre-generate trajectories
    print("Phase 1: generating trajectories...")
    simulator = GrowthModelSimulator(width=L, height=T, random_state=42)
    trajectories: dict[str, list[np.ndarray]] = {"EW": [], "KPZ": [], "BD": []}

    for short in ["EW", "KPZ", "BD"]:
        model_type, kwargs = MODEL_SPECS[short]
        print(f"  {short}: {model_type}")
        for i in range(n_samples):
            traj = simulator.generate_trajectory(model_type, **kwargs)
            trajectories[short].append(traj)

    # Compute fixed whitening from σ=0 (raw features)
    print("\nPhase 2: computing fixed whitening from σ=0...")
    features_sigma0: dict[str, np.ndarray] = {}
    for short in ["EW", "KPZ", "BD"]:
        feats = []
        for traj in trajectories[short]:
            h = traj[-1]  # Final time
            h_smooth = gaussian_smooth_1d(h, 0.0)
            feats.append(extract_gradient_moments_from_surface(h_smooth))
        features_sigma0[short] = np.array(feats)

    whiten_mean, whiten_std = compute_whitening_params(features_sigma0)
    print(f"  Fixed whitening computed: mean={whiten_mean[:3]}, std={whiten_std[:3]}")

    # Main experiment: sweep σ
    print("\nPhase 3: sweeping Gaussian smoothing scales...")
    print("-" * 78)

    distances = {
        "sigma": [],
        "d_bd_kpz": [],
        "d_ew_kpz": [],
        "d_bd_ew": [],
    }

    for sigma in sigmas:
        print(f"\nσ = {sigma:.1f}:")

        features_by_model: dict[str, np.ndarray] = {}
        for short in ["EW", "KPZ", "BD"]:
            feats = []
            for traj in trajectories[short]:
                h = traj[-1]
                h_smooth = gaussian_smooth_1d(h, sigma)
                feats.append(extract_gradient_moments_from_surface(h_smooth))
            # Apply fixed whitening
            features_by_model[short] = whiten(np.array(feats), whiten_mean, whiten_std)

        d_bd_kpz = mean_pairwise_distance(features_by_model["BD"], features_by_model["KPZ"])
        d_ew_kpz = mean_pairwise_distance(features_by_model["EW"], features_by_model["KPZ"])
        d_bd_ew = mean_pairwise_distance(features_by_model["BD"], features_by_model["EW"])

        distances["sigma"].append(sigma)
        distances["d_bd_kpz"].append(d_bd_kpz)
        distances["d_ew_kpz"].append(d_ew_kpz)
        distances["d_bd_ew"].append(d_bd_ew)

        print(f"  d(BD,KPZ) = {d_bd_kpz:.3f}")
        print(f"  d(EW,KPZ) = {d_ew_kpz:.3f}")
        print(f"  d(BD,EW)  = {d_bd_ew:.3f}")

    # Analysis and plotting
    print("\n" + "=" * 78)
    print("RESULTS SUMMARY")
    print("=" * 78)

    sigma_arr = np.array(distances["sigma"])
    d_bd_kpz = np.array(distances["d_bd_kpz"])
    d_ew_kpz = np.array(distances["d_ew_kpz"])
    d_bd_ew = np.array(distances["d_bd_ew"])

    # Compute contraction/drift metrics
    if d_bd_kpz[0] > 1e-12:
        bd_kpz_contraction = 1.0 - (d_bd_kpz[-1] / d_bd_kpz[0])
    else:
        bd_kpz_contraction = 0.0

    if d_ew_kpz[0] > 1e-12:
        ew_kpz_drift = (d_ew_kpz[-1] / d_ew_kpz[0]) - 1.0
    else:
        ew_kpz_drift = 0.0

    print(f"\nBD→KPZ contraction: {bd_kpz_contraction*100:.1f}%")
    print(f"EW↔KPZ drift: {ew_kpz_drift*100:.1f}%")

    # Success criterion check
    print("\n" + "-" * 78)
    if bd_kpz_contraction > 0.5 and abs(ew_kpz_drift) < 0.2:
        verdict = "✓ SUCCESS: Strong differential contraction observed!"
        print(verdict)
    elif bd_kpz_contraction > 0.3 and abs(ew_kpz_drift) < 0.3:
        verdict = "~ PARTIAL: Moderate contraction, some drift"
        print(verdict)
    else:
        verdict = "✗ FAILURE: No clear differential contraction"
        print(verdict)

    # Save results
    txt_path = out_dir / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Experiment 30: Gaussian-Smoothing RG Operator\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Parameters: L={L}, T={T}, n_samples={n_samples}\n")
        f.write(f"Smoothing scales: {sigmas}\n\n")
        f.write("sigma\td(BD,KPZ)\td(EW,KPZ)\td(BD,EW)\n")
        for i in range(len(sigma_arr)):
            f.write(f"{sigma_arr[i]:.1f}\t{d_bd_kpz[i]:.6f}\t{d_ew_kpz[i]:.6f}\t{d_bd_ew[i]:.6f}\n")
        f.write("\n")
        f.write(f"BD→KPZ contraction: {bd_kpz_contraction*100:.1f}%\n")
        f.write(f"EW↔KPZ drift: {ew_kpz_drift*100:.1f}%\n")
        f.write(f"\nVerdict: {verdict}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigma_arr, d_bd_kpz, 'o-', label='d(BD,KPZ) [within-class]', linewidth=2, markersize=6)
    ax.plot(sigma_arr, d_ew_kpz, 's-', label='d(EW,KPZ) [between-class]', linewidth=2, markersize=6)
    ax.set_xlabel('Gaussian smoothing scale σ', fontsize=11)
    ax.set_ylabel('Mean pairwise distance (fixed metric)', fontsize=11)
    ax.set_title('Gaussian-Smoothing RG Operator Test', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = out_dir / "gaussian_rg_operator.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {txt_path.name}, {fig_path.name}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    run()
