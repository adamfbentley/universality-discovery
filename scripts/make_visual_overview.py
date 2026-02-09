"""
Make Visual Overview Figures for universality-discovery
======================================================

This script generates a small set of "overview" figures that are useful for:
- Explaining the measure-space / pushforward-measure framing
- Explaining the diagnostic gate (scale-invariance) vs universality (cross-system collapse)
- Summarizing key Exp 50 results (KS/KPZ, Burgers/KPZ, KPZ/KPZ, alpha-only)

Outputs are written to:
  universality-discovery/figures/visual_overview/

Notes:
- Uses only local results folders and metadata (no web access).
- Uses a non-interactive backend (Agg) so it can run headless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.pairwise import rbf_kernel


ROOT = Path(__file__).resolve().parents[1]  # universality-discovery/
OUT_DIR = ROOT / "figures" / "visual_overview"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """
    MMD distance with RBF kernel, matching our experiment utilities.

    Returns sqrt(max(0, MMD^2)).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    gamma = 1.0 / (2.0 * float(sigma) ** 2)
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = float(XX.mean() + YY.mean() - 2.0 * XY.mean())
    return float(np.sqrt(max(0.0, mmd2)))


def _lin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple least-squares slope for y = a + b x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])


def _box(ax, xy: tuple[float, float], text: str, *, w: float = 0.16, h: float = 0.24) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor="#f7f7f7",
        edgecolor="#333333",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.2,
        color="#333333",
        transform=ax.transAxes,
    )
    ax.add_patch(arrow)


def plot_pipeline_schematic(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axis("off")

    _box(ax, (0.03, 0.62), "Model + Protocol\n(system, IC, params,\nL, T)")
    _box(ax, (0.23, 0.62), "Random Field\nh(x) ~ P")
    _box(ax, (0.43, 0.62), "RG Step\nC_b(h)\n(coarse-grain + rescale)")
    _box(ax, (0.63, 0.62), "Observable Map\nPhi(h) = x in R^d")
    _box(ax, (0.83, 0.62), "Feature Law\nmu^b = Law(x)\n(pushforward)")

    _arrow(ax, (0.19, 0.74), (0.23, 0.74))
    _arrow(ax, (0.39, 0.74), (0.43, 0.74))
    _arrow(ax, (0.59, 0.74), (0.63, 0.74))
    _arrow(ax, (0.79, 0.74), (0.83, 0.74))

    _box(ax, (0.10, 0.16), "Split-Half Gate\nA vs A, B vs B\nmust be flat vs b", w=0.24, h=0.22)
    _box(ax, (0.40, 0.16), "Cross-System Distance\nD(b) = Dist(mu_A^b, mu_B^b)\n(e.g., MMD w/ fixed sigma)", w=0.28, h=0.22)
    _box(ax, (0.74, 0.16), "Interpretation\nOnly if gate passes:\ntrend of D(b)\nmeaningful", w=0.22, h=0.22)

    _arrow(ax, (0.50, 0.62), (0.22, 0.38))
    _arrow(ax, (0.88, 0.62), (0.54, 0.38))
    _arrow(ax, (0.52, 0.27), (0.74, 0.27))

    ax.text(
        0.5,
        0.96,
        "Measure-Space RG Diagnostic: From Random Fields to Feature-Law Geometry",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scale_invariance_vs_universality(out_path: Path) -> None:
    """
    Conceptual plot: gate passing (self-consistency) does not guarantee universality collapse.
    """
    b = np.array([1, 2, 4, 8], dtype=float)
    x = np.log2(b)

    # Conceptual curves (not data): chosen to illustrate key logic.
    d_AA = 0.06 + 0.002 * x * 0  # flat
    d_BB = 0.05 + 0.002 * x * 0  # flat
    d_AB_universal = 0.10 * np.exp(-0.7 * x) + 0.01  # decays
    d_AB_nonuniv = 0.05 + 0.004 * x  # drifts up despite gate

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, d_AA, "o-", label="Gate: A vs A (scale-invariance)")
    ax.plot(x, d_BB, "s-", label="Gate: B vs B (scale-invariance)")
    ax.plot(x, d_AB_universal, "^-", label="A vs B (universal observable)")
    ax.plot(x, d_AB_nonuniv, "d-", label="A vs B (scale-invariant but non-universal)")

    ax.set_xlabel("log2(b) (coarse-graining scale)")
    ax.set_ylabel("Distance between feature-laws")
    ax.set_title("Scale-Invariance (Gate) vs Universality (Cross-System Collapse)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@dataclass(frozen=True)
class ExpCurve:
    label: str
    scales: np.ndarray
    distances: np.ndarray
    slope: float
    gate_passed: bool
    color: str


def _load_exp50_curves() -> list[ExpCurve]:
    curves: list[ExpCurve] = []

    # Exp 50n: KS vs KPZ (scale-free structure functions)
    meta_50n = _read_json(ROOT / "results_exp50n" / "metadata.json")
    curves.append(
        ExpCurve(
            label="50n: KS vs KPZ (structure functions)",
            scales=np.array(meta_50n["parameters"]["scales"], dtype=float),
            distances=np.array(meta_50n["ks_vs_kpz"]["distances"], dtype=float),
            slope=float(meta_50n["ks_vs_kpz"]["slope"]),
            gate_passed=bool(meta_50n["diagnostic_gate"]["passed"]),
            color="#1f77b4",
        )
    )

    # Exp 50o: Burgers vs KPZ (structure functions, height space)
    meta_50o = _read_json(ROOT / "results_exp50o" / "metadata.json")
    curves.append(
        ExpCurve(
            label="50o: Burgers vs KPZ (structure functions)",
            scales=np.array(meta_50o["parameters"]["scales"], dtype=float),
            distances=np.array(meta_50o["burgers_vs_kpz"]["distances"], dtype=float),
            slope=float(meta_50o["burgers_vs_kpz"]["slope"]),
            gate_passed=bool(meta_50o["diagnostic_gate"]["passed"]),
            color="#ff7f0e",
        )
    )

    # Exp 50q pilot: KPZ-A vs KPZ-B (structure functions)
    meta_50q = _read_json(ROOT / "results_exp50q_pilot" / "metadata.json")
    curves.append(
        ExpCurve(
            label="50q (pilot): KPZ-A vs KPZ-B (structure functions)",
            scales=np.array(meta_50q["parameters"]["scales"], dtype=float),
            distances=np.array(meta_50q["kpz_A_vs_B"]["distances"], dtype=float),
            slope=float(meta_50q["kpz_A_vs_B"]["slope"]),
            gate_passed=bool(meta_50q["diagnostic_gate"]["passed"]),
            color="#2ca02c",
        )
    )

    # Exp 50r: alpha-only (KPZ-A vs KPZ-B, and KS vs KPZ-A)
    meta_50r = _read_json(ROOT / "results_exp50r" / "metadata.json")
    curves.append(
        ExpCurve(
            label="50r: KPZ-A vs KPZ-B (alpha-only)",
            scales=np.array(meta_50r["parameters"]["scales"], dtype=float),
            distances=np.array(meta_50r["cross_distances"]["KPZ_A_vs_KPZ_B"]["distances"], dtype=float),
            slope=float(meta_50r["cross_distances"]["KPZ_A_vs_KPZ_B"]["slope"]),
            gate_passed=bool(meta_50r["diagnostic_gate"]["all_passed"]),
            color="#d62728",
        )
    )
    curves.append(
        ExpCurve(
            label="50r: KS vs KPZ-A (alpha-only)",
            scales=np.array(meta_50r["parameters"]["scales"], dtype=float),
            distances=np.array(meta_50r["cross_distances"]["KS_vs_KPZ_A"]["distances"], dtype=float),
            slope=float(meta_50r["cross_distances"]["KS_vs_KPZ_A"]["slope"]),
            gate_passed=bool(meta_50r["diagnostic_gate"]["all_passed"]),
            color="#9467bd",
        )
    )

    return curves


def plot_exp50_distance_summary(out_path: Path) -> None:
    curves = _load_exp50_curves()

    fig, ax = plt.subplots(figsize=(11, 6))

    for c in curves:
        x = np.log2(c.scales)
        gate_tag = "gate PASS" if c.gate_passed else "gate FAIL"
        ax.plot(
            x,
            c.distances,
            "o-",
            color=c.color,
            label=f"{c.label} | slope={c.slope:+.4f} | {gate_tag}",
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("log2(b) (coarse-graining scale)")
    ax.set_ylabel("Distance (MMD or equivalent)")
    ax.set_title("Exp 50 Summary: Distance vs Scale (Gate-Certified Curves)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_exp50r_alpha_distributions(out_path: Path) -> None:
    """
    Plots for Exp 50r:
    - Histogram of alpha at b=1 for KPZ-A, KPZ-B, KS.
    - Mean alpha vs scale b for each system.
    """
    meta = _read_json(ROOT / "results_exp50r" / "metadata.json")
    sigma = float(meta["sigma"])

    data_path = ROOT / "results_exp50r" / "kpzA_vs_kpzB_alpha_only.npz"
    with np.load(data_path) as d:
        scales = d["scales"].astype(int)
        kpzA = {int(b): d[f"KPZ_A_b{int(b)}"].reshape(-1) for b in scales}
        kpzB = {int(b): d[f"KPZ_B_b{int(b)}"].reshape(-1) for b in scales}
        ks = {int(b): d[f"KS_b{int(b)}"].reshape(-1) for b in scales}

    # Panel A: histogram at b=1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    b0 = int(scales[0])

    bins = np.linspace(0.0, 1.2, 35)
    ax.hist(kpzA[b0], bins=bins, alpha=0.55, label="KPZ-A", color="#d62728", density=True)
    ax.hist(kpzB[b0], bins=bins, alpha=0.55, label="KPZ-B", color="#ff9896", density=True)
    ax.hist(ks[b0], bins=bins, alpha=0.55, label="KS", color="#9467bd", density=True)

    ax.set_title("Exp 50r: alpha distributions at b=1")
    ax.set_xlabel("alpha")
    ax.set_ylabel("density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    # Panel B: mean alpha vs scale
    ax = axes[1]
    x = np.log2(scales.astype(float))
    ax.plot(x, [float(np.mean(kpzA[int(b)])) for b in scales], "o-", label="KPZ-A", color="#d62728")
    ax.plot(x, [float(np.mean(kpzB[int(b)])) for b in scales], "s-", label="KPZ-B", color="#ff9896")
    ax.plot(x, [float(np.mean(ks[int(b)])) for b in scales], "^-", label="KS", color="#9467bd")

    ax.set_title("Exp 50r: mean(alpha) vs scale")
    ax.set_xlabel("log2(b)")
    ax.set_ylabel("mean(alpha)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(f"Exp 50r Alpha-Only Positive Control (sigma={sigma:.4f})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_visual_index_md(out_path: Path, figure_paths: Iterable[Path]) -> None:
    rels = [p.relative_to(ROOT) for p in figure_paths]

    lines: list[str] = []
    lines.append("# Visual Overview\n")
    lines.append("This folder contains summary figures generated by `scripts/make_visual_overview.py`.\n")
    lines.append("## Figures\n")

    for rel in rels:
        lines.append(f"- `{rel.as_posix()}`")

    lines.append("\n## Notes\n")
    lines.append("- `pipeline_schematic.png`: conceptual flow from field laws to feature-laws and the diagnostic gate.")
    lines.append("- `scale_invariance_vs_universality.png`: conceptual distinction between gate passing and universality collapse.")
    lines.append("- `exp50_distance_summary.png`: key Exp 50 curves (only gate-certified comparisons).")
    lines.append("- `exp50r_alpha_distributions.png`: alpha-only distributions and mean vs scale.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    figures: list[Path] = []

    pipeline_png = OUT_DIR / "pipeline_schematic.png"
    plot_pipeline_schematic(pipeline_png)
    figures.append(pipeline_png)

    invariance_png = OUT_DIR / "scale_invariance_vs_universality.png"
    plot_scale_invariance_vs_universality(invariance_png)
    figures.append(invariance_png)

    exp50_png = OUT_DIR / "exp50_distance_summary.png"
    plot_exp50_distance_summary(exp50_png)
    figures.append(exp50_png)

    exp50r_png = OUT_DIR / "exp50r_alpha_distributions.png"
    plot_exp50r_alpha_distributions(exp50r_png)
    figures.append(exp50r_png)

    # One PDF bundle for quick scrolling.
    pdf_path = OUT_DIR / "visual_overview.pdf"
    with PdfPages(pdf_path) as pdf:
        for p in figures:
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    figures.append(pdf_path)

    index_md = OUT_DIR / "VISUAL_OVERVIEW.md"
    write_visual_index_md(index_md, figures)

    print(f"Wrote {len(figures)} outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()

