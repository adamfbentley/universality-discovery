# Execution prompt: exp83 — audit-response computations

Paste everything below the line into a fresh Sonnet session with repo access.
Context: an adversarial statistical audit (archive/ai_execution/SIMULATED_AUDIT.md) and
the theory note's new Appendix G identified five computations that harden the
floor paper. Each is small; all five are gated. The reviewer (Fable) audits
the output; report failed gates honestly.

---

You are implementing exp83 in `universality-discovery`. Read first:
`archive/ai_execution/SIMULATED_AUDIT.md`, `ml_paper/THEORY_minimax_floor.md` Appendix G
(and the main Le Cam section), `experiments/77_minimax_floor.py`, and the
"Sandbox gotchas" in `archive/ai_execution/EXP76_HANDOFF.md` (all still apply; also note:
if `git` reports a corrupt index, use `GIT_INDEX_FILE=/tmp/gidx git ...` and
rebuild `.git/index` after). Work on branch `exp81`. Do not touch
CLAIMS_REGISTER.md or MANUSCRIPT_OUTLINE.md. Output:
`experiments/83_audit_response.py`, `results_exp83_audit_response/*.json`,
`ml_paper/EXP83_REPORT.md` (same gate-ledger format as EXP81_REPORT.md).

## Task 1 — D² monotonicity guard (G-83a)

Add a Δα-monotonicity assertion to the confusion-gap path used for floors
(scan the Δα grid, assert nondecreasing D² up to optimizer tolerance; apply a
running max before bisection). Reproduce Appendix G5's independent scan
(11 points, Δα ∈ [0.02, 0.6], BD design/bounds) and confirm: monotone, BD
m=24 floor 0.27 ± one bisection step. Gate: matches G5's table within
optimizer scatter.

## Task 2 — Convex-hull floor vs class floor (G-83b)

Donoho sharpness needs a convex class; ours is a curved 2-parameter family.
Compute the hull floor by extending the adversary to convex combinations of
J atoms: g_hull(x) = Σ_j λ_j·log(1 + u_j·e^{−ω_j(x−x₁)}), λ_j ≥ 0,
Σλ_j = 1, each |u_j| within the class bounds. Run J = 1, 2, 4, 8 (J=1 must
reproduce the class floor — that is the nesting gate). Report
hull_floor(J)/class_floor for the BD design at m ∈ {6, 24}, U ∈ {1, 4}.
Gates: hull floor nondecreasing in J (nesting); J=1 anchor. Finding to
report either way: if the hull gap is small (< ~1.3×), Donoho affine
near-sharpness transfers to the physical class with little loss — the
sharpness program closes cheaply; if large, say so plainly.

## Task 3 — Van Trees (Bayesian Cramér–Rao) bound under the exp76 prior (G-83c)

The amortized estimator is judged by average RMSE; the Le Cam floor is
worst-case — mismatched objects (exp81 audit, anomaly 9). Compute the
matching average-case bound: van Trees over the full parameter vector
(α, c, correction params) with the exp76 sampling prior (families and bounds
as in `76_amortized_extrapolation.py`; if the exact prior is awkward, use
the F1 single-power family with its documented bounds and SAY SO). Gaussian
noise ⇒ the data-information term is the prior-averaged Fisher matrix;
report the α-marginal bound 1/√(VT_αα) at the real per-system σ, m ∈ {6,24}.
Gates: bound ≤ the fixed-nuisance CR bound (information can only decrease
with unknown nuisances — sanity direction); bound vs amortized synthetic
RMSE 0.106: estimator must NOT beat the bound (if it does, stop, diagnose,
report — that is a bug in bound or evaluation, not a result).

## Task 4 — Gaussianity diagnostic: the correction factor κ (G-83d)

From `results_exp76_amortized_extrapolation/wsat_perseed.csv`: per (system,
L), compute skewness and excess kurtosis of log W_sat across the 24 seeds
(with bootstrap CIs). Then estimate the Fisher information of the seed-mean
noise numerically: bootstrap seed-means (m=24 resamples), fit a kernel
density, compute I = E[(score)²], report κ = I·σ̄² per system (κ = 1 is
Gaussian; κ > 1 means the Gaussian floor overstates impossibility by √κ in
Δα). Label everything DIAGNOSTIC — a kernel-density score estimate at these
sample sizes is indicative, not theorem-grade; report CIs from the double
bootstrap and say if they straddle 1.

## Task 5 — Plug-in σ propagation (G-83e)

σ̂ per system uses ~24 seeds: propagate its sampling error (χ², m−1 dof per
L, combined across the 7 L's) into a floor interval: recompute the floor at
the 5th and 95th percentile σ. Report floor as [lo, mid, hi] per system at
m=24. Gate: interval contains the point floor; width consistent with the
√(2/(n_eff·(m−1))) relative error of σ̂.

## Report

`ml_paper/EXP83_REPORT.md`: gate ledger (every gate, PASS/FAIL, number,
proof file); one table per task; anomalies and deviations; "what we did not
do." No claims language. Every number five-tupled as in exp81. Commit on
branch `exp81`, message style matching the repo.
