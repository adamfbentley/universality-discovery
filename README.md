# Universality Discovery

[![Tests](https://github.com/adamfbentley/universality-discovery/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/adamfbentley/universality-discovery/actions/workflows/tests.yml)

Research notebook exploring whether unsupervised methods can recover
universality-class structure from simulated physics data, and what the
information-theoretic limits of that recovery are.

## Current Result: Minimax Resolution Floor for Finite-Size Scaling

The main result of the current phase (exp76–80) is a **computable minimax
resolution floor** for finite-size scaling (FSS) exponent estimation:

> At accessible system sizes (L ≤ 256), no estimator whatsoever can distinguish
> roughness exponents closer than Δα* — because adversarially chosen corrections
> to scaling make the data distributions statistically indistinguishable. At
> L ≤ 256, the floor is 0.27–0.44, large enough to explain every empirical
> negative in the project. It shrinks at a quantified rate with L_max (decades,
> not seeds).

This converts the project's empirical negatives into a theorem. The floor is
derived from Le Cam's two-point bound, computed by direct adversarial
optimization (`experiments/77_minimax_floor.py`), and documented in
`ml_paper/THEORY_minimax_floor.md` (including the algebraic mechanism, scaling
law, and connections to RG theory and sloppy models).

The companion positive result is an amortized estimator (trained on a prior over
correction families) that recovers BD's roughness exponent — α̂ = 0.522, honest
interval ≈ 0.50 ± 0.05 (syst) ± 0.03 (stat) — where classical ansatz-fitting
gives a scatter of 0.36–0.70. The estimator operates near the information limit
for the declared correction class.

## Next Phase: The Observable-Information Hierarchy (exp81, planned)

The floor is estimator-agnostic but **observable-conditional**: it bounds
inference from the declared summary (the W_sat ladder), not from richer data.
Exp81 turns that scope condition into the next research object: compute floors
across the observable hierarchy (single summary → multi-channel summaries →
spectra → raw configurations) on an exactly solvable fractional-EW testbed
where every level has closed-form information, and determine when richer
observables buy exponent resolution. Plan: `ml_paper/EXP81_PLAN.md`; theory:
`ml_paper/THEORY_minimax_floor.md` Appendix F; execution prompt:
`ml_paper/SONNET_EXP81_PROMPT.md`.

## Earlier Result: Clustering Negative (exp62–75)

The clustering phase showed that finite-size feature geometry can be locally
discriminative but globally incompatible with universality labels.

Local feature representations are often strongly discriminative, but the main
density-clustering hypothesis does not cleanly hold for the surface-growth
systems tested. In the best surface-growth experiments, k-nearest-neighbor
classification can be high (3-NN accuracy ~98% in exp63), while HDBSCAN and
KMeans still hit an ARI ceiling around 0.5. Later diagnostics showed this is
structural: the KPZ class is geometrically multimodal in the tested feature
space, especially because ballistic deposition (BD) carries a discrete-model
signature even at finite L. A protocol-dependent false positive (exp69: ARI
0.902) was traced to a single-seed protocol and collapsed to near-zero advantage
under a matched sweep (exp71).

That negative result prompted the current theory phase: why is FSS-based
exponent recovery hard? The answer is the floor theorem above.

## What This Project Shows

- Raw-field autoencoders learned simulator artifacts rather than universality
  structure; hand-designed spatial gradient features are more useful but still
  conflate physical classes with implementation details.
- Temporal features improve local discrimination between EW and KPZ-style
  dynamics but do not create clean density clusters.
- The KPZ quotient across continuum KPZ, BD, and Eden is the hard case; EW/KPZ
  binary separation is easy in most representations.
- Classical correction-to-scaling extrapolation cannot recover known exponents
  (EW/KPZ α=0.5) from L ≤ 256: systematic uncertainty from correction-form
  misspecification dominates statistics by ~10×.
- The floor theorem gives the information-theoretic reason: D²(0.1) ≈ 1.3×10⁻⁷,
  so resolving Δα = 0.1 would require ~160,000 seeds.
- An amortized estimator that declares a correction prior bridges the
  Bayes–minimax gap, with its interval sitting just above the floor for the
  declared correction class.
- The Ising finite-size-scaling experiments (exp52d) are the cleanest positive
  result: PCA features recover ν ≈ 1.07, ~7% from the exact value.
- The floor framework explains Ising precision: exact-solution constraints on
  the correction class are worth ~5–6× in resolution at that design.

## Representative Results

- **Exp 62: feature-space clustering.** Six spatial gradient features produce
  partial structure. HDBSCAN reaches ARI ~0.495 and 3-NN accuracy ~82%.
- **Exp 63: temporal features.** 3-NN accuracy rises to ~98%, but HDBSCAN ARI
  stays near the same ceiling.
- **Exp 69/71: effective exponent geometry.** Single-protocol ARI 0.902 collapses
  to near parity with raw multi-L features under a matched five-seed sweep.
- **Exp 75: classical extrapolation fails.** ω=1 fit gives EW→0.70, KPZ→0.59
  (known 0.5). Systematic uncertainty dominates statistics by ~10×.
- **Exp 76: amortized estimator recovers BD.** BD: α̂ = 0.522 [0.482, 0.529];
  honest α̂ ≈ 0.50 ± 0.05(syst) ± 0.03(stat). Classical scatter: 0.36–0.70.
  Leave-one-family-out: robust to prior family choice.
- **Exp 77: minimax floor.** D²(0.1) ≈ 1.3×10⁻⁷. Floor m=24: BD 0.27,
  others 0.44. Nearly flat in seeds; falls 0.44→0.14 from L_max 256→16384.
  Value of correction knowledge: u_max 4→0.1 shrinks floor 0.44→0.077.
- **Exp 78: referee checks.** KPZ gate failure attributed to Lam–Shin integrator
  stationary-measure distortion. Expert W²=aL+b ansatz rejected on BD
  (χ²/dof=6.2). Discriminability: Δα=0.05 separated at ~3σ; response slope 1.2
  (no shrinkage). Slice-conditional bias +0.05 documented.
- **Exp 79: N-scaling law.** E_N = c_N·√T·U·(ΔαT/U)^{N+1} numerically verified;
  c₁=0.0375 (exact), c₂=0.0216 (certified). Key identity: Richardson residual is
  Δα·(1-e^{-ωx})^N; substitution t=e^{-ωx} reduces to polynomial approximation
  of log.
- **Exp 80: floor transfers.** β floor ≈0.07–0.08 (EW/KPZ β-gap at floor);
  Ising ν: agnostic floor 0.31–0.39, strict floor 0.05–0.13 — exp52d's 7.3%
  deviation is consistent with the strict bound.
- **Exp 52d: Ising PCA-FSS.** PCA features recover ν ≈ 1.07, about 7% from the
  exact value. Positive control.

## Repository Structure

```text
experiments/              Numbered experiment scripts, each mostly self-contained
  52d_ising_finite_size_scaling.py   Ising PCA-FSS positive control
  62_feature_space_clustering.py     Spatial gradient feature clustering
  63_temporal_features.py            Temporal feature benchmark
  64_multiscale_peel.py              Coarse-graining and hierarchical peeling
  75_correction_to_scaling.py        Classical ansatz fits (exp75)
  76_amortized_extrapolation.py      Amortized estimator (exp76, main result)
  76b_regenerate_ladders.py          Per-seed W_sat generation (exp76b)
  77_minimax_floor.py                Le Cam floor computation (exp77)
  78_referee_checks.py               Referee-proofing checks (exp78)
  79_lemma_scaling_test.py           N-scaling law verification (exp79)
  79b_constant_certificates.py       c_1, c_2, c_3 certificates (exp79b)
  80_second_observable_floors.py     beta and nu floor transfers (exp80)

src/simulation/           Shared simulation utilities
src/models/               Autoencoder architectures (early experiments)
src/analysis/             Clustering and analysis helpers

ml_paper/
  THEORY_minimax_floor.md   Full theory note (~620 lines): derivation, scaling
                             law, Appendices A–D (construction, RG, prior art,
                             algebraic details)
  CLAIMS_REGISTER.md        All claims mapped to evidence (two-part: clustering
                             exp62–75, floor theorem exp76–80)
  MANUSCRIPT_OUTLINE.md     Paper outlines: floor paper (primary, MLST target)
                             + clustering paper (secondary)
  EXP76_HANDOFF.md          Complete project state and open items

docs/EXPERIMENT_LOG.md    Chronological research notes (exp1–80)
docs/literature_review.md Literature notes

results_exp76_amortized_extrapolation/
  wsat_perseed.csv          673 rows, 4 systems × 7 L × 24 seeds
  summary_full24seed.json   Full evaluation results
  lofo_control.json         Leave-one-family-out control
  referee_checks.json       Exp78 check results

results_exp77_minimax_floor/
  floor.json                All floor numbers

results_exp80_second_observable_floors/
  floors.json               Beta and nu floor transfers

archive/                  Obsolete manuscript drafts and earlier writeups
tests/                    Lightweight smoke tests
```

## Running

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run smoke tests:

```bash
python -m pytest -q
```

### Current-phase experiments (floor theorem arc, exp75–80)

```bash
# Classical fits (exp75 — shows failure)
python experiments/75_correction_to_scaling.py

# Amortized estimator (exp76 — full pipeline)
python experiments/76_amortized_extrapolation.py --stage gen
python experiments/76_amortized_extrapolation.py --stage train --prior mix
python experiments/76_amortized_extrapolation.py --stage eval

# Minimax floor computation (exp77)
python experiments/77_minimax_floor.py --part all

# Referee checks (exp78)
python experiments/78_referee_checks.py

# N-scaling law verification (exp79)
python experiments/79_lemma_scaling_test.py
python experiments/79b_constant_certificates.py

# Second-observable transfers (exp80)
python experiments/80_second_observable_floors.py
```

### Earlier clustering experiments

```bash
python experiments/62_feature_space_clustering.py --pilot
python experiments/63_temporal_features.py --pilot
python experiments/64_multiscale_peel.py --pilot
python experiments/52d_ising_finite_size_scaling.py
```

The experiment scripts compile Numba kernels on first run.

## Theory Note

`ml_paper/THEORY_minimax_floor.md` contains the full derivation of the
minimax resolution floor, including:

- Le Cam two-point bound → confusion gap D² → resolution floor Δα*.
- Computed floors for the surface-growth benchmark.
- Mechanism: why D² is ~10⁻⁷ (log L shadowed by correction ω→0 tail).
- Appendix A: Richardson construction, Legendre projection, amplitude binding.
- Appendix B: RG-equivariance corollary and Fisher/sloppy-models connection.
- Appendix C: positioning map (Lepage, Jay–Neil, Manski, Tikhonov, Sethna).
- Appendix D: binomial identity, log-polynomial reduction, optimal nodes.

## Relationship To Earlier Work

This repository builds on
[ml-universality-classification](https://github.com/adamfbentley/ml-universality-classification),
which tested supervised and anomaly-detection approaches for surface-growth
simulations. This project is broader and more exploratory: it investigates when
feature geometry aligns with physical universality, documents the cases where it
does not, and derives the information-theoretic reason.
