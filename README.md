# Universality Discovery

[![Tests](https://github.com/adamfbentley/universality-discovery/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/adamfbentley/universality-discovery/actions/workflows/tests.yml)

Research notebook exploring whether unsupervised methods can recover
universality-class structure from simulated physics data.

Instead of asking whether one simulator fits a known theory, this project asks a
harder question: if the labels are hidden, do physically motivated features make
systems with the same large-scale behavior organize together?

## Current Takeaway

The answer is mixed and useful, with the strongest current direction now on the
machine-learning side.

Local feature representations are often strongly discriminative, but the main
density-clustering hypothesis does not cleanly hold for the surface-growth
systems tested here. In the best current surface-growth experiments, k-nearest
neighbor classification can be high, while HDBSCAN and KMeans still hit an
Adjusted Rand Index ceiling around 0.5. Later diagnostics suggest this is not
just an algorithm choice: the KPZ class is geometrically multimodal in the tested
feature space, especially because ballistic deposition keeps a discrete-model
signature even when it belongs to the KPZ universality class.

The newer ML-paper track reframes this as a quotient-learning problem:
universality classes are asymptotic physical equivalence classes, while
off-the-shelf clustering finds finite-data geometry in the chosen representation.
The latest controls show that binary EW/KPZ separation can be easy, but the
hard KPZ quotient across continuum KPZ, ballistic deposition, and Eden remains
poorly recovered by finite-size feature clustering. Effective exponent geometry
helps on that hard subset, but is protocol-sensitive and no longer supports a
stable positive "universality recovery" headline.

That negative result is the most important result in the repository. It turns
the project from "unsupervised discovery works" into a more careful study of
when feature geometry agrees with physical universality, and when it does not.

## What This Project Shows

- Raw-field autoencoders mostly learned simulator artifacts rather than
  universality structure.
- Hand-designed spatial gradient features are much more useful, but they still
  conflate physical classes with implementation details.
- Temporal features improve local discrimination between EW and KPZ-style
  dynamics, but do not automatically create clean density clusters.
- Same-class controls are essential: some apparently strong cross-class results
  were later traced to normalization, bandwidth, or numerical-pipeline effects.
- The Ising finite-size-scaling experiments are the cleanest positive result:
  unsupervised PCA features recover a correlation-length exponent near the exact
  value, while the Potts experiments show a method boundary.
- The ML-focused controls separate local signal from quotient recovery: nearest
  neighbors can be mostly class-consistent even when global clusters remain
  incompatible with the intended universality labels.

## Representative Results

- **Exp 62: feature-space clustering.** Six spatial gradient features produce
  stable partial structure across EW, KPZ, ballistic deposition, Eden, random
  deposition, and Kuramoto-Sivashinsky simulations. HDBSCAN reaches ARI ~= 0.495
  and 3-NN accuracy ~= 82%.
- **Exp 63: temporal features.** Adding beta, velocity skew/kurtosis, and
  slope-growth coupling raises 3-NN accuracy to about 98%, but HDBSCAN remains
  near the same ARI ceiling on the full run.
- **Exp 64: multiscale/peel diagnostics.** Coarse-graining and hierarchical
  peeling show that the KPZ class can be disconnected in feature space. This
  supports the interpretation that the clustering limit is structural, not just
  a failed hyperparameter choice.
- **Exp 69-71: effective exponent geometry.** A single exp69 protocol gives a
  tempting exponent-geometry ARI of 0.902, but matched seed/protocol sweeps reduce
  its average advantage over raw multi-L features to approximately zero.
- **MLP 05/08/09: quotient diagnostics.** Local-vs-global, clusterer, hierarchy,
  and true exp70 matrix-refit controls show the obstruction is not simply the
  choice of KMeans or HDBSCAN. In the matrix-refit audit, EW/KPZ binary feature
  ARI reaches 1.0, but EW/KPZ/BD/Eden feature ARI remains about 0.17-0.19; matched
  effective exponents improve that hard subset only to about 0.44-0.50.
- **Exp 52d: Ising PCA-FSS.** PCA features recover nu ~= 1.07 for the 2D Ising
  model, about 7% from the exact value.
- **Exp 55-60: Potts controls.** Standard Binder analysis works, but the PCA-FSS
  approach does not transfer cleanly to 3-state Potts, which helps define the
  boundary of the method.

## Current Direction: Extrapolation Limits (Exp 76-79)

Exp 72-75 traced the BD split to a physical mechanism (intrinsic anomalous
roughening at small L) and ended with a negative: correction-to-scaling fits
cannot recover even the known EW/KPZ alpha = 0.5 from L <= 256 ladders,
because the result depends on the assumed correction form. The follow-up
question was whether anything can do better, and if not, why not.

- **Exp 76: amortized extrapolation.** Instead of fitting one correction form,
  train a regressor on synthetic W_sat(L) ladders drawn from several
  correction families, then ask it for alpha. On synthetic tests it beats the
  direct fits (RMSE 0.106 vs 0.165 for the best fixed-omega fit). On 24-seed
  regenerated ladders it gives EW 0.53, Eden 0.49, BD 0.52 — BD lands on its
  KPZ-class value, which no direct fit manages (they scatter 0.36-0.70 with
  the choice of ansatz). KPZ misses (0.62); exp78 traces that to the
  integrator, not the estimator.
- **Exp 77: a resolution floor.** A Le Cam two-point bound makes the
  difficulty quantitative: at L <= 256, exponent differences of ~0.1 can be
  absorbed almost exactly by ordinary correction terms, so resolving them
  would take on the order of 10^5 seeds at realistic noise. Extra statistics
  barely help; window length and assumptions about corrections are what buy
  resolution. This turns the earlier negative results from "our methods
  failed" into "no method could have succeeded at these sizes".
- **Exp 78: checks.** A discriminability control (the estimator separates
  true alpha 0.40 / 0.45 / 0.50 on BD-like ladders, so the BD recovery is not
  shrinkage toward the prior mean, though a +0.05 conditional bias widens the
  honest error bar); an exact-measure ch