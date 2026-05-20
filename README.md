# Universality Discovery

Research notebook exploring whether unsupervised methods can recover
universality-class structure from simulated physics data.

Instead of asking whether one simulator fits a known theory, this project asks a
harder question: if the labels are hidden, do physically motivated features make
systems with the same large-scale behavior organize together?

## Current Takeaway

The answer is mixed and useful.

Local feature representations are often strongly discriminative, but the main
density-clustering hypothesis does not cleanly hold for the surface-growth
systems tested here. In the best current surface-growth experiments, k-nearest
neighbor classification can be high, while HDBSCAN and KMeans still hit an
Adjusted Rand Index ceiling around 0.5. Later diagnostics suggest this is not
just an algorithm choice: the KPZ class is geometrically multimodal in the tested
feature space, especially because ballistic deposition keeps a discrete-model
signature even when it belongs to the KPZ universality class.

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
- **Exp 52d: Ising PCA-FSS.** PCA features recover nu ~= 1.07 for the 2D Ising
  model, about 7% from the exact value.
- **Exp 55-60: Potts controls.** Standard Binder analysis works, but the PCA-FSS
  approach does not transfer cleanly to 3-state Potts, which helps define the
  boundary of the method.

## Repository Structure

```text
experiments/              Numbered experiment scripts, each mostly self-contained
src/simulation/           Shared simulation utilities
src/models/               Autoencoder architectures used in early experiments
src/analysis/             Clustering and analysis helpers
docs/EXPERIMENT_LOG.md    Chronological research notes and corrections
docs/literature_review.md Literature notes
results*/                 Selected result snapshots used to document experiments
archive/                  Obsolete manuscript drafts and earlier writeups
tests/                    Lightweight smoke tests for import and core behavior
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

Run selected experiments:

```bash
python experiments/62_feature_space_clustering.py --pilot
python experiments/63_temporal_features.py --pilot
python experiments/64_multiscale_peel.py --pilot
python experiments/52d_ising_finite_size_scaling.py
```

The experiment scripts compile Numba kernels on first run, so the first execution
is slower than later runs.

## Relationship To Earlier Work

This repository builds on
[ml-universality-classification](https://github.com/adamfbentley/ml-universality-classification),
which tested supervised and anomaly-detection approaches for surface-growth
simulations. This project is broader and more exploratory: it investigates when
unsupervised geometry aligns with known physics, and documents the cases where
that assumption breaks.
