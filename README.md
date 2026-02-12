# Universality Discovery

Exploring whether unsupervised methods can discover universality classes from physics simulation data. Instead of asking "does this data fit KPZ theory?", the question is "what dynamical classes exist in this data?"

## Overview

This project ran 62+ experiments on 1D surface growth models (Edwards-Wilkinson, KPZ, Ballistic Deposition, Eden, Random Deposition, Kuramoto-Sivashinsky) as well as equilibrium criticality (2D Ising, 3-state Potts) and active matter (Vicsek). The central question was whether clustering and dimensionality reduction on physically-motivated features could recover universality class structure without supervision.

The short answer: yes, but you need the right features. Raw-field autoencoders mostly detect discreteness artifacts. Hand-crafted gradient statistics (variance, skewness, kurtosis of spatial gradients and Laplacians) actually carry strong universality signal, validated across 60 experiments before being used for the final clustering result.

## Key results

- **Feature-space clustering works** (Exp 62): HDBSCAN on 6 gradient features finds exactly 4 clusters matching the 4 true universality classes (EW, KPZ, KS, trivial). ARI=0.495, kNN accuracy 82%, stable across pilot and full runs.
- **KPZ coupling coordinate** (Exp 46/54): PC1 of gradient features tracks D/ν with r=0.961, explained by the known stationary slope measure.
- **Ising PCA-FSS** (Exp 52d): Unsupervised PCA features give ν=1.073 via finite-size scaling collapse, ~7% from the exact value. Works even after removing the order parameter.
- **Diagnostic gate** (Exp 50 series): Self-consistency protocol that catches pipeline artifacts before you over-interpret cross-class comparisons. Born from painful experience with false positives in the KS campaign.
- **IC-dependence** (Exp 27): The universality signal depends heavily on initial conditions. Droplet IC gives near-perfect EW/KPZ separation (r=-0.98), flat IC at stationarity gives nothing (r=-0.06). Connects to GOE/GUE/Baik-Rains structure in KPZ fixed point theory.
- **Potts method boundary** (Exp 55-59): PCA-FSS fails for 3-state Potts despite working for Ising. Standard Binder analysis still works — the failure is method-specific, not physics-specific.

## What didn't work

Autoencoders on raw height fields (Exp 1-8, 61), total correlation of local observables (Exp 17-19), RG-geodesic metric learning (Exp 12), naive coarse-graining as RG (Exp 14, 30), various wavelet decompositions (Exp 9, 11). Each failure taught something, documented in the experiment log.

## Project structure

```
experiments/          # Numbered scripts (01-62), each self-contained
src/
  simulation/         # Growth models: EW, KPZ, BD, Eden, RD, KS (Numba JIT)
  models/             # Autoencoder architectures
  analysis/           # Clustering, PCA, dimension estimation
docs/
  EXPERIMENT_LOG.md   # Chronological notes from all experiments
  literature_review.md
  main.tex            # Manuscript draft
  references.bib
manuscript/           # Paper drafts (LaTeX)
results_*/            # Outputs from individual experiments
figures/              # Generated visualisations
```

## Dependencies

```
numpy, scipy, matplotlib, scikit-learn
torch
hdbscan
umap-learn
numba
```

## Running things

Each experiment is standalone:
```bash
python experiments/62_feature_space_clustering.py          # full run (takes a few minutes)
python experiments/62_feature_space_clustering.py --pilot   # quick test
python experiments/52d_ising_finite_size_scaling.py         # Ising FSS
```

Data generation involves physics simulations compiled with Numba — first run is slow (JIT compilation), subsequent runs are much faster. KS in particular can take a while.

## Related

Builds on [ml-universality-classification](https://github.com/adamfbentley/ml-universality-classification), which used supervised learning for KPZ classification (~95% accuracy). This project asks the harder question: can you do it without labels?
