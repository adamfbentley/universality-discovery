# Manuscript Outline

Working title:

**Finite-Size Feature Geometry Can Misidentify Universality Classes in
Unsupervised Learning of Surface Growth**

Alternative title:

**A Protocol-Dependent False Positive in ML Universality Discovery**

## Abstract Shape

Unsupervised learning is often used to infer physical structure from simulated
or experimental data. In systems with universality, however, the target label is
an asymptotic equivalence relation, not necessarily a finite-data cluster. We
study 1+1D growth models spanning EW, KPZ, BD, Eden, RD, and KS dynamics. Across
spatial, temporal, height-distribution, and effective-exponent representations,
finite-size feature geometry is locally class-informative but not robustly
cluster-compatible with the intended universality labels. A single effective
exponent protocol appears to recover the classes with ARI 0.902, but a matched
five-seed protocol sweep reduces its advantage over raw multi-L features to
approximately zero. True matrix-archive subset refits show that EW/KPZ binary
separation is easy for the feature baselines, while the harder KPZ quotient
across continuum KPZ, BD, and Eden is not recovered. The benchmark shows that ML
universality discovery requires explicit invariance or finite-size-scaling
validation, not single-run clustering agreement.

## Section 1: Introduction

Goal:

- motivate unsupervised ML for physical discovery.
- state the core risk: clustering finds representation geometry, not necessarily
  physical equivalence.
- introduce universality as a quotient/RG-basin label.

Key message:

> Universality labels are quotient labels. Clustering labels are connected-density
> labels. These need not coincide in finite data.

## Section 2: Physical Benchmark

Models:

- EW
- continuum KPZ
- ballistic deposition
- Eden
- random deposition
- Kuramoto-Sivashinsky

Physics constraints:

- EW/KPZ stationary slope degeneracy in 1D.
- alpha = 1/2 for EW and KPZ, so roughness alone is not enough.
- BD/Eden have slow corrections to scaling.
- RD is a trivial easy class.
- KS is a finite-size/crossover/chaotic system, not a clean universality label.

## Section 3: Representations And ML Protocols

Representations:

- spatial morphology features.
- temporal features.
- height-distribution features.
- raw multi-L features.
- effective exponent geometry.

ML methods:

- HDBSCAN.
- KMeans.
- kNN as a local-informativeness probe, not an unsupervised discovery method.

Definitions to introduce:

- local class-informativeness.
- cluster compatibility.
- quotient/RG compatibility.

## Section 4: Feature Geometry Is Locally Informative But Globally Wrong

Evidence:

- Exp62 baseline.
- Exp65 robustness.
- Exp66 finite-size sweep.
- Exp67 height features.
- MLP-05 local-vs-global audit.
- MLP-08 clusterer and hierarchy controls.
- MLP-09 true exp70 feature-matrix subset refits.

Main figure:

- local-vs-global plot: kNN high, HDBSCAN/KMeans limited.

Main claim:

- finite-size feature geometry has signal, but not the right topology.
- EW/KPZ binary separation is not the hard test; merging continuum KPZ with
  discrete KPZ-class models is.

## Section 5: A Physics-Informed False Positive

Evidence:

- Exp68 continuum exponent success.
- Exp69 all-system exponent ARI 0.902.
- Exp70 matched recomputation.
- Exp71 protocol sweep.
- MLP-09 subset refits showing no all-six exponent advantage on average, but a
  modest hard-subset advantage over feature baselines.

Main figure:

- false-positive arc: feature ceiling -> exp69 high point -> exp71 sweep.

Main claim:

- effective exponent coordinates can look like a universality representation
  under one protocol but are not robust at accessible finite sizes.

## Section 6: Why This Happens

Mathematical framing:

- finite data law: `P_{m,L,t}`.
- feature map: `Phi(P_{m,L,t})`.
- universality: asymptotic quotient under RG flow.
- failure: `Phi` does not factor through the quotient.

Physics mechanisms:

- stationary-slope degeneracy for EW/KPZ.
- correction-heavy discrete KPZ models.
- effective exponent instability.
- easy/outlier systems inflating ARI.

## Section 7: Positive Controls

Evidence:

- Ising FSS control.
- Potts/Binder control.

Message:

- The failure is not generic ML incompetence. It is specific to the mismatch
  between finite surface-growth representations and the RG quotient.

## Section 8: Implications For ML Physical Discovery

Checklist for future papers:

- report seed/protocol sweeps.
- separate local predictability from clustering.
- report hard subsets without easy/outlier classes.
- test finite-size scaling explicitly.
- test whether the representation is invariant along known same-class nuisance
  directions.
- avoid single-run ARI as evidence of physical discovery.

## Conclusion

Core conclusion:

> Clustering finite-size observations can reveal stable finite-data structure
> without recovering the physical equivalence relation. For universality,
> representation learning must approximate RG invariants or scaling-collapse
> structure; otherwise high clustering scores can be finite-size mirages.
