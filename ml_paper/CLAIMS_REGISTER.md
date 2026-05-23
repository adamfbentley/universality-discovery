# Claims Register

This file is the guardrail for the ML-focused paper. Every manuscript claim must
map to an artifact and a robustness check.

## Central Claim

Finite-size feature geometry in this growth benchmark does not robustly factor
through the RG universality quotient. The failure is not a mere absence of
signal: local class information can be strong while global cluster structure
remains incompatible with the intended universality labels.

In ML language, universality labels are quotient labels, while standard
clustering expects cluster labels. This paper asks when a representation makes
that quotient visible, and documents a case where common finite-size
representations do not.

Required evidence:

- Feature clustering ceiling near ARI 0.5 across feature families and sizes.
- High kNN/local separability coexisting with low HDBSCAN/KMeans universality ARI.
- KPZ-class multimodality/disconnection in finite feature space.
- Effective-exponent false positive: exp69 high ARI reproduced under one protocol.
- Protocol sweep: exp71 reduces matched exponent advantage to parity with raw
  multi-L feature baseline.
- Hard-subset audit: removing RD and/or KS does not produce a stable, clean
  quotient-compatible clustering story; the EW/KPZ/BD/Eden subset remains weak
  for spatial features and only moderate/protocol-sensitive for effective
  exponent refits.
- Local-vs-global audit: across representations with kNN measurements, the mean
  kNN3-minus-HDBSCAN gap is `0.389`; in exp62, 1-neighbor universality purity is
  `0.842` while KMeans ARI is `0.185` and HDBSCAN core ARI is `0.495`.
- Clusterer/hierarchy control: replacing KMeans with standard fixed-k
  clusterers does not rescue exp62 feature geometry. The best all-six exp62
  fixed-k universality ARI is `0.250`, and the EW/KPZ/BD/Eden subset remains
  `-0.0687`. The exp62 centroid hierarchy never cleanly merges KPZ/BD/Eden
  before mixing non-KPZ systems.
- True exp70 feature-matrix subset refits: feature baselines can perfectly
  separate EW from continuum KPZ, but they do not recover the full KPZ quotient
  across continuum KPZ/BD/Eden. On EW/KPZ/BD/Eden, the best feature KMeans ARI is
  `0.186` under equal sampling and `0.169` under exp69 sampling, while matched
  effective exponents are `0.503` and `0.438`, respectively.

## Claims Allowed

1. **Representation geometry, not clustering alone, is the bottleneck.**
   Evidence: exp62, exp65, exp66, exp67, exp70, exp71.

2. **Local discriminability does not imply global universality recovery.**
   Evidence: high kNN in exp66/exp67 with HDBSCAN core ARI near 0.5, plus
   MLP-05. In exp62, 1-neighbor universality-class purity is `0.842`, but the
   same representation gives KMeans ARI `0.185` and HDBSCAN core ARI `0.495`.
   The average local-minus-HDBSCAN gap across rows with kNN measurements is
   `0.389`.

3. **Single-run high ARI can be a false positive in finite-size ML universality discovery.**
   Evidence: exp69 ARI 0.902 vs exp71 five-seed matched sweep.

4. **Effective exponent geometry is finite-size/protocol sensitive at L <= 128.**
   Evidence: exp70/exp71 matched exponent vectors and ARI intervals.

5. **Positive controls show the general workflow can work on cleaner FSS tasks.**
   Evidence: Ising exp52d and Potts/Binder exp57c.

6. **The result is finite-size scoped.**
   Evidence: all surface-growth simulations are at accessible finite L, T.

7. **The negative result is not solely an RD/KS artifact.**
   Evidence: MLP-02 hard-subset audit. In `subset_ari_summary.md`, exp62
   spatial-feature refits on EW/KPZ/BD/Eden give KMeans ARI `-0.0687` with kNN3
   accuracy `0.775`; matched exponent refits on the same subset average only
   `0.504` under equal sampling and `0.438` under exp69 sampling, with large
   seed spread.

8. **For the exp62 spatial representation, local KPZ purity mostly reflects
   microscopic-system neighborhoods rather than quotient mixing.**
   Evidence: MLP-05. Among KPZ-labeled points at `k=1`, KPZ-class neighbor
   purity is `0.833`, but cross-system KPZ/BD/Eden mixing is only `0.0125`.
   The feature-centroid quotient ratio is `24.7`, far above the desired
   below-one regime.

9. **The exp62 feature-space failure is not rescued by standard clusterer choice
   or a simple centroid hierarchy.**
   Evidence: MLP-08. Across KMeans, Gaussian mixtures, agglomerative variants,
   spectral clustering, and HDBSCAN, the best all-six fixed-k universality ARI
   for exp62 features is `0.250`. On EW/KPZ/BD/Eden it remains `-0.0687`.
   The KPZ centroid quotient ratio is `24.7` on all six and `37.0` on
   EW/KPZ/BD/Eden, with clean hierarchy fraction `0`.

10. **The hard continuum-vs-discrete KPZ quotient, not EW/KPZ binary separation,
    is the central obstruction in exp70 features.**
    Evidence: MLP-09 true matrix subset refits. Feature baselines reach KMeans
    ARI `1.0` on EW/KPZ only, but on EW/KPZ/BD/Eden the best feature KMeans ARI
    is only `0.186` under equal sampling and `0.169` under exp69 sampling.
    Matched effective exponents improve that hard subset to `0.503` and `0.438`,
    but remain protocol-sensitive and far from robust recovery.

## Claims Forbidden Unless New Evidence Is Added

1. **"Unsupervised ML cannot discover universality."**
   Too broad. The evidence only covers the tested representations and finite-size
   regimes.

2. **"Collapse geometry solves universality recovery."**
   False under exp71.

3. **"The exp69 advantage is representational, not informational."**
   Not supported. Raw multi-L features tie matched exponent geometry on average.

4. **"The current exponent vector method is a collapse metric."**
   Incorrect. It clusters fitted effective exponents, not collapse residuals.

5. **"The work discovers new KPZ physics."**
   Overclaim. The contribution is methodological and finite-size diagnostic.

## Claims That Require Additional Experiments

1. **True collapse-residual distances outperform finite-size features.**
   Requires implementation of a master-curve residual metric and repeated seed
   sweeps.

2. **Stored legacy exp70 artifacts alone establish true feature subset refits.**
   Resolved for new matrix-archive runs by MLP-09, but legacy artifacts remain
   label-only. Manuscript claims should cite the `codex_matrix_full_*` artifacts
   for true subset refits.

3. **The exponent instability is driven by specific systems.**
   Requires per-system exponent-cloud figures and between-cloud distance tables.

4. **Learned cross-L baselines cannot recover the quotient.**
   Requires stronger representation-learning baselines; current raw/slope/PCA
   controls are not exhaustive.

5. **No unsupervised method can recover the quotient from these data.**
   Not supported. MLP-08 tests only standard off-the-shelf clusterers on stored
   exp62 features and exponent vectors. A tailored representation learner or
   true collapse-residual distance remains outside the current evidence.

## One-Sentence Paper Claim

Unsupervised clustering of finite-size growth observables can find stable local
and protocol-specific structure without recovering the physical universality
equivalence relation; robust ML universality discovery requires explicit
invariance or finite-size-scaling validation.

## One-Sentence ML Research Question

When, and how can we test whether, a learned representation factors through the
physical quotient rather than merely organizing nuisance geometry?
