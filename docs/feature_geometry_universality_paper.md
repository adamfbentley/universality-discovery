# Feature Geometry Does Not Equal Universality:
# Diagnostic Failures in Unsupervised Surface-Growth Classification

**Author:** Adam Bentley  
**Status:** Working manuscript draft, May 2026  
**Repository:** `universality-discovery`

**2026-05-24 status note:** This draft remains the negative-result physics
manuscript. The newer `ml_paper/` track extends the same evidence into a
machine-learning failure-mode paper: finite-size representations can be locally
class-informative while failing to factor through the intended universality
quotient. The latest controls should be cited from `ml_paper/CLAIMS_REGISTER.md`
and the MLP-05/08/09 artifacts before making any claims about quotient learning
or effective-exponent geometry.

## Abstract

Unsupervised learning is often presented as a route to discovering physical
structure without labels. In statistical physics, this raises a sharper
question: can feature-space geometry recover universality classes rather than
merely separate simulators, amplitudes, or finite-size morphologies? I study
this question using simulated one-dimensional surface-growth systems, including
Edwards-Wilkinson (EW), continuum Kardar-Parisi-Zhang (KPZ), ballistic
deposition (BD), Eden growth, random deposition (RD), and
Kuramoto-Sivashinsky-like dynamics (KS). Six spatial gradient features produce
stable partial structure: HDBSCAN finds four clusters with adjusted Rand index
ARI = 0.495 against the intended physical classes, while 3-nearest-neighbour
classification reaches 82.1%. Adding four temporal features designed to probe
growth exponents, velocity asymmetry, and slope-growth coupling raises local
classification to 97.7% and nearly triples KMeans ARI, but leaves HDBSCAN at
the same ARI ceiling. Multi-scale block coarse-graining and hierarchical
peeling do not remove this ceiling; the KPZ-labelled class is geometrically
multimodal, with BD forming a disconnected component at microscopic scale.

These results support a negative but scientifically useful conclusion:
finite-size feature representations can be highly discriminative while still
failing to organize samples by universality class. The failure is not simply an
algorithmic defect. In 1D KPZ, the stationary slope measure is Gaussian and
independent of the nonlinear coupling, so stationary gradient features mostly
track a noise-to-diffusion amplitude coordinate rather than KPZ nonlinearity.
The same repository contains positive controls in equilibrium systems: a
PCA-based finite-size-scaling analysis recovers the 2D Ising correlation-length
exponent within about 7%, while Binder analysis recovers the 3-state Potts
exponent within about 6%. Together, these experiments motivate a diagnostic
protocol for ML-assisted universality studies: distinguish local
discriminability, density-cluster geometry, scale invariance, and physical
universality before making discovery claims.

## 1. Introduction

Universality is the principle that systems with different microscopic details
can share the same large-scale behaviour. In equilibrium critical phenomena,
the renormalization group explains this through fixed points and relevant
directions: microscopic models can flow to the same asymptotic description even
when their local rules are different. In non-equilibrium surface growth, the
Kardar-Parisi-Zhang equation is a central example. The KPZ equation,

```text
partial_t h = nu nabla^2 h + (lambda/2) (nabla h)^2 + eta,
```

defines a broad universality class of stochastic interface growth. In 1+1
dimensions, the standard exponents are alpha = 1/2, beta = 1/3, and z = 3/2,
while the linear Edwards-Wilkinson equation has beta = 1/4 and z = 2 but shares
alpha = 1/2. This overlap already shows why universality classification cannot
be reduced to a single finite-size roughness estimate.

The practical problem is that asymptotic exponent extraction is difficult.
Finite systems may not reach clean scaling windows; transient regimes can
dominate measurements; fitted exponents depend on time and length windows; and
experimental data may be noisy or incomplete. Machine learning therefore offers
an attractive possibility: perhaps local or distributional features can identify
universality structure before asymptotic fitting becomes reliable.

This paper asks a deliberately restricted question:

> When labels are hidden, do physically motivated finite-size features organize
> simulated systems by known universality class?

The answer from this repository is mixed. Local features often separate systems
very well. However, the resulting geometry does not cleanly coincide with
universality classes. The central contribution is therefore not a new universal
metric. It is a failure analysis: a map of where feature geometry agrees with
physical universality, where it breaks, and what controls are needed before ML
classification can be interpreted physically.

## 2. Literature Review

### 2.1 KPZ, EW, and finite-size surface-growth diagnostics

Kardar, Parisi, and Zhang introduced the nonlinear stochastic growth equation
now known as KPZ in 1986. The equation adds a local slope-squared term to
linear diffusive relaxation, making it a minimal model of lateral growth in
noisy interfaces. Family and Vicsek gave the finite-size scaling form commonly
used for roughening analysis. Standard surface-growth references, especially
Barabasi and Stanley, develop the relationship between interface width,
roughness exponents, growth exponents, and microscopic growth models such as
BD and Eden growth. Halpin-Healy and Zhang's review is important background
because it makes clear that kinetic roughening is already a mature literature
with known crossover problems, finite-size corrections, and links to directed
polymers. The point of the present work is therefore not to rediscover KPZ, but
to test whether a particular machine-learning geometry respects the distinctions
that the surface-growth literature already knows how to make.

The exact and probabilistic understanding of 1D KPZ has grown substantially.
Prahofer and Spohn connected one-dimensional growth models to random-matrix
fluctuation distributions. Corwin reviews the KPZ equation and its universality
class, including connections to directed polymers and exact one-point
distributions. Matetski, Quastel, and Remenik constructed the KPZ fixed point,
showing that initial conditions are not a nuisance detail but part of the
scaling-limit structure. Takeuchi's review emphasizes the modern experimental
and theoretical picture of KPZ fluctuations.

Renormalization-group work is the other essential background. Medina, Hwa,
Kardar, and Zhang analyze KPZ/Burgers dynamics with correlated noise and show
why coupling constants, noise structure, and coarse-graining matter. This is the
theoretical reason the project avoids treating raw clustering geometry as a
physical invariant: an embedding coordinate can be discriminative and still be
dominated by a non-universal amplitude or a correction-to-scaling direction.

For this project, one piece of theory is especially important: in 1D with
periodic boundary conditions, the stationary slope field of KPZ has a Gaussian
stationary measure that is independent of the nonlinear coupling `lambda`.
Equivalently, under `u = partial_x h`, the noisy Burgers stationary measure
depends on the noise-to-diffusion ratio and not directly on `lambda`. This means
stationary gradient moments cannot by themselves be interpreted as direct
measurements of KPZ nonlinearity. If gradient features separate simulations at
finite time, the separation may reflect amplitude, transient dynamics,
discretization, or morphology rather than pure universality class membership.

### 2.2 Machine learning for phases and criticality

The ML-for-physics literature has already shown that learning algorithms can
recover meaningful physical structure. Wang used PCA on spin configurations to
identify phases and phase transitions in many-body systems. Carrasquilla and
Melko showed that supervised neural networks can classify phases of matter from
raw configurations. Hu, Singh, and Scalettar gave an important cautionary study:
PCA can reveal physically interpretable structure, but the interpretation
depends on the model, representation, and phase-transition type.

This matters because the present project is not the first to observe that PCA
or clustering can detect phase structure. The novelty cannot be "ML finds
phases." More defensible contributions are:

1. testing whether unsupervised feature geometry aligns with known surface-growth
   universality classes;
2. distinguishing discriminability from density-cluster recovery;
3. showing how same-class controls expose amplitude and implementation
   artifacts;
4. using equilibrium Ising and Potts systems as controls for when PCA/FSS works
   and when it fails.

Yue, Wang, and Lyu explicitly combine PCA-style representations with
finite-size scaling in Ising systems. This means that an Ising PCA-FSS result is
a useful positive control, not by itself a new discovery. Tirelli, Costa, and
Carlon study unsupervised learning in Potts models, reinforcing the point that
symmetry, representation, and model-specific corrections matter.

### 2.3 Scale-invariant is not the same as universal

A recurring lesson in this repository is that a feature can be scale-invariant
without being universal. A structure function may scale as

```text
S_2(r) = A r^(2 alpha),
```

where the exponent is universal but the prefactor `A` is not. Normalized
features can still retain non-universal correction terms. This distinction
explains why some same-class KPZ comparisons pass a superficial scale-invariance
gate while still remaining separated in feature space. Wegner's correction-to-
scaling framework and Privman-Fisher finite-size scaling make this point in a
more general equilibrium-critical setting: universal scaling functions and
exponents often coexist with metric factors, amplitudes, and finite-size
corrections that are not universal. The correct hierarchy is:

1. local discriminability: can a classifier separate samples?
2. density clustering: do unlabeled samples form clusters?
3. scale invariance: do feature distributions remain stable under coarse-graining?
4. universality: do same-class systems converge to the same asymptotic
   quantities or scaling functions?

Only the fourth supports a universality claim.

### 2.4 What the literature implies for this paper

The literature gives this paper a narrow but defensible role. It is not a new
derivation of the KPZ class, and it is not a replacement for renormalization
group or finite-size scaling analysis. Its contribution is methodological:

1. It tests a common ML-for-physics inference: that visible unsupervised
   geometry corresponds to physical universality.
2. It uses surface growth as a hard case because EW and KPZ share the 1D
   roughness exponent, while discrete KPZ-class models can have large
   finite-size morphology.
3. It treats successful Ising/Potts analyses as controls, not as novelty
   claims, because PCA, Binder cumulants, and finite-size scaling are already
   established tools.
4. It converts earlier anomaly-detection results into a more precise claim:
   the algorithms identify finite-size morphology and dynamics unless same-class
   controls show otherwise.
5. It frames negative clustering results as useful evidence about
   representation limits rather than as a failure of universality itself.

## 3. Defensible Scientific Claims

| Claim | Status | Evidence in this workspace | Safe wording |
|---|---:|---|---|
| Spatial gradient features contain useful finite-size information. | Supported | Exp 62: 3-NN accuracy 82.1%; HDBSCAN ARI 0.495. | "Spatial gradient features provide partial finite-size structure." |
| Temporal features improve local discrimination. | Supported | Exp 63: 3-NN accuracy rises to 97.7%; EW-KPZ centroid distance increases from 0.874 to 1.326. | "Temporal observables expose local dynamics not visible to stationary gradient moments." |
| Density clustering recovers universality classes. | Not supported | Exp 62-64: HDBSCAN remains near ARI 0.49; peel and multi-scale features do not exceed the ceiling. | "Density clustering does not cleanly recover the intended universality classes for these features." |
| The KPZ class is geometrically connected in the tested feature space. | Not supported at microscopic scale | Exp 64: KPZ-labelled class has two kNN components at block size 1; BD separates from continuum KPZ/Eden. | "Finite-size KPZ-class samples are geometrically multimodal." |
| Gradient features directly measure KPZ nonlinearity at stationarity. | Not supported | Exp 54 and the stationary slope-measure theorem: gradient variance tracks `D/nu` and is lambda-blind at stationarity. | "Stationary gradient features mainly measure an amplitude coordinate." |
| A universal ML distance has been discovered. | Not supported | Same-class controls and BD/Eden geometry contradict this. | "Anomaly or clustering scores are operational diagnostics, not universal invariants." |
| The pipeline can recover real critical structure in simpler systems. | Supported as a control | Exp 52d: Ising PCA-FSS gives nu = 1.073 vs exact 1; Exp 57c: Potts Binder gives nu = 0.884 vs exact 5/6. | "The method family can work under favourable observability conditions." |

## 4. Methods

### 4.1 Surface-growth systems

The main surface-growth experiments use six systems:

- EW: linear stochastic diffusion, treated as its own class.
- KPZ: continuum nonlinear growth equation.
- BD: discrete ballistic deposition, usually associated with the KPZ class.
- Eden: discrete growth model associated with KPZ scaling.
- RD: random deposition, treated as a trivial uncorrelated class.
- KS: Kuramoto-Sivashinsky-like dynamics, treated as a distinct nontrivial
  dynamical class in these experiments.

For clustering evaluation, the intended class map is:

- `EW`: EW
- `KPZ`: KPZ, BD, Eden
- `KS`: KS
- `trivial`: RD

This class map is intentionally demanding. It asks the algorithm to merge
discrete and continuum realizations that should be asymptotically related while
separating EW from KPZ-style nonlinear growth.

### 4.2 Spatial features

Exp 62 uses six spatial features computed from late-time height profiles:

1. gradient variance;
2. gradient skewness;
3. gradient excess kurtosis;
4. Laplacian variance;
5. covariance of absolute gradient and Laplacian;
6. height variance.

These features were chosen because they are simple, interpretable, and linked to
local morphology. They are not assumed to be universal.

### 4.3 Temporal features

Exp 63 adds four temporal features:

1. `beta_eff`: effective growth exponent from a log-log fit of width growth;
2. `vel_skew`: skewness of multi-step velocity increments;
3. `vel_kurt`: excess kurtosis of velocity increments;
4. `slope_growth`: Pearson correlation between local velocity and squared
   gradient.

The temporal features target the EW-KPZ degeneracy. If stationary gradient
moments are lambda-blind, then transient growth and velocity statistics should
carry more information about the nonlinear term.

### 4.4 Clustering and diagnostics

The analysis standardizes features and applies HDBSCAN, KMeans, KMeans sweeps,
and k-nearest-neighbour classification. The important distinction is:

- kNN tests local discriminability with labels during evaluation;
- HDBSCAN tests whether unlabeled density structure matches the intended
  physical classes;
- KMeans tests whether a fixed number of convex-ish clusters improves recovery;
- kNN graph connectivity tests whether a class is geometrically connected in
  the chosen feature representation.

This distinction is central: high kNN accuracy does not imply unsupervised class
discovery.

### 4.5 Multi-scale and control experiments

Exp 64 applies block coarse-graining at block sizes 1, 2, 4, and 8, then repeats
clustering on single-scale and concatenated multi-scale features. It also
performs a hierarchical "peel" by removing the easy RD and KS classes and
focusing on the hard EW/KPZ/BD/Eden subproblem.

Control experiments include:

- KPZ-vs-KPZ exponent-only comparison: roughness exponent features keep two KPZ
  parameter regimes close while separating KS.
- 2D Ising PCA-FSS: PCA features recover a correlation-length exponent near the
  exact value.
- 3-state Potts Binder analysis: standard Binder analysis works even where the
  PCA-FSS approach struggles, showing a method boundary rather than a failure
  of the simulated physics.

## 5. Results

### 5.1 Spatial features produce partial but not universal clustering

Exp 62 is the clean baseline. With six spatial features at `L = 256`,
`T = 2000`, and `N = 80` samples per system, HDBSCAN finds four clusters but
only reaches ARI = 0.495 against the intended class labels. KMeans with four
clusters gives ARI = 0.185. However, 3-NN accuracy reaches 82.1%.

This means the representation contains class-relevant local information, but
the density geometry does not match the intended universality grouping. The
centroid distances show the problem: EW and KPZ are close in the standardized
spatial feature space, while RD and KS are much easier to separate.

### 5.2 Temporal features improve discrimination but not density discovery

Exp 63 tests whether temporal features can break the EW-KPZ degeneracy. They
do improve local separability:

- 3-NN accuracy rises from 82.7% to 97.7%;
- KMeans ARI rises from 0.185 to 0.498;
- EW-KPZ centroid distance rises from 0.874 to 1.326.

The temporal means are physically interpretable. EW has near-zero velocity skew
and near-zero slope-growth correlation. KPZ has positive velocity skew and
positive slope-growth correlation. RD separates through high effective beta.
KS becomes anomalous through clipped/high temporal behaviour.

But HDBSCAN remains at ARI = 0.496. This is the crucial result. The temporal
features create local class information without creating density clusters that
match the intended universality labels.

### 5.3 Multi-scale coarse-graining does not rescue the clustering hypothesis

Exp 64 attacks two structural barriers: EW/KPZ degeneracy and BD splitting from
the continuum KPZ/Eden group. It uses block coarse-graining and hierarchical
peeling.

The results are negative:

- best ARI remains about 0.493;
- all-system HDBSCAN at block size 1 gives ARI = 0.487;
- all-system KMeans at block size 1 gives ARI = 0.493;
- 3-NN remains high, about 97.8%, so information is present locally;
- peeling RD and KS does not improve ARI and can produce negative ARI.

The kNN graph explains why. At block size 1, the KPZ-labelled class has two
connected components: BD is disconnected from KPZ/Eden. At block size 2, the
KPZ-labelled class becomes connected, but by block size 8 it fragments again.
Coarse-graining helps some spatial aspects of the discrete-continuum gap while
hurting temporal features and resolution.

The conclusion is not "HDBSCAN failed." It is stronger: the tested feature
geometry does not make universality classes into clean density clusters.

### 5.4 Positive controls: the pipeline can work when the observable is right

The negative surface-growth result is more convincing because the repository
also contains positive controls.

In Exp 52d, a PCA-based finite-size-scaling analysis of the 2D Ising model
recovers

```text
nu = 1.073
```

against the exact value `nu = 1`, an error of about 7.3%. This shows that
PCA-like features can recover quantitative critical structure when the
representation couples cleanly to the dominant scaling direction.

For the 3-state Potts model, the PCA-FSS approach is much less reliable. But
Exp 57c shows that standard Binder analysis recovers

```text
nu = 0.884
```

against the exact value `nu = 5/6`, an error of about 6.1%. This indicates that
the physics is visible to standard observables, while the PCA-FSS pipeline has
a representation or metric boundary.

### 5.5 Same-class controls expose the scale-invariant vs universal distinction

The KPZ-vs-KPZ exponent-only positive control compares two KPZ parameter
regimes. The measured roughness exponents are close:

```text
KPZ-A alpha = 0.350
KPZ-B alpha = 0.337
KS alpha    = 0.895
```

The KPZ-A/KPZ-B distance is small and nearly flat under scale, while KS remains
far away. This supports the claim that exponent-like features are closer to
universal observables, even when richer scale-free structure-function features
retain non-universal amplitudes.

## 6. Discussion

### 6.1 What failed?

The original discovery hope was:

```text
universality class = density cluster in a good finite-size feature space
```

The experiments do not support this for the tested surface-growth systems.
Instead, the observed structure is hierarchical and multimodal:

1. RD and KS are easy to separate.
2. EW and continuum KPZ are locally separable with temporal features but not as
   clean density clusters.
3. BD carries a discrete-model signature that can remain disconnected from
   continuum KPZ/Eden in feature space.
4. Coarse-graining reduces some spatial discreteness but can damage temporal
   signals.

This explains why kNN can be very high while HDBSCAN stays near ARI 0.5. A
class can be locally recognizable but globally split into multiple components.

### 6.2 Why this is still scientifically useful

A negative result is valuable here because it identifies a common interpretive
trap in ML-for-physics work. Good classification performance does not imply a
learned universal invariant. A model may learn:

- numerical implementation;
- local morphology;
- amplitudes such as `D/nu`;
- initial-condition sector;
- transient dynamics;
- finite-size corrections;
- trivial separation of easy classes.

The project's value lies in making those possibilities explicit and testing
them. The phrase "feature geometry" is therefore more accurate than
"universality distance." The former is an empirical object; the latter suggests
an invariant that has not been established.

### 6.3 A revised interpretation of the earlier anomaly-detection work

The earlier anomaly-detection project should be reframed as a finite-size
diagnostic for simulated growth dynamics, not as evidence for a universal
distance. In hindsight, the BD result is especially instructive. Detecting BD as
an anomaly despite its KPZ association is not proof that gradients encode
universality. It is evidence that gradients encode finite-size morphology and
implementation-specific growth signatures.

The revised claim is:

> Anomaly and clustering methods can reveal useful finite-size morphology and
> dynamical signatures, but same-class controls are required before those
> signatures can be interpreted as universality structure.

### 6.4 Diagnostic protocol for future work

A defensible ML-assisted universality study should pass the following gates:

1. Same-class numerical variants remain close.
2. Different implementations of the same asymptotic class converge under
   appropriate coarse-graining.
3. Different classes remain separated under the same diagnostics.
4. The observable is checked for non-universal amplitudes.
5. Density-cluster claims are separated from supervised or kNN discriminability.
6. Results are repeated across seeds, system sizes, and time windows.
7. Obsolete or failed claims remain documented but are not used as headline
   evidence.

## 7. Limitations

The surface-growth experiments are simulated, one-dimensional, and finite in
both time and size. The cluster metrics are based on selected feature families,
not an exhaustive search over all possible observables. HDBSCAN and KMeans are
useful probes of geometry but not definitive tests of all unsupervised
structure. Some results are from single full-run snapshots and need repeated
uncertainty analysis. The KS simulations are treated operationally as a
separate class in this workspace, but the asymptotic relationship between KS
and KPZ can be subtle and parameter-dependent. No experimental data are tested
in the current draft.

These limitations are not incidental; they are part of the paper's message.
They are exactly the conditions under which overinterpreting ML classifications
as universality discovery becomes risky.

## 8. Conclusion

This project began from an optimistic hypothesis: unsupervised learning might
discover universality classes from simulated physics data. The current evidence
supports a more careful conclusion. Finite-size spatial and temporal features
can be highly informative, and in favourable systems PCA-style representations
can recover real critical structure. But in surface-growth simulations, feature
geometry does not cleanly equal universality geometry.

The strongest result is therefore a diagnostic failure analysis. The project
shows why high local classification accuracy, impressive anomaly scores, and
visually separated feature clouds are not enough. Universality claims require
same-class controls, scale-aware observables, and explicit separation between
finite-size morphology and asymptotic physics.

This is a useful result: it turns an overstrong discovery claim into a
methodological contribution for computational physics.

## References

[Kardar1986] Kardar, M., Parisi, G., and Zhang, Y.-C. "Dynamic Scaling of
Growing Interfaces." *Physical Review Letters* 56, 889-892 (1986).
https://doi.org/10.1103/PhysRevLett.56.889

[Edwards1982] Edwards, S. F. and Wilkinson, D. R. "The Surface Statistics of a
Granular Aggregate." *Proceedings of the Royal Society A* 381, 17-31 (1982).
https://doi.org/10.1098/rspa.1982.0056

[FamilyVicsek1985] Family, F. and Vicsek, T. "Scaling of the active zone in the
Eden process on percolation networks and the ballistic deposition model."
*Journal of Physics A* 18, L75-L81 (1985).
https://doi.org/10.1088/0305-4470/18/2/005

[BarabasiStanley1995] Barabasi, A.-L. and Stanley, H. E. *Fractal Concepts in
Surface Growth.* Cambridge University Press (1995).

[HalpinHealyZhang1995] Halpin-Healy, T. and Zhang, Y.-C. "Kinetic roughening
phenomena, stochastic growth, directed polymers and all that." *Physics
Reports* 254, 215-414 (1995).
https://doi.org/10.1016/0370-1573(94)00087-J

[Medina1989] Medina, E., Hwa, T., Kardar, M., and Zhang, Y.-C. "Burgers
equation with correlated noise: Renormalization-group analysis and applications
to directed polymers and interface growth." *Physical Review A* 39, 3053-3075
(1989). https://doi.org/10.1103/PhysRevA.39.3053

[PrahoferSpohn2000] Prahofer, M. and Spohn, H. "Universal Distributions for
Growth Processes in 1+1 Dimensions and Random Matrices." *Physical Review
Letters* 84, 4882-4885 (2000). https://doi.org/10.1103/PhysRevLett.84.4882

[Corwin2012] Corwin, I. "The Kardar-Parisi-Zhang equation and universality
class." *Random Matrices: Theory and Applications* 1, 1130001 (2012).
https://arxiv.org/abs/1106.1596

[Sasamoto2016] Sasamoto, T. "The 1D Kardar-Parisi-Zhang equation: Height
distribution and universality." *Progress of Theoretical and Experimental
Physics* 2016, 022A01 (2016). https://doi.org/10.1093/ptep/ptw002

[Takeuchi2018] Takeuchi, K. A. "An appetizer to modern developments on the
Kardar-Parisi-Zhang universality class." *Physica A* 504, 77-105 (2018).
https://arxiv.org/abs/1708.06060

[Matetski2021] Matetski, K., Quastel, J., and Remenik, D. "The KPZ fixed
point." *Acta Mathematica* 227, 115-203 (2021).
https://doi.org/10.4310/acta.2021.v227.n1.a3

[Wegner1972] Wegner, F. J. "Corrections to Scaling Laws." *Physical Review B*
5, 4529-4536 (1972). https://doi.org/10.1103/PhysRevB.5.4529

[PrivmanFisher1984] Privman, V. and Fisher, M. E. "Universal Critical
Amplitudes in Finite-Size Scaling." *Physical Review B* 30, 322-327 (1984).
https://doi.org/10.1103/PhysRevB.30.322

[Onsager1944] Onsager, L. "Crystal Statistics. I. A Two-Dimensional Model with
an Order-Disorder Transition." *Physical Review* 65, 117-149 (1944).
https://doi.org/10.1103/PhysRev.65.117

[Binder1981] Binder, K. "Finite size scaling analysis of Ising model block
distribution functions." *Zeitschrift fuer Physik B Condensed Matter* 43,
119-140 (1981). https://doi.org/10.1007/BF01293604

[Wang2016] Wang, L. "Discovering phase transitions with unsupervised learning."
*Physical Review B* 94, 195105 (2016).
https://doi.org/10.1103/PhysRevB.94.195105

[CarrasquillaMelko2017] Carrasquilla, J. and Melko, R. G. "Machine learning
phases of matter." *Nature Physics* 13, 431-434 (2017).
https://doi.org/10.1038/nphys4035

[Hu2017] Hu, W., Singh, R. R. P., and Scalettar, R. T. "Discovering phases,
phase transitions, and crossovers through unsupervised machine learning: A
critical examination." *Physical Review E* 95, 062122 (2017).
https://doi.org/10.1103/PhysRevE.95.062122

[Yue2022] Yue, Z., Wang, Y., and Lyu, P. "Incremental learning of phase
transition in Ising model: Preprocessing, finite-size scaling and critical
exponents." *Physica A* 600, 127538 (2022).
https://doi.org/10.1016/j.physa.2022.127538

[Tirelli2021] Tirelli, A., Costa, D. O., and Carlon, E. "Unsupervised machine
learning approaches to the q-state Potts model." arXiv:2112.06735 (2021).
https://arxiv.org/abs/2112.06735

## Appendix A: Repository Result Sources

- Exp 62 results: `results_exp62/results.json`
- Exp 63 results: `results_exp63/results.json`
- Exp 64 results: `results_exp64/results.json`
- Ising PCA-FSS control: `results_exp52d_full/results.json`
- Potts Binder control: `results_exp57c_pilot/summary.json`
- KPZ exponent-only control: `results_exp50r/metadata.json`

## Appendix B: Current Non-Claims

This draft does **not** claim:

1. discovery of a universal ML distance;
2. proof that HDBSCAN failure is universal across all feature maps;
3. experimental validation on real surface-growth data;
4. discovery of a new universality class;
5. replacement of exponent fitting or RG theory;
6. that old "universality axis" figures remain valid as headline evidence.
