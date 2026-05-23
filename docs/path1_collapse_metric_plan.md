# Path 1: Collapse Geometry as a Universality Metric

**Status:** in progress (started 2026-05-21)
**Experiment:** `experiments/68_collapse_metric.py` → `results_exp68_collapse/`

## Motivation

The feature-geometry paper showed that Euclidean density clustering of finite-size
features does **not** recover universality classes (ARI ceiling ~0.5), because the
feature space is dominated by non-universal nuisance directions (amplitudes such
as `D/nu`, discrete-model morphology, initial-condition sector). The diagnosis is
correct but the operationalization of "universality" was wrong.

**Reframe:** universality is not a proximity property in feature space; it is an
*equivalence under the RG / Family-Vicsek rescaling group*. Two systems are in the
same class iff a single rescaling collapses their large-scale scaling functions
onto a common master curve, after quotienting non-universal metric factors.

The right ML object is therefore not "distance between feature vectors" but
"residual of the best data collapse." This experiment builds that metric and tests
whether it recovers the class partition that feature geometry could not.

## The object to collapse: the Family-Vicsek width curve

For a 1D interface, the width `w(L,t) = <[h - <h>]^2>^{1/2}` obeys

    w(L,t) = L^alpha * F(t / L^z),     w ~ t^beta (growth),  beta = alpha/z.

This is the right object because it encodes the exponents that *separate* EW from
KPZ — beta (EW 1/4, KPZ 1/3) and z (EW 2, KPZ 3/2) — which the static gradient
features cannot see (alpha = 1/2 for both). Collapse over (alpha, z) extracts the
exponents and the universal scaling function F.

## Method

1. **Multi-L width curves.** For each system simulate at several L, average w(t)
   over seeds (smooths the curve). T(L) scaled ~ L^2 so even the slow EW (z=2)
   reaches saturation. Reuse the Exp 63 trajectory simulators via importlib.

2. **2D Family-Vicsek collapse.** Generalize the Exp 52d window-variance collapse
   quality to two parameters: for given (alpha, z), set u = t/L^z, y = w/L^alpha,
   pool over L, and measure window variance of log(y) in overlapping-u windows that
   contain >= 2 distinct L (so the metric reflects genuine cross-L overlap).
   Minimize over (alpha, z) by grid + Nelder-Mead. Output (alpha, z, beta) and the
   collapse master curve.

3. **Validation.** EW -> (alpha~0.5, z~2, beta~0.25); KPZ/BD/Eden -> (0.5, 1.5,
   0.33); RD -> beta~0.5, no saturation (flagged); KS -> its own. Confirms the
   collapse machinery is sound before trusting the metric.

4. **Per-sample exponent cloud + ARI.** Bootstrap seed-subsets per system -> a
   cloud of (alpha, beta, z) vectors. Cluster (HDBSCAN/KMeans) and report ARI vs
   the intended classes {EW | KPZ=KPZ+BD+Eden | KS | trivial=RD}. Headline test:
   does collapse-exponent geometry beat the old ~0.5 feature-geometry ceiling?

5. **System-level universality metric.** Pairwise distance combining exponent
   distance and normalized master-curve shape distance. Show EW far from the KPZ
   group, KPZ/BD/Eden mutually close, KS and RD separate. This is the "universality
   distance" the original project hoped for, done correctly (amplitude-quotiented).

## Success criteria (honest, pre-registered)

- **Validation:** extracted EW/KPZ exponents within ~15-20% of theory at accessible
  L (finite-size bias expected; report it, do not hide it).
- **Headline:** exponent-cloud ARI clearly exceeds the ~0.5 feature ceiling
  (target > 0.7) AND the KPZ subclass (KPZ+BD+Eden) is no longer split.
- **Metric:** EW-vs-KPZ distance > within-KPZ distances; KS, RD separate.

## Calibration findings (2026-05-22) — scope of v1

A calibration pass on the Exp 63 simulators established what is and is not
measurable at accessible sizes:

- **Width must be the seed-averaged variance** `W(t) = sqrt(<var_x h>_seeds)`;
  single-trajectory width curves are too noisy (non-monotonic) to collapse.
- **beta is robust with W-gating**: fit log W vs log t over the band
  `0.2 Wsat < W < 0.5 Wsat` (excludes the IC transient and saturation). EW
  recovers beta = 0.252 (theory 0.25).
- **KPZ beta is an effective, lambda-dependent crossover exponent** at accessible
  L: lambda=2 -> 0.25 (EW-like), lambda=4 -> 0.28, lambda=8 -> 0.40 (overshoot +
  instability). This is an honest instance of the main paper's thesis: even the
  canonical exponent observable is finite-time-contaminated. v1 uses lambda=4 and
  reports beta as effective, not asymptotic.
- **Discrete BD/Eden are strongly correction-limited at L <= 128.** The archived
  calibration (`results_exp68_calibration/summary.json`) gives BD_L96
  `t_sat_90pct` = 21.625 monolayers and Eden_L96 `t_sat_90pct` = 1752.75
  monolayers, with effective beta near 0.20 in both cases. The same calibration
  selected `intrinsic_width_Wi2 = 0.0` for BD and Eden, so the archived
  intrinsic-width subtraction is inactive rather than a demonstrated correction.

**v1 scope (this iteration):** continuum systems EW, KPZ, KS, RD, where time
resolution is clean. The headline test is whether collapse-exponent geometry
separates EW from KPZ -- the exact degeneracy gradient features cannot break --
and keeps KS, RD distinct, beating the ~0.5 feature ceiling.

**v2 (next step):** sub-monolayer-resolved BD/Eden width curves, with an
intrinsic-width scan reported explicitly, to test the discrete-continuum KPZ
merging.

## v1 results (2026-05-22, `--full`: L = 48..128, n_seed = 12, n_boot = 40)

Source: `results_exp68_collapse/summary.json`.

Per-system exponents (seed-mean curves), theory in brackets:

| system | alpha | beta | z | z_collapse | theory (a/b/z) |
|---|---|---|---|---|---|
| EW  | 0.533 | 0.267 | 1.998 | 1.86 | 0.5 / 0.25 / 2.0 |
| KPZ | 0.469 | 0.304 | 1.545 | 1.05 | 0.5 / 0.33 / 1.5 |
| KS  | (sep.) | -- | -- | -- | distinct class |
| RD  | 0.006 | 0.498 | 0.013 | -- | 0.0 / 0.5 / -- |

- **In this continuum-only run, z separates EW/KPZ cleanly**: EW z = 1.998,
  KPZ z = 1.545. This is the degeneracy gradient features cannot break
  (alpha = 1/2 for both).
- RD recovers (alpha~0, beta~0.5) essentially exactly.
- KPZ beta = 0.304 is an effective crossover value (asymptotic 1/3); honest.

**Headline clustering (exponent cloud, 4 classes):**
- KMeans ARI = **0.773**, HDBSCAN core ARI = **0.706**
- vs the old feature-geometry ceiling ARI ~ 0.50.
- The pilot-to-full comparison suggested improvement with system size, but this
  is only an early two-point observation and should not be treated as a finite-
  size trend.

**Current reading:** the continuum-only v1 result is a useful positive control
for exponent extraction, not a general universality-recovery method. Later
matched multi-seed controls show that the stronger all-system positive claim is
not stable.

**Caveat (honest):** the ARI comparison is indicative, not a perfectly controlled
head-to-head -- the old ceiling is from the 6-system set (incl. discrete BD/Eden)
with per-trajectory features, while this is the 4 continuum systems with
bootstrap-of-collapse samples. A matched head-to-head on identical samples, and
the discrete-model v2, are the next steps.

## v2 results (2026-05-22): discrete models + matched head-to-head

Experiment: `experiments/69_collapse_metric_full.py` -> `results_exp69_collapse_full/`.
Adds BD/Eden via sub-monolayer-resolved variance curves plus an intrinsic-width
scan; continuum systems keep the clean Exp 68 W-gated method. `--full`
ladder L = 48..128, continuum 12 seeds, discrete 20 seeds, 40 bootstraps.

**Historical exp69 head-to-head (identical 6 systems, 4-class labels
{EW | KPZ=KPZ+BD+Eden | KS | trivial}):**

| representation | KMeans ARI | HDBSCAN core ARI |
|---|---|---|
| feature geometry (old 10D Exp 63) | 0.490 | 0.495 |
| collapse geometry (alpha, beta, z) | **0.902** | **0.960** |

This single protocol reproduces the old ~0.5 feature ceiling and gives a much
higher exponent-geometry ARI. It is now interpreted as a high-performing
protocol instance, not as a stable method-level doubling of ARI.

Per-system effective exponents (a, b, z), theory in brackets:

| sys | alpha | beta | z | theory |
|---|---|---|---|---|
| EW   | 0.64 | 0.24 | 2.64 | 0.5/0.25/2.0 |
| KPZ  | 0.33 | 0.31 | 1.08 | 0.5/0.33/1.5 |
| BD   | 0.35 | 0.41 | 0.85 | 0.5/0.33/1.5 |
| Eden | 0.25 | 0.24 | 1.00 | 0.5/0.33/1.5 |
| RD   | -0.03 | 0.49 | 0.00 | 0.0/0.5/-- |

System distances (standardized exponents): KPZ-BD = 0.28, KPZ-Eden = 0.38,
BD-Eden = 0.50 (tight KPZ group) vs EW-KPZ = 2.25, RD-KPZ = 1.98, KS-KPZ = 3.3.

**Naming note:** "collapse geometry" here means clustering on the *exponent
vectors* (alpha, beta, z) extracted via the collapse/W-gating machinery. It is
NOT a clustering on a collapse-residual / master-curve distance -- that true
collapse-residual metric is not yet implemented and is Tranche-2 work. The 2D
Family-Vicsek collapse is used only as a cross-check on z (and gives z_col that
differs from alpha/beta; both are reported).

**Findings (calibrated against an external referee, 2026-05-22):**
1. On the identical 6-system set, exponent geometry gives much higher ARI
   (KMeans 0.90, HDBSCAN core 0.96) than the single-L feature pipeline (~0.49).
   But this is NOT an information-matched comparison (see caveats), so the correct
   reading is: *multi-L effective-exponent summaries are far more aligned with the
   class labels than single-L feature vectors* -- not that a new unsupervised
   universality metric has been established.
2. In exponent space BD/Eden land far closer to continuum KPZ (distances 0.28-0.38)
   than to EW/RD/KS (1.6-3.3), so the KPZ class is *largely* recovered. It is NOT
   cleanly merged: HDBSCAN over-segments (found 5 clusters), so "full 4-class
   recovery" overstates it.
3. The exponents are EFFECTIVE, not asymptotic (EW z~2.6 vs 2.0; KPZ z~1.1,
   alpha~0.33 vs 1.5/0.5; BD/Eden alpha~0.25-0.35). Clustering succeeds because
   same-class simulators have internally consistent *effective* exponents, which is
   useful but is weaker than "universality recovery" in the strict (asymptotic)
   sense. KPZ beta is also lambda-dependent (a tuned crossover), not universal.

**Honest caveats:** (a) the head-to-head is information-asymmetric -- collapse uses
multi-L curves (L=48..128), seed-averaged variance, and bootstrap-of-collapse
pseudo-samples, while the feature side is single-L (L=128), 25 trajectories,
separately standardized; a *matched multi-L feature baseline* is required and is
Tranche-2 (exp70). (b) EW z = 2.64 / KPZ z = 1.08 are off asymptotic but correctly
ordered. (c) **KS beta is not meaningfully fit (~5.6); KS separation is partly an
artifact of failed exponent extraction, not physical geometry** -- it must be
flagged or analyzed as a crossover. (d) Headline ARIs are single-run with no
confidence intervals; bootstrap-of-collapse is not a substitute for independent
seeds. (e) The pilot->full size-trend (~0.51 -> ~0.90) was observed in console
runs but the pilot artifact was overwritten; it needs archived pilot/full outputs
before being claimed.

### 2026-05-22 corrected + archived (post-referee)

Provenance fixes: calibration archived to `results_exp68_calibration/summary.json`;
the pilot/full comparison is now backed by archived artifacts -- exponent all-6
KMeans ARI **0.507 (pilot, `results_exp69_collapse_full_pilot/`) -> 0.902
(full, `results_exp69_collapse_full/`)**. This is a two-point comparison, not a
finite-size trend. exp69 now also stores HDBSCAN noise counts,
confusion matrices, system-identity ARI, and a KS-excluded variant.

Two clarifications the confusion matrices force:
- **Merging holds under KMeans, not HDBSCAN.** The KMeans (K=4) confusion matrix
  (full) puts the KPZ class (kpz+bd+eden, 120 samples) 112/120 in a single cluster,
  with EW, KS, RD each in their own pure cluster. This is a near-pure KMeans
  four-class partition, not a perfect recovery. HDBSCAN over-segments (5 clusters,
  14 noise points). So "BD/Eden merge with KPZ" is correct under KMeans but
  overstated for HDBSCAN.
- **The win is not solely a KS artifact under KMeans.** With KS removed, exponent
  geometry still beats the single-L feature pipeline by a wide margin: KMeans
  **0.843 vs 0.343** (KS-excluded, full). This does not by itself prove recovery
  on the hard EW/KPZ core, because RD remains an easy class and HDBSCAN's
  KS-excluded behavior is much less clean.

The central caveat that remained was the information asymmetry: exponent-
vectors-from-multi-L vs features-from-single-L. exp70 narrows this caveat but
does not settle it.

### exp70: raw multi-L feature baseline (narrows the asymmetry)

Experiment: `experiments/70_matched_baseline.py` -> `results_exp70_matched/`.
Same systems, fixed params, seed budget, and bootstrap construction as the
feature baselines. The current script also recomputes a matched exponent cloud
from the same seed subsets when rerun and adds explicit cross-L engineered and
PCA-whitened feature baselines. The archived `results_exp70_matched/` artifact
predates those code changes and still loads the exponent reference from exp69.
Archived pre-Codex representations:

| representation | dims | KMeans ARI | HDBSCAN core | KS-excluded KMeans |
|---|---|---|---|---|
| single-L features (Lmax only) | 10 | 0.605 | 0.566 | 0.500 |
| multi-L features (concat over L=48..128) | 40 | 0.605 | 0.566 | 0.500 |
| exponent geometry (alpha,beta,z; exp69 reference in archived artifact) | 3 | 0.902 | 0.960 | 0.843 |

**Codex reruns (tagged, archived without overwriting):**

| artifact | sampling | single-L | raw multi-L | cross-L engineered | PCA-whitened | matched exponent |
|---|---|---:|---:|---:|---:|---:|
| `results_exp70_matched_codex_full_20260522/` | seed-start 70000, 16 seeds/system | 0.605 | 0.605 | 0.496 | 0.496 | 0.584 |
| `results_exp70_matched_codex_full_seed69000_20260522/` | seed-start 69000, 16 seeds/system | 0.244 | 0.605 | 0.496 | 0.244 | 0.820 |
| `results_exp70_matched_codex_full_exp69sampling_20260522/` | seed-start 69000, exp69 seed counts (12 continuum, 20 discrete) | 0.244 | 0.605 | 0.496 | 0.244 | 0.902 |

The exact exp69-sampling rerun at seed-start 69000 reproduces the exp69 exponent
reference (KMeans 0.902, HDBSCAN core 0.960, KS-excluded KMeans 0.843), so the
original exp69 artifact is internally reproducible. But this point is not
representative of nearby seed-starts.

**Protocol sweep (Exp 71, `results_exp71_protocol_sweep/summary.json`; n=5
seed-starts 69000, 70000, 71000, 72000, 73000):**

| protocol | best feature KMeans | matched exponent KMeans | exponent - best feature | KS-excluded exponent KMeans |
|---|---:|---:|---:|---:|
| equal seeds (16/system) | 0.605 [0.605, 0.605] | 0.604 [0.440, 0.769] | -0.000 [-0.165, 0.164] | 0.576 [0.361, 0.792] |
| exp69 seed counts (12 continuum, 20 discrete) | 0.605 [0.605, 0.605] | 0.600 [0.374, 0.827] | -0.005 [-0.231, 0.222] | 0.565 [0.319, 0.810] |

Intervals are 95% t-intervals over seed-starts. The sweep preserves tagged
artifacts for every run (`results_exp70_matched_codex_sweep_full_*`) and writes
the flat run table to `results_exp71_protocol_sweep/runs.csv`.

**Result:** raw multi-L concatenation is the best feature baseline among the
tested feature-side controls and is essentially tied with matched exponent
geometry over this five-seed sweep. The simple cross-L engineered coordinates and
PCA-whitened multi-L block do not beat raw multi-L. The original exp69 headline
point is a high-performing seed/protocol instance, not a stable estimate of the
method's advantage.

**Conservative conclusion:** the strong "representational not informational"
claim should be withdrawn. The defensible statement is now negative/modest:
*effective exponent geometry can outperform feature baselines under some
sampling protocols, but across a five-seed matched protocol sweep its KMeans ARI
is statistically indistinguishable from the raw multi-L feature baseline.* This
is more consistent with a finite-size/protocol-sensitive representation than with
a robust universality-recovery method.

Remaining Tranche-2 items (still open): a true collapse-residual/master-curve
metric (vs exponent vectors); larger-L corrections-to-scaling fits;
leave-parameter-out for KPZ; learned cross-L feature baselines; and a larger
protocol sweep (more seed-starts and bootstrap constructions) only if this line
is still worth pursuing after the five-seed result.

### 2026-05-24 update: ML-track matrix-refit controls

The exp70 code now archives matched feature and exponent matrices in tagged
matrix runs, allowing true subset refits rather than post-hoc restriction of
stored all-system labels. The derived MLP-09 audit
(`ml_paper/results/mlp09_summary.json`,
`ml_paper/tables/exp70_matrix_subset_refit_summary.md`) sharpens the conclusion:

| subset | best feature KMeans | matched exponent KMeans | interpretation |
|---|---:|---:|---|
| all six, equal sampling | 0.605 | 0.604 | no stable all-system exponent advantage |
| all six, exp69 sampling | 0.605 | 0.600 | no stable all-system exponent advantage |
| EW/KPZ only, equal sampling | 1.000 | 0.662 | binary EW/KPZ separation is easy for features |
| EW/KPZ only, exp69 sampling | 1.000 | 0.517 | binary EW/KPZ separation is easy for features |
| EW/KPZ/BD/Eden, equal sampling | 0.186 | 0.503 | hard KPZ quotient remains only partially recovered |
| EW/KPZ/BD/Eden, exp69 sampling | 0.169 | 0.438 | hard KPZ quotient remains only partially recovered |

This means the old "collapse/exponent geometry doubles the ARI" storyline is no
longer the defensible headline. The stronger and more honest result is a
failure-mode result: finite-size feature geometry can separate local physical
regimes such as EW versus continuum KPZ while failing to factor through the
coarser universality quotient that should merge continuum KPZ, BD, and Eden.
Effective exponents contain more quotient-relevant information on the hard
subset, but at L <= 128 they are not robust enough to support a positive
universality-recovery claim.

Path 1 should therefore be treated as supporting evidence for the ML-focused
quotient-learning paper, not as a standalone positive collapse-metric method
unless a true master-curve residual distance and larger finite-size-scaling
study are added.

## Known risks / caveats

- Finite-size + finite-T bias the exponents; BD/Eden corrections are large and may
  keep them off the continuum-KPZ value even after collapse. If so, that is itself
  the (honest) result: collapse helps but discrete corrections persist at
  accessible L — consistent with the finite-size message of the main paper.
- EW saturation is slow (z=2); T(L) ~ L^2 keeps it feasible only for modest L.
- RD does not saturate; handled as a special non-collapsing (trivial) class.
- This is real research: a clean positive result is not guaranteed. Either outcome
  is informative, and the *framework* (collapse-residual distance) is the
  contribution regardless of the exact numbers.
