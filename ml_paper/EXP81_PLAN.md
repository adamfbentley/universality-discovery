# Exp 81 — The Observable-Information Hierarchy

**Status: PLANNED, 2026-07-02. No results yet.**
**Companion theory: `THEORY_minimax_floor.md`, Appendix F.**
**Execution prompt for the implementing agent: `SONNET_EXP81_PROMPT.md`.**

---

## Why this experiment (one paragraph)

The exp77 floor is estimator-agnostic but **observable-conditional**: it bounds
every estimator that consumes the log-W_sat ladder. It says nothing about
estimators that consume richer data — multi-observable summaries, full
spectra, or raw configurations. This scope condition is not a weakness; it is
the next research object. Exp81 asks: **how much exponent information is
gained at each level of the observable hierarchy, and under what correction
structure do richer observables actually help?** The answer determines whether
the floor has practical bite (width ladders are near-sufficient) or whether
the right response to the floor is richer measurement (the "better
observables, not better models" conclusion). Either outcome is a paper-grade
result, and it is the scope test the theorem needs before any "nature of
scientific ML" claim survives review.

## The hierarchy

    Level 0: single-summary ladder  (log W_sat)         — exp77 floor known
    Level 1: multi-summary ladders  (K channels)        — Part A (theory + pilot)
    Level 2: full spectrum / two-point statistics       — Part B (exact, Gaussian testbed)
    Level 3: raw configurations                          — Part B (empirical, Gaussian testbed)
    Level 4: non-Gaussian raw configurations (BD, KPZ)   — exp82+, out of scope here

Each level has its own floor. The scientific quantity is the **ratio of
floors between levels** as a function of the declared correction class.

---

## Part B first (it is cheap, exact, and decisive): the fractional-EW testbed

### Why a Gaussian testbed

We need a generative family with (i) **tunable true α**, (ii) **controllable
corrections to scaling**, (iii) **exact sampling of raw fields**, and (iv)
**closed-form information at every level of the hierarchy**. Stationary
fractional Edwards-Wilkinson delivers all four: the field is Gaussian, so the
power spectrum is a sufficient statistic, the raw-field KL between any two
parameter settings is a one-line mode sum, and the exact Fisher information
of α from raw fields is analytic. This gives ground truth at BOTH ends: the
Level-0 floor (exp77 machinery) and the Level-2/3 information (exact). The
experiment then measures where trained estimators sit between them.

### The family

1D stationary Gaussian field on L sites, Fourier modes k_n = 2πn/L,
n = 1..L/2, each mode independent complex Gaussian with

    S(k) = D / ( ν |k|^z  +  ν₂ |k|^{z + ω̃} )

- Leading roughness: **α = (z − 1)/2** (verify numerically; W² ∝ (D/ν) L^{2α}).
- Correction to scaling: relative correction b·L^{−ω_eff} in W²(L), amplitude
  b ∝ ν₂/ν. **Caution (mode-sum analysis): ω_eff = ω̃ only for ω̃ < 2α, where
  the correction integral is IR-dominated; for ω̃ > 2α it is UV-dominated and
  ω_eff saturates at 2α (intrinsic-width-like).** The implementer must MEASURE
  ω_eff from clean ladders and declare the measured class, not the nominal ω̃.
  Tunable range at α ≈ 0.5 is therefore ω_eff ∈ (0, 1].
- Special stress case: ω̃ = 2 − z reproduces a near-marginal correction at
  α ≈ 0.5 (the hard regime the floor identifies); include as a labeled case,
  not the default.

Sampling is exact and instant (draw each mode, inverse FFT). No dynamics.

### Sampling prior (leakage control — read carefully)

α_true ~ U[0.35, 0.65] (z ∈ [1.7, 2.3]). Nuisances drawn **independently of
α**: log D, log ν uniform over a decade; correction amplitude u := b at L_min
bounded |u| ≤ U (match exp77 conventions, U ∈ {0.5, 1, 4} grid); ω̃ ∈ [0.3, 2.5].
Independence of the prior is the defense against the discriminative-≠-universal
failure mode documented in exp62–71: if amplitudes correlated with α in the
prior, a network could estimate α from amplitude cues and the comparison would
be void. Document the prior in the results JSON.

### The three information computations (no training required)

1. **Level-0 floor**: rerun the exp77 exact optimizer on the 7-point design
   {32,48,64,96,128,192,256} with σ measured from this family's W_sat seed
   scatter, declared class (N=1, U, ω_min=0.3). This is the baseline.
2. **Level-2/3 floor (raw/spectral)**: Le Cam with the adversary living in the
   family itself: minimize KL between ensembles (z, D, ν, ν₂, ω̃)₁ and
   (z + 2Δα, ...)₂ over nuisances, where for independent zero-mean Gaussian
   modes KL = Σ_k [S₁/S₂ − 1 + ln(S₂/S₁)] (per complex mode; **verify the
   convention numerically against sampled log-likelihoods before using**).
   Multi-L: sum the per-L KLs (independent runs per L, matching exp77).
3. **Exact Fisher information** of α from raw fields at fixed nuisances:
   I(α) = Σ_k (∂ ln S/∂α)², ∂ ln S/∂α = −2 ln|k| · ν|k|^z/(ν|k|^z + ν₂|k|^{z+ω̃}).
   This is the no-nuisance ceiling; the gap between it and (2) prices the
   nuisance degeneracy at the raw-field level.

**Primary output: floor_L0 / floor_L3 as a function of (U, ω_min, design).**
Hypothesis (to be tested, not assumed): the raw-field floor is finitely but
not astronomically better — corrections enter the spectrum at small k where
few modes live, so the adversary's mimicry is harder to sustain against the
full spectrum than against one number per L. If instead floor_L3 ≈ floor_L0,
the width ladder is near-sufficient and the exp77 floor has full practical
bite for this family.

### The two trained estimators (after the floors exist)

- **Summary estimator**: exp76-style MLP on the 7-point log-W_sat ladder,
  trained on the sampling prior. Compare RMSE to floor_L0.
- **Raw-field estimator**: small 1D CNN consuming raw h(x) at all 7 sizes
  (per-L encoder, pooled). Same prior, same loss. Compare to floor_L3.
- **Built-in sanity ceiling**: because the family is Gaussian, no estimator
  can beat the exact-FIM/Le Cam numbers from (2)–(3). A CNN "beating" them
  indicates leakage or a bug, and the run is invalid. This is the
  self-auditing property that motivated the Gaussian choice.

### Part B acceptance gates

- G-B1: sampled spectra match S(k) within Monte Carlo error (plot + KS check).
- G-B2: measured α from W_sat ladders at ν₂=0 recovers (z−1)/2 across the z
  grid (slope regression, no corrections).
- G-B3: the KL formula validated against numerical log-likelihood differences
  on sampled fields (agreement to ~1%).
- G-B4: Level-0 optimizer reproduces exp77-style floors when fed this family's
  σ (order-of-magnitude consistency with the σ scaling in floor.json).
- G-B5: CNN and MLP both recover α to near-FIM precision at ν₂ = 0 (easy
  regime) before any hard-regime numbers are reported.

---

## Part A: multivariate floors (Level 1)

### Definition

Estimator observes K channels y_{i,s} ∈ R^K per size L_i:

    y_k(L_i) = c_k + θ_k(α)·x_i + Σ_j u_{k,j} e^{−ω_j x_i} + noise,
    noise ~ N(0, Σ) across channels (measured), independent across L and seeds.

θ_k(α): known per-channel exponent maps (width: θ = α; spectral band slope:
θ = −(1+2α); channels with θ_k = 0 are pure nuisance probes and still help iff
the correction spectrum {ω_j} is **shared** across channels).

    D²(Δα) = min_{c, {ω_j} shared, |u_{k,j}| ≤ U_k}  Σ_i Δμ(L_i)ᵀ Σ⁻¹ Δμ(L_i)

Floor as in exp77: Δα* = max{Δα : D² ≤ 1/m} (Σ absorbs σ²).

### Hypotheses

- **H-A1 (independent corrections per channel)**: extra channels give at most
  quadrature-level gains (~√K); the adversary defeats each channel separately.
- **H-A2 (shared correction spectrum)**: channels over-determine the shared
  {ω_j}; the adversary cannot mimic the tilt in all channels at once with
  consistent exponents; gain is superlinear in K ("triangulation"). This is
  the physically motivated case — the same irrelevant operators feed every
  observable.
- **H-A3 (correlated noise)**: strong cross-channel seed correlation (same
  run → same fluctuation) can cancel most of the nominal gain; only measured
  Σ decides. Do not report diagonal-Σ floors as if they were real.

The H-A1-vs-H-A2 contrast is the theory result of Part A: **whether richer
summaries help is decided by whether corrections are shared, not by K.**

### Data

Pilot only: adapt `76b_regenerate_ladders.py` to record, per (system, L, seed):
log W_sat, log gradient variance, height skewness and kurtosis at saturation,
two spectral band log-powers. Systems: EW + KPZ (skip BD in pilot). m = 8
seeds, design {32..256}. Measure Σ per (system, L). If wall-clock is
prohibitive, validate the multivariate optimizer fully on synthetic channels
first (H-A1/H-A2 curves) and defer measured-Σ to a second pass — the synthetic
contrast is the primary deliverable.

### Part A acceptance gates

- G-A1: K=1 reduction reproduces the exp77 exact floor for matching (σ, U,
  ω_min, design) to within optimizer tolerance (BD m=24 floor 0.27 is the
  anchor; state the tolerance achieved).
- G-A2: H-A1 vs H-A2 floor-vs-K curves produced under identical U, with the
  shared-spectrum gain quantified.
- G-A3: any measured-Σ floor reports its data provenance (file, seeds, date).

---

## Part C (stretch, only if A and B complete): the reflexive corollary

Apply the exp77 machinery, unchanged, to a neural-scaling-law design: target
exponent of L(N) = a·N^{−α_s}·(1 + b·N^{−ω}) on a 2-decade compute window
(7 log-spaced N), noise from published-scale scatter (take σ_logL ∈
{0.01, 0.03, 0.1} as a grid — no external data needed). Report the resolvable
Δα_s and the seed(=training-run) requirement. One table, one paragraph. This
is a floor for the ML community's own exponent fits; frame it exactly as
class-conditional, like every other floor number.

---

## Honesty rules (binding for all parts)

1. Every floor number states its five-tuple: (observable level, correction
   class N/U/ω_min, design, noise source, m).
2. No sharpness claims anywhere — the analytic lower bound on E_N is still
   open (see THEORY note).
3. Nothing enters `CLAIMS_REGISTER.md` in this phase; results go to
   `EXP81_REPORT.md` and `results_exp81_*/` for review first.
4. A trained estimator apparently beating an exact information bound is
   reported as a bug, not a result.
5. Negative/null outcomes (e.g. floor_L3 ≈ floor_L0) are primary results and
   are written up with the same care as positive ones.

## Non-claims (scope)

- Nothing here concerns non-Gaussian raw fields. BD/KPZ raw-configuration
  estimators are exp82+, and only make sense after Part B fixes the
  methodology on a family where the answer is checkable.
- Part B floors are for the declared parametric family; they do not bound
  estimators on data from outside it.

## Deliverables

    experiments/81_fractional_ew_testbed.py     Part B: family, sampling, floors, FIM
    experiments/81b_estimators.py               Part B: MLP + CNN training/eval
    experiments/81c_multivariate_floor.py       Part A: vector Le Cam optimizer
    experiments/81d_multiobservable_ladders.py  Part A: pilot data regeneration (optional pass 2)
    results_exp81_hierarchy/                    all JSONs + figures
    ml_paper/EXP81_REPORT.md                    findings, gates passed/failed, numbers

## Relation to thesis framing

Part B answers the referee question "does your theorem apply to learned
representations?" with mathematics instead of rhetoric: the floor binds
per-observable, the hierarchy quantifies what richer observables buy, and the
model-limited / optimization-limited / information-limited trichotomy becomes
measurable. Part C makes the point legible to an AI audience. This is the
scope test on which the "information limits of scientific ML" framing stands
or falls.
