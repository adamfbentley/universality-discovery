# Exp 83 Report — Audit-Response Computations

**Status: complete, 2026-07-03.**
**Plan: `archive/ai_execution/SONNET_EXP83_PROMPT.md`. Context: `archive/ai_execution/SIMULATED_AUDIT.md`,
`ml_paper/THEORY_minimax_floor.md` Appendix G.**
**Findings only — no claims language. Nothing here enters `CLAIMS_REGISTER.md`
or `MANUSCRIPT_OUTLINE.md`.**

All numbers below exist in `results_exp83_audit_response/audit_response.json`.
Code: `experiments/83_audit_response.py`.

---

## 1. Gate ledger

| Gate | What it checks | Result | Number | Proof path |
|---|---|---|---|---|
| G-83a | D²(Δα) monotonicity guard + Appendix G5 replication | **PASS** | 11-pt scan monotone; floor(monotone-grid path)=0.2703 vs anchor 0.27 (diff 0.0003) | `audit_response.json:task1_monotonicity` |
| G-83b | Convex-hull floor vs class floor: J=1 anchor + nondecreasing in J | **PASS** | J=1 matches class floor (≤2% at every U,m); hull/class floor nondecreasing in J at every (U,m) | `audit_response.json:task2_convex_hull` |
| G-83c | Van Trees bound: direction (≤ fixed-nuisance CR) + does not beat amortized RMSE | **PASS** | info ratio VT/fixed = 0.017 (≤1); VT bound 0.0634 < amortized RMSE 0.106 (does not beat) | `audit_response.json:task3_van_trees` |
| G-83d | Gaussianity diagnostic κ: computation runs, sane, CIs reported | **PASS** (diagnostic) | κ_median ≈ 0.92–0.93 across all 4 systems, 90% CI straddles 1 in every case | `audit_response.json:task4_kappa_diagnostic` |
| G-83e | Plug-in σ propagation: interval contains point floor + width matches prediction | **PASS** | all 4 systems: interval contains point; observed/predicted relative-error ratio ≈ 0.83 | `audit_response.json:task5_sigma_propagation` |

---

## 2. Task 1 (G-83a): D² monotonicity guard

Five-tuple: (Level-0, class N=1/U=4/ω_min=0.3, design={32,...,256}, σ=BD
measured 0.018977, m=24).

| Check | Value |
|---|---|
| 11-point scan (Δα∈[0.02,0.6], BD design/bounds), raw monotone | **True** |
| Floor via original bisection path | 0.2707 |
| Floor via new monotonized-grid path (60-pt fine grid, running max) | 0.2703 |
| Anchor (`results_exp77_minimax_floor/floor.json`, BD m=24) | 0.27 |
| Appendix G5's own independent-reimplementation anchor | 0.271 |

The fine 60-point grid *does* show 2 raw (pre-running-max) non-monotone dips,
both at Δα < 0.02 where D² itself is ~1e-7 to 1e-8 — the "astronomically
small confusion gap" regime the theory note already describes, where
near-perfect adversarial mimicry makes exact optimizer convergence hardest
with a fixed multi-start budget. This is not gated on (the 11-point scan,
matching Appendix G5's own check, is the gate criterion); it is exactly the
motivating case for applying the running-max transform in production,
which is now implemented (`floor_from_monotone_grid`).

## 3. Task 2 (G-83b): convex-hull floor vs class floor

Five-tuple: (Level-0, class N=1 [hull: J atoms]/U∈{1,4}/ω_min=0.3, design=
BD real design {32,...,256}, σ=BD measured 0.018977, m∈{6,24}).

| U | J | hull_floor (m=6) | hull_floor (m=24) | ratio to class_floor (m=6, m=24) |
|---|---|---|---|---|
| 1.0 | 1 (anchor) | 0.1535 | 0.1254 | 1.02, 0.98 |
| 1.0 | 2 | 0.2145 | 0.1441 | 1.42, 1.13 |
| 1.0 | 4 | 0.2449 | 0.1488 | 1.62, 1.17 |
| 1.0 | 8 | 0.2449 | 0.1488 | 1.62, 1.17 |
| 4.0 | 1 (anchor) | 0.3223 | 0.2707 | 1.00, 1.00 |
| 4.0 | 2 | 0.3926 | 0.3246 | 1.22, 1.20 |
| 4.0 | 4 | 0.4184 | 0.3410 | 1.30, 1.26 |
| 4.0 | 8 | 0.4184 | 0.3410 | 1.30, 1.26 |

(class_floor: U=1 → 0.1512 (m=6), 0.1277 (m=24); U=4 → 0.3223 (m=6), 0.2707
(m=24), all from `77_minimax_floor.py`'s own `floor()`, unchanged.)

**Finding (reported as instructed, either direction):** the hull gap
saturates by J=4 (J=8 adds nothing further at either U) and is **mixed in
size**, not uniformly small or large. At the loose bound U=4.0, the gap is
modest (ratio 1.26–1.30×), close to the ≈1.3× threshold the plan frames as
"Donoho affine near-sharpness transfers cheaply." At the tight bound U=1.0,
the gap is larger (ratio 1.17–1.62×, worse at the shorter m=6 design point).
So the "sharpness transfers cheaply" reading holds better at loose
correction bounds than tight ones for this design — a nuance, not a single
verdict either way.

## 4. Task 3 (G-83c): van Trees bound under the exp76 F1 prior

Five-tuple: (average-case/Bayes, class F1_power only [documented deviation
from the full exp76 5-family mixture — flagged, not hidden], design=real
design, σ=per-system measured, m∈{6,24}).

| System | σ (measured) | m | VT bound (RMSE scale) | fixed-nuisance CR bound | info ratio (VT/fixed) |
|---|---|---|---|---|---|
| bd | 0.0190 | 6 | 0.0049 | 0.00064 | 0.0170 |
| bd | 0.0190 | 24 | 0.0025 | 0.00032 | 0.0170 |
| eden | 0.1378 | 6 | 0.0356 | 0.0046 | 0.0170 |
| eden | 0.1378 | 24 | 0.0178 | 0.0023 | 0.0170 |
| ew | 0.1454 | 6 | 0.0376 | 0.0049 | 0.0170 |
| ew | 0.1454 | 24 | 0.0188 | 0.0024 | 0.0170 |
| kpz | 0.1541 | 6 | 0.0398 | 0.0052 | 0.0170 |
| kpz | 0.1541 | 24 | 0.0199 | 0.0026 | 0.0170 |

Jacobian verified against finite differences: max abs error 1.9e-10.
Direction gate holds everywhere (info ratio 0.017 ≤ 1, i.e. marginalizing
unknown nuisances under the prior reduces information to ~1.7% of the
fixed-nuisance ceiling — nuisance uncertainty dominates by roughly a factor
of 59, an intuitive quantification of "the nuisance is the hard part").

**Comparison to the amortized estimator's synthetic RMSE (0.106):** computed
at exp76's own noise ceiling (σ=0.10) and m=1 (matching the single-ladder-
per-example synthetic generation regime, not a real system) — direct
integration over exp76's σ~U[0,0.10] prior is not attempted, since 1/σ²
diverges as σ→0 under a uniform density with support touching zero; this
representative-ceiling choice is a documented deviation. VT bound = 0.0634;
amortized RMSE = 0.106 > bound (does **not** beat the bound — the correct
direction; had it come out below, that would indicate a bug per the plan's
own instruction).

## 5. Task 4 (G-83d): Gaussianity diagnostic κ — DIAGNOSTIC ONLY

A kernel-density score estimate at n≈24–300 (bootstrap-resampled) points is
indicative, not theorem-grade, per explicit instruction; label carried
through to every number below.

| System | κ_median (across L) | 90% CI envelope | Straddles 1? | √κ |
|---|---|---|---|---|
| bd | 0.924 | [0.789, 1.264] | **Yes** | 0.961 |
| eden | 0.927 | [0.796, 1.316] | **Yes** | 0.963 |
| ew | 0.919 | [0.802, 1.390] | **Yes** | 0.959 |
| kpz | 0.922 | [0.794, 1.363] | **Yes** | 0.960 |

Skewness/kurtosis of log W_sat across the 24 seeds are mild at every
(system, L) — point skewness in roughly [−0.4, 0.9], excess kurtosis in
roughly [−1.2, 0.7], with 90% bootstrap CIs (500 reps) that mostly contain
zero. No system shows dramatic, consistent non-Gaussianity.

**Reading:** every system's CI straddles 1 (Gaussian). Taken at face value
this says the Gaussian floor is neither clearly conservative nor clearly
anti-conservative for these systems' seed-mean noise — consistent with mild
skew/kurtosis. **However**, see anomaly 3 below: the underlying estimator
has a known, corrected bias (leave-one-out fix), and even after correction
retains enough finite-sample noise that a value straddling 1 should be read
as "not distinguishable from Gaussian at this sample size," not as a precise
measurement.

## 6. Task 5 (G-83e): plug-in σ propagation into a floor interval

Five-tuple: (Level-0, class N=1/U=4/ω_min=0.3, design=real design,
σ=per-system measured with χ²-based 90% interval combined_dof=161, m=24).

| System | σ [lo, mid, hi] | floor [lo, mid, hi] | predicted rel. err | observed rel. err |
|---|---|---|---|---|
| bd | [0.0174, 0.0190, 0.0209] | [0.2637, 0.2707, 0.2777] | 0.1115 | 0.0925 |
| eden | [0.1263, 0.1378, 0.1518] | [0.4254, 0.4348, 0.4441] | 0.1115 | 0.0925 |
| ew | [0.1333, 0.1454, 0.1602] | [0.4301, 0.4395, 0.4488] | 0.1115 | 0.0925 |
| kpz | [0.1413, 0.1541, 0.1698] | [0.4371, 0.4441, 0.4559] | 0.1115 | 0.0925 |

Every interval contains its point floor (true by construction of the
[5th-percentile-σ, point-σ, 95th-percentile-σ] ordering combined with the
floor's monotonicity in σ, but verified explicitly, not assumed). Observed
relative error (0.0925, identical across systems because it depends only on
`n_eff=7` and `m=24`, not on σ itself) sits at 0.83× the predicted
√(2/(n_eff·(m−1))) = 0.1115 — same order of magnitude, consistent with the
approximation noted below (anomaly 4).

---

## 7. Anomalies, bugs found, and deviations from plan

1. **Convex-hull optimizer under-convergence, same failure mode as exp81's
   multivariate private/shared bug, caught by the same kind of invariant.**
   J (more atoms) strictly nests J' < J (duplicating an atom set exactly
   reproduces the smaller mixture's function value), so hull_floor(J) must
   be nondecreasing in J. Initial runs (independent random multi-start per
   J) violated this at J=4 vs J=2 (J=4 reported a *worse*, i.e. smaller,
   gap than J=2 had already achieved would be required for validity —
   concretely, J=4's raw gap came out larger than J=2's, meaning the
   9-parameter jump to 25-parameter search under-converged). First fix
   attempt (warm-start using only the single highest-λ atom from the
   smaller solution) was insufficient — it under-represents true mixtures
   and still failed monotonicity (J=4 gap 1.84e-7 > J=2 gap 2.76e-8, a
   direct check performed before trusting the full task run). Root cause:
   the fix also needs the exact converged constant `c`, not just the
   atoms — a first correct-atoms-but-wrong-c version still failed
   (reconstructed warm-start objective value 1.04 vs the true smaller-J gap
   2.76e-8, an enormous mismatch, traced directly). Final fix: propagate
   the FULL mixture (all λ,u,w AND c) from J' to J via exact atom
   duplication (λ halved per duplicate), applied recursively (J=8 warm-
   starts from J=4, from J=2, from J=1). Verified: monotonicity now holds
   at every (U,m) cell in the final run.
2. **Read-modify-write race between two concurrently-running background
   tasks clobbered a completed task's JSON entry**, twice. Task 2's ~10-
   minute background run reads-then-rewrites the shared
   `audit_response.json` on completion; running Task 5 while Task 2 was
   still in flight caused Task 2's later write (based on an earlier read)
   to silently drop Task 5's entry. Caught by checking `sorted(rep.keys())`
   after the run and finding `task5_sigma_propagation` missing; fixed by
   re-running Task 5 once Task 2 had fully completed. Sequencing/locking
   note for future multi-task scripts sharing one output file, not a
   computational bug.
3. **KDE-based Fisher information estimator had a persistent (non-
   vanishing) downward bias, caught by a Gaussian sanity check.** The
   initial `kde_fisher_info` evaluated the fitted density and its
   derivative AT the same points used to build the KDE (self-inclusion);
   this inflates the density estimate exactly at evaluation points (the
   kernel's own maximum sits there), biasing the estimated Fisher
   information down. Diagnosed by fitting the estimator to KNOWN Gaussian
   data (I should → 1): the self-inclusion version gave κ≈0.76 at n=24,
   n=100, AND n=1000 — a bias that does NOT shrink with sample size,
   which is the signature of an estimator bug rather than ordinary finite-
   sample noise (which would shrink). Switched to leave-one-out KDE
   evaluation (each point's own kernel excluded from its own density/
   derivative estimate); recovers κ≈0.95–0.97 at n=1000–5000 on true
   Gaussian data, and moved the real per-system estimates from ≈0.76 (all
   systems, suspiciously identical, another tell) to ≈0.92–0.93 with CIs
   straddling 1 — the qualitatively different and far more defensible
   result reported in §5. Without this fix, the (wrong) conclusion would
   have been "all four systems show CI clearly below 1," which would have
   been read as violating the Cramér-Rao/Stam inequality (κ≥1 for any true
   distribution) — a red flag that motivated hunting for the bug rather
   than reporting the number.
4. **Task 5's χ²-interval-on-a-median is an approximation, not exact.** The
   per-system σ is defined (matching exp77) as the MEDIAN of 7 per-L
   sample standard deviations, not a pooled/mean estimate; treating
   σ_point² as if it were a pooled variance with combined_dof =
   n_eff·(m−1) = 161 borrows the clean χ² sampling theory of a pooled
   estimate for a statistic (the median) that does not exactly have it.
   The resulting interval is a reasonable, standard approximation (and the
   observed/predicted relative-error ratio of 0.83 across all four systems
   is consistent, not wildly off), but is flagged as approximate rather
   than exact, per the plan's own framing of this task as improving on a
   point estimate, not delivering an exact interval.
5. **Van Trees prior-curvature term (I_prior) not computed explicitly.**
   The standard van Trees inequality is Var ≥ 1/(E[I_data] + I_prior); this
   report uses the common simplification I_prior ≈ 0 (valid when the prior
   is wide/flat relative to the likelihood's curvature), which is not
   rigorously justified here for the piecewise-uniform F1 prior (uniform
   densities have zero curvature in their interior but a technical boundary
   term in the fully rigorous van Trees construction, which was not
   computed). Given the data term already dominates by a wide margin at
   real per-system σ (the resulting bounds, 0.002–0.04, are far below the
   prior's own width, ~0.9 for alpha alone), omitting I_prior is very
   unlikely to change the qualitative conclusion, but the simplification is
   stated explicitly rather than silently assumed.

## 8. What we did NOT do

- **Full exp76 5-family mixture prior for van Trees**: used F1_power only,
  as explicitly permitted by the plan's fallback clause. The other four
  families (F0 pure, F2 two-term, F3 intrinsic/Krug-Meakin, F4 log) were not
  incorporated; a mixture-prior van Trees bound would require either a
  discrete mixture-of-Jacobians treatment or per-family bounds reported
  separately, neither attempted here.
- **Rigorous van Trees I_prior term** for the uniform priors used (anomaly
  5 above) — the flat-prior simplification was used throughout.
- **Exact (non-approximate) sampling distribution for the median-of-7 σ
  estimator** in Task 5 — a χ²-pooled-variance approximation was used
  instead (anomaly 4).
- **Optimal atom placement for the convex hull** (Task 2): atoms were
  found by unconstrained multi-start optimization plus the exact-nesting
  warm start, not by any analytic construction; J=8's identical result to
  J=4 suggests 4 atoms already saturate what this design can achieve, but
  this was not proven, only observed.
- **A rigorous accounting of the double-bootstrap's own coverage
  properties** for Task 4's κ CIs — the double bootstrap was implemented
  as instructed and its output reported, but no separate validation (e.g.
  against a known-truth non-Gaussian synthetic case) was performed beyond
  the Gaussian sanity check that caught the KDE bias (anomaly 3).
- **Corrupt-index git workaround**: not needed this session (`git` behaved
  normally throughout); the `GIT_INDEX_FILE=/tmp/gidx` fallback mentioned in
  the plan was not exercised.

---

Committed on branch `exp81`. `CLAIMS_REGISTER.md` and `MANUSCRIPT_OUTLINE.md`
were not touched.
