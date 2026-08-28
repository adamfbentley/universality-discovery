# Claims Register

This file is the guardrail for the ML-focused paper. Every manuscript claim must
map to an artifact and a robustness check.

---

## Part I: Clustering Paper Claims (exp62–75)

### Central Claim

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

### Claims Allowed (exp62–75)

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

11. **Classical correction-to-scaling extrapolation cannot recover known exponents
    from L ≤ 256 data.**
    Evidence: exp75. At ω=1, EW extrapolates to α=0.70±0.15, KPZ to 0.59±0.16 —
    both known to be 0.5. The ω-sensitivity spans 0.43–0.97 across ansatz choices.
    The systematic uncertainty dominates the statistical error by ~10×.

### Claims Forbidden Unless New Evidence Is Added (exp62–75)

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

---

## Part II: Model-Conditional Risk Bound and Amortized Estimator Claims (exp76–80)

These claims support a second, separate paper. The central contribution is a
computable minimax resolution floor for finite-size scaling estimation,
accompanied by an amortized estimator that approaches the floor.

### Central Claim (Floor Paper)

For saturated-width ladders W_sat(L) under the declared Gaussian observation
model, finite-size design, and bounded correction class, a Le Cam two-point
construction gives a computable lower bound on worst-case expected absolute
exponent-estimation error. For an admissible pair separated by Δα with
D²(Δα) ≤ σ²/m, the bound is at least Δα/4 over that pair. The
reported threshold is conditional on the model, observable, design, and
nuisance bounds. It does not imply that every sub-threshold exponent pair is
indistinguishable, nor does it prove that richer-feature clustering failures
were inevitable.

### Claims Allowed (exp76–80)

**A. Amortized estimator outperforms classical correction-to-scaling fits.**

Evidence: exp76 (`results_exp76_amortized_extrapolation/summary_full24seed.json`).
- Synthetic benchmark (mixture test set, 2000 samples per prior): amortized RMSE
  0.106 vs best classical 0.165 (fit_w1); free-ω fit fails on 23% of samples.
- Transfer matrix: mixture-trained estimator generalizes across all five
  correction families (RMSE 0.088–0.123); single-family-trained estimators
  degrade to 0.19–0.33 off-diagonal.
- Required caveat: training prior is uniform α ∈ [0.05, 0.95]; the comparison is
  against classical ansatze, not against all possible Bayesian approaches.

**B. The amortized estimator recovers BD's roughness exponent where classical
fits scatter.**

> **RECONCILED 2026-07-04 (authority: DIRECTION_2026-07-04.md "Retired" + "Adversarial status ledger"; new evidence: EXP85B_REPORT.md real-data half-window).**
> The **strong form is RETIRED**: "amortized estimator recovers BD α" is prior
> selection on a ridge the data cannot adjudicate. Defensible form only:
> *correct propagation of declared knowledge*. New contradicting evidence:
> exp85b's blind BD half-window CI at the honest class (U=0.5) does **not**
> cover α=0.5 or the exp76 full-window value (only U=4 does). BD is bug-vs-
> mechanism **pending exp85c Task 2** — make no BD headline claim until it
> lands. Numbers below stand as the exp76 record but are no longer citable as
> "recovery."
>
> **UPDATE 2026-07-06 (exp85c Task 2 = BUG).** The exp85b half-window
> contradiction is diagnosed as an **estimator bug**, not a mechanism: the
> 4-point affine center is not amplitude-invariant (weights sum −0.9986 ≠ 0;
> log-L response −2.85 ≠ 1; reviewer-confirmed). So exp85b's "half-window fails
> to cover α=0.5" does **not** count against BD, and the exp76 **full-window**
> result is not impugned. Standing rule unchanged: the **strong "recovery" form
> stays retired** on the independent ridge argument (DIRECTION); and **no
> half-window BD claim** until the affine weights are constrained (Σw=0,
> Σw·x=1) and the pipeline re-run. See AUDIT_2026-07-04_recent10.md §6 (F2, N1).
>
> **UPDATE 2026-08-28 (exp86 Task 1).** Affine weights were constrained and BD
> was re-run. U=0.5 half-window CI [-0.6313, 1.4353] covers α=0.5. The interval
> is wide; this is not a recovery claim. Strong form stays retired.

Evidence: exp76 real-data evaluation on 24-seed ladders.
- BD: α̂ = 0.522, seed-bootstrap 90% interval [0.482, 0.529].
- Classical fits on the same data: BD spans 0.36–0.70 across ansatz choices.
- Leave-one-family-out control (Krug–Meakin family removed): BD → 0.532
  [0.486, 0.555]. BD recovery does not depend on having BD's textbook correction
  form in the training prior.
- EW gate: 0.532 ✓; Eden gate: 0.491 ✓; KPZ gate: 0.615 ✗ (by 0.015 over
  tolerance — attributed to integrator stationary-measure distortion, see claim D).
- **Honest combined error budget**: slice-conditional bias from discriminability
  control (claim E) implies BD α̂ ≈ 0.50 ± 0.05 (syst) ± 0.03 (stat). All claims
  about BD must report the systematic.

**C. A minimax resolution floor exists and is computable from the observed design.**

Evidence: exp77 (`results_exp77_minimax_floor/floor.json`), theory note
`ml_paper/THEORY_minimax_floor.md`.
- Le Cam two-point bound: confusion gap D²(Δα) computed by multistart bounded
  optimization; floor = largest Δα with KL(P₁‖P₂) ≤ 1/2.
- Key result: D²(0.1) ≈ 1.3×10⁻⁷ at the real L=[32,256] design. Resolving Δα=0.1
  would require ~160,000 seeds (EW/KPZ noise) or ~2,700 seeds (BD noise).
- Per-system worst-case floors, m=24: BD 0.27, EW/KPZ/Eden 0.44.
- **Floor vs m is nearly flat** (0.51/0.44/0.38 at m=6/24/96): the bottleneck is
  identifiability, not statistics. More seeds cannot overcome the floor.
- Resolution law at σ=0.15: L_max 256/512/1024/4096/16384 → floor 0.44/0.33/0.26/0.19/0.14
  (decades of L required, not seeds).
- Required caveat: adversary is single power correction with |u| ≤ 4, ω ∈ [0.3, 2.5].
  Richer families (log, two-term) can only deepen the confusion gap — the bound
  is conservative. Known implementation bug in the linearized closed-form (exact
  numerical results unaffected).

**D. The KPZ gate failure is attributed to integrator stationary-measure distortion.**

Evidence: exp78 check A (`results_exp76_amortized_extrapolation/referee_checks.json`).
- For 1D EW/KPZ in the stationary state, the exact result is W²_sat = L/12.
- EW: ratio W_sat/√(L/12) constant at 0.96 ± 0.03 — pure amplitude offset,
  explained by D/ν normalization; amplitude-invariant features make this harmless.
- KPZ: ratio swings 1.043 (L=32) → 0.918 (L=64), ≈4σ combined, then recovers.
  A genuine shape distortion at accessible L, consistent with known Lam–Shin
  discretization pathology (fluctuation–dissipation violation).
- The estimator correctly reads the distortion in the data; this is not estimator
  failure. Remedy: exact-stationary-measure integrator (not more seeds).

**E. The expert additive-width ansatz (W² = b + aL^{2α}) is rejected on BD.**

Evidence: exp78 check B.
- Free fit: α = 0.441 ± 0.011 (5σ from 0.5).
- Fixed α = 0.5: χ²/dof = 6.2 (decisively rejected).
- BD carries corrections beyond the pure additive form at L ≤ 256. This kills
  the anticipated referee rebuttal ("just use the textbook ansatz") and
  empirically demonstrates the thesis: a structurally informed but still-misspecified
  ansatz produces a confident wrong answer.

**F. The amortized estimator discriminates Δα = 0.05 at ~3σ (not prior-mean shrinkage).**

Evidence: exp78 check C (discriminability control, 300 BD-like ladders per α).
- True α ∈ {0.40, 0.45, 0.50, 0.55} → predicted means 0.425/0.496/0.548/0.607.
- Adjacent α separated by ~3σ; response slope ≈ 1.2 (not ≪ 1 as shrinkage would give).
- **Slice-conditional bias: +0.03 to +0.06** on this prior slice (true 0.50 →
  predicted 0.548). This is the systematic in claim B's honest error budget.
  Must be reported.

**G. The value of declared structural knowledge is quantified.**

Evidence: exp77 floor vs u_max sensitivity.
- Bounding correction amplitude: u_max = 4 → 0.1 shrinks floor from 0.44 → 0.077
  (σ=0.15, m=24, L≤256).
- BD's honest amplitude bound (|u| ≤ 0.5): floor → 0.023 at real design. The
  exp76 interval (±0.03 stat) sits just above it — the amortized estimator operates
  near the information limit, not beyond it.
- Interpretation: the Bayes-vs-minimax gap (declared vs smuggled prior) is the
  quantified value of structural knowledge. Classical fits smuggle the prior via
  the ansatz and fail silently when it is wrong; amortized inference declares it
  and marginalizes.

**H. The floor transfers to temporal scaling (growth exponent β).**

Evidence: exp80 part A (`results_exp80_second_observable_floors/floors.json`).
- Design: W(t) at 7 log-spaced times t=50..5000, L=1024, 8 seeds.
- Agnostic floor: β resolvable to ~0.07–0.08 at m=10–24.
- EW/KPZ β-gap (0.083 at face value) sits at/just above the β floor — β-based
  discrimination is marginally feasible at accessible scales.
- Retrodicts the exp74 asymmetry: the time window spans ~2 decades vs the size
  window's ~0.9 decades; β converged while α stayed anomalous because the β window
  is wider.
- Caveat: growth-regime t-points within a seed have temporal correlations not
  captured by the iid-noise model; floors are approximate for part A.

**I. The floor transfers to 2D Ising ν, and prices the value of exact-solution knowledge.**

Evidence: exp80 part B.
- Design: exp52d Ising (L ∈ {32,48,64,96}, 0.48 decades), noise σ~0.005–0.02.
- Agnostic floor (|u|≤1, ω≥0.3): 0.31–0.39 in 1/ν.
- Ising-honest floor (|u|≤0.3, ω≥1): 0.13–0.17.
- Ising-strict floor (|u|≤0.1, ω≥1): 0.05–0.13.
- exp52d's 7.3% deviation from ν=1 is consistent with (above) the strict-class
  floor — the observed precision is explained by, not in contradiction with, the
  floor.
- The Ising community's precision claims implicitly use exact-solution constraints
  that tighten the floor by ~5–6× at fixed design and noise. The floor framework
  prices that knowledge explicitly.

**J. The application of a Le Cam minimax lower bound to FSS may be novel; the
priority claim is provisional.**

> **RECONCILED 2026-07-04 (authority: DIRECTION_2026-07-04.md "Retired"; SIMULATED_AUDIT.md §1; LITERATURE_AUDIT.md Items 2, 9).**
> The priority claim below is **RETIRED**. The floor is an instance of the
> Donoho (1994) / Armstrong–Kolesár (2018) minimax-linear-functional theory;
> the machinery is imported, not new. Do **not** write "first minimax bound /
> lower bound for FSS" anywhere. Defensible novelty (state exactly this): (i)
> the *application* of the machinery to corrections-to-scaling; (ii) the
> physically-motivated adversary class + computed floors at real designs;
> (iii) the closed-form exponential-sum modulus (exp79) with its RG reading;
> (iv) the amortized declared-prior pairing. The original text is kept below
> for the ledger, struck.

~~Evidence: exp78 web search; discussion in~~ `ml_paper/THEORY_minimax_floor.md`
Appendix C.
- Hall–Welsh/Le Cam minimax theory is mature in extreme-value statistics (tail-index
  estimation under second-order regular variation), but no application to finite-size
  scaling was found.
- Closest related work: Lepage constrained curve fitting (hep-lat/0110175, 2001) for
  lattice QCD, which established the declared-prior philosophy; Jay–Neil Bayesian
  model averaging (different question); Braess–Hackbusch conditional stability theory
  (compactness argument, different framing). None compute a computable threshold for FSS.
- ~~Claim: "to our knowledge, first minimax lower bound on FSS exponent estimation."~~
  **RETIRED — see the RECONCILED note under the J heading.** origin/main (2026-07-16)
  had already narrowed this to a provisional "first application" pending a
  library pass. The working-tree register goes further: the machinery IS prior
  art (Donoho / Armstrong–Kolesár; LITERATURE_AUDIT Items 2/9). The remaining
  library-grade pass concerns only the *closed-form exponential-sum modulus*
  (exp79, Borwein–Erdélyi / Müntz–Szász), not the floor's priority.

**K. N-scaling law for the confusion gap: E_N ~ c_N · √T · U · (ΔαT/U)^{N+1}.**

Evidence: exp79 (`experiments/79_lemma_scaling_test.py`,
`experiments/79b_constant_certificates.py`).
- Numerical verification: N=1 ratio 0.030/0.029 (vs theory), N=2 ratio
  0.049/0.050; U-scaling ratio ~4 (N=1 → N=2), T-scaling ratio ~11.3 (matches
  √T·(T/U)^{N+1} prediction).
- Constants: c₁ = 0.0375 (exact; Richardson construction achieves it and optimizer
  cannot improve to 3 digits at tested parameters), c₂ = 0.0216 (optimizer beats
  Richardson construction by 14%), c₃ ≈ 0.019.
- **Central lemma**: minimum L² distance from Δα·x (slope on [0,T]) to the class of
  bounded N-term exponential sums is E_N = c_N·√T·U·(ΔαT/U)^{N+1}.
- **Key algebraic identity**: with Richardson weights, residual is
  Δα·(1-e^{-ωx})^N; substitution t=e^{-ωx} reduces the problem to polynomial
  approximation of log(1/t) on [0,1] (Legendre projection / Bernstein ellipse route).
- Caveat: c₁ exact claim means "construction matches optimizer to 3 digits at tested
  parameters" — not a proof of global optimality. c₂, c₃ are numerically certified
  lower bounds. Analytic lower bound proof (harmonic-node case) is the main open item.

### Claims Forbidden Unless New Evidence Is Added (exp76–80)

1. **"The amortized estimator gives the correct BD α to within ±0.03."**
   The ±0.03 is statistical only. The honest interval includes ±0.05 systematic
   from the slice-conditional bias (claim F). Must quote combined.

2. **"The floor is an absolute lower bound for all correction classes."**
   The floor depends on the declared class. With unbounded amplitudes the floor is
   zero (dense corrections; vacuous). The claim is: for declared class C_N with
   bounded amplitudes, the floor is E_N(Δα, design).

3. **"Classical extrapolation is useless."**
   Too strong. Classical fits work well when the correction form is correctly
   specified and L is large enough. The claim is: at L ≤ 256 and with the standard
   correction families, systematic uncertainty from correction-form misspecification
   exceeds the statistical precision.

4. **"The floor explains BD's clustering failure."**
   The exp62–74 clustering failures were diagnosed by exp77 as having a floor
   ≥ 0.27 (BD) or ≥ 0.44 (others), consistent with explaining them. But the
   clustering pipeline used effective exponents, not α directly; the connection
   is interpretive, not a deductive proof.

5. **"The N-scaling law is proven analytically."**
   The scaling form and constants are numerically verified and supported by the
   algebraic analysis of the Richardson residual. The analytic lower bound proof
   is an open item. Current status: numerically certified.

### Open Evidence Required Before Submission

1. **Borwein–Erdélyi prior-art pass**: Confirm no FSS minimax bound appears in the
   approximation theory or extreme-value statistics literature. Assign ~1 day.

2. **Analytic lower bound certificate for the N-scaling constant**: prove
   c_N = c_N^* (or give a tight analytic lower bound) via harmonic-node construction
   or Bernstein ellipse. Currently verified numerically.

3. **Optimal nodes for N≥2**: uniform nodes are suboptimal beyond N=1. Finding the
   optimal exponent placement is an open problem; not required for submission but
   strengthens the appendix.

4. **ω-range vs amplitude decomposition**: the 8× gap between log-family and linear
   adversary in exp77 is partly from different ω lower bounds (logged as Correction
   to Addendum 4 in the theory note). Verify the decomposition is properly attributed.

5. **KPZ integrator fix**: test whether an exact-stationary-measure discretization
   (Lam–Shin exact scheme) passes the gate. Not required for submission but would
   close the one open gate failure.

### One-Sentence Paper Claim (Floor Paper)

Under a specified Gaussian model, bounded correction class, `W_sat` summary,
and L ≤ 256 design, a Le Cam two-point construction gives a computable lower
bound on worst-case roughness-exponent estimation risk; an amortized estimator
with a declared correction prior is evaluated against that conditional bound.

### One-Sentence ML Research Question (Floor Paper)

What is the minimum structural assumption about corrections to scaling needed to
make a finite-size scaling exponent identifiable, and how does that assumption
value compare to the practical cost of obtaining more data?
