# Request for expert review: a minimax resolution floor for finite-size scaling

**Audience:** a statistician or information theorist with minimax/estimation-theory
background. Self-contained; no repository reading required. Full derivation chain:
`ml_paper/THEORY_minimax_floor.md`; computation: `experiments/77_minimax_floor.py`;
repository: github.com/adamfbentley/universality-discovery.
**Author:** Adam Bentley (adam.f.bentley@gmail.com). ~30–60 minutes of expert time
requested on five specific questions (§4).

## 1. The claim, in statistical language

Data: y_{i,s} = c + α·x_i + g(L_i; η) + σ·ξ_{i,s}, where x_i = log L_i at n = 7
design points (L ∈ {32,...,256}), s = 1..m replicates, ξ iid N(0,1), and g is a
nuisance "correction to scaling" from a declared class — primary case
g(L; B, ω) = log(1 + B·L^{−ω}), ω ∈ [0.3, 2.5], amplitude bounded
(u := B·L₁^{−ω} ∈ [−0.75, 4]). Target: the slope α (a critical exponent; its
value at L→∞ is the physics). σ is measured from replicates.

Claim: for any estimator α̂, a Le Cam two-point argument with the nuisance pair
chosen adversarially gives a computable resolution floor
Δα*(design, σ, m, class) = max{Δα : D²(Δα) ≤ σ²/m}, where the "confusion gap"

    D²(Δα) = min_{c, η₁, η₂} Σᵢ [Δα·xᵢ + c + g(Lᵢ; η₁) − g(Lᵢ; η₂)]²

is computed by direct bounded optimization (multi-start, exact — a construction,
so valid for the impossibility direction regardless of optimizer quality). For
the physical benchmark: Δα* ≈ 0.27–0.44 at m = 24, i.e. exponent differences
physicists routinely argue about are undecidable from these designs without
correction-structure assumptions. Companion results: the floor is nearly flat
in m (identifiability-limited), shrinks only with decades of L, and an analytic
scaling law for the gap, E_N ≈ c_N·√T·U·(ΔαT/U)^{N+1} for N-term correction
classes on a window of log-length T, with c₁ = 0.0375 exact (construction
optimal at N=1) and c₂, c₃ numerically certified.

## 2. What we believe is proven vs. numerical vs. open

Proven / theorem-grade: the Le Cam reduction (standard); the impossibility
direction (explicit confusion pairs are constructions); the Richardson-weights
upper-bound construction and its closed-form constant (Appendix A of the theory
note); the reduction of the harmonic-node case to constrained polynomial
approximation of log(1/t).

Numerical, certified: the exact-optimization floor values; c₂ ≈ 0.0216,
c₃ ≈ 0.019; scale-law checks in all four variables.

Open, and disclosed: the analytic *lower* bound on E_N (so no sharpness claims
are made anywhere); optimal nodes for N ≥ 2; a known-buggy linearized code path
(never used for published numbers); Gaussianity and independence-across-L are
modeling idealizations (checked approximately in data; independence holds by
construction — each (L, s) is a separate simulation).

## 3. Context for why this matters

The floor converts a decade of "our method failed to recover exponents at small
L" folklore into a computable statement, quantifies the value of prior knowledge
about corrections (amplitude bounds are worth orders of magnitude in replicates),
and pairs with an amortized (simulation-based) estimator that operates near the
floor under a declared correction prior. The same normal form covers
finite-size scaling, lattice-QCD energy extraction, tail-index estimation, and
neural-scaling-law fits. The nearest literature we have found: Hall & Welsh
(Ann. Statist. 1984) and Drees on tail-index minimax under second-order regular
variation; Lepage-style constrained fits in lattice QCD; Manski partial
identification. We have found no minimax/identifiability treatment of
corrections-to-scaling in the FSS literature.

## 4. The five questions we would most value expert judgment on

1. **Robustness of the bound to the Gaussian model.** The KL is computed for
   Gaussian seed-means. How much of the floor survives realistic departures
   (the data are log-transformed sums; approximate checks look mildly
   non-Gaussian)? Is there a cheap Hellinger/robust-divergence version worth
   computing alongside?
2. **Two-point vs. richer constructions.** Would a Fano/Assouad or
   multi-point construction materially strengthen the floor here, or is the
   two-point bound essentially the right tool given the one-dimensional target
   and the nuisance structure?
3. **The sharpness gap.** Without the analytic lower bound on E_N, we claim
   only impossibility ("no estimator resolves below Δα*"), never optimality of
   the constant. Is our proposed route (annihilator operator + Newman-type
   Markov inequalities for exponential sums / Borwein–Erdélyi restricted
   Müntz systems) the right attack, and is any of this already known?
4. **Class-conditionality in practice.** Every floor is conditional on
   (N, U, ω_min). We report the tuple with every number. Is there a better
   framing (e.g. Manski-style assumption hierarchies) that makes the
   conditionality harder to misuse by practitioners?
5. **Priority, and a specific mapping to check.** Our own follow-up analysis
   (ml_paper/SIMULATED_AUDIT.md) suggests this problem is an instance of the
   Donoho (Ann. Statist. 1994) optimal-recovery/modulus-of-continuity theory
   for linear functionals over convex classes, in the form applied by
   Armstrong & Kolesár (Econometrica 2018) to fixed-design regression with a
   convex nuisance class. Is that mapping exact? If so: (a) does affine
   near-sharpness (the ≈1.25 factor) transfer once our nonconvex correction
   class is convexified, and (b) is there prior work applying this machinery
   to corrections-to-scaling or continuum extrapolation that we must defer
   to?

## 5. Known weaknesses, disclosed upfront

The floor numbers depend 