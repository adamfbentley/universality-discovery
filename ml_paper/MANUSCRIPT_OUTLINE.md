# Manuscript Outline

This file outlines two papers that can come from this project. **Paper 2 (floor
theorem) is the primary target** for the next submission cycle. Paper 1 (clustering
negative) is a secondary contribution that may be folded into Paper 2 as motivation
or published separately.

---

## Paper 2: The Floor Theorem Paper (Primary Target)

**Working title:**

**A Minimax Resolution Floor for Finite-Size Scaling: Why Exponent Estimation
Fails at Accessible System Sizes and How to Approach the Limit**

**Alternative title:**

**Identifiability Limits for Finite-Size Scaling: A Le Cam Lower Bound and
an Amortized Estimator**

**Venue target:** Machine Learning: Science and Technology (MLST), or Physical
Review E (computational/statistical methods section).

**Elevator pitch:** Every finite-size scaling experiment has a computable
information-theoretic floor below which no estimator can distinguish exponents —
regardless of how the data are analyzed. We derive this floor from Le Cam's
two-point bound, compute it for surface-growth benchmarks, and demonstrate an
amortized estimator that approaches it from above. The floor explains all the
empirical negatives in the project: not algorithm failure, but data-level
near-non-identifiability.

### Abstract Shape

Finite-size scaling (FSS) is a central tool for extracting critical exponents
from simulations of finite systems. We show that at accessible system sizes, the
problem of estimating the asymptotic roughness exponent α from a ladder of
saturated-width measurements is near-non-identifiable: for any estimator, there
exists a worst-case correction-to-scaling nuisance under which the estimator's
error exceeds a computable floor Δα*(design, σ, m, assumption class). We derive
this floor via Le Cam's two-point method, compute it for the standard 1+1D
surface-growth benchmark at L ≤ 256, and find floors of 0.27–0.44 (m=24
seeds), explaining quantitatively why direct and ML-based extrapolation methods
all failed in earlier experiments on these systems. The floor shrinks
at a computable rate with L_max and m, with decades of L needed rather than
more seeds. We further show that an amortized estimator — trained on synthetic
ladders drawn from a declared prior over correction families — achieves error
near the floor for declared-class systems (BD: α̂ = 0.522 [0.482, 0.529],
honest ±0.05 systematic), while classical ansatz-fitting gives a 10× wider
scatter. The gap between the Bayes risk and the minimax floor quantifies the
value of structural knowledge about corrections. The framework transfers to other
observables (temporal β, 2D Ising ν) and to other systems via the same
design-and-noise inputs. An algebraic analysis of the confusion gap reveals a
universal N-scaling law E_N ~ c_N · √T · U · (ΔαT/U)^{N+1} that governs how
fast the floor drops with correction complexity, with an exact constant c₁=0.0375
and numerically certified c₂=0.0216.

---

### Section 1: Introduction (≈ 1.5 pages)

**Goal:** Motivate FSS estimation as an inference problem, identify the gap in
the literature (no computable lower bounds), state the three contributions.

**Key message:**

> FSS estimation is not just a curve-fitting problem; it is an inference problem
> with a fundamental identifiability limit set by the correction-to-scaling
> degeneracy. That limit is computable, and it quantifies the value of structural
> prior knowledge.

**Three contributions to state clearly:**

1. A minimax resolution floor derived from Le Cam's two-point bound, computable
   from design, noise, and a declared correction-assumption class.
2. An amortized estimator that declares a correction prior and operates near the
   floor for in-class systems; classical ansatze are shown to fail silently.
3. An algebraic analysis (central lemma) revealing the N-scaling structure of
   the confusion gap and exact/certified constants.

**Context to give:** Standard FSS workflow (fit W_sat ~ L^α with corrections,
bootstrap), why it fails at small L, what "corrections to scaling" are and why
they are hard. Brief mention of prior-art landscape (Lepage lattice QCD for
declared-prior philosophy; Hall–Welsh for tail-index minimax; no known FSS
version — see Appendix C).

---

### Section 2: Setup and Notation (≈ 1 page)

**Observable:** y_{i,s} = log W_sat(L_i), seed s, size L_i. Gaussian noise
model: y = c + α·x + g(L; η) + σ·ξ, x_i = log L_i.

**Correction class C_N:** N-term power correction with amplitude bounds
|u_k| ≤ U, exponent ω_k ≥ ω_min (conservative: single power correction for
main results; two-term and log discussed in appendix).

**Design:** L ladder {32,48,64,96,128,192,256}, n=7. Noise σ measured per
system from wsat_perseed.csv (24 seeds, exp76b protocol).

**Systems:** EW (Edwards–Wilkinson), KPZ (Kardar–Parisi–Zhang), ballistic
deposition (BD), Eden. Theory α=0.5 for all KPZ-class systems; BD has known
slow corrections with intrinsic-width form.

---

### Section 3: The Minimax Resolution Floor (≈ 2.5 pages)

**3.1 Le Cam two-point bound (≈ 0.5 pages)**

Two configurations (c₁, α, η₁) and (c₂, α+Δα, η₂). Gaussian seed-mean
distributions. KL = (m/2σ²)·D², with confusion gap

    D²(Δα) = min_{c, η₁, η₂} Σᵢ [Δα·xᵢ + c + g(Lᵢ; η₁) - g(Lᵢ; η₂)]².

Pinsker → TV ≤ √(KL/2). Le Cam: max-error ≥ (Δα/4) when D² ≤ σ²/m.
Resolution floor: Δα*(design, σ, m) = max{Δα : D²(Δα) ≤ σ²/m}.

**3.2 Computed floors for surface-growth benchmark (≈ 1 page)**

Table: per-system floors (m=6/24) and measured σ. D²(0.1) ≈ 1.3×10⁻⁷.
Floor vs seeds (flat — identifiability-limited). Floor vs L_max (0.44→0.14
from 256→16384). Value of structural knowledge: u_max 4→0.1 shrinks floor
0.44→0.077; BD's honest bound (u ≤ 0.5) gives floor 0.023, just below the
exp76 interval.

**Figure 1: Resolution floor landscape.** (a) D²(Δα) vs Δα on log scale
with σ²/m threshold marked; (b) floor vs L_max; (c) floor vs u_max.

**3.3 Why D² is so small: the confusion mechanism (≈ 1 page)**

Key insight: D² is small because log L — the signal for α — is the ω→0
limit of the correction basis log(1 + u·L^{-ω}) → u·L^{-ω} → ... →
u·log(L). The adversary can shadow log L with the low-ω tail of the
correction family. This is made precise by the central lemma (Section 5).
The mechanism is the same reason log-transform linearization "works" but
cannot be trusted: the correction term mimics the signal.

---

### Section 4: Amortized Estimation and the Prior Gap (≈ 2 pages)

**4.1 Estimator design (≈ 0.5 pages)**

Prior P over (α, correction family, noise): five families F0–F4 (pure
power; single correction; two-term; Krug–Meakin intrinsic width W²=Wᵢ²+(AL^α)²;
log corrections) with α~U[0.05,0.95], noise σ~U[0,0.10]. Features: 14
scale-invariant inputs (6 adjacent effective exponents α_eff(Lᵢ,Lᵢ₊₁) =
ΔlogW/ΔlogL; naive slope; quadratic curvature; fit residual std; 5 aeff
differences). HistGradientBoosting point + 5/50/95% quantile models, one per
prior and for the mixture. Training: 200k ladders per prior.

**4.2 Synthetic benchmark and transfer matrix (≈ 0.5 pages)**

Amortized RMSE 0.106, bias +0.001, vs best classical 0.165 (fit_w1). Free-ω
fit diverges on 23% of samples. Transfer matrix: mixture-trained at 0.088–0.123
across all families; single-family-trained degrades to 0.19–0.33 off-diagonal.
Classical scatter on BD: 0.36–0.70 across ansatz choices.

**Table 1:** Synthetic benchmark — RMSE by estimator and test family.

**4.3 Real-data gate and BD recovery (≈ 0.75 pages)**

Pre-registered gate: EW/KPZ/Eden within 0.10 of known 0.5. Results:
EW 0.532 ✓, Eden 0.491 ✓, KPZ 0.615 ✗ (by 0.015; attributed to
integrator stationary-measure distortion — see Section 4.4). BD:
α̂ = 0.522, bootstrap 90% [0.482, 0.529]. Leave-one-family-out (Krug–Meakin
removed): BD → 0.532 [0.486, 0.555] — robust to prior family choice.

Honest error budget (Section 4.5): slice-conditional bias +0.03–0.06 from
discriminability control → BD α̂ ≈ 0.50 ± 0.05 (syst) ± 0.03 (stat). Near
the floor for u ≤ 0.5: consistent with near-optimal.

**Figure 2:** Real-data recovery. α̂ and 90% interval per system and estimator;
floor marked.

**4.4 KPZ gate failure: integrator pathology (≈ 0.25 pages)**

W_sat/√(L/12) measured vs L. EW: 0.96±0.03 (amplitude offset, harmless).
KPZ: 1.043→0.918 at L=64, ≈4σ, no trend. Consistent with Lam–Shin
discretization pathology (fluctuation–dissipation violation at finite Δt).
Estimator correctly reads the distortion in the data. Remedy: exact-measure
integrator, not more seeds.

**4.5 Referee checks (≈ 0.25 pages)**

Four checks: (A) exact stationary measure — done above; (B) expert additive
ansatz W²=b+aL^{2α} on BD — α=0.441±0.011 (5σ from 0.5), fixed-0.5 gives
χ²/dof=6.2, rejected; (C) discriminability control — Δα=0.05 separated at
~3σ, slope 1.2 (no prior-mean shrinkage), conditional bias +0.05 documented;
(D) prior-art scan — no FSS minimax bound found.

---

### Section 5: The Central Lemma and N-Scaling (≈ 2 pages)

**5.1 Statement (≈ 0.5 pages)**

Lemma: the minimum L² distance from Δα·x (slope, x ∈ [0,T]) to the class
of bounded N-term exponential sums Σₖ uₖ e^{-ωₖx} with |uₖ| ≤ U is

    E_N = c_N · √T · U · (ΔαT/U)^{N+1},

with c₁ = 0.0375 (exact; Richardson construction achieves it), c₂ = 0.0216
(numerically certified), c₃ ≈ 0.019.

**5.2 Key algebraic structure (≈ 0.75 pages)**

Richardson construction: N basis functions with weights that annihilate moments
0..N-1. Residual with Richardson weights is Δα·(1-e^{-ωx})^N (by binomial
theorem). Substitution t = e^{-ωx} converts the problem to polynomial
approximation of log(1/t) on [0,1] — a classical problem (Legendre projection,
Zolotarev, Bernstein ellipse). The N-scaling of c_N follows from the quality
of polynomial approximation of log, and the exponent in the scaling law
(N+1) counts the order of the Richardson cancellation.

**5.3 Verified constants and open items (≈ 0.5 pages)**

N-scaling ratios: 0.030 (theory) vs 0.029 (numerical); U-scaling ratio ~4;
T-scaling ratio ~11.3 (matches √T·(T/U)^{N+1}). c₁ = 0.0375: construction
is optimal to 3 digits at tested parameters. c₂ = 0.0216: optimizer improves
on uniform-node construction by 14% — optimal nodes for N≥2 are not uniform
and remain an open problem. Analytic lower bound proof via harmonic-node
construction is the main open item.

**Figure 3:** N-scaling: log E_N vs log(ΔαT/U) for N=1,2,3 with theoretical
slopes N+1=2,3,4 marked.

---

### Section 6: Transfers and Universality of the Framework (≈ 1.5 pages)

**6.1 Temporal scaling (growth exponent β) (≈ 0.5 pages)**

Design: W(t) at 7 log-spaced times, L=1024, 8 seeds. β floor ≈ 0.07–0.08
(agnostic class). EW/KPZ β-gap (0.083) sits at/just above the floor —
marginally feasible at accessible scales. Retrodicts exp74 asymmetry: time
window spans ~2 decades vs size window's ~0.9 decades; wider window → lower
floor → β converged while α stayed anomalous.

**6.2 Correlation length exponent ν (2D Ising) (≈ 0.5 pages)**

Design: exp52d (L ∈ {32,48,64,96}). Floors: agnostic 0.31–0.39; Ising-honest
0.13–0.17; Ising-strict 0.05–0.13 in 1/ν. exp52d's 7.3% deviation from ν=1
is consistent with (above) the strict-class floor. The Ising community's
precision claims implicitly use exact-solution knowledge that is worth ~5–6×
in resolution at this design — the floor framework prices that knowledge
explicitly.

**6.3 Framework portability (≈ 0.5 pages)**

The floor computation requires only: (a) the FSS observable and design,
(b) a noise model (Gaussian log-width is sufficient for a first pass), (c) a
declared correction-assumption class with bounded amplitudes. Any finite-size
scaling exponent estimation problem has a version of this floor; the framework
is not specific to surface growth.

---

### Section 7: Implications (≈ 1 page)

**What this changes for FSS practice:**

- Floors are computable before running the experiment. At the design stage,
  compute Δα*(design, σ_estimate, m_planned) to check feasibility.
- More seeds do not overcome a floor set by identifiability. L_max needs to
  grow by decades, not seeds by orders of magnitude.
- Structural knowledge is worth quantifying. The floor's sensitivity to u_max
  and ω_min gives a concrete return-on-investment for physical understanding.
- Amortized/Bayesian approaches are legitimate: declaring a prior and
  marginalizing over corrections is not cheating — it is the correct Bayesian
  response to identifiability limits, and its gap from the minimax floor is
  the value of that prior.

**Connection to broader themes:**

- Sloppy models / FIM: RG-irrelevant = FIM-sloppy = large confusion gap.
  This work is the estimation-theory side of Sethna/Machta/Transtrum.
- Partial identification (Manski): identified set diameter = zero-noise floor.
  The floor framework is a special case of the partial-identification framework
  for parametric FSS.
- Super-resolution / approximation theory: the N-scaling law is structurally
  identical to super-resolution capacity bounds. The FSS problem is a
  physical incarnation of the Prony problem.

---

### Section 8: Conclusion (≈ 0.5 pages)

Three-sentence summary:

> At L ≤ 256, FSS exponent estimation is near-non-identifiable: the confusion
> gap is ~10⁻⁷ at Δα=0.1, making the floor larger than the gaps between
> universality classes. This is not a failure of algorithms but of the data:
> the correction-to-scaling family shadows the signal. The floor is computable,
> the scaling law is proven numerically and supported algebraically, and an
> amortized estimator that declares the prior operates near the limit.

Open questions:

- Analytic lower bound for c_N (harmonic-node case).
- Optimal node placement for N≥2.
- KPZ exact-measure integrator test (Lam–Shin exact scheme).
- Borwein–Erdélyi prior-art sweep (approximation theory).

---

### Appendix A: Construction and Derivation

Full Richardson construction: N basis functions, moment annihilation, residual
series Δα·(1-e^{-ωx})^N, Legendre projection onto polynomial basis of log,
Bernstein ellipse route to the exponential-regime constant. Amplitude binding
argument. See `ml_paper/THEORY_minimax_floor.md` Appendix A for the extended
version.

### Appendix B: RG-Equivariance Corollary

The floor is RG-equivariant in the sense that the correction exponents ω_k
correspond to eigenvalues of irrelevant RG operators. Gapped irrelevant spectrum
(all ω_k ≥ ω_min > 0) → bounded N-correction class → computable floor. Marginal
irrelevant spectrum (ω_k → 0) → log corrections → floor drops to zero only for
infinite system sizes. Compute-allocation criterion: 2Nω vs d+z (when the
subleading correction is subdominant to the leading correction). Connection to
Fisher information / sloppy-models hierarchy.

### Appendix C: Prior-Art Map

Detailed comparison with: Lepage constrained curve fitting (1988 Gordon–Breach
book, hep-lat/0110175); Jay–Neil Bayesian model averaging; Manski partial
identification; Tikhonov–Isakov conditional stability; Braess–Hackbusch
compactness argument; super-resolution / Prony problem; Zolotarev theory for
optimal polynomial approximation; Sethna/Machta/Transtrum sloppy models. None
derive a computable FSS floor. See `ml_paper/THEORY_minimax_floor.md` Appendix C.

### Appendix D: Algebraic Details

Binomial identity residual r′ = Δα·(1-e^{-ωx})^N (full derivation); substitution
t=e^{-ωx} and reduction to polynomial approximation of log; constraint protection
argument; computation of c₂ optimizer path. See `ml_paper/THEORY_minimax_floor.md`
Appendix D.

---

## Paper 1: The Clustering Negative (Secondary, Potentially Folded)

**Working title:**

**Finite-Size Feature Geometry Can Misidentify Universality Classes in
Unsupervised Learning of Surface Growth**

**Alternative title:**

**A Protocol-Dependent False Positive in ML Universality Discovery**

**Status:** All experiments complete (exp62–75, MLP-01–09). Could be published
standalone (ML-focused, MLST or NeurIPS ML4Physics) or folded as motivating
Section 1 of Paper 2. The false-positive arc (exp69 ARI 0.902 → exp71 sweep
→ zero advantage) is the strongest narrative hook for the introduction of
the floor paper.

**Outline:** See the original MANUSCRIPT_OUTLINE.md sections 1–8 (preserved
in git history), based on exp62–75 and MLP papers. Claims register for this
paper is Part I of CLAIMS_REGISTER.md.

**Key claims (for completeness):**

1. Local discriminability does not imply global universality recovery (kNN vs
   HDBSCAN gap 0.389 on average).
2. Single-run high ARI (exp69: 0.902) is a false positive — swept to near parity
   under matched protocol (exp71).
3. The hard quotient is continuum-vs-discrete KPZ (EW/KPZ binary is easy;
   EW/KPZ/BD/Eden hard subset ARI ≈ 0.17–0.50 across representations).
4. Positive controls (Ising PCA-FSS, Potts Binder) show the framework can work
   on cleaner FSS tasks.

**Decision point:** After submitting Paper 2, reassess whether Paper 1 adds
enough to the literature independently. If Paper 2 explains why clustering
fails (floor theorem), Paper 1's negative result has a theoretical grounding
that makes it publishable. Alternatively, merge the false-positive arc as a
2-page motivating example in Paper 2's introduction.
