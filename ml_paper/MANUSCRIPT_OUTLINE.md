# Manuscript Outline (v2, 2026-07-03 — post-Donoho repositioning)

Supersedes the v1 outline (see git history). Paper 2 remains the primary
target. Every claim below is backed by a gated, audited computation on disk;
the five-tuple discipline (observable level, class N/U/ω_min, design, noise
source, m) applies to every reported number.

---

## Paper 2 (primary): the identifiability paper

**Working title:**
Identifiability of critical exponents from finite scaling windows:
exact floors, honest confidence intervals, and the value of correction
knowledge

**Alternative:**
The information content of a finite-size scaling window

**Venue:** Machine Learning: Science & Technology (primary), SciPost Physics
(alternative). arXiv: cond-mat.stat-mech + stat.ME cross-list (the stat.ME
cross-list is new and deliberate — the Armstrong–Kolesár audience).

**Elevator pitch:** Estimating an asymptotic critical exponent from a finite
scaling window is a linear-functional inference problem of the
Donoho (1994) / Armstrong–Kolesár (2018) type: the exponent is a linear
functional of the mean vector, and corrections to scaling are a bounded
nuisance class. This identification makes every question computable for
real designs: a model-conditional two-point lower bound on worst-case risk,
honest confidence intervals under declared assumptions, the value of selected
forms of prior knowledge (amplitude bounds, spectral bounds, sparsity), and the
value of richer observables. Computed for the standard surface-growth
benchmark, these quantities diagnose one source of difficult exponent recovery;
they do not establish a universal optimum outside the specified model,
observable, design, and nuisance class.

### Contributions (state in §1, exactly these, exactly this delineation)

Imported (cited, not claimed): the minimax linear-functional machinery
(Le Cam; Ibragimov–Khasminskii; Donoho 1994; Donoho–Liu; Cai–Low;
Armstrong–Kolesár 2018), and the declared-prior fitting philosophy
(Lepage et al.).

New:
1. The identification itself: FSS extrapolation (and its normal-form
   siblings: growth exponents, correlation lengths, lattice-correlator
   energies, neural-scaling exponents) as a Donoho-type problem, with the
   RG dictionary (ω_min = smallest irrelevant eigenvalue; amplitude bound =
   irrelevant-operator amplitude; marginal operators destroy
   identifiability).
2. The closed-form modulus for bounded exponential-sum classes:
   E_N = c_N·√T·U·(ΔαT/U)^{N+1}, c₁ = 0.0375 exact, c₂/c₃ certified;
   binomial-identity reduction to polynomial approximation of log(1/t);
   the three-regime identifiability hierarchy (fixed-N power law /
   free-N exponential / unbounded-amplitude vacuity).
3. Computed floors for the surface-growth benchmark at real designs and
   measured noise, with robustness: floors 0.27–0.44 at L ≤ 256 (m=24);
   nearly flat in seeds; decades-not-seeds resolution law; ±3% under σ
   error; Gaussianity diagnostic κ consistent with 1; nuisance
   marginalization destroys 98.3% of the Fisher information about α (59×,
   design-only) — the one-sentence reason FSS is hard.
4. The observable-hierarchy law (exactly solvable fractional-EW testbed,
   DPI-gated): full-spectrum observation resolves α 6–10× finer than the
   summary ladder under tight amplitude bounds; the advantage vanishes at
   loose bounds, where α is unidentifiable from ANY observable at the
   design. Data richness and correction knowledge are complements, not
   substitutes. Multivariate summary gains decompose as best-channel
   rescaling + small increments (+7% shared-spectrum, +37% correlated-noise
   common-mode rejection).
5. Estimation at the limit: amortized (declared-prior) estimator recovers
   BD's α̂ ≈ 0.50 ± 0.05 (syst) ± 0.03 (stat) where classical ansatz fits
   scatter 0.36–0.70; sits within ~1.7× of its van Trees bound; exact
   finite-sample honest CIs from the (hull) modulus via the A–K
   construction [exp84 numbers to drop in].
6. The reporting standard: every floor/CI carries its five-tuple; a
   data-driven lower bound on U (A–K-style specification test) replaces
   arbitrary class declaration [exp84].

### Section plan → results mapping

1. **Introduction** (1.5 pp) — the inference-problem framing, the
   contribution list above, the audit-trail note (gate ledgers and
   adversarial audits as supplementary material — state it plainly as
   methodology).
2. **The normal form and the mapping** (1.5 pp) — setup, modulus identity
   D↔ω, scope conditions (per-observable, class-conditional, Gaussian
   model with κ diagnostic), convexity status and the hull.
   [THEORY note §§1–3, Appendix G]
3. **Floors for the surface-growth benchmark** (2 pp) — exp77 floors +
   exp83 robustness (σ intervals, κ, monotonicity guard); the 59× van
   Trees number; seeds-vs-decades; value-of-knowledge tables (U, ω_min,
   sparsity). [floor.json, audit_response.json]
4. **The modulus in closed form** (1.5 pp) — exp79 lemma, constants,
   binomial/log-polynomial reduction, three-regime hierarchy, RG
   dictionary. [79 certificates; THEORY Addenda 2–3, Appendix D]
5. **What richer observables buy** (1.5 pp) — exp81f aligned hierarchy
   table + complementarity law; multivariate decomposition; the
   no-identifiability regime. [fixpass.json]
6. **Inference at the limit** (1.5 pp) — exp76 amortized + LOFO + referee
   checks; van Trees comparison; exact honest CIs [exp84]; the
   Bayes-vs-minimax gap as the price of the prior.
7. **Transfers** (1 p) — β, Ising ν (pricing exact-solution knowledge),
   lattice correlators (Lepage connection), neural-scaling-law designs
   (floors 0.05–0.18 at published-scale noise; class-conditional framing
   only). [exp80, exp81e]
8. **Discussion** (1 p) — sloppy-models connection; limitations (Gaussian
   idealization with κ caveat; class-conditionality; one testbed for the
   hierarchy; U=0.5/1.0 saturation wrinkle); open: analytic lower bound,
   optimal nodes, adaptation in nonconvex classes, non-Gaussian raw fields.

Appendices: Richardson construction + constants; hull computation; gate
ledger index (pointer to repo audit trail); reproducibility statement.

### Pre-submission blockers (all small)

- exp84: smooth-prior van Trees; σ-resolved bound-vs-RMSE; exact optimal
  CIs from the hull modulus; data-driven U lower bound. [archive/ai_execution/SONNET_EXP84_PROMPT.md]
- Human audit reply folded in (Donoho-mapping confirmation, constants
  conventions).
- Borwein–Erdélyi / Müntz–Szász-preprints priority pass (library-grade).
- Figures: floor landscape; hierarchy heatmap; modulus N-scaling log-log;
  estimator-vs-bounds; scaling-law floor table.

## Paper 1 (secondary): clustering negative, v2

As per the external review (docs/external_reviews/): add the null-partition
fingerprint table (exp82, verified); fold in the exp72–74 mechanism; add the
exp69/71 false-positive case study to the protocol section; Lam–Shin
integrator rerun of the exp62/63 headline; N-sweep control; unsupervised
positive control; exp67b sign-flip test. Cite Paper 2 as the reason the
negative was necessary. Venue: SciPost Physics Core / PRE / JSTAT.
Cut a repo release tagged at the paper's state.
