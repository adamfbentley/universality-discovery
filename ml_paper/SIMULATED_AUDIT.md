# Simulated expert audit of the floor theorem (adversarial statistical review)

**Status: AI-simulated review, 2026-07-03 — preparation for, not a substitute
for, the human audit requested in AUDIT_BRIEF.md. Written from the standpoint
of a hostile referee in the nonparametric-minimax tradition. Literature
anchors verified via web search where noted; the human expert should still
check the mapping in §1.**

---

## 0. Verdict in three sentences

The construction is sound and the impossibility direction is valid as stated
for the Gaussian model. However, the *framework* is not new: the setup —
fixed design, Gaussian errors, a linear functional of a mean vector
constrained to lie in a class — is precisely the Donoho (1994) optimal-recovery
/ modulus-of-continuity theory, whose modern applied form is Armstrong &
Kolesár (Econometrica 2018). This is simultaneously the review's harshest
finding (the priority claim must be narrowed) and its most valuable gift
(near-sharpness, the project's largest open item, may follow from that theory
essentially for free once the class is convexified).

## 1. The central finding: this is the Donoho / Armstrong–Kolesár framework

The floor problem is: y = μ + noise, μ ∈ M = {c·1 + α·x + g : g ∈ G(U, ω)},
estimate the linear functional T(μ) = α. Donoho, "Statistical Estimation and
Optimal Recovery" (Ann. Statist. 22, 1994) shows that for CONVEX M under
Gaussian noise, the minimax risk for a linear functional is characterized —
up to small explicit constants — by the modulus of continuity

    ω(ε) = sup{ T(μ₁) − T(μ₂) : ‖μ₁ − μ₂‖ ≤ ε, μᵢ ∈ M },

and that minimax *affine* estimators come within a factor ≈1.25 of fully
minimax (building on Ibragimov–Khasminskii). The confusion gap D²(Δα) is
exactly the inverse of this modulus; the "resolution floor" is ω(σ/√m) in
Donoho's notation. Armstrong & Kolesár ("Optimal Inference in a Class of
Regression Models," Econometrica 2018; also their nonparametric-regression
papers) develop the same machinery for exactly this situation — fixed-design
regression, normal errors, linear functional, convex nuisance class — with
honest confidence intervals and software, applied to regression-discontinuity
designs in econometrics.

Consequences:

**(a) Priority must be restated.** "To our knowledge the first minimax bound
for FSS" survives only in the narrow sense: the first *application* of this
machinery to corrections-to-scaling in statistical physics. The machinery
itself is thirty years old and actively used in econometrics. The paper's
positioning must cite Donoho 1994, Donoho–Liu, Ibragimov–Khasminskii, Cai–Low
(nonconvex adaptation), and Armstrong–Kolesár, and present the contributions
as: (i) recognizing FSS extrapolation as a linear-functional problem of this
type; (ii) the physically motivated adversary class and computed floors for
real designs; (iii) the closed-form modulus lemma (exp79) with its
approximation-theory content and RG reading; (iv) the amortized estimator as
the declared-prior counterpart. That is still a good paper. It is not the
paper the current abstract describes.

**(b) Sharpness may be free.** The project's largest open item — the analytic
lower bound needed for optimality claims — may be bypassable: for the
CONVEXIFIED correction class, Donoho's theory supplies matching upper bounds
(affine estimators within ≈1.25 of minimax). The physical class
{log(1+u·e^{−ωx})} with bounded u and an ω-interval is nonconvex; its convex
hull is a larger adversary, so hull-floors are ≥ class-floors (conservative
impossibility, same direction as every other approximation in the project),
and sharpness statements hold relative to the hull. Whether the hull is much
larger than the class at these designs is a computable question (one
experiment: compare hull-floor to class-floor numerically). The exp79 lemma
then becomes: *the closed form of Donoho's modulus for bounded exponential-sum
classes* — a cleaner and better-positioned contribution than a bespoke
"confusion gap" lemma.

## 2. Gaussianity (brief Q1) — the sharpest legitimate attack, and its answer

The floor needs an UPPER bound on distinguishability (KL/TV). Gaussian noise
minimizes Fisher information at fixed variance, so for real (non-Gaussian)
noise the true KL between the adversarial pair is LARGER than the Gaussian
formula: the computed floor could overstate impossibility for real data. This
is the one place where the idealization cuts against the theorem rather than
for it. Three-part response, all implementable:

1. The confusion-gap shifts are minuscule (D ~ 1e-4..1e-7), so the small-shift
   quadratic regime applies exactly: KL ≈ I·Δμ²/2 with I the Fisher
   information of the *seed-mean* noise.
2. The seed-mean Gaussianizes by CLT, and Fisher information is monotone along
   convolution (Stam; Artstein–Ball–Barthe–Naor), so the excess information
   I·σ̄² − 1 decays with m and is controlled by the single-seed
   non-Gaussianity.
3. Therefore: estimate single-seed skewness/kurtosis from wsat_perseed.csv,
   report the correction factor κ = I·σ̄² as a band, and state the floor as
   Δα*/√κ-robust. One lemma, one diagnostic plot. Also do the heteroscedastic
   per-L version (already flagged in the note) at the same time.

## 3. Two-point vs. richer bounds (brief Q2)

For a scalar functional, two-point/modulus IS the right tool; Fano/Assouad
buy nothing material here (they matter for adaptive/multi-parameter rates).
One genuine addition instead: a **van Trees (Bayesian Cramér–Rao) bound under
the exp76 sampling prior**. The amortized estimator is evaluated by average
RMSE; the minimax floor is a worst-case object; the referee-facing mismatch
(exp81 audit, anomaly 9) is resolved by exhibiting the matching average-case
lower bound. This is a short computation and closes a real gap in the paper's
claim structure ("operates near the information limit" currently compares an
average risk to a worst-case floor).

## 4. Practice framing (brief Q4)

Armstrong–Kolesár practice answers this: report sensitivity curves in the
class constants (floor vs U — already computed) and adopt breakdown-analysis
language from partial identification ("the largest U under which resolution
δ is achievable"). No new mathematics required; adopt their reporting
conventions and cite them.

## 5. Priority scan directions (brief Q5)

Anchors now known: Donoho 1994; Donoho–Liu (geometrizing rates); Ibragimov–
Khasminskii 1985; Cai–Low (adaptation, nonconvex classes); Armstrong–Kolesár
2018 + software; Hall–Welsh 1984 / Drees (tail index, the second-order-
regular-variation cousin); Lepage (declared priors); plus the note's existing
approximation-theory anchors (Braess–Hackbusch, Borwein–Erdélyi, super-
resolution). Still-unclaimed territory after this scan: the application to
corrections-to-scaling/FSS, the physical adversary class, the closed-form
modulus for exponential sums with the RG spectrum reading, and the amortized-
estimator pairing. Search terms for the library pass: "minimax linear
functional convex class regression discontinuity", "modulus of continuity
exponential sums", "honest confidence intervals nuisance smoothness class".

## 6. Minor technical items (fix cheaply, none fatal)

- **Monotonicity of D²(Δα)**: for a nonconvex class this is not automatic;
  the floor definition max{Δα : D² ≤ σ²/m} implicitly assumes it. Add a
  numerical assertion (or take the running max) — same invariant-discipline
  class as the exp81 audit's U-monotonicity gate.
- **Optimizer under-convergence is safe** for impossibility (computed D² ≥
  true D² ⇒ computed floor ≤ true floor ⇒ claims conservative). State this
  one-line argument explicitly in the paper; it immunizes the numerics.
- **Plug-in σ**: propagate σ̂'s sampling error (χ² with (m−1) dof per L) into
  a floor interval rather than a point value.
- **Temporal transfer (exp80 A)**: within-seed correlations remain unmodeled;
  either whiten or downgrade the β-floor to "approximate" in the paper (the
  note already flags this — keep it flagged in the paper too).
- The Δα/4 constants and Pinsker step are standard and correct as written.

## 7. What survives, what changes

Survives: the model, the reduction, the computed floors (as class- and
model-conditional statements), the mechanism analysis (ω→0 degeneracy), the
exp79 lemma and constants, the observable-agnostic normal form, the
amortized-estimator pairing, and the physics reading. Changes: the framing
(from "new bound" to "old, powerful machinery newly applied — plus a new
closed-form modulus and a physics dictionary"), the sharpness route (convexify
and inherit Donoho, rather than prove Newman-type inequalities from scratch —
keep the annihilator route as the mathematically interesting appendix), and
the abstract's priority sentence. Expected referee outcome after these
changes: the paper is *stronger*, because it stands on a thirty-year-old
foundation instead of beside it, and its genuinely new parts are exactly
delineated.

## 8. Limits of this simulation

This review was produced by the same class of model that produced the theorem
notes; blind spots may be correlated. The Donoho/Armstrong–Kolesár mapping
was verified against the published abstracts (fixed design, Gaussian errors,
linear functional, convex class — the match is structural, not superficial),
but the exact constants (the ≈1.25 affine-minimax factor) and the nonconvex-
hull gap need expert or primary-source confirmation. The human audit remains
gate one; its brief (AUDIT_BRIEF.md) has been updated to ask the expert to
check precisely this mapping.
