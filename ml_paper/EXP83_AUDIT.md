# Exp 83 Audit — reviewer sign-off

**Auditor pass 2026-07-03, against `results_exp83_audit_response/audit_response.json`.
Verdict: ACCEPTED — all five gates verified against the JSON; the two
disclosed estimator bugs (hull warm-start, KDE self-inclusion) were caught by
the correct mechanisms (nesting invariant; Stam-inequality sanity check) and
their fixes are sound. The exp81f fix-pass remains the only outstanding
experiment before manuscript work.**

## The derivation the report stops short of: a sharpness bracket

Combining Task 2 with Appendix G: the class floor lower-bounds the class
minimax risk; any estimator's class worst-case is ≤ its hull worst-case; and
over the (convex) hull, Donoho affine-minimax theory bounds achievable risk
within a small constant (≈1.25, conventions to be fixed at the primary
source) of the hull modulus, which the hull floor computes. Chain:

    class_floor ≤ minimax_class ≤ ~1.25 × hull_floor

With the measured hull/class ratios (1.17–1.62), **the computed floors
bracket the true minimax risk within a factor ≈1.5–2.0 for this design** —
worst at (U=1, m=6), best ≈1.5 at U=4. This is the sharpness statement the
paper can make *today*, with no new inequality proven: the impossibility
numbers are not just valid, they are order-unity tight. The exp79
analytic-lower-bound program is now purely enrichment. (Caveat for the
paper: Donoho's constants are stated for specific risk conventions —
estimation vs. testing separation; the ≈1.25 and the exact chain need the
human auditor's or the primary source's sign-off before the bracket is
stated with numbers rather than "order unity.")

**Correction (2026-07-03, after primary-source verification — see
SIMULATED_AUDIT.md §8a):** the ≈1.25 constant in the chain above is an
MSE-estimation constant and was misapplied to the CI context; for confidence
intervals the Armstrong–Kolesár machinery is exact with no unknown constant.
The bracket above remains valid but is superseded as a goal: the correct
deliverable is exact minimax CIs computed from the (hull) modulus already
implemented in this experiment's Task 2 — see the exp84 candidate in
SIMULATED_AUDIT.md §8a. Centrosymmetry caveat also applies: constructive
results transfer (convexity only); A–K's non-adaptation efficiency numbers
do not, since our class is asymmetric.

## Paper-facing numbers this experiment produced

- **Nuisance information destruction: 59×** (info ratio 0.017, design-only,
  σ-independent). "Unknown corrections destroy 98.3% of the Fisher
  information about the exponent" is the single best one-sentence
  explanation of why FSS is hard the project now owns.
- Gaussianity: κ CIs straddle 1 in all four systems; with the estimator's
  measured residual bias (κ≈0.95–0.97 on true Gaussian data), the
  bias-aware reading is "no detectable excess information; the Gaussian
  idealization stands within ~±5% in Δα (√κ)." DIAGNOSTIC grade, as labeled.
- Floor robustness to plug-in σ: ±3% at m=24. Negligible next to
  class-conditionality; report as an interval and move on.
- Amortized estimator vs Bayes bound: 0.106 vs 0.0634 at the noise ceiling —
  within ~1.7× of the (most generous) van Trees bound. Direction safe a
  fortiori: bounds at smaller σ are smaller, so no refinement can flip it.

## Required fixes before these numbers enter the manuscript

1. **Van Trees regularity**: the uniform prior violates van Trees'
   absolute-continuity/boundary conditions (anomaly 5 is right to flag it).
   Fix is standard and cheap: recompute with a smooth bump (e.g. cosine²)
   prior on the same support; numbers will move by O(boundary mass), not
   qualitatively.
2. **σ-resolved VT comparison**: "near the information limit" eventually
   needs the bound and the RMSE conditioned on the same σ-slices, not a
   ceiling-vs-average comparison. One plot.
3. **J=1 anchor noise defines the table's error bar**: the hull/class ratio
   carries ±2% optimizer/bisection noise (visible as the 0.98 anchor cell);
   state it with the table.
4. The race-condition 