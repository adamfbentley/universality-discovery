# Exp 81f Audit — reviewer sign-off; exp81 arc CLOSED

**Auditor pass 2026-07-03, against `results_exp81f_fixpass/fixpass.json`.
Verdict: ACCEPTED. Both audit blockers resolved; gates G-B6/G-B7/G-A4 verified
against the JSON; the fix-pass's methodology upgrades (exact separability for
diagonal-Σ modes, structural monotonicity, optimizer-worse-than-start guard)
are sound and the anomaly log — including the honest reversal of the
reduced-restart "optimization" after a 10⁷ error — is exemplary. The
hierarchy ratio and multivariate gains are citable subject to the two
audit re-statements below.**

## Re-statement 1: the U=4 row is the no-identifiability regime, not a ratio

The U=4 floors (L23 1.106; L0_aligned 1.144–1.181) sit within 1.6–5% of the
1.2 search ceiling AND exceed the α-prior's own width — physically, "with
essentially unconstrained correction amplitudes, α is unidentifiable from
either observable at this design/noise." The DPI margin there (3.4%) is
smaller than the 5% bisection tolerance, so the 1.03–1.07× "ratios" are
≈1 within numerical precision and must not be cited as measurements. The
paper's statement for that row: "the advantage vanishes (≈1× within
precision); both floors exceed the prior range."

## Re-statement 2: the multivariate gains are mostly max|θ| rescaling

Private K=2 equals K=1/2 EXACTLY (0.13828125 = 0.2765625/2). Mechanism: the
confusion gap per channel is g(|θ_k|·Δα) with g extremely steep near the
floor, so the floor of any channel set is ≈ floor_single/max_k|θ_k| — here
the θ=−2 spectral-slope channel halves the floor by itself, and further
channels move nothing beyond tolerance. Consequently:
- The "quadrature prediction √K" is the wrong null for floors under steep
  gap growth (correct null: K_eff^{1/(2(N+1))}, much flatter); do not cite
  "at quadrature" as confirmed theory.
- The decomposition of the measured gains: |θ|-dominance baseline 2.00×;
  shared-spectrum triangulation adds only ~7% (2.15/2.00); correlated-noise
  common-mode rejection adds ~37% (2.74/2.00). The honest H-A3 statement is
  "a real but modest ~1.4× on top of the best single channel," not 2.74×
  as a standalone multivariate effect (and not the superseded 11.8×).

## The citable hierarchy result (exp81 arc, final form)

Under the ALIGNED physical adversary (G-B7 data-processing gate passing at
every cell), with five-tuple as recorded in fixpass.json:

    Tight amplitude bounds (U=0.5–1.0): full spectrum resolves α 6–10× finer
    than the 7-point W_sat summary (m=24: 0.020 vs 0.15–0.19).
    Loose bounds (U=4): no meaningful advantage; α effectively unidentifiable
    from either observable.

Synthesis for the paper (and the thesis's central sentence): **the value of
richer observables is conditional on prior knowledge of the correction
structure — data richness and correction knowledge are complements, not
substitutes.** Combined with exp77/83 (statistics cannot substitute for
window or knowledge) the full exchange-rate picture is now: seeds ≪
observables ≤ window ≈ correction knowledge, with all four quantified on
one testbed.

## Status

All exp81-arc experiments (81, 81f, 82, 83) complete and audited. Remaining
before manuscript: none (experimental); the human audit reply is the only
open external gate. Next action per the standing queue: rewrite
MANUSCRIPT_OUTLINE.md and draft the Paper 2 introduction under the
post-Donoho framing, folding in: exact-CI deliverable (SIMULATED_AUDIT 8a),
59× nuisance destruction (exp83), the hierarchy result above, and the
aligned-adversary methodology as a worked example of class-alignment
discipline.

## Countersignature (second auditor, 2026-07-03)

Independent verification performed against fixpass.json: both re-statements
confirmed by direct recomputation (private K=2 = K=1/2 exactly; increments
+7.3% triangulation, +37% correlated noise; U=4 DPI margin 3.4% < 5%
tolerance; U=4 floors exceed the alpha-prior width). Both gates G-B6/B7 and
all three G-A4 curves re-checked from the JSON directly. Verdict ACCEPTED is
countersigned; the exp81 arc is closed.

Two additions to the record:

1. **Provenance of the wrong quadrature null.** Re-statement 2 corrects an
   error that originated in EXP81_PLAN.md's H-A1 hypothesis (written by the
   planning/audit session, not the executor): under gap growth
   D ~ (Δα)^{N+1}, K equal channels give floor gains ~K^{1/(2(N+1))}
   (K=5, N=1: ~1.50×), not √K (2.24×). The plan's null was wrong; exp81's
   original "at quadrature" finding was numerically right for the wrong
   reason (the measured 2.0× is max-|θ| rescaling, not K-aggregation).
   Auditors are in the error ledger too; this is the second entry
   (see archive/ai_execution/EXP83_AUDIT.md, commit-verification failure).

2. **Unexplained wrinkle, non-blocking, for the paper to address:** between
   U=0.5 and U=1.0, floor_L23 is identical (b-saturation collapses the two
   classes for the spectral observable) but floor_L0_aligned is NOT
   (0.1523 vs 0.1922 at m=24). Same physical family, same nominal
   saturation — so the summary-pushforward must bind the amplitude
   constraint differently (plausibly via the asymmetric b ∈ (−1, ∞) range
   acting differently on the two KLs). Monotonicity holds at both levels, so
   no gate is violated, but the paper should either explain the mechanism
   or collapse U=0.5/1.0 into a single declared class for this family.
