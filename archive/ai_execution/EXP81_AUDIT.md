# Exp 81 Audit — reviewer pass on EXP81_REPORT.md

**Auditor pass 2026-07-02, against `results_exp81_hierarchy/*.json` on branch
`exp81`. Verdict: gates accepted; the two headline quantitative findings are
NOT accepted pending the fix-pass below. Do not merge to main or cite any §2/§4
ratio until exp81f completes.**

## What the audit confirms

- G-B1–B3, G-B4, G-B5, G-A1 verified directly against the JSONs. The G-B5
  two-ceiling resolution (summary-level Fisher 0.0477 vs raw FIM 0.0273) is
  legitimate — it is the data-processing inequality doing its job, and it is
  itself a clean mini-result: the 7-point ladder discards ≈(0.0477/0.0273)² ≈
  3.1× of the raw-field Fisher information about α even with zero corrections.
- Anomaly 9's minimax-vs-average-risk distinction is correct and important.
- H-A1 at quadrature, the private⊇shared warm-start fix, and the honest
  ω_eff measurements are all sound.

## Blocker 1 (unflagged in the report): nesting violation in floor_L23 vs U

`level23_floors`, m=24: U=0.5 → 0.0422, U=1.0 → 0.0234, U=4.0 → 1.1016.
The class |u|≤0.5 is a subset of |u|≤1.0, so the floor must be nondecreasing
in U. 0.0234 < 0.0422 is mathematically impossible for exactly-solved
optimizations — same invariant class as the report's own anomaly 2, but in the
headline table and uncaught. Diagnosis: the small-floor cells sit at the
bisection grid scale (values are exact multiples of 1.2/64 ≈ 0.019, i.e.
0.0234375 and 0.0421875 differ by ONE bisection step), so the sub-0.05 cells
carry 40–90% relative error from grid granularity plus optimizer noise. The
same applies to the H-A3 K=5 cell (0.0234375 — the identical grid value),
which means the 11.8× gain denominator is a single coarse cell; the true gain
could be materially different in either direction.

## Blocker 2 (flagged as anomaly 7, but under-weighted): the L0-vs-L23
## comparison is not yet a hierarchy measurement

The 10–20× "richer observables help" ratio compares floors under *different
adversaries* (abstract log-ladder correction at Level 0; the parametric
(ν,ν₂,ω̃) family at Level 2/3, whose achievable correction amplitude
saturates below nominal U — the b_fit plateau). The ratio therefore confounds
"more data" with "weaker adversary" and is not citable as a hierarchy result.
The U=4 reversal is the visible symptom; the tight-U cells have the same
disease with the opposite sign.

## Required fix-pass (exp81f) before any §2/§4 number is citable

1. **Bisection resolution**: refine the Δα search to relative precision ≤5%
   per cell (adaptive bisection; do not inherit exp77's fixed grid at scales
   it was never used for).
2. **Monotonicity as a gate**: warm-start each (U, m) optimization from the
   converged solution of the next-smaller U (and each K from K−1); assert
   floor nondecreasing in U and nonincreasing in K as hard gates
   (G-B6, G-A4). Any violation = under-convergence = rerun with larger
   budget, never report.
3. **Aligned adversary classes**: recompute floor_L0 with the adversary
   restricted to the *pushforward of the physical family* — i.e., the same
   (ν, ν₂, ω̃) nuisance pair generating the summary means, so both levels
   face the identical physical adversary. With aligned classes the
   data-processing inequality floor_L23 ≤ floor_L0 becomes a validation gate
   (G-B7), and the L23/L0 ratio becomes the hierarchy measurement the plan
   asked for. Report both the aligned ratio and (separately, clearly
   labeled) the original class-conditional comparison.
4. Re-run the H-A3 curve with items 1–2 in place; report the corrected K=5
   gain with a convergence statement.

## Standing after the fix-pass (expected)

Solid regardless: the testbed validations, the K=1 anchor, H-A1 quadrature,
the qualitative H-A3 common-mode-rejection mechanism, the summary-lossiness
Fisher result, the ω_eff discipline, and the Task-5 scaling-law floor table
(class-conditional, as labeled). The hierarchy ratio and the quantitative
H-A3 gain are pending exp81f.
