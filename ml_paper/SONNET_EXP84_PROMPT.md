# Execution prompt: exp84 — pre-submission computations (paper-blocking)

Paste everything below the line into a fresh Sonnet session with repo access.
Four small tasks close the manuscript's remaining numerical gaps (see
MANUSCRIPT_OUTLINE.md "Pre-submission blockers" and EXP83_AUDIT.md "Required
fixes"). PROCESS RULE (new, binding): you are the executor. Report findings;
run gates; NEVER write an audit verdict, "ACCEPTED", or "arc closed" — that
is the reviewer's role, in a separate session. Your report ends at the gate
ledger and anomaly log.

---

You are implementing exp84 in `universality-discovery`, branch `exp81`.
Read first: `ml_paper/EXP83_AUDIT.md` (required fixes), `ml_paper/
SIMULATED_AUDIT.md` §8a (the A–K CI formulas and U-lower-bound idea),
`experiments/83_audit_response.py` (reuse its hull-modulus machinery via
importlib, unmodified), EXP76_HANDOFF.md sandbox gotchas (all apply; also:
verify committed blobs after committing; if git index corrupts, use
GIT_INDEX_FILE=/tmp/gidx). Output: `experiments/84_presubmission.py`,
`results_exp84_presubmission/*.json`, `ml_paper/EXP84_REPORT.md` (gate-ledger
format).

## Task 1 — Van Trees with a regular prior (G-84a)

Replace exp83's uniform prior with a smooth bump prior (cos² taper on the
same support) satisfying van Trees regularity (prior density → 0 at
boundary, finite prior information I_prior — compute it explicitly this
time, no flat-prior shortcut). Recompute the α-marginal bound per system,
m ∈ {6,24}. Gates: bound within ~20% of exp83's flat-prior values (the
correction should be small — if it is large, STOP and report); still ≤
fixed-nuisance CR bound; amortized RMSE still does not beat it.

## Task 2 — σ-resolved bound-vs-RMSE comparison (G-84b)

exp83 compared the amortized estimator's prior-averaged RMSE (0.106) to a
bound computed at the noise ceiling. Do it properly: bin exp76's synthetic
test set by σ (4–6 bins), compute per-bin RMSE and the per-bin van Trees
bound (Task 1 prior, σ fixed per bin), plot/tabulate ratio vs σ. Gate: no
bin has RMSE < bound (a violation = bug, stop and diagnose). Finding either
way: is the ~1.7× headroom uniform in σ or concentrated?

## Task 3 — Exact honest CIs from the modulus (G-84c) — the paper's §6 core

Implement the A–K fixed-length CI at the BD design: half-length
χ_α = cv_α( ω(δ)/(2σ_m ω'(δ)) − δ/(2σ_m) ) · σ_m ω'(δ), minimized over
δ > 0, where ω is the SINGLE-CLASS modulus over the convex hull (J=4 atoms,
exp83 Task 2 machinery; σ_m = σ/√m; ω' by finite differences on the
computed modulus curve; cv_α from the A–K Table I formula — the 1−α
quantile of |N(t,1)|). Compute for BD measured σ, m ∈ {6,24}, U ∈ {0.5,1,4},
α = 0.05. Also the one-sided minimax excess length ω(σ_m(z_{1−α}+z_β)) at
β = 0.8. Gates: (i) ω computed on a fine enough δ-grid that χ's minimizer
is interior and stable to grid refinement (~2%); (ii) CI half-length ≥
floor/2 sanity relation at matching class (if violated, conventions are
mixed up — stop); (iii) five-tuple on every number. Report the comparison:
honest CI half-length vs the exp76 amortized interval (±0.03 stat) vs the
floor — three objects, clearly distinguished in one table.

## Task 4 — Data-driven lower bound on U (G-84d)

A–K Supplemental E.3 analogue: over exp76b's real per-seed ladders
(wsat_perseed.csv), for each system fit the pure power law (no correction)
and compute the minimum correction amplitude u_min needed to explain the
residual beyond noise (profile: u_min = max over subwindows of
|residual|/(expected correction shape), with a noise-floor deduction;
document the exact estimator you implement — simple and explicit beats
clever). Report u_min per system with bootstrap CIs, and check consistency:
u_min ≤ the declared U values used throughout (gate) — and specifically
compare BD's u_min to the u ~ 0.4 the theory note quotes from the
literature. Label: DIAGNOSTIC (an estimator of a lower bound, not a
theorem).

## Report

`ml_paper/EXP84_REPORT.md`: gate ledger, one table per task, anomalies,
"what we did not do", five-tuples everywhere, no claims language, NO
verdicts. Commit on `exp81`; verify committed blobs (git show HEAD:<file>).
