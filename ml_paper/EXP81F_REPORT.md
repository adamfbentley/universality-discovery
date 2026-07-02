# Exp 81f Report — Fix-Pass Responding to EXP81_AUDIT.md

**Status: complete, 2026-07-03.**
**Audit: `ml_paper/EXP81_AUDIT.md`. Original results: `ml_paper/EXP81_REPORT.md`,
`results_exp81_hierarchy/*.json`.**
**Findings only — no claims language. Nothing here enters `CLAIMS_REGISTER.md`
or `MANUSCRIPT_OUTLINE.md`.**

All numbers below exist in `results_exp81f_fixpass/fixpass.json`. Code:
`experiments/81f_hierarchy_fixpass.py` (reuses `81_fractional_ew_testbed.py`
and `81c_multivariate_floor.py` via `importlib`, unmodified — this file only
adds new, independently-gated computations on top).

**Verdict: both audit blockers are resolved. The §2/§4 hierarchy ratio and
the H-A3 multivariate gain are now citable**, with numbers materially
different from (more conservative than) the originally-reported ones.

---

## 1. Gate ledger

| Gate | What it checks | Result | Number | Proof path |
|---|---|---|---|---|
| G-B6 (floor_L23) | floor nondecreasing in U | **PASS** | m=24: 0.0196 (U=0.5) = 0.0196 (U=1.0) ≤ 1.1062 (U=4.0) | `fixpass.json:hierarchy_fixed` |
| G-B6 (floor_L0_aligned) | floor nondecreasing in U | **PASS** | m=24: 0.1523 ≤ 0.1922 ≤ 1.1438 | `fixpass.json:hierarchy_fixed` |
| G-B7 | data-processing: floor_L23 ≤ floor_L0_aligned (aligned adversary) | **PASS**, every (U,m) cell | tightest margin: U=4.0,m=24: 1.1062 ≤ 1.1438 | `fixpass.json:hierarchy_fixed` |
| G-A4 (private) | floor nonincreasing in K | **PASS** (exact decomposition) | 0.2766 → 0.1383, flat K=2..5 | `fixpass.json:multivariate_fixed` |
| G-A4 (shared) | floor nonincreasing in K | **PASS** | 0.2766 → 0.1289, flat K=2..5 | `fixpass.json:multivariate_fixed` |
| G-A4 (shared_correlated) | floor nonincreasing in K | **PASS** | 0.2766 → 0.1102 → 0.1055 → 0.1055 → 0.1008 | `fixpass.json:multivariate_fixed` |

Both fix-passes converged (satisfied every gate) at the **first** escalation
level tried (`n_starts=8`); no retry was needed.

---

## 2. Item 1+2 fix: adaptive relative tolerance + U-monotone chaining

The original code used a fixed absolute bisection tolerance (`tol=1e-2`),
which the audit correctly identified as 25–90% relative error on the small
`floor_L23` cells (e.g. 0.0234 vs 0.0422 differing by exactly one bisection
step). Replaced with an adaptive stopping rule (`(hi-lo) <= max(0.05*lo,
1e-3)`), giving uniform ~5% relative precision from the smallest floor
(~0.02) to the largest (~1.1).

U-monotonicity is enforced by **exact box-nesting**: since the adversary's
per-config amplitude bound is `(-0.75, U)`, a solution valid at smaller U is
*automatically* feasible (unchanged, no reparametrization) at any larger U.
Each U in `{0.5, 1.0, 4.0}` (sorted ascending) is solved by warm-starting
from the immediately-smaller U's converged solution, recursively. This alone
was not sufficient reliability-wise for every case (see anomaly 2 below);
the final code also re-evaluates every warm-start point's own raw objective
value directly and takes the minimum against the optimizer's return, guarding
against the local optimizer occasionally returning a point worse than where
it started (observed directly — see anomalies).

## 3. Item 3 fix: aligned adversary (the citable hierarchy ratio)

**floor_L0_aligned** replaces the abstract exp77 log-ladder adversary with
one using the *exact same* physical `(z, D, ν, ν₂, ω̃)` nuisance pair that
generates the Level-2/3 spectral ensembles, observed only through the
7-point `log(W_sat)` summary (not the full spectrum). This is the
pushforward of the Level-2/3 family through the summary map, making
`floor_L23 ≤ floor_L0_aligned` a genuine data-processing consequence rather
than an assumption — and it holds at every cell (G-B7), including U=4.0
where the *original*, non-aligned comparison had the impossible ordering
`floor_L23 > floor_L0` (audit blocker 1).

Five-tuple: (Level 0-aligned / Level 2-3, class N=1/U∈{0.5,1,4}/ω_min=0.3,
design={32,...,256}, σ=0.6005 measured at α≈0.5 for this family, m∈{6,24}).

| U | m | floor_L23 | floor_L0_aligned | ratio (L0_aligned / L23) |
|---|---|---|---|---|
| 0.5 | 6 | 0.0381 | 0.2297 | 6.03× |
| 1.0 | 6 | 0.0381 | 0.2953 | 7.75× |
| 4.0 | 6 | 1.1062 | 1.1812 | 1.07× |
| 0.5 | 24 | 0.0196 | 0.1523 | 7.76× |
| 1.0 | 24 | 0.0196 | 0.1922 | 9.79× |
| 4.0 | 24 | 1.1062 | 1.1438 | 1.03× |

**Corrected hierarchy finding.** Richer observables (full spectrum vs.
7-point summary) buy a real, substantial resolution improvement — roughly
6–10× — when the correction-amplitude bound is tight (U=0.5, 1.0). That
advantage nearly **vanishes** (~1.03–1.07×) when the bound is loose (U=4.0):
with almost no constraint on the adversary's amplitude, neither observable
resolves much better than the other. This replaces the original report's
10–20× ratio (computed under two non-nested adversary classes, per audit
blocker 2) with a properly-aligned, gate-validated measurement showing the
advantage is *real but U-dependent*, not a fixed multiplier.

`floor_L23` is identical between U=0.5 and U=1.0 (both m): this reflects the
same `b`-saturation mechanism documented in the original report's ω_eff
table (`floors_hierarchy.json:omega_eff_measurements`) — the spectral
adversary's achievable amplitude at these ω̃ values saturates below the
nominal U=1.0 bound, so U=0.5 and U=1.0 are not actually distinguishable
classes for this observable at this design.

## 4. Item 4 fix: H-A3 corrected K-curve

Five-tuple: (Level 1, class N=1/U=1.0/ω_min=0.3, design={32,...,256},
σ=0.15 synthetic, m=24).

| K | private (H-A1) | shared (H-A2) | shared + correlated noise ρ=0.8 (H-A3) |
|---|---|---|---|
| 1 | 0.2766 | 0.2766 | 0.2766 |
| 2 | 0.1383 | 0.1289 | 0.1102 |
| 3 | 0.1383 | 0.1289 | 0.1055 |
| 4 | 0.1383 | 0.1289 | 0.1055 |
| 5 | 0.1383 | 0.1289 | 0.1008 |
| **K1→K5 gain** | **2.00×** | **2.15×** | **2.74×** |

Quadrature prediction at K=5 (`shared_K1/√5`): 2.236×.

**Corrected finding.** All three gains now sit in a modest, mutually
comparable range (2.0×–2.74×), all near or below the naive quadrature
prediction (2.236×). Correlated noise (H-A3) gives the *largest* gain of
the three, consistently with the qualitative common-mode-rejection
mechanism from the original report, but the effect is **moderate, not
dramatic** — a corrected 2.74× at K=5, not the originally-reported 11.8×.
The original number was traced (audit blocker 1, "the same invariant class"
note) to an unconverged, non-monotone K-curve whose K=5 cell sat at
bisection-grid resolution; that specific number is superseded by the one
above.

## 5. Anomalies and bugs found during the fix-pass

1. **First warm-start attempt (single extra-start, no result verification)
   was insufficient for the "shared" multivariate mode**, caught by a
   direct diagnostic: chaining K=3 from K=2's own solution should, since
   channel 3 is a pure-nuisance channel (b=0, contributes exactly 0 under
   diagonal noise), give `D²(K=3) == D²(K=2)` at every Δα — but K=3's
   reported floor (0.1336) came out *larger* than K=2's (0.1055), and a
   direct check at the same Δα found `gap(K=3) = 0.0437 < gap(K=2) = 0.0974`
   at the identical point — an impossible ordering (K=3 nests K=2 exactly
   here), proving **K=2's own solve**, not the K=3 chain, was the weak link.
   Root cause: `shared` mode's joint (ω_A, ω_B) optimization over a
   `(3K+2)`-dimensional space is a genuinely hard, multi-modal problem for
   blind multistart, even with only 8 starts.
2. **Fix: exploited exact separability.** For *fixed* (ω_A, ω_B), the
   per-channel minimization is independent across channels whenever Σ is
   diagonal (true for both `shared` and `private`, only NOT true for
   `shared_correlated`). This turns the `shared` problem into a reliable
   outer 2D grid search (9×9) over the only genuinely-coupled parameters,
   with cheap, robust 3-parameter per-channel inner solves — replacing the
   fragile high-dimensional joint multistart entirely for this case, and
   making K-monotonicity for `shared` **structurally guaranteed** (for any
   fixed ω pair, the K-channel sum is the (K−1)-channel sum plus one more
   nonnegative term, evaluated over the *same* grid for both K and K−1, so
   `min(K) ≥ min(K−1)` always — no chaining needed for this case at all).
3. **`private` mode decomposes exactly** under diagonal noise with fully
   independent per-channel rates: the K-channel problem separates into K
   independent single-channel problems (each identical in structure to
   exp77's own K=1 anchor case), with pure-nuisance channels contributing
   *exactly* zero. Used as an exact (not merely gated) computation for
   H-A1, replacing generic joint multistart.
4. **`shared_correlated` (H-A3) still needs genuine joint optimization**
   (Σ is not diagonal, so the separability trick does not apply), but is
   now warm-started from *both* the K−1 chained solution and the reliable
   uncorrelated (`shared`, ρ=0) outer-2D solution at the same K — a much
   better-informed starting point than blind random restarts, since the
   underlying signal structure is identical between the two cases and only
   the noise correlation differs.
5. **Optimizer-returns-worse-than-its-own-start guard.** Independently of
   the above, `L-BFGS-B` was observed capable of returning a point *worse*
   than a warm start that is analytically a stationary point (a
   zero-amplitude new channel has exactly zero gradient contribution to the
   objective, verified by direct differentiation) — plausibly because a
   `maxiter`/`maxfun` cutoff caught it mid-overshoot during an imperfect
   line search. Fixed defensively: every warm-started call also directly
   re-evaluates the raw objective at each of its own warm-start points and
   takes the minimum against whatever the optimizer returns. Applied to
   both the Level-0/2-3 U-chaining (Part A) and the multivariate K-chaining
   (Part B).
6. **Directory-creation ordering bug (tooling, not math).** An early attempt
   to launch the hierarchy computation in the background failed silently at
   the shell level (`> .../results_exp81f_fixpass/....log`) because the
   output directory did not yet exist — shell redirection requires the
   parent directory to exist *before* the redirected process starts, but
   the script's own `os.makedirs(..., exist_ok=True)` only runs *after*
   Python starts, too late to help. Fixed by creating the directory
   explicitly before launching.
7. **Reduced random-restart budget for warm-started ("chained") calls was
   tried and reverted.** Reasoning that a good warm start needs little
   further exploration, an early version used `n_starts_chained=3` for
   non-base U levels in the Level 2/3 chain. Verified directly to be
   badly wrong: at Δα=0.6, U=4.0, `n_starts_chained=3` gave gap=21.9 vs.
   `n_starts_chained=8` giving gap=1.1e-6 — a >10⁷ ratio. The warm-started
   point, while feasible, is often far from the larger-U-specific optimum
   (which needs genuine exploration of the *expanded* amplitude range a
   smaller-U warm start never visits); the local optimizer alone cannot
   reliably bridge that gap. Reverted to using the full `n_starts` budget
   uniformly for every U level; only `maxiter`/`maxfun` per individual
   optimization were reduced (independently verified as NOT responsible
   for the degradation) to control wall-clock.

## 6. What we did NOT do

- **Did not re-verify the original (non-aligned) `floor_L0` or the original
  H-A3 curve's specific numbers beyond what the audit already stated** —
  they are superseded here, not re-derived; `EXP81_REPORT.md` and
  `results_exp81_hierarchy/*.json` remain as the historical record of what
  was originally computed and why it was flagged.
- **Did not extend the aligned-adversary treatment to Part A's Task 3
  (trained estimators) or Task 5 (scaling-law floor)** in the original
  report — those were not flagged by the audit and are unaffected by this
  fix-pass.
- **Did not attempt an exact closed-form monotonicity proof for
  `shared_correlated`** — G-A4 for this case rests on the empirical
  verification (K-chained warm start + multi-start, converged at the first
  escalation level tried), not a structural guarantee like `private` and
  `shared` now have.
- **Did not sweep the escalation ladder beyond the first level** (`n_starts
  = 8`) since every gate passed there; the `(8, 16, 24)`/`(10, 20, 32)`
  fallback levels coded into `task_hierarchy`/`task_multivariate` were
  never exercised this run.
- **Did not revisit G-A1/G-A2's own numbers** from `81c_multivariate_floor.py`
  (already gated and passing); this fix-pass only adds the K-chained,
  monotonicity-guaranteed re-computation on top for the audit's specific
  ask (item 4).

---

Committed on branch `exp81`. `CLAIMS_REGISTER.md` and `MANUSCRIPT_OUTLINE.md`
were not touched.
