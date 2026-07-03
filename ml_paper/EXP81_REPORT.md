# Exp 81 Report — The Observable-Information Hierarchy

**Status: complete (Parts B, A, and the C stretch), 2026-07-02.**
**Plan: `EXP81_PLAN.md`. Theory: `THEORY_minimax_floor.md` Appendix F.**
**Findings only — no claims language. Nothing here enters `CLAIMS_REGISTER.md`
or `MANUSCRIPT_OUTLINE.md`.**

All numbers below exist in `results_exp81_hierarchy/*.json`. Code:
`experiments/81_fractional_ew_testbed.py` (Tasks 1–2),
`experiments/81b_estimators.py` (Task 3),
`experiments/81c_multivariate_floor.py` (Task 4),
`experiments/81e_scaling_law_floor.py` (Task 5, stretch).

---

## 1. Gate ledger

| Gate | What it checks | Result | Number | Proof file |
|---|---|---|---|---|
| G-B1 | sampled ensemble spectrum matches L·S(k) | **PASS** | max per-band rel. err 4.4% vs. MC SEM 1.6% | `gates_task1.json:gate_b1`, `figs/gate_b1_spectrum_overlay.png` |
| G-B2 | ν₂=0 ladder recovers (z−1)/2 | **PASS** | asymptotic slope matches target to 2e-4; MC (unbiased, log-mean) estimator matches exact design-ladder slope within 4σ at all three z | `gates_task1.json:gate_b2` |
| G-B3 | KL formula vs. empirical log-likelihood-ratio | **PASS** | KL analytic 2.3695 vs. empirical 2.3622±0.0246 (rel. err 0.31%); FIM finite-diff rel. err 1.7e-10 | `gates_task1.json:gate_b3` |
| G-B4 | Level-0 floor sanity vs. `results_exp77_minimax_floor/floor.json` | **PASS** | monotonic increasing in U (both here and in exp77's reference); same order of magnitude | `floors_hierarchy.json:gate_b4` |
| G-B5 | MLP/CNN recover α near-FIM at ν₂=0 before hard regimes | **PASS** | MLP ratio to its own **summary-level** FIM ceiling = 1.50×; CNN ratio to **raw-field** FIM = 1.15× (both < 2×) | `estimators.json:gate_b5` |
| G-A1 | K=1 multivariate reduction reproduces exp77 exact floor | **PASS** | 0.2672 vs. anchor 0.27 (BD, m=24); diff 0.0028 | `multivariate_floor.json:gate_a1` |
| G-A2 | floor-vs-K curves (shared / private / shared+correlated) | **PASS** (produced; see §4) | K1→K5: shared 0.277→0.117 (2.36×), private 0.277→0.136 (2.03×), shared+ρ=0.8 0.277→0.023 (11.8×) | `multivariate_floor.json:gate_a2` |
| G-A3 | measured-Σ pilot | **NOT RUN** (optional) | — | see §6 |

---

## 2. Floors table (Level 0 vs. Level 2/3 vs. FIM)

Five-tuple for every number below: **(observable level, class N=1 / U /
ω_min=0.3, design={32,48,64,96,128,192,256}, noise=this-family's measured
σ(log W_sat)=0.6005 at α≈0.5, m)**. σ here is far larger than the real
EW/KPZ/BD/Eden σ (0.02–0.15) used in exp77 — see §5 anomaly 4 for the
mechanism. `da_hi` (bisection search ceiling) = 1.2 throughout; no cell is
saturated at that ceiling (all reported values are genuine bisection
solutions, not search-boundary artifacts — see §5 anomaly 6 for the earlier
0.6-ceiling version of this table that *was* saturated).

| ω̃ | U | m | floor_L0 | floor_L23 | Δα_FIM = 1/√(m·I(α)) |
|---|---|---|---|---|---|
| 1.0 | 0.5 | 6  | 0.6176 | 0.0516 | 0.000169 |
| 1.0 | 0.5 | 24 | 0.4184 | 0.0422 | 0.0000844 |
| 1.0 | 1.0 | 6  | 0.6926 | 0.0516 | 0.000169 |
| 1.0 | 1.0 | 24 | 0.4723 | 0.0234 | 0.0000844 |
| 1.0 | 4.0 | 6  | 0.9363 | 1.1109 | 0.000169 |
| 1.0 | 4.0 | 24 | 0.6809 | 1.1016 | 0.0000844 |
| 0.3 (stress) | 0.5 | 6  | 0.6176 | 0.0516 | 0.00576 |
| 0.3 (stress) | 0.5 | 24 | 0.4184 | 0.0422 | 0.00288 |
| 0.3 (stress) | 1.0 | 6  | 0.6926 | 0.0516 | 0.00226 |
| 0.3 (stress) | 1.0 | 24 | 0.4723 | 0.0234 | 0.00113 |
| 0.3 (stress) | 4.0 | 6  | 0.9363 | 1.1109 | 0.000135 |
| 0.3 (stress) | 4.0 | 24 | 0.6809 | 1.1016 | 0.0000677 |

`floor_L0`/`floor_L23` do not depend on ω̃ (ω_A, ω_B are free adversary
nuisances in [0.3, 2.5] at both levels); ω̃ enters only the FIM column
(evaluated at fixed, non-adversarial nuisances — see §3) and the
correction-exponent table below. The "0.3 (stress)" row is the near-marginal
case (plan's ω̃=2−z at z=1.7, i.e. the ω_min edge — the literal z=2.0
formula degenerates, see §5 anomaly 2).

**Reading:** at tight amplitude bounds (U=0.5, 1.0), the raw/spectral
observable resolves α roughly 10–20× finer than the single-summary ladder
(floor_L23 ≪ floor_L0) — the "richer observables help substantially" branch
of the plan's hypothesis. At the loosest bound (U=4.0), this reverses:
floor_L23 exceeds floor_L0. This is not a data-processing-inequality
violation; see §5 anomaly 5 for why the two levels use non-nested adversary
classes and are not guaranteed ordered.

**Correction-exponent measurement** (mandatory per plan before declaring any
class), at α=0.5, fit on a clean (noise-free, exact) 30-point ladder to
L=2¹⁶:

| ω̃ declared | u declared | ω_eff measured | b_fit |
|---|---|---|---|
| 0.3 | 0.5 | 0.330 | 1.40 |
| 0.3 | 1.0 | 0.437 | 3.94 |
| 0.3 | 4.0 | 0.480 | 6.90 |
| 0.5 | 0.5 | 0.539 | 2.85 |
| 0.5 | 1.0/4.0 | 0.595 | 5.62 |
| 1.0 | any | 0.672 | 2.23 |
| 1.5 | any | 0.631 | 0.76 |
| 2.0 | any | 0.670 | 0.30 |

Consistent with the plan's caveat (ω_eff saturates below the nominal ω̃ for
ω̃ > 2α): measured ω_eff clusters around 0.63–0.67 for ω̃ ≥ 1.0, well short
of the naive-saturation prediction 2α=1.0 — a real, not fully explained,
discrepancy (see §5 anomaly 3). At ω̃=0.3 and ω̃=0.5 (< 2α), ω_eff tracks
ω̃ reasonably (0.33–0.60), as expected for the IR-dominated regime. **b_fit
plateaus below the nominal u** for u≥1 at ω̃≥1.0 (e.g. ω̃=1.0: b_fit=2.23
regardless of whether u_declared is 0.5, 1.0, or 4.0) — the nu2↔u inversion
saturates (§3), meaning the Level-2/3 adversary cannot actually reach the
nominal U for these ω̃ even though it is nominally "declared."

## 3. Level-2/3 construction notes (for interpreting the tables above)

- **KL/FIM formulas**: per-complex-mode convention (mode variance = L·S(k),
  verified in G-B1/G-B3) makes the plan's literal KL formula
  Σ_k[S₁/S₂−1+ln(S₂/S₁)] and FIM formula exact with no correction factor
  (the shared L-rescaling cancels in the KL ratio and drops from the FIM's
  log-derivative). See the module docstring in `81_fractional_ew_testbed.py`.
- **FIM convention**: the plan's closed form
  ∂lnS/∂α = −2ln|k|·ν|k|^z/(ν|k|^z+ν₂|k|^{z+ω̃}) corresponds to holding the
  *absolute* correction exponent z+ω̃ fixed as α (i.e. z) varies, not ω̃
  itself — verified by finite difference (rel. err 1.7e-10, G-B3). The FIM
  table uses a representative nu2 (the value that saturates the declared U
  bound at the design center, z=2.0) as the "fixed nuisance" point, i.e. "if
  you already knew the correction sat at the class boundary, with perfect
  knowledge (no adversarial marginalization), how precise could you be" —
  it is a best-case ceiling, not a class-averaged number.
- **u→ν₂ inversion**: "u := b(L_min)" (the plan's sampling-prior
  correction-amplitude convention) requires inverting the nonlinear relation
  between ν₂ and the relative correction b(L_min) := W²_full(L_min)/
  W²_leadonly(L_min) − 1. This is done by bisection (`nu2_from_u`), exact
  (not linearized). b is bounded in (−1, +∞), asymmetric — requesting
  u ≤ −1 saturates at b=−1 (the ν₂→+∞ limit); for large ω̃, b(L_min) also
  saturates well below +U even for ν₂→−∞ (finite pole location), which is
  the mechanism behind the b_fit plateau noted above.

## 4. Multivariate (Level-1) floors — H-A1/H-A2/H-A3

Design: K=1..5 channels, θ_b = (1, −2, 0, 0, 1) (width, spectral-band-slope,
two pure-nuisance, repeat-width), σ=0.15, U=1.0, m=24, 7-point design. Per
plan, the per-channel correction uses exp77's own log-form
g(x;u,w)=log(1+u·e^{−w(x−x₁)}) (not the Appendix F1 schematic's bare linear
exponential — required for the K=1 anchor to literally coincide with
exp77's confusion_gap; see §5 anomaly 1).

| K | shared (H-A2 case) | private (H-A1 case) | shared + correlated noise ρ=0.8 |
|---|---|---|---|
| 1 | 0.2766 | 0.2766 | 0.2766 |
| 2 | 0.1359 | 0.1359 | 0.1078 |
| 3 | 0.1359 | 0.1359 | 0.0891 |
| 4 | 0.1172 | 0.1359 | 0.0797 |
| 5 | 0.1172 | 0.1359 | 0.0234 |

Quadrature prediction at K=5 (shared_K1/√5): 0.1237.

**H-A1 (private/independent corrections):** gain K1→K5 = 2.03×, essentially
at the quadrature prediction (2.24×) — consistent with the hypothesis that
independent per-channel corrections give at most quadrature-level gains. No
improvement is found beyond K=2 (channels 3,4 are pure-nuisance and
contribute nothing under diagonal noise by construction — a Δc_k=0,
u_k=0 solution exactly zeroes their contribution regardless of what the
other channels do; channel 5 repeats channel 1's exponent map but the
private optimizer does not find any gain from it either, within the
optimization budget used — see §5 anomaly 7).

**H-A2 (shared correction spectrum):** gain K1→K5 = 2.36×, modestly above
quadrature but not dramatically superlinear as the "triangulation" framing
anticipated. The private-parametrization floor is, correctly, always ≥ the
shared floor at every K (private nests shared) — this is a hard
mathematical requirement, not an empirical finding, and was violated before
a warm-start fix (§5 anomaly 1).

**H-A3 (correlated noise):** the largest effect by far — 11.8× gain at K=5
under ρ=0.8, dwarfing both H-A1 and H-A2. This is the *opposite* direction
from the plan's framing ("correlated noise can cancel most of the nominal
gain"): here it amplifies the gain enormously. Mechanism (not a bug):
compound-symmetric Σ has a large eigenvalue along the common-mode
(all-channels-equal) direction and a small eigenvalue (σ²(1−ρ)) along
contrast directions; the channel exponent maps here are NOT all
same-signed (θ_b = 1, −2, 0, 0, 1), so a genuine α-tilt produces a
contrast-like pattern across channels, and Σ⁻¹ up-weights exactly that
pattern. Whether real physical channels have same-signed or
opposite-signed responses (and how correlated their noise really is) is an
empirical question outside this synthetic gate — flagged, not resolved.

**Net Part-A finding:** whether richer summaries help is decided by
correction-sharing AND noise correlation structure jointly, not K alone,
matching the plan's framing — but the *dominant* lever measured here is
noise correlation (H-A3), not correction-sharing (H-A2), which is a
different emphasis than the plan's narrative suggested.

## 5. Anomalies, bugs found, and deviations from plan

1. **Multivariate K=1 anchor required exp77's log-form correction, not the
   Appendix F1 schematic's linear exponential.** Using
   `u·e^{−ωx}` gave K=1 floor 0.183 vs. anchor 0.27 (G-A1 failed, diff
   0.087). Switching to exp77's `log(1+u·e^{−ω(x−x1)})` (matching bounds
   too: U_BOUNDS=(−0.75,U), not symmetric (−U,U)) fixed this exactly
   (0.2672 vs. 0.27). Documented as a deliberate deviation from F1's literal
   formula, required by G-A1's own instruction to reproduce exp77 exactly.
2. **Private-parametrization optimizer under-convergence, caught by a
   mathematical invariant.** Private nests shared (replicating a shared ω
   across channels is a valid private point), so D²_private ≤ D²_shared must
   hold, hence floor_private ≥ floor_shared always. Initial runs (24
   independent random starts per case) violated this (private floor
   *smaller* than shared at K=3,5) because the higher-dimensional private
   search (up to 5K params) under-converged relative to shared's easier
   (2+3K)-param search. Fixed by warm-starting every private optimization
   from the converged shared solution (`shared_solution_as_private_start`),
   which guarantees the inequality and is now satisfied exactly (equality)
   or with private strictly worse, at every K.
3. **`exact_W2` had a dropped `/L` factor**, caught by re-deriving the
   normalization from scratch: an earlier simplification of
   `2*L*sum(S)/L**2` to `2*sum(S)` accidentally dropped the L in the
   denominator, giving G-B2 asymptotic slopes double the target (e.g. 0.85
   instead of the target 0.35 for z=1.7). Fixed; re-verified against the
   L→∞ asymptotic slope to 2e-4 precision for all three z.
4. **Degenerate stress case ω̃=2−z at z=2.0 (=0.0) caused optimizer
   overflow**, not a hang but a very slow least-squares fit (bounded
   parameter search wandering to extreme ω_eff, `L_grid**(−ω_eff)`
   overflowing for L up to 2¹⁶). ω̃=0 is physically degenerate (z+ω̃=z, so
   ν₂ just redefines ν, not a genuine second operator) and outside
   OMEGA_BOUNDS=[0.3,2.5] anyway. Replaced with the plan's own ω_min=0.3 as
   the in-bounds near-marginal stress case (also bounded `least_squares`
   with `trf`+bounds instead of unbounded `lm`).
5. **This family's Level-0 σ (0.60) is far larger than exp77's real-system
   σ (0.02–0.15).** Mechanism: W²(L) is a sum over ~L/2 modes weighted by
   S(k)∝k⁻ᶻ, so at α≈0.5 the smallest-k mode dominates the sum (weight
   ratio ~4:1 or more over the next mode), giving W² the fluctuation
   character of a single (roughly) exponential-distributed variable
   (CV≈1) rather than a well-self-averaged sum over many
   effectively-independent contributions — unlike a real simulated
   interface, which has more effective degrees of freedom per realization.
   This is a genuine, reported property of the synthetic family (not a
   bug), and is the reason `da_hi` needed widening from exp77's default 0.6
   to 1.2 (see next item).
6. **floor_L0/floor_L23 saturated at the search ceiling in an earlier pass**
   (da_hi=0.6, inherited from exp77's default): several (U,m) cells,
   especially m=6, hit exactly 0.6 for every U, meaning "no resolution
   within the search range" rather than a precise number. Widened da_hi to
   1.2 (still less than the full alpha-prior range's 2× width) and
   re-ran; the table in §2 is the re-run, with no cell saturated.
7. **floor_L23 > floor_L0 at U=4.0** looks like it violates
   data-processing (richer data should never resolve worse), but Level-0
   and Level-2/3 use methodologically *different, non-nested* adversary
   classes (Level-0: exp77's abstract log-ladder correction, applied
   directly to the scalar log-W_sat summary, with no tie to the physical
   spectral family; Level-2/3: the family's own (ν,ν₂,ω̃) parametrization,
   whose achievable b(L_min) saturates well below nominal U for large ω̃ —
   see the b_fit plateau in §2). There is therefore no theorem forcing
   floor_L23 ≤ floor_L0 in this construction; the crossover at U=4 is a
   consequence of the two adversary classes not being apples-to-apples at
   large U, not a violation of any information-theoretic bound. This is
   flagged as a methodological limitation of the comparison as built, not
   resolved further here.
8. **Summary MLP's fair ceiling is not the raw-field FIM.** The 7-point
   log-W_sat ladder is a lossy function of the raw field even at ν₂=0 (it
   discards mode-by-mode shape); comparing the summary MLP's RMSE to the
   raw-field FIM gave ratio 2.62× (would have failed G-B5 at the literal
   2× threshold). Computed a separate "summary-level" Fisher ceiling
   (delta-method Gaussian ansatz on the independent-per-L log-W2, same
   convention as `THEORY_minimax_floor.md`/exp77) — 0.0477 vs. the raw
   FIM's 0.0273 — against which the MLP's ratio is 1.50×, passing cleanly.
   The CNN (which consumes raw fields) is correctly compared to the raw
   FIM (ratio 1.15×). Using the wrong ceiling for the wrong model would
   have been a false-alarm "gate failure."
9. **Trained-estimator RMSE is far below both floor_L0 and floor_L23 in the
   hard regimes** (e.g. MLP RMSE 0.072–0.078 vs. floor_L0 0.91–1.20 at
   m=1). This is expected, not a violation of honesty rule 4: the Le Cam
   floor bounds *worst-case, adversarially-paired* two-point testing error,
   while the reported RMSE is an *average-case* test error over the full
   sampling prior. The two are different statistical objects; an estimator
   can have small average risk while still failing badly on the specific
   adversarial pairs the floor is about. Neither model beat the FIM
   (Cramér-Rao-type, per-point) ceiling in any regime — that is the
   relevant "beating an information bound would be a bug" check, and it
   passes (`summary_mlp_beats_raw_fim`/`cnn_beats_raw_fim` are `false`
   everywhere in `estimators.json`).
10. **G-B4's own monotonicity check was initially backwards** (asserted
    floor decreasing in U; the correct — and actually observed — direction,
    matching exp77's own `floor_vs_umax` table, is floor increasing with
    U). Fixed in the gate code; not a data problem.

## 6. What we did NOT do

- **G-A3 (measured-Σ pilot, EW+KPZ via `76b_regenerate_ladders.py`)**: not
  run. Explicitly optional per the plan ("If wall-clock is prohibitive...
  the synthetic contrast is the primary deliverable"); the synthetic-channel
  gates (G-A1, G-A2) were prioritized and are complete.
- **`81d_multiobservable_ladders.py`** (Part A pilot data regeneration,
  pass 2): not written; superseded by the G-A3 skip above.
- **Analytic lower bound / sharpness for any floor**: out of scope per the
  plan's honesty rules ("No sharpness claims anywhere"); only the exact
  adversarial upper-bound-by-construction optimization was run.
- **Exhaustive convergence audits of the multivariate optimizer at K=4,5
  private**: the private floor is flat at 0.1359 for K=2..5 (§4); we did not
  determine whether this is the true optimum or an artifact of a
  24-random-start budget in up to a 25-dimensional space beyond the
  shared-solution warm start. Flagged as an open convergence question, not
  resolved.
- **CNN architecture exploration**: a single small per-L conv encoder
  (2 conv layers, global-average-pool, shared MLP head) was used, with no
  architecture search. It passes G-B5 (ratio 1.15× to raw FIM) so was not
  iterated further.
- **Task 5 (scaling-law floor)**: run exactly as specified (exp77 machinery
  unchanged, 2-decade N window, σ_logL∈{0.01,0.03,0.1}, m∈{6,24}) — one
  table, produced in `scaling_law_floor.json`; no further analysis
  (e.g. comparison to any specific published scaling law) was attempted, per
  the plan's "no external data needed" scope.
- **Reconciling the ω_eff-saturates-below-2α discrepancy** (§2/§5 anomaly
  3: measured ω_eff plateaus near 0.63–0.67, not the naively-predicted 2α=
  1.0, for ω̃ ≥ 1.0): measured and reported, not explained mechanistically.

## Task 5 (stretch): scaling-law design floor

Exp77 machinery, unchanged; design = 7 log-spaced N over 2 decades (1e6 to
1e8); σ_logL grid matching typical published compute-scan scatter.

| σ_logL | floor (m=6) | floor (m=24) |
|---|---|---|
| 0.01 | 0.0926 | 0.0457 |
| 0.03 | 0.1605 | 0.1230 |
| 0.10 | 0.2121 | 0.1816 |

At typical reported compute-scan noise (σ_logL≈0.01–0.03) and m≈6–24
training runs, the resolvable neural-scaling exponent gap is Δα_s≈0.05–0.16
— comparable in scale to the disagreements seen between independent
scaling-law fits in the literature (not compared to any specific paper
here; this is a class-conditional floor for that class of fit, exactly like
every other number in this project).
