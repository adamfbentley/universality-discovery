# Execution prompt: exp81 — observable-information hierarchy

Copy everything below the line into a fresh Claude Sonnet session with access
to this repository. A senior reviewer will audit the output against the
acceptance gates; report failures honestly — a failed gate reported clearly is
a good outcome, a fudged pass is the worst possible outcome.

---

You are implementing experiment 81 in the `universality-discovery` repository.
You are the implementer, not the scientific judge: follow the plan, run the
gates, and report. Do not reframe claims, do not touch `CLAIMS_REGISTER.md`,
`MANUSCRIPT_OUTLINE.md`, or any existing results directories.

## Read first, in this order

1. `ml_paper/EXP81_PLAN.md` — the full plan; it is binding, including the
   honesty rules and acceptance gates. Where this prompt and the plan
   disagree, the plan wins.
2. `ml_paper/THEORY_minimax_floor.md` — Appendix F (definitions you will
   implement), the main Le Cam section, and the "Known issue" note (never use
   the linearized floor; only exact optimization).
3. `experiments/77_minimax_floor.py` — the exact confusion-gap optimizer you
   will extend. Reuse its multi-start bounded-optimization pattern.
4. `ml_paper/EXP76_HANDOFF.md` — section "Sandbox gotchas". All of them still
   apply. In particular: no background processes; time-budgeted foreground
   chunks that close files before exit; `pip install ... --break-system-packages`;
   rewrite (don't patch) any script that develops a mid-file SyntaxError.

## Task order (do not reorder; later tasks depend on earlier gates)

### Task 1 — Fractional-EW testbed core (`experiments/81_fractional_ew_testbed.py`)

Implement the family S(k) = D/(ν|k|^z + ν₂|k|^{z+ω̃}) on 1D lattices,
exact stationary sampling via independent Fourier modes + inverse FFT.
Be careful and explicit about FFT normalization and real-field Hermitian
symmetry; document the convention in a docstring.

Then pass gates before anything else:
- **G-B1**: sampled ensemble spectrum matches S(k) (overlay plot saved to
  `results_exp81_hierarchy/figs/`, plus a quantitative check per k-band).
- **G-B2**: with ν₂ = 0, measured α from log W² vs log L regression on the
  design {32,48,64,96,128,192,256} recovers (z−1)/2 across z ∈ {1.7, 2.0, 2.3}
  to within stated stochastic error (use enough seeds that the check is sharp;
  this is cheap).
- **G-B3**: implement the per-mode Gaussian log-likelihood and verify the KL
  formula KL = Σ_k [S₁/S₂ − 1 + ln(S₂/S₁)] (per complex mode) against the
  empirical mean log-likelihood-ratio on sampled fields, agreement ~1%.
  If your convention differs by a factor (real vs complex modes, one-sided vs
  two-sided spectrum), FIX THE FORMULA to match the verified likelihood and
  record the corrected form in the report. The verified likelihood is the
  ground truth, not the formula as written.

### Task 2 — Floors at Level 0 and Level 2/3 (same script)

- Level 0: measure σ of log W_sat across seeds for this family (per L, per
  parameter setting near α = 0.5), then run the exp77-style exact two-point
  optimization on the 7-point design with declared class (N=1, ω_min = 0.3,
  U ∈ {0.5, 1, 4}). **G-B4**: sanity-check the numbers against the σ-scaling
  behavior in `results_exp77_minimax_floor/floor.json` (order of magnitude and
  monotonicity in U; exact agreement is not expected since σ differs).
- Level 2/3: Le Cam with the adversary inside the family — minimize the
  verified KL between (z, D, ν, ν₂, ω̃)₁ and (z + 2Δα, ...)₂ over nuisances
  (bounded as in the plan's sampling prior), summed over the 7 sizes,
  multi-start (≥ 24 starts; record the best and the start-to-start scatter).
  Also compute the exact no-nuisance Fisher information of α (formula in
  Appendix F2; validate its convention the same way as the KL).
- **Primary output**: table of floor_L0 vs floor_L23 vs Δα_FIM = 1/sqrt(m·I(α))
  as functions of (U, ω̃, m ∈ {6, 24}), written to
  `results_exp81_hierarchy/floors_hierarchy.json` with the five-tuple
  (observable level, class N/U/ω_min, design, noise source, m) attached to
  every number. Include the near-marginal stress case ω̃ = 2−z.
- **Correction-exponent measurement (mandatory before declaring any class):**
  the plan's mode-sum caveat says ω_eff = ω̃ only for ω̃ < 2α, saturating at
  2α above. Measure ω_eff from clean high-seed ladders (fit the relative
  deviation of W² from pure power law) for each ω̃ you use, and declare the
  MEASURED ω_eff in every floor five-tuple.

### Task 3 — Trained estimators (`experiments/81b_estimators.py`)

Sampling prior exactly as in EXP81_PLAN.md ("Sampling prior" section);
α-independence of nuisances is mandatory and must be asserted in code
(document the prior in the output JSON).

- Summary MLP on the 7-point log-W_sat ladder; small 1D CNN on raw fields
  (per-L encoder, pooled head). Same prior, same α-regression loss, ~100k
  training samples is plenty (sampling is instant). Torch if available
  (`pip install torch --break-system-packages`, CPU is fine at these sizes);
  if torch install fails, use sklearn MLPRegressor for the summary model and
  a numpy/scipy ridge regression on log-spectrum band features as the
  "raw-field" model — declare the substitution prominently in the report.
- **G-B5** first: at ν₂ = 0 (no corrections), both models must recover α with
  RMSE within ~2× of the FIM bound before any hard-regime run is reported.
- Then the hard regimes (U ∈ {0.5, 1, 4}): report test RMSE of both models
  against their respective floors from Task 2. If any model beats an exact
  information bound, that is a bug (leakage or floor error) — stop, diagnose,
  report; do not present it as a result.

### Task 4 — Multivariate floor (`experiments/81c_multivariate_floor.py`)

Implement the Level-1 confusion gap from Appendix F1 (vector channels,
measured or synthetic Σ, SHARED-ω vs private-ω adversaries, bounded
amplitudes). Gates:
- **G-A1**: K=1 reduction reproduces the exp77 exact floor for matching
  (σ, U, ω_min, design) — anchor: BD, m=24, floor 0.27. State the tolerance
  achieved.
- **G-A2**: floor-vs-K curves (K = 1..5) under identical U for (i) shared
  correction spectrum, (ii) private per-channel corrections, (iii) shared
  spectrum + strongly correlated noise (Σ with off-diagonal 0.8). Synthetic
  channels are sufficient for this deliverable: θ_k grid including θ = α,
  θ = −(1+2α), and two pure-nuisance channels θ = 0.
- Measured-Σ pilot (adapting `76b_regenerate_ladders.py`, EW+KPZ, m=8) is
  OPTIONAL — only if wall-clock allows after everything above; G-A3 applies
  (provenance recorded).

### Task 5 — Stretch only: scaling-law design floor

Only if Tasks 1–4 are complete and gated: run the exp77 machinery on the
neural-scaling-law design in EXP81_PLAN.md Part C. One table.

### Task 6 — Report (`ml_paper/EXP81_REPORT.md`)

Structure: (1) gate ledger — every gate, PASS/FAIL, the number, the file that
proves it; (2) the floors table; (3) estimator-vs-floor table; (4) H-A1/H-A2/
H-A3 outcome; (5) anomalies, bugs found, deviations from plan; (6) explicitly:
what you did NOT do. Every number in the report must exist in a JSON/CSV in
`results_exp81_hierarchy/`. No claims language ("proves", "demonstrates",
"first") — findings only; the reviewer decides what they mean.

## Binding rules

- The honesty rules in EXP81_PLAN.md apply verbatim.
- Never cite or use `floor_linearized`-style closed forms; exact optimization
  only.
- Multi-start every adversarial optimization; report scatter across starts.
  A confusion gap is an upper-bound-by-construction: a better start can only
  lower it, so report the minimum and note convergence.
- Pilot scale throughout: everything here runs in minutes to tens of minutes
  on CPU. If something takes hours, your design is wrong — stop and simplify.
- Commit experiments + results + report on a branch named `exp81`, message
  style matching the repo's history. Do not push to main.
