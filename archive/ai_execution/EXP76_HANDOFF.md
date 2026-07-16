# Exp 76–80 Handoff — Amortized Extrapolation and Minimax Floor

**Status as of 2026-06-13. All experiments complete.**

This file documents the completed state of the exp76–80 research arc, open
items, and the narrative for the paper. It supersedes the earlier draft that
described in-progress work.

---

## The arc in one paragraph

exp75 showed that classical correction-to-scaling extrapolation fails at L ≤ 256:
even EW and KPZ (with known α=0.5) extrapolate to 0.70 and 0.59 when the
correction exponent ω is fixed to 1.0, and the answer swings wildly with ω.
exp76 replaced the fixed-ansatz fit with an amortized estimator trained on a
prior over correction families, which successfully recovered BD's roughness
exponent (α̂ = 0.522 [0.482, 0.529]) — something no classical fit could pin.
exp77 supplied a model-conditional Le Cam two-point risk bound for the declared
Gaussian model, `W_sat` ladder, finite-size design, and bounded correction
class. The reported 0.27–0.44 thresholds imply expected worst-case error of at
least one quarter of the selected pair separation; they are not universal
indistinguishability thresholds. exp78 hardened exp76's claims against four
referee checks.
exp79 analyzed the algebraic structure of the confusion gap, finding a universal
N-scaling law. exp80 demonstrated the floor transfers to other observables
(temporal β, Ising ν). These are conditional risk bounds for the stated
models and observables, not a theorem explaining the richer clustering
experiments.

---

## Experiment status

### exp76: Amortized finite-size extrapolation — COMPLETE

**Files:** `experiments/76_amortized_extrapolation.py`,
`experiments/76b_regenerate_ladders.py`,
`results_exp76_amortized_extrapolation/`

**Key outputs:**
- `wsat_perseed.csv`: 673 rows, 4 systems × 7 L × 24 seeds (complete).
- `summary_full24seed.json`: full 24-seed evaluation.
- `lofo_control.json`: leave-one-family-out (Krug–Meakin removed) control.
- `classical_on_real.json`: classical baselines on 24-seed data.
- `kpz_window.csv`: KPZ gate failure diagnostic.
- `referee_checks.json`: exp78 check results.

**Results:**
- Synthetic benchmark: amortized RMSE 0.106 vs best classical 0.165.
- Transfer matrix: mixture-trained stays at 0.088–0.123 across all families.
- Real gate: EW ✓, Eden ✓, KPZ ✗ (by 0.015; integrator pathology).
- BD: α̂ = 0.522, bootstrap 90% [0.482, 0.529].
- LOFO control: BD → 0.532 [0.486, 0.555] without Krug–Meakin family.

**Honest BD claim:** α̂ ≈ 0.50 ± 0.05 (syst) ± 0.03 (stat). The ±0.05
systematic comes from the slice-conditional bias in exp78 check C. Must be
reported with the statistical interval.

---

### exp77: Le Cam minimax resolution floor — COMPLETE

**Files:** `experiments/77_minimax_floor.py`,
`results_exp77_minimax_floor/floor.json`,
`ml_paper/THEORY_minimax_floor.md`

**Key outputs:**
- `floor.json`: all floor numbers (per-system, resolution law, floor vs m).
- Theory note: full derivation + Appendices A–D (~620 lines).

**Results:**
- D²(0.1) ≈ 1.3×10⁻⁷. Resolving Δα=0.1 needs ~160k seeds (continuum).
- Per-system floors m=24: BD 0.27, EW/KPZ/Eden 0.44.
- Floor vs seeds: nearly flat (0.51/0.44/0.38 at m=6/24/96). Seeds don't help.
- Resolution law: L_max 256→16384 → floor 0.44→0.14 (decades of L required).
- Value of u_max: 4→0.1 → floor 0.44→0.077. Knowledge worth ~6× in resolution.

**Known implementation issue:** The linearized closed-form (`linearized_floor()`
in exp77) has a bug where the 40-vector ω-grid trivially spans the 7-point
feature space (P_perp x ≈ 0). The linearized numbers in floor.json are
therefore unreliable; use the exact `floor_exact` values throughout. The
exact minimax optimization is unaffected.

---

### exp78: Referee-proofing checks — COMPLETE

**Files:** `experiments/78_referee_checks.py`,
`results_exp76_amortized_extrapolation/referee_checks.json`

Four checks:

**A. Exact stationary measure (W²_sat = L/12 for 1D EW/KPZ):**
EW sits at 0.96±0.03 (amplitude offset only, harmless for amplitude-invariant
features). KPZ swings 1.043→0.918 at L=64, ≈4σ. Lam–Shin discretization
pathology. Explains KPZ gate failure. Literature anchor: Lam & Shin, PRE 1998.

**B. Expert additive ansatz (W² = b + aL^{2α}) on BD:**
Free fit α = 0.441 ± 0.011 (5σ from 0.5). Fixed α=0.5: χ²/dof = 6.2.
The textbook two-parameter ansatz is rejected on BD at L ≤ 256.

**C. Discriminability control:**
300 BD-like ladders at true α ∈ {0.40, 0.45, 0.50, 0.55}. Predictions
0.425/0.496/0.548/0.607. Adjacent α separated ~3σ. Response slope 1.2
(no shrinkage). Slice-conditional bias +0.05 (true 0.50 → predicted 0.548).

**D. Literature priority scan:**
Hall–Welsh/Le Cam minimax theory mature in extreme-value statistics, but no
application to FSS found. Claim "to our knowledge, first" survives initial
scan; Borwein–Erdélyi pass still pending.

---

### exp79: Lemma scaling test and constant certificates — COMPLETE

**Files:** `experiments/79_lemma_scaling_test.py`,
`experiments/79b_constant_certificates.py`

**Central lemma (numerically verified):** Minimum L² distance from Δα·x to
bounded N-term exponential sums on [0,T]:

    E_N = c_N · √T · U · (ΔαT/U)^{N+1}

**Verified constants:** c₁ = 0.0375 (construction optimal at tested parameters),
c₂ = 0.0216 (optimizer beats uniform-node construction by 14%), c₃ ≈ 0.019.

**Algebraic insight:** With Richardson weights, the residual is exactly
Δα·(1-e^{-ωx})^N (binomial theorem). Substitution t = e^{-ωx} converts the
problem to polynomial approximation of log(1/t) — a classical problem solvable
via Legendre projection or Bernstein ellipse.

**Open items:**
1. Analytic lower bound proof (harmonic-node case is tractable).
2. Optimal nodes for N≥2 (uniform nodes are suboptimal; open problem).

---

### exp80: Second-observable floor transfers — COMPLETE

**Files:** `experiments/80_second_observable_floors.py`,
`results_exp80_second_observable_floors/floors.json`

**Part A (temporal β):** Design: 7 log-spaced times t=50..5000, L=1024, 8 seeds.
β floor ≈ 0.07–0.08. EW/KPZ β-gap (0.083) at/just above floor. Retrodicts exp74
asymmetry: time window spans ~2 decades vs size window's ~0.9 decades.
Caveat: temporal correlations within a seed not captured by iid-noise model.

**Part B (Ising ν):** Design: exp52d L ∈ {32,48,64,96}. Floors:
agnostic 0.31–0.39; Ising-honest 0.13–0.17; Ising-strict 0.05–0.13 (in 1/ν).
exp52d's 7.3% deviation consistent with strict-class floor. Exact-solution
knowledge worth ~5–6× in resolution.

---

## Files on disk (complete state)

```
experiments/
  75_correction_to_scaling.py        classical fits, exp75
  76_amortized_extrapolation.py      amortized estimator, full pipeline
  76b_regenerate_ladders.py          per-seed W_sat generation
  77_minimax_floor.py                Le Cam floor computation
  78_referee_checks.py               4 referee checks
  79_lemma_scaling_test.py           N-scaling verification
  79b_constant_certificates.py       c_1, c_2, c_3 certificates
  80_second_observable_floors.py     beta and nu floor transfers

results_exp76_amortized_extrapolation/
  wsat_perseed.csv                   673 rows, 4 × 7 × 24
  summary_full24seed.json            full 24-seed eval
  lofo_control.json                  leave-one-family-out
  classical_on_real.json             classical baselines
  kpz_window.csv                     KPZ gate diagnostic
  referee_checks.json                exp78 results

results_exp77_minimax_floor/
  floor.json                         all floor numbers

results_exp80_second_observable_floors/
  floors.json                        beta and nu floors

ml_paper/
  THEORY_minimax_floor.md            ~620-line theory note (Appendices A–D)
  CLAIMS_REGISTER.md                 all claims, Part I (exp62–75) + Part II (exp76–80)
  MANUSCRIPT_OUTLINE.md              Paper 2 (floor, primary) + Paper 1 (clustering)
  archive/ai_execution/EXP76_HANDOFF.md                   this file

docs/
  EXPERIMENT_LOG.md                  entries through exp80
```

---

## Paper-writing state

**Theory note (`THEORY_minimax_floor.md`):** Complete draft including:
- Main body: setup, Le Cam bound, confusion gap, computed floors.
- Addendums 1–5: mechanism (ω→0 degeneracy), resolution law, c₁ exact,
  verification ledger, ω-range/amplitude correction.
- Appendix A: full Richardson construction and proof.
- Appendix B: RG-equivariance corollary, compute-allocation criterion.
- Appendix C: prior-art positioning map (9 references).
- Appendix D: binomial identity, log-polynomial reduction, Bernstein ellipse,
  constraint protection.

**What maps to the paper:**
- Theory note sections → paper sections (renumbered, condensed).
- floor.json → Tables 1–2.
- Figure 1: floor landscape (D² vs Δα; floor vs L_max; floor vs u_max).
- Figure 2: real-data recovery with floor marked.
- Figure 3: N-scaling log-log plot with theoretical slopes.

---

## Open items before submission

**Blocking:**
1. Write the paper (MLST format, ~8–10 pages + appendices).
   - Outline is in MANUSCRIPT_OUTLINE.md.
   - All numerical results are in place.
   - All figures need to be generated from scripts (no figure generation code
     yet — must be written, or generated interactively from floor.json etc.).

**Non-blocking (strengthening):**
2. Borwein–Erdélyi prior-art pass: confirm no FSS minimax bound in the
   approximation-theory literature.
3. Analytic lower bound for c_N (harmonic-node construction).
4. KPZ exact-measure integrator (Lam–Shin exact scheme) to test gate.
5. ω-range vs amplitude decomposition: verify Correction to Addendum 4 is
   properly attributed in the note.
6. Optimal nodes for N≥2 (open mathematical problem; not required for submission).

---

## Sandbox gotchas (cost real time — do not rediscover)

- Background/nohup processes are killed when a bash call ends. Always run
  time-budgeted foreground chunks that **close files before exit**.
- Mount sync corrupts files edited mid-process: if a script suddenly has a
  SyntaxError, rewrite it whole.
- `pip install scikit-learn numba --break-system-packages` needed per fresh VM.
- `numpy.math.factorial` removed in newer numpy — use `import math; math.factorial`.
- Git: branch `exp76-79` has the experiment commits. If pushing from VM,
  use `/tmp/ud` clone to avoid stale lock files from crashed processes.
  `git push origin HEAD:exp76-79` from there.
- The linearized floor formula (see exp77 known issue above) is buggy; never
  cite `floor_linearized` values from floor.json.
