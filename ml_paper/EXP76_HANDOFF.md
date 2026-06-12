# Exp 76 Handoff — Amortized Finite-Size Extrapolation

Status as of 2026-06-10. Resume point for the next session.

## The chosen direction (decided after exp72–75 review)

The highest-value novel result: **replace the parametric correction-to-scaling
fit (which exp75 showed fails even on known EW/KPZ at L ≤ 256) with an
amortized estimator** — a regressor trained on synthetic W_sat(L) ladders drawn
from a *prior over correction families*, so it implicitly marginalizes over
correction forms instead of assuming one. This is the minimal testable core of
the broader "learned RG coordinates" program (disentangle fixed-point class
from RG-trajectory position).

Prior-art check (done): CNN-FSS work (Li & Luo, arXiv:1711.04252) learns from
spin configurations near criticality; SBI/amortized-inference literature is
generic. Nobody has aimed amortized inference at corrections-to-scaling
extrapolation with a known-class sanity gate. Novel and citeable if the gate
passes — and a rigorous impossibility statement if it fails.

## What exists

- `experiments/76_amortized_extrapolation.py` — full pipeline, stages
  `gen|train|eval` (+ `--pilot`, `--prior`). Synthetic generator with 5
  correction families (pure power law; single power correction ω∈[0.3,2.5];
  two-term; Krug–Meakin intrinsic-width form W²=Wi²+(AL^α)² — BD's textbook
  correction; log corrections), multiplicative noise σ~U[0,0.10],
  scale-invariant features (6 adjacent effective exponents + naive slope +
  curvature + residual + aeff differences = 14), HistGradientBoosting point +
  quantile (5/50/95%) models per training prior + mixture, classical baselines
  (naive slope, fixed-ω=1, fixed-ω=0.5, free-ω fits) reimplementing exp75,
  transfer matrix (train prior × test family), real-data gate. Eval prefers
  high-seed `wsat_perseed.csv` (exp76b) over exp75 6-seed means, and adds a
  seed-bootstrap 90% interval.
- `experiments/76b_regenerate_ladders.py` — regenerates per-seed W_sat(L)
  ladders, 24 seeds, exact exp75 protocol (same simulators via exp63,
  T = 30·L^1.5, late-20% W_sat). Resumable: skips rows already in the CSV.
- `results_exp76_amortized_extrapolation/`: `datasets.npz`, six
  `model_*.joblib` (PILOT-sized: 40k train), `summary.json` (pilot eval),
  `wsat_perseed.csv` (in progress: 304/672 rows — ew complete 7L×24;
  kpz complete through L=128, L=192 at 16/24; bd and eden not started).
- `/tmp/chunk76b.py` (sandbox-only, will be lost) — time-budgeted chunk runner;
  recreate trivially or run 76b directly if process limits allow.

## Pilot result (6-seed exp75 ladders, 40k training)

```
benchmark (synthetic mixture test):  naive 0.307 | fit_w1 0.173 | fit_w0p5 0.206
                                     | fit_free 0.903 | amortized 0.111 RMSE
real:  ew 0.671  kpz 0.564  eden 0.576  bd 0.483  (rd 0.79, degenerate control)
gate (EW/KPZ/Eden within 0.10 of 0.5): FAIL (EW worst)
```

Two findings already: (1) the amortized estimator clearly beats all classical
fits on synthetic data with near-zero bias — the methodological claim holds;
(2) the real-data gate fails on EW, and the diagnosis is **seed noise in the
6-seed exp75 ladders** (EW effective exponents swing 0.19–1.07), not estimator
failure. Tantalizing: BD's pilot α̂ = 0.483, 90% [0.395, 0.675] — consistent
with its KPZ-class value 0.5 — but unusable until the gate passes. Hence
exp76b: 24-seed ladders.

## Sandbox gotchas (cost real time — do not rediscover)

- Background/nohup processes are killed when a bash call ends. Run long jobs
  as foreground time-budgeted chunks that **close files before exit** — rows
  flushed but not closed are lost when `timeout` kills the process.
- The mounted-folder sync occasionally corrupts files edited while a process
  runs (a stray char broke 76 once; 76b got truncated once). If a script
  suddenly has a SyntaxError, rewrite it whole; both current files are clean.
  Verify with `python -c "import ast; ast.parse(open(f).read())"`.
- `pip install scikit-learn numba --break-system-packages` needed per fresh VM.
- One simulator seed costs ~0.1 s (small L) to ~3–7 s (L=256); numba recompiles
  ~5 s per process.

## Resume sequence (in order)

1. Finish `wsat_perseed.csv`: repeatedly run
   `timeout 43 python3 -u experiments/76b_regenerate_ladders.py --seeds 24`
   in ~35 s foreground chunks (or recreate the budgeted runner) until
   "all done" — remaining: kpz L=192 (8) + L=256 (24), all of bd and eden
   (~368 rows, ≈ 10–15 chunks).
2. Full-size training: `python3 experiments/76_amortized_extrapolation.py
   --stage gen` then `--stage train --prior <P>` one prior per call
   (200k samples; each prior trains 4 GBMs, ~1–2 min total per prior).
3. `--stage eval`: check (a) synthetic benchmark table, (b) transfer matrix
   off-diagonal (misspecification robustness), (c) **sanity gate on 24-seed
   EW/KPZ/Eden ladders**, (d) BD α̂ + quantile interval + prior spread +
   seed bootstrap.
4. Decision point:
   - Gate PASSES → headline: "amortized estimator recovers known α where
     classical extrapolation fails, and places BD at α̂ ≈ …" → exp76 closes the
     exp72–75 arc constructively. Write EXPERIMENT_LOG.md entry + new claim in
     CLAIMS_REGISTER.md (evidence: summary.json) + manuscript Section 6/7 hook.
   - Gate FAILS with 24 seeds → equally citeable negative: "even an estimator
     that is Bayes-optimal under a broad correction prior cannot recover known
     exponents from L ≤ 256 / 24-seed data" — quantifies the information limit,
     strengthening the exp75 conclusion. Then check whether failure is seed
     noise (bootstrap interval covers 0.5) vs representation limit.
5. Robustness before claiming anything: leave-one-family-out training
   (train on mix minus F3, test gate again) to show the result isn't an
   artifact of including BD's known correction form in the prior.
6. Documentation: EXPERIMENT_LOG.md entry (follow exp72–75 style, with
   caveats), update CLAIMS_REGISTER.md (new claim + required evidence),
   note in MANUSCRIPT_OUTLINE.md Section 6.

## Paper framing reminder

Venue target: MLST. The exp76 result slots in as the constructive capstone of
the mechanism arc: exp72 (mechanism) → 73 (survives normalization) → 74
(partial recovery, α blocked) → 75 (classical extrapolation fails) → 76
(amortized estimator: either fixes it or proves the information limit).
Either outcome completes the arc. Must-do referee defenses unchanged:
leave-one-family-out control (above) and honest prior-sensitivity spread.
