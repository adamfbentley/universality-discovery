# Exp 85 Report -- Preregistered Validation

Findings only. `CLAIMS_REGISTER.md` was not edited.

## Gate Ledger

| Gate | Check | Result | Proof path |
|---|---|---|---|
| G-85a bd | local floor <= honest global floor; D2 monotone | met | `results_exp85_preregistered/task1_local_floor.json` |
| G-85a eden | local floor <= honest global floor; D2 monotone | met | `results_exp85_preregistered/task1_local_floor.json` |
| G-85a ew | local floor <= honest global floor; D2 monotone | met | `results_exp85_preregistered/task1_local_floor.json` |
| G-85a kpz | local floor <= honest global floor; D2 monotone | met | `results_exp85_preregistered/task1_local_floor.json` |
| G-85b | BD additive-width alpha=0.5 p < 0.05 power anchor | met | `results_exp85_preregistered/task2_gof.json` |
| G-85c pre U0.5 | pure nu2=0 95% CI coverage within binomial 2sigma | not met | `results_exp85_preregistered/task3_pregate.json` |
| G-85c pre U1 | pure nu2=0 95% CI coverage within binomial 2sigma | not met | `results_exp85_preregistered/task3_pregate.json` |
| Blinding | predictions hash committed before scoring | not reached | G-85c pre-gate blocked phase 1 before `predictions.json` was written |

## Task 1 Decision Table

Level-0, class N=1/log-form/U as tabulated/omega_min=0.3, design={32,48,64,96,128,192,256}, sigma=measured per-system median sd(log W), m=24

| System | sigma | fitted u | honest U | global U=4 floor | honest global floor | instance-local floor | exp76 abs err | fit_w1 abs err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bd | 0.0190 | 3.3190 | 6.6381 | 0.2707 | 0.3420 | 0.2431 | 0.0216 | 0.0817 |
| eden | 0.1378 | -0.3035 | 0.6070 | 0.4348 | 0.2144 | 0.1616 | 0.0086 | 0.0649 |
| ew | 0.1454 | 4.0000 | 8.0000 | 0.4395 | 0.5951 | 0.3486 | 0.0315 | 0.0051 |
| kpz | 0.1541 | 0.1417 | 0.2835 | 0.4441 | 0.1486 | 0.0615 | 0.1147 | 0.1187 |

## Task 2 Adequacy

Level-0, classes N=1/U=0.5/omega_min=0.3, N=1/U=4/omega_min=0.3, and additive-width alpha=0.5; design={32,48,64,96,128,192,256}; sigma=measured per-system; m=24

| System | N1 U=0.5 p | N1 U=4 p | additive alpha=0.5 p | additive chi2/dof |
|---|---:|---:|---:|---:|
| bd | 0.667 | 0.766 | 0.002 | 6.04 |
| eden | 0.792 | 0.790 | 0.563 | 0.70 |
| ew | 0.395 | 0.413 | 0.359 | 1.09 |
| kpz | 0.661 | 0.599 | 0.158 | 1.61 |

## Task 3 Coverage

Five-tuple: Level-0 fractional-EW seed log-width ladder, class N=1 hull J=4 with U in {0.5,1}, omega_min=0.3, design={32,48,64,96,128,192,256}, sigma=config-measured median sd(log W), m=24.

| Class | Pre-gate coverage | Pre-gate window | Main in-class coverage | Main window | Out-of-class coverage | median half/floor |
|---|---:|---|---:|---|---:|---:|
| U0.5 | 1.000 | [0.919, 0.981] | pending | pending | pending | pending |
| U1 | 1.000 | [0.919, 0.981] | pending | pending | pending | pending |

Main blind coverage and falsifiability tables were not generated because the pre-gate stopped the run before `predictions.json`.

## Task 4 Real-Data Half-Window

Five-tuple BD: Level-0 real BD log-width ladder, class N=1/U=0.5/omega_min=0.3, design={32,48,64,96}, sigma=BD half-window measured median sd(log W), m=24.

BD half-window blind prediction was not generated because G-85c pre-gate blocked phase 1 before Task 4 prediction output.

Ising half-window status: not reached.

## What We Did Not Do

- No entry was added to `CLAIMS_REGISTER.md`.
- `results_exp84_presubmission/` was absent in this checkout, so the A-K fixed-length CI machinery was implemented locally in `experiments/85_preregistered_validation.py`.
- `predictions.json`, `predictions.sha256`, blind main coverage, falsifiability scoring, and real-data half-window predictions were not produced because G-85c pre-gate failed.

## Anomalies And Bugs

- G-85c pre-gate failed by overcoverage after the affine center was corrected to have zero response to constants and unit response to `log L`: U=0.5 coverage 1.000 and U=1 coverage 1.000, outside the preregistered [0.919, 0.981] window.
- Before the affine-normalization fix, the U=1 calibration undercovered (0.830), revealing that raw extremal-pair weights did not define a properly normalized alpha estimator. The script now records normalized and raw weights.
