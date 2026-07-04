# Exp 85b Report -- Corrected A-K Validation

Findings only. `CLAIMS_REGISTER.md` was not edited.

## Gate Ledger

| Gate | Check | Result | Proof path |
|---|---|---|---|
| G-85b-A | true A-K center shrinkage, half-length, grid risk | met | `results_exp85b_validation/taskA_ak.json` |
| G-85b-B pure U0.5 | coverage >= one-sided binomial 2sigma lower | met | `results_exp85b_validation/taskB_pregates.json` |
| G-85b-B pure U1 | coverage >= one-sided binomial 2sigma lower | met | `results_exp85b_validation/taskB_pregates.json` |
| G-85b-B least-fav U0.5 | coverage in [0.92,0.995] with pooled misses | met | `results_exp85b_validation/taskB_pregates.json` |
| G-85b-B least-fav U1 | coverage in [0.92,0.995] with pooled misses | met | `results_exp85b_validation/taskB_pregates.json` |
| G-85b-C bd | floor nondecreasing in U; truth-pinned local <= global | met | `results_exp85b_validation/taskC_identification_floor.json` |
| G-85b-C eden | floor nondecreasing in U; truth-pinned local <= global | met | `results_exp85b_validation/taskC_identification_floor.json` |
| G-85b-C ew | floor nondecreasing in U; truth-pinned local <= global | met | `results_exp85b_validation/taskC_identification_floor.json` |
| G-85b-C kpz | floor nondecreasing in U; truth-pinned local <= global | met | `results_exp85b_validation/taskC_identification_floor.json` |
| G-85b-D U0.5 | in-class coverage within binomial 2sigma of 0.95 | not met | `results_exp85b_validation/score.json` |
| G-85b-D U1 | in-class coverage within binomial 2sigma of 0.95 | not met | `results_exp85b_validation/score.json` |
| Blinding | predictions hash committed before scoring | recorded | phase-1 commit `ba761bfd18935bf2623cc9f6a740dc01bbddc72e`; phase-2 score commit `3269d4aac4df2487f438680d882eedc08bf24173` |

## A-K Center

| Class | Sigma scenario | A-K logL response | GLS logL response | A-K half | GLS half | A-K worst risk | GLS worst risk |
|---|---|---:|---:|---:|---:|---:|---:|
| U0.5 | fractional_reference | 0.9950 | 1.0000 | 0.2279 | 0.2284 | 0.1482 | 0.1738 |
| U1 | fractional_reference | 0.3874 | 1.0000 | 0.3181 | 0.3537 | 0.2084 | 0.6763 |

## Calibration

| Class | Pure coverage | Pure lower | Least-fav pooled coverage | Least-fav misses |
|---|---:|---:|---:|---:|
| U0.5 | 1.000 | 0.919 | 0.963 | 15 |
| U1 | 1.000 | 0.919 | 0.953 | 19 |

## Identified Sets

| System | U | alpha at profile min | 68 width | 95 width |
|---|---:|---:|---:|---:|
| bd | 0.5 | 0.485 | 0.060 | 0.095 |
| bd | 1 | 0.540 | 0.110 | 0.155 |
| bd | 4 | 0.710 | 0.275 | 0.335 |
| eden | 0.5 | 0.380 | 0.245 | 0.350 |
| eden | 1 | 0.380 | 0.245 | 0.440 |
| eden | 4 | 0.380 | 0.245 | 0.620 |
| ew | 0.5 | 0.580 | 0.215 | 0.340 |
| ew | 1 | 0.625 | 0.280 | 0.420 |
| ew | 4 | 0.770 | 0.485 | 0.585 |
| kpz | 0.5 | 0.535 | 0.145 | 0.200 |
| kpz | 1 | 0.535 | 0.225 | 0.295 |
| kpz | 4 | 0.535 | 0.395 | 0.425 |

## Floor Curves

| System | U=0.25 | U=0.5 | U=1 | U=2 | U=4 | U=8 |
|---|---:|---:|---:|---:|---:|---:|
| bd | 0.0416 | 0.0787 | 0.1194 | 0.1882 | 0.2679 | 0.3713 |
| eden | 0.1279 | 0.1736 | 0.2963 | 0.3038 | 0.4310 | 0.5873 |
| ew | 0.1314 | 0.1773 | 0.3040 | 0.3064 | 0.4367 | 0.5952 |
| kpz | 0.1356 | 0.1818 | 0.3119 | 0.3181 | 0.4436 | 0.6029 |

## Truth-Pinned Local Floors

| System | U | local floor | global floor |
|---|---:|---:|---:|
| bd | 1 | 0.0800 | 0.1194 |
| bd | 4 | 0.2239 | 0.2679 |
| eden | 1 | 0.1553 | 0.2963 |
| eden | 4 | 0.3373 | 0.4310 |
| ew | 1 | 0.1801 | 0.3040 |
| ew | 4 | 0.3659 | 0.4367 |
| kpz | 1 | 0.2029 | 0.3119 |
| kpz | 4 | 0.4260 | 0.4436 |

## Blind Phase

| Class | In-class coverage | 2sigma window | Out-of-class coverage | Median half/floor |
|---|---:|---|---:|---:|
| U0.5 | 1.000 | [0.910, 0.990] | 0.975 | 0.81 |
| U1 | 1.000 | [0.910, 0.990] | 1.000 | 0.77 |

| Class | adequacy pass/cover | pass/miss | fail/cover | fail/miss |
|---|---:|---:|---:|---:|
| U0.5 | 67 | 1 | 11 | 1 |
| U1 | 69 | 0 | 11 | 0 |

## Real-Data Half-Window

| Target | Class | Blind CI | Center | After unblind |
|---|---|---|---:|---|
| BD alpha | U0.5 | [-0.8684, -0.5628] | -0.7156 | covers alpha=0.5: no; covers full-window exp76: no |
| BD alpha | U4 | [-0.2473, 0.5514] | 0.1520 | covers alpha=0.5: yes; covers full-window exp76: yes |
| Ising 1/nu | | not_run_missing_per_L_collapse_ladder | NA | stored artifact has no per-L half-window ladder |

## What We Did Not Do

- No entry was added to `CLAIMS_REGISTER.md`.
- No honest U was estimated from fitted correction amplitudes.
- The Ising L={32,48} blind CI was not constructed because exp52d does not store the per-L collapse observable ladder required before unblinding.
- Blind adequacy p-values used the asymptotic chi-square single-fit diagnostic recorded in `predictions.json`; the calibrated pre-gates used the generated calibration ladders.
- Phase-1 predictions contain no truth fields; phase 2 reconstructs truth configs only after the committed SHA256 check.

## Anomalies And Bugs

- The Ising half-window prediction remains blocked by missing stored per-L exp52d data.
