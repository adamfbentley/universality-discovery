# Exp 85c Report -- Confirmatory Coverage And Calibration Debts

Findings only. `CLAIMS_REGISTER.md` was not edited.

## Gate Ledger

| Gate | Check | Result | Proof path |
|---|---|---|---|
| G-85c-1a U0.5 | exp85b observed 1.000 inside analytic predicted band | met | `results_exp85c_confirmatory/task1_analytic_prediction.json` |
| G-85c-1a U1 | exp85b observed 1.000 inside analytic predicted band | met | `results_exp85c_confirmatory/task1_analytic_prediction.json` |
| G-85c-1b validity U0.5 | in-class coverage >= 0.95 minus one-sided binomial 2sigma | met | `results_exp85c_confirmatory/score.json` |
| G-85c-1b consistency U0.5 | observed in-class coverage within 2sigma of Task-1a prediction | met | `results_exp85c_confirmatory/score.json` |
| G-85c-1b validity U1 | in-class coverage >= 0.95 minus one-sided binomial 2sigma | met | `results_exp85c_confirmatory/score.json` |
| G-85c-1b consistency U1 | observed in-class coverage within 2sigma of Task-1a prediction | met | `results_exp85c_confirmatory/score.json` |
| G-85c-2 | BD U0.5 synthetic center reproduction | bug | `results_exp85c_confirmatory/task2_bd_diagnosis.json` |
| G-85c-3 | calibrated profile widths within 25% of Wilks widths | not met | `results_exp85c_confirmatory/task3_calibration_debts.json` |
| G-85c-4 | Ising procedural commit order; coverage reported, not gated | recorded | phase-1 commit `4f55069ec22080d1351da6280c18973785fe490f`; phase-2 score commit `0f24f4becdac7ac3163a3ee7de387e89c040ea9e` |
| Blinding | phase-1 hashes committed before scoring | recorded | phase-1 commit `4f55069ec22080d1351da6280c18973785fe490f`; phase-2 score commit `0f24f4becdac7ac3163a3ee7de387e89c040ea9e` |

## Task 1

| Class | Task-1a predicted mean | 2sigma band n=120 | exp85b observed | Confirmatory in-class | Valid lower | Consistency band |
|---|---:|---|---:|---:|---:|---|
| U0.5 | 0.9973 | [0.9878, 1.0000] | 1.000 | 1.000 | 0.906 | [0.9869, 1.0000] |
| U1 | 0.9995 | [0.9952, 1.0000] | 1.000 | 1.000 | 0.906 | [0.9948, 1.0000] |

## Task 2

| Check | Value |
|---|---|
| BD U0.5 design | [32.0, 48.0, 64.0, 96.0]; 4-point: yes |
| BD U0.5 weights | [2.33076, -3.64820, -4.28174, 4.60060] |
| BD U4 design | [32.0, 48.0, 64.0, 96.0]; 4-point: yes |
| Synthetic centers | observed -0.7156; mean -2.2746; 95% spread [-2.3301, -2.2146] |
| Bug bisection | weight sum -0.9986; deterministic c=0 center -2.2743; required c -1.5609 moves center to observed |
| Half-window dof | 0 |
| BD U4 width / alpha prior width | 1.141 for [0.2,0.9]; 1.997 for [0.3,0.7] |
| BD U4 plausible-grid analytic coverage | median 0.000, min 0.000, share >=0.95 0.127 |

## Task 3

| System | U | q68 cal/Wilks | q95 cal/Wilks | 68 width cal/Wilks | 95 width cal/Wilks | Gate |
|---|---:|---:|---:|---:|---:|---|
| bd | 0.5 | 0.825/1.000 | 3.503/3.840 | 0.050/0.060 | 0.095/0.095 | met |
| bd | 4 | 0.787/1.000 | 3.792/3.840 | 0.265/0.275 | 0.335/0.335 | met |
| eden | 0.5 | 0.628/1.000 | 3.252/3.840 | 0.175/0.245 | 0.340/0.350 | not met |
| eden | 4 | 0.595/1.000 | 3.231/3.840 | 0.170/0.245 | 0.595/0.620 | not met |

| System | U | pinned chi2/dof | p-value | p<0.05 |
|---|---:|---:|---:|---|
| bd | 1 | 0.466 | 0.7608 | no |
| bd | 4 | 0.466 | 0.7608 | no |
| eden | 1 | 0.561 | 0.6910 | no |
| eden | 4 | 0.561 | 0.6910 | no |
| ew | 1 | 0.968 | 0.4233 | no |
| ew | 4 | 0.968 | 0.4233 | no |
| kpz | 1 | 0.890 | 0.4688 | no |
| kpz | 4 | 0.890 | 0.4688 | no |

Eden profile curve table: `results_exp85c_confirmatory/eden_profile_curve.csv`.

| Eden U | min alpha | min boundary | 68-width boundary-active | 68 width |
|---|---:|---|---|---:|
| 0.5 | 0.380 | no | yes | 0.245 |
| 1 | 0.380 | no | yes | 0.245 |
| 4 | 0.380 | no | yes | 0.245 |

| System | U | local floor | global floor | naive err | w1 err | w0.5 err | free err | exp76 err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bd | 0.5 | 0.0620 | 0.0787 | 0.1367 | 0.0817 | 0.0120 | 0.1965 | 0.0216 |
| bd | 1.0 | 0.0800 | 0.1194 | 0.1367 | 0.0817 | 0.0120 | 0.1965 | 0.0216 |
| bd | 4.0 | 0.2239 | 0.2679 | 0.1367 | 0.0817 | 0.0120 | 0.1965 | 0.0216 |
| eden | 0.5 | 0.1352 | 0.1736 | 0.0277 | 0.0649 | 0.0918 | 0.1092 | 0.0086 |
| eden | 1.0 | 0.1553 | 0.2963 | 0.0277 | 0.0649 | 0.0918 | 0.1092 | 0.0086 |
| eden | 4.0 | 0.3373 | 0.4310 | 0.0277 | 0.0649 | 0.0918 | 0.1092 | 0.0086 |
| ew | 0.5 | 0.1251 | 0.1773 | 0.0018 | 0.0051 | 0.0359 | NA | 0.0315 |
| ew | 1.0 | 0.1801 | 0.3040 | 0.0018 | 0.0051 | 0.0359 | NA | 0.0315 |
| ew | 4.0 | 0.3659 | 0.4367 | 0.0018 | 0.0051 | 0.0359 | NA | 0.0315 |
| kpz | 0.5 | 0.1246 | 0.1818 | 0.0298 | 0.1187 | 0.2155 | 0.0260 | 0.1147 |
| kpz | 1.0 | 0.2029 | 0.3119 | 0.0298 | 0.1187 | 0.2155 | 0.0260 | 0.1147 |
| kpz | 4.0 | 0.4260 | 0.4436 | 0.0298 | 0.1187 | 0.2155 | 0.0260 | 0.1147 |

## Task 4

Ising regeneration note: Pilot one (L,seed) took 10.42s; projected full 4L x 24 seeds 1000.6s within guard 1800.0s.
| Class | Blind CI | Center | Covers exact 1/nu | Covers heldout L64-96 slope | Covers full L32-96 slope |
|---|---|---:|---|---|---|
| loose_U1_omega_ge_1 | [-819951.3090, -763041.4706] | -791496.3898 | no | no | no |
| strict_U0.3_omega_ge_1 | [-0.4612, 0.6939] | 0.1163 | no | yes | yes |

Heldout OLS slope: 0.1101; full OLS slope: 0.1661; exact 1/nu: 1.0000.

## Post-hoc Notes

- Single-session blinding here is a discipline device, not information isolation.
- The Ising per-L ladder reconstructs exp52d's PC1 collapse observable as a local PC1-vs-temperature slope, because exp52d stored only the final collapse summary and PNG.

## What We Did Not Do

- No entry was added to `CLAIMS_REGISTER.md`.
- No failed gate was relabeled or rerun until it passed.
- No tightness gate was added to G-85c-1b.

## Anomalies And Bugs

- BD synthetic reproduction did not contain the observed center; The failed synthetic reproduction occurs before sampling noise: the exp85b BD U=0.5 four-point affine center has weight sum near -1 rather than 0, so the alpha center is sensitive to the additive log-amplitude c. The c=0 synthetic mean centers near the synthetic distribution, while changing only c moves the center toward the observed value. The blind half-window center is therefore dominated by an amplitude-invariance failure in the 85b affine-center pipeline on the four-point design, not by the declared correction mechanism alone.
- At least one calibrated profile width is outside 25% of the Wilks width.
