# Exp 86 Report -- Validation Spine

Findings only. `CLAIMS_REGISTER.md` was not edited.

## Gate Ledger

| Gate | Check | Result | Proof path |
|---|---|---|---|
| G-86-1 | fixed constraints, amplitude invariance, 7-point no-regression | met | `results_exp86_task1/task1_score.json` |
| Task 1 blinding | fixed phase-1 predictions committed before truth scoring | recorded | phase-1 commit `dbb0fe1bdcbae19cdd7a5deff49f5bec8d8ac7a6`; phase-2 score commit `49026fdc89948b24215f18b98557e17e2dc9726a` |
| G-86-2 | real Ising external anchor procedural gate | met | phase-1 commit `576d95dca33a8e1afcd3a2119d555ad4825fa87d`; phase-2 score commit `d7d3318d21a2eda38e6d62e85023d8be39c3b093` |
| G-86-3 | exact honest CIs, smooth-prior van Trees, data-driven U | pending | not run yet |
| G-86-4 | modulus prior-art pass | pending | not run yet |

## Task 1

| Quantity | Value |
|---|---:|
| exp85b BD U0.5 sum(w) | -0.9986 |
| exp85b BD U0.5 w dot logL | -2.8537 |
| fixed 7pt_U0.5 sum(w) | -0.0000000000 |
| fixed 7pt_U0.5 w dot logL | 1.0000000000 |
| fixed 7pt_U1 sum(w) | 0.0000000000 |
| fixed 7pt_U1 w dot logL | 1.0000000000 |
| fixed bd4_U0.5 sum(w) | 0.0000000000 |
| fixed bd4_U0.5 w dot logL | 1.0000000000 |
| fixed bd4_U4 sum(w) | 0.0000000000 |
| fixed bd4_U4 w dot logL | 1.0000000000 |

| Class | fixed in-class coverage | exp85c coverage | 2sigma slack | no-regression |
|---|---:|---:|---:|---|
| U0.5 | 1.000 | 1.000 | 0.044 | yes |
| U1 | 1.000 | 1.000 | 0.044 | yes |

| BD class | fixed CI | center | covers alpha=0.5 | covers exp76 full-window |
|---|---|---:|---|---|
| U0.5 | [-0.6313, 1.4353] | 0.4020 | yes | yes |
| U4 | [-13.1844, 14.1854] | 0.5005 | yes | yes |

## Task 2

| Quantity | Value |
|---|---:|
| half-window actual collapse nu_opt | 1.4271 |
| half-window actual collapse 1/nu_opt | 0.7007 |
| half-window auxiliary slope vs logL | 0.1360 |
| heldout actual collapse 1/nu_opt | 0.7007 |
| full actual collapse 1/nu_opt | 0.5482 |
| exact 1/nu | 1.0000 |

| Class | auxiliary fixed CI | center | covers exact 1/nu | covers heldout actual 1/nu | covers full actual 1/nu |
|---|---|---:|---|---|---|
| loose_abs_u_le_1_omega_ge_1 | [-49.9206, 50.1927] | 0.1360 | yes | yes | yes |
| strict_abs_u_le_0.3_omega_ge_1 | [-0.5395, 0.8116] | 0.1360 | no | yes | yes |

Observable note: Exp52d extracts 1/nu by global PC1 collapse-quality minimization. It does not define a native affine per-L ladder; the Task-1 fixed estimator can only be applied to an auxiliary leading-slope ladder.

## Post-hoc Notes

- Single-session blinding is a discipline device, not information isolation.
- Task 1 uses the already-committed exp85c blind ladders for the 7-point no-regression check; fixed predictions were written and committed before truth configs were loaded for exp86 scoring.
- Task 2 uses the actual exp52d collapse-quality minimizer for the Ising anchor; the fixed affine CI is reported on the auxiliary leading-slope ladder because the source estimator is global, not a native per-L affine observable.

## What We Did Not Do

- No entry was added to `CLAIMS_REGISTER.md`.
- Tasks 3 and 4 are pending.

## Anomalies And Bugs

- Exp52d extracts 1/nu by global PC1 collapse-quality minimization. It does not define a native affine per-L ladder; the Task-1 fixed estimator can only be applied to an auxiliary leading-slope ladder.
- Tasks 3-4 are not yet run in this report snapshot.
