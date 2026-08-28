# Audit — most recent 10 items (2026-07-02 → 07-04)

**Auditor pass 2026-07-04. Reviewer role only (no executor authority; no gate
relabels). Scope: the July 2–4 arc — six experiments (exp82, exp81, exp81f,
exp83, exp85, exp85b) and four doc updates (adversarial/simulated audit + audit
brief; theory Appendix G + A–K primary-source pass; v2 manuscript outline +
Paper 2 intro; DIRECTION freeze + register/log updates).**

Evidentiary authority: the gate ledgers and result JSONs. Interpretive
authority: DIRECTION_2026-07-04.md. This file records findings and recommended
actions; the reconciliation edits it triggered are listed in §5.

---

## 0. Verdict in three sentences

The verification *process* is in excellent health — blinding, invariant gates,
executor/reviewer separation, and error ledgers that implicate the auditors
themselves are all catching real bugs. The *scientific* state is more
precarious than the manuscript activity implies: the paper's positioning pivoted
(Donoho reframe) on a self-audit the human expert has not checked, calibration
has failed two preregistered gates with zero clean passes, the only external
anchor (Ising) is still blocked, and the claims register lagged the retractions
until this pass. None fatal; all tracked; the manuscript is running ahead of its
load-bearing inputs.

## 1. Per-item status

| Item | Date | Status | Flag |
|---|---|---|---|
| exp82 — null-partition ARI (Paper 1) | 07-03 | Accepted; verification only | low |
| exp81 — observable hierarchy | 07-02 | Gates pass; headline ratios **superseded by 81f** | stale numbers in report (bannered this pass) |
| exp81f — fix-pass | 07-03 | ACCEPTED, arc CLOSED, countersigned | clean |
| exp83 — audit-response | 07-03 | ACCEPTED; ≈1.25 bracket self-corrected same day | resolved |
| exp85 — preregistered validation | 07-04 | **G-85c calibration FAILED** (overcoverage); phase 1 blocked | live |
| exp85b — corrected A–K validation | 07-04 | A/B/C pass; **G-85b-D FAILED** (overcoverage); Ising not run | live |
| SIMULATED_AUDIT + AUDIT_BRIEF | 07-03 | Donoho reframe; **human audit still pending** | positioning risk |
| Theory Appendix G + A–K read | 07-03 | Exact-CI machinery; sound | good |
| Manuscript v2 + Paper 2 intro | 07-03 | Draft; ≥3 inputs unfinished | ahead of evidence |
| DIRECTION_2026-07-04 | 07-04 | Freeze + tiers + kill criteria | disciplined |

## 2. Material findings

**F1 — Calibration is the live wound; two failures, both overcoverage.**
exp85 G-85c failed (U0.5/U1 coverage 1.000 vs preregistered [0.919, 0.981]) and
blocked the blind phase before predictions were written. exp85b re-ran corrected
and G-85b-D failed again (in-class coverage 1.000 vs [0.910, 0.990]). Saving
grace, honestly stated in both reports: the failures are **over**coverage, so CI
*validity* holds; what is unproven is calibration/efficiency. exp85 also surfaced
a real bug en route — raw extremal-pair weights did not define a normalized α
estimator (U1 undercovered 0.830 before the affine-normalization fix), so the
estimator construction was wrong until 07-04. Watch: exp85b's A–K center at U1
still shows sub-unit log-L response (0.3874 vs GLS 1.0000), consistent with a
lingering normalization asymmetry and with 81f's unexplained U=0.5/U=1 wrinkle.
Correct posture (per DIRECTION): "honest/conservative CIs," never "calibrated,"
until exp85c Task 1 lands a clean pass.

**F2 — BD real-data half-window contradicts the BD headline.** exp85b: the blind
BD half-window CI at the honest class (U=0.5) does **not** cover α=0.5 or the
exp76 full-window value; only the loose U=4 class covers. This is the queued
bug-vs-mechanism item (exp85c Task 2). Until diagnosed, BD — the flagship
recovery (exp76 α̂≈0.522) — is the most exposed claim in the program. Consistent
with DIRECTION already retiring claim B's strong form; exp85b is positive
evidence it should stay retired.

**F3 — Ising, the only external anchor, still hasn't run.** Confirmed this pass:
`results_exp52d_full/` holds only `results.json` (624 B) + a PNG — no per-L
collapse ladder. The only per-seed ladder on disk is exp76's surface-growth
`wsat_perseed.csv`. Ising is a Tier-0 requirement *and* a kill-criterion input;
every floor number is currently self-generated on the same fractional-EW
machinery. exp85c Task 4 regenerates the minimal Ising ladder (L∈{32,48,64,96},
m=24) — this is data plumbing, not new science, and is the highest-value
missing piece.

**F4 — The Donoho reframe is right and honest — but rests on a self-audit.**
SIMULATED_AUDIT correctly identifies the floor as an instance of Donoho (1994) /
Armstrong–Kolesár (2018) linear-functional minimax theory, narrows priority, and
re-delineates the genuinely new parts (FSS application; closed-form
exponential-sum modulus; RG dictionary; amortized pairing). Exposure: the v2
outline and Paper 2 intro already commit hard to this framing, but the mapping
was verified by an AI audit plus a ~70%-of-text read of A–K. The human expert
audit (AUDIT_BRIEF gate one) has not happened. Do not finalize the abstract's
positioning on the self-audit alone.

**F5 — Claims register lagged the retractions (fixed this pass).** CLAIMS_REGISTER
claim J still read "first minimax lower bound on FSS," and claim B still presented
BD recovery, both contradicted by DIRECTION_2026-07-04. Reconciled this pass (§5);
gate results untouched, original text struck-not-deleted for the ledger.

**F6 — exp81's superseded numbers lived on in its report (fixed this pass).**
EXP81_REPORT §4 still printed 11.8× and "at quadrature," demoted by 81f (honest
H-A3 ~1.4× over best channel; U=4 "ratios" are ≈1). Bannered this pass at the top
of the report and inline at §4.

**F7 — Manuscript ahead of evidence.** Confirmed exp84 is **absent** from this
checkout (only `SONNET_EXP84_PROMPT.md`; no code, no `results_exp84*`). The v2
outline's "pre-submission blockers (all small)" understates reality: exp84
un-run; calibration failed twice; Ising blocked. The intro draft is correctly
marked draft with [bracketed] exp84/van-Trees placeholders — keep it non-final
until those exist.

## 3. Process health (the standout)

Genuinely strong; worth the planned methods section. Invariant gates caught real
bugs across the arc: nesting invariant → hull warm-start bug (exp83) and
private-nests-shared under-convergence (exp81); Stam inequality → KDE
self-inclusion (exp83); DPI gate → hierarchy discipline (81f). Blinding executed
properly in exp85b (phase-1 hash committed before phase-2 scoring; both SHAs on
record). Error ledger implicates the reviewers: the wrong quadrature null traces
to the *plan*, not the executor; and the exp83 audit records that abfeb32/822dd05
silently missed their edits (mount unsynced at `git add`), yielding the rule
"verify the committed blob, not the working tree" — added this pass to
EXP76_HANDOFF sandbox gotchas.

## 4. Recommended actions (priority order)

1. **[DOC — done this pass]** Reconcile CLAIMS_REGISTER claims J and B with the
   Donoho retraction. See §5.
2. **[EXPERIMENT — exp85c Task 2]** Diagnose the BD half-window before BD appears
   in any abstract; it currently fails its own honest-class CI.
3. **[EXPERIMENT — exp85c Task 4]** Unblock Ising by regenerating the exp52d
   per-L ladder; only out-of-family validation and a kill criterion.
4. **[EXPERIMENT — exp85c Task 1]** Get one clean calibration pass; until then,
   never write "calibrated."
5. **[PROCESS]** Gate the manuscript's Donoho positioning on the human audit
   reply, not the self-audit.
6. **[DOC — done this pass]** Banner superseded exp81 numbers; confirm exp84
   exists before the outline leans on its bracketed values (confirmed **absent**).

Actions 2–4 are already fully scoped in `GPT_EXP85C_PROMPT.md` and require a
fresh **executor** session with committed blinding — deliberately NOT run inside
this reviewer/audit session (executor/reviewer separation).

## 5. Reconciliation edits made in this pass (documentation only)

- `CLAIMS_REGISTER.md` claim **J**: added RECONCILED banner retiring the "first
  minimax bound for FSS" priority claim (authority: DIRECTION_2026-07-04,
  SIMULATED_AUDIT §1, LITERATURE_AUDIT Items 2/9); struck the priority sentence
  and the stale prior-art-pass framing; original kept for the ledger.
- `CLAIMS_REGISTER.md` claim **B**: added RECONCILED banner retiring the strong
  "recovers BD α" form and recording the exp85b half-window contradiction;
  exp76 numbers retained but flagged non-citable as "recovery" pending exp85c
  Task 2.
- `EXP81_REPORT.md`: top-of-file SUPERSEDED banner + inline §4 note demoting
  11.8× / "at quadrature"; points to EXP81F_AUDIT.md for citable numbers.
- `EXP76_HANDOFF.md`: added the commit-blob-verification gotcha.

No gate result was relabeled; no failure was amended. All changes are additive
reconciliation to the already-decided DIRECTION_2026-07-04 retractions.

---

## 6. exp85c outcome (2026-07-06 update; reviewer pass)

exp85c ran (blinding intact: phase-1 `4f55069`, phase-2 `0f24f4b`). It resolves
the three live findings, and surfaces one new, decisive bug. Independent checks
by this reviewer noted inline.

**F1 (calibration) → RESOLVED, favourably.** G-85c-1 met. The exp85/85b
overcoverage (1.000) is now shown to be **analytically expected**: Task 1a's
per-config prediction gives mean coverage 0.9973 (U0.5) / 0.9995 (U1), and the
observed 1.000 sits inside the band. A fresh-seed confirmatory round (seed 85300,
n=100) reproduced in-class coverage 1.000 with the validity gate met (≥0.906).
Reading: the honest CIs are **valid and provably conservative**, not miscalibrated.
"Two failed gates, zero clean passes" is superseded — say "valid, analytically
conservative," never "tightly calibrated."

**F2 (BD half-window) → DIAGNOSED as a bug, not a mechanism.** G-85c-2 verdict:
**bug**. The exp85b 4-point affine center is not amplitude-invariant — its weight
vector sums to −0.9986 (must be 0) and gives response −2.85 to log L (must be 1),
so it is dominated by the additive log-amplitude, not the α slope. *Independently
confirmed by this reviewer* from the reported weights [2.331, −3.648, −4.282,
4.601] on design {32,48,64,96}. Consequence: the scary "BD half-window does not
cover α=0.5" is an **estimator-construction artifact on the short design**, NOT
evidence against BD. The exp76 **full-window** BD result is not impugned by it.
But the half-window CI pipeline is broken and no half-window BD claim may be made
until the weights are constrained (Σw=0, Σw·x=1) and it is re-run. The U=4
"success" was illusory too (Task 2: analytic coverage over plausible BD truths
has median 0, share≥0.95 only 0.127).

**F3 (Ising anchor) → UNBLOCKED but NOT YET ACHIEVED.** G-85c-4: the per-L ladder
was regenerated (pilot 10.4s; full ~1000s < 1800s guard), clearing the data
blocker. But the result does not yet validate the framework: (i) exp52d stored
only a summary + PNG, so the observable is a **reconstructed PC1-vs-temperature
slope proxy**, not exp52d's actual 1/ν observable; (ii) the strict-class CI
[−0.461, 0.694] covers the empirical slopes (0.110, 0.166) but **not** the exact
1/ν = 1.000 — and the empirical slopes themselves are nowhere near 1.0, so the
proxy does not track ν; (iii) the loose-class CI blew up to ~[−820000, −763000],
the **same amplitude-invariance bug** as F2 on this short design. Honest status:
Ising attempted, plumbing solved, **external anchor still not satisfied** — the
DIRECTION Tier-0 "Ising external anchor" and its kill-criterion remain OPEN.

**NEW FINDING (N1) — one bug runs through all four short-design failures.** The
non-amplitude-invariant affine center (Σw≠0) is the common cause of: exp85
raw-weight undercoverage (0.830), exp85b's sub-unit A-K log-L response at U1
(0.3874), the BD half-window artifact (F2), and the Ising loose-class blowup
(F3). It is localized to the **short/4-point-design CI center**; the 7-point
confusion-gap floors, the identified sets, and the main in-class coverage use
different code paths and are unaffected. **Highest-priority fix:** enforce the two
linear constraints on the affine weights (annihilate the constant, unit slope
response) and re-run every 4-point half-window analysis (BD + Ising).

**Minor (Task 3).** Calibrated profile widths meet the 25%-of-Wilks gate for BD
but **not Eden** (68-width 0.175 cal vs 0.245 Wilks); the exp85b/c identified-set
table needs a calibrated column for Eden. Truth-pinned fits are all
non-rejected (χ²/dof < 1, p > 0.05) — the pinned configs are representative.

**Net.** exp85c is a clean, honest confirmatory pass: it rescued the calibration
story (conservative-valid, analytically backed), converted the BD "contradiction"
into a diagnosed and re-fixable estimator bug (good for BD full-window), and
showed the Ising anchor is not yet real. Two things gate the manuscript now that
did not before: **(a)** fix the short-design affine center (N1) and re-run BD +
Ising half-windows; **(b)** obtain a genuine Ising anchor on exp52d's real
observable, since the proxy does not recover ν. Neither is a theory problem; both
are on the Tier-0 path.

*Recommended (for Adam to approve, not edited here):* update
DIRECTION_2026-07-04's "Pending" list — calibration is no longer pending (F1
landed); BD half-window is diagnosed (bug, F2); Ising remains pending with the
proxy caveat (F3/N1).
