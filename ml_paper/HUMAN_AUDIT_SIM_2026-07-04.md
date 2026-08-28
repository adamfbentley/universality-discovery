# Simulated expert audit — the FSS minimax resolution floor

**Standing in for the human expert review requested in `AUDIT_BRIEF.md` (gate one),
which is not yet available. Reviewer stance: a statistician in the
nonparametric-minimax / estimation-theory tradition, reading adversarially.**

**What makes this more than a second correlated AI pass:** the headline numbers
were *independently recomputed from the theory-note formulas alone*, in fresh
code that imports none of the exp77/79/83 pipeline
(`ml_paper/experiments/audit_2026-07-04_numeric_checks.py`). Arithmetic is not
opinion and does not share the generator's blind spots. The *judgment* portions
(§2–§8) remain AI judgment and still warrant a human, especially §8.

Read alongside the prior AI pass (`SIMULATED_AUDIT.md`); this audit confirms most
of it, sharpens Q1 and Q5, and disagrees with it in one place (§8, the priority
risk it under-weighted).

---

## 0. Verdict

The construction is sound and the impossibility direction is valid as stated for
the Gaussian model. I **independently reproduced** the confusion gap
(D²(0.10) ≈ 1.6×10⁻⁷ vs the note's 1.3×10⁻⁷ — both upper bounds by construction,
consistent), the per-system floors (BD 0.27, EW/Eden ≈ 0.35–0.44, KPZ 0.44,
m=24), the monotonicity of D² in Δα, and the closed-form modulus constant
(c₁ → 0.0375 with fitted exponent 2.010 vs theory N+1=2). The Donoho / Armstrong–
Kolesár reframing is correct and makes the paper stronger. The genuinely new
contributions survive. Four residual risks are load-bearing and none are fatal:
(1) Gaussianity is the one idealization that cuts *against* the bound; (2) the
tail-index / second-order-regular-variation precedent is closer than the note
credits and is the top priority-attack surface; (3) the correct deliverable is
*exact honest CIs*, which depend on exp84 (does not yet exist); (4) the BD
half-window contradiction (exp85b) is unresolved and BD is the flagship empirical
claim.

## 1. Independent numerical replication (the part that isn't opinion)

Fresh reimplementation from the `AUDIT_BRIEF §1` / theory-note formulas:

| Quantity | Note / claim | This audit (independent) | Verdict |
|---|---|---|---|
| D²(0.10), real 7-pt design | ≈1.3×10⁻⁷ | 1.61×10⁻⁷ | ✔ consistent (both ctor upper bounds) |
| BD floor, m=24 | 0.27 | ≥0.27 (grid) | ✔ |
| EW / Eden floor, m=24 | ≈0.44 | 0.35–0.44 (grid) | ✔ |
| KPZ floor, m=24 | ≈0.44 | 0.44 | ✔ |
| D²(Δα) monotone | assumed (running-max) | monotone on [0.05,0.55] | ✔ assumption holds |
| Seeds to resolve Δα=0.1 (EW σ≈0.145) | ~1.6×10⁵ | σ²/D² ≈ 1.3×10⁵ | ✔ order-of-magnitude |
| Modulus constant c₁ (ΔαT/U→0) | 0.0375 exact | 0.0374–0.0375 | ✔ |
| Modulus exponent N+1 | 2 | 2.010 (fit) | ✔ |

The near-non-identifiability headline is real and reproducible. The
"optimizer-under-convergence is safe" argument (computed D² ≥ true D² ⇒ computed
floor ≤ true floor ⇒ impossibility conservative) is correct and should be stated
as a one-line lemma in the paper — it immunizes every floor number against the
multistart's quality.

## 2. Q1 — Robustness to the Gaussian model

This is the single place the idealization cuts *against* you, and the brief is
right to lead with it. The floor needs an **upper** bound on distinguishability;
Gaussian noise **minimizes** Fisher information at fixed variance, so real
(non-Gaussian) seed noise is generally *more* distinguishable, the true KL is
*larger*, TV is larger, and the two-point bound `sup error ≥ (Δα/2)(1−TV)` gets
*weaker*. So the Gaussian floor is **not automatically conservative** for
non-Gaussian noise — it can overstate impossibility. The κ diagnostic is
therefore load-bearing, not cosmetic.

The clean resolution (tighter than SIMULATED_AUDIT §2): because the adversarial
shifts are minuscule (D ~ 10⁻⁴…10⁻⁷), you are deep in the local-quadratic regime
where **every** f-divergence coincides to leading order — KL, Hellinger, χ² all
equal ½·I·Δμ² with `I` the Fisher information of the *seed-mean*. The entire
non-Gaussian correction collapses to the single scalar κ = I·σ̄². So:

- Report κ per system as a band and state the floor as Δα\*/√κ. (The note already
  does this; make it lead, not appear in a caveat.)
- κ → 1 as m grows: Fisher information is monotone toward Gaussian under
  convolution (Stam; Artstein–Ball–Barthe–Naor), and the seed-mean is an m-fold
  convolution. This is a theorem, not a hope. (SIMULATED_AUDIT verified the
  direction numerically: t(5)→1.25, CLT monotonicity 2.00→1.19→1.06→1.02; I did
  not re-run it — it is textbook.)
- **Yes, compute the Hellinger version the brief asks about** — but note *why* it
  is reassuring: since all divergences agree to leading order here, a Hellinger
  floor that matches the KL floor is positive evidence the local-quadratic
  approximation is valid, i.e. it certifies the approximation rather than
  changing the number. Cheap, and it retires the "is the quadratic KL legitimate"
  question outright.

Bottom line: robustness is *adequately* handled provided κ leads and a Hellinger
cross-check is shown. Do not state any floor without its κ band.

## 3. Q2 — Two-point vs. richer constructions

For a **scalar** (1-D) target functional, two-point/modulus is the right and
essentially sharp tool (Donoho–Liu; Ibragimov–Khasminskii): affine-minimax within
the small constant for convex classes. Fano/Assouad buy strengthening for
**multi-parameter or rate** problems (estimating a whole function, or several
exponents jointly) — not here. They would add nothing material.

The one richer object worth computing is **not** a stronger impossibility but the
**van Trees (Bayesian Cramér–Rao) bound under the exp76 sampling prior**: it
produces an *average-case* lower bound that matches the amortized estimator's
*average* RMSE, resolving the worst-case-floor-vs-average-RMSE category mismatch
flagged in exp81 anomaly 9 ("operates near the information limit" currently
compares an average risk to a worst-case floor). Caveat the project already
caught (exp83 anomaly 5): the **uniform** prior violates van Trees'
boundary/absolute-continuity conditions — recompute with a smooth bump (cosine²)
prior on the same support before quoting the ~1.7× number.

## 4. Q3 — The sharpness gap

The proposed attack (annihilator + Newman/Markov inequalities for exponential
sums, Borwein–Erdélyi restricted Müntz systems) is a legitimate route to a lower
bound on approximation by bounded exponential sums. But the Donoho mapping (§7)
makes it **largely unnecessary**: for the convexified class, Donoho / A–K supply
matching upper bounds, so the modulus itself is the sharp quantity up to known
constants (exactly, for CIs — see §7). Demoting the analytic lower bound from
blocking to enrichment (theory-note G4) is the correct call.

What *is* a clean, self-contained new result is the **closed form of the modulus
for bounded exponential-sum classes** — the c_N lemma. I confirmed c₁ = 0.0375 in
the ΔαT/U→0 limit and the exponent N+1 = 2. Two cautions for the write-up:

- State c₁ as **"construction-optimal at N=1, numerically certified to 0.0375,"**
  not "proven exact." My replication strengthens the claim but is still numerical;
  global optimality of the constant is not proven.
- For **N ≥ 2 the Richardson construction is provably loose** (14% at N=2, 40% at
  N=3 per Addendum 3), so c₂, c₃ are upper bounds only. The paper says this — keep
  it explicit; do not let c₂≈0.0216 read as exact.

The binomial-identity reduction to polynomial approximation of log(1/t) is the
right machinery and is the natural place to cite Müntz–Szász (the two 2026
preprints in `LITERATURE_AUDIT` Item 1 are genuinely adjacent here).

## 5. Q4 — Class-conditionality in practice

The five-tuple discipline is good; push the Manski framing harder. Report the
floor as a **breakdown frontier** — "the largest U (weakest correction
assumption) under which resolution δ is achievable" — rather than a single number
at a declared U. That is standard partial-identification language and is much
harder for a practitioner to misuse than a lone floor value. The default output
should be a *curve* (floor vs U), which the project already produces (exp85b floor
curves) — make it the headline object, not a sensitivity appendix.

The highest-value upgrade, already scoped (exp84/85c): the **A–K data-driven
lower bound on the smoothness constant** maps to a data-driven lower bound on U
(and possibly ω_min), converting "we declare U" into "the data force U ≥ U_min,
and here is the floor above that." That answers the inevitable referee question
"how would I choose U?" with a procedure instead of a declaration. Manski's "law
of decreasing credibility" is the right epigraph (already in the intro draft).

## 6. Q5 — Priority and the Donoho / Armstrong–Kolesár mapping

**The mapping is exact in structure.** Fixed design, Gaussian errors, a linear
functional of a mean vector constrained to a class — this is Donoho (1994)
optimal recovery / A–K (2018) verbatim. The confusion-gap minimization *is* the
least-favorable-pair / modulus computation (A–K Eq. 2, including the free
intercept). I concur with the identification.

Two consequences, one of which corrects the earlier AI pass:

- **(a) Sharpness transfer.** The ≈1.25 affine-vs-minimax factor is an
  **MSE-estimation** constant and does **not** belong in the CI story. For
  confidence intervals — the object you actually want — the A–K construction is
  **exact, with no unknown constant** (their Thm 3.1 / §3.4), computed from the
  same modulus the hull code already evaluates. So the right deliverable is
  *exact honest CIs for α under declared corrections* ("FSSHonest"), not a
  1.5–2× bracket. This supersedes EXP83_AUDIT's sharpness bracket and matches
  SIMULATED_AUDIT §8a — I endorse that correction. **But note the dependency:**
  this deliverable is exp84, which does **not exist in the repo yet** (prompt
  only). Until it is computed, the paper cannot state the exact-CI result.
- **(b) Centrosymmetry.** A–K's *non-adaptation* efficiency bounds require the
  class convex **and centrosymmetric**; the correction class (u ∈ [−0.75, 4],
  asymmetric, curved) is neither. So the constructive minimax-CI machinery
  transfers (convex hull suffices), but the adaptation-impossibility numbers do
  **not** transfer verbatim. Correct as recorded; keep it flagged.

Convexity: the class is a 2-parameter curved (nonconvex) manifold, so all
sharpness statements are **relative to the convex hull**; hull ⊇ class ⇒ hull
floor ≥ class floor ⇒ conservative, same direction as every other approximation.
The size of the hull-vs-class gap is an empirical question (exp83 task 2) and
should be reported, not assumed small.

## 7. The Donoho mapping — confirmed, with the identifiability nuance

One subtlety worth stating precisely (theory-note G2 gets it right): α is a
functional of the *parameters*, and is well-defined on the mean-image M only if
no two parameter settings with different α produce the same μ. ω(0)=0 (no exact
mimicry on the bounded class) holds, so the functional is well-defined; the whole
result is that ω(ε) is *enormous at ε = σ/√m* despite ω(0)=0. That is the honest
statement of "near-non-identifiability" — not non-identifiability. Keep the
distinction crisp; a minimax referee will check exactly this.

## 8. Where I disagree with the earlier AI pass — the under-weighted priority risk

`SIMULATED_AUDIT` lists Hall–Welsh (1984) and Drees among the anchors but treats
the priority risk as essentially discharged by the Donoho reframe. I think that
**under-weights the nearest precedent.** Tail-index estimation under
**second-order regular variation** is, stripped of vocabulary, the *same problem*:
a leading power law plus a bounded correction of unknown amplitude/rate, with the
exponent as the target, and a known minimax theory (Hall–Welsh; Drees; Cheng–Peng)
for exactly how the second-order term caps the achievable rate. A referee from
extreme-value statistics will see FSS-corrections-to-scaling as a change of
clothes and will ask why this is new. The defensible distinctions are real —
**fixed small design + a declared parametric correction *class* + finite-sample
computation**, versus asymptotic second-order-RV *conditions* and rate statements —
but they must be argued in a dedicated paragraph, not a citation. As currently
framed this is the **most likely "you missed prior art" attack**, ahead of the
Donoho point (which the paper now handles well). Recommend: read Hall–Welsh (1984)
and Drees (1998/2001) directly and write the "why FSS ≠ tail-index" paragraph
before submission.

## 9. What I did NOT verify (residual need for a human)

- The van Trees ~1.7× number (needs the smooth-prior recomputation; exp84).
- The exact A–K honest-CI half-lengths (exp84 does not exist yet; I verified the
  modulus that feeds them, not the CI construction).
- Donoho's exact constant conventions (MSE vs testing separation) — a primary-
  source point where I am relying on A–K's statements, same limitation as
  SIMULATED_AUDIT §8a; a human should confirm at the source.
- The empirical BD claim: exp85b's blind half-window CI at the honest class does
  not cover α=0.5 (audit ledger F2). This is an *empirical* contradiction, not a
  flaw in the floor theory, but BD is the flagship recovery — resolve exp85c
  Task 2 before BD appears in any abstract.
- I remain an AI reviewer. The replication in §1 is independent arithmetic; §2–§8
  are judgment and can share blind spots with the generator. A human statistician
  is still the gate — most valuably on §8 (tail-index positioning) and the Donoho
  constant conventions.

## 10. Bottom line and actions

The theorem is correct, conservative in the right direction (modulo the
Gaussianity caveat), independently reproduced, and correctly repositioned onto a
30-year-old foundation. It is a good paper. To de-risk it:

1. Lead every floor with its κ band; add the Hellinger cross-check (Q1).
2. Write the "why FSS ≠ tail-index / second-order RV" paragraph — top priority
   risk (§8).
3. Make exact honest CIs (not the 1.25 bracket) the deliverable — blocked on
   exp84, which must actually be run (§6a, §9).
4. Report floors as breakdown frontiers over U, with the data-driven U lower
   bound (Q4).
5. Recompute van Trees with a smooth prior before quoting ~1.7× (Q2).
6. Resolve the BD half-window (exp85c Task 2) before any BD headline (§9).
7. State the "under-convergence is safe for impossibility" lemma explicitly (§1).

None of these is fatal; all are addressable within the existing Tier-0 plan.
