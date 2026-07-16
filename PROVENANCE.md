# Provenance and AI Assistance

This is self-directed research conducted without academic supervision. AI
systems were used as development infrastructure; the work should therefore be
described as **self-directed and AI-assisted**, not as unaided authorship.

## Division of labour

The repository owner selected the research questions, simulation systems,
evaluation criteria, and acceptance gates; decided which experiments to retain;
and is responsible for the interpretation and every public claim.

AI systems assisted with:

- drafting and revising experiment code;
- implementing experiment plans and running scripted gates;
- proposing diagnostics and adversarial checks;
- reviewing mathematical arguments and source code;
- editing reports, theory notes, and repository documentation.

Experiments 81, 83, and 84 used explicit Claude Sonnet execution prompts. Those
prompts and AI-oriented audit records are retained under
`archive/ai_execution/` for transparency. They are process records, not
scientific evidence.

Commit authorship does not establish where an idea originated, and an AI audit
is not an independent review. The evidence for a claim must be a derivation,
test, stored result, or external source that can be checked without relying on
the model's authority.

## Reproduction and validation

The main numerical and analytical checks are:

- smoke tests in `tests/` and deterministic experiment entry points;
- machine-readable results stored with the corresponding experiment;
- synthetic recovery and exact-model controls where the ground truth is known;
- direct multistart optimization for the exp77 two-point construction;
- referee-style sensitivity checks in exp78;
- numerical coefficient certificates and residual-identity checks in exp79;
- exact fractional-EW and data-processing checks in exp81.

The Gaussian mean-shift KL calculation, Pinsker step, and Le Cam two-point risk
bound are written out in `ml_paper/THEORY_minimax_floor.md` and can be
recomputed by `experiments/77_minimax_floor.py`. The repository currently treats
the exp79 higher-order approximation result as numerically supported where an
analytic proof is not present. Literature-priority claims and extensions beyond
the declared observation model remain provisional.

This file records repository-verifiable checks; it does not claim that every
derivation was first produced without AI help. Before external submission, the
owner should independently re-derive each headline mathematical step and record
that sign-off here.

## Claim boundary

The minimax result is conditional on a specified Gaussian observation model,
finite-size design, `W_sat` summary observable, and bounded correction class.
For an admissible two-point construction separated by `Δα`, the condition
`D²(Δα) ≤ σ²/m` yields a lower bound of `Δα/4` on worst-case expected
absolute error over that pair. It does not prove that every smaller exponent
difference is indistinguishable, and it does not prove that earlier clustering
failures using richer features were inevitable.
