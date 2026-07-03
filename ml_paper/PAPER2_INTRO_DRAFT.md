# Paper 2 — §1 Introduction (draft v1, 2026-07-03)

Status: prose draft for the identifiability paper, post-Donoho framing.
Numbers in [brackets] are placeholders pending exp84 / human-audit reply.

---

Finite-size scaling (FSS) is the standard instrument for extracting
asymptotic critical exponents from simulations of finite systems: measure an
observable across a ladder of sizes, fit a power law, correct for finite-size
effects, extrapolate. The corrections are the difficulty. Every practitioner
knows that corrections to scaling can masquerade as shifts in the leading
exponent, that the fitted exponent can depend alarmingly on the assumed
correction form, and that the cure is "larger systems" rather than "more
statistics." What has been missing is the quantitative form of this
knowledge: how finely can the asymptotic exponent be resolved, in principle,
from a given window, at a given noise level, under a given assumption about
corrections — by any method of analysis whatsoever?

This paper answers that question exactly, by recognizing it as a problem
statistics solved in general form three decades ago. Written on the
logarithmic size axis, an FSS ladder is a fixed-design regression whose mean
vector is an affine function of the target exponent plus a nuisance drawn
from a bounded class of decaying exponentials. The exponent is therefore a
linear functional of the mean vector constrained to a known class — precisely
the setting of the minimax linear-functional theory of Ibragimov and
Khas'minskii and Donoho [refs], in the finite-sample form developed for
econometric regression-discontinuity inference by Armstrong and Kolesár
[ref]. In that theory, everything of interest is governed by a single
computable object, the modulus of continuity of the functional over the
class: the minimax resolution (no estimator can distinguish exponents closer
than the modulus at the noise scale), the exact form of optimal estimators,
and finite-sample honest confidence intervals. None of this machinery is
ours. What is new here is the identification, its consequences for
computational statistical mechanics, and the closed-form theory of the
modulus for the specific nuisance class that scaling corrections form.

We make six contributions. **First**, the identification itself, with its
renormalization-group dictionary: the correction class's spectral edge is
the smallest irrelevant RG eigenvalue, its amplitude bound is the
irrelevant-operator amplitude, and the degeneracy that makes extrapolation
ill-posed — small-ω corrections are exponent shifts up to Taylor remainder —
is the statistical face of marginal operators. The same normal form covers
growth exponents, correlation-length fits, lattice-correlator energy
extraction, and the fitting of neural-network scaling laws. **Second**, a
closed form for the modulus over bounded N-term exponential-sum classes,
E_N = c_N √T U (Δα T/U)^{N+1}, with the leading constant exact
(c₁ = 0.0375), higher constants numerically certified, and a reduction — via
a binomial identity — to classical polynomial approximation of the
logarithm. The resulting identifiability hierarchy (power-law floor at fixed
sparsity; exponential floor at free sparsity; vacuity without amplitude
bounds) says that exponent measurement is possible exactly insofar as the
irrelevant spectrum visible in the window is sparse and bounded. **Third**,
computed floors for the standard 1+1-dimensional surface-growth benchmark at
real designs and measured noise: at L ≤ 256, no estimator can resolve
exponent differences of 0.27–0.44 against adversarial standard corrections;
the floor is nearly flat in replicate count; and a van Trees computation
shows that marginalizing unknown corrections destroys 98.3% of the Fisher
information about the exponent — a factor of 59, set by the design alone.
These numbers explain, quantitatively and retroactively, a large empirical
literature of failed and scattered exponent recovery, including our own
[Paper 1 ref]. **Fourth**, an observable-hierarchy law, computed on an
exactly solvable Gaussian testbed where every level of observation has
closed-form information and the data-processing inequality serves as a
validation gate: observing the full spectrum rather than a scalar summary
buys a 6–10× finer floor when correction amplitudes are tightly constrained,
and essentially nothing when they are not — at loose amplitude bounds the
exponent is unidentifiable from any observable at this design. Data richness
and correction knowledge are complements, not substitutes. **Fifth**,
inference that achieves the limits: an amortized estimator trained on a
declared prior over correction families recovers the ballistic-deposition
roughness exponent (α̂ ≈ 0.50 ± 0.05 syst ± 0.03 stat) where classical
ansatz fitting scatters over 0.36–0.70, operates within a factor [~1.7] of
its Bayes bound, and — via the Armstrong–Kolesár construction applied to the
computed modulus — finite-sample honest confidence intervals [exp84].
**Sixth**, a reporting standard: every floor and interval in this paper
carries its full conditioning (observable, correction class, design, noise
source, replicates), and we give a data-driven lower bound on the correction
amplitude that converts the class declaration from an arbitrary choice into
a testable one [exp84].

Two methodological notes. All results are class-conditional by construction:
a floor is a statement about a declared set of possible corrections, and we
regard the explicitness of that conditioning not as a limitation but as the
content — it is the quantitative form of Manski's law of decreasing
credibility, and the analysis prices every strengthening of assumptions.
Second, the results were produced under an adversarial verification
protocol — independent replication of headline numbers from formulas alone,
mathematical invariants (nesting, monotonicity, data-processing) enforced as
automated gates, and staged audits that caught and corrected errors at every
layer including the authors' own; the gate ledgers are included as
supplementary material. We believe this protocol is of independent interest
for AI-assisted computational research.

The paper is organized as follows. Section 2 states the normal form and the
mapping to the linear-functional theory, including its scope conditions.
Section 3 computes the floors and their robustness for the surface-growth
benchmark. Section 4 develops the closed-form modulus. Section 5 presents
the observable-hierarchy law. Section 6 presents estimation at the limit.
Section 7 transfers the framework (temporal exponents, Ising ν, lattice
correlators, neural scaling laws). Section 8 discusses limitations — the
Gaussian noise model and its measured κ diagnostic, the single testbed for
the hierarchy law — and open problems: the analytic lower bound on the
modulus constants, optimal node placement, adaptation in nonconvex
correction classes, and non-Gaussian raw-configuration inference.
