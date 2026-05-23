In my view, the **ML side has the higher-value publishable path**, while the **physics side has the deeper but much harder path**.

**ML Side: Higher Probability, Clearer Contribution**

The ML-side contribution is already visible:

> Unsupervised clustering can identify finite-data geometry while failing to recover the intended physical equivalence relation.

That is valuable because it generalizes beyond KPZ. It speaks to:
- representation learning,
- scientific clustering,
- benchmark design,
- false positives in unsupervised discovery,
- local prediction vs global concept recovery,
- invariance/quotient learning.

This path is publishable if framed as a cautionary benchmark. It does not need to discover new KPZ physics. It needs to convincingly show that a realistic scientific ML workflow can fool itself.

The deeper ML question is:

> When does a learned representation factor through the physical quotient?

That is genuinely interesting. Universality classes are quotient labels. Most clustering assumes cluster labels. Your work shows those are not the same.

A strong ML paper could be titled around:
- finite-size universality mirages,
- local separability vs global physical discovery,
- clustering failure under quotient labels,
- protocol-dependent false positives in unsupervised scientific ML.

That is the path I would prioritize.

**Physics Side: Deeper But Harder**

The physics-side path would ask:

> Can we construct observables or distances that actually approximate the RG quotient for finite KPZ-class data?

That is deeper, but much harder. It would require:
- true Family-Vicsek collapse-residual metrics,
- larger L and t,
- correction-to-scaling fits,
- clean separation of EW/KPZ/BD/Eden/RD/KS,
- maybe abandoning KS or treating it as a crossover system,
- serious finite-size scaling rather than clustering.

This could become a meaningful statistical-physics paper, but only if you produce a robust positive or a very clean finite-size scaling analysis. Current results are not there yet.

The deeper physics question is:

> How far along the RG trajectory must a system be before finite observations become quotient-compatible?

That is interesting. But answering it well is expensive and would need more controlled simulations.

**Which Should You Pursue?**

I would pursue the ML side as the main paper, with enough physics rigor to make it credible.

The main thesis:

> Physical discovery by clustering requires representations that respect the physical equivalence relation. In surface-growth universality, finite-size feature and exponent representations do not; they encode transient RG trajectory geometry instead.

Then keep the physics side as the explanatory backbone.

**Best Combined Path**

The strongest version is hybrid:

- ML claim: clustering finds representation geometry, not physics.
- Physics mechanism: finite-size KPZ observables do not approximate the RG quotient.
- Empirical demonstration: feature-space ceiling, high kNN/low clustering, exp69 false positive, exp71 collapse of advantage.
- Positive controls: Ising/Potts show method can work when quotient geometry is accessible.

That combination is more compelling than either alone.

**Deeper Future Paths**

ML deeper path:
1. Formalize “quotient-compatible representation.”
2. Use diagnostics: local separability, cluster compatibility, invariance under nuisance directions.
3. Develop benchmark tests for false physical discovery.

Physics deeper path:
1. Build true collapse-residual distances.
2. Study convergence of finite feature geometry under coarse-graining.
3. Quantify when BD/Eden approach continuum KPZ in representation space.
4. Treat universality recovery as estimating RG basins from finite trajectories.

**My Recommendation**

Main submission: MLST-style cautionary benchmark.

Follow-up, if energy remains: physics-heavy collapse-residual/RG trajectory paper.

If you try to make the current work a positive physics paper, it will be vulnerable. If you make it a rigorous ML-for-physics failure-mode paper, it has a much clearer reason to exist.

## Status Update: 2026-05-24

The ML-paper track has now implemented the first layer of controls needed to
make that failure-mode story scientific rather than rhetorical:

- MLP-05 separates local signal from global clustering: exp62 has strong
  nearest-neighbor universality purity but poor global quotient recovery.
- MLP-08 shows the exp62 failure is not rescued by a standard clusterer swap or
  a simple centroid hierarchy.
- MLP-09 reruns exp70 with archived feature matrices and performs true hard-subset
  refits. EW/KPZ binary separation is easy for feature baselines, but the
  continuum-KPZ/BD/Eden quotient is not recovered; matched effective exponents
  help only moderately and remain protocol-sensitive.

The current best paper path is therefore:

> finite-size scientific representations can contain real local physical signal
> while failing to factor through the intended physical quotient.

The next decisive control, if the project continues beyond the current paper
draft, is a stronger learned or explicitly invariant quotient baseline rather
than another ordinary clustering rerun.
