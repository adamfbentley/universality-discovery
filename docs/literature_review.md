# Literature Review — Universality Discovery via Local Observables, PCA, and RG Diagnostics

**Last updated**: February 11, 2026

## Executive Summary (Project-Relevant)

This project sits at the intersection of:
- **Renormalization group (RG)** and **finite-size scaling (FSS)** (how universal behavior emerges under coarse-graining).
- **Exactly/rigorously characterized universality** in 1D KPZ (fixed point/subclasses; stationary measures).
- **Unsupervised representation learning** (PCA/embeddings) applied to **ensembles** of configurations, not single samples.

What the current literature strongly supports:
- **Scale-invariant does not imply universal**: “scale-free” observables can still encode **non-universal amplitudes** and **corrections-to-scaling** that prevent cross-system convergence at accessible sizes/times. (Wegner 1972; Privman & Fisher 1984; Cardy 1996)
- **PCA can find phase/critical structure** in benchmark lattice models (Ising/XY/Potts), but *what PCA learns* depends on the representation and symmetry sector. (Wang 2016; Hu–Singh–Scalettar 2017; Wetzel 2017)
- There is **prior art** for combining PCA-like representations with **finite-size scaling** to estimate critical exponents in Ising-like settings. This means our novelty must be positioned carefully (local-observable charts, cross-system transfer, and “method boundary”/failure analysis). (Yue, Wang & Lyu 2022)

What our project adds (as framed against the literature):
- A **local-observable chart** perspective (work with Φ(config) in low dimension, then compare distributions/flows).
- A **diagnostic gate / failure-mode** methodology (system-vs-system invariance, identifiability checks) that is rarely made explicit in ML-for-physics papers.
- A **boundary case**: Potts demonstrates that “PCA + FSS” can fail due to objective/representation identifiability and symmetry/gauge issues even when standard observables succeed.
- A clean KPZ-specific theoretical anchor: the **1D KPZ stationary slope measure is Gaussian and λ-independent**, explaining why variance-type local features align with a noise-to-diffusion coordinate.

---

## 1. Universality, Scaling Fields, and Finite-Size Scaling (FSS)

### 1.1 Scaling Fields vs “Scale-Free Features”

RG emphasizes that “good coordinates” near a fixed point are **scaling fields** (eigen-directions of linearized RG). Universal critical exponents describe how these scaling fields transform under coarse-graining.

However, the objects measured in numerics are typically contaminated by:
- **Analytic/background terms** (non-singular contributions).
- **Corrections to scaling** governed by irrelevant operators.

A common practical form is:
`S2(r) = A r^(2α) + B r^(2α-ω) + …`
where `A` and `B` are typically **non-universal** amplitudes, and `ω > 0` is a corrections-to-scaling exponent. (Wegner 1972)

**Project relevance**: This is exactly the conceptual lesson behind the “scale-invariant ≠ universal” result. Passing a self-consistency (scale-invariance) gate is necessary but not sufficient to claim universality convergence.

### 1.2 Finite-Size Scaling “Gold Standard”

FSS formalizes how observables depend on reduced temperature `t = (T-Tc)/Tc` and system size `L`:
- correlation length: `ξ ~ |t|^(-ν_corr)`
- scaling collapse: `O(t, L) ≈ L^y f(t L^(1/ν_corr))` (exact form depends on the observable)

Canonical references: Privman & Fisher (1984); Cardy (1996).

---

## 2. KPZ / EW / Burgers (1D) — What Is Known and Why It Matters Here

### 2.1 KPZ Basics and Universality Structure

The KPZ equation (Kardar, Parisi & Zhang 1986) defines a nonequilibrium universality class with well-studied 1D exponents and rich fluctuation structure. Reviews and modern summaries:
- Halpin-Healy & Zhang (1995)
- Corwin (2012)
- Takeuchi (2018)

### 2.2 KPZ Fixed Point and Subclass (IC Sector) Structure

The modern viewpoint treats the KPZ scaling limit as a **stochastic fixed point**, with **subclasses** determined by initial conditions (flat/droplet/stationary). A mathematically precise construction is given by:
- Matetski, Quastel & Remenik (2021)

Exact results for narrow-wedge/droplet regimes connect to Tracy–Widom and Airy processes (Prähofer & Spohn 2002; Sasamoto & Spohn 2010).

**Project relevance**: IC-dependence is not a nuisance; it is part of the KPZ universality “sector decomposition”.

### 2.3 Stationary Measures and the “λ-Blindness” of Slope Variance

In 1D, the KPZ stationary **slope (gradient) field** has a Gaussian stationary measure that is independent of λ (with periodic boundary conditions). This explains why gradient-variance-type observables are dominated by a **noise-to-diffusion** ratio and may not reflect nonlinearity strength.

Useful references discussing stationary distributions / Burgers/KPZ probability measures include:
- Fogedby (1999) (noisy Burgers/KPZ probability distributions; stationary measure analysis)
- standard KPZ reviews (Halpin-Healy & Zhang 1995; Corwin 2012)

### 2.4 Caution: Burgers–KPZ Formal Relations vs Observable-Level Fluctuations

Even when equations are formally linked by `u = ∂x h`, *observable-level fluctuation universality* can differ depending on which field is examined and which sector is probed. A concrete example:
- Rodríguez-Fernández & Cuerno (2020): “non-KPZ fluctuations” in derivative observables of KPZ.

**Project relevance**: “Burgers → KPZ must converge” is not guaranteed for arbitrary ensemble observables; careful observable choice is essential.

---

## 3. Ising and Potts as Benchmark Critical Systems

### 3.1 Symmetry Sectors and Observability

For equilibrium models, symmetry matters:
- Z2 symmetry (Ising) means Z2-even features naturally couple to energy-like (thermal) operators, while Z2-odd features are needed for magnetic-sector scaling.
- For q-state Potts, the order parameter lives in a multi-component internal space with permutation symmetry; “order-parameter components” can be invisible without gauge-fixing or symmetry-breaking fields, depending on how features are aggregated.

**Project relevance**: this frames why a two-parameter (t, h) test can “see” only the thermal direction if features are symmetry-even, and why Potts order-parameter-component features may require gauge-fixing to contribute meaningfully to PCA variance.

### 3.2 Cluster Algorithms (Implementation Correctness Matters)

Potts/Ising Monte Carlo efficiency and correctness rely on cluster algorithms:
- Swendsen & Wang (1987) (cluster formulation)
- Wolff (1989) (single-cluster update)

For Potts, the correct bond-addition probability is `p_add = 1 - exp(-βJ)` (not the Ising-specific `1 - exp(-2βJ)`).
- Tomita & Okabe (2001) provides a clear statement in the Potts setting.

**Project relevance**: This directly explains why any Potts exponent claim must be grounded in a validated implementation (and why a wrong p_add contaminates exponent extraction).

---

## 4. Unsupervised Learning of Phases, Criticality, and Exponents

### 4.1 PCA/Embeddings for Phase Structure

Seminal and widely cited works:
- Wang (2016): PCA detects phase structure in Ising configurations.
- Hu–Singh–Scalettar (2017): “critical examination” of what unsupervised methods learn (important for avoiding overinterpretation).
- Wetzel (2017): PCA-to-VAE comparisons on phase transitions.
- Carrasquilla & Melko (2017): supervised NN phase classification.

### 4.2 PCA + Finite-Size Scaling (Important Prior Art)

There is explicit prior work combining PCA-like representations with FSS/exponent extraction in Ising contexts:
- Yue, Wang & Lyu (2022): incremental learning / PCA pipeline with preprocessing + FSS and critical exponents for Ising-like settings.
- Qi & Wang (2025): PCA for percolation transitions; emphasizes when representation/preprocessing changes what PCA “sees”.

**Project relevance**: This means the strongest novelty claim is not “PCA can recover ν_corr in Ising.” Rather:
- using a **local-observable** representation Φ (transferable across systems),
- producing a **systematic failure boundary** (Potts),
- and providing a **diagnostic protocol** that predicts when PCA-FSS is trustworthy.

### 4.3 Unsupervised Learning in Potts Models (Comparison Point)

Potts models are also studied in the ML-for-physics literature:
- Tirelli, Costa & Carlon (2021): unsupervised approaches applied to q-state Potts; emphasizes robustness of nonlinear methods and finite-size considerations.

**Project relevance**: Our Potts campaign provides detailed diagnostics for *why* a particular “PCA-on-local-features + collapse metric” pipeline can fail, complementing the broader Potts ML literature.

---

## 5. ML ↔ RG, Information Theory, and Information Geometry

### 5.1 Variational/Generative RG Connections

Well-known conceptual bridges:
- Mehta & Schwab (2014): variational RG ↔ deep learning mapping
- Koch-Janusz & Ringel (2018): construct RG transformations maximizing mutual information
- Li & Wang (2018): neural-network RG / generative RG viewpoints

### 5.2 Information Geometry and RG Flows on Measures

Information-geometric viewpoints on RG and coarse-graining:
- Bény & Osborne (2015) (information-geometric approach to RG)

**Project relevance**: This provides theoretical context for treating “the object” as a family of measures (pushforwards under Φ) and analyzing RG-like motion via information distances (KL, Bhattacharyya, etc.).

### 5.3 Fisher Information and Exponential Families (A Useful Theoretical Hook)

If a model family is written as an exponential family:
`p(x|θ) = exp(θ · T(x) - A(θ))`
then the covariance of sufficient statistics satisfies:
`Cov[T(x)] = Fisher(θ)`

Textbook anchors:
- Amari & Nagaoka (2000) (information geometry)
- Wainwright & Jordan (2008) (graphical models / exponential families)

**Project relevance**: If our feature map Φ acts as an approximate sufficient-statistic map for the dominant scaling field, then PCA on Φ is closely related to the dominant eigendirections of Fisher information. This provides a plausible route to a more rigorous “PCA finds scaling-field-like coordinates” statement.

### 5.4 Parameter-Sensitivity, “Sloppy Models”, and RG (Directly Relevant)

There is a well-developed viewpoint (Sethna group) in which:
- “relevant”/“stiff” parameter combinations are those that most strongly affect observables,
- and the corresponding geometry is controlled by Fisher information / parameter sensitivity.

Key references:
- Machta et al. (2013): “parameter space compression” as the mechanism behind emergent theories.
- Raju–Machta–Sethna (2018): information loss under coarse-graining, framed geometrically.
- Quinn–Machta–Sethna (2022): review-style synthesis in information-geometry terms.

**Project relevance**: This is arguably the cleanest literature bridge for the Ising result: “PC1 behaves like a thermal scaling field and supports FSS” is a statement about the data’s stiffest direction, which is exactly what Fisher-information geometry formalizes.

---

## 6. Distances Between Distributions (Two-Sample Statistics) and RG

### 6.1 Maximum Mean Discrepancy (MMD)

Kernel two-sample distances/tests:
- Gretton et al. (2012): MMD as a consistent two-sample test and practical metric in feature space.

**Project relevance**: MMD naturally compares *ensembles* (pushforward measures) rather than individual configurations, matching the project’s “measure-space” framing.

### 6.2 Optimal Transport and RG

Links between RG flow and optimal transport:
- Cotler & Rezchikov (2023): RG as Wasserstein gradient flow (context for using distributional geometry).

---

## 7. Practical Implications for This Project (How Literature Explains the Arc)

1. **“Scale-invariant ≠ universal” is expected in RG** and must be treated as a diagnostic criterion (Wegner; Privman–Fisher). Our experiments make this operational.
2. **KPZ stationary slope measure** explains why gradient-variance features align with a noise-to-diffusion amplitude coordinate and can be λ-blind (KPZ reviews; Fogedby).
3. **Symmetry sectors matter**: if features are symmetry-even, they will miss symmetry-odd scaling directions unless symmetry is broken or gauge-fixed (Ising h-direction; Potts OP components).
4. **PCA + FSS has prior art** (Ising), so our papers should emphasize what is actually new: local features, cross-system transfer, rigorous diagnostics, and the method boundary.
5. **Fisher information is the right “math language” for visibility**: PCA-like methods preferentially surface the stiff/relevant directions because they maximize variance/sensitivity. This connects the ML pipeline to the geometry of parameterized measures and to RG relevance. (Machta 2013; Raju 2018; Quinn 2022)

---

## References (Curated, With “Why It Matters”)

1. Wegner, F. J. (1972). “Corrections to scaling laws.” *Phys. Rev. B* 5, 4529. doi:10.1103/PhysRevB.5.4529
2. Privman, V. & Fisher, M. E. (1984). “Universal critical amplitudes in finite-size scaling.” *Phys. Rev. B* 30, 322. doi:10.1103/PhysRevB.30.322
3. Cardy, J. (1996). *Scaling and Renormalization in Statistical Physics*. Cambridge.
4. Kardar, M., Parisi, G., & Zhang, Y.-C. (1986). “Dynamic scaling of growing interfaces.” *Phys. Rev. Lett.* 56, 889.
5. Halpin-Healy, T. & Zhang, Y.-C. (1995). *Physics Reports* 254, 215–414.
6. Corwin, I. (2012). “The KPZ equation and universality class.” *Random Matrices: Theory and Applications* 1, 1130001.
7. Takeuchi, K. A. (2018). “An appetizer to modern developments on KPZ.” *J. Phys. A* 51, 164001.
8. Matetski, K., Quastel, J., & Remenik, D. (2021). “The KPZ fixed point.” *Acta Math.* 227, 115–203.
9. Prähofer, M. & Spohn, H. (2002). *J. Stat. Phys.* 108, 1071–1106.
10. Sasamoto, T. & Spohn, H. (2010). *Phys. Rev. Lett.* 104, 230602.
11. Fogedby, H. C. (1999). “Canonical phase-space approach to the noisy Burgers equation: probability distributions.” *Phys. Rev. E* 59, 5065.
12. Rodríguez-Fernández, A. & Cuerno, R. (2020). “Non-KPZ fluctuations in the derivative of the KPZ equation.” *Phys. Rev. E* 101, 052126.
13. Swendsen, R. H. & Wang, J.-S. (1987). *Phys. Rev. Lett.* 58, 86.
14. Wolff, U. (1989). “Collective Monte Carlo updating for spin systems.” *Phys. Rev. Lett.* 62, 361.
15. Tomita, Y. & Okabe, Y. (2001). “Efficient cluster algorithm for q-state Potts model…” *Phys. Rev. E* 64, 036114.
16. Wang, L. (2016). “Discovering phase transitions with unsupervised learning.” *Phys. Rev. B* 94, 195105.
17. Hu, W., Singh, R. R. P., & Scalettar, R. T. (2017). *Phys. Rev. E* 95, 062122.
18. Wetzel, S. J. (2017). *Phys. Rev. E* 96, 022140.
19. Carrasquilla, J. & Melko, R. G. (2017). *Nat. Phys.* 13, 431.
20. Yue, Z., Wang, Y., & Lyu, P. (2022). “Incremental learning of phase transition in Ising model: preprocessing, finite-size scaling and critical exponents.” *Physica A* 600, 127538.
21. Qi, X. & Wang, L. (2025). “Principal component analysis for percolation transitions with and without preprocessing.” *Phys. Rev. E* 111, 045303.
22. Tirelli, A., Costa, N. C., & Carlon, E. (2021). “Unsupervised machine learning approaches to the q-state Potts model.” arXiv:2112.06735.
23. Xu, J., Salas, J., & Deng, Y. (2025). “Finite-size scaling corrections from subleading magnetic scaling field and application to 2D Ising and Potts models.” *Entropy* 27(1), 24.
24. Machta, B. B., Chachra, R., Transtrum, M. K., & Sethna, J. P. (2013). “Parameter space compression underlies emergent theories and predictive models.” *Science* 342(6158), 604–607.
25. Raju, A. A., Machta, B. B., & Sethna, J. P. (2018). “Information loss under coarse graining: a geometric approach.” *Phys. Rev. E* 98, 052112.
26. Quinn, K. N., Machta, B. B., & Sethna, J. P. (2022). “Information geometry for multiparameter models: new perspectives on the origin of simplicity.” *Rep. Prog. Phys.* 86, 035901.
27. Mehta, P. & Schwab, D. J. (2014). arXiv:1410.3831.
28. Koch-Janusz, M. & Ringel, Z. (2018). *Nat. Phys.* 14, 578–582.
29. Li, S.-H. & Wang, L. (2018). *Phys. Rev. Lett.* 121, 260601.
30. Bény, C. & Osborne, T. J. (2015). “Information-geometric approach to RG.” *Phys. Rev. A* 92, 022330.
31. Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS/OUP.
32. Wainwright, M. J. & Jordan, M. I. (2008). “Graphical models, exponential families, and variational inference.” *Found. Trends Mach. Learn.* 1(1–2).
33. Gretton, A. et al. (2012). “A Kernel Two-Sample Test.” *JMLR* 13, 723–773.
34. Cotler, J. & Rezchikov, S. (2023). “Renormalization Group Flow as Optimal Transport.” *Phys. Rev. D* 108, 025003.
