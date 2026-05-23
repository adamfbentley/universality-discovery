# Literature Review — KPZ Universality, Surface Growth, and ML Methods

**Last updated**: May 24, 2026
**Purpose**: Physics-first reference document. Read this before designing experiments.

**May 2026 scan note**: ResearchGate-facing searches did not reveal a close
recent prior result using unsupervised local-feature geometry to classify EW,
KPZ, BD, Eden, RD, and KS surface-growth ensembles. The closest credibility
anchors remain the modern KPZ scaling literature, ML-for-phase-transition
papers, and cautionary work on what unsupervised representations do and do not
prove.

**May 24 project note**: The current ML-paper direction is not a positive
universality-discovery claim. It is a quotient-learning failure-mode benchmark:
finite-size feature and effective-exponent representations can contain strong
local physical signal without robustly identifying the coarser RG equivalence
relation under ordinary clustering.

---

## 1. The KPZ Equation and Its Universality Class

### 1.1 The Equation

Kardar, Parisi & Zhang (1986) proposed the stochastic PDE for interface growth:

∂ₜh = ν∇²h + (λ/2)(∇h)² + η(x,t)

where ν is the surface tension (diffusion), λ is the nonlinear coupling, and η is space-time white noise with ⟨η(x,t)η(x',t')⟩ = 2Dδ(x−x')δ(t−t'). The three terms represent relaxation, lateral growth, and stochastic driving.

Setting λ = 0 gives the Edwards-Wilkinson (EW) equation, which is exactly solvable (linear SPDE with Gaussian statistics).

**Key references:**
- Kardar, Parisi & Zhang (1986). "Dynamic scaling of growing interfaces." *Phys. Rev. Lett.* 56, 889.
- Edwards & Wilkinson (1982). "The surface statistics of a granular aggregate." *Proc. R. Soc. A* 381, 17.

### 1.2 Scaling Exponents

The interface width W(L,t) = ⟨[h(x,t) − ⟨h⟩]²⟩^{1/2} obeys the Family-Vicsek scaling ansatz:

W(L,t) ~ L^α f(t/L^z)

where α is the roughness exponent, β = α/z is the growth exponent, and z = α/β is the dynamic exponent. In 1+1 dimensions:

| Class | α | β | z |
|-------|---|---|---|
| EW | 1/2 | 1/4 | 2 |
| KPZ | 1/2 | 1/3 | 3/2 |

Note that **α = 1/2 for both EW and KPZ** in 1D. The roughness exponent alone does not distinguish them. The difference is in β (and therefore z).

**Key references:**
- Family & Vicsek (1985). "Scaling of the active zone in the Eden process." *J. Phys. A* 18, L75.
- Barabási & Stanley (1995). *Fractal Concepts in Surface Growth.* Cambridge University Press. **[Essential textbook — Chapter 4 covers the stationary measure.]**

### 1.3 Exact Solutions and the KPZ Fixed Point

A revolution in mathematical understanding of KPZ began in the 2000s with connections to random matrix theory and integrable probability. The one-point height distribution of 1D KPZ converges (in the scaling limit) to Tracy-Widom distributions, with the specific distribution depending on the initial condition:

| Initial condition | Limiting distribution | Symmetry class |
|---|---|---|
| Droplet (narrow wedge) | Tracy-Widom GUE (F₂) | Unitary |
| Flat | Tracy-Widom GOE (F₁) | Orthogonal |
| Stationary | Baik-Rains (F₀) | — |

This means the KPZ universality class has **internal substructure** determined by initial conditions. This is not a finite-size artifact — it persists in the scaling limit.

**Key references:**
- Prähofer & Spohn (2000). "Universal distributions for growth processes in 1+1 dimensions and random matrices." *Phys. Rev. Lett.* 84, 4882.
- Johansson (2000). "Shape fluctuations and random matrices." *Commun. Math. Phys.* 209, 437.
- Sasamoto & Spohn (2010). "One-dimensional Kardar-Parisi-Zhang equation: an exact solution and its universality." *Phys. Rev. Lett.* 104, 230602.
- Amir, Corwin & Quastel (2011). "Probability distribution of the free energy of the continuum directed random polymer in 1+1 dimensions." *Commun. Pure Appl. Math.* 64, 466.
- Corwin (2012). "The Kardar-Parisi-Zhang equation and universality class." *Random Matrices: Theory Appl.* 1, 1130001. **[Best modern review for a physicist.]**
- Matetski, Quastel & Remenik (2021). "The KPZ fixed point." *Acta Math.* 227, 115. **[Rigorous construction of the full scaling limit.]**
- Takeuchi (2018). "An appetizer to modern developments on the KPZ universality class." *Physica A* 504, 77.

### 1.4 The Stationary Slope Measure — Why Gradient Features Fail

**This is the single most important result for this project.**

The KPZ equation can be transformed via u = ∇h into the noisy Burgers equation:

∂ₜu = ν∇²u + λu∂ₓu + ∂ₓη

The **stationary measure** for u(x) on a ring of length L with periodic boundary conditions is:

P_stat[u] ∝ exp(−(ν/2D) ∫₀ᴸ u(x)² dx)

This is **Gaussian** and **independent of λ**. The nonlinear coupling λ does not appear. This means:

1. **Any observable computed from the stationary slope field u = ∇h is identical for EW and KPZ** (with matched ν, D). Gradient variance, gradient skewness, gradient kurtosis, Laplacian variance — all of them.
2. The only parameters that matter at stationarity are D/ν (the noise-to-diffusion ratio).
3. This explains why PC1 of gradient features tracks D/ν (Exp 46/54) — it's not discovering universality, it's measuring the amplitude of a Gaussian.

**Practical consequences for this project:**
- Gradient features cannot separate EW from KPZ at stationarity. This is a mathematical theorem, not an empirical finding.
- Separation requires either (a) transient (pre-stationary) observables, (b) higher-order temporal statistics, or (c) observables that probe the height field h directly (not just ∇h).
- At longer simulation times T, EW and KPZ gradient distributions converge. The EW-KPZ centroid distance should **decrease** with T, which is exactly what Exp 62 found (1.04 → 0.87).

**Key references:**
- Barabási & Stanley (1995), Chapter 4. **[The stationary measure is derived here.]**
- Fogedby (1999). "Canonical phase-space approach to the noisy Burgers equation." *Phys. Rev. E* 59, 5065.
- Halpin-Healy & Zhang (1995). "Kinetic roughening phenomena, stochastic growth, directed polymers and all that." *Physics Reports* 254, 215. **[Comprehensive early review.]**

### 1.5 What DOES Distinguish EW from KPZ

If gradient statistics are degenerate, what observables carry the EW-KPZ distinction?

**Height distribution shape:** KPZ has Tracy-Widom statistics (skewed, heavy-tailed). EW has Gaussian height fluctuations. But this requires large L and T to converge — at accessible finite sizes, the convergence is slow and noisy (as found in Exp 31-36).

**Temporal growth rate:** β_EW = 1/4 vs β_KPZ = 1/3. Measurable from W(t) growth curves, but requires the system to be in the growth regime (not yet saturated).

**Slope-growth coupling:** The term λ(∇h)² in the KPZ equation means local growth rate correlates with local slope squared. This is directly detectable via regression of ∂ₜh against (∇h)² (Exp 13). This is a **transient** observable — it's most detectable before stationarity.

**Velocity statistics:** Multi-step velocity Δh(x, t+Δt) − h(x, t) accumulates the nonlinear signal. Skewness of velocity is non-zero for KPZ, zero for EW. Again, strongest pre-stationarity.

**The deep issue:** The observables that distinguish EW from KPZ are all either temporal, transient, or require the full height field (not just gradients). This is a direct consequence of the stationary slope measure theorem.

---

## 2. Discrete Growth Models and the Continuum Limit

### 2.1 Models in the KPZ Universality Class

Several discrete lattice models are believed to be in the KPZ universality class:

- **Ballistic Deposition (BD):** Particles drop vertically and stick upon first contact with the surface or a neighbour. Creates overhangs. Known to give α ≈ 1/2, β ≈ 1/3 in 1D, but convergence to continuum KPZ is very slow.
- **Eden model:** Cells are added at random to the perimeter of a growing cluster. In a restricted geometry (strip), gives KPZ exponents.
- **Polynuclear growth (PNG):** Nucleation and lateral spreading. Exactly solvable; height fluctuations converge to Tracy-Widom.
- **TASEP / last passage percolation:** Integrable models with exact KPZ connections.

### 2.2 The Discrete-Continuum Gap

Discrete models exhibit lattice artifacts that persist for a long time before asymptotic KPZ behaviour emerges. This is well-known but often underestimated:

- BD has very large corrections to scaling. Numerical estimates of β often give values slightly different from 1/3 at accessible system sizes.
- The gradient variance of BD is orders of magnitude larger than continuum KPZ because discrete growth rules create large local slope fluctuations.
- Convergence to the continuum limit requires spatial coarse-graining — averaging over blocks of sites. Block size b = 2-4 reduces the discrete-continuum gap significantly (as found in Exp 23), but temporal features may diverge under the same coarse-graining (Exp 64).

**Key references:**
- Meakin (1998). *Fractals, Scaling and Growth Far from Equilibrium.* Cambridge. **[Comprehensive treatment of discrete growth models.]**
- Krug & Spohn (1991). "Kinetic roughening of growing surfaces." In *Solids Far from Equilibrium.* Cambridge.
- Family & Vicsek (1991). *Dynamics of Fractal Surfaces.* World Scientific.

### 2.3 Non-KPZ Classes Used in This Project

- **Random Deposition (RD):** Each column grows independently. β = 1/2, no spatial correlations. Trivially different from all other classes.
- **Kuramoto-Sivashinsky (KS):** ∂ₜh = −ν₂∇²h − κ∇⁴h + (λ/2)(∇h)² + η. The negative diffusion (−ν₂∇²h) creates instability; the 4th-order term (−κ∇⁴h) provides stabilisation. Produces spatiotemporal chaos. Whether KS flows to KPZ at very large scales in 1D is debated — numerical evidence is mixed. Recent work suggests it may be in the KPZ class asymptotically, but the crossover is extremely slow.

**Key references:**
- Sneppen et al. (1992). "Dynamic scaling and crossover analysis for the Kuramoto-Sivashinsky equation." *Phys. Rev. A* 46, R7351.
- Jayaprakash, Hayot & Pandit (1993). "Universal properties of the two-dimensional Kuramoto-Sivashinsky equation." *Phys. Rev. Lett.* 71, 12.

---

## 3. Renormalization Group and Universality

### 3.1 RG Basics for Surface Growth

The RG approach to surface growth involves integrating out short-wavelength modes and rescaling. For the KPZ equation, the coupling constant g = λ²D/ν³ determines the flow:

- d = 1: g flows to a finite fixed-point value (strong coupling). KPZ universality is stable.
- d = 2: This is the critical dimension. Logarithmic corrections appear.
- d > 2: Weak-coupling (EW) fixed point is stable for small g; strong-coupling KPZ fixed point may exist but is not perturbatively accessible.

**Key point for this project:** In 1D, the KPZ nonlinearity is always relevant (in the RG sense). There is no perturbative small-λ regime. EW behaviour is unstable to any nonzero λ. This means EW and KPZ are genuinely different fixed points, and the distinction is physically sharp — but accessing it requires observing the system on long enough time scales for the nonlinearity to manifest.

### 3.2 RG-Relevant vs RG-Irrelevant Operators

An operator O is:
- **Relevant** if it grows under coarse-graining (scaling dimension > 0)
- **Irrelevant** if it shrinks under coarse-graining (scaling dimension < 0)
- **Marginal** if it stays constant

Gradient moments (Var(∇h), etc.) are **RG-relevant** in the sense that they grow under spatial coarse-graining for KPZ-class systems. This was found experimentally in Exp 24 (EW-KPZ distance expands by 45% under CG) and Exp 47 (KL divergence increases with CG).

**This creates a fundamental tension:** The features that best discriminate universality classes at finite size (gradient moments) are precisely the ones whose values blow up under coarse-graining. You cannot simultaneously have features that (a) separate classes at finite size and (b) converge to stable fixed-point values under RG, if those features are relevant operators.

**Key references:**
- Medina, Hwa, Kardar & Zhang (1989). "Burgers equation with correlated noise." *Phys. Rev. A* 39, 3053. **[RG for KPZ]**
- Krug & Spohn (1991), as above.
- Barabási & Stanley (1995), Chapter 7 (RG for growth equations).

### 3.3 Scale-Invariant ≠ Universal

A critical distinction found empirically in Exp 50q/50r and supported by standard RG theory:

- **Scale-invariant observables** are those whose statistical properties are unchanged under rescaling x → bx, h → b^α h, t → b^z t. Both universal exponents AND non-universal amplitudes can be scale-invariant.
- **Universal quantities** are those that take the same value for all systems in the same universality class. Exponent ratios, amplitude ratios, and scaling functions are universal. Individual amplitudes are generally not.

Example: The structure function S₂(r) = ⟨[h(x+r) − h(x)]²⟩ ~ A·r^{2α}. The exponent 2α is universal. The prefactor A is not — it depends on microscopic details. But both are "scale-invariant" in the sense that S₂(br) = b^{2α} S₂(r).

**Key references:**
- Wegner (1972). "Corrections to scaling laws." *Phys. Rev. B* 5, 4529.
- Privman & Fisher (1984). "Universal critical amplitudes in finite-size scaling." *Phys. Rev. B* 30, 322.
- Cardy (1996). *Scaling and Renormalization in Statistical Physics.* Cambridge.

---

## 4. Equilibrium Critical Phenomena (Ising, Potts)

### 4.1 The 2D Ising Model

The benchmark system for critical phenomena. Exact solution by Onsager (1944) gives:
- T_c = 2J/[k_B ln(1+√2)] ≈ 2.269J/k_B
- ν = 1 (correlation length exponent)
- α = 0 (log divergence of specific heat)
- β = 1/8, γ = 7/4 (magnetisation, susceptibility)

**Why Ising works for PCA-FSS (Exp 52):** Near criticality, the Ising model has a single dominant relevant operator (thermal). PCA on local observables (energy-like features) naturally finds this direction because it's the direction of maximum variance. FSS collapse along this direction gives ν because the correlation length diverges as ξ ~ |t|^{−ν}.

The key theoretical connection is via Fisher information / sloppy models (Machta et al. 2013): the stiffest direction in parameter space (the one that most affects observables) corresponds to the thermal relevant direction. PCA finds the stiffest direction by construction.

### 4.2 The q-State Potts Model

Generalises Ising from Z₂ to Z_q symmetry. For q = 3 in 2D:
- The transition is second-order with ν = 5/6, α = 1/3
- The order parameter lives in a 2D internal space (not a scalar)
- Z₃-symmetric features (like energy) see only the thermal sector
- The magnetic sector requires explicitly breaking the permutation symmetry or including order parameter components

**Why Potts fails for PCA-FSS (Exp 55-59):** Multiple competing effects:
1. Z₃ symmetry means Z₃-even features are blind to the magnetic direction
2. Corrections to scaling are larger than for Ising
3. The PCA collapse metric (window variance) may have metric degeneracy — multiple ν values give comparably good collapses
4. Standard Binder cumulant analysis still works (Exp 57c: ν ≈ 0.884, ~6% error), so the physics is accessible — the method is specifically limited

### 4.3 Prior Art: ML for Phase Transitions

The application of ML to phase transitions is well-established:
- **Wang (2016):** PCA detects Ising phase structure unsupervised
- **Carrasquilla & Melko (2017):** Supervised neural networks classify phases with high accuracy
- **Wetzel (2017):** Comparison of PCA and variational autoencoders for phase detection
- **Hu, Singh & Scalettar (2017):** Critical examination of what unsupervised methods actually learn — important cautionary paper
- **Yue, Wang & Lyu (2022):** PCA + preprocessing + FSS for Ising critical exponents — **direct prior art for Exp 52**

**Implication for novelty:** "PCA recovers ν for Ising" is not novel by itself. The contribution must be in what's different: the specific observable map (local gradient/energy features), the cross-system comparison, the diagnostic protocol, and especially the **failure analysis** for Potts.

**Key references:**
- Wang (2016). *Phys. Rev. B* 94, 195105.
- Carrasquilla & Melko (2017). *Nature Physics* 13, 431.
- Wetzel (2017). *Phys. Rev. E* 96, 022140.
- Hu, Singh & Scalettar (2017). *Phys. Rev. E* 95, 062122.
- Yue, Wang & Lyu (2022). *Physica A* 600, 127538.
- Tirelli, Costa & Carlon (2021). "Unsupervised ML approaches to the q-state Potts model." arXiv:2112.06735.

---

## 5. Information Geometry and RG

### 5.1 Fisher Information and Parameter Sensitivity

If a statistical model is parameterised as p(x|θ), the Fisher information matrix is:

F_ij(θ) = E[∂_i log p · ∂_j log p]

This defines a Riemannian metric on parameter space (the Fisher-Rao metric). Directions with large Fisher information are "stiff" — observables change rapidly. Directions with small Fisher information are "sloppy" — observables barely respond.

The connection to RG: relevant operators correspond to stiff directions; irrelevant operators correspond to sloppy directions. Coarse-graining preferentially removes sloppy directions, which is why effective theories have fewer parameters than microscopic ones.

**Key references:**
- Amari & Nagaoka (2000). *Methods of Information Geometry.* AMS/OUP.
- Machta, Chachra, Transtrum & Sethna (2013). "Parameter space compression underlies emergent theories." *Science* 342, 604.
- Raju, Machta & Sethna (2018). "Information loss under coarse graining." *Phys. Rev. E* 98, 052112.
- Quinn, Machta & Sethna (2022). "Information geometry for multiparameter models." *Rep. Prog. Phys.* 86, 035901.

### 5.2 Why This Matters for PCA

If the observable map Φ acts as an approximate sufficient statistic for the dominant scaling field, then PCA on Φ extracts the stiffest direction — which is the thermal/relevant direction. This is why PCA works for Ising (single dominant relevant operator) and may fail for Potts (competing operators, corrections to scaling).

The Fisher-PCA connection: Cov[Φ(x)] evaluated near criticality is approximately proportional to the Fisher information matrix in the scaling-field basis, if Φ captures the dominant fluctuations.

### 5.3 Information-Geometric Distances Under CG

KL divergence and Bhattacharyya distance between EW and KPZ induced measures **increase** under spatial coarse-graining (Exp 47). This is the information-geometric signature of RG relevance: the gradient features are relevant operators, so their distributions diverge (become more distinguishable) under CG rather than converging.

**Key reference:**
- Bény & Osborne (2015). "Information-geometric approach to RG." *Phys. Rev. A* 92, 022330.

---

## 6. Practical Lessons from the Literature for This Project

### 6.1 What Was Known Before Any Experiment Was Run

1. EW and KPZ share the same stationary slope measure in 1D. Gradient features are degenerate at stationarity. **(Barabási & Stanley Ch. 4)**
2. KPZ height fluctuations converge to Tracy-Widom, but slowly. Finite-size convergence requires careful treatment. **(Prähofer & Spohn 2000; Corwin 2012)**
3. KPZ has IC-dependent subclasses (GOE/GUE/Baik-Rains). **(Matetski, Quastel & Remenik 2021)**
4. Discrete models (BD, Eden) have large corrections to scaling and converge slowly to continuum KPZ behaviour. **(Krug & Spohn 1991; Meakin 1998)**
5. PCA on Ising configurations recovers phase structure. **(Wang 2016)**
6. Scale-invariant ≠ universal. Non-universal amplitudes persist under scaling. **(Wegner 1972; Privman & Fisher 1984)**

### 6.2 What the Experiments Added Beyond the Literature

1. **Exp 13:** Slope-growth regression directly detects the KPZ nonlinearity λ(∇h)² at finite size. This is a practical diagnostic not derived from the exact solution literature.
2. **Exp 24/47:** Numerical demonstration that gradient moments are RG-relevant operators — their inter-class distances grow under CG. Known in principle from RG theory, but the quantitative measurement in feature space is new.
3. **Exp 27:** IC-dependence of PCA separation connects to the GOE/GUE structure. The fact that the PCA loading vector *rotates* between ICs is a concrete numerical demonstration of subclass geometry.
4. **Exp 50q/50r:** Operational distinction between scale-invariant and universal observables, with a concrete diagnostic protocol.
5. **Exp 52:** PCA-FSS for Ising with local-observable features (not raw configurations). Methodologically simpler than prior art but uses a different representation.
6. **Exp 55-59/57:** Systematic failure analysis for PCA-FSS on Potts, with control (Binder works, PCA doesn't).
7. **Exp 62-64:** Quantitative characterisation of the ARI ≈ 0.5 clustering ceiling, with structural explanation (EW-KPZ gradient degeneracy + BD topological disconnection).

### 6.3 What Should Have Been Done Differently

1. **Literature review first.** The stationary slope measure theorem should have been identified before Exp 1, not at Exp 54.
2. **Start with temporal/transient features.** Knowing the gradient degeneracy, the project should have targeted β, velocity statistics, and slope-growth coupling from the beginning.
3. **Check finite-size convergence rigorously.** Tracy-Widom convergence at accessible L, T should have been benchmarked early (Exp 34 found L >> √T is needed) rather than assumed.
4. **Separate the discrete-continuum problem.** BD and Eden should have been treated as a separate convergence study, not mixed with the EW-KPZ discrimination problem.

---

## 7. Core References (Annotated)

### Essential (Read First)
1. **Barabási & Stanley (1995).** *Fractal Concepts in Surface Growth.* Cambridge. — The textbook. Ch. 4 (stationary measures), Ch. 7 (RG). Read before doing anything else.
2. **Corwin (2012).** "The KPZ equation and universality class." — Best modern review. Exact solutions, Tracy-Widom, integrable structure.
3. **Takeuchi (2018).** "An appetizer to modern developments on KPZ." — Accessible introduction with experimental context.

### KPZ Exact Results
4. Kardar, Parisi & Zhang (1986). *Phys. Rev. Lett.* 56, 889.
5. Prähofer & Spohn (2000). *Phys. Rev. Lett.* 84, 4882.
6. Sasamoto & Spohn (2010). *Phys. Rev. Lett.* 104, 230602.
7. Amir, Corwin & Quastel (2011). *Commun. Pure Appl. Math.* 64, 466.
8. Matetski, Quastel & Remenik (2021). *Acta Math.* 227, 115.
9. Johansson (2000). *Commun. Math. Phys.* 209, 437.

### KPZ Reviews and Textbooks
10. Halpin-Healy & Zhang (1995). *Physics Reports* 254, 215.
11. Krug & Spohn (1991). "Kinetic roughening of growing surfaces." In *Solids Far from Equilibrium.*
12. Meakin (1998). *Fractals, Scaling and Growth Far from Equilibrium.* Cambridge.

### Stationary Measure / Noisy Burgers
13. Fogedby (1999). *Phys. Rev. E* 59, 5065.
14. Rodríguez-Fernández & Cuerno (2020). *Phys. Rev. E* 101, 052126.

### RG and Scaling
15. Wegner (1972). *Phys. Rev. B* 5, 4529.
16. Privman & Fisher (1984). *Phys. Rev. B* 30, 322.
17. Cardy (1996). *Scaling and Renormalization in Statistical Physics.* Cambridge.
18. Medina, Hwa, Kardar & Zhang (1989). *Phys. Rev. A* 39, 3053.

### ML for Physics
19. Wang (2016). *Phys. Rev. B* 94, 195105.
20. Carrasquilla & Melko (2017). *Nature Physics* 13, 431.
21. Hu, Singh & Scalettar (2017). *Phys. Rev. E* 95, 062122.
22. Wetzel (2017). *Phys. Rev. E* 96, 022140.
23. Yue, Wang & Lyu (2022). *Physica A* 600, 127538.
24. Tirelli, Costa & Carlon (2021). arXiv:2112.06735.

### Information Geometry / Sloppy Models
25. Machta, Chachra, Transtrum & Sethna (2013). *Science* 342, 604.
26. Raju, Machta & Sethna (2018). *Phys. Rev. E* 98, 052112.
27. Quinn, Machta & Sethna (2022). *Rep. Prog. Phys.* 86, 035901.
28. Amari & Nagaoka (2000). *Methods of Information Geometry.* AMS/OUP.
29. Bény & Osborne (2015). *Phys. Rev. A* 92, 022330.

### RG and Optimal Transport / Deep Learning
30. Cotler & Rezchikov (2023). *Phys. Rev. D* 108, 025003.
31. Mehta & Schwab (2014). arXiv:1410.3831.
32. Koch-Janusz & Ringel (2018). *Nature Physics* 14, 578.

### Experimental KPZ
33. Takeuchi & Sano (2010). "Universal fluctuations of growing interfaces: evidence in turbulent liquid crystals." *Phys. Rev. Lett.* 104, 230601. **[Real experimental verification of KPZ Tracy-Widom statistics.]**
34. Takeuchi & Sano (2012). "Evidence for geometry-dependent universal fluctuations of the Kardar-Parisi-Zhang interfaces in liquid-crystal turbulence." *J. Stat. Phys.* 147, 853.
35. Widmann et al. (2026). "Observation of Kardar-Parisi-Zhang universal scaling in two dimensions." *Science* 392, 221. **[Recent experimental 2D KPZ scaling benchmark; reinforces that scaling evidence, not clustering alone, is the standard for universality claims.]**

### KS-KPZ Crossover
36. Sneppen et al. (1992). *Phys. Rev. A* 46, R7351.
37. Jayaprakash, Hayot & Pandit (1993). *Phys. Rev. Lett.* 71, 12.

### Cluster Algorithms
38. Swendsen & Wang (1987). *Phys. Rev. Lett.* 58, 86.
39. Wolff (1989). *Phys. Rev. Lett.* 62, 361.
40. Tomita & Okabe (2001). *Phys. Rev. E* 64, 036114.

### Distribution Theory
41. Gretton et al. (2012). "A kernel two-sample test." *JMLR* 13, 723.
42. Wainwright & Jordan (2008). *Found. Trends Mach. Learn.* 1(1-2).
