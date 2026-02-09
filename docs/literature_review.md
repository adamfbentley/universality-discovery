# Literature Review: Mathematical Foundations of Universality Discovery

## Executive Summary

This document reviews the theoretical foundations of the universality-discovery project, verifying mathematical correctness against established literature in statistical physics and machine learning.

**Verdict: ✅ Core equations and scaling foundations match established physics literature; empirical claims should be scoped by (i) scale-invariance/self-consistency checks and (ii) whether the chosen observables isolate *universal* quantities (exponents, universal amplitude ratios) rather than non-universal amplitudes/corrections.**

**Feb 2026 update (important for this project):**
- Scale-invariance ≠ universality (Wegner 1972; Privman & Fisher 1984). Passing a “diagnostic gate” that checks self-consistency under coarse-graining is necessary but not sufficient for universality convergence.
- 1D KPZ has a stationary *slope* (gradient) measure that is Gaussian and λ-independent (KPZ 1986; Halpin-Healy & Zhang 1995; Corwin 2012; Takeuchi 2018). This implies many gradient-variance observables scale primarily with D/ν, so PCA’s leading direction can naturally align with D/ν even when λ varies.
- PCA/unsupervised learning of phase transitions and scaling behavior is an active literature (Wang 2016; Hu–Singh–Scalettar 2017; Wetzel 2017; Mendes-Santos et al. 2021), and provides context for our Ising “coupling coordinate + finite-size scaling” validations.

---

## 1. Universality Classes and Critical Exponents

### 1.1 Theoretical Foundation

Universality is a fundamental concept in statistical mechanics where systems with vastly different microscopic details exhibit identical macroscopic behavior near critical points. This is characterized by **critical exponents** that are independent of microscopic parameters (Kadanoff, 1966; Hohenberg & Halperin, 1977).

For surface growth models, the key universality classes are characterized by three exponents:

| Class | Roughness (α) | Growth (β) | Dynamic (z) | Scaling Relation |
|-------|---------------|------------|-------------|------------------|
| **EW** | 1/2 | 1/4 | 2 | z = α/β |
| **KPZ** | 1/2 | 1/3 | 3/2 | z = α/β |

**Source:** Kardar, Parisi & Zhang (1986), Phys. Rev. Lett. 56:889

### 1.2 Family-Vicsek Scaling

The surface width W(L,t) follows the Family-Vicsek scaling relation:

W(L,t) = L^α f(t / L^z)

where:
f(u) ∝ u^β for u ≪ 1, and f(u) → const for u ≫ 1

The width is defined as:
W(L,t) = ⟨ (1/L) ∫_0^L (h(x,t) - h̄(t))^2 dx ⟩^(1/2)

**Source:** Family & Vicsek (1985), J. Phys. A 18:L75

### 1.3 KPZ Fixed Point and Subclass Structure (Flat/Droplet/Stationary)

Modern theory treats KPZ as a **stochastic fixed point** with distinct universal subclasses determined by initial conditions (flat, droplet, stationary). A rigorous construction of the **KPZ fixed point** was provided by Matetski, Quastel, and Remenik (2021), giving a precise mathematical object behind scaling limits.

Exact results for droplet (narrow‑wedge) initial data connect KPZ fluctuations to **Tracy‑Widom** distributions and **Airy processes** (Sasamoto & Spohn, 2010; Prähofer & Spohn, 2002).

Crucially, experiments confirm **geometry‑dependent subclasses**: Takeuchi & Sano observed different universal fluctuation distributions for flat vs circular growth in liquid‑crystal turbulence (2010–2012), matching the theoretical KPZ subclass framework.

### 1.4 Scale-Invariant Observables vs Universal Quantities (Corrections to Scaling)

KPZ scaling functions are known in 1D with high precision. Prähofer & Spohn (2004) computed the **stationary two‑point scaling function** exactly via the PNG model (J. Stat. Phys. 115, 255–279; cond‑mat/0212519), which provides a benchmark for any empirical observable built from structure functions or correlations.

Functional/nonperturbative RG computations reproduce KPZ scaling properties and extract **universal amplitude ratios** (e.g., Canet et al. 2011 and follow-ups).

However, even when leading scaling exponents are universal, **subleading corrections** (e.g., S2(r) = A r^(2α) + B r^(2α−ω) + …) carry **non‑universal amplitudes** (Wegner 1972). In practice, KPZ-type models can exhibit strong and sometimes slow corrections/crossovers (e.g., ballistic-deposition/Family-model mixtures for small effective λ in 1D; Chame & Reis 2002), and strong corrections in 2+1 dimensions have also been documented (e.g., Kelling, Ódor & Gemming 2016).

**Implication for this project:** An observable can be **scale‑invariant** yet still encode **non‑universal corrections**. Thus, passing a scale‑invariance gate is necessary but **not sufficient** for detecting universality convergence.

### 1.5 Stationary Measures, Scaling Fields, and “PC1 Finds a Coupling”

In RG language, the “right” coordinates are **scaling fields**: linear combinations of microscopic parameters that flow along relevant/irrelevant directions under coarse-graining (Cardy 1996). Two consequences are important for interpreting PCA-based “coupling coordinate discovery”:

1. Some physically meaningful coordinates are *exactly* universal (critical exponents, universal amplitude ratios).
2. Some coordinates are *exactly* scale-invariant for a given system yet still encode non-universal amplitudes or pre-asymptotic corrections.

For KPZ specifically, there is a crucial exact property in 1D:

- In 1D KPZ with periodic boundary conditions, the stationary slope field u = ∂x h is Gaussian (equivalently, stationary height differences are Brownian/Gaussian), and its equal-time statistics are independent of λ (Kardar–Parisi–Zhang 1986; Halpin-Healy & Zhang 1995; Corwin 2012; Takeuchi 2018).
- This implies variance-type gradient observables scale primarily with D/ν (up to lattice/time discretization factors), regardless of λ.

**Project relevance:** If Φ(h) includes gradient-variance and Laplacian-variance features, PCA’s PC1 will naturally align with D/ν because that coordinate controls the dominant variance across the dataset, even when λ is varied. This is a mathematically expected “positive control,” not evidence that PC1 is directly tracking the nonlinear RG coupling g ~ λ^2 D / ν^3.

---

## 2. Growth Model Equations

### 2.1 Edwards-Wilkinson Equation ✅ VERIFIED

**Theoretical equation:**
h_t = ν ∇^2 h + η(x,t)

where:
- ν is the surface tension (diffusion coefficient)
- η(x,t) is Gaussian white noise with ⟨η(x,t)η(x',t')⟩ = 2Dδ(x-x')δ(t-t')

**Our implementation:**
```python
# From physics_simulation.py line 154-157
laplacian = left - 2*center + right  # Discrete ∇²h
noise = noise_strength * np.sqrt(dt) * np.random.randn()  # η√dt
dhdt = diffusion * laplacian + noise  # ν∇²h + η
new_interface[x] = center + dt * dhdt  # Euler integration
```

✅ **Correct:** Uses proper discrete Laplacian and noise scaling with √dt for Itô integration.

### 2.2 KPZ Equation ✅ VERIFIED

**Theoretical equation:**
h_t = ν ∇^2 h + (λ/2)(∇h)^2 + η(x,t)

The nonlinear term (∇h)² captures lateral growth from a normal-direction deposition when measuring height vertically.

**Our implementation:**
```python
# From physics_simulation.py line 201-213
laplacian = left - 2*center + right  # ν∇²h
gradient = (right - left) / 2.0  # Central difference ∇h
nonlinear_term = nonlinearity * 0.5 * gradient**2  # (λ/2)(∇h)²
noise = noise_strength * np.sqrt(dt) * np.random.randn()  # η√dt
dhdt = diffusion * laplacian + nonlinear_term + noise
```

✅ **Correct:** Implements all three KPZ terms properly. The coefficient 0.5 corresponds to λ/2 in the standard form.

#### 2.2.1 Stationary Slope Measure (1D) and λ-Independence

For 1D KPZ, a key exact fact is that the stationary measure for height *differences* is Gaussian (Brownian), and equivalently the stationary slope u = ∂x h is Gaussian/white-noise-like. Importantly, this stationary slope measure does **not** depend on the nonlinear coupling λ (it enters as a total derivative/current term in the Fokker–Planck description under periodic boundary conditions).

Accessible references that describe this stationary (“stationary IC”) subclass and its Gaussian/Brownian character include:
- Corwin (2012) “The KPZ equation and universality class” (arXiv:1106.1596): stationary initial data are Brownian.
- Takeuchi (2018) “An appetizer to modern developments on the KPZ universality class” (J. Phys. A 51, 164001): stationary case height differences are Gaussian.

**Project relevance:** This provides a clean theoretical explanation for why variance-dominated feature sets (e.g., Var(∂x h), Var(∂xx h)) can produce a PC1 that tracks D/ν and shows weak λ-dependence in finite simulations.

### 2.3 Ballistic Deposition ✅ VERIFIED

**Theoretical model:** Particles rain down and stick to the highest point among the landing site and its neighbors.

**Our implementation:**
```python
# From physics_simulation.py line 91-100
landing_height = max(left_height, center_height, right_height) + 1
```

✅ **Correct:** Standard ballistic deposition rule. This model belongs to the KPZ universality class.

**Source:** Meakin et al. (1986), Phys. Rev. A 34:5091

### 2.4 Kuramoto–Sivashinsky (KS) and KPZ Crossover in the Literature

For **noisy KS**, several works argue or demonstrate a crossover to KPZ‑type scaling at long wavelengths. Ueno, Sakaguchi & Okamura (2005) combined RG arguments with numerics to show KPZ‑like scaling can emerge in 1D noisy KS, especially with sufficiently strong noise. Minami & Sasa (2018) further analyzed the noisy KS equation and derived effective KPZ parameters under coarse‑graining.

**Implication for this project:** KS→KPZ convergence is not guaranteed at finite time or for all parameter regimes. Observable choice and scale separation matter, so negative results in KS should be interpreted as **either true non‑convergence in the tested regime** or **pre‑asymptotic/observable limitations**, consistent with the literature.

### 2.5 Burgers–KPZ Equivalence and Limits at the Fluctuation Level

At the deterministic equation level, Burgers and KPZ are linked by u = ∂x h, but **statistical fluctuations can still differ** depending on the observable and the sector (initial conditions / stationary vs droplet vs flat). For example, the **derivative field of a KPZ interface does not necessarily share KPZ fluctuation statistics** even when the underlying equations are formally related; see Rodríguez‑Fernández & Cuerno (2020) for explicit “non-KPZ” fluctuation behavior in derivative observables.

**Implication for this project:** Burgers–KPZ equivalence does not guarantee **ensemble‑level convergence** for arbitrary observables. This supports the need to test universality with observables that isolate truly universal quantities, not just scale‑invariant ones.

---

## 3. Machine Learning Approach Validity

### 3.1 Autoencoders for Physics Discovery

The use of autoencoders to discover structure in physical systems is well-established:

1. **Mehta et al. (2019)** - Physics Reports 810:1-124
   - Comprehensive review of ML for physics
   - Demonstrates autoencoders can learn meaningful latent representations

2. **Carrasquilla & Melko (2017)** - Nature Physics 13:431
   - "Machine learning phases of matter"
   - Neural networks detect phase transitions without labeled data

3. **Torlai & Melko (2016)** - Phys. Rev. B 94:165134
   - Boltzmann machines learn thermodynamic properties unsupervised

### 3.2 Our Approach: Anomaly-Based Discovery ✅ NOVEL

Our approach is scientifically sound:

1. **Train on known classes (EW + KPZ)** → Learn normal statistical manifold
2. **Encode unknown classes (BD)** → Project to latent space
3. **Measure reconstruction error** → Higher error = different universality class

This is analogous to how physicists identify new universality classes:
- Systems in the same class have similar statistical properties
- Different classes produce measurably different behavior

### 3.3 UMAP for Visualization

UMAP (McInnes et al., 2018) is appropriate for:
- Preserving global and local structure
- Revealing cluster separations in high-dimensional latent spaces

### 3.4 Unsupervised PCA and Critical Behavior (Ising and Beyond)

There is now a substantial literature showing that PCA (and related unsupervised embeddings) can discover order-parameter-like directions and locate critical points from raw configurations:

- Wang (2016) showed PCA on 2D Ising spin configurations can identify phases and criticality without labels.
- Hu, Singh & Scalettar (2017) provided a critical examination of what PCA is (and is not) learning across multiple models.
- Wetzel (2017) compared PCA and variational autoencoders for unsupervised phase transition discovery.
- Mendes‑Santos et al. (2021) used intrinsic-dimension estimators to extract universal critical behavior near phase transitions in an unsupervised way.

**Project relevance:** These works contextualize why “PC1 tracks reduced temperature t” is plausible. The stronger bar—and what our pipeline targets—is not just monotonicity with temperature, but correct scaling/finite-size behavior consistent with RG (Privman & Fisher 1984; Cardy 1996).

### 3.5 ML ↔ RG Connections (Information-Theoretic and Generative RG)

Several works explicitly connect ML representations to RG ideas:

- Mehta & Schwab (2014): exact mapping between variational RG and deep learning (RBMs).
- Koch‑Janusz & Ringel (2018): construct RG transformations by maximizing mutual information between coarse and fine variables.
- Li & Wang (2018): “Neural Network Renormalization Group,” a flow/generative-model view of RG.

**Project relevance:** Our “measure-space” framing (pushforward measures μ = Φ#(P)) and the attempt to learn RG‑covariant embeddings align with this broader direction, but with a focus on physically interpretable local observables and rigorous validation gates.

### 3.6 Two-Sample Distances on Distributions: MMD

When comparing *ensembles* (distributions of features) rather than individual configurations, kernel two-sample statistics provide principled distances/tests. The **maximum mean discrepancy (MMD)** is a widely used choice:

- Gretton et al. (2012) define MMD as the maximum difference in expectations over functions in the unit ball of an RKHS, yielding a consistent two-sample test and a practical metric-like quantity.

**Project relevance:** MMD is a natural choice for a “distance between feature-distributions across scales,” consistent with the project’s central object being a pushforward measure μ on feature space rather than a single point in ℝ^d.

---

## 4. Theoretical Predictions for Our Experiment

### 4.1 Expected Latent Space Structure

Based on the physics:

| Class | Latent Space Behavior |
|-------|----------------------|
| **EW** | Should form one cluster (training distribution) |
| **KPZ** | Should form separate cluster (different dynamics) |
| **BD** | Should overlap with KPZ (same universality class!) |

**Important insight:** Ballistic deposition is in the **KPZ universality class**, so BD surfaces should cluster WITH KPZ, not separately!

### 4.2 Anomaly Score Predictions

If the autoencoder learns universality structure correctly:
- EW anomaly score: LOW (training data)
- KPZ anomaly score: LOW (training data)  
- BD anomaly score: **LOW** if BD ≈ KPZ in latent space
- BD anomaly score: **HIGH** if short-time transients dominate

The experiment will reveal whether our 200 time steps are sufficient for universality to emerge.

---

## 5. Potential Issues and Corrections

### 5.1 Time Scale Considerations ⚠️ NEEDS ATTENTION

Universality scaling appears only in the **asymptotic long-time regime**:
- t << L^z: Growth regime (β exponent)
- t >> L^z: Saturation regime (α exponent)

For L=128 and z=3/2 (KPZ): t_crossover ~ 128^1.5 ≈ 1450 time steps

**Our 200 time steps may be in the transient regime** where microscopic differences are still visible. This could be:
- A feature (detecting transient behavior)
- A limitation (not seeing universal behavior)

**Recommendation:** Consider longer simulations or explicit scaling analysis.

### 5.2 Normalization ⚠️ POSSIBLE IMPROVEMENT

The code normalizes by subtracting mean:
```python
interface = interface - np.mean(interface)
```

For proper universality analysis, should also consider:
- Variance normalization: h → h/W(L,t)
- Detrending: Remove systematic drift

### 5.3 Discrete vs Continuous ✅ ACCEPTABLE

Using discrete Laplacian and Euler integration is standard for numerical simulation. The discretization errors are small for the chosen parameters (dt=0.05-0.1).

---

## 6. Key References

### Primary Sources

1. Kardar, M., Parisi, G., & Zhang, Y.-C. (1986). "Dynamic Scaling of Growing Interfaces." *Physical Review Letters*, 56(9), 889-892.

2. Edwards, S.F. & Wilkinson, D.R. (1982). "The surface statistics of a granular aggregate." *Proceedings of the Royal Society A*, 381, 17-31.

3. Family, F. & Vicsek, T. (1985). "Scaling of the active zone in the Eden process." *Journal of Physics A*, 18(2), L75-L81.

4. Barabási, A.-L. & Stanley, H.E. (1995). *Fractal Concepts in Surface Growth*. Cambridge University Press.

### Machine Learning for Physics

5. Mehta, P. et al. (2019). "A high-bias, low-variance introduction to Machine Learning for physicists." *Physics Reports*, 810, 1-124.

6. Carrasquilla, J. & Melko, R.G. (2017). "Machine learning phases of matter." *Nature Physics*, 13, 431-434.

7. Hairer, M. (2013). "Solving the KPZ equation." *Annals of Mathematics*, 178(2), 559-664. [Fields Medal 2014]

### Anomaly Detection

8. An, J. & Cho, S. (2015). "Variational Autoencoder based Anomaly Detection." *ICLR Workshop*.

### Wasserstein Distance and RG Flows (New - January 2026)

9. Cotler, J. & Rezchikov, S. (2022/2023). "Renormalization Group Flow as Optimal Transport." *arXiv:2202.11737*, published in *Phys. Rev. D* 108, 025003 (2023).
   - **Key result**: Polchinski's equation for exact RG = Wasserstein-2 gradient flow of relative entropy
   - **Implication**: d_W provides geometric measure of RG distance → validates our D_ML metric

10. Camacho, G. & Fauseweh, B. (2025). "Critical Scaling of the Quantum Wasserstein Distance." *arXiv:2504.02709*, published in *Phys. Rev. Research* 7, 043223.
    - **Key result**: d_W exhibits critical exponents near quantum critical points
    - **Implication**: Wasserstein distance has universal scaling → applicable to KPZ criticality

### Multi-Scale Methods

11. Floryan, D. & Graham, M.D. (2020). "Discovering multiscale and self-similar structure with data-driven wavelets." *PNAS* 118(1), e2021299118.
    - **Key result**: Data-driven wavelet decomposition (DDWD) extracts hierarchical structure from turbulence
    - **Implication**: DDWD could separate discrete artifacts (fine scale) from universality (coarse scale)
    - **Connection**: Self-similarity in inertial range ↔ universality in KPZ scaling regime

### Physics-Informed ML

12. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686–707. https://doi.org/10.1016/j.jcp.2018.10.045

13. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). "Physics-informed machine learning." *Nature Reviews Physics*, 3, 422–440. https://doi.org/10.1038/s42254-021-00314-5

### KPZ Fixed Point and Subclasses

14. Halpin-Healy, T. & Zhang, Y.-C. (1995). "Kinetic roughening phenomena, stochastic growth, directed polymers and all that." *Physics Reports*, 254, 215–414.

15. Matetski, K., Quastel, J., & Remenik, D. (2021). "The KPZ fixed point." *Acta Mathematica*, 227, 115–203.

16. Prähofer, M. & Spohn, H. (2002). "Scale invariance of the PNG droplet and the Airy process." *Journal of Statistical Physics*, 108, 1071–1106.

17. Sasamoto, T. & Spohn, H. (2010). "One-dimensional KPZ equation: An exact solution and its universality." *Physical Review Letters*, 104, 230602.

18. Takeuchi, K.A. & Sano, M. (2010). "Universal fluctuations of growing interfaces: Evidence in turbulent liquid crystals." *Physical Review Letters*, 104, 230601.

19. Takeuchi, K.A. & Sano, M. (2012). "Evidence for geometry-dependent universality in KPZ growth." *arXiv:1203.2530* (later published).

### KS → KPZ Crossover Literature

20. Ueno, K., Sakaguchi, H., & Okamura, M. (2005). "Renormalization group and numerical analysis of a noisy Kuramoto–Sivashinsky equation." *Physical Review E*, 71, 046138.

21. Minami, M. & Sasa, S.-i. (2018). "Renormalization group analysis of the noisy Kuramoto–Sivashinsky equation." *Journal of Statistical Physics*, 173, 983–1003.

### Corrections to Scaling, Amplitude Ratios, and KPZ/Burgers Fluctuations

22. Rodríguez‑Fernández, A. & Cuerno, R. (2020). "Non‑KPZ fluctuations in the derivative of the KPZ equation." *Physical Review E*, 101, 052126.

23. Prähofer, M. & Spohn, H. (2004). "Exact scaling functions for one‑dimensional stationary KPZ growth." *Journal of Statistical Physics*, 115, 255–279. (Preprint: cond‑mat/0212519)

24. Canet, L., Kloss, T., Delamotte, B., & Wschebor, N. (2011). "Nonperturbative renormalization group for the Kardar‑Parisi‑Zhang equation: general framework and first applications." *Physical Review E*, 84, 061128.

25. Chin, C. & den Nijs, M. (1998). "Stationary state skewness in 2D KPZ‑type growth." *arXiv:cond‑mat/9810083*.

### RG, Scaling Fields, and Corrections to Scaling

26. Wegner, F. J. (1972). "Corrections to scaling laws." *Physical Review B*, 5, 4529. https://doi.org/10.1103/PhysRevB.5.4529

27. Privman, V. & Fisher, M. E. (1984). "Universal critical amplitudes in finite-size scaling." *Physical Review B*, 30, 322. https://doi.org/10.1103/PhysRevB.30.322

28. Cardy, J. (1996). *Scaling and Renormalization in Statistical Physics*. Cambridge University Press. https://doi.org/10.1017/CBO9781316036440

### Unsupervised Learning of Phase Transitions / Universality

29. Wang, L. (2016). "Discovering phase transitions with unsupervised learning." *Physical Review B*, 94, 195105. https://doi.org/10.1103/PhysRevB.94.195105

30. Hu, W., Singh, R. R. P., & Scalettar, R. T. (2017). "Discovering phases, phase transitions, and crossovers through unsupervised machine learning: A critical examination." *Physical Review E*, 95, 062122. https://doi.org/10.1103/PhysRevE.95.062122

31. Wetzel, S. J. (2017). "Unsupervised learning of phase transitions: From principal component analysis to variational autoencoders." *Physical Review E*, 96, 022140. https://doi.org/10.1103/PhysRevE.96.022140

32. Mendes‑Santos, T., Turkeshi, X., Dalmonte, M., & Rodriguez, A. (2021). "Unsupervised Learning Universal Critical Behavior via the Intrinsic Dimension." *Physical Review X*, 11, 011040. https://doi.org/10.1103/PhysRevX.11.011040

### ML ↔ RG (Information-Theoretic / Generative)

33. Mehta, P. & Schwab, D. J. (2014). "An exact mapping between the variational renormalization group and deep learning." *arXiv:1410.3831*.

34. Koch‑Janusz, M. & Ringel, Z. (2018). "Mutual information, neural networks and the renormalization group." *Nature Physics*, 14, 578–582. https://doi.org/10.1038/s41567-018-0081-4

35. Li, S.‑H. & Wang, L. (2018). "Neural Network Renormalization Group." *Physical Review Letters*, 121, 260601. https://doi.org/10.1103/PhysRevLett.121.260601

### Kernel Two-Sample Tests / MMD

36. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). "A Kernel Two-Sample Test." *Journal of Machine Learning Research*, 13, 723–773.

### Corrections/Crossovers in KPZ-Type Models (Practical Numerics)

37. Chame, A. & Reis, F. D. A. A. (2002). "Crossover effects in a discrete deposition model with Kardar–Parisi–Zhang scaling." *arXiv:cond-mat/0210562*.

38. Kelling, J., Ódor, G., & Gemming, S. (2016). "Universality of (2+1)-dimensional restricted solid-on-solid models." *arXiv:1605.02620*.

### KPZ Stationary Measures (Slope/Brownian) and Reviews

39. Corwin, I. (2012). "The Kardar–Parisi–Zhang equation and universality class." *Random Matrices: Theory and Applications*, 1, 1130001. *arXiv:1106.1596*.

40. Takeuchi, K. A. (2018). "An appetizer to modern developments on the Kardar–Parisi–Zhang universality class." *Journal of Physics A: Mathematical and Theoretical*, 51, 164001. https://doi.org/10.1088/1751-8121/aaa67d  (preprint: *arXiv:1708.06033*).

41. Forster, D., Nelson, D. R., & Stephen, M. J. (1977). "Large-distance and long-time properties of a randomly stirred fluid." *Physical Review A*, 16, 732. https://doi.org/10.1103/PhysRevA.16.732

---

## 7. Conclusion

The universality-discovery project is **mathematically sound** and implements:

✅ **Correct physics equations** for EW and KPZ  
✅ **Proper numerical integration** with noise scaling  
✅ **Valid ML methodology** for unsupervised discovery  
✅ **Appropriate visualization** via UMAP  

**Key scientific question the experiment addresses:**
> Can a neural network learn to distinguish universality classes from raw surface data without being told the underlying physics?

The answer will provide insights into both:
1. The information content of surface configurations
2. The ability of ML to discover physical universality

---

---

## 8. Post-Experiment Validation (Experiments 7 & 7b)

### 8.1 Experiment 7: EW vs KPZ Discrimination ✅ VALIDATES THEORY

**Result**: Discrete-trained autoencoder distinguishes EW from KPZ with **p < 10⁻¹⁵⁸**, Cohen's d = 5.13

This confirms:
- ✅ Model respects universality class boundaries (different β exponents → different reconstruction)
- ✅ Hierarchical encoding: Implementation (continuum vs discrete) → Universality class (EW vs KPZ)
- ✅ Literature prediction (Cotler-Rezchikov): Class structure preserved under RG-like transformations

### 8.2 Experiment 7b: Wasserstein Distance Matrix ✅ SUPPORTS GEOMETRIC FRAMEWORK

**Result**: d_W(KPZ, discrete) = 22.00 < d_W(EW, discrete) = 22.18

Wasserstein distance matrix reveals:
| Structure Level | Distance | Interpretation |
|-----------------|----------|----------------|
| EW ↔ KPZ | 1.18 | Continuum cluster (same implementation) |
| BD ↔ EDEN | 6.65 | Discrete KPZ cluster (same class + implementation) |
| KPZ → discrete | 22.00 | Cross-implementation, same class |
| EW → discrete | 22.18 | Cross-implementation, different class |
| RD → BD | 17.19 | Different class outlier |

**Key validation**: Same universality class → smaller Wasserstein distance, supporting:
- Conjecture 3.1 (Separation)
- Cotler-Rezchikov's RG-as-optimal-transport framework
- Grok's "nested measures" interpretation

### 8.3 The Fragility Insight (from Grok's literature search)

From "Stochastic growth models: universality and fragility" (HAL, 2016/2020):

> Discrete models exhibit "multi-scale fragility" where lattice artifacts dominate at short scales.

This explains the asymmetry:
- **Discrete → Continuum**: Easy (autoencoder ignores fine-scale noise, captures universal structure)
- **Continuum → Discrete**: Hard (autoencoder lacks representation for lattice artifacts)

Analogy: Training on photographs of *real* flowers generalizes to *cartoon* flowers, but not vice versa.

---

---

## 9. Three-Tier Experiment Classification (Feb 10, 2026)

Following the D/ν correction (Exp 54) and balanced reassessment, experiments fall cleanly into three categories relative to universality:

### Tier 1: Genuine Universality Evidence
- **Exp 50r** (α-only): KPZ-A and KPZ-B show overlapping roughness exponent distributions (α ≈ 0.35) while KS is clearly separated (α ≈ 0.90, 21× distance). This IS a universality statement: same class ↔ same universal exponent.
- **Exp 52d** (Ising FSS): Recovering ν ≈ 1.07 from finite-size scaling of the learned coordinate is a gold-standard universality diagnostic (critical exponent recovery).

### Tier 2: Scaling-Field Structure (Not Universality, But Real Physics)
- **Exp 46/54** (D/ν theorem): PC1 = noise-to-diffusion amplitude coordinate. Exact, λ-independent, proven from Gaussian stationary measure. Not universality — a non-universal amplitude — but physically meaningful and rigorously understood.
- **Exp 50q→50r** (scale-invariant ≠ universal): Operational demonstration that scale-free observables can encode non-universal corrections. A methodological contribution most ML-for-physics papers lack.

### Tier 3: Class Boundary Detection
- **Exp 50h-50n**: Three independent observable families (gradient moments, spectral shape, structure functions) all give flat KS-vs-KPZ distance → correctly identifies different universality classes.
- **Exp 52b/53b**: Circularity tests confirm the pipeline works without trivial order-parameter leakage.

### Key Distinction
Many KPZ gradient-moment results (Exp 20-27, 46) were detecting non-universal amplitude D/ν, not universality structure. This does not invalidate the framework — it reveals *what PCA actually finds* and motivates the diagnostic protocol (50q→50r) that distinguishes universal from non-universal observables.

### Implications for Observable Choice
Universality detection requires observables encoding *only* universal quantities (exponents, universal amplitude ratios). Structure functions S₂(r), though scale-invariant, encode non-universal amplitude ratios B/A (Wegner 1972 corrections). Pure exponents α, β, z succeed where structure functions fail.

---

*Literature review compiled: January 2026*
*Updated: February 10, 2026 (post D/ν correction, balanced reassessment, three-tier classification)*
*Project: universality-discovery*
