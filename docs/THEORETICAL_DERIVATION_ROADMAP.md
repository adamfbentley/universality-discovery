# Theoretical Derivation: Why PC1 ~ D/ν³ [SUPERSEDED]

**Date**: February 9, 2026  
**Status**: ⚠️ SUPERSEDED by THEORETICAL_DERIVATION_COMPLETE.md  
**Correction**: PC1 ~ D/ν (r=0.961), NOT D/ν³ (r=0.857)  

> **See [THEORETICAL_DERIVATION_COMPLETE.md](THEORETICAL_DERIVATION_COMPLETE.md) for the proven result.**
>
> The correct relationship is PC1 ~ D/ν, which follows from the EXACT Gaussian 
> stationary measure of the 1D KPZ gradient field: P_stat[g] ∝ exp(−ν/(4D) ∫ g² dx).
> This was validated numerically in Experiment 54.

---

## ORIGINAL ROADMAP (preserved for historical reference)

**Original Experiment 46 Result** (partially incorrect):
- PC1 correlates with D/ν³: r = 0.857 ← this is real but NOT the best fit
- **Corrected**: PC1 correlates with D/ν: r = 0.961 (much stronger!)
- PC1 loads primarily on: grad_var (+0.607), lap_var (+0.586), h_var (+0.536)

**Original Question**: WHY does PCA "discover" D/ν³?  
**Corrected Question**: WHY does PCA discover D/ν? → ANSWERED (exact KPZ theorem)

---

## THEORETICAL FRAMEWORK

### Starting Point: KPZ Equation in 1+1D

$$\frac{\partial h}{\partial t} = \nu \nabla^2 h + \frac{\lambda}{2} (\nabla h)^2 + \eta(x,t)$$

Where:
- h(x,t) is the interface height
- ν is the diffusion coefficient (surface tension)
- λ is the nonlinearity coefficient
- η(x,t) is Gaussian white noise: ⟨η(x,t)η(x',t')⟩ = 2D δ(x-x')δ(t-t')

**Key parameter**: D/ν³ = λ²/(2ν³) (dimensionless coupling)

### Our Observable Space: Gradient Moments

Define spatial gradient: g(x,t) = ∂h/∂x

**Our 6D feature vector** (from Exp 21):
1. grad_var = Var[g]
2. grad_skew = Skew[g]
3. grad_kurt = Kurt[g]
4. lap_var = Var[∇²h]
5. grad_lap_cov = Cov[g, ∇²h]
6. h_var = Var[h]

**Question**: How do these moments evolve under the KPZ dynamics? How does D/ν³ control them?

---

## DERIVATION STRATEGY: 3-Stage Approach

### **Stage 1: Moment Evolution Equations** (Weeks 1-2)

**Goal**: Derive ∂⟨gⁿ⟩/∂t from the KPZ equation

**Approach**: Ito calculus / Fokker-Planck

Starting from KPZ, compute evolution of g = ∂h/∂x:

$$\frac{\partial g}{\partial t} = \frac{\partial}{\partial x}\left[\nu \nabla^2 h + \frac{\lambda}{2} g^2 + \eta\right]$$

$$= \nu \nabla^2 g + \lambda g \frac{\partial g}{\partial x} + \frac{\partial \eta}{\partial x}$$

**Key observation**: The nonlinear term λg(∂g/∂x) couples gradients multiplicatively.

**Derive**:
1. d⟨g⟩/dt = 0 (by symmetry)
2. d⟨g²⟩/dt = f₁(ν, λ, D, L)
3. d⟨g³⟩/dt = f₂(ν, λ, D, L)
4. d⟨g⁴⟩/dt = f₃(ν, λ, D, L)

**Expected result**: Cumulants will depend on combinations like λ²/ν = 2D/ν (the KPZ ratio).

**Literature to check**:
- Fogedby (2005): "Langevin equations for continuous time Lévy flights" - moment closure methods
- Kardar, Parisi, Zhang (1986): Original KPZ paper - scaling arguments
- Prähofer & Spohn (2004): "Exact scaling functions for 1D driven interfaces" - might have moment results

---

### **Stage 2: Stationary Distribution Analysis** (Weeks 3-4)

**Goal**: Find steady-state gradient statistics in terms of (ν, λ, D)

**Approach**: Find the stationary Fokker-Planck solution

The gradient field g(x) in steady state satisfies a probability distribution P[g(x)].

**For EW** (λ=0):
- ∂h/∂t = ν∇²h + η → Gaussian gradient field
- P[g] ~ exp[-∫ g²/2σ² dx]
- Var[g] ~ D/ν (noise/diffusion balance)

**For KPZ** (λ≠0):
- Nonlinearity breaks Gaussianity
- Need to solve: ∂P/∂t = ℒ†P where ℒ† is adjoint Fokker-Planck operator
- Stationary solution: ℒ†P_stat = 0

**Key calculation**: Show that P_stat depends on λ²/ν not λ and ν separately.

**Dimensional analysis check**:
- [λ] = length^(-1/2) time^(-1/2)
- [ν] = length^2 time^(-1)
- [D] = length^2 time^(-1)

Dimensionless ratio: λ²/(νD) or equivalently D/ν³ after substituting D = λ²/(2ν).

**Expected result**: 
$$\text{Var}[g] \sim D^{1/3} \cdot f(D/\nu^3)$$

where f is some scaling function.

---

### **Stage 3: PCA Eigenvector Derivation** (Weeks 5-6)

**Goal**: Show that the covariance matrix C_ij of moment features has dominant eigenvector aligned with (D/ν³) direction.

**Setup**: 6×6 covariance matrix

$$C = \begin{bmatrix}
\text{Var}[\text{grad\_var}] & \text{Cov}[\text{grad\_var}, \text{grad\_skew}] & \cdots \\
\text{Cov}[\text{grad\_skew}, \text{grad\_var}] & \text{Var}[\text{grad\_skew}] & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}$$

**From Stage 2**: Each moment has functional dependence on (ν, λ, D).

**Key insight**: If we vary D/ν³ while holding other parameters fixed:
- All moments change
- Changes are correlated (all driven by same physical parameter)
- **Maximum variance direction = direction of D/ν³ variation**

**Mathematical argument**:

Let Φ = (Var[g], Skew[g], Kurt[g], Var[∇²h], Cov[g,∇²h], Var[h])

If Φ_i = F_i(D/ν³) for smooth functions F_i, then:

$$\frac{dΦ_i}{d(D/\nu^3)} = F_i'(D/\nu^3)$$

The covariance matrix in the D/ν³ direction is:

$$C_{ij} \propto \left\langle \frac{dΦ_i}{d(D/\nu^3)} \frac{dΦ_j}{d(D/\nu^3)} \right\rangle$$

**Eigenvector**: The vector v = (F_1', F_2', ..., F_6') is the dominant eigenvector.

**Prediction**: 
- PC1 ∝ (F_1', F_2', ..., F_6')
- The loadings (0.607, -0.004, 0.026, 0.586, ...) reflect the derivatives F_i'

**To prove**: Show that F_i' has the observed pattern (grad_var and lap_var dominate).

---

## DETAILED CALCULATION PLAN

### **Week 1-2: Moment Closure**

**Task 1**: Derive gradient moment hierarchy

Starting from:
$$\frac{\partial g}{\partial t} = \nu \nabla^2 g + \lambda g \frac{\partial g}{\partial x} + \xi(x,t)$$

where ξ = ∂η/∂x is space-derivative noise.

**Calculate**:

**Second moment** (variance):
$$\frac{d}{dt}\langle g^2 \rangle = 2\nu \langle g \nabla^2 g \rangle + 2\lambda \langle g^2 \frac{\partial g}{\partial x} \rangle + \langle \xi^2 \rangle$$

Use integration by parts and periodic boundary conditions:
- $\langle g \nabla^2 g \rangle = -\langle (\nabla g)^2 \rangle$ (negative definite)
- $\langle g^2 \partial_x g \rangle = 0$ (by antisymmetry)
- $\langle \xi^2 \rangle = 2D \partial_x^2 δ(0)$ (divergent - needs regularization)

**Regularization**: Consider spatial average over length L:
$$\text{Var}[g] = \frac{1}{L}\int_0^L g^2 dx - \left(\frac{1}{L}\int_0^L g dx\right)^2$$

**Expected steady-state balance**:
$$0 = -2\nu \langle (\nabla g)^2 \rangle + 2D k_{\text{max}}^2$$

where k_max ~ 1/a is the short-distance cutoff.

Result: $\langle g^2 \rangle \sim D/(\nu a)$ where a is lattice spacing.

**Third moment** (skewness):
$$\frac{d}{dt}\langle g^3 \rangle = 3\nu \langle g^2 \nabla^2 g \rangle + 3\lambda \langle g^3 \frac{\partial g}{\partial x} \rangle + \langle g \xi^2 \rangle$$

The nonlinear term: $\langle g^3 \partial_x g \rangle = \frac{1}{4}\partial_x \langle g^4 \rangle \neq 0$ (generates skewness!)

**Expected result**: In steady state, $\langle g^3 \rangle \propto \lambda \langle g^4 \rangle / \nu$

Since $\langle g^4 \rangle \sim (D/\nu a)^2$, we get:
$$\text{Skew}[g] \sim \frac{\lambda}{D/\nu a} \cdot \frac{(D/\nu a)^{3/2}}{1} \sim \frac{\lambda (D/\nu)^{1/2}}{\nu a^{1/2}}$$

Substitute D = λ²/(2ν):
$$\text{Skew}[g] \sim \frac{\lambda^2}{\nu^{3/2} a^{1/2}} \sim \frac{D}{\nu^{3/2} a^{1/2}}$$

**Key dimensionless ratio**: Skew[g] / Var[g]^{3/2} ~ λ/(D/ν)^{1/2} ~ (D/ν³)^{1/2}

---

### **Week 3-4: Fokker-Planck Solution**

**Task 2**: Solve for stationary distribution P[g]

The Fokker-Planck equation for g(x,t) is:

$$\frac{\partial P}{\partial t} = -\frac{\partial}{\partial g}\left[\left(\nu \frac{\partial^2 h}{\partial x^2} + \lambda g \frac{\partial g}{\partial x}\right)P\right] + D \frac{\partial^2 P}{\partial g^2}$$

**Simplification**: Consider the 1-point distribution (single x location):

In steady state:
$$0 = -\frac{\partial}{\partial g}[\mu(g) P] + D \frac{\partial^2 P}{\partial g^2}$$

where μ(g) is the effective drift (from deterministic terms + correlations).

**For KPZ**: The nonlinearity creates a drift μ(g) ~ λ⟨g²⟩ (needs self-consistency).

**Ansatz**: Try P(g) ~ exp[-V(g)/(2D)] where V(g) is effective potential.

Steady-state condition gives:
$$\frac{dV}{dg} = \frac{\mu(g)}{D}$$

**Expected**: V(g) ~ g²/2σ² - α g³ + β g⁴ (non-Gaussian with skew and kurtosis)

Coefficients α, β will depend on λ/ν ratio.

---

### **Week 5-6: Covariance Matrix Structure**

**Task 3**: Compute C_ij analytically

For each pair (Φ_i, Φ_j) where Φ = (Var[g], Skew[g], Kurt[g], Var[∇²h], ...):

$$C_{ij} = \langle (Φ_i - \langle Φ_i \rangle)(Φ_j - \langle Φ_j \rangle) \rangle$$

**From Stages 1-2**: We have Φ_i = F_i(D/ν³, L, ...)

Vary D/ν³ in a dataset:
$$C_{ij} \approx \text{Var}[D/\nu^3] \cdot \frac{\partial F_i}{\partial (D/\nu^3)} \frac{\partial F_j}{\partial (D/\nu^3)}$$

**The dominant eigenvector is**:
$$v_i \propto \frac{\partial F_i}{\partial (D/\nu^3)}$$

**Compute**: 
- ∂Var[g]/∂(D/ν³) — gradient variance sensitivity
- ∂Skew[g]/∂(D/ν³) — skewness sensitivity
- ∂Var[∇²h]/∂(D/ν³) — Laplacian variance sensitivity

**Prediction**: If Var[g] ~ (D/ν³)^α and Var[∇²h] ~ (D/ν³)^β, then:
$$v \propto (\alpha, \text{small}, \text{small}, \beta, \text{small}, \gamma)$$

**Match to observation**: PC1 = (0.607, -0.004, 0.026, 0.586, 0.000, 0.536)

This requires α ≈ β ≈ γ (all three variances scale similarly with D/ν³).

---

## VALIDATION STRATEGY

### **Check 1: Numerical Experiment**

**Exp 54**: Verify moment scaling

- Simulate KPZ with varying (ν, λ) but fixed D/ν³
- **Prediction**: All moments should collapse when plotted vs D/ν³
- **Test**: Plot Var[g] vs D/ν³ for (ν=1,λ=1), (ν=2,λ=√8), (ν=0.5,λ=0.5) — should all lie on same curve

**Exp 55**: Verify covariance scaling

- Construct C_ij for different parameter ranges
- Compute eigenvector v₁ (PC1)
- **Prediction**: v₁ should be invariant across different (ν,λ) choices, depend only on D/ν³ range

---

### **Check 2: Analytic Limit**

**Weak coupling limit**: λ → 0 (D/ν³ → 0)

- KPZ → EW
- Should recover Gaussian results
- Var[g] ~ D/ν, Skew[g] → 0

**Strong coupling**: λ → ∞ (D/ν³ → ∞)

- Highly nonlinear regime
- Check against numerical simulations
- May require resummation / RG analysis

---

### **Check 3: Connection to RG**

**From RG theory** (Dynamic RG, Forster-Nelson-Stephen 1977):

KPZ has RG flow:
$$\frac{d\tilde{\lambda}}{d\ell} = (1-d/2)\tilde{\lambda} - C \tilde{\lambda}^2 / \tilde{\nu}$$
$$\frac{d\tilde{\nu}}{d\ell} = z \tilde{\nu}$$

In 1D: the nonlinearity is **relevant** (flows to strong coupling).

**Fixed point**: λ* ~ ν^(3/2) (dimensional analysis).

**Connection to our result**: Near fixed point, D/ν³ ~ λ²/ν³ ~ ν^0 (dimensionless).

Our PC1 ~ D/ν³ direction corresponds to **distance from fixed point**.

**Interpretation**: PC1 measures "how KPZ-like" the system is relative to EW.

---

## LITERATURE TO CONSULT

### **Essential Papers**

1. **Kardar, Parisi, Zhang (1986)**: "Dynamic Scaling of Growing Interfaces"
   - Original KPZ paper
   - Scaling arguments for exponents

2. **Spohn (1991)**: "Large Scale Dynamics of Interacting Particles"
   - Rigorous derivation of KPZ from microscopic models
   - Chapter on moment methods

3. **Prähofer & Spohn (2004)**: "Exact Scaling Functions for One-Dimensional Stationary KPZ Growth"
   - Exact solutions for special cases
   - Tracy-Widom connection

4. **Fogedby (2005)**: "Lévy flights in random environments"
   - Moment closure techniques
   - Fokker-Planck methods

5. **Corwin (2012)**: "The Kardar-Parisi-Zhang equation and universality class"
   - Modern review
   - Connection to integrable probability

6. **Quastel & Spohn (2015)**: "The One-Dimensional KPZ Equation and Its Universality Class"
   - Comprehensive review
   - Height fluctuation theory (may have moment results)

### **Advanced Theory**

7. **Sasamoto & Spohn (2010)**: "One-dimensional KPZ equation: an exact solution and its universality"
   - Exact solution for flat IC
   - May contain moment formulas

8. **Amir, Corwin, Quastel (2011)**: "Probability distribution of the free energy of the continuum directed random polymer in 1+1 dimensions"
   - Connection to directed polymers
   - Free energy = height in KPZ

9. **Matetski, Quastel, Remenik (2021)**: "The KPZ fixed point"
   - Most recent exact results
   - Fixed point structure

---

## SUCCESS METRICS

### **Minimum Viable Derivation** (PRE-level)

✅ Show that Var[g], Var[∇²h], Var[h] all scale with same power of (D/ν³)
✅ Derive approximate moment evolution equations
✅ Validate numerically that moments collapse onto D/ν³ curves

**Result**: "Gradient moments are naturally controlled by D/ν³ coupling"

**Impact**: Solid PRE paper, explains "why PC1 works"

---

### **Strong Derivation** (PRL-level)

✅ Derive exact steady-state distribution P[g | D/ν³]
✅ Compute covariance matrix C_ij analytically
✅ Prove dominant eigenvector aligns with D/ν³ direction

**Result**: "PC1 is the natural coordinate for KPZ universality"

**Impact**: PRL if connected to RG theory

---

### **Breakthrough Derivation** (Nature Physics-level)

✅ Connect to KPZ fixed point structure (Tracy-Widom)
✅ Generalize to other universality classes (EW, MBE, etc.)
✅ Predict optimal observables for any growth process

**Result**: "A theory of observable geometry for universality classes"

**Impact**: Nature Physics if sufficiently general + validated experimentally

---

## TIMELINE & MILESTONES

### **Month 1: Moment Evolution** (Feb-Mar 2026)
- Week 1: Derive d⟨g²⟩/dt, d⟨g³⟩/dt
- Week 2: Numerical validation (Exp 54)
- Week 3: Write up moment hierarchy results
- Week 4: Check against literature (Fogedby, Spohn)

**Milestone**: Moment evolution paper draft

---

### **Month 2: Steady-State Analysis** (Mar-Apr 2026)
- Week 1: Solve Fokker-Planck approximately (perturbation theory)
- Week 2: Numerical validation (compare to simulation)
- Week 3: Dimensional analysis (confirm D/ν³ scaling)
- Week 4: Connect to RG fixed point

**Milestone**: Scaling theory derivation

---

### **Month 3: PCA Connection** (Apr-May 2026)
- Week 1: Compute covariance matrix from moment theory
- Week 2: Analytical eigenvector derivation
- Week 3: Match to empirical PC1 loadings
- Week 4: Write full theory paper

**Milestone**: Complete theoretical framework

---

### **Month 4: Generalization** (May-Jun 2026)
- Week 1: Extend to EW, Ising, other systems
- Week 2: Derive optimal observables for each class
- Week 3: Validate on 2D systems
- Week 4: Write generalization paper

**Milestone**: Universal theory of observable coordinates

---

## RISK ASSESSMENT & CONTINGENCIES

### **Risk 1: Moments don't scale cleanly with D/ν³**

**Mitigation**: 
- May depend on multiple parameters (D/ν³, L, T)
- Refine to "D/ν³ is dominant, but not only, direction"
- Still publishable: "PC1 tracks primary coupling coordinate"

### **Risk 2: Fokker-Planck is intractable**

**Mitigation**:
- Use perturbation theory (weak coupling λ → 0)
- Use strong coupling expansion (large λ)
- Numerical continuation between limits
- Still publishable: "Approximate theory validated numerically"

### **Risk 3: Can't match PC1 loadings exactly**

**Mitigation**:
- Show qualitative agreement (grad_var and lap_var dominate)
- Quantitative mismatch may be due to finite-size effects
- Still publishable: "Theory predicts structure, details differ"

### **Risk 4: Derivation only works for KPZ, not generalizable**

**Mitigation**:
- Focus on KPZ as proof-of-concept
- Frame as "template for other systems"
- Still publishable: PRE instead of Nature Physics

---

## COLLABORATION STRATEGY

### **Potential Collaborators**

1. **Herbert Spohn** (TU Munich)
   - Expert on KPZ equation
   - Co-author of Tracy-Widom papers
   - Could validate moment calculations

2. **Ivan Corwin** (Columbia)
   - Expert on KPZ universality class
   - Integrable probability perspective
   - Could connect to exact solutions

3. **Jeremy Quastel** (Toronto)
   - Co-author KPZ fixed point paper
   - May have unpublished moment results
   - Could accelerate derivation

4. **Tomohiro Sasamoto** (Tokyo Institute of Technology)
   - Exact solution methods
   - May have calculated gradient moments for special cases

**Approach**:
- Email with empirical result (PC1 ~ D/ν³, r=0.857)
- Ask: "Have you seen gradient moment scaling with D/ν³?"
- Propose collaboration: "We have numerics, need theory"

---

## FALLBACK: NUMERICAL THEORY

If analytic derivation proves too hard, we can still publish:

### **"Numerical Theory of Coupling Coordinates"** (PRE)

**Content**:
1. Numerical evidence that moments scale with D/ν³ (many parameter scans)
2. Empirical covariance matrix structure
3. Dimensional analysis explaining scaling
4. Connection to RG flow (qualitative)

**Impact**: Solid PRE paper, less prestigious but still valuable

**Timeline**: 2-3 months instead of 4-6

---

## NEXT IMMEDIATE STEPS

### **This Week** (Feb 9-15, 2026)

**Day 1-2**: Literature deep dive
- Read Kardar-Parisi-Zhang 1986 carefully
- Check Spohn 1991 Chapter 5 (moment methods)
- Search for "gradient moments KPZ" on arXiv

**Day 3-4**: Preliminary calculation
- Derive d⟨g²⟩/dt by hand
- Estimate steady-state balance
- Check dimensional analysis

**Day 5-6**: Numerical validation
- Run Exp 54 (moment scaling with D/ν³)
- Plot Var[g], Skew[g], Kurt[g] vs D/ν³
- Check if curves collapse for different (ν,λ) choices

**Day 7**: Decision point
- If numerics work → proceed with full derivation
- If numerics fail → reassess approach

---

## CONCLUSION: THE PRIZE

**If we succeed in deriving PC1 ~ D/ν³ from first principles**:

✅ Transforms empirical observation into theoretical understanding
✅ Provides predictive framework (no need for PCA, just compute derivatives)
✅ Generalizable to other systems (template for any growth process)
✅ Publishable in **Nature Physics / PRL** (theory + experiment)

**The payoff**: From "we found that ML works" → "we understand why it must work"

That's the difference between a methods paper and a theory paper.

**Let's do this.** 🎯

---

*Created: February 9, 2026*  
*Target completion: June 2026*  
*Estimated impact: Nature Physics-level if successful*
