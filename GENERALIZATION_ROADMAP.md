# Beyond KPZ: Testing Framework Universality

**Date**: February 3, 2026  
**Status**: Main research goal - determine framework scope  
**Question**: Does the coupling-coordinate / RG-relevance framework generalize beyond surface growth?

---

## The Scope Question

### What We've Proven (KPZ-Family Surface Growth)
✅ PC1 tracks coupling D/ν³ in KPZ systems (r=0.857)  
✅ Features yield RG-covariant embeddings (r=-1.000)  
✅ Information-geometric distances increase with scale (+0.31)  
✅ Linear separability across EW/KPZ/BD (100% logistic regression)

### The Critical Unknown
> **Does this framework identify coupling coordinates and RG structure in OTHER universality classes?**

**Two possible outcomes**:

**Scenario A: KPZ-Specific** (Still respectable)
- Gradient moments happen to align with D/ν³ for this particular family
- Method is powerful but limited to surface growth phenomena
- Contribution: Strong KPZ-family analysis tool

**Scenario B: Universal** (High-impact)
- Same observables (or natural generalizations) work across systems
- Framework identifies relevant operators generically
- Contribution: General method for discovering universality structure

---

## Test Systems for Generalization

### Tier 1: Related Growth Models (Nearest Test)

**1. Kuramoto-Sivashinsky (KS) Equation**
```
∂h/∂t = -∇⁴h - ∇²h - (∇h)²/2 + η
```
- **Universality class**: KS (different from KPZ)
- **Parameters**: Viscosity ν, dispersion coefficient
- **Physical coupling**: Similar structure to KPZ but with ∇⁴ term
- **Test**: Does PC1 track some combination of ν and dispersion?
- **Difficulty**: Medium (similar PDE structure)

**2. Molecular Beam Epitaxy (MBE)**
```
∂h/∂t = -∇²(∇²h) + λ∇²(∇h)² + η
```
- **Universality class**: Conserved KPZ / Villain-Lai-Das Sarma
- **Parameters**: Surface tension κ, nonlinearity λ
- **Physical coupling**: λ/κ ratio controls roughening
- **Test**: Does PC1 track λ/κ?
- **Difficulty**: Medium (conservation changes scaling)

**3. Anisotropic KPZ**
```
∂h/∂t = ν_x ∂²h/∂x² + ν_y ∂²h/∂y² + λ(∇h)² + η
```
- **Universality class**: Anisotropic KPZ
- **Parameters**: ν_x, ν_y, λ
- **Physical coupling**: Anisotropy ratio ν_x/ν_y, λ²/(ν_x ν_y)
- **Test**: Does framework detect anisotropy axis?
- **Difficulty**: Easy (extension of existing code)

### Tier 2: Different Non-Equilibrium Classes (Stronger Test)

**4. Burgers Equation (1D Turbulence)**
```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x² + f
```
- **Universality class**: KPZ via Cole-Hopf (h = ∫u dx)
- **Parameters**: Viscosity ν, forcing strength
- **Physical coupling**: Reynolds number, forcing scale
- **Test**: Does PC1 track coupling in velocity space?
- **Difficulty**: Medium (velocity vs height field)

**5. Reaction-Diffusion (Directed Percolation)**
```
∂ρ/∂t = D∇²ρ + aρ - bρ² + η
```
- **Universality class**: Directed percolation (DP)
- **Parameters**: Diffusion D, reaction rates a, b
- **Physical coupling**: Distance from critical point (a - a_c)
- **Test**: Does framework identify critical coupling?
- **Difficulty**: Hard (fundamentally different physics)

**6. Active Matter (Vicsek Model)**
```
v_i(t+1) = v₀ (⟨v_j⟩ + noise)
```
- **Universality class**: Active matter phase transition
- **Parameters**: Noise strength, alignment radius, density
- **Physical coupling**: Noise/alignment ratio
- **Test**: Does PC1 track order parameter?
- **Difficulty**: Hard (particle-based, not continuum)

### Tier 3: Equilibrium Critical Phenomena (Ultimate Test)

**7. Ising Model (2D)**
```
H = -J Σ s_i s_j - h Σ s_i
```
- **Universality class**: 2D Ising
- **Parameters**: Temperature T, magnetic field h
- **Physical coupling**: (T - T_c)/T_c, h/T
- **Test**: Does framework identify thermal coupling near criticality?
- **Difficulty**: Very hard (equilibrium vs non-equilibrium)

**8. XY Model (Kosterlitz-Thouless)**
```
H = -J Σ cos(θ_i - θ_j)
```
- **Universality class**: BKT transition
- **Parameters**: Temperature T, stiffness J
- **Physical coupling**: T/T_c
- **Test**: Does PC1 track approach to BKT transition?
- **Difficulty**: Very hard (topological transition)

---

## Observable Design Strategy

### Option A: Direct Translation (Gradient Moments)
Use same features but adapted to system:
- **KPZ**: Gradient moments m_k = ⟨|∇h|^k⟩
- **Burgers**: Velocity gradient moments ⟨|∂u/∂x|^k⟩
- **Reaction-diffusion**: Density gradient moments ⟨|∇ρ|^k⟩
- **Ising**: Magnetization gradient moments ⟨|∇m|^k⟩

**Test**: Do these generalized gradient moments capture coupling structure?

### Option B: Structure Functions (Scale-Invariant)
Use multi-scale observables:
- **General**: S_q(r) = ⟨|φ(x+r) - φ(x)|^q⟩
- **Advantage**: More universal, less system-specific
- **Test**: Does S_q(r) structure encode universality class?

### Option C: Correlation Functions (Statistical)
Use two-point correlators:
- **General**: C(r) = ⟨φ(x)φ(x+r)⟩ - ⟨φ⟩²
- **Fourier**: Power spectrum P(k)
- **Test**: Do correlation function shapes encode coupling?

### Option D: Learned Features (Data-Driven)
Train autoencoder on each system independently:
- **Architecture**: Same as Exp 45b but per-system
- **Test**: Do learned features converge to similar structure?
- **Ultimate test**: Transfer learning across systems

---

## Experimental Design: Minimal Viable Test

### Phase 1: Kuramoto-Sivashinsky (4 weeks)

**Week 1: Simulation Setup**
- Implement KS equation solver (spectral method)
- Generate data: 50 trajectories × 3 parameter regimes
- Compute gradient moments (same as KPZ)

**Week 2: Coupling Coordinate Test**
- PCA on KS gradient moments
- Test: Does PC1 correlate with ν or ν/dispersion ratio?
- Compare: KS vs KPZ PC1 structure (cosine similarity)

**Week 3: RG Structure Test**
- Information geometry: KL divergence vs scale
- Test: Do KS parameter regimes separate under coarse-graining?
- Compare: KS vs KPZ RG-relevance

**Week 4: Cross-System Test**
- Train on KPZ, test on KS (transfer learning)
- Train joint model on KPZ+KS
- Test: Can single embedding separate both?

**Success criteria**:
- ✅ Strong: PC1 tracks KS coupling with r>0.7
- ✅ Weak: Some PC tracks coupling with r>0.5
- ❌ Failure: No PC correlates with physical parameters

### Phase 2: MBE (if Phase 1 succeeds, 4 weeks)
Same protocol as KS

### Phase 3: Reaction-Diffusion (if Phase 2 succeeds, 6 weeks)
Same protocol, but fundamental physics differs more

---

## Decision Tree

```
Phase 1 (KS) Result:
├─ Strong success (r>0.7)
│  ├─ Continue to Phase 2 (MBE)
│  └─ Draft theory paper: "Universal coupling identification framework"
│
├─ Weak success (r>0.5)
│  ├─ Refine observables (structure functions)
│  └─ Continue to Phase 2 with caution
│
└─ Failure (r<0.3)
   ├─ Accept KPZ-family scope
   ├─ Publish current results as strong KPZ contribution
   └─ Explore why: Is it gradient moments? Or deeper?

Phase 2 (MBE) Result:
├─ Success → Phase 3 (RD)
└─ Failure → Surface-growth specific framework
    (Still valuable: unified KPZ-KS-MBE analysis)

Phase 3 (RD) Result:
├─ Success → UNIVERSAL FRAMEWORK (PRL-level)
└─ Failure → Growth-phenomena framework
    (Still strong: works across different growth classes)
```

---

## Timeline & Resource Estimate

**Optimistic (all succeed)**: 14 weeks to universal claim
**Realistic (some succeed)**: 8 weeks to "surface growth framework" claim  
**Conservative (KS only)**: 4 weeks to "beyond KPZ" demonstration

**Computational**: Moderate (KS/MBE similar to KPZ, RD cheaper)  
**Conceptual**: High (each system requires physics understanding)  
**Writing**: Major theory paper if universal, extensions if limited

---

## Expected Outcomes & Publication Strategy

### Outcome A: Universal (KS + MBE + RD succeed)
**Paper**: "Geometric Framework for Universality Discovery in Non-Equilibrium Systems"  
**Venue**: PRL or Nature Physics  
**Claims**: 
- General method for identifying relevant operators
- Coupling coordinates emerge from observable geometry
- RG structure encoded in information geometry

### Outcome B: Growth-Specific (KS + MBE succeed, RD fails)
**Paper**: "Unified Geometric Analysis of Surface Growth Universality"  
**Venue**: PRE or Physical Review Materials  
**Claims**:
- Framework works across growth phenomena (KPZ, KS, MBE)
- Gradient moments encode universality for interface dynamics
- Method applicable to thin-film growth, erosion, deposition

### Outcome C: KPZ-Family (Only KS succeeds partially)
**Paper**: "RG-Relevant Coordinate Charts for KPZ-Family Systems"  
**Venue**: PRE or J. Stat. Mech.  
**Claims**:
- Strong tool for KPZ-family analysis
- Coupling identification for growth equations
- Extensions to related models

---

## Immediate Next Steps (This Week)

1. **Literature review**: KS universality class, known scaling relations
2. **Code setup**: Spectral KS solver (pseudospectral method)
3. **Parameter space**: Identify relevant coupling combinations
4. **Baseline simulation**: Generate KS trajectories, validate scaling exponents
5. **Feature extraction**: Apply gradient moment pipeline to KS

**Deliverable**: First KS vs KPZ comparison plot showing PC1 correlation with coupling

**Decision point**: If Week 1 shows promising correlation (r>0.4), commit to full Phase 1

---

## Notes on Risk & Scope Acceptance

**If framework is KPZ-specific**:
- Still a major contribution (published paper stands)
- Provides deep understanding of ONE universality class
- Methods may inspire similar analyses for other classes
- Not a failure - it's defining a scope boundary

**If framework is universal**:
- Transforms into foundational methodology paper
- Opens field of "geometric universality discovery"
- Potential paradigm shift in non-equilibrium statistical mechanics

**Either way**: The question is worth answering definitively
