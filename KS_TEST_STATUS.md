# KS Generalization Test - Status Update

**Date**: Feb 3, 2026  
**Context**: Testing if framework generalizes beyond KPZ-family

---

## What We Fixed

### ❌ Exp 50 (Original): INVALID
**Problem**: Features ~ 10^-10 (way too small)

**Root cause**: KS equation had **wrong sign on κ term**
- Incorrect: `∂h/∂t = -ν∇⁴h - κ∇²h ...` (both negative → pure damping)
- Correct: `∂h/∂t = -ν∇⁴h + κ∇²h ...` (κ term is anti-diffusion/destabilizing)

**Impact**: Simulation decayed to zero instead of maintaining chaotic dynamics
- Before fix: std(h) = 5×10^-4 (decaying)
- After fix: std(h) = 14 (chaotic stationary state) ✅

### ✅ Exp 50c: PROPER TEST (but flawed implementation)
**Correct question**: Does KS → KPZ under coarse-graining?

**Method**: Compute d(KS, KPZ) at scales b = 1, 2, 4, 8, 16, 32
- If KS ∈ KPZ universality at IR: distance should **decrease** with b
- If KS ≠ KPZ: distance flat or increases

**Result**: 
```
b=1:  d = 0.803
b=2:  d = 0.872
b=4:  d = 0.936
b=8:  d = 0.979

Slope: +0.085 (R² = 0.990, p = 0.005)
```

**⚠️ CRITICAL FLAW DISCOVERED (Exp 50e):**
- Bandwidth σ was **recomputed at each b** using median heuristic
- This creates **spurious trends** even for identical physics
- **Diagnostic C (KPZ vs KPZ)**: Shows +0.113 slope with recomputed σ, but -0.020 (flat) with fixed σ
- **Conclusion**: Exp 50c slope is **contaminated by bandwidth artifact**

**Status**: Result INVALID. Need to rerun with fixed bandwidth.

**⚠️ INTERPRETATION GUARDRAIL:**
This result is **protocol- and regime-conditional**. It tests whether KS approaches KPZ in our observable-induced metric under our coarse-graining map. It does **not** yet rule out KS→KPZ convergence under:
- Alternative coarse-graining (spectral low-pass vs block averaging)
- Alternative observables (spectral features, structure functions vs gradient moments)  
- Different parameter regimes (ν, κ, λ)
- Larger system sizes (L) or longer equilibration times
- Different sampling protocols

The clean negative result tells us **this particular path doesn't show convergence**, which narrows the search space.

---

## What This Means

### Valid findings:
1. ✅ KS simulator now works correctly (chaotic dynamics)
2. ✅ Proper RG-flow-in-measure-space test implemented
3. ✅ KS does NOT flow to KPZ in tested regime

### Open questions:
1. **Is this the right KS regime?**
   - Literature claims: 1D KS shows KPZ-like scaling at long wavelengths
   - But depends on: system size L, parameter regime (ν, κ, λ)
   - Tested: ν=1.0, κ=2.0, λ=1.0, L=256

2. **Feature scaling issue?**
   - KS raw features: m2 ~ 10^-3
   - KPZ (from Exp 46): m2 ~ 10^-4 (but **pre-standardized**)
   - Need to compare RAW features from both

3. **Observable appropriateness?**
   - Gradient moments work for KPZ (growth-driven)
   - KS has different physics (dispersion + anti-diffusion)
   - Might need: Fourier modes, correlation functions, or structure functions

---

## Next Steps (NOT rushing to publish)

### Immediate diagnostics (this week) - highest ROI:

**Diagnostic A: Coarse-grainer sanity** ⭐ HIGHEST PRIORITY
- Problem: Block averaging in feature space might be wrong RG map for KS
- Test: Use **spectral low-pass coarse-graining** instead
  - Method: Truncate high |k| modes, inverse FFT, extract features
  - If distance now DECREASES: framework generalizes, just wrong coarse-grainer
  - If distance still INCREASES: not a coarse-grainer artifact
- Why first: Isolates whether the issue is the RG map itself

**Diagnostic B: Observables sanity**
- Problem: Gradient moments might be KPZ-specific operators
- Test: Use **KS-native observables**:
  - Low-k spectral slope / power fraction
  - Correlation length from spectrum peak width
  - Structure function slope S(r) at large r
- If convergence appears with spectral features: gradient moments are KPZ-biased
- This tests if observable choice is the limiting factor

**Diagnostic C: Matched KPZ sanity**  
- Problem: Preprocessing/implementation mismatch can create fake scale trends
- Test: Generate fresh KPZ with **identical** setup:
  - Same L=256, sampling cadence, preprocessing
  - Same coarse-graining implementation
  - Verify d(KPZ, KPZ) is flat under coarse-graining
- This prevents subtle discretization artifacts

### After diagnostics: Parameter exploration (if needed)
4. **If diagnostics pass but distance still increases**:
   - Test MBE (conserved growth: ∂h/∂t ~ ∇²∇²h + noise)
   - Test anisotropic KPZ
   - Build taxonomy: which systems does framework work for?

5. **If KS can show KPZ-like behavior**:
   - Find the regime (parameters, system size)
   - Test if framework detects the crossover
   - This would be a **beautiful result**: framework detects IR universality

---

## Honest Assessment

### What Exp 50 was NOT:
- ❌ Test of "PC1 vs coupling" (wrong test logic for KS)
- ❌ "Variance split" as RG test (not a universality comparison)
- ❌ Rushed to "KPZ-specific" conclusion

### What Exp 50c IS:
- ✅ Correct test logic (d(KS,KPZ) vs scale)
- ✅ Found meaningful result (divergence, not convergence)
- ✅ Opens scientific questions (why diverge? wrong regime? wrong observables?)

### Current status:
**We have a working KS-KPZ comparison pipeline.**

The result (divergence) could mean:
1. Framework is KPZ-specific (respectable finding)
2. Wrong KS regime (need to explore parameters)
3. Wrong observables for KS (need different features)
4. KS-KPZ crossover exists but not at these scales

**This is EXACTLY the kind of exploration we should be doing.** We're testing the boundaries of the framework systematically.

---

## Key Technical Points

### KS Equation (correct form):
```
∂h/∂t = -ν∇⁴h + κ∇²h - (λ/2)(∇h)² + η
         ↑        ↑        ↑           ↑
    stabilize  destabilize nonlinear  noise
```

### KS Universality:
- Different from KPZ (no KPZ-like (∇h)² growth term)
- Known to show spatiotemporal chaos
- Some literature: KPZ-like scaling at long wavelengths (system-dependent)

### Test Implementation:
- **Pseudospectral method** (FFT for spatial derivatives)
- **ETDRK4 time integration** (exponential integrator for stiff PDEs)
- **Distance metric**: MMD (Maximum Mean Discrepancy)
  - RBF kernel: `K(x,y) = exp(-||x-y||²/(2σ²))`
  - Bandwidth: median heuristic (σ = median of pairwise distances)
  - Formula: `MMD² = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]`
- **Coarse-graining**: Block averaging in feature space
  - Method: Reshape to (n_blocks, block_size, n_features), average over block_size
  - **No rescaling applied** (just averaging)
  - Applied to standardized features
- **Feature standardization**: StandardScaler on combined KS+KPZ before distance computation

**Key limitation**: Block averaging in feature space may not be the right RG map for KS spectral dynamics. Diagnostic A will test this.

---

## What We're NOT Doing

❌ Rushing to "publish KPZ-specific result"  
❌ Giving up after one negative test  
❌ Ignoring parameter/observable exploration  

## What We ARE Doing

✅ Systematically testing framework boundaries  
✅ Fixing bugs when found (sign error in KS)  
✅ Using correct test logic (RG flow in measure space)  
✅ Being open to: framework works / doesn't work / needs refinement  

**This is how you find the deep mathematical structure.**
