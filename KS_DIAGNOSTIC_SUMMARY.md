# KS Generalization Diagnostic Summary

**Date**: Feb 3, 2026  
**Goal**: Test if framework generalizes beyond KPZ to Kuramoto-Sivashinsky

---

## Three Experiments Compared

### Exp 50c: Initial KS→KPZ Test (FLAWED)
- **Method**: Feature-space block averaging, **recomputed bandwidth**
- **Result**: Slope = **+0.085** (distance increases 0.803 → 0.979)
- **Conclusion**: "KS diverges from KPZ"
- **Fatal flaw**: Bandwidth σ recomputed at each scale → artifact

### Exp 50e: Diagnostic C - Pipeline Validation
- **Test**: KPZ vs KPZ (identical physics) with same pipeline
- **Result with recomputed σ**: Slope = **+0.113** (spurious drift!)
- **Result with fixed σ**: Slope = **-0.020** (flat, as expected)
- **Conclusion**: **Bandwidth artifact confirmed**. Exp 50c result invalidated.

### Exp 50f: Diagnostic A - Proper Field-Level RG
- **Method**: Spectral low-pass on fields, then extract features, **fixed bandwidth**
- **Result**: Slope = **+0.0185** (distance 0.803 → 0.877, ~9% increase)
- **Conclusion**: **Roughly constant**. KS and KPZ maintain separation, no convergence but no dramatic divergence.

---

## Key Findings

### ✅ What We Fixed:
1. **KS simulation**: Sign error in equation (was decaying, now chaotic)
2. **Bandwidth artifact**: Fixed σ prevents spurious trends
3. **Coarse-graining**: Field-level (not feature-level) is proper RG

### 📊 Current Best Result (Exp 50f):
- **Distance trend**: Very slight increase (+0.0185 slope)
- **Physical interpretation**: KS and KPZ are **distinct but parallel** under this protocol
- **Not converging**: No evidence KS flows to KPZ in this regime
- **Not diverging dramatically**: Separation is stable, not exploding

### ⚠️ Caveats:
1. **KPZ limitation**: Using pre-extracted features, not regenerated fields
   - Ideally should regenerate KPZ with same L=256, sampling protocol
   - Current test uses feature-space averaging for KPZ b>1 (not ideal)

2. **Single KS regime**: Only tested ν=1.0, κ=2.0, λ=1.0
   - Literature suggests KS→KPZ convergence is regime-dependent
   - Need parameter sweep

3. **Observable choice**: Gradient moments natural for KPZ, may not capture KS physics
   - KS lives in Fourier space (dispersion + anti-diffusion)
   - Should try spectral observables (Diagnostic B)

---

## Interpretation

### What the +0.0185 slope means:
**Not a definitive "no"**, but **no evidence of convergence yet**.

Three possible explanations:
1. **Wrong regime**: KS parameters don't put it in KPZ basin
2. **Wrong observables**: Gradient moments miss KS→KPZ crossover
3. **Genuinely different**: KS doesn't flow to KPZ (also scientifically valid)

### Why this is progress:
- ✅ Eliminated measurement artifacts (bandwidth, improper coarse-graining)
- ✅ Established stable pipeline (Diagnostic C passes with fixed σ)
- ✅ Have baseline result: KS and KPZ are distinct at these scales
- ⏭️ Ready for systematic exploration (regimes, observables)

---

## Next Diagnostics

### Diagnostic B: KS-Native Observables
**Question**: Do spectral features show KS→KPZ convergence where gradient moments don't?

**Test observables**:
- Low-k spectral slope (power-law exponent)
- Spectral centroid / peak location
- Power in different k-bands
- Structure functions S₂(r), S₄(r)

**Why**: KS physics is spectral (ν k⁴ vs κ k²). Gradient moments might be insensitive to KS→KPZ crossover if it happens in Fourier space.

### Parameter Sweep (if Diagnostic B fails)
**Question**: Is there a KS regime where convergence occurs?

**Sweep**:
- ν/κ ratio (changes balance of stabilizing/destabilizing)
- System size L (IR behavior might need larger L)
- Noise strength (KPZ is stochastic, KS deterministic)

**Literature guidance**: Some papers claim KS shows KPZ-like scaling at long wavelengths in certain parameter regimes.

---

## Technical Details

### Exp 50f Implementation:
- **Field coarse-graining**: Spectral low-pass
  - FFT → keep |k| ≤ k_c → inverse FFT
  - k_c fractions: 1.0, 0.5, 0.25, 0.125, 0.0625 (b=1,2,4,8,16)
- **Feature extraction**: Gradient moments on coarse-grained fields
- **Distance metric**: MMD with RBF kernel
- **Bandwidth**: σ = 3.19 (estimated from b=1, **frozen for all scales**)
- **Standardization**: StandardScaler fitted on b=1 combined data, applied to all scales

### Remaining Limitations:
1. KPZ side uses feature-space averaging (not field-level) for b>1
   - Need to regenerate KPZ trajectories for fully fair comparison
2. Single KS parameter point
3. Single observable family (gradient moments)

---

## Status: Where We Are

### Honest assessment:
**We have not found evidence that KS flows to KPZ under our current protocol**, but we've also:
- Fixed critical bugs (KS equation, bandwidth artifact)
- Established valid measurement pipeline (Diagnostic C)
- Narrowed the question to: observables vs regime vs genuinely different

### Not yet tested:
- [ ] KS-native spectral observables
- [ ] Different KS parameter regimes
- [ ] Matched KPZ regeneration with fields
- [ ] Larger system sizes (L>256)
- [ ] Cross-system embeddings (can one model classify both?)

### Publication-ready?
**Not yet**. Current state:
- Strong negative control (C passes)
- Proper methodology (field-level RG, fixed σ)
- Inconclusive physics result (parallel, not convergent)

Need either:
1. Find convergence with different observables/regime → "framework generalizes"
2. Systematically rule out convergence → "KS boundary established" (also publishable)

**This is good science**: We're testing systematically, not rushing to conclusions.

---

## Bottom Line

**Exp 50c**: ❌ Invalid (bandwidth artifact)  
**Exp 50e**: ✅ Diagnostic passes (pipeline stable)  
**Exp 50f**: ⚠️ Inconclusive (no convergence, no dramatic divergence)

**Verdict**: Framework generalization to KS remains **open question**. Need Diagnostic B (observables) before concluding.
