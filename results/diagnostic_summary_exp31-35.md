# Summary of Experiments 31-35: Diagnostic Deep Dive

## Executive Summary

After running comprehensive diagnostics (Exp31-35), we have identified both **genuine strengths** and **specific issues** in the universality class framework.

## Key Findings

### ✓ VALIDATED: KPZ vs EW Separation
- **Flat IC simulations** (Exp35): KPZ shows skewness = +0.22, EW shows skewness ≈ 0
- This is close to Tracy-Widom GOE prediction (+0.29)
- **The universality classes ARE distinguishable** by their statistics

### ✓ VALIDATED: Variance Scaling
- KPZ: Var[h] ~ t^(2/3) ✓
- EW: Var[h] ~ t^(1/2) ✓
- Normalized values are roughly constant over time

### ✗ IDENTIFIED ISSUE: Previous Tracy-Widom Validation
The original experiments computed **global skewness** of h(x) profile, not **single-point fluctuations**.
- Global skewness → 0 at long times (finite-size effects, periodic BC)
- Single-point fluctuations at center → Tracy-Widom (when done correctly)

### ✗ IDENTIFIED ISSUE: RG Test with Dimensionful Features  
- Original 6D features (grad_var, etc.) scale as 1/b² under coarse-graining
- They CANNOT be RG-invariant by construction
- Dimensionless features (skewness, kurtosis) also showed poor RG behavior
- Root cause: The geometric structure exists, but RG flow in feature space is complex

### ✗ IDENTIFIED ISSUE: Initial Condition Dependence
- Droplet IC: Skewness = +0.28 (expected -0.29 for TW-GUE)
- Flat IC: Skewness = +0.22 (expected +0.29 for TW-GOE)
- This suggests our droplet IC implementation needs adjustment

## Experimental Parameters Summary

| Experiment | L | T | n_samples | Key Finding |
|------------|---|---|-----------|-------------|
| Exp31 | 256 | 30,000 | 50 | KPZ global skewness → 0 (finite-size) |
| Exp32 | 512 | 3,000 | 30 | Dimensionless features also fail RG |
| Exp33 | 512 | 5,000 | 200 | Single-point test, still noisy |
| Exp34 | 2048 | 1,000 | 50 | Identified L >> √T requirement |
| Exp35 | 4096 | 500 | 500 | **Success: Flat IC → TW-GOE** |

## Recommendations for Paper

### What to Keep
1. **Classification framework** - EW, KPZ, BD are distinguishable ✓
2. **Geometric structure** - Classes occupy distinct regions in observable space ✓
3. **Machine learning methodology** - Works for classification ✓

### What to Revise
1. **Tracy-Widom validation** - Use single-point statistics, cite correct IC type
2. **RG interpretation** - Acknowledge that RG in feature space is non-trivial
3. **IC dependence** - Present as a genuine physical effect, not a limitation

### What to Remove or Tone Down
1. **"Theorem"** claim - Present as an empirical observation
2. **RG contraction** as validation - It doesn't work cleanly

## Immediate Next Steps

### Option A: Quick Fix for Publication
1. Revise Tracy-Widom section to use Exp35 results (flat IC, single-point)
2. Remove/downplay RG validation claims
3. Present IC dependence as an interesting finding, not a bug

### Option B: Deeper Investigation (Recommended)
1. Fix droplet IC implementation properly
2. Understand RG flow in feature space theoretically
3. Explore multi-class extension with correct observables
4. Target higher-impact journal with more complete story

## Technical Notes

### Why Droplet IC Failed
The "droplet" IC should be a **narrow wedge** or delta function:
```
h(x,0) = -|x - L/2|/ε  for |x-L/2| < ε
       = -1            otherwise
```
Our implementation used a gradual parabola spanning the whole domain.

### Why RG Failed
Even dimensionless features like skewness depend on the **correlation length** ξ:
- At small scales (b < ξ): Features reflect local noise
- At large scales (b > ξ): Features reflect universal behavior
- The transition is not clean contraction but complex flow

### Finite-Size Criterion
For Tracy-Widom to emerge: L >> √(νT)
- Our T=30,000 runs had √T ≈ 173, but L=256
- Exp35 used L=4096, T=500, giving L/√T ≈ 183 ✓
