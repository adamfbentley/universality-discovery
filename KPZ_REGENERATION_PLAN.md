# Action Plan: Regenerate KPZ with Field-Level Matching

## Current KPZ Data Status

**What we have** (from Exp 46):
- **95 parameter combinations** (λ, ν, D varied independently)
- **Features only** (6D gradient moments: m₂, m₃, m₄, m₅, m₆, m₇)
- **No raw fields** stored
- **Simulation parameters**:
  - L = 128 (spatial lattice points)
  - T = 2000 timesteps
  - dt = 0.05
  - save_interval = 10
  - KPZ equation: ∂h/∂t = ν∇²h + (λ/2)(∇h)² + η

**Problem for Exp 50f**:
- Cannot apply spectral coarse-graining (no fields!)
- Sample count collapses: 95 → 47 → 23 → 11 → 5 at b=16
- Used feature-space averaging (not field-level) as workaround
- Creates asymmetry with KS (which uses field-level spectral coarse-graining)

## What Needs to Match KS Setup

**KS parameters** (from Exp 50f):
- L = **256** (spatial grid points) ← **DIFFERENT from KPZ's L=128**
- T = 2000 timesteps
- dt = 0.01
- Equation: ∂h/∂t = -ν∇⁴h + κ∇²h - (λ/2)(∇h)² + η
- Parameters: ν=1.0, κ=2.0, λ=1.0, noise_strength=0.1
- Generated: 20 trajectories × 50 snapshots = 1000 samples

**KS field-level coarse-graining**:
- Method: Spectral low-pass (FFT → keep |k| ≤ k_c → iFFT)
- Scales: k_c fractions = [1.0, 0.5, 0.25, 0.125, 0.0625] (b=1,2,4,8,16)
- Features extracted AFTER coarse-graining field

## Regeneration Plan

### Option A: Full Regeneration (Recommended)

Generate fresh KPZ data matching KS setup exactly:

**Parameters**:
- L = **256** (match KS)
- T = 2000
- dt = 0.01 (match KS)
- Fixed KPZ parameters: λ=?, ν=?, D=? (choose one "standard" KPZ regime)
  - Could use: λ=1.0, ν=1.0, D=1.0 (neutral regime)
  - Or match g_eff from KS if possible

**Sampling strategy**:
- Generate **20-30 trajectories** (match KS count)
- Save **every 20 steps** from stationary regime (skip transient)
- Target: **~1000-1500 samples** at b=1
- After coarse-graining: Still have >100 samples at b=16

**Field storage**:
- Store raw fields, not just features
- Apply spectral coarse-graining identically to KS
- Extract gradient moments from coarse-grained fields

**Code modifications**:
```python
def generate_kpz_fields_matched(n_trajectories=25, L=256, T=2000, 
                                lambda_=1.0, nu=1.0, D=1.0):
    """Generate KPZ matching KS setup."""
    all_fields = []
    
    for traj_idx in range(n_trajectories):
        trajectory = simulate_kpz_trajectory(
            L=L, T=T, 
            lambda_=lambda_, nu=nu, D=D,
            dt=0.01,  # Match KS
            save_interval=20
        )
        
        # Skip transient (first half)
        for t in range(len(trajectory) // 2, len(trajectory)):
            all_fields.append(trajectory[t])
    
    return np.array(all_fields)

# Then apply SAME coarse-graining as KS
def apply_spectral_coarsening_to_kpz(fields, scales):
    """Apply spectral low-pass at each scale."""
    features_by_scale = {}
    
    for scale in scales:
        coarse_fields = []
        for h in fields:
            h_coarse = coarse_grain_field_spectral(h, scale)
            coarse_fields.append(h_coarse)
        
        # Extract features from coarse-grained fields
        features = [extract_gradient_moments(h) for h in coarse_fields]
        features_by_scale[scale] = np.array(features)
    
    return features_by_scale
```

**Estimated compute**:
- 25 trajectories × 2000 steps × L=256 ≈ 2-3 minutes per trajectory
- Total: ~1 hour (reasonable)

### Option B: Minimal Fix (Quick Test)

If compute is limited, regenerate just **100-200 samples** at b=1:
- Fewer trajectories but ensure consistent sample count across b
- Still use field-level coarse-graining
- Test if trend changes

## Expected Outcomes After Regeneration

### If distance still shows +0.02 slope (flat/slight increase):
- **Robust result**: KS and KPZ don't converge in this regime with gradient moments
- Move to Diagnostic B (spectral observables)

### If distance now DECREASES:
- **Observable artifact**: Gradient moments work, but KPZ asymmetry was hiding convergence
- Framework generalizes!

### If distance INCREASES more:
- **Regime-specific**: This KS/KPZ combination genuinely diverges
- Try different parameters or accept boundary

## Implementation Priority

**Immediate** (this session):
1. Create `generate_kpz_matched()` function (L=256, dt=0.01, field storage)
2. Generate 25 trajectories → ~1000 samples
3. Apply spectral coarse-graining (same as KS)
4. Rerun Exp 50f comparison with symmetric data

**After regeneration**:
5. If still flat → Move to Diagnostic B (spectral observables)
6. Document final result in KS_DIAGNOSTIC_SUMMARY.md

## Technical Details to Match

| Parameter | KS (Exp 50f) | KPZ (Exp 46) | Need to Match |
|-----------|--------------|--------------|---------------|
| L (lattice size) | 256 | 128 | ✅ Change to 256 |
| dt | 0.01 | 0.05 | ✅ Change to 0.01 |
| T (steps) | 2000 | 2000 | ✅ Already matched |
| save_interval | 20 | 10 | ⚠️ Keep 20 |
| Coarse-graining | Spectral on fields | N/A (features only) | ✅ Implement spectral |
| Samples at b=1 | 1000 | 95 | ✅ Generate ~1000 |
| Samples at b=16 | 1000 | 5 | ✅ Keep ~1000 |

## Code Location

**New experiment**: `experiments/50g_regenerate_kpz_matched.py`
- Generate KPZ with L=256, dt=0.01
- Store fields, not just features
- Apply same spectral coarse-graining as KS
- Save as `results/kpz_fields_matched_L256/`

**Then update**: `experiments/50f_diagnostic_a_field_cg.py`
- Load new matched KPZ data
- Rerun comparison with symmetric protocol
- Check if slope changes

---

## Bottom Line

**Current limitation**: KPZ has only 5 samples at b=16 and uses feature-averaging, not field-level coarse-graining.

**Fix**: Regenerate KPZ with L=256, store fields, apply spectral coarse-graining symmetrically.

**Estimated time**: ~1 hour to generate, 10 minutes to rerun comparison.

**Payoff**: Know if +0.0185 slope is real or due to asymmetry.
