# Experiment 45 Failure Analysis

**Date**: February 3, 2026  
**Status**: ❌ FAILED - Degenerate Solution

---

## What Happened

**Training Results**:
- RG covariance loss → 0.0000 (perfect!)
- Learned feature correlation: r = **nan** (collapsed)
- Baseline gradient moments: r = -0.17 (weak but functional)
- All eigenvalues ~10⁻⁷ (near-zero RG transformations)

**Diagnosis**: Network learned **trivial solution Φ ≈ 0** (constant features).

---

## Why This Is a Degenerate Solution

The loss function was:
```
L = ||Φ(coarse_grain(h)) - (A·Φ(h) + b)||²
```

**Trivial solution**: If Φ(h) = 0 for all h, then:
- Left side: Φ(coarse_grain(h)) = 0
- Right side: A·0 + b = b
- Loss: ||0 - b||² → minimized when b ≈ 0

Network found: Φ ≈ 0, A ≈ 0, b ≈ 0 → loss = 0 ✓

This satisfies the RG covariance constraint perfectly but is **physically meaningless**.

---

## Why Hand-Crafted Features Still Weak (r = -0.17)

The baseline gradient moments showed r = -0.17 (much weaker than expected). This is likely because:

1. **Validation data issue**: Only 10 samples per class (1000 total snapshots from 20 trajectories)
2. **Flat IC by default**: Simulations used flat IC → we know from Exp 27 this gives r ≈ -0.06
3. **Time-averaging artifact**: Using last 50 frames per trajectory dilutes signal

**Comparison to Exp 27**:
- Exp 27 (flat IC): r = -0.06
- Exp 45 baseline: r = -0.17
- Slightly better but still in "flat IC regime" where separation is weak

---

## How to Fix Exp 45: Anti-Collapse Strategies

### Option 1: Feature Normalization (Simplest)
Add constraint: ||Φ(h)|| = 1 (unit norm features)

```python
class RGCovariantNetwork(nn.Module):
    def forward(self, h):
        features = self.encoder(h)
        # Normalize to unit sphere
        features = F.normalize(features, p=2, dim=1)
        return features
```

**Why this helps**: Can't collapse to 0 anymore.

### Option 2: Contrastive Loss (SimCLR-style)
Maximize separation between classes while maintaining RG covariance.

```python
# Multi-objective loss
loss_rg = ||Φ(coarse(h)) - (A·Φ(h) + b)||²
loss_contrastive = -log[exp(sim(Φ(h_i), Φ(h_j)))/Σ_k exp(sim(Φ(h_i), Φ(h_k)))]
loss_total = loss_rg + λ * loss_contrastive
```

**Why this helps**: Forced to learn discriminative features.

### Option 3: Auxiliary Supervised Task (Multi-Task Learning)
Add classification head as regularization.

```python
# Dual objective
features = model.encoder(h)
logits = model.classifier(features)  # EW vs KPZ

loss_rg = RG_covariance_loss(features)
loss_class = CrossEntropy(logits, labels)
loss_total = loss_rg + α * loss_class
```

**Why this helps**: Classification forces non-degenerate features, RG loss adds structure.

### Option 4: Variance Regularization
Explicitly penalize low-variance features.

```python
loss_rg = RG_covariance(features)
loss_var = -log(var(features) + ε)  # Maximize variance
loss_total = loss_rg + β * loss_var
```

---

## Recommended Next Steps

### Immediate (Most Likely to Succeed):
**Exp 45b: Multi-Task RG-Covariant Learning**
- Add classification head (EW vs KPZ labels)
- Joint loss: `L = L_RG + 0.1 * L_classification`
- This leverages both RG structure AND class separation
- Expected result: Non-degenerate features with RG structure

**Timeline**: 1-2 hours to implement and run

### Alternative (More Novel):
**Exp 47: Information Geometry**
- Skip neural network entirely
- Use parametric Gaussian models on gradient moments
- Compute KL divergence / Fisher-Rao distances across scales
- More interpretable, no collapse issues

**Timeline**: 1 day to implement

### Pragmatic (Return to Immediate Path):
**Exp 37-39: IC Invariance Studies**
- Address manuscript concerns directly
- Deep learning can wait for follow-up paper
- Get publishable results in 2-3 weeks

---

## Key Insights

1. **Self-supervised RG loss alone is insufficient** - has trivial solution
2. **Exp 46 remains valid** - PC1 tracks D/ν³ (profound result!)
3. **Hand-crafted features work better than degenerate learned features** (not surprising)
4. **Flat IC used by default** - explains weak baseline correlation (r=-0.17)

## What Still Stands

✅ **Exp 46**: PC1 is coupling coordinate (r=0.857) - **this is the breakthrough**  
✅ **Assessment 2 framework**: Validated conceptually  
✅ **Path forward**: Clear fixes for Exp 45 or pivot to Exp 47

The deep theoretical path is **still viable** - we just hit a known ML pitfall (feature collapse) that has well-understood solutions.

---

## Decision Point

**Recommended**: Implement **Exp 45b (multi-task)** quickly (~1 hour) to see if non-degenerate solution exists.

**If Exp 45b also fails**: Pivot to **Exp 47 (information geometry)** - more interpretable, no neural network collapse issues.

**If time is critical**: Return to **Exp 37-39 (IC invariance)** - address manuscript immediately, deep path becomes follow-up paper.

All three paths are scientifically sound. Choice depends on your timeline and risk tolerance.
