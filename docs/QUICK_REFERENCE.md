# Quick Reference: Critical Actions

**Date**: Feb 3, 2026 | **Status**: Ready to Execute

---

## 🎯 Immediate Actions (This Week)

### 1. Experiment 37: Conditioning Per IC (Start Today)
**File**: `experiments/37_conditioning_per_ic.py`

**Quick Implementation**:
```python
# For each IC type separately:
for ic in ['flat', 'droplet', 'stationary']:
    # Subset data
    data_ic = data[ic_type == ic]
    # Fit PCA on subset only  
    pca = PCA().fit(data_ic)
    # Test separation
    r = correlation(pca.components_[0], labels)
    print(f"{ic}: r={r:.3f}")
```

**Expected Time**: 3 days  
**Success**: r > 0.8 for all IC types

---

### 2. Experiment 38: Invariant Features (Parallel)
**File**: `experiments/38_invariant_features.py`

**Key Features to Test**:
```python
# Dimensionless ratios (IC-invariant by construction)
f1 = grad_skew / (grad_var ** 1.5)  # Standardized skewness
f2 = grad_kurt / (grad_var ** 2)    # Standardized kurtosis  
f3 = lap_var / grad_var              # Curvature/gradient ratio

# Detrended (remove IC baseline)
f4 = grad_var - mean_grad_var_per_IC[ic]
```

**Expected Time**: 3 days  
**Success**: r > 0.7 with invariant features only

---

### 3. Experiment 39: Roughness Control (Quick)
**File**: `experiments/39_roughness_matched_control.py`

```python
# Generate EW/KPZ with matched σ_h
ew = generate_with_width(target=1.0)
kpz = generate_with_width(target=1.0)
assert abs(ew.std() - kpz.std()) < 0.05

# Extract features, test if separation persists
```

**Expected Time**: 2 days  
**Success**: Separation with matched roughness

---

## 📝 Manuscript Edits (Parallel with Experiments)

### Edit 1: Section IV - "Proof" → "Heuristic" (30 min)
```markdown
OLD: "We prove that gradient variance must separate..."
NEW: "We present a heuristic scaling argument suggesting..."
ADD: "Caveats: (1) assumes late time, (2) IC effects not controlled, 
      (3) finite-size corrections not accounted for"
```

### Edit 2: Tracy-Widom Hedge (30 min)
```markdown
ADD: "Note: EW single-point also shows skewness ≈ -0.299, indicating 
      finite-size contamination. This is a consistency check, not 
      definitive asymptotic validation."
```

### Edit 3: RG Reframe (30 min)
```markdown
REPLACE: "RG flow merges manifolds"
WITH: "Gradient moments show unexpected RG behavior - they are 
       RG-relevant operators whose variance grows under coarse-graining,
       not RG-invariant observables."
```

**Total Time**: 90 minutes of focused editing

---

## 🚨 Critical Issues Summary

| Issue | Evidence | Fix | Timeline |
|-------|----------|-----|----------|
| PC1 is IC-dependent | Exp 27: flat r=-0.06, droplet r=-0.98 | Exp 37-38 | Week 1 |
| "Proof" overstated | Reviewer: "handwavy ansatz" | Manuscript edit | 30 min |
| TW overinterpreted | EW also -0.299 skew | Manuscript edit | 30 min |
| RG doesn't merge | EW↔KPZ expands 45% | Manuscript edit | 30 min |

---

## ✅ Decision Points

**After Exp 37:**
- ✅ Success (r>0.8 all IC) → Write up, move to controls (Exp 39-40)
- ❌ Failure → Immediately start Exp 38 (invariant features)

**After Exp 38:**
- ✅ Success (r>0.7 invariant) → Strong result, write up
- ❌ Failure → Consider long-time study (Exp 43-44) or honest negative

**Decision by**: Feb 10 (end of Week 1)

---

## 📊 Expected Outcomes

**Best Case** (60%): Conditioning or invariant features work → Strong paper  
**Good Case** (30%): Partial success + long-time → Solid paper  
**Honest Negative** (10%): IC-dependent method → Still publishable

**All outcomes are scientifically valuable!**

---

## 🎓 Key Lessons from Assessment

1. **Overclaiming hurts**: "Universality axis" → "effective discriminator in IC ensemble"
2. **Honest science is valued**: Reporting Exp 27 failure mode increased credibility
3. **Theory must match rigor**: "Proof" → "heuristic argument"
4. **IC-dependence is the centerpiece**: Must demonstrate invariance or explain dependence

---

## 📁 File Locations

**Context Docs**:
- Full context: `PROJECT_CONTEXT_ONGOING.md`
- Action plan: `docs/ACTION_PLAN_POST_ASSESSMENT.md`
- This file: `docs/QUICK_REFERENCE.md`

**Experiments**:
- Template: `experiments/36_proper_droplet.py` (most recent)
- New: `experiments/37_conditioning_per_ic.py` (to create)
- Config: `src/simulation/config.py`

**Manuscript**:
- Main: `docs/main.tex`
- Sections to edit: IV (theory), V (TW), VI (RG)

**Results**:
- Exp 27: `results/exp27_ic_control/`
- New: `results/exp37_conditioning/` (to create)

---

## ⏰ Timeline

```
Week 1 (Feb 3-9):   Exp 37, 38, 39 + manuscript edits
Week 2 (Feb 10-16): Exp 40, 41 + integrate results
Week 3 (Feb 17-23): Add to manuscript, generate figures
Week 4 (Feb 24-Mar 2): Final polish, response document

Target: Revised preprint by March 2
```

---

## 🔬 Next Step RIGHT NOW

**Start Experiment 37**: Conditioning approach

1. Copy `experiments/36_proper_droplet.py` → `37_conditioning_per_ic.py`
2. Modify to loop over IC types
3. Run overnight simulation
4. Analyze results tomorrow

**Estimated start time**: 2 hours to set up and launch

---

**Questions? Check**:
- Technical details → `PROJECT_CONTEXT_ONGOING.md`
- Experimental design → `docs/ACTION_PLAN_POST_ASSESSMENT.md`
- Previous results → `docs/EXPERIMENT_LOG.md`
