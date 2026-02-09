# Action Plan: Post-Assessment Revision Strategy

**Date**: February 3, 2026  
**Based on**: External scientific assessment of "Geometric Structure of Universality Classes"  
**Status**: Ready to execute

---

## Executive Summary

The assessment is **constructive and actionable**. The work is scientifically sound but overclaims "universality" when we have demonstrated "effective discrimination in specific IC ensemble."

**Key Message**: This is fixable with 4-6 weeks of focused work. The path is clear.

---

## Critical Issues Identified

### Issue 1: PC1 "Universality Axis" is IC-Dependent ⚠️

**Evidence**: Exp 27 showed:
- Flat IC: r = -0.060 (no separation)
- Droplet IC: r = -0.982 (perfect separation)
- PC1 loading vectors rotate with IC

**Consequence**: Cannot claim PC1 is a "universal coordinate" - it's protocol-dependent.

**Fix**: Demonstrate IC invariance via:
1. Conditioning (separate analysis per IC)
2. Invariant features (ratios, detrended)
3. Long-time convergence (flat IC at T→∞)

---

### Issue 2: Theoretical "Proof" is Overstated 📝

**Problem**: Section IV claims to "prove" gradient variance separates EW from KPZ via scaling argument.

**Reviewer critique**:
> "Handwavy scaling ansatz, not a derivation... the 𝐿 scaling in Eq. 11 is especially likely to be wrong... calling it a 'proof' will backfire."

**Fix**: 
- Retitle: "Heuristic Scaling Argument"
- Add caveats: "Not rigorous without controlling IC and finite-size effects"
- Cite existing results OR present as testable conjecture

**Timeline**: 1 day manuscript edit

---

### Issue 3: Tracy-Widom Validation Overinterpreted 🔬

**Problem**: Claimed TW skewness "validates asymptotic regime"

**Issue**: EW single-point also shows -0.299 (should be 0 for Gaussian)

**Reviewer critique**:
> "TW skewness here is not evidence of asymptotia; it's evidence your statistic isn't specific."

**Fix**:
- Downgrade to "consistency check"
- Acknowledge finite-size contamination
- Note: spatial vs ensemble averaging distinction

**Timeline**: 1 day manuscript edit

---

### Issue 4: RG Diagnostic Doesn't Support Narrative 🔄

**Problem**: Claimed coarse-graining reveals universality

**Evidence**:
- BD→KPZ: 15% contraction (not 90%)
- EW↔KPZ: 45% expansion (opposite direction!)

**Reviewer critique**:
> "Your block transform + whitening distance is not acting like an RG flow in this observable space."

**Fix**:
- Reframe: "Gradient moments are RG-relevant operators"
- Note: Separation persists but amplifies (discrimination works, universality story unclear)
- Optional: Try RG-invariant observables (S_2 scaling exponents)

**Timeline**: 1 day manuscript edit

---

## Experimental Priority Queue

### 🚨 **URGENT: Week 1 (Feb 3-9)**

#### Experiment 37: Conditioning Approach (3 days)
**Goal**: Does separation work within each IC family?

**Method**:
```python
for ic_type in ['flat', 'droplet', 'stationary']:
    # Subset data to single IC
    ew_ic = ew_data[ic == ic_type]
    kpz_ic = kpz_data[ic == ic_type]
    
    # Fit PCA on this subset only
    pca_ic = PCA().fit(combined_features_ic)
    
    # Compute r(PC1, model) within IC
    print(f"{ic_type}: r = {correlation}")
```

**Expected Outcome**:
- If r remains strong (>0.8) for ALL IC → separation is IC-robust
- If r varies wildly → need invariant features

**Files**:
- `experiments/37_conditioning_per_ic.py`
- Results → `results/exp37_conditioning/`

---

#### Experiment 38: Invariant Features (3 days)
**Goal**: Find IC-invariant coordinates

**Candidate Features**:
1. **Dimensionless ratios**:
   - `grad_skew / (grad_var)^{3/2}` (standardized skewness)
   - `lap_var / grad_var` (diffusion vs gradient ratio)

2. **Detrended statistics**:
   - `grad_var - <grad_var>_IC` (remove IC-specific baseline)
   - Compute IC-dependent means, subtract before PCA

3. **Temporal derivatives**:
   - `∂(grad_var)/∂t` (rate of change, not absolute value)

**Method**:
```python
# Dimensionless ratios
features_invariant = [
    grad_skew / (grad_var ** 1.5),
    grad_kurt / (grad_var ** 2),
    lap_var / grad_var,
    # ... more ratios
]

# Test separation with invariant features only
pca_inv = PCA().fit(features_invariant)
r_invariant = correlation(pca_inv.PC1, model_label)
```

**Success Criteria**: r > 0.7 with invariant features across all IC

**Files**:
- `experiments/38_invariant_features.py`
- Results → `results/exp38_invariant_features/`

---

#### Experiment 39: Roughness-Matched Control (2 days)
**Goal**: Show separation isn't just "roughness scale"

**Method**:
```python
# Generate EW and KPZ with matched interface width
ew_matched = generate_ew(target_width=sigma_target)
kpz_matched = generate_kpz(target_width=sigma_target)

# Verify: np.std(ew_heights) ≈ np.std(kpz_heights)

# Extract features, compute PC1
# Test if separation persists despite matched σ_h
```

**Expected**: Separation persists (because nonlinearity, not just scale)

**Files**:
- `experiments/39_roughness_matched_control.py`
- Results → `results/exp39_roughness_control/`

---

### 📊 **HIGH PRIORITY: Week 2 (Feb 10-16)**

#### Experiment 40: Variance-Normalized Features (2 days)
**Goal**: Remove absolute scale dependence

**Method**:
```python
# Relative roughness features
features_normalized = [
    grad_var / h_var,      # Relative gradient roughness
    lap_var / h_var,       # Relative curvature
    grad_skew,             # Already dimensionless
    grad_kurt,             # Already dimensionless
]
```

**Files**: `experiments/40_variance_normalized.py`

---

#### Experiment 41: Structure Functions (5 days)
**Goal**: Use RG-invariant observables

**Theory**: S_2(r) = ⟨|h(x+r) - h(x)|²⟩ ~ r^{2α}

**Method**:
```python
# Compute structure functions at multiple r
S2_values = []
for r in [1, 2, 4, 8, 16, 32]:
    S2_values.append(compute_structure_function(h, r))

# Extract scaling exponent α via log-log fit
alpha_ew = fit_exponent(S2_ew)
alpha_kpz = fit_exponent(S2_kpz)

# Use α as feature (should be IC-invariant by theory)
```

**Expected**: α_EW = 0.5, α_KPZ = 0.5 (both same!), but approach differs

**Challenge**: This might NOT separate if both converge to α=1/2

**Alternative**: Use crossover scales, not exponents

**Files**: `experiments/41_structure_functions.py`

---

### ⏳ **CONDITIONAL: Week 3 (Feb 17-23)**

#### Experiment 43-44: Extended Time Series (Only if needed)
**Goal**: Test if flat IC separates at T→∞

**Rationale**: Droplet IC may just reach separation faster

**Method**:
```python
for T in [1000, 3000, 5000, 10000, 20000]:
    simulate_ew_kpz(T=T, IC='flat')
    compute_pc1_separation()
    plot_r_vs_T()
```

**Decision Point**: Only do this if Exp 37-38 fail to find IC-invariant features

**Timeline**: 1 week per T value (compute-intensive)

---

## Manuscript Revision Tasks

### Task 1: Section IV Rewrite (1 day) 📝

**Current Title**: "Theoretical Derivation: Why Gradient Variance Separates Classes"

**New Title**: "Physical Argument: Expected Separation from KPZ Nonlinearity"

**Changes**:
```markdown
REMOVE:
- "We prove that..."
- "This rigorously establishes..."
- "Theorem: ..."

ADD:
- "We present a heuristic scaling argument..."
- "This suggests (but does not rigorously prove)..."
- "Caveats: This argument assumes (1) sufficiently late time, 
   (2) suppressed IC effects, (3) bulk scaling regime."
- "Empirical validation: Exp 21 confirms for droplet IC (r=-0.98), 
   but Exp 27 shows flat IC fails (r=-0.06), indicating 
   IC-dependence not accounted for in this heuristic."
```

**File**: `docs/main.tex` Section IV

---

### Task 2: Tracy-Widom Section Hedge (1 day) 🔬

**Current**: "Tracy-Widom validation confirms asymptotic regime"

**Revised**: "Tracy-Widom consistency check"

**Changes**:
```markdown
ADD NEW PARAGRAPH:
"We note that single-point height statistics show KPZ skewness 
≈ -0.297 (matching TW-GOE) but EW also shows ≈ -0.299 at the 
same system parameters. This indicates finite-size effects 
contaminate the single-point statistic, and this measurement 
should be viewed as a consistency check rather than definitive 
proof of asymptotic universality. Proper TW validation would 
require ensemble averaging over spatial positions or much 
larger L,T."
```

**File**: `docs/main.tex` Section V

---

### Task 3: RG Section Reframe (1 day) 🔄

**Current**: "RG flow merges discrete→continuum manifolds"

**Revised**: "Gradient moments show unexpected RG behavior"

**Changes**:
```markdown
REWRITE PARAGRAPH:
"Surprisingly, our block coarse-graining analysis (Exp 24) showed 
that EW↔KPZ distance INCREASES by 45% under spatial averaging, 
while BD→KPZ contracts only 15%. This suggests gradient moment 
observables are RG-relevant operators (whose variance grows under 
coarse-graining) rather than RG-invariant quantities. 

This does not invalidate our discrimination framework - separation 
persists robustly across scales - but indicates these observables 
may be closer to 'critical exponents' themselves rather than 
fixed-point coordinates. Future work should explore explicitly 
RG-invariant observables such as structure function scaling 
exponents S_2(r) ~ r^{2α}."
```

**File**: `docs/main.tex` Section VI

---

### Task 4: Add Exp 27 IC-Dependence (1 day) ⚡

**New Subsection**: "Initial Condition Dependence"

**Content**:
```markdown
"A critical test (Exp 27) revealed that the PC1 'universality 
axis' is strongly IC-dependent:

- Flat IC: r = -0.06, Cohen's d = -0.12 (no separation)
- Droplet IC: r = -0.98, Cohen's d = -10.31 (perfect separation)
- Stationary IC: r = -0.24, Cohen's d = -0.50 (marginal)

Furthermore, the PC1 loading vector itself rotates with IC 
(cosine similarity = 0.12 between flat and droplet), indicating 
the dominant variance direction changes.

This demonstrates that PC1 is not a universal coordinate in the 
strong sense, but rather an effective discriminator for specific 
IC ensembles. The IC-dependence may reflect that droplet/curved 
ICs break spatial symmetry and amplify the signature of the KPZ 
nonlinearity λ(∇h)², while flat IC preserves symmetries that 
suppress this signal at finite time/size.

[New experiments 37-38 investigating IC-invariant features]"
```

**File**: `docs/main.tex` - New subsection in Section VII (Results)

---

## Timeline Summary

```
Week 1 (Feb 3-9):
├─ Exp 37: Conditioning (3 days)
├─ Exp 38: Invariant features (3 days)
└─ Exp 39: Roughness control (2 days, parallel)

Week 2 (Feb 10-16):
├─ Exp 40: Variance-normalized (2 days)
├─ Exp 41: Structure functions (5 days)
└─ Manuscript edits: Tasks 1-3 (3 days, parallel)

Week 3 (Feb 17-23):
├─ Add Exp 27 results to manuscript (Task 4, 1 day)
├─ Incorporate Exp 37-41 results (3 days)
└─ Conditional: Start Exp 43 if needed

Week 4 (Feb 24-Mar 2):
├─ Generate updated figures (2 days)
├─ Response to critiques document (2 days)
└─ Final manuscript polish (3 days)
```

**Target**: Revised preprint by March 2, 2026

---

## Decision Tree

```
START: Exp 37 (Conditioning)
    │
    ├─ Success (r>0.8 for all IC) ──→ CLAIM: "Robust within IC families"
    │                                  ADD: Exp 39 control
    │                                  DONE: Strong result
    │
    └─ Failure (r varies) ──→ Exp 38 (Invariant features)
                             │
                             ├─ Success (r>0.7 invariant) ──→ CLAIM: "IC-invariant coords found"
                             │                                 ADD: Exp 40-41 validation
                             │                                 DONE: Very strong result
                             │
                             └─ Failure ──→ Exp 43-44 (Long time)
                                           │
                                           ├─ Success (flat IC separates at T=10k+) ──→ CLAIM: "Asymptotic separation"
                                           │                                            DONE: Acceptable result
                                           │
                                           └─ Failure ──→ HONEST CONCLUSION:
                                                          "Separation is IC-dependent.
                                                           Droplet IC enables discrimination,
                                                           flat IC does not at accessible scales.
                                                           Framework is effective but not
                                                           universal in strong RG sense."
                                                          DONE: Honest negative result (publishable)
```

---

## Expected Outcomes

### Best Case (60% probability)
- Exp 37 succeeds OR Exp 38 finds invariant features
- Manuscript claims: "IC-robust discrimination via [conditioning/invariant coords]"
- Paper strength: Strong empirical + honest limitations
- Outcome: Accept with minor revisions (PRE or J. Stat. Mech.)

### Good Case (30% probability)
- Exp 37-38 partially succeed (2 of 3 IC families)
- Long-time Exp 43 shows flat IC eventual separation
- Manuscript claims: "Discrimination in specific regimes"
- Outcome: Accept with major revisions

### Honest Negative (10% probability)
- All experiments fail to find IC invariance
- Manuscript pivots: "IC-dependent discrimination method"
- Still scientifically valuable (methods paper)
- Outcome: Accept in specialized journal (Phys. Rev. E Rapid Comm.)

**None of these outcomes are "failure"** - honest science is publishable!

---

## Success Metrics

### Minimal Success ✅
- [ ] Demonstrate conditioning OR invariant features work
- [ ] Reframe theory as heuristic
- [ ] Add roughness-matched control

### Strong Success ✅✅
- Above PLUS
- [ ] IC-invariant coordinates identified
- [ ] Separation quantified for all 3 IC families
- [ ] Manuscript revised with honest framing

### Exceptional Success ✅✅✅
- Above PLUS
- [ ] Theoretical understanding of IC-dependence
- [ ] RG-invariant observable (S_2 exponents) demonstrated
- [ ] Extension to real experimental data

---

## Communication Strategy

### For Manuscript Revision
**Framing**: "We discovered an effective geometric discrimination method whose performance depends on initial condition ensemble. This IC-dependence reveals important physics about how universality signatures emerge."

**Key Points**:
1. Honest about IC-dependence (Exp 27)
2. Investigated systematically (Exp 37-38)
3. Found [conditioning/invariant features/regime] that works
4. Explains why (physical mechanism for IC amplification)

### For Response to Reviewers
**Tone**: "Thank you for this constructive assessment. We agree the initial manuscript overclaimed 'universality'. We've strengthened the work by..."

**Structure**:
1. Acknowledge critique validity
2. Show additional experiments addressing concern
3. Reframe claims appropriately
4. Highlight what's now stronger

---

## Aspirational Path: Deep Theoretical Framework (Assessment 2)

### Overview

A second theoretical assessment provided a profound reframing: we're not "finding universality axes" but rather building **coordinate charts on spaces of coarse-grained measures**, where RG is literally a flow.

**Key Insight**: The right object is probability measures μ over height fields, not surfaces themselves. RG acts on these measures.

### New Experiments (Longer-Term)

#### Experiment 45: RG-Covariant Embedding Learning (2-3 weeks) 🔬

**Goal**: Find Φ such that Φ(coarse_grain(h)) ≈ A·Φ(h) + b

**Method**:
```python
# Self-supervised optimization
loss = 0
for h in trajectories:
    for scale_b in [2, 4, 8]:
        h_coarse = coarse_grain(h, block_size=scale_b)
        feature = Phi(h)
        feature_coarse = Phi(h_coarse)
        predicted_coarse = A_b @ feature + b_b
        loss += ||feature_coarse - predicted_coarse||²
```

**Success Criteria**: 
- Low loss → genuine RG structure exists
- RG becomes linear dynamical system in Φ-space
- Fixed points = attractors, relevant directions = unstable eigenvectors

**Impact**: If successful, this is **profound** - we've found coordinates where RG has simple dynamics.

---

#### Experiment 46: Coupling Coordinate Calibration (1 week) 🔬

**Goal**: Test if PC1 tracks effective coupling g_eff = (λ²D/ν³) ℓ^(2-d)

**Method**:
- Vary λ, ν, D **independently** (not just λ/ν ratio)
- Simulate multiple (λ,ν,D) combinations
- Compute constructed g_eff for each
- Plot Φ(h) against g_eff

**Success Criteria**: Data collapse onto single curve

**Impact**: If successful, PC1 is genuine **coupling coordinate** tracking crossover trajectory between weak/strong coupling regimes (not just classifier).

---

#### Experiment 47: Information-Geometric Distances (2 weeks) 🔬

**Goal**: Replace Euclidean PCA with Fisher-Rao / KL divergence framework

**Method**:
```python
# Fit parametric models to observables
def fit_model(data):
    # E.g., multivariate Gaussian
    return mu, Sigma

# Compute KL divergence
KL_distance = 0.5 * (log(det(Sigma2)/det(Sigma1)) 
                     + trace(inv(Sigma2) @ Sigma1)
                     + (mu2-mu1).T @ inv(Sigma2) @ (mu2-mu1)
                     - d)

# Measure at multiple coarse-graining scales
for scale in [1, 2, 4, 8]:
    compute KL_distance between EW and KPZ at scale
```

**Hypothesis**: 
- Relevant directions maintain/increase distinguishability
- Irrelevant directions fade

**Impact**: Could explain why our RG diagnostic looked messy - we used wrong metric (Euclidean instead of information-geometric).

---

#### Experiment 48: Domain-Adversarial IC Factorization (1 week) 🔬

**Goal**: Learn Φ maximally predictive of class, minimally predictive of IC

**Method**:
```python
# Adversarial training
Phi = FeatureExtractor()
ClassPredictor = Classifier()
ICPredictor = Classifier()

# Loss
loss_class = CrossEntropy(ClassPredictor(Phi(h)), true_class)
loss_ic = CrossEntropy(ICPredictor(Phi(h)), true_ic)
loss_total = loss_class - lambda * loss_ic  # Gradient reversal
```

**Success Criteria**: High class accuracy (>90%), low IC accuracy (<60%)

**Impact**: Disentangles class from IC sector - isolates truly universal features.

---

### Integration with Immediate Work

**Relationship**: These deep experiments are **complementary** to immediate priorities:

- **Immediate** (Exp 37-39): Address IC-dependence empirically (conditioning, invariant features, controls)
- **Deep** (Exp 45-48): Build theoretical framework explaining **why** IC-dependence exists and **how** to properly handle it

**Timeline**: 
- Weeks 1-4: Execute immediate priorities (IC invariance, adversarial controls, manuscript revisions)
- Weeks 5-8: If immediate experiments successful, begin deep theoretical framework
- Weeks 9-12: Integrate deep theory into revised manuscript

**Decision Point**: If Exp 37-38 **fail** (no separation even after conditioning/invariant features), then deep experiments become **critical** to salvage contribution. If Exp 37-38 **succeed**, deep experiments become **aspirational enhancement** for future follow-up paper.

---

## Key Insights from Assessment

### Assessment 1: Empirical Methods

### What Reviewer Values ✅
- Honest failure mode reporting
- Physically motivated features
- Interpretable framework
- Reproducible results

### What Reviewer Rejects ❌
- Overstated theoretical claims ("proof" → "heuristic")
- Universal claims without IC invariance demonstration
- Overinterpreting consistency checks as validation

### Actionable Advice
> "If you do only a few things, do these:"
> 1. IC invariance study as centerpiece
> 2. Replace "theorem/proof" language
> 3. Adversarial controls against variance triviality

**We're doing exactly this.** ✅

---

### Assessment 2: Deep Theoretical Framework

### Core Insight
> "The most credible deep link is not 'PCA finds an RG axis.' It's that you're accidentally building a finite-dimensional coordinate chart on a space of coarse-grained measures, and RG is literally a flow on that space."

### What This Means
- Each model + protocol defines probability measure μ_C,L,T
- Feature map Φ pushes this to measure on ℝ^d
- RG is map on measures: R: μ ↦ μ'
- Fixed points = universality classes (attractors)
- We're approximating Φ_#(Rμ) but generic Φ doesn't commute with R

### Six Deep Connections Identified
1. **RG-covariant embeddings** (most promising)
2. **Gradient variance tracks effective coupling** g_eff
3. **IC-dependence as sector decomposition** of fixed point
4. **Information geometry > Euclidean clusters**
5. **Projection artifacts explain messy RG**
6. **Concrete testable conjecture** about flow structure

### Impact on Project
- Provides **mathematical framework** for understanding what we're doing
- Explains **why** IC-dependence exists (sector decomposition)
- Suggests **how** to fix it (RG-covariant learning, information geometry)
- Turns potential failure into **profound contribution** if deep experiments succeed

---

## Notes for Future Reference

- This assessment is from someone who:
  - Understands the physics (TW statistics, RG theory)
  - Values honest science over hype
  - Provides constructive path forward
  - Expects rigor but acknowledges empirical contributions

- The critique is **strengthening** the work, not rejecting it

- Main message: "You have strong empirical results. Don't overclaim the theory. Be honest about IC-dependence. Do the IC invariance study. You'll be fine."

- Estimated final paper quality after revisions: **Strong PRE paper** (Physical Review E) or **Solid J. Stat. Mech.** paper

---

**END OF ACTION PLAN**

Next steps: Execute Exp 37 (conditioning approach) immediately.
