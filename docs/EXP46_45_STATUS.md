# Experiment 46 & 45 Status Update

**Date**: February 3, 2026

## Exp 46: Coupling Coordinate Calibration ✅ SUCCESS

### Hypothesis
Does PC1 track effective coupling g_eff = (λ²D/ν³)L?

### Results
- **Best correlation**: PC1 vs D/ν³ with **r = 0.857** (p < 10⁻²⁸)
- Linear g_eff: r = 0.738 (moderate)
- Log(g_eff): r = 0.662 (weaker)
- λ alone: r = 0.164 (very weak)

### Key Finding
**PC1 primarily tracks noise-to-dissipation ratio (D/ν³), not full KPZ coupling!**

### Physical Interpretation
1. Gradient moment features are **sensitive to fluctuation strength** vs smoothing
2. They **weakly encode nonlinearity** (λ dependence)
3. This validates "coupling coordinate" concept (PC1 tracks physics, not just classification)
4. But it's tracking the **fluctuation regime**, not the KPZ-specific nonlinearity

### Implications
- ✅ Validates Assessment 2's coupling coordinate hypothesis
- ⚠️ Features may need augmentation to capture λ-dependence
- ✅ Ready to proceed with RG-covariant embedding (Exp 45)
- RG-covariant learning should naturally find features capturing ALL relevant physics

---

## Exp 45: RG-Covariant Embedding Learning 🔄 RUNNING

### Status
- **Phase**: Training neural network (epoch 0/50)
- **Data**: 3000 training snapshots, 1000 validation
- **Architecture**: Conv1D → AdaptivePooling → Dense layers → 8D features
- **Learning**: Self-supervised RG covariance loss

### What It's Doing
Learning embedding Φ such that:
```
Φ(coarse_grain(h)) ≈ A·Φ(h) + b
```
across scales 2, 4, 8.

### Expected Outcomes

**If low RG loss (<0.1)**:
- We've found coordinates where RG has simple dynamics
- Profound: discovered natural coordinate system for RG flow

**If better separation than gradient moments**:
- Learned features > hand-crafted features
- Validates deep learning for physics

**If interpretable eigenstructure (A matrices)**:
- Can identify RG-relevant directions (|λ| > 1)
- Can identify marginal/irrelevant directions
- Direct connection to RG theory

### Timeline
- Training: ~10 minutes (50 epochs on CPU)
- Should complete by 15:45

---

## What This Means for the Project

### Immediate Impact
We've moved from "PC1 separates classes" (shallow) to "PC1 tracks physical coupling D/ν³" (deep).

### Next Steps After Exp 45
1. **If Exp 45 succeeds**: Write profound paper on RG-covariant embeddings
2. **If Exp 45 partial**: Still have coupling coordinate result from Exp 46
3. **Parallel**: Can start manuscript revisions (Sections IV-VI) while Exp 47-48 run

### Publication Path
- **Best case**: Physical Review Letters - "Learned Coordinates for Renormalization Group Flow"
- **Good case**: Physical Review E - "Self-Supervised Discovery of RG Structure"
- **Safe case**: J. Stat. Mech. - "Coupling Coordinates for Surface Growth Classification"

All viable, all publishable, all scientifically honest.

---

## Files Generated
- `experiments/46_coupling_coordinate_test.py` - Main coupling test
- `experiments/46b_log_scale_coupling.py` - Log-scale analysis
- `experiments/45_rg_covariant_embedding.py` - RG-covariant learning (RUNNING)
- `results/exp46_coupling_coordinate/` - Figures and data
- `results/exp45_rg_covariant/` - Will contain trained model + analysis

## Key Figures
- `coupling_collapse_test.png` - PC1 vs various coupling definitions
- `coupling_coordinate_comparison.png` - 6 different coordinates tested
- `best_coupling_coordinate.png` - D/ν³ correlation breakdown
- `rg_covariant_analysis.png` - (pending) Training curves, separation, eigenspectra
