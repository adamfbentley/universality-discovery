# Preprint Submission Checklist

**Target**: arXiv (cond-mat.stat-mech)  
**Timeline**: Ready for submission after figure generation  
**Date**: January 20, 2026

---

## Documents Completed ✅

- [x] **MANUSCRIPT_DRAFT.md** - Complete paper draft (~6500 words)
- [x] **THEORETICAL_DERIVATION.md** - Mathematical proofs supporting main claims
- [x] **EXPERIMENT_LOG.md** - Full experimental record (26 experiments)

---

## Required Figures

### Main Text Figures ✅ ALL AVAILABLE

- [x] **Figure 2**: The universality axis (Exp 21)
  - `figure2a_coordinates_4panel.png` - Full 4-panel analysis ✅
  - `figure2b_coordinates_annotated.png` - Annotated PC space ✅
  - **Source**: Experiment 21

- [x] **Figure 3**: Robustness to parameters (Exp 22)
  - `figure3a_generalization.png` - BD/Eden generalization ✅
  - `figure3b_nuisance_invariance.png` - Cohen's d heatmap ✅
  - `figure3c_coordinate_free.png` - Logistic regression AUC ✅
  - **Source**: Experiment 22

- [x] **Figure 4**: RG convergence (Exp 23/24)
  - `figure4c_rg_flow.png` - Distance vs block size ✅
  - `figure4_alt_killer_plot.png` - Exp 24 version ✅
  - **Source**: Experiments 23, 24

- [x] **Figure 5**: Tracy-Widom validation (Exp 26) ✅
  - `figure5_tracy_widom.png` - Height fluctuation distributions
  - **Source**: Experiment 26

### Figure 1 (Intrinsic Dimension)
- [x] Experiment 20 has been run with numerical results
- [ ] Need to add figure saving to experiment 20 script
- **Key numbers available**: d(EW)=2.03, d(KPZ)=1.88, d(BD)=4.80

### Supplementary Figures (Priority 2)

- [ ] **Figure S1**: Full 6×6 correlation matrix
- [ ] **Figure S2**: PC3-PC6 scatter plots
- [ ] **Figure S3**: Time evolution of features
- [ ] **Figure S4**: Feature histograms
- [ ] **Figure S5**: RG trajectories in high-D space

---

## Action Items

### Immediate ✅ COMPLETE

1. [x] Complete theoretical derivation
2. [x] Write manuscript draft
3. [x] Compile all existing figures (10 figures available)
4. [x] Create figure reference document
5. [x] Run Experiment 20 for intrinsic dimension numbers

### Short-term (Before Submission)

6. [ ] Add figure generation to Exp 20 script (optional - numbers available)
7. [ ] Write detailed figure captions (draft in FIGURE_REFERENCE.md)
8. [ ] Compile all references with DOIs
9. [ ] Proofread manuscript for clarity
10. [ ] Check all equations for typos

### Medium-term (Before Submission)

11. [ ] Get co-author feedback (if applicable)
12. [ ] Spell check and grammar review
13. [ ] Verify all numerical values match experiments
14. [ ] Create LaTeX version for arXiv submission
15. [ ] Prepare submission cover letter

---

## arXiv Submission Requirements

### Manuscript Format
- [ ] LaTeX source (.tex file)
- [ ] Bibliography file (.bib)
- [ ] All figures as separate files (.pdf or .png)
- [ ] README with compilation instructions

### Metadata
- [ ] Title (finalized)
- [ ] Author list with affiliations
- [ ] Abstract (≤1920 characters)
- [ ] Subject categories: cond-mat.stat-mech (primary), cs.LG (secondary)
- [ ] Keywords list
- [ ] Comments field (optional)

### Technical Requirements
- [ ] Figures: 300 DPI minimum
- [ ] File size: <50 MB total
- [ ] Compilation: pdflatex or similar
- [ ] License: CC BY 4.0 (recommended)

---

## Post-Submission Plan

### After arXiv Posting

1. **Tweet/social media** announcement with key figure
2. **GitHub release** with full code and data
3. **Blog post** explaining results for broader audience
4. **Contact experimentalists** working on KPZ systems (turbulent liquid crystals, bacterial colonies)

### Journal Submission Options

**Tier 1 (High Impact)**:
- Physical Review Letters (PRL) - Fast, high visibility
- Nature Communications - Broader readership
- PNAS - Interdisciplinary appeal

**Tier 2 (Solid Physics Journals)**:
- Physical Review E - Natural fit for stat mech
- Journal of Statistical Physics - Specialized audience
- New Journal of Physics - Open access

**Tier 3 (ML/Interdisciplinary)**:
- Machine Learning: Science and Technology
- Journal of Physics: Complexity
- Physical Review Research

**Recommendation**: Start with **Physical Review E** (good fit, reasonable acceptance rate) or **Physical Review Letters** (if we think it's groundbreaking enough).

---

## Estimated Timeline

| Task | Duration | Deadline |
|------|----------|----------|
| Generate Figures 1-4 | 4 hours | Jan 20 |
| Supplementary figures | 2 hours | Jan 21 |
| Manuscript polish | 2 hours | Jan 21 |
| LaTeX conversion | 3 hours | Jan 22 |
| Final review | 2 hours | Jan 22 |
| **arXiv submission** | — | **Jan 22, 2026** |

After arXiv: 1 week buffer, then submit to journal (Jan 29).

---

## Key Strengths of This Work

1. **Validation triad**: Math (Tracy-Widom) + Geometry (manifolds) + Dynamics (RG)
2. **Practical impact**: Finite-time classification (10-100× speedup)
3. **Theoretical grounding**: Proved why PC1 separates classes
4. **Honest science**: 26 experiments, many negative results documented
5. **Reproducibility**: Full code + data will be public

---

## Potential Reviewer Concerns & Responses

### Concern 1: "Only two classes (EW, KPZ)?"

**Response**: These are the two most important surface growth classes, studied for 40+ years. Extension to more classes is straightforward (Section 5.5).

### Concern 2: "Only 1D interfaces?"

**Response**: 1D is where KPZ theory is most rigorous (Tracy-Widom proven). 2D extension is future work (Section 5.5) but method should generalize.

### Concern 3: "Why not just measure scaling exponents?"

**Response**: Our method works at **finite time/size** without asymptotic scaling. Table in Section 5.1 shows 10-100× speedup.

### Concern 4: "Isn't this just PCA?"

**Response**: Yes, but on the **right observables** (gradients). We show this is **theoretically necessary** (Section 4), not empirical luck. Autoencoders failed (Section 5.3).

### Concern 5: "No experimental data?"

**Response**: Agreed, this is simulation only. But Tracy-Widom validation (2% error) shows our simulations are trustworthy. Experimental application is next (Section 5.5).

---

## Collaboration Opportunities

Potential collaborators for follow-up work:
- **Experimentalists**: Takeuchi (Tokyo), Hallatschek (Berkeley) - KPZ experiments
- **Theorists**: Corwin (Columbia), Quastel (Toronto) - KPZ math
- **ML/Physics**: Carrasquilla (Vector Institute), Melko (Waterloo) - ML for physics

---

*Last updated: January 20, 2026*
