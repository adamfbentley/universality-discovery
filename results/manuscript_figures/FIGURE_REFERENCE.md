# Manuscript Figure Reference

## Main Text Figures

### Figure 1: Low-Dimensional Manifolds
- **Source**: Experiment 20 (intrinsic_dimension)
- **Panels**: Need to generate
- **Key result**: d ≈ 2 for EW/KPZ in 6D moment space

### Figure 2: The Universality Axis  
- **Source**: Experiment 21 (coordinates_of_universality)
- **Files**:
  - figure2a_coordinates_4panel.png - Full 4-panel analysis
  - figure2b_coordinates_annotated.png - Annotated PC space
- **Key result**: PC1 separates with r = -0.956

### Figure 3: Robustness to Parameters
- **Source**: Experiment 22 (robustness_tests)  
- **Files**:
  - figure3a_generalization.png - BD/Eden generalization test
  - figure3b_nuisance_invariance.png - L,T variation test
  - figure3c_coordinate_free.png - Logistic regression test
- **Key result**: Separation robust across L,T (CV = 0.08)

### Figure 4: RG Convergence
- **Source**: Experiment 23 (gap_investigation) or Experiment 24
- **Files**:
  - figure4a_feature_distributions.png - Feature gap diagnosis
  - figure4b_scale_invariant.png - Normalized features
  - figure4c_rg_flow.png - Distance vs block size
  - figure4_alt_killer_plot.png - Exp 24 distance evolution
- **Key result**: 90% distance reduction under RG

### Figure 5: Tracy-Widom Validation
- **Source**: Experiment 26 (tracy_widom_validation)
- **Files**:
  - figure5_tracy_widom.png - Height fluctuation statistics
- **Key result**: KPZ skewness = -0.297 (theory = -0.29)

## Recommended Figure Selection for Paper

For a 4-figure paper (e.g., PRL):
1. **Figure 2a** (coordinates_4panel) - Shows the full story
2. **Figure 3b** (nuisance_invariance) - Robustness evidence  
3. **Figure 4c** or **figure4_alt** - RG convergence
4. **Figure 5** - Tracy-Widom validation

For a 5-figure paper (e.g., PRE):
- Add Figure 1 (intrinsic dimension) as opening figure

## Figure Captions

### Figure 1
Low-dimensional manifolds in gradient moment space. (A) PC1-PC2 scatter plot 
showing EW (blue), KPZ (orange), and BD (green) samples. (B) Intrinsic dimension
estimates from three methods (PCA, MLE, TwoNN). (C) Cumulative explained variance.
EW and KPZ have d ≈ 2 while BD has d ≈ 5.

### Figure 2  
The universality axis. (A) PC1 score vs model label showing near-perfect separation
(r = -0.956). (B) PC1-PC2 plane with class clustering. (C) Feature loadings on
principal components showing PC1 loads on variance terms. (D) PC1 distributions
for EW and KPZ showing non-overlapping clusters.

### Figure 3
Robustness of universality classification. (A) Generalization test showing BD/Eden
do not project onto KPZ cluster. (B) Cohen's d heatmap across (L,T) parameter space
showing stable separation (CV = 0.08). (C) Logistic regression achieves AUC = 1.000,
confirming classification is coordinate-free.

### Figure 4
Renormalization group convergence. Distance between BD and KPZ manifolds (green)
decreases by 90% under block coarse-graining (b = 1 to 16), while EW-KPZ distance
(blue) remains constant. This validates the RG picture of universality.

### Figure 5
Tracy-Widom validation. (A) Height fluctuation histograms for EW and KPZ with
Gaussian fits. (B) Q-Q plots vs Gaussian. KPZ exhibits Tracy-Widom signature
(skewness = -0.297 ± 0.03) matching theory (-0.29) to 2% accuracy.
