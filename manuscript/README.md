# Manuscript: Machine Learning Universality

## Files

| File | Description | Target |
|------|-------------|--------|
| `universality_geometric.tex` | Short version (~4 pages) | PRL (Physical Review Letters) |
| `universality_geometric_PRE.tex` | Full version (~8 pages) | PRE (Physical Review E) |

## Compilation

### Using pdflatex:
```bash
pdflatex universality_geometric.tex
bibtex universality_geometric
pdflatex universality_geometric.tex
pdflatex universality_geometric.tex
```

### Using latexmk (recommended):
```bash
latexmk -pdf universality_geometric.tex
```

## Figures

Copy figures from `../results/manuscript_figures/` to this directory:

```bash
cp ../results/manuscript_figures/*.png ./figures/
```

Then add figure includes in the LaTeX:
```latex
\begin{figure}
\includegraphics[width=\columnwidth]{figures/figure2a_coordinates_4panel.png}
\caption{The universality axis...}
\end{figure}
```

## Figure Mapping

| LaTeX Figure | Source File | Description |
|--------------|-------------|-------------|
| Figure 1 | (intrinsic dimension - to generate) | Low-dimensional manifolds |
| Figure 2 | `figure2a_coordinates_4panel.png` | Universality axis |
| Figure 3 | `figure3b_nuisance_invariance.png` | Robustness heatmap |
| Figure 4 | `figure4_alt_killer_plot.png` | RG convergence |
| Figure 5 | `figure5_tracy_widom.png` | Tracy-Widom validation |

## arXiv Submission

1. Create submission folder:
```bash
mkdir arxiv_submission
cp universality_geometric_PRE.tex arxiv_submission/main.tex
cp -r figures arxiv_submission/
```

2. Package:
```bash
cd arxiv_submission
tar -czvf submission.tar.gz *
```

3. Submit to arXiv category: `cond-mat.stat-mech`

## Key Results for Abstract

- PC1 correlation with class: r = -0.956
- Tracy-Widom skewness: -0.297 ± 0.03 (theory: -0.29)
- RG convergence: 90% distance reduction
- Intrinsic dimension: d ≈ 2 for EW/KPZ
- Robustness: CV = 0.08 across L, T
