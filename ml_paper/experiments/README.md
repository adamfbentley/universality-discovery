# ML Paper Experiment Scripts

This folder should contain only scripts that serve the ML-focused paper. They
should mostly aggregate, audit, or visualize existing artifacts rather than
launching new long simulations.

## Implemented Scripts

1. `mlp01_consolidate_evidence.py`
   - read existing JSON/CSV artifacts.
   - write `../tables/representation_performance.csv`.

2. `mlp02_hard_subset_audit.py`
   - compute ARI on all/no-KS/no-RD/no-KS-RD/EW-KPZ-BD-Eden/EW-KPZ subsets.
   - use stored labels/predictions where available and true refits where
     matrices exist.

3. `mlp03_exponent_clouds.py`
   - plot exponent vectors from exp70/exp71 tagged summaries.
   - write exponent-cloud instability figures.

4. `mlp04_false_positive_summary.py`
   - make the exp69-to-exp71 false-positive figure and table.

5. `mlp05_local_vs_global.py`
   - compare kNN/local metrics with HDBSCAN/KMeans global clustering.

6. `mlp06_positive_controls.py`
   - condense Ising/Potts controls into a short table.

7. `mlp08_cluster_hierarchy_controls.py`
   - test standard clusterers and centroid hierarchy controls on stored exp62
     features and exponent-vector artifacts.

8. `mlp09_exp70_matrix_subset_refits.py`
   - refit raw multi-L, cross-L, PCA-whitened, and exponent representations on
     hard subsets using the new exp70 `matched_matrices.npz` archives.

## Still Open

- `mlp04_false_positive_summary.py` for the polished exp69-to-exp71 narrative
  figure.
- `mlp06_positive_controls.py` for a compact Ising/Potts credibility table.
- Stronger learned quotient baselines if the manuscript claims more than a
  benchmarked failure mode.

## Rules

- Do not overwrite original `results_exp*` directories.
- Derived outputs go only to `ml_paper/results`, `ml_paper/figures`, or
  `ml_paper/tables`.
- Every script should accept `--dry-run` or be read-only by default until output
  paths are explicit.
- Every output table should include artifact provenance columns.
