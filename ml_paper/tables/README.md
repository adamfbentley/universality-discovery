# Tables For ML Paper

Generated tables are derived from archived `results_exp*/` artifacts and should
include provenance columns whenever possible.

- `representation_performance.csv/.md` - consolidated local/global evidence
  across surface-growth representations.
- `subset_ari_audit.csv`, `subset_ari_summary.csv/.md` - hard-subset ARI audits,
  including no-RD, no-KS, EW/KPZ/BD/Eden, and EW/KPZ.
- `local_global_gap.csv` - kNN/local signal versus HDBSCAN/KMeans global
  clustering gaps.
- `exp62_neighborhood_purity.csv` - nearest-neighbor purity curves for the
  baseline spatial representation.
- `exp62_feature_quotient_geometry.csv` - centroid and quotient-distance
  diagnostics for exp62 feature geometry.
- `exponent_quotient_metrics.csv`, `exponent_system_means.csv`,
  `exponent_system_stds.csv` - finite exponent-cloud diagnostics.
- `clusterer_control_rows.csv`, `clusterer_control_summary.csv` - standard
  clusterer stress tests.
- `hierarchical_merge_audit.csv`, `hierarchical_pairwise_distances.csv`,
  `hierarchical_merge_steps.csv` - centroid hierarchy and KPZ merge controls.
- `exp70_matrix_subset_refit_rows.csv`, `exp70_matrix_subset_refit_summary.csv/.md`
  - true subset refits from exp70 matrix archives.

Still planned:

- `false_positive_summary.md` - polished exp69 high point versus exp71 repeated
  estimates.
- `positive_controls.md` - Ising/Potts finite-size-scaling controls.
