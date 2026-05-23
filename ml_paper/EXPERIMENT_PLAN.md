# Experiment Plan For A Cite-Worthy ML Paper

The paper should demonstrate a controlled failure mode of unsupervised physical
discovery, not advertise a new universality classifier.

## Research Question

When does a learned representation factor through the physical quotient, rather
than merely organize nuisance geometry?

In this benchmark, the physical equivalence relation is RG universality class.
The ML object is a finite-dimensional feature embedding of finite-size simulated
growth data. The core test is whether the embedding makes quotient labels
cluster-compatible, not merely locally predictable.

## Target Contribution

1. Define the distinction between:
   - local class-informativeness,
   - cluster compatibility,
   - physical/RG quotient compatibility.
2. Show these notions separate in 1+1D surface-growth data.
3. Document a high-ARI false positive from effective exponent geometry.
4. Provide a reproducible benchmark and validation checklist for ML universality
   claims.

## Experiment MLP-01: Evidence Consolidation Table

Purpose: produce one canonical table of every representation and its performance.

Inputs:

- `results_exp62/results.json`
- `results_exp65_robustness/summary.json`
- `results_exp66_finite_size/sweep_rows.csv`
- `results_exp67_height_dist/representation_rows.csv`
- `results_exp69_collapse_full/summary.json`
- `results_exp71_protocol_sweep/summary.json`

Outputs:

- `ml_paper/tables/representation_performance.csv`
- `ml_paper/tables/representation_performance.md`

Required columns:

- representation
- systems included
- sample protocol
- HDBSCAN core ARI
- KMeans ARI
- kNN accuracy if available
- notes on whether the representation is local, temporal, height, exponent, or
  multi-L

Decision criterion:

- Establish the repeated local-vs-global pattern: kNN can be high while
  unsupervised ARI remains near the ceiling.

## Experiment MLP-02: Hard-Subset Audit

Purpose: remove easy or ambiguous classes and test whether conclusions survive.

Subsets:

- all six systems: EW, KPZ, BD, Eden, RD, KS
- no KS
- no RD
- no RD and no KS
- EW/KPZ/BD/Eden only
- EW/KPZ only

Inputs:

- Existing labels and predictions where stored.
- Exp70/Exp71 labels and fitted labels.
- Rerun only if an artifact lacks labels required for subset ARI.

Outputs:

- `ml_paper/tables/subset_ari_audit.csv`
- `ml_paper/tables/subset_ari_summary.csv`
- `ml_paper/tables/subset_ari_summary.md`
- `ml_paper/figures/subset_ari_audit.png`
- `ml_paper/results/mlp02_summary.json`

Status:

- Completed as a derived audit in `ml_paper/experiments/mlp02_hard_subset_audit.py`.
- Exp62 spatial features and matched exponent vectors are true subset refits.
- Exp70 feature baselines are post-hoc restrictions of stored all-system
  partitions because their feature matrices were not archived.
- This limitation is addressed for new tagged matrix-archive runs by MLP-09.

Decision criterion:

- The paper cannot rely on RD or KS. The hard-core subsets must be shown
  explicitly.

## Experiment MLP-03: Exponent-Cloud Instability

Purpose: show that effective exponent geometry is not a stable RG quotient at
the tested sizes.

Inputs:

- `results_exp70_matched_codex_sweep_full_equal_seed*/summary.json`
- `results_exp70_matched_codex_sweep_full_exp69_seed*/summary.json`

Outputs:

- `ml_paper/figures/exponent_clouds_by_seed.png`
- `ml_paper/tables/exponent_cloud_means.csv`
- `ml_paper/tables/exponent_cloud_overlap.csv`

Required plots:

- alpha-beta panels by seed-start.
- alpha-z or beta-z panels by seed-start.
- highlight exp69 seed-start 69000 as the tempting high-ARI case.

Decision criterion:

- Demonstrate visually that exp69 is a favorable finite-size/protocol instance,
  not the stable geometry of the method.

## Experiment MLP-04: False-Positive Case Study

Purpose: make the paper's most memorable ML lesson precise.

Inputs:

- `results_exp69_collapse_full/summary.json`
- `results_exp70_matched_codex_full_exp69sampling_20260522/summary.json`
- `results_exp71_protocol_sweep/summary.json`

Outputs:

- `ml_paper/figures/false_positive_protocol_sweep.png`
- `ml_paper/tables/false_positive_summary.md`

Message:

- The same exponent method gives ARI 0.902 under one protocol, but its mean
  matched advantage over raw multi-L features is approximately zero over five
  seed-starts.

## Experiment MLP-05: Local Vs Global Geometry

Purpose: formalize the distinction between local class-informativeness and
cluster compatibility.

Inputs:

- Exp66 and Exp67 rows with kNN, KMeans, HDBSCAN.
- Exp62 feature matrix if needed for new geometry diagnostics.

Outputs:

- `ml_paper/figures/local_vs_global.png`
- `ml_paper/tables/local_global_gap.csv`
- `ml_paper/tables/exp62_neighborhood_purity.csv`
- `ml_paper/tables/exp62_feature_quotient_geometry.csv`
- `ml_paper/results/mlp05_summary.json`

Status:

- Completed as a derived audit in `ml_paper/experiments/mlp05_local_vs_global.py`.
- Mean kNN3-minus-HDBSCAN gap across rows with kNN measurements is `0.389`.
- Exp62 has 1-neighbor universality purity `0.842`, KMeans ARI `0.185`, and
  HDBSCAN core ARI `0.495`.
- For KPZ-labeled exp62 points at `k=1`, only `0.0125` of nearest neighbors are
  other KPZ-class microscopic systems; local KPZ purity is mostly same-system
  locality, not quotient mixing.

Metrics:

- kNN accuracy
- HDBSCAN core ARI
- KMeans ARI
- local-global gap = kNN accuracy - HDBSCAN core ARI

Decision criterion:

- Show that the data are not signal-free; rather, the representation's global
  topology is incompatible with universality clustering.

## Experiment MLP-06: Positive Control Compression

Purpose: keep positive controls as credibility anchors without letting them
dominate the paper.

Inputs:

- `results_exp52d_full/results.json`
- `results_exp57c_pilot/summary.json`

Outputs:

- `ml_paper/tables/positive_controls.md`

Message:

- The pipeline can recover clean FSS structure when the observable is appropriate.
  The surface-growth failure is therefore not explained by generic incompetence
  of clustering or code.

## Experiment MLP-07: Optional Stronger Learned Baselines

Purpose: address the strongest ML-reviewer objection: simple feature baselines
are not the whole space of learned representations.

Candidate baselines:

- supervised probe only for diagnostic, not headline.
- self-supervised contrastive learning with augmentations that preserve system
  identity but vary seed/time crop.
- domain-adversarial baseline that removes model-specific nuisance amplitude.
- multi-L learned representation that sees per-L blocks and can learn slopes.

This is optional for first submission if the paper is framed as a benchmark and
not a proof that no learned representation can work.

## Experiment MLP-08: Clusterer And Hierarchy Controls

Purpose: test whether the quotient-failure claim is merely an artifact of
KMeans/HDBSCAN or of refusing to hierarchically merge microscopic subclusters.

Inputs:

- `results_exp62/features.npz`
- `results_exp70_matched_codex_sweep_full_equal_seed*/summary.json`
- `results_exp70_matched_codex_sweep_full_exp69_seed*/summary.json`

Outputs:

- `ml_paper/tables/clusterer_control_rows.csv`
- `ml_paper/tables/clusterer_control_summary.csv`
- `ml_paper/tables/hierarchical_merge_audit.csv`
- `ml_paper/tables/hierarchical_pairwise_distances.csv`
- `ml_paper/tables/hierarchical_merge_steps.csv`
- `ml_paper/figures/clusterer_control_best_ari.png`
- `ml_paper/figures/system_vs_universality_alignment.png`
- `ml_paper/figures/hierarchical_kpz_merge_audit.png`
- `ml_paper/results/mlp08_summary.json`

Status:

- Completed as a derived audit in
  `ml_paper/experiments/mlp08_cluster_hierarchy_controls.py`.
- Standard fixed-k clusterers do not rescue exp62 features: best all-six ARI is
  `0.250`, and EW/KPZ/BD/Eden remains `-0.0687`.
- Exp62 centroid hierarchy does not cleanly merge KPZ/BD/Eden before non-KPZ
  mixing; clean hierarchy fraction is `0` and the EW/KPZ/BD/Eden quotient ratio
  is `37.0`.
- Matched exponent vectors are better for EW/KPZ only, but still weak on the
  EW/KPZ/BD/Eden quotient: best fixed-k ARI is about `0.50` under equal sampling
  and `0.46` under exp69 sampling.

Remaining limitation:

- Exp70 feature matrices were not stored, so true subset refits of raw/cross-L
  feature baselines still require a tagged rerun that archives those matrices.

## Experiment MLP-09: Exp70 Matrix-Archive Subset Refits

Purpose: close the MLP-02 limitation by rerunning exp70 with archived feature
matrices and refitting feature/exponent baselines after selecting hard subsets.

Inputs:

- `results_exp70_matched_codex_matrix_full_equal_seed*/matched_matrices.npz`
- `results_exp70_matched_codex_matrix_full_exp69_seed*/matched_matrices.npz`

Outputs:

- `ml_paper/tables/exp70_matrix_subset_refit_rows.csv`
- `ml_paper/tables/exp70_matrix_subset_refit_summary.csv`
- `ml_paper/tables/exp70_matrix_subset_refit_summary.md`
- `ml_paper/figures/exp70_matrix_subset_refits.png`
- `ml_paper/results/mlp09_summary.json`

Status:

- Completed as `ml_paper/experiments/mlp09_exp70_matrix_subset_refits.py`.
- Exp70 now archives `matched_matrices.npz` in new tagged runs.
- On EW/KPZ only, feature baselines reach KMeans ARI `1.0`, so binary
  continuum separation is easy for these features.
- On EW/KPZ/BD/Eden, the best feature KMeans ARI is `0.186` under equal
  sampling and `0.169` under exp69 sampling; matched effective exponents improve
  this to `0.503` and `0.438`, respectively, but remain unstable.
- All-six matched exponent geometry still ties raw multi-L features on average:
  delta `-0.0005` under equal sampling and `-0.0046` under exp69 sampling.

## Recommended Execution Order

1. MLP-01 evidence consolidation.
2. MLP-03 exponent-cloud instability figure.
3. MLP-02 hard-subset audit.
4. MLP-05 local-vs-global geometry.
5. MLP-04 false-positive case study.
6. MLP-06 positive controls.
7. MLP-08 clusterer/hierarchy controls.
8. MLP-09 true exp70 matrix subset refits.
9. Decide whether MLP-07 is necessary after seeing the first draft.

## Publication Criterion

The paper is ready when the following sentence is visibly supported by figures
and tables:

> In this benchmark, finite-size representations can be locally predictive and
> can even produce high-ARI single-protocol false positives, but they do not
> robustly approximate the RG universality quotient under unsupervised clustering.
