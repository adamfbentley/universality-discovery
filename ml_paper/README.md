# ML-Focused Paper Track

This folder is the clean workspace for the machine-learning paper that grows out
of the surface-growth universality experiments. It deliberately separates the
publishable ML story from the older exploratory experiment log.

## Working Thesis

The deeper ML question is:

> When does a learned representation factor through the physical quotient?

Unsupervised clustering discovers geometry induced by a representation, metric,
sampling protocol, and finite-size regime. It does not automatically discover a
physical equivalence relation. Universality classes are quotient labels under RG
flow; most clustering algorithms assume cluster labels in an observed metric
space. For 1+1D growth models, finite-size morphology, height-distribution
features, and effective exponent summaries are informative but do not robustly
approximate the RG universality quotient.

The paper should be framed as a cautionary benchmark for ML-for-physics:

> Finite-size feature geometry can be locally class-informative while globally
> incompatible with universality-class clustering. A high-ARI exponent-geometry
> result can be reproduced under one protocol, then disappear under matched
> seed/protocol repetition.

Candidate title language:

- finite-size universality mirages
- local separability vs global physical discovery
- clustering failure under quotient labels
- protocol-dependent false positives in unsupervised scientific ML

## Folder Map

- `EXPERIMENT_PLAN.md` - ordered experiment plan for the ML paper.
- `CLAIMS_REGISTER.md` - allowed claims, forbidden claims, and required evidence.
- `MANUSCRIPT_OUTLINE.md` - target paper structure.
- `experiments/` - ML-paper-specific scripts or wrappers.
- `results/` - derived summaries for the ML paper only.
- `figures/` - generated ML-paper figures.
- `tables/` - generated ML-paper tables.

## Current Evidence Backbone

Existing artifacts to reuse:

- `results_exp62/results.json` - baseline feature-space clustering.
- `results_exp65_robustness/summary.json` - seed and HDBSCAN sensitivity.
- `results_exp66_finite_size/summary.json`, `sweep_rows.csv` - finite-size sweep.
- `results_exp67_height_dist/summary.json`, `representation_rows.csv` - height/TW features.
- `results_exp68_collapse/summary.json` - continuum effective-exponent success case.
- `results_exp69_collapse_full/summary.json` - high-ARI exponent false-positive candidate.
- `results_exp70_matched*` - matched feature/exponent baselines.
- `results_exp71_protocol_sweep/summary.json`, `runs.csv` - repeated protocol sweep.
- `results_exp52d_full/results.json` and `results_exp57c_pilot/summary.json` - positive controls.

ML-paper derived artifacts now include:

- `tables/representation_performance.md` - consolidated local/global evidence.
- `tables/subset_ari_summary.md` - hard-subset audit removing RD and/or KS.
- `tables/local_global_gap.csv` - local probe versus global clustering gaps.
- `tables/exp62_neighborhood_purity.csv` - nearest-neighbor purity curves.
- `tables/exp62_feature_quotient_geometry.csv` - feature-space quotient-distance diagnostic.
- `tables/exponent_quotient_metrics.csv` - finite exponent-cloud quotient diagnostics.
- `tables/clusterer_control_summary.csv` - standard clusterer stress test.
- `tables/hierarchical_merge_audit.csv` - centroid hierarchy and KPZ merge audit.
- `tables/exp70_matrix_subset_refit_summary.md` - true subset refits from archived exp70 feature matrices.
- `figures/subset_ari_audit.png` - subset KMeans ARI comparison.
- `figures/local_vs_global.png` - local separability versus global quotient recovery.
- `figures/exponent_ari_by_seed.png` - protocol sensitivity of exponent geometry.
- `figures/clusterer_control_best_ari.png` - best fixed-k clusterer by subset.
- `figures/hierarchical_kpz_merge_audit.png` - quotient-distance hierarchy control.
- `figures/exp70_matrix_subset_refits.png` - true feature/exponent subset refits from matrix archives.

## Non-Negotiable Framing

This is not a claim that universality cannot be learned. It is a claim that the
tested finite-size representations do not robustly recover the intended RG
equivalence relation by ordinary clustering.

Avoid saying "collapse metric" unless a true master-curve residual is computed.
Use "effective exponent geometry" for the current Path 1 results.
