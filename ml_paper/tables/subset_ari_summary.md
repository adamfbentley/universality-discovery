# MLP-02 Hard-Subset ARI Summary

Rows marked `restrict_full_partition` are post-hoc restrictions of stored all-system cluster labels, not subset refits.

| source_family | protocol | representation | evaluation_mode | subset | kmeans_ari_mean | kmeans_ari_min | kmeans_ari_max | hdbscan_core_ari_mean | knn3_accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp62 | stored feature matrix, L=128, T=1000, N=80/system | 6D spatial morphology | refit_subset | all_six | 0.184919 | 0.184919 | 0.184919 | 0.494763 | 0.8375 |
| exp62 | stored feature matrix, L=128, T=1000, N=80/system | 6D spatial morphology | refit_subset | ew_kpz_only | -0.00191673 | -0.00191673 | -0.00191673 | -0.00584873 | 0.725 |
| exp62 | stored feature matrix, L=128, T=1000, N=80/system | 6D spatial morphology | refit_subset | no_ks | 0.348571 | 0.348571 | 0.348571 | 0.345689 | 0.8125 |
| exp62 | stored feature matrix, L=128, T=1000, N=80/system | 6D spatial morphology | refit_subset | no_rd | -0.116418 | -0.116418 | -0.116418 | 0.348571 | 0.81 |
| exp62 | stored feature matrix, L=128, T=1000, N=80/system | 6D spatial morphology | refit_subset | no_rd_no_ks | -0.0686767 | -0.0686767 | -0.0686767 | -0.0686767 | 0.775 |
| exp70_sweep | equal | matched effective exponent geometry | refit_subset | all_six | 0.60446 | 0.469378 | 0.81984 | 0.732626 | 0.956667 |
| exp70_sweep | equal | matched effective exponent geometry | refit_subset | ew_kpz_only | 0.661983 | 0.29358 | 1 | 0.846284 | 0.98 |
| exp70_sweep | equal | matched effective exponent geometry | refit_subset | no_ks | 0.576177 | 0.434711 | 0.784017 | 0.645712 | 0.954 |
| exp70_sweep | equal | matched effective exponent geometry | refit_subset | no_rd | 0.648263 | 0.404031 | 0.983747 | 0.630626 | 0.958 |
| exp70_sweep | equal | matched effective exponent geometry | refit_subset | no_rd_no_ks | 0.503614 | 0.19872 | 0.973486 | 0.425398 | 0.95 |
| exp70_sweep | equal | raw multi-L features | restrict_full_partition | all_six | 0.604959 | 0.604959 | 0.604959 | 0.566243 |  |
| exp70_sweep | equal | raw multi-L features | restrict_full_partition | ew_kpz_only | 1 | 1 | 1 | 1 |  |
| exp70_sweep | equal | raw multi-L features | restrict_full_partition | no_ks | 0.499581 | 0.499581 | 0.499581 | 0.47769 |  |
| exp70_sweep | equal | raw multi-L features | restrict_full_partition | no_rd | 0.499581 | 0.499581 | 0.499581 | 0.47769 |  |
| exp70_sweep | equal | raw multi-L features | restrict_full_partition | no_rd_no_ks | 0.246445 | 0.246445 | 0.246445 | 0.329114 |  |
| exp70_sweep | exp69 | matched effective exponent geometry | refit_subset | all_six | 0.600314 | 0.438731 | 0.90171 | 0.674224 | 0.946667 |
| exp70_sweep | exp69 | matched effective exponent geometry | refit_subset | ew_kpz_only | 0.517293 | -0.0102561 | 1 | 0.70124 | 0.9675 |
| exp70_sweep | exp69 | matched effective exponent geometry | refit_subset | no_ks | 0.564451 | 0.374539 | 0.843234 | 0.596175 | 0.95 |
| exp70_sweep | exp69 | matched effective exponent geometry | refit_subset | no_rd | 0.62221 | 0.381674 | 0.936031 | 0.526587 | 0.946 |
| exp70_sweep | exp69 | matched effective exponent geometry | refit_subset | no_rd_no_ks | 0.437755 | 0.0900332 | 0.921943 | 0.333333 | 0.9425 |
| exp70_sweep | exp69 | raw multi-L features | restrict_full_partition | all_six | 0.604959 | 0.604959 | 0.604959 | 0.566243 |  |
| exp70_sweep | exp69 | raw multi-L features | restrict_full_partition | ew_kpz_only | 1 | 1 | 1 | 1 |  |
| exp70_sweep | exp69 | raw multi-L features | restrict_full_partition | no_ks | 0.499581 | 0.499581 | 0.499581 | 0.47769 |  |
| exp70_sweep | exp69 | raw multi-L features | restrict_full_partition | no_rd | 0.499581 | 0.499581 | 0.499581 | 0.47769 |  |
| exp70_sweep | exp69 | raw multi-L features | restrict_full_partition | no_rd_no_ks | 0.246445 | 0.246445 | 0.246445 | 0.329114 |  |
