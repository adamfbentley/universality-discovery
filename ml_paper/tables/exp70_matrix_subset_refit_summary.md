# Exp70 Matrix Subset Refit Summary

These rows use archived feature matrices and refit KMeans after selecting each subset.

| protocol | subset | best_feature_representation | best_feature_kmeans_ari_mean | matched_exponent_kmeans_ari_mean | exponent_minus_best_feature |
| --- | --- | --- | --- | --- | --- |
| equal | all_six | raw multi-L features | 0.604959 | 0.604459 | -0.000500000000000056 |
| equal | no_ks | raw multi-L features | 0.499581 | 0.574735 | 0.075154 |
| equal | no_rd | cross-L engineered features | 0.34647 | 0.648263 | 0.30179300000000003 |
| equal | no_rd_no_ks | PCA-whitened multi-L features | 0.185552 | 0.503341 | 0.31778900000000004 |
| equal | ew_kpz_only | PCA-whitened multi-L features | 1.0 | 0.661983 | -0.338017 |
| exp69 | all_six | raw multi-L features | 0.604959 | 0.600314 | -0.00464500000000001 |
| exp69 | no_ks | raw multi-L features | 0.499581 | 0.564451 | 0.06487000000000004 |
| exp69 | no_rd | cross-L engineered features | 0.34647 | 0.623164 | 0.27669400000000005 |
| exp69 | no_rd_no_ks | PCA-whitened multi-L features | 0.169329 | 0.437755 | 0.268426 |
| exp69 | ew_kpz_only | PCA-whitened multi-L features | 1.0 | 0.517293 | -0.482707 |
