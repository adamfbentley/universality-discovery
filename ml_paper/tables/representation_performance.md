| representation | protocol | hdbscan_core_ari_mean | hdbscan_core_ari_std | kmeans_ari_mean | kmeans_ari_std | knn3_accuracy_mean | knn3_accuracy_std | local_global_gap | quotient_compatibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6D spatial morphology | L=256, T=2000, N=80 | 0.494763 | 0 | 0.184919 | 0 | 0.820833 | 0 | 0.32607 | fails |
| 10d_spatial_temporal | 5 seed blocks, L=128, T=500 | 0.497444 | 0.00570178 | 0.49364 | 0.00103044 | 0.973333 | 0.0106863 | 0.475889 | fails |
| 6d_spatial | 5 seed blocks, L=128, T=500 | 0.493882 | 0.00359407 | 0.185552 | 0.00491623 | 0.834444 | 0.0163865 | 0.340562 | fails |
| 10d_spatial_temporal | finite-size sweep L=64..256, 3 seeds/size | 0.507678 | 0.0387443 | 0.491283 | 0.00107157 | 0.98 | 0.0109834 | 0.472322 | fails |
| 6d_spatial | finite-size sweep L=64..256, 3 seeds/size | 0.47695 | 0.0199426 | 0.182939 | 0.00392483 | 0.832 | 0.0292987 | 0.35505 | fails |
| 10d_spatial_temporal | height/TW feature audit, 5 seed blocks | 0.510115 | 0.0272095 | 0.49293 | 0.000929385 | 0.97 | 0.00496904 | 0.459885 | fails |
| 12d_all | height/TW feature audit, 5 seed blocks | 0.495075 | 0.0104567 | 0.494092 | 0.000577433 | 0.962222 | 0.0138332 | 0.467147 | fails |
| 2d_height_only | height/TW feature audit, 5 seed blocks | 0.302206 | 0.0397351 | 0.238224 | 0.0158945 | 0.616667 | 0.0180021 | 0.314461 | fails |
| 6d_spatial | height/TW feature audit, 5 seed blocks | 0.483845 | 0.0160963 | 0.181965 | 0.00154159 | 0.818889 | 0.0268167 | 0.335044 | fails |
| 8d_spatial_height | height/TW feature audit, 5 seed blocks | 0.49525 | 0.00620421 | 0.49435 | 0 | 0.834444 | 0.0320108 | 0.339194 | fails |
| single-L Exp63 feature geometry | exp69 all-six 4-class head-to-head | 0.495201 | 0 | 0.489929 | 0 |  |  |  | fails |
| effective exponent geometry | exp69 all-six 4-class head-to-head | 0.959571 | 0 | 0.90171 | 0 |  |  |  | single_protocol_success |
| single-L features | exp71 five seed-start sweep, protocol=equal | 0.566243 | 0 | 0.532701 | 0.161573 |  |  |  | not_robust |
| raw multi-L features | exp71 five seed-start sweep, protocol=equal | 0.566243 | 0 | 0.604959 | 0 |  |  |  | not_robust |
| cross-L engineered features | exp71 five seed-start sweep, protocol=equal | 0.504742 | 0.0516089 | 0.495781 | 0 |  |  |  | not_robust |
| PCA-whitened multi-L features | exp71 five seed-start sweep, protocol=equal | 0.483013 | 0.0925284 | 0.394937 | 0.138086 |  |  |  | not_robust |
| matched effective exponent geometry | exp71 five seed-start sweep, protocol=equal | 0.732626 | 0.149243 | 0.60446 | 0.132405 |  |  |  | not_robust |
| exponent minus best feature | exp71 five seed-start sweep, protocol=equal |  |  | -0.000498981 | 0.132405 |  |  |  | no_advantage |
| single-L features | exp71 five seed-start sweep, protocol=exp69 | 0.558609 | 0.017071 | 0.532701 | 0.161573 |  |  |  | not_robust |
| raw multi-L features | exp71 five seed-start sweep, protocol=exp69 | 0.566243 | 0 | 0.604959 | 0 |  |  |  | not_robust |
| cross-L engineered features | exp71 five seed-start sweep, protocol=exp69 | 0.513733 | 0.0589733 | 0.495583 | 0.000441348 |  |  |  | not_robust |
| PCA-whitened multi-L features | exp71 five seed-start sweep, protocol=exp69 | 0.439861 | 0.0803748 | 0.294093 | 0.112747 |  |  |  | not_robust |
| matched effective exponent geometry | exp71 five seed-start sweep, protocol=exp69 | 0.674224 | 0.199287 | 0.600314 | 0.182461 |  |  |  | not_robust |
| exponent minus best feature | exp71 five seed-start sweep, protocol=exp69 |  |  | -0.00464479 | 0.182461 |  |  |  | no_advantage |
| Ising PCA/FSS positive control | L=32..96 finite-size scaling |  |  |  |  |  |  |  | works_on_clean_control |
| Potts Binder positive control | pilot Binder derivative |  |  |  |  |  |  |  | works_on_clean_control |
