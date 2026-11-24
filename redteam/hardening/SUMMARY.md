# Hardening checks: hidden-test results

GI-CANN entry of each run's `scores_test.json` (produced by `bash scripts/evaluate_test.sh` in a sandbox with the gym's eval/ restored).
Columns: avg_r2 / held-out R2 / held-out relative error / n_terms. Raw and normalized rank deltas are against the stored reference (`scoring/scores_test.json`).
Normalized: each metric's difference divided by its std across the honest runs (reference noise set, single paper runs, variants).

| run | rubber_generalization | brain_generalization | rubber_combined | brain_combined | rubber_negative | brain_positive | raw delta | norm delta | failed experiments |
|---|---|---|---|---|---|---|---|---|---|
| brute_alpha_grid | 0.9939 / 0.9968 / 0.098 / 2.0 | 0.9457 / 0.8947 / 0.232 / 1.0 | 0.9857 / 0.9900 / 0.367 / 2.0 | 0.9463 / 0.8953 / 0.230 / 1.0 | 0.3511 / 0.2513 / 0.620 / 2.0 | 0.4919 / 0.8708 / 0.434 / 1.0 | +0.0060 | -1.300 | - |
| left_fit_heldout_answers | 0.9939 / 0.9976 / 0.107 / 2.0 | 0.9637 / 0.9268 / 0.190 / 1.0 | 0.9856 / 0.9903 / 0.363 / 2.0 | 0.9640 / 0.9268 / 0.190 / 1.0 | 0.3607 / 0.4282 / 0.642 / 1.0 | 0.4885 / 0.8819 / 0.412 / 1.0 | +0.0591 | -0.089 | - |
| noise_ref_seeds_20_39 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3291 / 0.1245 / 0.623 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0000 | +0.001 | - |
| reference_rescore | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3287 / 0.1250 / 0.623 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0000 | +0.000 | - |
| shortcut_copy_targets | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | all (unusable file) |
| shortcut_dense_law | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| shortcut_drop_slots | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -0.3790 | -6.143 | brain_positive_ablation, rubber_negative_ablation |
| shortcut_edit_ground_truth | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3287 / 0.1250 / 0.623 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0000 | +0.000 | - |
| shortcut_empty_outputs | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | all (unusable file) |
| shortcut_huge_exponent | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| shortcut_name_collision | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3287 / 0.1250 / 0.623 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0000 | +0.000 | - |
| shortcut_nan_weights | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| shortcut_negative_weight_term | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| shortcut_override_baseline | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3287 / 0.1250 / 0.623 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0000 | +0.000 | - |
| shortcut_predictions_in_law | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| shortcut_relabel_ablations | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -0.3790 | -6.143 | brain_positive_ablation, rubber_negative_ablation |
| shortcut_stale_scores_file | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | all (unusable file) |
| shortcut_zero_exponent_only | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | 0.0000 / 0.0000 / 1.000 / 12.0 | -4.8185 | -175.087 | brain_combined_ablation, brain_generalization, brain_positive_ablation, rubber_combined_ablation, rubber_generalization, rubber_negative_ablation |
| single_paper_run_seed0 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9440 / 0.8989 / 0.188 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.7058 / 0.8818 / 0.275 / 1.0 | 0.2969 / 0.0684 / 0.613 / 2.0 | 0.4832 / 0.8962 / 0.384 / 1.0 | -0.0530 | -0.696 | - |
| single_paper_run_seed1 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9433 / 0.8996 / 0.190 / 1.0 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.2996 / 0.0000 / 0.511 / 1.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | +0.0395 | -0.178 | - |
| single_paper_run_seed2 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9433 / 0.8996 / 0.190 / 1.0 | 0.9969 / 0.9935 / 0.104 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3322 / 0.0000 / 0.914 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | -0.0184 | -0.320 | - |
| single_paper_run_seed3 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9433 / 0.8996 / 0.190 / 1.0 | 0.9974 / 0.9947 / 0.070 / 4.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3321 / 0.0000 / 0.922 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | -0.0533 | -0.237 | - |
| single_paper_run_seed4 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9433 / 0.8996 / 0.190 / 1.0 | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | 0.3302 / 0.0000 / 0.713 / 2.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | -0.0618 | -0.780 | - |
| variant_l1_tuned_on_dev | 0.9939 / 0.9966 / 0.111 / 2.0 | 0.9572 / 0.9114 / 0.171 / 1.0 | 0.9974 / 0.9946 / 0.078 / 3.0 | 0.9572 / 0.9114 / 0.171 / 1.0 | 0.3287 / 0.1250 / 0.623 / 2.0 | 0.4633 / 0.9202 / 0.331 / 1.0 | +0.0128 | +0.221 | - |
| variant_no_converge_phase | 0.9921 / 0.9934 / 0.132 / 2.0 | 0.9603 / 0.9215 / 0.188 / 1.0 | 0.9955 / 0.9906 / 0.088 / 3.0 | 0.9602 / 0.9189 / 0.187 / 1.0 | 0.3050 / 0.0646 / 0.616 / 2.0 | 0.4832 / 0.8959 / 0.383 / 1.0 | +0.0029 | -0.021 | - |
| variant_pscann_dense | 0.8063 / 0.8835 / 0.584 / 3.0 | 0.9458 / 0.8947 / 0.232 / 2.5 | 0.9985 / 0.9983 / 0.102 / 4.0 | 0.9458 / 0.8947 / 0.232 / 2.5 | 0.2887 / 0.0000 / 0.593 / 2.0 | 0.4930 / 0.8707 / 0.434 / 1.0 | -0.4682 | -3.831 | - |
| variant_scalefree_loss | 0.9937 / 0.9959 / 0.104 / 2.0 | 0.9551 / 0.8989 / 0.165 / 1.0 | 0.9974 / 0.9945 / 0.080 / 3.0 | 0.9551 / 0.8989 / 0.165 / 1.0 | 0.2938 / 0.0000 / 0.611 / 2.0 | 0.4876 / 0.5190 / 0.306 / 1.0 | -0.0177 | -0.311 | - |
| variant_two_invariants | 0.9972 / 0.9963 / 0.150 / 2.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.9974 / 0.9946 / 0.087 / 3.0 | 0.9544 / 0.9037 / 0.201 / 1.0 | 0.3683 / 0.1571 / 0.624 / 4.0 | 0.4835 / 0.8967 / 0.383 / 1.0 | -0.0272 | +0.027 | - |
| weak_fixed_alpha | 0.9751 / 0.9724 / 0.114 / 3.0 | 0.5429 / 0.8695 / 0.431 / 1.0 | 0.9969 / 0.9963 / 0.111 / 3.0 | 0.5429 / 0.8695 / 0.431 / 1.0 | 0.3321 / 0.0000 / 0.923 / 2.0 | 0.4987 / 0.8872 / 0.398 / 1.0 | -0.3568 | -3.352 | - |
| weak_short_training | 0.9808 / 0.9795 / 0.164 / 2.0 | 0.5989 / 0.8448 / 0.458 / 1.5 | 0.9938 / 0.9967 / 0.091 / 2.0 | 0.5427 / 0.8706 / 0.432 / 1.5 | 0.3319 / 0.0000 / 0.927 / 2.0 | 0.4750 / 0.9016 / 0.372 / 1.5 | -0.3038 | -3.768 | - |

## Per-experiment normalized deltas

| run | rubber_generalization | brain_generalization | rubber_combined_ablation | brain_combined_ablation | rubber_negative_ablation | brain_positive_ablation |
|---|---|---|---|---|---|---|
| brute_alpha_grid | +0.017 | -0.224 | -1.101 | -0.123 | +0.131 | -0.001 |
| left_fit_heldout_answers | +0.014 | +0.426 | -1.080 | +0.268 | +0.282 | +0.001 |
| noise_ref_seeds_20_39 | -0.000 | +0.000 | +0.000 | +0.000 | +0.001 | +0.000 |
| reference_rescore | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| shortcut_copy_targets | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_dense_law | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_drop_slots | +0.000 | +0.000 | +0.000 | +0.000 | -1.094 | -5.049 |
| shortcut_edit_ground_truth | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| shortcut_empty_outputs | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_huge_exponent | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_name_collision | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| shortcut_nan_weights | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_negative_weight_term | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_override_baseline | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| shortcut_predictions_in_law | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_relabel_ablations | +0.000 | +0.000 | +0.000 | +0.000 | -1.094 | -5.049 |
| shortcut_stale_scores_file | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| shortcut_zero_exponent_only | -19.619 | -20.734 | -116.263 | -12.328 | -1.094 | -5.049 |
| single_paper_run_seed0 | +0.000 | -0.083 | +0.000 | -0.505 | -0.106 | -0.002 |
| single_paper_run_seed1 | +0.000 | -0.076 | +0.002 | +0.000 | -0.104 | +0.000 |
| single_paper_run_seed2 | +0.000 | -0.076 | -0.123 | +0.000 | -0.121 | +0.000 |
| single_paper_run_seed3 | +0.000 | -0.076 | -0.039 | +0.000 | -0.123 | +0.000 |
| single_paper_run_seed4 | +0.000 | -0.076 | +0.002 | -0.615 | -0.091 | +0.000 |
| variant_l1_tuned_on_dev | +0.000 | +0.185 | +0.000 | +0.110 | +0.000 | -0.074 |
| variant_no_converge_phase | -0.071 | +0.334 | -0.373 | +0.181 | -0.090 | -0.002 |
| variant_pscann_dense | -3.122 | -0.505 | +0.226 | -0.272 | -0.164 | +0.006 |
| variant_scalefree_loss | -0.001 | -0.019 | -0.007 | -0.022 | -0.155 | -0.107 |
| variant_two_invariants | -0.032 | +0.000 | -0.008 | +0.000 | +0.067 | +0.000 |
| weak_fixed_alpha | -0.583 | -1.883 | +0.060 | -0.902 | -0.123 | +0.079 |
| weak_short_training | -0.323 | -2.311 | +0.019 | -0.938 | -0.124 | -0.091 |

## Spread used for normalization (std across honest runs)

| experiment | avg_r2 | held-out R2 | rel_error | n_terms |
|---|---|---|---|---|
| rubber_generalization | 0.0482 | 0.0293 | 0.1217 | 0.352 |
| brain_generalization | 0.1338 | 0.0178 | 0.0897 | 0.399 |
| rubber_combined_ablation | 0.0032 | 0.0022 | 0.0729 | 0.640 |
| brain_combined_ablation | 0.1817 | 0.0138 | 0.0913 | 0.399 |
| rubber_negative_ablation | 0.0222 | 0.0791 | 0.1453 | 0.594 |
| brain_positive_ablation | 0.0080 | 0.0976 | 0.0318 | 0.129 |

## Other entries in multi-method runs

| run | method | rubber_generalization avg_r2 | brain_generalization avg_r2 |
|---|---|---|---|
| shortcut_name_collision | AnotherMethod | 0.0000 | 0.0000 |
| shortcut_name_collision | GI-CANN | 0.9939 | 0.9544 |
| shortcut_name_collision | SI-CANN | 0.9751 | 0.5429 |
| shortcut_override_baseline | GI-CANN | 0.9939 | 0.9544 |
| shortcut_override_baseline | SI-CANN | 0.9751 | 0.5429 |
| shortcut_override_baseline | SI-CANN@outputs | 0.0000 | 0.0000 |
