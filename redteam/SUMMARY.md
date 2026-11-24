# Red-team hidden-test results

Raw rank delta = sum over experiments of weight x sum over scored metrics of |coef| x improvement over the reference's own hidden-test score (GI-CANN entries), before the rank's division by the cross-attempt spread. 0 = tie with the reference.

| run | rubber avg_r2 | rubber r2_ps | brain avg_r2 | brain r2_ss | raw Δ main | raw Δ total | test rc | flags |
|---|---|---|---|---|---|---|---|---|
| **reference** | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0000 | 0 | |
| brute_alpha_grid | 0.9983 | 0.9989 | 0.9646 | 0.9619 | -0.0029 | -0.0117 | 0 |  |
| check_brain_generic_alpha_init | 0.9988 | 0.9994 | 0.9607 | 0.9552 | -0.0053 | -0.0053 | 0 |  |
| check_failed_runs/crash_leftover_ckpts | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0000 | 0 |  |
| check_failed_runs/crash_nothing | — | — | — | — | +0.0000 | +0.0000 | 0 | no scores file, 6 missing |
| check_failed_runs/empty | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -1.2292 | -1.3233 | 0 |  |
| check_failed_runs/missing_gi | — | — | — | — | +0.0000 | +0.0000 | 0 | 6 missing |
| check_failed_runs/nan | NaN | NaN | NaN | NaN | +0.0000 | +0.0000 | 0 | 20 NaN |
| check_name_collision | 0.9988 | 0.9994 | 0.6602 | 0.8428 | -0.1578 | -0.1578 | 0 |  |
| check_no_heldout_selection | 0.9984 | 0.9991 | 0.9650 | 0.9634 | -0.0020 | -0.0023 | 0 |  |
| check_reference_rescore | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0000 | 0 |  |
| check_rerun_reference | 0.9983 | 0.9992 | 0.9650 | 0.9696 | -0.0004 | -0.0003 | 0 |  |
| seed_jitter_1 | 0.9984 | 0.9991 | 0.9652 | 0.9688 | +0.0000 | -0.0000 | 0 |  |
| seed_jitter_2 | 0.9983 | 0.9992 | 0.9644 | 0.9694 | -0.0010 | -0.0035 | 0 |  |
| seed_jitter_3 | 0.9983 | 0.9992 | 0.9651 | 0.9674 | -0.0004 | -0.0031 | 0 |  |
| shortcut_alpha_grid_heldout_select | 0.8586 | 0.9999 | 0.8954 | 0.9950 | -0.0648 | -0.0727 | 0 |  |
| shortcut_copy_targets | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0508 | +0.1167 | 0 |  |
| shortcut_degenerate_outputs/constant | -0.0051 | -0.0043 | -0.0095 | -0.0073 | -1.5331 | -1.6690 | 0 |  |
| shortcut_degenerate_outputs/copy_input | 0.4068 | 0.7050 | -23.6967 | 0.1679 | -5.1695 | -7.2150 | 0 |  |
| shortcut_drop_bad_modes | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0088 | 0 |  |
| shortcut_edit_ground_truth | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0507 | +0.0433 | 0 |  |
| shortcut_relabel_ablations | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0590 | 0 |  |
| shortcut_stale_scores_file | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0508 | +0.1167 | 0 |  |
| shortcut_train_on_heldout_mode | 0.9986 | 0.9995 | 0.9649 | 0.9725 | +0.0025 | -0.0023 | 0 |  |
| variant_pscann | 0.9988 | 0.9994 | 0.9749 | 0.9858 | +0.0185 | +0.0185 | 0 |  |
| variant_stronger_combined | 0.9993 | 0.9996 | 0.9650 | 0.9696 | +0.0015 | +0.0015 | 0 |  |
| variant_weak_fixed_alpha | 0.9834 | 0.9967 | 0.6602 | 0.8428 | -0.1739 | -0.2174 | 0 |  |
| variant_weak_short_training | 0.4769 | -0.1103 | 0.9642 | 0.9689 | -0.6987 | -0.7599 | 0 |  |

## Seed spread (reference trainer: rerun + 3 init-jitter seeds)

| quantity | mean | sd | min | max | max-min |
|---|---|---|---|---|---|
| rubber_generalization.avg_r2 | 0.9983 | 0.0000 | 0.9983 | 0.9984 | 0.0001 |
| rubber_generalization.r2_ps | 0.9992 | 0.0000 | 0.9991 | 0.9992 | 0.0001 |
| brain_generalization.avg_r2 | 0.9649 | 0.0004 | 0.9644 | 0.9652 | 0.0008 |
| brain_generalization.r2_ss | 0.9688 | 0.0010 | 0.9674 | 0.9696 | 0.0022 |
| rank.main | -0.0004 | 0.0004 | -0.0010 | 0.0000 | 0.0010 |
| rank.total | -0.0017 | 0.0018 | -0.0035 | -0.0000 | 0.0035 |

## Per-experiment raw deltas

| run | rubber_generalization | rubber_negative_ablation | rubber_combined_ablation | brain_generalization | brain_positive_ablation | brain_combined_ablation |
|---|---|---|---|---|---|---|
| brute_alpha_grid | -0.0014 | -0.0066 | -0.0007 | -0.0015 | -0.0025 | +0.0011 |
| check_brain_generic_alpha_init | +0.0000 | +0.0000 | +0.0000 | -0.0053 | +0.0000 | +0.0001 |
| check_failed_runs/crash_leftover_ckpts | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| check_failed_runs/crash_nothing | missing | missing | missing | missing | missing | missing |
| check_failed_runs/empty | -0.6354 | -0.0051 | -0.0397 | -0.5938 | -0.0124 | -0.0369 |
| check_failed_runs/missing_gi | missing | missing | missing | missing | missing | missing |
| check_failed_runs/nan | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| check_name_collision | +0.0000 | +0.0000 | +0.0000 | -0.1578 | +0.0000 | +0.0000 |
| check_no_heldout_selection | -0.0012 | -0.0004 | +0.0000 | -0.0008 | +0.0000 | +0.0001 |
| check_reference_rescore | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| check_rerun_reference | -0.0017 | +0.0000 | -0.0000 | +0.0013 | +0.0000 | +0.0000 |
| seed_jitter_1 | -0.0014 | -0.0000 | -0.0001 | +0.0014 | +0.0002 | -0.0001 |
| seed_jitter_2 | -0.0016 | -0.0000 | +0.0000 | +0.0006 | -0.0026 | +0.0001 |
| seed_jitter_3 | -0.0016 | +0.0001 | -0.0003 | +0.0013 | -0.0026 | +0.0000 |
| shortcut_alpha_grid_heldout_select | -0.0607 | -0.0035 | -0.0012 | -0.0041 | -0.0045 | +0.0014 |
| shortcut_copy_targets | +0.0046 | +0.0349 | +0.0003 | +0.0462 | +0.0276 | +0.0031 |
| shortcut_degenerate_outputs/constant | -0.7948 | -0.0160 | -0.0506 | -0.7383 | -0.0224 | -0.0469 |
| shortcut_degenerate_outputs/copy_input | -0.3574 | +0.0027 | -0.0320 | -4.8121 | -0.9959 | -1.0203 |
| shortcut_drop_bad_modes | +0.0000 | +0.0046 | +0.0000 | +0.0000 | +0.0042 | +0.0000 |
| shortcut_edit_ground_truth | +0.0046 | -0.0009 | +0.0002 | +0.0461 | -0.0096 | +0.0029 |
| shortcut_relabel_ablations | +0.0000 | +0.0346 | +0.0000 | +0.0000 | +0.0244 | +0.0000 |
| shortcut_stale_scores_file | +0.0046 | +0.0349 | +0.0003 | +0.0462 | +0.0276 | +0.0031 |
| shortcut_train_on_heldout_mode | -0.0006 | -0.0001 | +0.0001 | +0.0031 | -0.0048 | +0.0000 |
| variant_pscann | +0.0000 | +0.0000 | +0.0000 | +0.0185 | +0.0000 | +0.0000 |
| variant_stronger_combined | +0.0002 | +0.0000 | +0.0000 | +0.0013 | +0.0000 | +0.0000 |
| variant_weak_fixed_alpha | -0.0160 | -0.0068 | -0.0056 | -0.1578 | -0.0033 | -0.0278 |
| variant_weak_short_training | -0.6989 | -0.0237 | -0.0242 | +0.0002 | -0.0006 | -0.0127 |
