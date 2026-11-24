# Red-team hidden-test results

Raw rank delta = sum over experiments of weight x sum over scored metrics of |coef| x improvement over the reference's own hidden-test score (GI-CANN entries), before the rank's division by the cross-attempt spread. 0 = tie with the reference.

| run | rubber avg_r2 | rubber r2_ps | brain avg_r2 | brain r2_ss | raw Δ main | raw Δ total | test rc | flags |
|---|---|---|---|---|---|---|---|---|
| **reference** | 0.9988 | 0.9994 | 0.9643 | 0.9683 | +0.0000 | +0.0000 | 0 | |

## Per-experiment raw deltas

| run | rubber_generalization | rubber_negative_ablation | rubber_combined_ablation | brain_generalization | brain_positive_ablation | brain_combined_ablation |
|---|---|---|---|---|---|---|
