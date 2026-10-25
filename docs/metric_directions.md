# Metric-direction contract

Every metric result carries a `direction`, `target`, `unit`, `comparison`,
`feature`, applicability flag and data-role audit. Interpretation is
metric-specific.

| Direction value | Interpretation | Typical examples |
|---|---|---|
| `lower_is_better` | Smaller divergence, error or observed replication is preferred, subject to the use case | Wasserstein distance, total variation, Jensen–Shannon divergence, exact-match rate |
| `higher_is_better` | Larger coverage or predictive performance is preferred | macro-F1, balanced accuracy, category coverage |
| `closest_to_0.5_is_lower_observed_attack_signal` | Values nearer 0.5 show less signal for the declared binary attack | distance-based membership AUC |
| `higher_is_generally_lower_isolation_risk` | Larger values are heuristically preferable | nearest-neighbour distance ratio |
| `lower_is_generally_lower_replication_risk` | Smaller values mean fewer declared rare combinations were reproduced | rare-combination replication recall |
| `closest_to_zero` | Smaller absolute difference is preferred | difference in utility score |

The phrases “generally” and “observed attack signal” are intentional. Privacy
diagnostics depend on attacker knowledge, distance representation and sampling;
they are not anonymity guarantees. Likewise, low rare-combination replication
can be poor fidelity rather than good privacy.

No metric is transformed into a universal quality score. A release decision
must state the intended use, acceptable directions and thresholds before
examining candidate outcomes.
