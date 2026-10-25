from __future__ import annotations

import numpy as np

from anpr_syntheval import SynthEvaluator


def metric(report, metric_id: str, comparison: str = "synthetic_vs_real_holdout", feature=None):
    matches = [
        item for item in report.metrics
        if item.metric_id == metric_id and item.comparison == comparison
        and (feature is None or item.feature == feature)
    ]
    return matches[0].value


def test_shuffling_damages_dependency_more_than_marginals(roles, schema_path, config_mapping) -> None:
    train, holdout, _ = roles
    generator = np.random.default_rng(111)
    shuffled = train.copy()
    for column in shuffled.columns:
        shuffled[column] = generator.permutation(shuffled[column].to_numpy())
    audit_config = dict(config_mapping)
    audit_config["validation_mode"] = "audit"
    report = SynthEvaluator(train, holdout, shuffled, schema_path, audit_config).evaluate()
    assert metric(report, "continuous.ks_statistic", feature="duration_seconds") < 0.15
    assert metric(report, "dependency.pearson_correlation_matrix_mae") > 0.05


def test_severe_corruption_is_worse_than_minor_shift(roles, schema_path, config_mapping) -> None:
    train, holdout, synthetic = roles
    minor = synthetic.copy()
    minor["route_distance_m"] *= 1.03
    minor["duration_seconds"] *= 1.03
    severe = synthetic.copy()
    severe["route_distance_m"] *= 3
    severe["duration_seconds"] *= 0.35
    severe["derived_average_speed_kph"] *= 2.5
    audit_config = dict(config_mapping)
    audit_config["validation_mode"] = "audit"
    minor_report = SynthEvaluator(train, holdout, minor, schema_path, audit_config).evaluate()
    severe_report = SynthEvaluator(train, holdout, severe, schema_path, audit_config).evaluate()
    assert metric(severe_report, "continuous.normalised_wasserstein", feature="route_distance_m") > metric(
        minor_report, "continuous.normalised_wasserstein", feature="route_distance_m"
    )
    assert metric(severe_report, "transport.distance_duration_speed_consistency_rate", comparison="synthetic_vs_real_holdout_candidate_only") < metric(
        minor_report, "transport.distance_duration_speed_consistency_rate", comparison="synthetic_vs_real_holdout_candidate_only"
    )
