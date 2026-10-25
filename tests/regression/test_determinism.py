from __future__ import annotations

from anpr_syntheval import SynthEvaluator


REQUIRED_METRICS = {
    "continuous.ks_statistic",
    "continuous.normalised_wasserstein",
    "categorical.total_variation",
    "dependency.pearson_correlation_matrix_mae",
    "dependency.propensity_classifier_auc",
    "utility.macro_f1",
    "privacy.exact_training_row_match_rate",
    "privacy.train_holdout_distance_ratio.median",
    "transport.od_pair_distribution_jsd",
    "transport.distance_duration_speed_consistency_rate",
}


def test_same_inputs_and_seed_have_identical_scientific_digest(roles, schema_path, config_mapping) -> None:
    first = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    second = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    assert first.canonical_digest() == second.canonical_digest()


def test_metric_registry_regression_surface(roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    identifiers = {item.metric_id for item in report.metrics}
    assert REQUIRED_METRICS <= identifiers
    assert set(report.metadata["metric_registry"]) == {
        "continuous_fidelity", "categorical_fidelity", "dependency_fidelity",
        "utility", "empirical_privacy_risk", "transport_alignment"
    }


def test_no_unqualified_overall_score_is_emitted(roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    assert not any(item.metric_id in {"overall_score", "quality_score"} for item in report.metrics)
