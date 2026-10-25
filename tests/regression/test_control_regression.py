from __future__ import annotations

import numpy as np

from anpr_syntheval import SynthEvaluator


def get(report, metric_id, comparison=None):
    return next(
        item.value for item in report.metrics
        if item.metric_id == metric_id and (comparison is None or item.comparison == comparison)
    )


def test_copied_rows_show_apparent_fidelity_and_privacy_warning(roles, schema_path, config_mapping) -> None:
    train, holdout, _ = roles
    copied = train.copy()
    report = SynthEvaluator(train, holdout, copied, schema_path, config_mapping).evaluate()
    assert get(report, "privacy.exact_training_row_match_rate") == 1.0
    assert get(report, "privacy.closest_training_distance.minimum") == 0.0
    assert get(report, "continuous.ks_statistic.mean", "synthetic_vs_real_holdout") < 0.2


def test_invalid_control_is_flagged_in_audit_mode(roles, schema_path, config_mapping) -> None:
    train, holdout, synthetic = roles
    invalid = synthetic.copy()
    invalid.loc[:49, "duration_seconds"] = -5
    invalid.loc[:49, "derived_average_speed_kph"] = 300
    config = dict(config_mapping)
    config["validation_mode"] = "audit"
    report = SynthEvaluator(train, holdout, invalid, schema_path, config).evaluate()
    assert any(item.code == "physical_range_violation" for item in report.validation)
    assert get(report, "transport.non_positive_duration_rate", "synthetic_vs_real_holdout_candidate_only") > 0.1
    assert get(report, "transport.impossible_speed_rate", "synthetic_vs_real_holdout_candidate_only") > 0.1
