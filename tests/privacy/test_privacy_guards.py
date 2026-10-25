from __future__ import annotations

import json

import pytest

from anpr_syntheval import SynthEvaluator
from anpr_syntheval.validation import DataValidationError, validate_output_path


@pytest.mark.parametrize("column", ["sequence_key", "vehicle_token", "registration_number", "origin_timestamp"])
def test_prohibited_identifier_columns_fail_before_evaluation(
    column, roles, schema_path, config_mapping
) -> None:
    train, holdout, synthetic = roles
    synthetic = synthetic.copy()
    synthetic[column] = [f"sensitive-{index}" for index in range(len(synthetic))]
    with pytest.raises(DataValidationError):
        SynthEvaluator(train, holdout, synthetic, schema_path, config_mapping).evaluate()


def test_same_training_and_holdout_object_is_rejected(roles, schema_path, config_mapping) -> None:
    train, _, synthetic = roles
    with pytest.raises(DataValidationError):
        SynthEvaluator(train, train, synthetic, schema_path, config_mapping).validate()


def test_strict_mode_rejects_impossible_values(roles, schema_path, config_mapping) -> None:
    train, holdout, synthetic = roles
    synthetic = synthetic.copy()
    synthetic.loc[0, "duration_seconds"] = 0
    with pytest.raises(DataValidationError):
        SynthEvaluator(train, holdout, synthetic, schema_path, config_mapping).evaluate()


def test_restricted_output_path_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        validate_output_path(tmp_path / "restricted" / "report.json", ["restricted"])


def test_report_contains_only_aggregate_metric_rows(tmp_path, roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    path = report.to_json(tmp_path / "result.json")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "rows" not in payload
    assert "records" not in payload
    assert "sensitive-" not in text.lower()
    for role in payload["metadata"]["datasets"].values():
        assert not ({"registration_number", "vehicle_token", "journey_token", "sequence_key"}
                    & set(role["projected_columns"]))


def test_empirical_privacy_notes_disclaim_formal_guarantee(roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    boundaries = " ".join(report.metadata["interpretation_boundaries"]).lower()
    assert "do not prove anonymity" in boundaries
