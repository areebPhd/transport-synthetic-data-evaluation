from __future__ import annotations

import pandas as pd
import pytest

from anpr_syntheval import EvaluationConfig, SchemaConfig
from anpr_syntheval.schema import SchemaError
from anpr_syntheval.validation import prohibited_columns, validate_frame


def test_schema_loads_and_roles_are_disjoint(schema_path) -> None:
    schema = SchemaConfig.from_yaml(schema_path)
    assert schema.transport_fields["origin_camera"] == "origin_camera"
    assert len(schema.all_features) == len(set(schema.all_features))
    assert "sequence_key" in schema.prohibited_identifiers["exact_names"]


def test_overlapping_feature_types_are_rejected() -> None:
    with pytest.raises(SchemaError):
        SchemaConfig.from_mapping({
            "continuous_features": ["x"], "categorical_features": ["x"],
            "required_columns": ["x"]
        })


def test_configuration_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        EvaluationConfig.from_mapping({"validation_mode": "permissive", "sampling": {}})


def test_prohibited_identifier_detection_is_case_insensitive(schema_path) -> None:
    schema = SchemaConfig.from_yaml(schema_path)
    assert prohibited_columns(["Vehicle_Token", "duration_seconds"], schema) == ["Vehicle_Token"]


def test_audit_mode_records_physical_violation_as_warning(schema_path, roles) -> None:
    schema = SchemaConfig.from_yaml(schema_path)
    frame = roles[0].copy()
    frame.loc[0, "duration_seconds"] = -1
    issues = validate_frame(frame, "candidate", schema, "audit")
    assert any(item.code == "physical_range_violation" and item.severity == "warning" for item in issues)
