from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anpr_syntheval import SynthEvaluator
from anpr_syntheval.datasets import DatasetAudit
from anpr_syntheval.metrics import dependency as dependency_module
from anpr_syntheval.metrics.categorical import evaluate_categorical
from anpr_syntheval.metrics.dependency import evaluate_dependency
from anpr_syntheval.metrics.privacy import evaluate_privacy
from anpr_syntheval.metrics.transport import evaluate_transport
from anpr_syntheval.schema import SchemaConfig


def audit(role: str, size: int) -> DatasetAudit:
    return DatasetAudit(
        role=role,
        source_type="dataframe",
        path=None,
        format="in_memory",
        checksum_sha256=None,
        full_size=size,
        evaluated_size=size,
        available_columns=[],
        projected_columns=[],
        sampling_method="full_data",
        sampling_seed=None,
        sample_index_sha256=None,
        sampling_reason=None,
    )


def minimal_schema() -> SchemaConfig:
    return SchemaConfig.from_mapping({
        "schema_version": "test",
        "name": "code_review_fixture",
        "continuous_features": ["x"],
        "count_features": [],
        "categorical_features": ["origin", "destination", "category"],
        "ordinal_features": {"hour": {"ordered_values": [0, 1]}},
        "binary_features": {},
        "temporal_features": {"hour_of_day": "hour"},
        "cyclical_temporal_features": {},
        "transport_fields": {
            "origin_camera": "origin",
            "destination_camera": "destination",
        },
        "target_variables": [],
        "required_columns": ["x", "origin", "destination", "category", "hour"],
        "optional_columns": [],
        "prohibited_identifiers": {},
        "grouping_variables": [],
        "overlap_audit_columns": [],
        "rare_categories": {},
        "valid_physical_ranges": {},
        "physical_consistency": {},
        "leakage_rules": {},
        "privacy_distance_features": {
            "continuous": ["x"],
            "categorical": ["category"],
        },
        "rare_combination_features": [],
    })


def test_duplicate_train_holdout_paths_rejected_with_dataframe_synthetic(
    tmp_path: Path, schema_path: Path, config_mapping: dict, roles: tuple[pd.DataFrame, ...]
) -> None:
    shared = tmp_path / "shared.csv"
    roles[0].to_csv(shared, index=False)
    evaluator = SynthEvaluator(shared, shared, roles[2], schema_path, config_mapping)
    codes = {issue.code for issue in evaluator._preflight_issues()}
    assert "same_input_path" in codes


def test_disjoint_categorical_support_has_zero_top_k_overlap() -> None:
    reference = pd.DataFrame({"category": ["A", "A"]})
    candidate = pd.DataFrame({"category": ["B", "B"]})
    metrics = evaluate_categorical(
        reference,
        candidate,
        ["category"],
        {"top_k": 10},
        {},
        audit("reference", 2),
        audit("candidate", 2),
        "test",
    )
    selected = next(item for item in metrics if item.metric_id == "categorical.top_k_overlap")
    assert selected.value == 0.0


def test_disjoint_od_support_has_zero_top_k_overlap() -> None:
    reference = pd.DataFrame({
        "origin": ["A", "A"], "destination": ["B", "B"], "hour": [0, 1],
    })
    candidate = pd.DataFrame({
        "origin": ["C", "C"], "destination": ["D", "D"], "hour": [0, 1],
    })
    metrics = evaluate_transport(
        reference,
        candidate,
        {"top_k_od": 20},
        minimal_schema(),
        audit("reference", 2),
        audit("candidate", 2),
        "test",
    )
    selected = next(item for item in metrics if item.metric_id == "transport.top_k_od_overlap")
    assert selected.value == 0.0


def test_one_row_privacy_second_neighbour_is_unavailable() -> None:
    frame = pd.DataFrame({
        "x": [1.0],
        "origin": ["A"],
        "destination": ["B"],
        "category": ["C"],
        "hour": [0],
    })
    metrics = evaluate_privacy(
        frame,
        frame.copy(),
        frame.copy(),
        {},
        minimal_schema(),
        audit("train", 1),
        audit("holdout", 1),
        audit("synthetic", 1),
        seed=11,
        maximum_rows=1,
    )
    ratio = next(
        item for item in metrics
        if item.metric_id == "privacy.nearest_neighbour_distance_ratio.median"
    )
    assert ratio.applicable is False
    assert ratio.value is None


def test_ordinal_propensity_predictor_is_encoded_once(monkeypatch) -> None:
    size = 40
    reference = pd.DataFrame({
        "x": np.linspace(0, 1, size),
        "hour": np.arange(size) % 2,
        "category": np.where(np.arange(size) % 3, "A", "B"),
    })
    candidate = reference.sample(frac=1, random_state=4).reset_index(drop=True)
    observed: list[list[str]] = []
    original = dependency_module.DesignEncoder.fit.__func__

    def recording_fit(cls, frame, features, maximum_levels=100):
        observed.append(list(features))
        return original(cls, frame, features, maximum_levels)

    monkeypatch.setattr(
        dependency_module.DesignEncoder,
        "fit",
        classmethod(recording_fit),
    )
    evaluate_dependency(
        reference,
        candidate,
        numeric_features=["x", "hour"],
        categorical_features=["hour", "category"],
        settings={
            "propensity_iterations": 1,
            "propensity_learning_rate": 0.1,
            "propensity_maximum_rows_per_class": 30,
        },
        reference_audit=audit("reference", size),
        candidate_audit=audit("candidate", size),
        comparison="test",
        seed=9,
        maximum_rows=size,
    )
    assert observed
    assert observed[-1].count("hour") == 1
