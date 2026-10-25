from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from anpr_syntheval import SchemaConfig
from anpr_syntheval.datasets import DatasetAudit
from anpr_syntheval.metrics.categorical import evaluate_categorical
from anpr_syntheval.metrics.continuous import evaluate_continuous
from anpr_syntheval.metrics.transport import _combined_series, evaluate_transport


def audit(role: str, size: int) -> DatasetAudit:
    return DatasetAudit(role, "dataframe", None, "in_memory", None, size, size, [], [], "full_data", None, None, None)


def value(results, metric_id: str, feature: str):
    return next(item.value for item in results if item.metric_id == metric_id and item.feature == feature)


def test_vectorised_transport_combination_matches_row_join() -> None:
    frame = pd.DataFrame(
        {
            "hour": pd.Series([0, 17, None, 23], dtype="Int64"),
            "origin": ["A", "B", None, "D"],
            "destination": ["Q", None, "S", "T"],
        }
    )
    columns = ["hour", "origin", "destination"]
    legacy = (
        frame[columns]
        .astype("string")
        .fillna("<MISSING>")
        .agg("|".join, axis=1)
    )
    pd.testing.assert_series_equal(_combined_series(frame, columns), legacy)


def test_continuous_identity_is_zero(roles) -> None:
    reference = roles[0]
    results = evaluate_continuous(
        reference, reference.copy(), ["duration_seconds"], {}, audit("ref", len(reference)),
        audit("cand", len(reference)), "identity"
    )
    assert value(results, "continuous.ks_statistic", "duration_seconds") == 0
    assert value(results, "continuous.normalised_wasserstein", "duration_seconds") == 0
    assert value(results, "continuous.out_of_range_rate", "duration_seconds") == 0


def test_categorical_missing_support_is_detected(roles) -> None:
    reference = roles[0]
    candidate = roles[1].copy()
    candidate["time_of_day"] = "night"
    results = evaluate_categorical(
        reference, candidate, ["time_of_day"], {"top_k": 3},
        {"maximum_reference_frequency": 0.2, "minimum_reference_count": 2},
        audit("ref", len(reference)), audit("cand", len(candidate)), "test"
    )
    assert value(results, "categorical.category_coverage", "time_of_day") < 1
    assert value(results, "categorical.total_variation", "time_of_day") > 0


def test_transport_invalid_records_are_detected(schema_path, roles) -> None:
    schema = SchemaConfig.from_yaml(schema_path)
    reference = roles[0]
    candidate = roles[1].copy()
    candidate.loc[:19, "duration_seconds"] = 0
    candidate.loc[:19, "derived_average_speed_kph"] = 250
    results = evaluate_transport(
        reference, candidate, {"impossible_speed_threshold_kph": 130}, schema,
        audit("ref", len(reference)), audit("cand", len(candidate)), "invalid"
    )
    assert value(results, "transport.non_positive_duration_rate", "duration_seconds") > 0
    assert value(results, "transport.impossible_speed_rate", "derived_average_speed_kph") > 0
    assert value(results, "transport.distance_duration_speed_consistency_rate", "DISTANCE|DURATION|SPEED") < 1
