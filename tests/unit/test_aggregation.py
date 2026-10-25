from __future__ import annotations

import pytest

from anpr_syntheval import (
    EvaluationReport,
    RunObservation,
    aggregate_run_observations,
)
from anpr_syntheval.registry import MetricResult


def report(value: float | None, *, applicable: bool = True, direction: str = "lower_is_better"):
    return EvaluationReport(
        metadata={},
        validation=[],
        metrics=[
            MetricResult(
                metric_id="fidelity.distance",
                family="fidelity",
                comparison="candidate_vs_holdout",
                feature="x",
                value=value,
                direction=direction,
                ideal_value=0.0,
                applicable=applicable,
                reason=None if applicable else "declared test condition",
            )
        ],
    )


def test_two_level_aggregation_uses_independent_unit_means() -> None:
    observations = [
        RunObservation(report(1.0), "seed_11", 0),
        RunObservation(report(3.0), "seed_11", 1),
        RunObservation(report(5.0), "seed_22", 0),
        RunObservation(report(7.0), "seed_22", 1),
        RunObservation(report(None, applicable=False), "seed_22", 2),
    ]
    result = aggregate_run_observations(
        observations, bootstrap_repetitions=500, bootstrap_seed=91
    )[0]
    assert result.mean == 4.0
    assert result.independent_units_applicable == 2
    assert result.observations_expected == 5
    assert result.observations_attempted == 5
    assert result.observations_applicable == 4
    assert result.ci_low is not None
    assert result.ci_high is not None


def test_aggregation_is_deterministic() -> None:
    observations = [
        RunObservation(report(1.0), "seed_11"),
        RunObservation(report(2.0), "seed_22"),
        RunObservation(report(4.0), "seed_33"),
    ]
    first = aggregate_run_observations(
        observations, bootstrap_repetitions=250, bootstrap_seed=45
    )
    second = aggregate_run_observations(
        observations, bootstrap_repetitions=250, bootstrap_seed=45
    )
    assert first == second


def test_single_independent_unit_does_not_manufacture_interval() -> None:
    observations = [
        RunObservation(report(1.0), "seed_11", 0),
        RunObservation(report(2.0), "seed_11", 1),
    ]
    result = aggregate_run_observations(observations)[0]
    assert result.independent_units_applicable == 1
    assert result.ci_low is None
    assert result.ci_high is None
    assert "two applicable independent units" in str(result.reason)


def test_duplicate_observation_identity_and_metadata_mismatch_fail() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        aggregate_run_observations(
            [
                RunObservation(report(1.0), "seed_11", 0),
                RunObservation(report(2.0), "seed_11", 0),
            ]
        )
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_run_observations(
            [
                RunObservation(report(1.0), "seed_11"),
                RunObservation(report(2.0, direction="higher_is_better"), "seed_22"),
            ]
        )
