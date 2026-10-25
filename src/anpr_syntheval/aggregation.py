"""Two-level uncertainty aggregation for independently generated evaluations.

Evaluation repetitions within one trained model are averaged before inference
across independent model runs.  This prevents repeated evaluation of one model
from being misrepresented as additional model-training evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .api import EvaluationReport


@dataclass(frozen=True)
class RunObservation:
    """One evaluation report with its genuine independent-unit identity."""

    report: EvaluationReport
    independent_unit: str
    evaluation_repetition: str | int = 0


@dataclass(frozen=True)
class AggregateMetricResult:
    """One metric aggregated over within-unit means."""

    metric_id: str
    family: str
    comparison: str
    feature: str | None
    direction: str
    ideal_value: float | str | None
    unit: str | None
    observations_expected: int
    observations_attempted: int
    observations_applicable: int
    independent_units_attempted: int
    independent_units_applicable: int
    mean: float | None
    standard_deviation: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    uncertainty_unit: str = "independent_unit_mean"
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric_key(item: Any) -> tuple[str, str, str, str | None]:
    return item.metric_id, item.family, item.comparison, item.feature


def _key_seed(seed: int, key: tuple[str, str, str, str | None]) -> int:
    payload = "\x1f".join("" if item is None else str(item) for item in key).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return int((int(seed) + offset) % (2**63 - 1))


def aggregate_run_observations(
    observations: Iterable[RunObservation],
    *,
    confidence_level: float = 0.95,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260729,
) -> list[AggregateMetricResult]:
    """Aggregate metric results without inflating the independent denominator.

    Values from evaluation repetitions are first averaged within each declared
    ``independent_unit``. Percentile bootstrap intervals are then calculated
    over those independent-unit means. Non-applicable results remain in the
    attempted denominator.
    """

    items = list(observations)
    if not items:
        raise ValueError("At least one RunObservation is required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    identities = [
        (str(item.independent_unit), str(item.evaluation_repetition)) for item in items
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("independent_unit and evaluation_repetition pairs must be unique")

    declarations: dict[
        tuple[str, str, str, str | None], tuple[str, float | str | None, str | None]
    ] = {}
    values: dict[tuple[str, str, str, str | None], dict[str, list[float]]] = {}
    attempts: dict[tuple[str, str, str, str | None], int] = {}
    applicable: dict[tuple[str, str, str, str | None], int] = {}
    attempted_units: dict[tuple[str, str, str, str | None], set[str]] = {}

    for observation in items:
        seen: set[tuple[str, str, str, str | None]] = set()
        unit_name = str(observation.independent_unit)
        for metric in observation.report.metrics:
            key = _metric_key(metric)
            if key in seen:
                raise ValueError(f"Report contains a duplicate metric key: {key}")
            seen.add(key)
            declaration = (metric.direction, metric.ideal_value, metric.unit)
            if key in declarations and declarations[key] != declaration:
                raise ValueError(f"Metric metadata are inconsistent across reports: {key}")
            declarations[key] = declaration
            attempts[key] = attempts.get(key, 0) + 1
            attempted_units.setdefault(key, set()).add(unit_name)
            if metric.applicable and metric.value is not None and np.isfinite(metric.value):
                applicable[key] = applicable.get(key, 0) + 1
                values.setdefault(key, {}).setdefault(unit_name, []).append(float(metric.value))

    output: list[AggregateMetricResult] = []
    for key in sorted(declarations, key=lambda item: tuple("" if x is None else str(x) for x in item)):
        direction, ideal_value, unit = declarations[key]
        unit_means = np.asarray(
            [np.mean(unit_values) for unit_values in values.get(key, {}).values()],
            dtype=float,
        )
        mean = float(unit_means.mean()) if len(unit_means) else None
        standard_deviation = (
            float(unit_means.std(ddof=1)) if len(unit_means) >= 2 else None
        )
        if len(unit_means) >= 2:
            generator = np.random.default_rng(_key_seed(bootstrap_seed, key))
            means = np.empty(bootstrap_repetitions, dtype=float)
            for index in range(bootstrap_repetitions):
                means[index] = generator.choice(
                    unit_means, size=len(unit_means), replace=True
                ).mean()
            tail = (1.0 - confidence_level) / 2.0
            ci_low, ci_high = (
                float(value)
                for value in np.quantile(means, [tail, 1.0 - tail])
            )
            reason = None
        else:
            ci_low = None
            ci_high = None
            reason = (
                "No applicable independent unit"
                if not len(unit_means)
                else "At least two applicable independent units are required for an interval"
            )
        output.append(
            AggregateMetricResult(
                metric_id=key[0],
                family=key[1],
                comparison=key[2],
                feature=key[3],
                direction=direction,
                ideal_value=ideal_value,
                unit=unit,
                observations_expected=len(items),
                observations_attempted=attempts[key],
                observations_applicable=applicable.get(key, 0),
                independent_units_attempted=len(attempted_units[key]),
                independent_units_applicable=len(unit_means),
                mean=mean,
                standard_deviation=standard_deviation,
                ci_low=ci_low,
                ci_high=ci_high,
                confidence_level=confidence_level,
                bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_seed=bootstrap_seed,
                reason=reason,
            )
        )
    return output
