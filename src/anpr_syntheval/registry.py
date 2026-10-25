"""Metric result model and registry metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class MetricResult:
    """One auditable metric observation."""

    metric_id: str
    family: str
    comparison: str
    feature: str | None
    value: float | None
    direction: str
    ideal_value: float | str | None
    unit: str | None = None
    applicable: bool = True
    reason: str | None = None
    reference_size_full: int | None = None
    candidate_size_full: int | None = None
    reference_size_evaluated: int | None = None
    candidate_size_evaluated: int | None = None
    sampling_method: str = "full_data"
    sampling_seed: int | None = None
    sampling_reason: str | None = None
    uncertainty_type: str = "none_single_evaluation"
    ci_low: float | None = None
    ci_high: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.value is not None:
            result["value"] = float(self.value)
        return result


class MetricRegistry:
    """Small explicit registry used for documentation and execution audits."""

    def __init__(self) -> None:
        self._metrics: dict[str, Callable[..., list[MetricResult]]] = {}
        self._descriptions: dict[str, str] = {}

    def register(
        self, name: str, function: Callable[..., list[MetricResult]], description: str
    ) -> None:
        if name in self._metrics:
            raise ValueError(f"Metric family already registered: {name}")
        self._metrics[name] = function
        self._descriptions[name] = description

    def get(self, name: str) -> Callable[..., list[MetricResult]]:
        return self._metrics[name]

    def describe(self) -> dict[str, str]:
        return dict(self._descriptions)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._metrics)
