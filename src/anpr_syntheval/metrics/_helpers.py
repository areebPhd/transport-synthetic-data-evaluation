"""Shared construction of sampling-aware metric results."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult


def result(
    metric_id: str,
    family: str,
    comparison: str,
    feature: str | None,
    value: float | None,
    direction: str,
    ideal_value: float | str | None,
    reference: DatasetAudit,
    candidate: DatasetAudit,
    unit: str | None = None,
    applicable: bool = True,
    reason: str | None = None,
    notes: str | None = None,
    **extra: Any,
) -> MetricResult:
    sampled = reference.sampling_method != "full_data" or candidate.sampling_method != "full_data"
    numeric = None
    if value is not None:
        numeric = float(value)
        if not np.isfinite(numeric):
            numeric = None
            applicable = False
            reason = reason or "Metric is undefined for the evaluated values"
    return MetricResult(
        metric_id=metric_id,
        family=family,
        comparison=comparison,
        feature=feature,
        value=numeric,
        direction=direction,
        ideal_value=ideal_value,
        unit=unit,
        applicable=applicable,
        reason=reason,
        reference_size_full=reference.full_size,
        candidate_size_full=candidate.full_size,
        reference_size_evaluated=reference.evaluated_size,
        candidate_size_evaluated=candidate.evaluated_size,
        sampling_method=(
            "deterministic_role_sampling" if sampled else "full_evaluated_inputs"
        ),
        sampling_seed=(candidate.sampling_seed or reference.sampling_seed),
        sampling_reason=(
            candidate.sampling_reason or reference.sampling_reason if sampled else None
        ),
        notes=notes,
        **extra,
    )


def not_applicable(
    metric_id: str,
    family: str,
    comparison: str,
    feature: str | None,
    reason: str,
    reference: DatasetAudit,
    candidate: DatasetAudit,
    direction: str = "not_applicable",
) -> MetricResult:
    return result(
        metric_id,
        family,
        comparison,
        feature,
        None,
        direction,
        None,
        reference,
        candidate,
        applicable=False,
        reason=reason,
    )


def cap_frame(
    frame: pd.DataFrame,
    audit: DatasetAudit,
    maximum_rows: int,
    seed: int,
    reason: str,
) -> tuple[pd.DataFrame, DatasetAudit]:
    """Apply and disclose a metric-specific deterministic cap."""

    if len(frame) <= maximum_rows:
        return frame, audit
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(frame), size=maximum_rows, replace=False))
    sampled = frame.iloc[indices].reset_index(drop=True)
    updated = replace(
        audit,
        evaluated_size=len(sampled),
        sampling_method="metric_specific_uniform_without_replacement",
        sampling_seed=seed,
        sampling_reason=reason,
    )
    return sampled, updated
