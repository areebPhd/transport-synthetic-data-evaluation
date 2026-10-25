"""Continuous-feature held-out fidelity metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..statistics import (
    freedman_diaconis_edges,
    js_divergence,
    ks_statistic,
    numeric_values,
    wasserstein_distance,
)
from ._helpers import not_applicable, result


def evaluate_continuous(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    features: Sequence[str],
    settings: dict[str, Any],
    reference_audit: DatasetAudit,
    candidate_audit: DatasetAudit,
    comparison: str,
) -> list[MetricResult]:
    """Evaluate marginal continuous fidelity against held-out real data."""

    output: list[MetricResult] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    quantiles = [float(item) for item in settings.get(
        "quantiles", [0.05, 0.25, 0.5, 0.75, 0.95]
    )]
    maximum_bins = int(settings.get("maximum_histogram_bins", 80))
    for feature in features:
        if feature not in reference_frame or feature not in candidate_frame:
            output.append(not_applicable(
                "continuous.feature_missing",
                "continuous_fidelity",
                comparison,
                feature,
                "Feature is absent from one evaluated role",
                reference_audit,
                candidate_audit,
            ))
            continue
        reference = numeric_values(reference_frame[feature])
        candidate = numeric_values(candidate_frame[feature])
        if len(reference) < 2 or len(candidate) < 2:
            output.append(not_applicable(
                "continuous.insufficient_values",
                "continuous_fidelity",
                comparison,
                feature,
                "At least two finite values are required in each role",
                reference_audit,
                candidate_audit,
            ))
            continue
        mean_error = abs(float(candidate.mean() - reference.mean()))
        median_error = abs(float(np.median(candidate) - np.median(reference)))
        std_error = abs(float(np.std(candidate, ddof=1) - np.std(reference, ddof=1)))
        quantile_error = float(np.mean(np.abs(
            np.quantile(candidate, quantiles) - np.quantile(reference, quantiles)
        )))
        lower, upper = float(reference.min()), float(reference.max())
        boundary_coverage = float(np.mean((candidate >= lower) & (candidate <= upper)))
        reference_range = upper - lower
        candidate_range = float(candidate.max() - candidate.min())
        range_ratio = (
            min(reference_range, candidate_range) / max(reference_range, candidate_range)
            if max(reference_range, candidate_range) > 0 else 1.0
        )
        ks = ks_statistic(reference, candidate)
        wasserstein = wasserstein_distance(reference, candidate)
        q25, q75 = np.quantile(reference, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale <= 0:
            scale = float(np.std(reference, ddof=1))
        normalised = wasserstein / scale if scale > 0 else float("nan")
        edges = freedman_diaconis_edges(np.concatenate([reference, candidate]), maximum_bins)
        ref_hist = np.histogram(reference, bins=edges)[0]
        cand_hist = np.histogram(candidate, bins=edges)[0]
        histogram_jsd = js_divergence(ref_hist, cand_hist)
        out_of_range = 1.0 - boundary_coverage
        definitions = [
            ("continuous.mean_absolute_error", mean_error, "lower_is_better", 0.0, "feature_unit"),
            ("continuous.median_absolute_error", median_error, "lower_is_better", 0.0, "feature_unit"),
            ("continuous.standard_deviation_error", std_error, "lower_is_better", 0.0, "feature_unit"),
            ("continuous.quantile_mae", quantile_error, "lower_is_better", 0.0, "feature_unit"),
            ("continuous.boundary_coverage", boundary_coverage, "higher_is_better", 1.0, "proportion"),
            ("continuous.range_coverage_ratio", range_ratio, "higher_is_better", 1.0, "ratio"),
            ("continuous.ks_statistic", ks, "lower_is_better", 0.0, "statistic"),
            ("continuous.wasserstein_distance", wasserstein, "lower_is_better", 0.0, "feature_unit"),
            ("continuous.normalised_wasserstein", normalised, "lower_is_better", 0.0, "holdout_iqr"),
            ("continuous.histogram_jsd", histogram_jsd, "lower_is_better", 0.0, "bits"),
            ("continuous.out_of_range_rate", out_of_range, "lower_is_better", 0.0, "proportion"),
        ]
        for metric_id, value, direction, ideal, unit in definitions:
            output.append(result(
                metric_id,
                "continuous_fidelity",
                comparison,
                feature,
                value,
                direction,
                ideal,
                reference_audit,
                candidate_audit,
                unit=unit,
                notes=(
                    "Normalised by the held-out interquartile range; falls back to held-out standard deviation"
                    if metric_id == "continuous.normalised_wasserstein" else None
                ),
            ))
            aggregates[metric_id].append(value)
    for metric_id in (
        "continuous.ks_statistic",
        "continuous.normalised_wasserstein",
        "continuous.histogram_jsd",
        "continuous.out_of_range_rate",
    ):
        values = [item for item in aggregates.get(metric_id, []) if np.isfinite(item)]
        output.append(result(
            f"{metric_id}.mean",
            "continuous_fidelity",
            comparison,
            "ALL_CONTINUOUS_FEATURES",
            float(np.mean(values)) if values else None,
            "lower_is_better",
            0.0,
            reference_audit,
            candidate_audit,
            unit="mean_across_features",
            applicable=bool(values),
            reason=None if values else "No applicable continuous features",
        ))
    return output
