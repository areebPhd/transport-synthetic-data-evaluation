"""ANPR-specific temporal, OD and physical-alignment diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..schema import SchemaConfig
from ..statistics import aligned_counts, js_divergence, wasserstein_distance
from ._helpers import not_applicable, result


def _combined_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].astype("string").fillna("<MISSING>")
    combined = values.iloc[:, 0]
    for column in values.columns[1:]:
        combined = combined.str.cat(values[column], sep="|")
    return combined.astype(object).rename(None)


def _profile_jsd(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: list[str],
) -> float:
    reference_series = reference[columns[0]] if len(columns) == 1 else _combined_series(reference, columns)
    candidate_series = candidate[columns[0]] if len(columns) == 1 else _combined_series(candidate, columns)
    left, right, _ = aligned_counts(reference_series, candidate_series)
    return js_divergence(left, right)


def _normalised_wasserstein(reference: pd.Series, candidate: pd.Series) -> float:
    left = pd.to_numeric(reference, errors="coerce").to_numpy(float)
    right = pd.to_numeric(candidate, errors="coerce").to_numpy(float)
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    distance = wasserstein_distance(left, right)
    q25, q75 = np.quantile(left, [0.25, 0.75])
    scale = q75 - q25
    if scale <= 0:
        scale = np.std(left, ddof=1)
    return float(distance / scale) if scale > 0 else float("nan")


def evaluate_transport(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    settings: dict[str, Any],
    schema: SchemaConfig,
    reference_audit: DatasetAudit,
    candidate_audit: DatasetAudit,
    comparison: str,
) -> list[MetricResult]:
    output: list[MetricResult] = []
    candidate_only_comparison = f"{comparison}_candidate_only"
    fields = schema.transport_fields
    temporal = schema.temporal_features

    profile_definitions = [
        ("transport.hourly_profile_jsd", [str(temporal.get("hour_of_day", "hour_of_day"))]),
        ("transport.day_of_week_profile_jsd", [str(temporal.get("day_of_week", "day_of_week"))]),
        ("transport.origin_distribution_jsd", [str(fields.get("origin_camera", "origin_camera"))]),
        ("transport.destination_distribution_jsd", [str(fields.get("destination_camera", "destination_camera"))]),
    ]
    for metric_id, columns in profile_definitions:
        if all(column in reference_frame and column in candidate_frame for column in columns):
            output.append(result(
                metric_id,
                "transport_alignment",
                comparison,
                "|".join(columns),
                _profile_jsd(reference_frame, candidate_frame, columns),
                "lower_is_better",
                0.0,
                reference_audit,
                candidate_audit,
                unit="bits",
            ))
        else:
            output.append(not_applicable(
                metric_id, "transport_alignment", comparison, "|".join(columns),
                "Required transport profile column is absent", reference_audit, candidate_audit
            ))

    working = fields.get("working_period", {})
    hour = str(working.get("hour_feature", temporal.get("hour_of_day", "hour_of_day")))
    if hour in reference_frame and hour in candidate_frame:
        start = int(working.get("start_inclusive", 7))
        end = int(working.get("end_exclusive", 19))
        reference_working = pd.to_numeric(reference_frame[hour], errors="coerce").between(
            start, end - 1
        ).mean()
        candidate_working = pd.to_numeric(candidate_frame[hour], errors="coerce").between(
            start, end - 1
        ).mean()
        output.append(result(
            "transport.working_period_rate_absolute_error",
            "transport_alignment",
            comparison,
            hour,
            abs(float(candidate_working - reference_working)),
            "lower_is_better",
            0.0,
            reference_audit,
            candidate_audit,
            unit="proportion_difference",
        ))
    else:
        output.append(not_applicable(
            "transport.working_period_rate_absolute_error", "transport_alignment", comparison,
            hour, "Hour feature is absent", reference_audit, candidate_audit
        ))

    origin = str(fields.get("origin_camera", "origin_camera"))
    destination = str(fields.get("destination_camera", "destination_camera"))
    od_available = all(
        column in reference_frame and column in candidate_frame for column in (origin, destination)
    )
    if od_available:
        reference_od = _combined_series(reference_frame, [origin, destination])
        candidate_od = _combined_series(candidate_frame, [origin, destination])
        ref_counts, cand_counts, labels = aligned_counts(reference_od, candidate_od)
        ref_probability = ref_counts / ref_counts.sum()
        cand_probability = cand_counts / cand_counts.sum()
        output.append(result(
            "transport.od_pair_distribution_jsd", "transport_alignment", comparison, "OD_PAIR",
            js_divergence(ref_counts, cand_counts), "lower_is_better", 0.0,
            reference_audit, candidate_audit, unit="bits"
        ))
        top_k = int(settings.get("top_k_od", 20))
        ref_positive = pd.Series(ref_counts, index=labels)
        cand_positive = pd.Series(cand_counts, index=labels)
        ref_top = set(ref_positive.loc[ref_positive > 0].nlargest(top_k).index)
        cand_top = set(cand_positive.loc[cand_positive > 0].nlargest(top_k).index)
        overlap = len(ref_top & cand_top) / len(ref_top | cand_top) if ref_top | cand_top else float("nan")
        output.append(result(
            "transport.top_k_od_overlap", "transport_alignment", comparison, "OD_PAIR", overlap,
            "higher_is_better", 1.0, reference_audit, candidate_audit, unit="jaccard"
        ))
        output.append(result(
            "transport.weighted_od_flow_error", "transport_alignment", comparison, "OD_PAIR",
            float(np.abs(ref_probability - cand_probability).sum()), "lower_is_better", 0.0,
            reference_audit, candidate_audit, unit="normalised_l1"
        ))
        reference_present = {labels[index] for index, count in enumerate(ref_counts) if count > 0}
        candidate_present = {labels[index] for index, count in enumerate(cand_counts) if count > 0}
        output.append(result(
            "transport.camera_pair_coverage", "transport_alignment", comparison, "OD_PAIR",
            len(reference_present & candidate_present) / len(reference_present), "higher_is_better", 1.0,
            reference_audit, candidate_audit, unit="proportion"
        ))
        rare_maximum = int(settings.get("minimum_released_od_count", 10)) - 1
        rare = {labels[index] for index, count in enumerate(ref_counts) if 0 < count <= rare_maximum}
        rare_coverage = len(rare & candidate_present) / len(rare) if rare else float("nan")
        output.append(result(
            "transport.rare_od_coverage", "transport_alignment", comparison, "OD_PAIR",
            rare_coverage if np.isfinite(rare_coverage) else None, "higher_is_better", 1.0,
            reference_audit, candidate_audit, unit="proportion", applicable=bool(np.isfinite(rare_coverage)),
            reason=None if np.isfinite(rare_coverage) else "No held-out OD pair met the rare-count definition",
            notes="Only a scalar coverage rate is released; low-count OD labels are never emitted"
        ))
        if hour in reference_frame and hour in candidate_frame:
            output.append(result(
                "transport.temporal_od_interaction_jsd", "transport_alignment", comparison,
                "HOUR|OD_PAIR", _profile_jsd(reference_frame, candidate_frame, [hour, origin, destination]),
                "lower_is_better", 0.0, reference_audit, candidate_audit, unit="bits",
                notes="Computed internally over aggregate cells; no low-count cell labels are released"
            ))
        else:
            output.append(not_applicable(
                "transport.temporal_od_interaction_jsd", "transport_alignment", comparison,
                "HOUR|OD_PAIR", "Hour feature is absent", reference_audit, candidate_audit
            ))
    else:
        for metric_id in (
            "transport.od_pair_distribution_jsd", "transport.top_k_od_overlap",
            "transport.weighted_od_flow_error", "transport.camera_pair_coverage",
            "transport.rare_od_coverage", "transport.temporal_od_interaction_jsd",
        ):
            output.append(not_applicable(
                metric_id, "transport_alignment", comparison, "OD_PAIR",
                "Origin or destination camera field is absent", reference_audit, candidate_audit
            ))

    distance = str(fields.get("distance", "route_distance_m"))
    duration = str(fields.get("duration", "duration_seconds"))
    speed = str(fields.get("derived_speed", "derived_average_speed_kph"))
    for metric_id, feature in (
        ("transport.distance_distribution_normalised_wasserstein", distance),
        ("transport.duration_distribution_normalised_wasserstein", duration),
        ("transport.speed_distribution_normalised_wasserstein", speed),
    ):
        if feature in reference_frame and feature in candidate_frame:
            output.append(result(
                metric_id, "transport_alignment", comparison, feature,
                _normalised_wasserstein(reference_frame[feature], candidate_frame[feature]),
                "lower_is_better", 0.0, reference_audit, candidate_audit, unit="holdout_iqr"
            ))
        else:
            output.append(not_applicable(
                metric_id, "transport_alignment", comparison, feature,
                "Required physical feature is absent", reference_audit, candidate_audit
            ))

    if duration in candidate_frame:
        candidate_duration = pd.to_numeric(candidate_frame[duration], errors="coerce").to_numpy(float)
        nonpositive = float(np.mean(~np.isfinite(candidate_duration) | (candidate_duration <= 0)))
        output.append(result(
            "transport.non_positive_duration_rate", "transport_alignment", candidate_only_comparison, duration,
            nonpositive, "lower_is_better", 0.0, reference_audit, candidate_audit, unit="proportion"
        ))
    else:
        output.append(not_applicable(
            "transport.non_positive_duration_rate", "transport_alignment", candidate_only_comparison, duration,
            "Duration feature is absent", reference_audit, candidate_audit
        ))
    if speed in candidate_frame:
        candidate_speed = pd.to_numeric(candidate_frame[speed], errors="coerce").to_numpy(float)
        threshold = float(settings.get(
            "impossible_speed_threshold_kph",
            schema.physical_consistency.get("impossible_speed_threshold_kph", 130.0),
        ))
        impossible = float(np.mean(~np.isfinite(candidate_speed) | (candidate_speed > threshold) | (candidate_speed < 0)))
        output.append(result(
            "transport.impossible_speed_rate", "transport_alignment", candidate_only_comparison, speed,
            impossible, "lower_is_better", 0.0, reference_audit, candidate_audit, unit="proportion"
        ))
    else:
        output.append(not_applicable(
            "transport.impossible_speed_rate", "transport_alignment", candidate_only_comparison, speed,
            "Derived-speed feature is absent", reference_audit, candidate_audit
        ))

    if all(feature in candidate_frame for feature in (distance, duration, speed)):
        candidate_distance = pd.to_numeric(candidate_frame[distance], errors="coerce").to_numpy(float)
        candidate_duration = pd.to_numeric(candidate_frame[duration], errors="coerce").to_numpy(float)
        candidate_speed = pd.to_numeric(candidate_frame[speed], errors="coerce").to_numpy(float)
        valid = (
            np.isfinite(candidate_distance) & np.isfinite(candidate_duration) & np.isfinite(candidate_speed)
            & (candidate_duration > 0) & (candidate_distance >= 0)
        )
        expected = np.full(len(candidate_frame), np.nan)
        expected[valid] = 3.6 * candidate_distance[valid] / candidate_duration[valid]
        relative_error = np.full(len(candidate_frame), np.inf)
        relative_error[valid] = np.abs(candidate_speed[valid] - expected[valid]) / np.maximum(
            np.abs(expected[valid]), 1e-9
        )
        tolerance = float(settings.get(
            "speed_consistency_relative_tolerance",
            schema.physical_consistency.get("relative_tolerance", 0.02),
        ))
        consistency = float(np.mean(relative_error <= tolerance))
        finite_error = relative_error[np.isfinite(relative_error)]
        output.append(result(
            "transport.distance_duration_speed_consistency_rate", "transport_alignment",
            candidate_only_comparison, "DISTANCE|DURATION|SPEED", consistency, "higher_is_better", 1.0,
            reference_audit, candidate_audit, unit="proportion"
        ))
        output.append(result(
            "transport.distance_duration_speed_relative_error.mean", "transport_alignment",
            candidate_only_comparison, "DISTANCE|DURATION|SPEED",
            float(finite_error.mean()) if len(finite_error) else None, "lower_is_better", 0.0,
            reference_audit, candidate_audit, unit="relative_error",
            applicable=bool(len(finite_error)), reason=None if len(finite_error) else "No valid physical triples"
        ))
    else:
        output.append(not_applicable(
            "transport.distance_duration_speed_consistency_rate", "transport_alignment",
            candidate_only_comparison, "DISTANCE|DURATION|SPEED", "A physical triple feature is absent",
            reference_audit, candidate_audit
        ))
    return output
