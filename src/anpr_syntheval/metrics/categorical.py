"""Categorical support and frequency fidelity metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..statistics import aligned_counts, js_divergence, spearman_correlation, total_variation
from ._helpers import not_applicable, result


def evaluate_categorical(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    features: Sequence[str],
    settings: dict[str, Any],
    rare_settings: dict[str, Any],
    reference_audit: DatasetAudit,
    candidate_audit: DatasetAudit,
    comparison: str,
) -> list[MetricResult]:
    output: list[MetricResult] = []
    aggregates: dict[str, list[float]] = defaultdict(list)
    top_k = int(settings.get("top_k", 10))
    rare_frequency = float(rare_settings.get("maximum_reference_frequency", 0.01))
    rare_minimum = int(rare_settings.get("minimum_reference_count", 2))
    for feature in features:
        if feature not in reference_frame or feature not in candidate_frame:
            output.append(not_applicable(
                "categorical.feature_missing",
                "categorical_fidelity",
                comparison,
                feature,
                "Feature is absent from one evaluated role",
                reference_audit,
                candidate_audit,
            ))
            continue
        reference = reference_frame[feature].astype("string").fillna("<MISSING>")
        candidate = candidate_frame[feature].astype("string").fillna("<MISSING>")
        if reference.nunique() < 1 or candidate.nunique() < 1:
            output.append(not_applicable(
                "categorical.insufficient_values",
                "categorical_fidelity",
                comparison,
                feature,
                "At least one category is required in each role",
                reference_audit,
                candidate_audit,
            ))
            continue
        ref_counts, cand_counts, labels = aligned_counts(reference, candidate)
        tvd = total_variation(ref_counts, cand_counts)
        divergence = js_divergence(ref_counts, cand_counts)
        reference_present = {labels[i] for i, value in enumerate(ref_counts) if value > 0}
        candidate_present = {labels[i] for i, value in enumerate(cand_counts) if value > 0}
        coverage = len(reference_present & candidate_present) / len(reference_present)
        unseen_rate = float(candidate.isin(candidate_present - reference_present).mean())
        frequencies = ref_counts / ref_counts.sum()
        rare = {
            labels[i]
            for i, count in enumerate(ref_counts)
            if count >= rare_minimum and frequencies[i] <= rare_frequency
        }
        rare_recall = len(rare & candidate_present) / len(rare) if rare else float("nan")
        common_positions = [i for i, label in enumerate(labels) if label in reference_present & candidate_present]
        rank_agreement = (
            spearman_correlation(ref_counts[common_positions], cand_counts[common_positions])
            if len(common_positions) >= 2 else float("nan")
        )
        ref_positive = pd.Series(ref_counts, index=labels)
        cand_positive = pd.Series(cand_counts, index=labels)
        ref_top = set(ref_positive.loc[ref_positive > 0].nlargest(top_k).index)
        cand_top = set(cand_positive.loc[cand_positive > 0].nlargest(top_k).index)
        top_overlap = len(ref_top & cand_top) / len(ref_top | cand_top) if ref_top | cand_top else float("nan")
        definitions = [
            ("categorical.total_variation", tvd, "lower_is_better", 0.0, "proportion"),
            ("categorical.jsd", divergence, "lower_is_better", 0.0, "bits"),
            ("categorical.category_coverage", coverage, "higher_is_better", 1.0, "proportion"),
            ("categorical.rare_category_recall", rare_recall, "higher_is_better", 1.0, "proportion"),
            ("categorical.unseen_category_rate", unseen_rate, "lower_is_better", 0.0, "proportion"),
            ("categorical.frequency_rank_spearman", rank_agreement, "higher_is_better", 1.0, "correlation"),
            ("categorical.top_k_overlap", top_overlap, "higher_is_better", 1.0, "jaccard"),
        ]
        for metric_id, value, direction, ideal, unit in definitions:
            applicable = bool(np.isfinite(value))
            output.append(result(
                metric_id,
                "categorical_fidelity",
                comparison,
                feature,
                value if applicable else None,
                direction,
                ideal,
                reference_audit,
                candidate_audit,
                unit=unit,
                applicable=applicable,
                reason=(
                    "No reference categories met the declared rare-category definition"
                    if metric_id == "categorical.rare_category_recall" and not applicable
                    else ("Fewer than two shared categories" if not applicable else None)
                ),
            ))
            if applicable:
                aggregates[metric_id].append(float(value))
    for metric_id in (
        "categorical.total_variation",
        "categorical.jsd",
        "categorical.category_coverage",
        "categorical.unseen_category_rate",
    ):
        values = aggregates.get(metric_id, [])
        higher = metric_id == "categorical.category_coverage"
        output.append(result(
            f"{metric_id}.mean",
            "categorical_fidelity",
            comparison,
            "ALL_CATEGORICAL_FEATURES",
            float(np.mean(values)) if values else None,
            "higher_is_better" if higher else "lower_is_better",
            1.0 if higher else 0.0,
            reference_audit,
            candidate_audit,
            unit="mean_across_features",
            applicable=bool(values),
            reason=None if values else "No applicable categorical features",
        ))
    return output
