"""Dependency, mixed-association, mutual-information and propensity diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..statistics import (
    DesignEncoder,
    association_matrix,
    correlation_matrix,
    fit_binary_logistic,
    mutual_information,
    predict_binary_logistic,
    roc_auc_score,
    upper_triangle_mae,
)
from ._helpers import cap_frame, not_applicable, result


def evaluate_dependency(
    reference_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    settings: dict[str, Any],
    reference_audit: DatasetAudit,
    candidate_audit: DatasetAudit,
    comparison: str,
    seed: int,
    maximum_rows: int,
) -> list[MetricResult]:
    output: list[MetricResult] = []
    reference_frame, reference_audit = cap_frame(
        reference_frame,
        reference_audit,
        maximum_rows,
        seed + 101,
        "bounded dependency and association computation",
    )
    candidate_frame, candidate_audit = cap_frame(
        candidate_frame,
        candidate_audit,
        maximum_rows,
        seed + 102,
        "bounded dependency and association computation",
    )
    numeric = [
        item for item in numeric_features
        if item in reference_frame.columns and item in candidate_frame.columns
    ]
    categorical = [
        item for item in categorical_features
        if item in reference_frame.columns and item in candidate_frame.columns
    ]
    if len(numeric) >= 2:
        for method in ("pearson", "spearman"):
            reference_matrix = correlation_matrix(reference_frame, numeric, method)
            candidate_matrix = correlation_matrix(candidate_frame, numeric, method)
            error = upper_triangle_mae(reference_matrix, candidate_matrix)
            output.append(result(
                f"dependency.{method}_correlation_matrix_mae",
                "dependency_fidelity",
                comparison,
                "NUMERIC_MATRIX",
                error,
                "lower_is_better",
                0.0,
                reference_audit,
                candidate_audit,
                unit="absolute_correlation",
                notes="Mean absolute error over finite upper-triangle entries",
            ))
    else:
        for method in ("pearson", "spearman"):
            output.append(not_applicable(
                f"dependency.{method}_correlation_matrix_mae",
                "dependency_fidelity",
                comparison,
                "NUMERIC_MATRIX",
                "At least two common numerical features are required",
                reference_audit,
                candidate_audit,
            ))

    mixed_columns = numeric + categorical
    if len(mixed_columns) >= 2:
        reference_matrix, _ = association_matrix(reference_frame, numeric, categorical)
        candidate_matrix, _ = association_matrix(candidate_frame, numeric, categorical)
        output.append(result(
            "dependency.mixed_association_matrix_mae",
            "dependency_fidelity",
            comparison,
            "MIXED_ASSOCIATION_MATRIX",
            upper_triangle_mae(reference_matrix, candidate_matrix),
            "lower_is_better",
            0.0,
            reference_audit,
            candidate_audit,
            unit="absolute_association",
            notes="Absolute Pearson, Cramer's V and correlation-ratio associations by pair type",
        ))
    else:
        output.append(not_applicable(
            "dependency.mixed_association_matrix_mae",
            "dependency_fidelity",
            comparison,
            "MIXED_ASSOCIATION_MATRIX",
            "At least two common features are required",
            reference_audit,
            candidate_audit,
        ))

    mi_errors: list[float] = []
    bins = int(settings.get("numeric_bin_count", 10))
    for declared_pair in settings.get("mutual_information_pairs", []):
        if len(declared_pair) != 2:
            continue
        left, right = str(declared_pair[0]), str(declared_pair[1])
        feature_name = f"{left}|{right}"
        if any(item not in reference_frame.columns or item not in candidate_frame.columns for item in (left, right)):
            output.append(not_applicable(
                "dependency.mutual_information_absolute_error",
                "dependency_fidelity",
                comparison,
                feature_name,
                "A declared mutual-information feature is absent",
                reference_audit,
                candidate_audit,
            ))
            continue
        reference_mi = mutual_information(reference_frame[left], reference_frame[right], bins)
        candidate_mi = mutual_information(candidate_frame[left], candidate_frame[right], bins)
        error = abs(candidate_mi - reference_mi)
        mi_errors.append(error)
        output.append(result(
            "dependency.mutual_information_absolute_error",
            "dependency_fidelity",
            comparison,
            feature_name,
            error,
            "lower_is_better",
            0.0,
            reference_audit,
            candidate_audit,
            unit="nats",
            notes=f"Numerical variables discretised into at most {bins} quantile bins",
        ))
    output.append(result(
        "dependency.mutual_information_absolute_error.mean",
        "dependency_fidelity",
        comparison,
        "DECLARED_MI_PAIRS",
        float(np.mean(mi_errors)) if mi_errors else None,
        "lower_is_better",
        0.0,
        reference_audit,
        candidate_audit,
        unit="mean_nats",
        applicable=bool(mi_errors),
        reason=None if mi_errors else "No declared mutual-information pair was applicable",
    ))

    propensity_features = list(dict.fromkeys(mixed_columns))
    maximum_propensity = int(settings.get("propensity_maximum_rows_per_class", maximum_rows))
    equal_size = min(len(reference_frame), len(candidate_frame), maximum_propensity)
    if equal_size < 20 or not propensity_features:
        output.append(not_applicable(
            "dependency.propensity_classifier_auc",
            "dependency_fidelity",
            comparison,
            "ALL_COMMON_FEATURES",
            "Propensity classification requires common features and at least 20 rows per role",
            reference_audit,
            candidate_audit,
        ))
        return output
    generator = np.random.default_rng(seed + 103)
    reference_sample = reference_frame.iloc[
        np.sort(generator.choice(len(reference_frame), equal_size, replace=False))
    ].reset_index(drop=True)
    candidate_sample = candidate_frame.iloc[
        np.sort(generator.choice(len(candidate_frame), equal_size, replace=False))
    ].reset_index(drop=True)
    reference_order = generator.permutation(equal_size)
    candidate_order = generator.permutation(equal_size)
    train_count = max(10, int(0.7 * equal_size))
    train = pd.concat([
        reference_sample.iloc[reference_order[:train_count]],
        candidate_sample.iloc[candidate_order[:train_count]],
    ], ignore_index=True)
    train_labels = np.concatenate([np.zeros(train_count, dtype=int), np.ones(train_count, dtype=int)])
    test = pd.concat([
        reference_sample.iloc[reference_order[train_count:]],
        candidate_sample.iloc[candidate_order[train_count:]],
    ], ignore_index=True)
    test_labels = np.concatenate([
        np.zeros(equal_size - train_count, dtype=int),
        np.ones(equal_size - train_count, dtype=int),
    ])
    encoder = DesignEncoder.fit(train, propensity_features)
    train_design = encoder.transform(train)
    test_design = encoder.transform(test)
    weights = fit_binary_logistic(
        train_design,
        train_labels,
        iterations=int(settings.get("propensity_iterations", 300)),
        learning_rate=float(settings.get("propensity_learning_rate", 0.1)),
    )
    auc = roc_auc_score(test_labels, predict_binary_logistic(test_design, weights))
    detectability = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
    propensity_reference_audit = DatasetAudit(
        **{**reference_audit.to_dict(), "evaluated_size": equal_size,
           "sampling_method": "equal_size_propensity_subsample", "sampling_seed": seed + 103,
           "sampling_reason": "balanced propensity-classifier comparison"}
    )
    propensity_candidate_audit = DatasetAudit(
        **{**candidate_audit.to_dict(), "evaluated_size": equal_size,
           "sampling_method": "equal_size_propensity_subsample", "sampling_seed": seed + 103,
           "sampling_reason": "balanced propensity-classifier comparison"}
    )
    output.append(result(
        "dependency.propensity_classifier_auc",
        "dependency_fidelity",
        comparison,
        "ALL_COMMON_FEATURES",
        detectability,
        "closest_to_0.5",
        0.5,
        propensity_reference_audit,
        propensity_candidate_audit,
        unit="roc_auc",
        notes=(
            "A value near 0.5 means only that this classifier had difficulty distinguishing the samples; "
            "it does not prove distributional equivalence"
        ),
    ))
    output.append(result(
        "dependency.propensity_auc_distance_from_half",
        "dependency_fidelity",
        comparison,
        "ALL_COMMON_FEATURES",
        abs(detectability - 0.5),
        "lower_is_better",
        0.0,
        propensity_reference_audit,
        propensity_candidate_audit,
        unit="roc_auc_absolute_difference",
    ))
    return output
