"""Empirical privacy-risk diagnostics with held-out comparison."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..schema import SchemaConfig
from ..statistics import EPSILON, roc_auc_score
from ._helpers import not_applicable, result


def _row_hashes(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    normalised = frame[list(columns)].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(normalised[column]):
            normalised[column] = pd.to_numeric(normalised[column], errors="coerce").round(10)
        else:
            normalised[column] = normalised[column].astype("string").fillna("<MISSING>")
    return pd.util.hash_pandas_object(normalised, index=False).to_numpy(dtype="uint64")


class _GowerEncoder:
    def __init__(self, train: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.lower: dict[str, float] = {}
        self.scale: dict[str, float] = {}
        for feature in self.numeric:
            values = pd.to_numeric(train[feature], errors="coerce")
            lower, upper = values.quantile([0.05, 0.95])
            median = values.median()
            if not np.isfinite(lower):
                lower = median if np.isfinite(median) else 0.0
            difference = float(upper - lower) if np.isfinite(upper) else 0.0
            self.lower[feature] = float(lower)
            self.scale[feature] = difference if difference > EPSILON else 1.0

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numeric_columns: list[np.ndarray] = []
        for feature in self.numeric:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(self.lower[feature])
            scaled = np.clip(
                (values.to_numpy(float) - self.lower[feature]) / self.scale[feature], 0.0, 1.0
            )
            numeric_columns.append(scaled[:, None])
        categorical_columns: list[np.ndarray] = []
        for feature in self.categorical:
            categorical_columns.append(
                frame[feature].astype("string").fillna("<MISSING>").to_numpy()[:, None]
            )
        numeric = np.hstack(numeric_columns) if numeric_columns else np.empty((len(frame), 0))
        categorical = (
            np.hstack(categorical_columns) if categorical_columns else np.empty((len(frame), 0), dtype=str)
        )
        return numeric, categorical


def _nearest_distances(
    query: tuple[np.ndarray, np.ndarray],
    reference: tuple[np.ndarray, np.ndarray],
    chunk_size: int,
    second: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    query_numeric, query_categorical = query
    reference_numeric, reference_categorical = reference
    component_count = query_numeric.shape[1] + query_categorical.shape[1]
    if component_count == 0 or len(reference_numeric) == 0:
        raise ValueError("Gower distance requires features and reference rows")
    nearest = np.empty(len(query_numeric), dtype=float)
    second_available = second and len(reference_numeric) >= 2
    second_nearest = np.empty(len(query_numeric), dtype=float) if second_available else None
    for start in range(0, len(query_numeric), chunk_size):
        end = min(len(query_numeric), start + chunk_size)
        distances = np.zeros((end - start, len(reference_numeric)), dtype=float)
        for index in range(query_numeric.shape[1]):
            distances += np.abs(
                query_numeric[start:end, index, None] - reference_numeric[None, :, index]
            )
        for index in range(query_categorical.shape[1]):
            distances += (
                query_categorical[start:end, index, None] != reference_categorical[None, :, index]
            )
        distances /= component_count
        if second_available:
            partitioned = np.partition(distances, kth=1, axis=1)[:, :2]
            partitioned.sort(axis=1)
            nearest[start:end] = partitioned[:, 0]
            assert second_nearest is not None
            second_nearest[start:end] = partitioned[:, 1]
        else:
            nearest[start:end] = distances.min(axis=1)
    return nearest, second_nearest


def _equal_privacy_samples(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    maximum_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    size = min(len(train), len(holdout), len(synthetic), maximum_rows)
    generator = np.random.default_rng(seed)
    frames = []
    for frame in (train, holdout, synthetic):
        indices = np.sort(generator.choice(len(frame), size=size, replace=False))
        frames.append(frame.iloc[indices].reset_index(drop=True))
    return frames[0], frames[1], frames[2], size


def evaluate_privacy(
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synthetic: pd.DataFrame,
    settings: dict[str, Any],
    schema: SchemaConfig,
    train_audit: DatasetAudit,
    holdout_audit: DatasetAudit,
    synthetic_audit: DatasetAudit,
    seed: int,
    maximum_rows: int,
) -> list[MetricResult]:
    output: list[MetricResult] = []
    common = [
        feature for feature in schema.all_features
        if all(feature in frame.columns for frame in (real_train, real_holdout, synthetic))
    ]
    if not common:
        return [not_applicable(
            "privacy.empirical_diagnostics",
            "empirical_privacy_risk",
            "synthetic_vs_training_and_holdout",
            None,
            "No common non-identifier features are available",
            train_audit,
            synthetic_audit,
        )]

    train_hash = set(_row_hashes(real_train, common))
    holdout_hash = set(_row_hashes(real_holdout, common))
    synthetic_hash = _row_hashes(synthetic, common)
    exact_train = float(np.mean([item in train_hash for item in synthetic_hash]))
    exact_holdout = float(np.mean([item in holdout_hash for item in synthetic_hash]))
    duplicate_rate = float(1.0 - len(set(synthetic_hash)) / len(synthetic_hash))
    for metric_id, value, audit in (
        ("privacy.exact_training_row_match_rate", exact_train, train_audit),
        ("privacy.exact_holdout_row_match_rate", exact_holdout, holdout_audit),
        ("privacy.duplicate_synthetic_row_rate", duplicate_rate, synthetic_audit),
    ):
        output.append(result(
            metric_id,
            "empirical_privacy_risk",
            "synthetic_vs_training_and_holdout",
            "ALL_COMMON_FEATURES",
            value,
            "lower_is_better",
            0.0,
            audit,
            synthetic_audit,
            unit="proportion",
            notes="Exact matching is a warning diagnostic, not proof of privacy or its absence",
        ))

    numeric = [
        item for item in schema.privacy_distance_features.get("continuous", [])
        if all(item in frame.columns for frame in (real_train, real_holdout, synthetic))
    ]
    categorical = [
        item for item in schema.privacy_distance_features.get("categorical", [])
        if all(item in frame.columns for frame in (real_train, real_holdout, synthetic))
    ]
    if not numeric and not categorical:
        output.append(not_applicable(
            "privacy.nearest_neighbour_distance",
            "empirical_privacy_risk",
            "synthetic_vs_training_and_holdout",
            None,
            "No declared privacy-distance features are common to all roles",
            train_audit,
            synthetic_audit,
        ))
        return output

    train, holdout, candidate, sample_size = _equal_privacy_samples(
        real_train, real_holdout, synthetic, maximum_rows, seed + 301
    )
    privacy_reason = "equal-size bounded mixed-type nearest-neighbour computation"
    train_audit = replace(
        train_audit, evaluated_size=sample_size, sampling_method="equal_size_privacy_subsample",
        sampling_seed=seed + 301, sampling_reason=privacy_reason
    )
    holdout_audit = replace(
        holdout_audit, evaluated_size=sample_size, sampling_method="equal_size_privacy_subsample",
        sampling_seed=seed + 301, sampling_reason=privacy_reason
    )
    synthetic_audit = replace(
        synthetic_audit, evaluated_size=sample_size, sampling_method="equal_size_privacy_subsample",
        sampling_seed=seed + 301, sampling_reason=privacy_reason
    )
    encoder = _GowerEncoder(train, numeric, categorical)
    encoded_train = encoder.transform(train)
    encoded_holdout = encoder.transform(holdout)
    encoded_candidate = encoder.transform(candidate)
    chunk_size = int(settings.get("nearest_neighbour_chunk_size", 128))
    distance_train, second_train = _nearest_distances(
        encoded_candidate, encoded_train, chunk_size, second=True
    )
    distance_holdout, _ = _nearest_distances(encoded_candidate, encoded_holdout, chunk_size)
    distance_real = np.minimum(distance_train, distance_holdout)
    nn_ratio = (
        distance_train / np.maximum(second_train, EPSILON)
        if second_train is not None else None
    )
    train_holdout_ratio = distance_train / np.maximum(distance_holdout, EPSILON)
    summaries = (
        ("privacy.closest_training_distance.minimum", float(distance_train.min()), train_audit),
        ("privacy.closest_training_distance.mean", float(distance_train.mean()), train_audit),
        ("privacy.closest_training_distance.median", float(np.median(distance_train)), train_audit),
        ("privacy.closest_holdout_distance.minimum", float(distance_holdout.min()), holdout_audit),
        ("privacy.closest_holdout_distance.mean", float(distance_holdout.mean()), holdout_audit),
        ("privacy.closest_holdout_distance.median", float(np.median(distance_holdout)), holdout_audit),
        ("privacy.closest_real_distance.mean", float(distance_real.mean()), holdout_audit),
        ("privacy.closest_real_distance.median", float(np.median(distance_real)), holdout_audit),
    )
    for metric_id, value, audit in summaries:
        output.append(result(
            metric_id,
            "empirical_privacy_risk",
            "synthetic_vs_training_and_holdout",
            "GOWER_MIXED_FEATURES",
            value,
            "higher_is_generally_lower_proximity_risk",
            "context_dependent",
            audit,
            synthetic_audit,
            unit="gower_distance",
            notes="Robust training-range scaled numerical components plus categorical mismatch components",
        ))
    if nn_ratio is None:
        output.append(not_applicable(
            "privacy.nearest_neighbour_distance_ratio.median",
            "empirical_privacy_risk",
            "synthetic_vs_training",
            "GOWER_MIXED_FEATURES",
            "At least two training-reference rows are required for a second-neighbour ratio",
            train_audit,
            synthetic_audit,
        ))
    else:
        output.append(result(
            "privacy.nearest_neighbour_distance_ratio.median",
            "empirical_privacy_risk",
            "synthetic_vs_training",
            "GOWER_MIXED_FEATURES",
            float(np.median(nn_ratio)),
            "higher_is_generally_lower_isolation_risk",
            1.0,
            train_audit,
            synthetic_audit,
            unit="ratio",
        ))
    output.append(result(
        "privacy.train_holdout_distance_ratio.median",
        "empirical_privacy_risk",
        "training_relative_to_holdout",
        "GOWER_MIXED_FEATURES",
        float(np.median(train_holdout_ratio)),
        "closer_to_or_above_1_is_lower_training_proximity_risk",
        ">=1",
        holdout_audit,
        synthetic_audit,
        unit="ratio",
    ))
    output.append(result(
        "privacy.synthetic_closer_to_training_rate",
        "empirical_privacy_risk",
        "training_relative_to_holdout",
        "GOWER_MIXED_FEATURES",
        float(np.mean(distance_train < distance_holdout)),
        "lower_is_better",
        0.5,
        holdout_audit,
        synthetic_audit,
        unit="proportion",
        notes="A value near 0.5 is a sampling reference, not a formal threshold",
    ))

    train_to_synthetic, _ = _nearest_distances(encoded_train, encoded_candidate, chunk_size)
    holdout_to_synthetic, _ = _nearest_distances(encoded_holdout, encoded_candidate, chunk_size)
    membership_labels = np.concatenate([
        np.ones(len(train_to_synthetic), dtype=int), np.zeros(len(holdout_to_synthetic), dtype=int)
    ])
    membership_scores = -np.concatenate([train_to_synthetic, holdout_to_synthetic])
    membership_auc = roc_auc_score(membership_labels, membership_scores)
    output.append(result(
        "privacy.membership_inference_distance_auc",
        "empirical_privacy_risk",
        "training_vs_holdout_membership_baseline",
        "GOWER_MIXED_FEATURES",
        membership_auc,
        "closest_to_0.5_is_lower_observed_attack_signal",
        0.5,
        holdout_audit,
        synthetic_audit,
        unit="roc_auc",
        notes="Distance-threshold baseline only; not an exhaustive membership-inference attack",
    ))

    rare_columns = [
        item for item in schema.rare_combination_features
        if all(item in frame.columns for frame in (real_train, synthetic))
    ]
    if rare_columns:
        maximum_count = int(settings.get("rare_combination_maximum_count", 10))
        train_combinations = real_train[rare_columns].astype("string").value_counts()
        rare_index = set(train_combinations[train_combinations <= maximum_count].index)
        synthetic_combinations = synthetic[rare_columns].astype("string")
        synthetic_tuples = list(synthetic_combinations.itertuples(index=False, name=None))
        produced = set(synthetic_tuples)
        recall = len(rare_index & produced) / len(rare_index) if rare_index else float("nan")
        row_rate = float(np.mean([item in rare_index for item in synthetic_tuples]))
        output.append(result(
            "privacy.rare_combination_replication_recall",
            "empirical_privacy_risk",
            "synthetic_vs_training",
            "DECLARED_RARE_COMBINATION",
            recall if np.isfinite(recall) else None,
            "lower_is_generally_lower_replication_risk",
            0.0,
            train_audit,
            synthetic_audit,
            unit="proportion",
            applicable=bool(np.isfinite(recall)),
            reason=None if np.isfinite(recall) else "No combination met the rare-count definition",
        ))
        output.append(result(
            "privacy.synthetic_row_rare_combination_match_rate",
            "empirical_privacy_risk",
            "synthetic_vs_training",
            "DECLARED_RARE_COMBINATION",
            row_rate,
            "lower_is_better",
            0.0,
            train_audit,
            synthetic_audit,
            unit="proportion",
        ))
    else:
        output.append(not_applicable(
            "privacy.rare_combination_replication_recall",
            "empirical_privacy_risk",
            "synthetic_vs_training",
            None,
            "Declared rare-combination features are unavailable",
            train_audit,
            synthetic_audit,
        ))

    for task in settings.get("attribute_inference_tasks", []):
        target = str(task.get("target", ""))
        known = [str(item) for item in task.get("known_features", [])]
        if not target or any(
            item not in frame.columns for item in known + [target] for frame in (holdout, candidate)
        ):
            output.append(not_applicable(
                "privacy.attribute_inference_accuracy",
                "empirical_privacy_risk",
                "synthetic_to_holdout_attribute_baseline",
                target or "UNNAMED_TARGET",
                "Attribute-inference columns are unavailable",
                holdout_audit,
                synthetic_audit,
            ))
            continue
        task_numeric = [item for item in known if item in schema.numeric_features]
        task_categorical = [item for item in known if item not in task_numeric]
        task_encoder = _GowerEncoder(train, task_numeric, task_categorical)
        query = task_encoder.transform(holdout)
        reference = task_encoder.transform(candidate)
        query_numeric, query_category = query
        reference_numeric, reference_category = reference
        component_count = query_numeric.shape[1] + query_category.shape[1]
        if component_count == 0:
            output.append(not_applicable(
                "privacy.attribute_inference_accuracy", "empirical_privacy_risk",
                "synthetic_to_holdout_attribute_baseline", target,
                "No known attributes are usable", holdout_audit, synthetic_audit
            ))
            continue
        predictions: list[str] = []
        candidate_target = candidate[target].astype("string").fillna("<MISSING>").to_numpy()
        for start in range(0, len(holdout), chunk_size):
            end = min(len(holdout), start + chunk_size)
            distances = np.zeros((end - start, len(candidate)), dtype=float)
            for index in range(query_numeric.shape[1]):
                distances += np.abs(
                    query_numeric[start:end, index, None] - reference_numeric[None, :, index]
                )
            for index in range(query_category.shape[1]):
                distances += (
                    query_category[start:end, index, None] != reference_category[None, :, index]
                )
            predictions.extend(candidate_target[np.argmin(distances, axis=1)])
        truth = holdout[target].astype("string").fillna("<MISSING>").to_numpy()
        accuracy = float(np.mean(truth == np.asarray(predictions)))
        majority = float(candidate[target].astype("string").value_counts(normalize=True).max())
        output.append(result(
            "privacy.attribute_inference_accuracy",
            "empirical_privacy_risk",
            "synthetic_to_holdout_attribute_baseline",
            target,
            accuracy,
            "lower_is_generally_lower_observed_inference_risk",
            "context_dependent",
            holdout_audit,
            synthetic_audit,
            unit="accuracy",
        ))
        output.append(result(
            "privacy.attribute_inference_advantage_over_majority",
            "empirical_privacy_risk",
            "synthetic_to_holdout_attribute_baseline",
            target,
            accuracy - majority,
            "lower_is_better",
            0.0,
            holdout_audit,
            synthetic_audit,
            unit="accuracy_difference",
        ))
    return output
