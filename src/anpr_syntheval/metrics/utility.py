"""Train-Real/Test-Real and Train-Synthetic/Test-Real utility evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from ..datasets import DatasetAudit
from ..registry import MetricResult
from ..schema import SchemaConfig
from ..statistics import (
    DesignEncoder,
    classification_scores,
    nearest_centroid_predict,
    regression_scores,
    ridge_fit,
    ridge_predict,
    roc_auc_score,
)
from ._helpers import cap_frame, not_applicable, result


def _binary_centroid_auc(
    train_design: np.ndarray,
    train_labels: pd.Series,
    test_design: np.ndarray,
    test_labels: pd.Series,
) -> float:
    labels = train_labels.astype("string").fillna("<MISSING>").to_numpy()
    classes = np.unique(labels)
    if len(classes) != 2:
        return float("nan")
    centroids = np.vstack([train_design[labels == item].mean(axis=0) for item in classes])
    distances = (
        np.sum(test_design ** 2, axis=1)[:, None]
        + np.sum(centroids ** 2, axis=1)[None, :]
        - 2.0 * test_design @ centroids.T
    )
    score = distances[:, 0] - distances[:, 1]
    truth = (test_labels.astype("string").fillna("<MISSING>").to_numpy() == classes[1]).astype(int)
    return roc_auc_score(truth, score)


def _task_leakage_reason(task: dict[str, Any], schema: SchemaConfig) -> str | None:
    target = str(task.get("target", ""))
    features = {str(item) for item in task.get("features", [])}
    if target in features:
        return "Target appears directly among predictors"
    rule = schema.leakage_rules.get(target, {})
    components = {str(item) for item in rule.get("direct_components", [])}
    if components and components.issubset(features):
        return str(rule.get("reason", "Predictors contain all direct target components"))
    return None


def _utility_not_applicable(
    task_name: str,
    reason: str,
    holdout_audit: DatasetAudit,
    synthetic_audit: DatasetAudit,
) -> list[MetricResult]:
    return [not_applicable(
        "utility.task",
        "utility",
        "TSTR_relative_to_TRTR",
        task_name,
        reason,
        holdout_audit,
        synthetic_audit,
    )]


def evaluate_utility(
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
    real_train, train_audit = cap_frame(
        real_train, train_audit, maximum_rows, seed + 201, "bounded utility training"
    )
    real_holdout, holdout_audit = cap_frame(
        real_holdout, holdout_audit, maximum_rows, seed + 202, "bounded held-out utility testing"
    )
    synthetic, synthetic_audit = cap_frame(
        synthetic, synthetic_audit, maximum_rows, seed + 203, "bounded synthetic utility training"
    )
    equal_train_size = min(len(real_train), len(synthetic))
    generator = np.random.default_rng(seed + 204)
    real_train = real_train.iloc[
        np.sort(generator.choice(len(real_train), equal_train_size, replace=False))
    ].reset_index(drop=True)
    synthetic = synthetic.iloc[
        np.sort(generator.choice(len(synthetic), equal_train_size, replace=False))
    ].reset_index(drop=True)
    train_audit = replace(
        train_audit,
        evaluated_size=equal_train_size,
        sampling_method="equal_size_utility_training",
        sampling_seed=seed + 204,
        sampling_reason="equal training sizes required for TRTR/TSTR comparison",
    )
    synthetic_audit = replace(
        synthetic_audit,
        evaluated_size=equal_train_size,
        sampling_method="equal_size_utility_training",
        sampling_seed=seed + 204,
        sampling_reason="equal training sizes required for TRTR/TSTR comparison",
    )
    tasks = settings.get("tasks", [])
    if not tasks:
        return _utility_not_applicable(
            "NO_TASK", "No utility task is configured", holdout_audit, synthetic_audit
        )
    for task in tasks:
        name = str(task.get("name", task.get("target", "unnamed_task")))
        target = str(task.get("target", ""))
        task_type = str(task.get("type", "classification"))
        features = [str(item) for item in task.get("features", [])]
        missing = [
            item for item in features + [target]
            if any(item not in frame.columns for frame in (real_train, real_holdout, synthetic))
        ]
        if missing:
            output.extend(_utility_not_applicable(
                name,
                "Task columns are missing from one or more roles",
                holdout_audit,
                synthetic_audit,
            ))
            continue
        leakage = _task_leakage_reason(task, schema)
        if leakage:
            output.extend(_utility_not_applicable(name, leakage, holdout_audit, synthetic_audit))
            continue
        if not features:
            output.extend(_utility_not_applicable(
                name, "Task has no predictors", holdout_audit, synthetic_audit
            ))
            continue
        encoder = DesignEncoder.fit(real_train, features)
        x_real = encoder.transform(real_train)
        x_holdout = encoder.transform(real_holdout)
        x_synthetic = encoder.transform(synthetic)
        if x_real.shape[1] == 0:
            output.extend(_utility_not_applicable(
                name, "Predictors produced an empty design matrix", holdout_audit, synthetic_audit
            ))
            continue
        if task_type == "classification":
            real_target = real_train[target].astype("string").fillna("<MISSING>")
            holdout_target = real_holdout[target].astype("string").fillna("<MISSING>")
            synthetic_target = synthetic[target].astype("string").fillna("<MISSING>")
            minimum = int(task.get("minimum_class_count", 2))
            if real_target.nunique() < 2 or holdout_target.nunique() < 2 or real_target.value_counts().min() < minimum:
                output.extend(_utility_not_applicable(
                    name,
                    "Classification requires at least two sufficiently represented real-training classes",
                    holdout_audit,
                    synthetic_audit,
                ))
                continue
            if synthetic_target.nunique() < 2:
                output.extend(_utility_not_applicable(
                    name,
                    "Synthetic training target contains fewer than two classes",
                    holdout_audit,
                    synthetic_audit,
                ))
                continue
            trtr_prediction = nearest_centroid_predict(x_real, real_target, x_holdout)
            tstr_prediction = nearest_centroid_predict(x_synthetic, synthetic_target, x_holdout)
            protocol_scores = {
                "TRTR": classification_scores(holdout_target, trtr_prediction),
                "TSTR": classification_scores(holdout_target, tstr_prediction),
            }
            if bool(settings.get("include_trts_diagnostic", True)):
                trts_prediction = nearest_centroid_predict(x_real, real_target, x_synthetic)
                protocol_scores["TRTS_DIAGNOSTIC"] = classification_scores(
                    synthetic_target, trts_prediction
                )
            for protocol, scores in protocol_scores.items():
                candidate_audit = train_audit if protocol.startswith("TRT") else synthetic_audit
                if protocol == "TRTS_DIAGNOSTIC":
                    candidate_audit = synthetic_audit
                for metric_name in ("macro_f1", "weighted_f1", "balanced_accuracy"):
                    output.append(result(
                        f"utility.{metric_name}",
                        "utility",
                        protocol,
                        name,
                        scores[metric_name],
                        "higher_is_better",
                        1.0,
                        holdout_audit,
                        candidate_audit,
                        unit="score",
                        notes="TRTS is diagnostic only" if protocol == "TRTS_DIAGNOSTIC" else None,
                    ))
            for metric_name in ("macro_f1", "weighted_f1", "balanced_accuracy"):
                baseline = protocol_scores["TRTR"][metric_name]
                tstr = protocol_scores["TSTR"][metric_name]
                relative = tstr / baseline if baseline > 0 else float("nan")
                output.append(result(
                    f"utility.relative_{metric_name}",
                    "utility",
                    "TSTR_relative_to_TRTR",
                    name,
                    relative,
                    "higher_is_better",
                    1.0,
                    holdout_audit,
                    synthetic_audit,
                    unit="ratio",
                ))
            trtr_auc = _binary_centroid_auc(x_real, real_target, x_holdout, holdout_target)
            tstr_auc = _binary_centroid_auc(x_synthetic, synthetic_target, x_holdout, holdout_target)
            for protocol, value, audit in (
                ("TRTR", trtr_auc, train_audit),
                ("TSTR", tstr_auc, synthetic_audit),
            ):
                applicable = bool(np.isfinite(value))
                output.append(result(
                    "utility.roc_auc",
                    "utility",
                    protocol,
                    name,
                    value if applicable else None,
                    "higher_is_better",
                    1.0,
                    holdout_audit,
                    audit,
                    unit="roc_auc",
                    applicable=applicable,
                    reason=None if applicable else "ROC-AUC is reported only for a binary target",
                ))
        elif task_type == "regression":
            y_real = pd.to_numeric(real_train[target], errors="coerce").to_numpy(float)
            y_holdout = pd.to_numeric(real_holdout[target], errors="coerce").to_numpy(float)
            y_synthetic = pd.to_numeric(synthetic[target], errors="coerce").to_numpy(float)
            if not all(np.isfinite(values).all() for values in (y_real, y_holdout, y_synthetic)):
                output.extend(_utility_not_applicable(
                    name, "Regression target contains non-finite values", holdout_audit, synthetic_audit
                ))
                continue
            alpha = float(settings.get("ridge_alpha", 1.0))
            trtr_scores = regression_scores(y_holdout, ridge_predict(x_holdout, ridge_fit(x_real, y_real, alpha)))
            tstr_scores = regression_scores(y_holdout, ridge_predict(x_holdout, ridge_fit(x_synthetic, y_synthetic, alpha)))
            for protocol, scores, audit in (
                ("TRTR", trtr_scores, train_audit),
                ("TSTR", tstr_scores, synthetic_audit),
            ):
                for metric_name, value in scores.items():
                    direction = "lower_is_better" if metric_name in {"mae", "rmse"} else "higher_is_better"
                    output.append(result(
                        f"utility.{metric_name}", "utility", protocol, name, value, direction,
                        0.0 if direction == "lower_is_better" else 1.0,
                        holdout_audit, audit, unit="target_unit" if metric_name != "r_squared" else "score"
                    ))
            for metric_name in ("mae", "rmse"):
                relative = trtr_scores[metric_name] / tstr_scores[metric_name] if tstr_scores[metric_name] > 0 else float("nan")
                output.append(result(
                    f"utility.relative_{metric_name}", "utility", "TSTR_relative_to_TRTR", name,
                    relative, "higher_is_better", 1.0, holdout_audit, synthetic_audit, unit="inverse_error_ratio"
                ))
            output.append(result(
                "utility.r_squared_difference", "utility", "TSTR_minus_TRTR", name,
                tstr_scores["r_squared"] - trtr_scores["r_squared"], "closest_to_zero", 0.0,
                holdout_audit, synthetic_audit, unit="score_difference"
            ))
        else:
            output.extend(_utility_not_applicable(
                name, f"Unsupported utility task type: {task_type}", holdout_audit, synthetic_audit
            ))
    return output
