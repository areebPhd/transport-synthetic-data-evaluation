"""Dependency-light statistical primitives used by SynthEval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


EPSILON = 1e-12


def numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values[np.isfinite(values)]


def probabilities_from_counts(counts: Sequence[float]) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def js_divergence(probability: Sequence[float], comparison: Sequence[float]) -> float:
    """Base-two Jensen--Shannon divergence in [0, 1]."""

    p = probabilities_from_counts(probability)
    q = probabilities_from_counts(comparison)
    if p.shape != q.shape:
        raise ValueError("Jensen--Shannon inputs must have equal shape")
    mixture = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return float(0.5 * kl(p, mixture) + 0.5 * kl(q, mixture))


def total_variation(probability: Sequence[float], comparison: Sequence[float]) -> float:
    p = probabilities_from_counts(probability)
    q = probabilities_from_counts(comparison)
    return float(0.5 * np.abs(p - q).sum())


def aligned_counts(reference: pd.Series, candidate: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ref = reference.astype("string").fillna("<MISSING>").value_counts(dropna=False)
    cand = candidate.astype("string").fillna("<MISSING>").value_counts(dropna=False)
    labels = sorted(set(ref.index.astype(str)).union(cand.index.astype(str)))
    return (
        np.asarray([ref.get(item, 0) for item in labels], dtype=float),
        np.asarray([cand.get(item, 0) for item in labels], dtype=float),
        labels,
    )


def ks_statistic(reference: Sequence[float], candidate: Sequence[float]) -> float:
    left = np.sort(np.asarray(reference, dtype=float))
    right = np.sort(np.asarray(candidate, dtype=float))
    if not len(left) or not len(right):
        return float("nan")
    combined = np.sort(np.concatenate([left, right]))
    cdf_left = np.searchsorted(left, combined, side="right") / len(left)
    cdf_right = np.searchsorted(right, combined, side="right") / len(right)
    return float(np.max(np.abs(cdf_left - cdf_right)))


def wasserstein_distance(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """Numerical integral between empirical one-dimensional quantile functions."""

    left = np.sort(np.asarray(reference, dtype=float))
    right = np.sort(np.asarray(candidate, dtype=float))
    if not len(left) or not len(right):
        return float("nan")
    points = max(len(left), len(right), 2)
    quantiles = np.linspace(0.0, 1.0, points)
    # The sorted vectors are empirical quantile values already.  Linear
    # interpolation on their probability grids is algebraically equivalent to
    # ``np.quantile(..., method="linear")`` while avoiding an expensive repeat
    # quantile calculation for every point.  Equal-size samples, the principal
    # evaluation case, require no interpolation.
    left_q = (
        left
        if len(left) == points
        else np.interp(quantiles, np.linspace(0.0, 1.0, len(left)), left)
    )
    right_q = (
        right
        if len(right) == points
        else np.interp(quantiles, np.linspace(0.0, 1.0, len(right)), right)
    )
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(np.abs(left_q - right_q), quantiles))


def freedman_diaconis_edges(values: Sequence[float], maximum_bins: int = 80) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2 or np.min(data) == np.max(data):
        centre = float(data[0]) if len(data) else 0.0
        return np.asarray([centre - 0.5, centre + 0.5])
    q25, q75 = np.quantile(data, [0.25, 0.75])
    width = 2.0 * (q75 - q25) / np.cbrt(len(data))
    if width <= 0 or not np.isfinite(width):
        bins = int(np.ceil(np.log2(len(data)) + 1))
    else:
        bins = int(np.ceil((data.max() - data.min()) / width))
    bins = max(2, min(int(maximum_bins), bins))
    return np.linspace(float(data.min()), float(data.max()), bins + 1)


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= EPSILON or np.std(y) <= EPSILON:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return pearson_correlation(rankdata(x[mask]), rankdata(y[mask]))


def correlation_matrix(frame: pd.DataFrame, columns: Sequence[str], method: str) -> np.ndarray:
    matrix = np.full((len(columns), len(columns)), np.nan, dtype=float)
    for row, left in enumerate(columns):
        for column, right in enumerate(columns):
            if row == column:
                matrix[row, column] = 1.0
            elif column < row:
                matrix[row, column] = matrix[column, row]
            else:
                function = pearson_correlation if method == "pearson" else spearman_correlation
                matrix[row, column] = function(frame[left], frame[right])
    return matrix


def upper_triangle_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Matrices must have equal two-dimensional shape")
    mask = np.triu(np.ones(reference.shape, dtype=bool), k=1)
    mask &= np.isfinite(reference) & np.isfinite(candidate)
    return float(np.mean(np.abs(reference[mask] - candidate[mask]))) if mask.any() else float("nan")


def cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left.astype("string"), right.astype("string"), dropna=False).to_numpy(float)
    total = table.sum()
    if total <= 0 or min(table.shape) < 2:
        return float("nan")
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    mask = expected > 0
    chi_square = np.sum(((table - expected) ** 2)[mask] / expected[mask])
    denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(max(0.0, chi_square / denominator))) if denominator > 0 else float("nan")


def correlation_ratio(categories: pd.Series, numerical: pd.Series) -> float:
    values = pd.to_numeric(numerical, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    labels = categories.astype("string").fillna("<MISSING>").to_numpy()
    mask = np.isfinite(values)
    values = values[mask]
    labels = labels[mask]
    if len(values) < 2 or np.var(values) <= EPSILON:
        return float("nan")
    mean = values.mean()
    between = 0.0
    for category in np.unique(labels):
        group = values[labels == category]
        between += len(group) * (group.mean() - mean) ** 2
    return float(np.sqrt(max(0.0, between / np.sum((values - mean) ** 2))))


def association_matrix(
    frame: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    columns = list(numeric) + [item for item in categorical if item not in numeric]
    numeric_set = set(numeric)
    matrix = np.full((len(columns), len(columns)), np.nan, dtype=float)
    for row, left in enumerate(columns):
        matrix[row, row] = 1.0
        for column in range(row + 1, len(columns)):
            right = columns[column]
            if left in numeric_set and right in numeric_set:
                value = abs(pearson_correlation(frame[left], frame[right]))
            elif left not in numeric_set and right not in numeric_set:
                value = cramers_v(frame[left], frame[right])
            elif left in numeric_set:
                value = correlation_ratio(frame[right], frame[left])
            else:
                value = correlation_ratio(frame[left], frame[right])
            matrix[row, column] = matrix[column, row] = value
    return matrix, columns


def discretise(series: pd.Series, bins: int = 10) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.nunique() <= bins:
            return pd.factorize(values.astype("string").fillna("<MISSING>"), sort=True)[0]
        edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 2:
            return np.zeros(len(series), dtype=int)
        return np.digitize(values.fillna(finite.median()), edges[1:-1], right=True)
    return pd.factorize(series.astype("string").fillna("<MISSING>"), sort=True)[0]


def mutual_information(left: pd.Series, right: pd.Series, bins: int = 10) -> float:
    x = discretise(left, bins)
    y = discretise(right, bins)
    table = pd.crosstab(x, y).to_numpy(float)
    joint = table / table.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    positive = y == 1
    negative = y == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_sum = ranks[positive].sum()
    auc = (rank_sum - positive.sum() * (positive.sum() + 1) / 2) / (
        positive.sum() * negative.sum()
    )
    return float(auc)


@dataclass
class DesignEncoder:
    numerical: list[str]
    categorical: list[str]
    medians: dict[str, float]
    scales: dict[str, float]
    levels: dict[str, list[str]]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Sequence[str], maximum_levels: int = 100) -> "DesignEncoder":
        numerical: list[str] = []
        categorical: list[str] = []
        medians: dict[str, float] = {}
        scales: dict[str, float] = {}
        levels: dict[str, list[str]] = {}
        for feature in features:
            if pd.api.types.is_numeric_dtype(frame[feature]) and not pd.api.types.is_bool_dtype(frame[feature]):
                numerical.append(feature)
                values = pd.to_numeric(frame[feature], errors="coerce")
                median = float(values.median())
                scale = float(values.quantile(0.75) - values.quantile(0.25))
                medians[feature] = median if np.isfinite(median) else 0.0
                scales[feature] = scale if np.isfinite(scale) and scale > EPSILON else 1.0
            else:
                categorical.append(feature)
                counts = frame[feature].astype("string").fillna("<MISSING>").value_counts()
                levels[feature] = [str(item) for item in counts.index[:maximum_levels]]
        return cls(numerical, categorical, medians, scales, levels)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for feature in self.numerical:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(self.medians[feature])
            columns.append(((values.to_numpy(float) - self.medians[feature]) / self.scales[feature])[:, None])
        for feature in self.categorical:
            values = frame[feature].astype("string").fillna("<MISSING>").to_numpy()
            for level in self.levels[feature]:
                columns.append((values == level).astype(float)[:, None])
        if not columns:
            return np.empty((len(frame), 0), dtype=float)
        return np.hstack(columns)


def fit_binary_logistic(
    design: np.ndarray,
    labels: Sequence[int],
    iterations: int = 300,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> np.ndarray:
    x = np.hstack([np.ones((len(design), 1)), design])
    y = np.asarray(labels, dtype=float)
    weights = np.zeros(x.shape[1], dtype=float)
    for _ in range(iterations):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = x.T @ (probability - y) / len(x)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def predict_binary_logistic(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.hstack([np.ones((len(design), 1)), design])
    logits = np.clip(x @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def nearest_centroid_predict(
    train_design: np.ndarray,
    train_labels: Sequence[object],
    test_design: np.ndarray,
) -> np.ndarray:
    labels = pd.Series(train_labels, dtype="string").fillna("<MISSING>").to_numpy()
    classes = np.unique(labels)
    centroids = np.vstack([train_design[labels == item].mean(axis=0) for item in classes])
    distances = (
        np.sum(test_design ** 2, axis=1)[:, None]
        + np.sum(centroids ** 2, axis=1)[None, :]
        - 2.0 * test_design @ centroids.T
    )
    return classes[np.argmin(distances, axis=1)]


def classification_scores(actual: Sequence[object], predicted: Sequence[object]) -> dict[str, float]:
    truth = pd.Series(actual, dtype="string").fillna("<MISSING>").to_numpy()
    pred = pd.Series(predicted, dtype="string").fillna("<MISSING>").to_numpy()
    classes = np.unique(np.concatenate([truth, pred]))
    f1_values: list[float] = []
    recalls: list[float] = []
    supports: list[int] = []
    for label in classes:
        true_positive = int(((truth == label) & (pred == label)).sum())
        false_positive = int(((truth != label) & (pred == label)).sum())
        false_negative = int(((truth == label) & (pred != label)).sum())
        support = int((truth == label).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        supports.append(support)
    support_array = np.asarray(supports, dtype=float)
    return {
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": float(np.average(f1_values, weights=support_array)) if support_array.sum() else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)),
        "accuracy": float(np.mean(truth == pred)),
    }


def ridge_fit(design: np.ndarray, target: Sequence[float], alpha: float = 1.0) -> np.ndarray:
    x = np.hstack([np.ones((len(design), 1)), design])
    y = np.asarray(target, dtype=float)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def ridge_predict(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(design), 1)), design]) @ weights


def regression_scores(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    residual = truth - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > EPSILON else float("nan")
    return {"mae": mae, "rmse": rmse, "r_squared": r_squared}


def percentile_bootstrap_ci(
    values: Sequence[float], repetitions: int, confidence: float, seed: int
) -> tuple[float, float]:
    data = np.asarray([item for item in values if np.isfinite(item)], dtype=float)
    if not len(data):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = generator.choice(data, size=len(data), replace=True).mean()
    tail = (1.0 - confidence) / 2.0
    return tuple(float(item) for item in np.quantile(means, [tail, 1.0 - tail]))
