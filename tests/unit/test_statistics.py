from __future__ import annotations

import numpy as np
import pandas as pd

from anpr_syntheval.statistics import (
    classification_scores,
    js_divergence,
    ks_statistic,
    mutual_information,
    pearson_correlation,
    roc_auc_score,
    spearman_correlation,
    total_variation,
    wasserstein_distance,
)


def test_probability_distances_have_expected_identity_and_extremes() -> None:
    assert js_divergence([5, 5], [5, 5]) == 0
    assert total_variation([5, 5], [5, 5]) == 0
    assert np.isclose(js_divergence([10, 0], [0, 10]), 1.0)
    assert np.isclose(total_variation([10, 0], [0, 10]), 1.0)


def test_continuous_distances_detect_shift() -> None:
    base = np.arange(100, dtype=float)
    shifted = base + 5
    assert ks_statistic(base, base) == 0
    assert wasserstein_distance(base, base) == 0
    assert np.isclose(wasserstein_distance(base, shifted), 5.0, atol=0.1)
    assert ks_statistic(base, shifted) > 0


def test_wasserstein_matches_legacy_linear_quantile_definition() -> None:
    generator = np.random.default_rng(20260728)
    integrate = getattr(np, "trapezoid", np.trapz)
    for left_size, right_size in ((2, 5), (17, 17), (31, 79), (500, 431)):
        left = generator.normal(size=left_size)
        right = generator.lognormal(size=right_size)
        points = max(left_size, right_size, 2)
        quantiles = np.linspace(0.0, 1.0, points)
        legacy = float(
            integrate(
                np.abs(
                    np.quantile(np.sort(left), quantiles)
                    - np.quantile(np.sort(right), quantiles)
                ),
                quantiles,
            )
        )
        assert np.isclose(
            wasserstein_distance(left, right),
            legacy,
            rtol=1e-13,
            atol=1e-13,
        )


def test_correlations_and_mutual_information() -> None:
    values = np.arange(50, dtype=float)
    assert np.isclose(pearson_correlation(values, values * 2), 1.0)
    assert np.isclose(spearman_correlation(values, -values), -1.0)
    dependent = mutual_information(pd.Series(values), pd.Series(values), bins=5)
    independent = mutual_information(pd.Series(values), pd.Series(np.roll(values, 7)), bins=5)
    assert dependent >= independent


def test_auc_and_classification_scores() -> None:
    assert np.isclose(roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
    scores = classification_scores(["a", "a", "b", "b"], ["a", "a", "b", "b"])
    assert scores["macro_f1"] == 1.0
    assert scores["balanced_accuracy"] == 1.0
