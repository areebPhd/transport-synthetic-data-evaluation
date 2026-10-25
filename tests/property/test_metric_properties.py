from __future__ import annotations

import numpy as np
import pandas as pd

from anpr_syntheval.statistics import js_divergence, ks_statistic, total_variation, wasserstein_distance


def test_probability_metrics_are_symmetric_and_bounded_across_random_cases() -> None:
    for seed in range(20):
        generator = np.random.default_rng(seed)
        left = generator.integers(0, 100, 12)
        right = generator.integers(0, 100, 12)
        for function in (js_divergence, total_variation):
            forward = function(left, right)
            backward = function(right, left)
            assert np.isclose(forward, backward)
            assert 0 <= forward <= 1 + 1e-12


def test_continuous_distances_are_symmetric_nonnegative_across_random_cases() -> None:
    for seed in range(20):
        generator = np.random.default_rng(seed)
        left = generator.normal(size=200)
        right = generator.normal(loc=seed / 20, size=170)
        assert np.isclose(ks_statistic(left, right), ks_statistic(right, left))
        assert np.isclose(wasserstein_distance(left, right), wasserstein_distance(right, left))
        assert ks_statistic(left, right) >= 0
        assert wasserstein_distance(left, right) >= 0


def test_column_shuffle_preserves_each_marginal_exactly(roles) -> None:
    frame = roles[0]
    generator = np.random.default_rng(91)
    shuffled = frame.copy()
    for column in frame.columns:
        shuffled[column] = generator.permutation(frame[column].to_numpy())
    for column in ("duration_seconds", "route_distance_m", "time_of_day", "origin_camera"):
        assert sorted(frame[column].astype(str)) == sorted(shuffled[column].astype(str))
