from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "configs" / "anpr_schema.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default_evaluation.yaml"


def make_frame(size: int = 320, seed: int = 7, shift: float = 0.0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    origin_index = generator.integers(0, 5, size=size)
    step = generator.integers(1, 5, size=size)
    destination_index = (origin_index + step) % 5
    hour = generator.integers(0, 24, size=size)
    day = generator.integers(0, 7, size=size)
    pair_effect = (origin_index * 5 + destination_index) * 280.0
    distance = 1800.0 + pair_effect + generator.gamma(2.5, 700.0, size=size) + shift * 900.0
    speed = 32.0 + 7.0 * np.sin(2 * np.pi * hour / 24.0) + generator.normal(0, 3.0, size)
    speed = np.clip(speed + shift * 4.0, 12.0, 80.0)
    duration = 3.6 * distance / speed
    period = np.select(
        [hour < 6, hour < 10, hour < 16, hour < 20],
        ["night", "morning_peak", "inter_peak", "evening_peak"],
        default="late_evening",
    )
    road_class = np.select(
        [distance < 4500, distance < 7500], ["residential", "primary"], default="motorway"
    )
    return pd.DataFrame({
        "duration_seconds": duration,
        "route_distance_m": distance,
        "derived_average_speed_kph": speed,
        "origin_confidence": np.clip(generator.normal(93, 3, size), 70, 100),
        "destination_confidence": np.clip(generator.normal(92, 4, size), 70, 100),
        "route_average_kph": np.clip(speed * generator.normal(1.08, 0.04, size), 1, 120),
        "direction_compatibility": generator.uniform(0.3, 1.0, size),
        "origin_camera": [f"CAM_{item}" for item in origin_index],
        "destination_camera": [f"CAM_{item}" for item in destination_index],
        "time_of_day": period,
        "predominant_road_class": road_class,
        "day_of_week": day,
        "hour_of_day": hour,
        "cross_week": generator.random(size) < 0.01,
    })


def make_roles(size: int = 320) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_frame(size, 10), make_frame(size, 20), make_frame(size, 30)


def evaluation_mapping(mode: str = "strict", provenance: str = "controlled_validation") -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    payload = deepcopy(payload)
    payload["validation_mode"] = mode
    payload["candidate_provenance"] = provenance
    payload["sampling"].update({
        "maximum_rows_per_role": 400,
        "dependency_maximum_rows": 240,
        "propensity_maximum_rows_per_class": 160,
        "utility_maximum_rows_per_role": 240,
        "privacy_maximum_rows_per_role": 100,
        "memory_limit_mb": 256,
    })
    payload["utility"]["tasks"][0]["minimum_class_count"] = 3
    payload["uncertainty"]["bootstrap_repetitions"] = 0
    return payload


@pytest.fixture
def roles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_roles()


@pytest.fixture
def schema_path() -> Path:
    return SCHEMA_PATH


@pytest.fixture
def config_mapping() -> dict:
    return evaluation_mapping()
