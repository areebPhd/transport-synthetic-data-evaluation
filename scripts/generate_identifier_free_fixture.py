#!/usr/bin/env python3
"""Generate fictional, identifier-free transport roles for documentation/tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_frame(size: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin = rng.integers(0, 6, size=size)
    destination = (origin + rng.integers(1, 6, size=size)) % 6
    hour = rng.integers(0, 24, size=size)
    day = rng.integers(0, 7, size=size)
    distance = (
        1600.0
        + (origin * 6 + destination) * 220.0
        + rng.gamma(2.3, 650.0, size=size)
        + shift * 400.0
    )
    speed = np.clip(
        31.0
        + 6.0 * np.sin(2.0 * np.pi * hour / 24.0)
        + rng.normal(0.0, 3.0, size=size)
        + shift,
        10.0,
        90.0,
    )
    duration = 3.6 * distance / speed
    period = np.select(
        [hour < 6, hour < 10, hour < 16, hour < 20],
        ["night", "morning_peak", "inter_peak", "evening_peak"],
        default="late_evening",
    )
    road_class = np.select(
        [distance < 4500.0, distance < 7500.0],
        ["residential", "primary"],
        default="motorway",
    )
    return pd.DataFrame(
        {
            "duration_seconds": duration,
            "route_distance_m": distance,
            "derived_average_speed_kph": speed,
            "origin_confidence": np.clip(rng.normal(93.0, 3.0, size), 70.0, 100.0),
            "destination_confidence": np.clip(
                rng.normal(92.0, 4.0, size), 70.0, 100.0
            ),
            "route_average_kph": np.clip(
                speed * rng.normal(1.08, 0.04, size), 1.0, 120.0
            ),
            "direction_compatibility": rng.uniform(0.3, 1.0, size),
            "origin_camera": [f"FICTIONAL_NODE_{item}" for item in origin],
            "destination_camera": [
                f"FICTIONAL_NODE_{item}" for item in destination
            ],
            "time_of_day": period,
            "predominant_road_class": road_class,
            "day_of_week": day,
            "hour_of_day": hour,
            "cross_week": rng.random(size) < 0.01,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=320)
    arguments = parser.parse_args()
    if arguments.rows < 100:
        raise ValueError("--rows must be at least 100")
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    roles = {
        "real_train.csv": make_frame(arguments.rows, 1101),
        "real_holdout.csv": make_frame(arguments.rows, 2202, 0.15),
        "synthetic.csv": make_frame(arguments.rows, 3303, 0.05),
    }
    for filename, frame in roles.items():
        path = output / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite fixture role: {path}")
        frame.to_csv(path, index=False, lineterminator="\n")
        print(f"{path}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
