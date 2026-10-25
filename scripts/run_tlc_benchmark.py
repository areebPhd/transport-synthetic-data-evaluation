#!/usr/bin/env python3
"""Run a bounded SynthEval example on caller-supplied public NYC TLC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anpr_syntheval import SynthEvaluator


MILE_TO_METRE = 1609.344


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def period(hour: pd.Series) -> np.ndarray:
    return np.select(
        [hour < 6, hour < 10, hour < 16, hour < 20],
        ["night", "morning_peak", "inter_peak", "evening_peak"],
        default="late_evening",
    )


def distance_band(distance: pd.Series) -> np.ndarray:
    return np.select(
        [distance < 3218.688, distance < 8046.72],
        ["short", "medium"],
        default="long",
    )


def project(frame: pd.DataFrame, maximum_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    required = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TLC input lacks required fields: {missing}")
    pickup = pd.to_datetime(frame["tpep_pickup_datetime"], errors="coerce")
    dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"], errors="coerce")
    duration = (dropoff - pickup).dt.total_seconds()
    distance = pd.to_numeric(frame["trip_distance"], errors="coerce") * MILE_TO_METRE
    speed = 3.6 * distance / duration
    valid = (
        pickup.notna()
        & dropoff.notna()
        & duration.gt(0.0)
        & duration.le(10800.0)
        & distance.gt(0.0)
        & distance.le(160934.0)
        & speed.gt(0.0)
        & speed.le(160.0)
        & pd.to_numeric(frame["PULocationID"], errors="coerce").gt(0)
        & pd.to_numeric(frame["DOLocationID"], errors="coerce").gt(0)
    )
    selected = pd.DataFrame(
        {
            "duration_seconds": duration[valid],
            "route_distance_m": distance[valid],
            "derived_average_speed_kph": speed[valid],
            "origin_zone": "ZONE_" + frame.loc[valid, "PULocationID"].astype("Int64").astype(str),
            "destination_zone": "ZONE_"
            + frame.loc[valid, "DOLocationID"].astype("Int64").astype(str),
            "payment_type": "PAYMENT_"
            + frame.loc[valid, "payment_type"].astype("Int64").astype(str),
            "day_of_week": pickup[valid].dt.dayofweek.astype(int),
            "hour_of_day": pickup[valid].dt.hour.astype(int),
        }
    ).reset_index(drop=True)
    times = pickup[valid].reset_index(drop=True)
    if len(selected) > maximum_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(selected), maximum_rows, replace=False))
        selected = selected.iloc[indices].reset_index(drop=True)
        times = times.iloc[indices].reset_index(drop=True)
    selected["time_of_day"] = period(selected["hour_of_day"])
    selected["distance_band"] = distance_band(selected["route_distance_m"])
    return selected, times


def independent_reference(train: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    generated = pd.DataFrame(index=np.arange(rows))
    sampled = [
        "duration_seconds",
        "route_distance_m",
        "origin_zone",
        "destination_zone",
        "payment_type",
        "day_of_week",
        "hour_of_day",
    ]
    for column in sampled:
        values = train[column].to_numpy()
        generated[column] = values[rng.integers(0, len(values), size=rows)]
    generated["derived_average_speed_kph"] = (
        3.6 * generated["route_distance_m"] / generated["duration_seconds"]
    )
    valid = generated["derived_average_speed_kph"].le(160.0)
    while not bool(valid.all()):
        count = int((~valid).sum())
        generated.loc[~valid, "duration_seconds"] = train["duration_seconds"].to_numpy()[
            rng.integers(0, len(train), size=count)
        ]
        generated["derived_average_speed_kph"] = (
            3.6 * generated["route_distance_m"] / generated["duration_seconds"]
        )
        valid = generated["derived_average_speed_kph"].le(160.0)
    generated["time_of_day"] = period(generated["hour_of_day"])
    generated["distance_band"] = distance_band(generated["route_distance_m"])
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260728)
    arguments = parser.parse_args()
    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir() if output.is_dir() else [output]):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    projected, pickup = project(
        pd.read_parquet(source), arguments.maximum_rows, arguments.seed
    )
    order = np.argsort(pickup.to_numpy(), kind="stable")
    projected = projected.iloc[order].reset_index(drop=True)
    boundary = max(1, min(len(projected) - 1, int(0.7 * len(projected))))
    train = projected.iloc[:boundary].reset_index(drop=True)
    holdout = projected.iloc[boundary:].reset_index(drop=True)
    synthetic = independent_reference(train, len(holdout), arguments.seed + 1)
    roles = output / "projected_roles"
    roles.mkdir()
    paths = {}
    for name, frame in (
        ("real_train", train),
        ("real_holdout", holdout),
        ("synthetic", synthetic),
    ):
        path = roles / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
    project_root = Path(__file__).resolve().parents[1]
    report = SynthEvaluator(
        paths["real_train"],
        paths["real_holdout"],
        paths["synthetic"],
        project_root / "examples/tlc/tlc_schema.yaml",
        project_root / "examples/tlc/evaluation.yaml",
    ).evaluate()
    report_dir = output / "report"
    report.to_json(report_dir / "results.json")
    report.to_csv(report_dir / "csv")
    report.to_html(report_dir / "report.html")
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "rows_after_filter_and_cap": len(projected),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "synthetic_rows": len(synthetic),
        "split": "first 70% versus last 30% after pickup-time ordering",
        "synthetic_reference": "training-only independent marginals; speed and bands deterministically reconstructed",
        "seed": arguments.seed,
        "scientific_content_sha256": report.canonical_digest(),
        "role_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
