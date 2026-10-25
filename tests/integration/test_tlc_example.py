from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_tlc_adapter_runs_without_releasing_exact_timestamps(tmp_path: Path) -> None:
    size = 600
    rng = np.random.default_rng(20260729)
    pickup = pd.date_range("2024-01-01", periods=size, freq="5min")
    duration = rng.integers(300, 1800, size=size)
    source = pd.DataFrame(
        {
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(duration, unit="s"),
            "trip_distance": rng.uniform(1.0, 12.0, size=size),
            "PULocationID": rng.integers(1, 21, size=size),
            "DOLocationID": rng.integers(21, 41, size=size),
            "payment_type": rng.integers(1, 4, size=size),
        }
    )
    input_path = tmp_path / "tlc_structural_fixture.parquet"
    output = tmp_path / "output"
    source.to_parquet(input_path, index=False)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_tlc_benchmark.py"),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--maximum-rows",
            "400",
            "--seed",
            "20260729",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows_after_filter_and_cap"] == 400
    assert manifest["train_rows"] == 280
    assert manifest["holdout_rows"] == 120
    prohibited = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "vehicle_id",
        "sequence_key",
    }
    for path in (output / "projected_roles").glob("*.parquet"):
        assert prohibited.isdisjoint(pd.read_parquet(path).columns)
    assert (output / "report/results.json").is_file()
