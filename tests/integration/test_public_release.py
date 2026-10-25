from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from anpr_syntheval import SynthEvaluator, __version__


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_release_version_is_consistent() -> None:
    assert __version__ == "0.2.1"


def test_identifier_free_fixture_generator_and_evaluator(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_identifier_free_fixture.py"),
            "--output",
            str(fixture),
            "--rows",
            "160",
        ],
        check=True,
    )
    forbidden = {
        "registration",
        "vehicle_id",
        "sequence_key",
        "origin_timestamp",
        "destination_timestamp",
    }
    for path in fixture.glob("*.csv"):
        assert forbidden.isdisjoint(pd.read_csv(path, nrows=1).columns)
    config = yaml.safe_load(
        (PROJECT_ROOT / "examples/tiny_fixture/evaluation.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["sampling"]["maximum_rows_per_role"] = 160
    config["sampling"]["utility_maximum_rows_per_role"] = 120
    report = SynthEvaluator(
        fixture / "real_train.csv",
        fixture / "real_holdout.csv",
        fixture / "synthetic.csv",
        PROJECT_ROOT / "configs/anpr_schema.yaml",
        config,
    ).evaluate()
    assert report.metadata["metric_count"] > 0
    assert report.metadata["framework_version"] == "0.2.1"
