"""Machine-readable CSV bundle writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..api import EvaluationReport


def write_csv_bundle(report: "EvaluationReport", directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([item.to_dict() for item in report.metrics]).to_csv(
        directory / "metrics.csv", index=False
    )
    pd.DataFrame([item.to_dict() for item in report.validation]).to_csv(
        directory / "validation.csv", index=False
    )
    with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(report.metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
