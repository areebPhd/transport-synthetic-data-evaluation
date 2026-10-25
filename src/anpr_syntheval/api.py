"""Evaluation report API and privacy-safe serialisation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .registry import MetricResult
from .reporting.csv_report import write_csv_bundle
from .reporting.html_report import write_html
from .reporting.json_report import write_json
from .validation import ValidationIssue, validate_output_path


@dataclass
class EvaluationReport:
    """Aggregate-only SynthEval result with JSON, CSV and HTML exporters."""

    metadata: dict[str, Any]
    validation: list[ValidationIssue]
    metrics: list[MetricResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "validation": [item.to_dict() for item in self.validation],
            "metrics": [item.to_dict() for item in self.metrics],
        }

    @property
    def applicable_metrics(self) -> list[MetricResult]:
        return [item for item in self.metrics if item.applicable]

    def metric(self, metric_id: str, comparison: str | None = None) -> list[MetricResult]:
        return [
            item for item in self.metrics
            if item.metric_id == metric_id and (comparison is None or item.comparison == comparison)
        ]

    def _safe_path(self, path: str | Path) -> Path:
        reporting = self.metadata.get("reporting_policy", {})
        forbidden = reporting.get("forbidden_output_path_components", [])
        inputs: Iterable[str] = [
            item.get("path") for item in self.metadata.get("datasets", {}).values()
            if item.get("path")
        ]
        return validate_output_path(path, forbidden, inputs)

    def to_json(self, path: str | Path) -> Path:
        target = self._safe_path(path)
        write_json(self.to_dict(), target)
        return target

    def to_csv(self, directory: str | Path) -> Path:
        target = self._safe_path(directory)
        write_csv_bundle(self, target)
        return target

    def to_html(self, path: str | Path) -> Path:
        target = self._safe_path(path)
        write_html(self, target)
        return target

    def canonical_payload(self) -> dict[str, Any]:
        """Return deterministic scientific content for API/CLI reconciliation."""

        metric_rows = [item.to_dict() for item in self.metrics]
        metric_rows.sort(key=lambda row: (
            str(row["family"]), str(row["comparison"]), str(row["metric_id"]), str(row["feature"])
        ))
        validation_rows = [item.to_dict() for item in self.validation]
        validation_rows.sort(key=lambda row: (
            str(row["role"]), str(row["code"]), str(row.get("column"))
        ))
        return {
            "framework_version": self.metadata.get("framework_version"),
            "schema_version": self.metadata.get("schema_version"),
            "config_version": self.metadata.get("config_version"),
            "random_seed": self.metadata.get("random_seed"),
            "candidate_provenance": self.metadata.get("candidate_provenance"),
            "validation": validation_rows,
            "metrics": metric_rows,
        }

    def canonical_digest(self) -> str:
        payload = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
