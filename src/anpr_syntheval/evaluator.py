"""Orchestration engine for the SynthEval reference implementation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .api import EvaluationReport
from .config import EvaluationConfig
from .datasets import DataSource, LoadedDataset, inspect_columns, load_dataset, sha256_file
from .metrics import (
    evaluate_categorical,
    evaluate_continuous,
    evaluate_dependency,
    evaluate_privacy,
    evaluate_transport,
    evaluate_utility,
)
from .registry import MetricRegistry
from .schema import SchemaConfig
from .validation import (
    ValidationIssue,
    audit_train_holdout_overlap,
    prohibited_columns,
    raise_for_errors,
    validate_frame,
)


FRAMEWORK_VERSION = "0.2.1"


@dataclass
class _Prepared:
    train: LoadedDataset
    holdout: LoadedDataset
    synthetic: LoadedDataset
    issues: list[ValidationIssue]


def _mapping_digest(mapping: Mapping[str, Any]) -> str:
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class SynthEvaluator:
    """Evaluate synthetic data against separate real training and holdout roles.

    Parameters follow the public thesis API. Data sources may be pandas data
    frames, CSV/TSV files or Parquet files. Path-based inputs are inspected for
    prohibited identifiers before column projection.
    """

    def __init__(
        self,
        real_train: DataSource,
        real_holdout: DataSource,
        synthetic: DataSource,
        schema: SchemaConfig | Mapping[str, Any] | str | Path,
        config: EvaluationConfig | Mapping[str, Any] | str | Path,
    ) -> None:
        self.real_train_source = real_train
        self.real_holdout_source = real_holdout
        self.synthetic_source = synthetic
        self.schema = SchemaConfig.coerce(schema)
        self.config = EvaluationConfig.coerce(config)
        self._prepared: _Prepared | None = None
        self.registry = MetricRegistry()
        self.registry.register(
            "continuous_fidelity", evaluate_continuous,
            "Held-out univariate location, scale, quantile, range and distribution distances",
        )
        self.registry.register(
            "categorical_fidelity", evaluate_categorical,
            "Held-out category frequency, support, rarity and rank preservation",
        )
        self.registry.register(
            "dependency_fidelity", evaluate_dependency,
            "Numerical, mixed-type, mutual-information and propensity diagnostics",
        )
        self.registry.register(
            "utility", evaluate_utility,
            "TRTR baseline, TSTR transfer and optional TRTS diagnostic",
        )
        self.registry.register(
            "empirical_privacy_risk", evaluate_privacy,
            "Matching, mixed-type proximity, membership, attribute and rare-combination diagnostics",
        )
        self.registry.register(
            "transport_alignment", evaluate_transport,
            "Temporal, OD-flow and distance-duration-speed alignment",
        )

    def _source_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for role, source in (
            ("real_train", self.real_train_source),
            ("real_holdout", self.real_holdout_source),
            ("synthetic", self.synthetic_source),
        ):
            if not isinstance(source, pd.DataFrame):
                paths[role] = Path(source).expanduser().resolve()
        return paths

    def _preflight_issues(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        roles = {
            "real_train": self.real_train_source,
            "real_holdout": self.real_holdout_source,
            "synthetic": self.synthetic_source,
        }
        for role, source in roles.items():
            try:
                columns = inspect_columns(source)
            except Exception as error:
                issues.append(ValidationIssue(
                    role, "input_inspection_failed", "error", f"Input could not be inspected: {type(error).__name__}"
                ))
                continue
            for column in prohibited_columns(columns, self.schema):
                issues.append(ValidationIssue(
                    role, "prohibited_identifier_present", "error",
                    "A prohibited identifier or exact timestamp column is present", column
                ))
            for column in self.schema.required_columns:
                if column not in columns:
                    issues.append(ValidationIssue(
                        role, "required_column_missing", "error", "A required column is absent", column
                    ))
        if isinstance(self.real_train_source, pd.DataFrame) and self.real_train_source is self.real_holdout_source:
            issues.append(ValidationIssue(
                "real_train+real_holdout", "same_input_object", "error",
                "Training and holdout roles reference the same in-memory object"
            ))
        paths = self._source_paths()
        if (
            "real_train" in paths
            and "real_holdout" in paths
            and paths["real_train"] == paths["real_holdout"]
        ):
            issues.append(ValidationIssue(
                "real_train+real_holdout", "same_input_path", "error",
                "Training and holdout roles reference the same file"
            ))
        if (
            "real_train" in paths
            and "synthetic" in paths
            and paths["real_train"] == paths["synthetic"]
            and self.config.candidate_provenance != "controlled_validation"
        ):
            issues.append(ValidationIssue(
                "synthetic", "synthetic_equals_training_input", "error",
                "Synthetic and training roles reference the same file"
            ))
        issues.append(ValidationIssue(
            "synthetic", "candidate_provenance_declared", "info",
            f"Candidate provenance: {self.config.candidate_provenance}"
        ))
        return issues

    def _prepare(self) -> _Prepared:
        if self._prepared is not None:
            return self._prepared
        issues = self._preflight_issues()
        raise_for_errors(issues)
        maximum = int(self.config.sampling.get("maximum_rows_per_role", 50000))
        batch_size = int(self.config.sampling.get("batch_size", 65536))
        seed = self.config.random_seed
        train = load_dataset(
            self.real_train_source, "real_train", self.schema, maximum, seed + 1, batch_size
        )
        holdout = load_dataset(
            self.real_holdout_source, "real_holdout", self.schema, maximum, seed + 2, batch_size
        )
        synthetic = load_dataset(
            self.synthetic_source, "synthetic", self.schema, maximum, seed + 3, batch_size
        )
        if (
            self.config.candidate_provenance != "controlled_validation"
            and train.audit.checksum_sha256
            and train.audit.checksum_sha256 == synthetic.audit.checksum_sha256
        ):
            issues.append(ValidationIssue(
                "synthetic", "synthetic_equals_training_checksum", "error",
                "Synthetic and training files have identical checksums"
            ))
        for loaded in (train, holdout, synthetic):
            issues.extend(validate_frame(
                loaded.frame, loaded.audit.role, self.schema, self.config.validation_mode
            ))
        issues.extend(audit_train_holdout_overlap(train.frame, holdout.frame, self.schema))
        memory_bytes = sum(
            int(loaded.frame.memory_usage(index=True, deep=True).sum())
            for loaded in (train, holdout, synthetic)
        )
        limit_bytes = int(self.config.sampling.get("memory_limit_mb", 2048)) * 1024 * 1024
        if memory_bytes > limit_bytes:
            issues.append(ValidationIssue(
                "all_roles", "configured_memory_limit_exceeded", "error",
                "Projected evaluation frames exceed the configured memory limit",
                affected_count=memory_bytes,
            ))
        else:
            issues.append(ValidationIssue(
                "all_roles", "configured_memory_limit_passed", "info",
                "Projected evaluation frames remain within the configured memory limit",
                affected_count=memory_bytes,
            ))
        raise_for_errors(issues)
        self._prepared = _Prepared(train, holdout, synthetic, issues)
        return self._prepared

    def _base_metadata(self, prepared: _Prepared, runtime_seconds: float) -> dict[str, Any]:
        schema_checksum = (
            sha256_file(self.schema.source_path) if self.schema.source_path else _mapping_digest(self.schema.to_dict())
        )
        config_checksum = (
            sha256_file(self.config.source_path) if self.config.source_path else _mapping_digest(self.config.to_dict())
        )
        try:
            import psutil

            peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            peak_memory_mb = None
        return {
            "framework": "SynthEval",
            "framework_version": FRAMEWORK_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_name": self.schema.name,
            "schema_version": self.schema.schema_version,
            "schema_checksum_sha256": schema_checksum,
            "config_version": self.config.config_version,
            "config_checksum_sha256": config_checksum,
            "random_seed": self.config.random_seed,
            "candidate_provenance": self.config.candidate_provenance,
            "validation_mode": self.config.validation_mode,
            "runtime_seconds": runtime_seconds,
            "peak_resident_memory_mb": peak_memory_mb,
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python": sys.version.split()[0],
            },
            "dependencies": {
                "numpy": _version("numpy"),
                "pandas": _version("pandas"),
                "pyarrow": _version("pyarrow"),
                "PyYAML": _version("PyYAML"),
            },
            "datasets": {
                "real_train": prepared.train.audit.to_dict(),
                "real_holdout": prepared.holdout.audit.to_dict(),
                "synthetic": prepared.synthetic.audit.to_dict(),
            },
            "metric_registry": self.registry.describe(),
            "uncertainty": {
                "single_evaluation": "metric confidence intervals are not inferred from one deterministic run",
                "configured": dict(self.config.uncertainty),
                "model_run_uncertainty": "requires independently trained model outputs",
            },
            "reporting_policy": dict(self.config.reporting),
            "interpretation_boundaries": [
                "No single overall quality score is produced",
                "Propensity AUC near 0.5 does not prove equivalence",
                "Empirical privacy diagnostics do not prove anonymity or formal privacy",
                "TRTS is diagnostic and does not replace held-out-real TSTR evidence",
            ],
        }

    def validate(self) -> EvaluationReport:
        started = time.perf_counter()
        prepared = self._prepare()
        metadata = self._base_metadata(prepared, time.perf_counter() - started)
        metadata["evaluation_status"] = "validation_passed"
        metadata["metric_count"] = 0
        return EvaluationReport(metadata, list(prepared.issues), [])

    def evaluate(self) -> EvaluationReport:
        started = time.perf_counter()
        prepared = self._prepare()
        train = prepared.train
        holdout = prepared.holdout
        synthetic = prepared.synthetic
        continuous_features = self.schema.continuous_features + self.schema.count_features
        categorical_features = (
            self.schema.categorical_features + self.schema.ordinal_features + self.schema.binary_features
        )
        metrics = []
        for candidate, comparison in (
            (synthetic, "synthetic_vs_real_holdout"),
            (train, "real_train_vs_real_holdout_baseline"),
        ):
            metrics.extend(evaluate_continuous(
                holdout.frame, candidate.frame, continuous_features, dict(self.config.continuous),
                holdout.audit, candidate.audit, comparison
            ))
            metrics.extend(evaluate_categorical(
                holdout.frame, candidate.frame, categorical_features,
                dict(self.config.categorical), dict(self.schema.rare_categories),
                holdout.audit, candidate.audit, comparison
            ))
            dependency_settings = dict(self.config.dependency)
            dependency_settings.setdefault(
                "propensity_maximum_rows_per_class",
                int(self.config.sampling.get("propensity_maximum_rows_per_class", 10000)),
            )
            metrics.extend(evaluate_dependency(
                holdout.frame,
                candidate.frame,
                self.schema.numeric_features,
                categorical_features,
                dependency_settings,
                holdout.audit,
                candidate.audit,
                comparison,
                self.config.random_seed,
                int(self.config.sampling.get("dependency_maximum_rows", 12000)),
            ))
            metrics.extend(evaluate_transport(
                holdout.frame, candidate.frame, dict(self.config.transport), self.schema,
                holdout.audit, candidate.audit, comparison
            ))
        metrics.extend(evaluate_utility(
            train.frame,
            holdout.frame,
            synthetic.frame,
            dict(self.config.utility),
            self.schema,
            train.audit,
            holdout.audit,
            synthetic.audit,
            self.config.random_seed,
            int(self.config.sampling.get("utility_maximum_rows_per_role", 20000)),
        ))
        metrics.extend(evaluate_privacy(
            train.frame,
            holdout.frame,
            synthetic.frame,
            dict(self.config.privacy),
            self.schema,
            train.audit,
            holdout.audit,
            synthetic.audit,
            self.config.random_seed,
            int(self.config.sampling.get("privacy_maximum_rows_per_role", 2000)),
        ))
        metadata = self._base_metadata(prepared, time.perf_counter() - started)
        metadata["evaluation_status"] = "completed"
        metadata["metric_count"] = len(metrics)
        metadata["applicable_metric_count"] = sum(item.applicable for item in metrics)
        report = EvaluationReport(metadata, list(prepared.issues), metrics)
        metadata["scientific_content_sha256"] = report.canonical_digest()
        return report
