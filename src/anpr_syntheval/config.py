"""Typed loading and validation of evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration is missing or internally inconsistent."""


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable evaluation settings.

    Unknown nested keys are retained in ``raw`` so that manifests preserve the
    exact declared configuration while typed accessors cover execution-critical
    values.
    """

    config_version: str = "1.0.0"
    random_seed: int = 20260718
    candidate_provenance: str = "generated_from_authorised_train"
    validation_mode: str = "strict"
    sampling: Mapping[str, Any] = field(default_factory=dict)
    continuous: Mapping[str, Any] = field(default_factory=dict)
    categorical: Mapping[str, Any] = field(default_factory=dict)
    dependency: Mapping[str, Any] = field(default_factory=dict)
    utility: Mapping[str, Any] = field(default_factory=dict)
    privacy: Mapping[str, Any] = field(default_factory=dict)
    transport: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    reporting: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ConfigurationError(f"Evaluation configuration does not exist: {source}")
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ConfigurationError("Evaluation configuration root must be a mapping")
        return cls.from_mapping(payload, source_path=source)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], source_path: Path | None = None
    ) -> "EvaluationConfig":
        mode = str(payload.get("validation_mode", "strict"))
        if mode not in {"strict", "audit"}:
            raise ConfigurationError("validation_mode must be 'strict' or 'audit'")
        provenance = str(payload.get("candidate_provenance", "generated_from_authorised_train"))
        if provenance not in {
            "generated_from_authorised_train",
            "controlled_validation",
        }:
            raise ConfigurationError("candidate_provenance is not recognised")
        seed = int(payload.get("random_seed", 20260718))
        sampling = dict(payload.get("sampling", {}))
        maximum = int(sampling.get("maximum_rows_per_role", 50000))
        if maximum < 2:
            raise ConfigurationError("sampling.maximum_rows_per_role must be at least two")
        memory = int(sampling.get("memory_limit_mb", 2048))
        if memory < 64:
            raise ConfigurationError("sampling.memory_limit_mb must be at least 64")
        return cls(
            config_version=str(payload.get("config_version", "1.0.0")),
            random_seed=seed,
            candidate_provenance=provenance,
            validation_mode=mode,
            sampling=sampling,
            continuous=dict(payload.get("continuous", {})),
            categorical=dict(payload.get("categorical", {})),
            dependency=dict(payload.get("dependency", {})),
            utility=dict(payload.get("utility", {})),
            privacy=dict(payload.get("privacy", {})),
            transport=dict(payload.get("transport", {})),
            uncertainty=dict(payload.get("uncertainty", {})),
            reporting=dict(payload.get("reporting", {})),
            raw=dict(payload),
            source_path=source_path,
        )

    @classmethod
    def coerce(cls, value: "EvaluationConfig | Mapping[str, Any] | str | Path") -> "EvaluationConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        return cls.from_yaml(value)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        mapping = getattr(self, section)
        return mapping.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw) if self.raw else {
            "config_version": self.config_version,
            "random_seed": self.random_seed,
            "candidate_provenance": self.candidate_provenance,
            "validation_mode": self.validation_mode,
            "sampling": dict(self.sampling),
            "continuous": dict(self.continuous),
            "categorical": dict(self.categorical),
            "dependency": dict(self.dependency),
            "utility": dict(self.utility),
            "privacy": dict(self.privacy),
            "transport": dict(self.transport),
            "uncertainty": dict(self.uncertainty),
            "reporting": dict(self.reporting),
        }
