"""Schema configuration and feature-role validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class SchemaError(ValueError):
    """Raised for an invalid or incompatible SynthEval schema."""


def _keys_or_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise SchemaError("Feature declarations must be lists or mappings")


@dataclass(frozen=True)
class SchemaConfig:
    """Immutable feature and governance schema."""

    schema_version: str
    name: str
    continuous_features: tuple[str, ...]
    count_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    ordinal_features: tuple[str, ...]
    binary_features: tuple[str, ...]
    temporal_features: Mapping[str, str]
    cyclical_temporal_features: Mapping[str, Any]
    transport_fields: Mapping[str, Any]
    target_variables: tuple[str, ...]
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    prohibited_identifiers: Mapping[str, Any]
    grouping_variables: tuple[str, ...]
    overlap_audit_columns: tuple[str, ...]
    rare_categories: Mapping[str, Any]
    valid_physical_ranges: Mapping[str, Any]
    physical_consistency: Mapping[str, Any]
    leakage_rules: Mapping[str, Any]
    privacy_distance_features: Mapping[str, Any]
    rare_combination_features: tuple[str, ...]
    raw: Mapping[str, Any] = field(repr=False)
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SchemaConfig":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise SchemaError(f"Schema does not exist: {source}")
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise SchemaError("Schema root must be a mapping")
        return cls.from_mapping(payload, source_path=source)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], source_path: Path | None = None
    ) -> "SchemaConfig":
        feature_groups = {
            "continuous": _keys_or_values(payload.get("continuous_features", [])),
            "count": _keys_or_values(payload.get("count_features", [])),
            "categorical": _keys_or_values(payload.get("categorical_features", [])),
            "ordinal": _keys_or_values(payload.get("ordinal_features", [])),
            "binary": _keys_or_values(payload.get("binary_features", [])),
        }
        owners: dict[str, str] = {}
        for group, columns in feature_groups.items():
            for column in columns:
                if column in owners:
                    raise SchemaError(
                        f"Column {column!r} is declared as both {owners[column]} and {group}"
                    )
                owners[column] = group
        required = tuple(str(item) for item in payload.get("required_columns", []))
        undeclared = [item for item in required if item not in owners]
        if undeclared:
            raise SchemaError(f"Required columns lack feature types: {undeclared}")
        targets = tuple(str(item) for item in payload.get("target_variables", []))
        for target in targets:
            if target not in owners:
                raise SchemaError(f"Target {target!r} lacks a declared feature type")
        return cls(
            schema_version=str(payload.get("schema_version", "1.0.0")),
            name=str(payload.get("name", "unnamed_schema")),
            continuous_features=tuple(feature_groups["continuous"]),
            count_features=tuple(feature_groups["count"]),
            categorical_features=tuple(feature_groups["categorical"]),
            ordinal_features=tuple(feature_groups["ordinal"]),
            binary_features=tuple(feature_groups["binary"]),
            temporal_features=dict(payload.get("temporal_features", {})),
            cyclical_temporal_features=dict(payload.get("cyclical_temporal_features", {})),
            transport_fields=dict(payload.get("transport_fields", {})),
            target_variables=targets,
            required_columns=required,
            optional_columns=tuple(str(item) for item in payload.get("optional_columns", [])),
            prohibited_identifiers=dict(payload.get("prohibited_identifiers", {})),
            grouping_variables=tuple(str(item) for item in payload.get("grouping_variables", [])),
            overlap_audit_columns=tuple(
                str(item) for item in payload.get("overlap_audit_columns", [])
            ),
            rare_categories=dict(payload.get("rare_categories", {})),
            valid_physical_ranges=dict(payload.get("valid_physical_ranges", {})),
            physical_consistency=dict(payload.get("physical_consistency", {})),
            leakage_rules=dict(payload.get("leakage_rules", {})),
            privacy_distance_features=dict(payload.get("privacy_distance_features", {})),
            rare_combination_features=tuple(
                str(item) for item in payload.get("rare_combination_features", [])
            ),
            raw=dict(payload),
            source_path=source_path,
        )

    @classmethod
    def coerce(cls, value: "SchemaConfig | Mapping[str, Any] | str | Path") -> "SchemaConfig":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        return cls.from_yaml(value)

    @property
    def all_features(self) -> tuple[str, ...]:
        ordered = (
            self.continuous_features
            + self.count_features
            + self.categorical_features
            + self.ordinal_features
            + self.binary_features
        )
        return tuple(dict.fromkeys(ordered))

    @property
    def numeric_features(self) -> tuple[str, ...]:
        return self.continuous_features + self.count_features + self.ordinal_features

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)
