"""Privacy-aware input, overlap, range and output-path validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .schema import SchemaConfig


@dataclass(frozen=True)
class ValidationIssue:
    role: str
    code: str
    severity: str
    message: str
    column: str | None = None
    affected_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DataValidationError(ValueError):
    """Raised when strict validation finds one or more unsafe conditions."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = list(issues)
        summary = "; ".join(f"{item.role}:{item.code}" for item in self.issues[:8])
        if len(self.issues) > 8:
            summary += f"; and {len(self.issues) - 8} more"
        super().__init__(f"SynthEval input validation failed: {summary}")


def prohibited_columns(columns: Iterable[str], schema: SchemaConfig) -> list[str]:
    declaration = schema.prohibited_identifiers
    exact = {str(item).lower() for item in declaration.get("exact_names", [])}
    patterns = [re.compile(str(item), re.IGNORECASE) for item in declaration.get(
        "case_insensitive_patterns", []
    )]
    found: list[str] = []
    for column in columns:
        name = str(column)
        if name.lower() in exact or any(pattern.search(name) for pattern in patterns):
            found.append(name)
    return sorted(set(found))


def _range_failure_mask(series: pd.Series, rule: dict[str, object]) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(len(values), dtype=bool)
    minimum = rule.get("minimum")
    maximum = rule.get("maximum")
    if minimum is not None:
        if bool(rule.get("minimum_inclusive", True)):
            mask |= values < float(minimum)
        else:
            mask |= values <= float(minimum)
    if maximum is not None:
        if bool(rule.get("maximum_inclusive", True)):
            mask |= values > float(maximum)
        else:
            mask |= values >= float(maximum)
    return mask


def validate_frame(
    frame: pd.DataFrame,
    role: str,
    schema: SchemaConfig,
    mode: str = "strict",
) -> list[ValidationIssue]:
    """Validate one role without including row values in diagnostics."""

    issues: list[ValidationIssue] = []
    columns = [str(item) for item in frame.columns]
    missing = [item for item in schema.required_columns if item not in frame.columns]
    for column in missing:
        issues.append(ValidationIssue(
            role, "required_column_missing", "error", "A required column is absent", column
        ))
    for column in prohibited_columns(columns, schema):
        issues.append(ValidationIssue(
            role,
            "prohibited_identifier_present",
            "error",
            "A prohibited identifier or exact timestamp column is present",
            column,
        ))

    for column in schema.numeric_features:
        if column not in frame.columns:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        incompatible = int(converted.isna().sum() - frame[column].isna().sum())
        if incompatible > 0:
            issues.append(ValidationIssue(
                role,
                "incompatible_numeric_type",
                "error",
                "Values could not be converted to the declared numerical type",
                column,
                incompatible,
            ))
        finite = np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan))
        nonfinite = int((~finite).sum())
        if nonfinite > 0:
            issues.append(ValidationIssue(
                role,
                "non_finite_or_missing_numeric_value",
                "error",
                "The declared numerical feature contains missing or non-finite values",
                column,
                nonfinite,
            ))
        rule = schema.valid_physical_ranges.get(column)
        if rule and finite.any():
            failure = _range_failure_mask(frame[column], dict(rule)) & finite
            count = int(failure.sum())
            if count:
                issues.append(ValidationIssue(
                    role,
                    "physical_range_violation",
                    "error" if mode == "strict" else "warning",
                    "Values fall outside the declared physical range",
                    column,
                    count,
                ))

    ordinal_declarations = schema.raw.get("ordinal_features", {})
    if isinstance(ordinal_declarations, dict):
        for column, declaration in ordinal_declarations.items():
            if column not in frame.columns:
                continue
            allowed = set(declaration.get("ordered_values", []))
            invalid_count = int((~frame[column].isin(allowed)).sum())
            if invalid_count:
                issues.append(ValidationIssue(
                    role,
                    "invalid_ordinal_value",
                    "error" if mode == "strict" else "warning",
                    "Values are outside the declared ordinal levels",
                    column,
                    invalid_count,
                ))
    binary_declarations = schema.raw.get("binary_features", {})
    if isinstance(binary_declarations, dict):
        for column, declaration in binary_declarations.items():
            if column not in frame.columns:
                continue
            allowed = set(declaration.get("allowed_values", [False, True]))
            invalid_count = int((~frame[column].isin(allowed)).sum())
            if invalid_count:
                issues.append(ValidationIssue(
                    role,
                    "invalid_binary_value",
                    "error" if mode == "strict" else "warning",
                    "Values are outside the declared binary levels",
                    column,
                    invalid_count,
                ))

    origin = schema.transport_fields.get("origin_camera")
    destination = schema.transport_fields.get("destination_camera")
    if origin in frame.columns and destination in frame.columns:
        same = int((frame[str(origin)].astype(str) == frame[str(destination)].astype(str)).sum())
        if same:
            issues.append(ValidationIssue(
                role,
                "same_origin_destination",
                "error" if mode == "strict" else "warning",
                "Origin and destination are identical for inter-camera records",
                affected_count=same,
            ))

    if not issues:
        issues.append(ValidationIssue(
            role, "schema_validation_passed", "info", "All declared input checks passed"
        ))
    return issues


def audit_train_holdout_overlap(
    train: pd.DataFrame, holdout: pd.DataFrame, schema: SchemaConfig
) -> list[ValidationIssue]:
    columns = list(schema.overlap_audit_columns)
    if not columns:
        return [ValidationIssue(
            "real_train+real_holdout",
            "overlap_key_not_available",
            "info",
            "No governance-approved overlap key was supplied; external split audit is required",
        )]
    missing = [item for item in columns if item not in train.columns or item not in holdout.columns]
    if missing:
        return [ValidationIssue(
            "real_train+real_holdout",
            "overlap_key_missing",
            "error",
            "A configured overlap audit column is absent",
            ",".join(missing),
        )]
    train_hash = set(pd.util.hash_pandas_object(train[columns], index=False).astype("uint64"))
    holdout_hash = set(pd.util.hash_pandas_object(holdout[columns], index=False).astype("uint64"))
    overlap = len(train_hash.intersection(holdout_hash))
    return [ValidationIssue(
        "real_train+real_holdout",
        "overlap_detected" if overlap else "overlap_audit_passed",
        "error" if overlap else "info",
        "The configured split-key audit detected overlap" if overlap else "No overlap was detected",
        affected_count=overlap,
    )]


def raise_for_errors(issues: Sequence[ValidationIssue]) -> None:
    errors = [item for item in issues if item.severity == "error"]
    if errors:
        raise DataValidationError(errors)


def validate_output_path(
    path: str | Path,
    forbidden_components: Iterable[str],
    input_paths: Iterable[str | Path] = (),
) -> Path:
    """Reject paths that could place reports inside restricted input locations."""

    target = Path(path).expanduser().resolve()
    forbidden = {str(item).lower() for item in forbidden_components}
    components = {item.lower() for item in target.parts}
    matched = sorted(components.intersection(forbidden))
    if matched:
        raise ValueError(f"Unsafe output path contains a forbidden component: {matched[0]}")
    for source in input_paths:
        source_path = Path(source).expanduser().resolve()
        source_root = source_path if source_path.is_dir() else source_path.parent
        try:
            target.relative_to(source_root)
        except ValueError:
            continue
        if any(item.lower() in forbidden for item in source_root.parts):
            raise ValueError("Output path resolves inside a restricted input directory")
    return target
