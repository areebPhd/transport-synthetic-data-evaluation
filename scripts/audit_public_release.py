#!/usr/bin/env python3
"""Audit and manifest the SynthEval public-package release boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


GENERATED_FILES = {
    "MANIFEST.json",
    "PUBLIC_RELEASE_AUDIT.md",
    "SHA256SUMS.txt",
}
LOCAL_ROOT_PREFIX = "/" + "home/user/"
EXCLUDED_PARTS = {
    ".pytest_cache",
    "__pycache__",
    "build",
}
EXCLUDED_SUFFIXES = {".egg-info", ".pyc", ".pyo"}
FORBIDDEN_DATA_SUFFIXES = {
    ".ckpt",
    ".db",
    ".feather",
    ".h5",
    ".joblib",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
}
PROHIBITED_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"number.?plate",
        r"registration",
        r"vehicle.?id",
        r"vehicle.?token",
        r"journey.?token",
        r"sequence.?key",
        r"(^|_)timestamp($|_)",
        r"latitude",
        r"longitude",
    )
]


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() in GENERATED_FILES:
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _check_fixture(root: Path) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    fixture_root = root / "examples/tiny_fixture"
    expected = {"real_train.csv", "real_holdout.csv", "synthetic.csv"}
    observed = {path.name for path in fixture_root.glob("*.csv")}
    checks.append(
        AuditCheck(
            "fixture_roles",
            observed == expected,
            f"expected={sorted(expected)}; observed={sorted(observed)}",
        )
    )
    for path in sorted(fixture_root.glob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
        prohibited = sorted(
            column
            for column in columns
            if any(pattern.search(column) for pattern in PROHIBITED_COLUMN_PATTERNS)
        )
        camera_columns = [
            column for column in ("origin_camera", "destination_camera") if column in columns
        ]
        nonfictional = sum(
            1
            for row in rows
            for column in camera_columns
            if not str(row[column]).startswith("FICTIONAL_NODE_")
        )
        checks.append(
            AuditCheck(
                f"fixture_{path.stem}",
                bool(rows) and not prohibited and nonfictional == 0,
                (
                    f"rows={len(rows)}; prohibited_columns={prohibited}; "
                    f"nonfictional_location_labels={nonfictional}"
                ),
            )
        )
    return checks


def _check_aggregate_report(root: Path) -> list[AuditCheck]:
    result_path = root / "reports/example_report/results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    top_level = set(payload)
    permitted = {"metadata", "metrics", "validation"}
    text = result_path.read_text(encoding="utf-8")
    return [
        AuditCheck(
            "aggregate_report_shape",
            top_level == permitted and isinstance(payload["metrics"], list),
            f"top_level_keys={sorted(top_level)}; metric_rows={len(payload.get('metrics', []))}",
        ),
        AuditCheck(
            "aggregate_report_portability",
            LOCAL_ROOT_PREFIX not in text and str(root) not in text,
            "no machine-specific workspace path is embedded",
        ),
    ]


def _check_release_files(root: Path, files: list[Path], version: str) -> list[AuditCheck]:
    forbidden = [
        path.relative_to(root).as_posix()
        for path in files
        if path.suffix.lower() in FORBIDDEN_DATA_SUFFIXES
    ]
    wheels = [path for path in files if path.suffix == ".whl"]
    wheel_names = [path.name for path in wheels]
    wheel_entries: list[str] = []
    wheel_forbidden: list[str] = []
    for wheel_path in wheels:
        with zipfile.ZipFile(wheel_path) as archive:
            names = archive.namelist()
            wheel_entries.extend(names)
            wheel_forbidden.extend(
                name
                for name in names
                if Path(name).suffix.lower() in FORBIDDEN_DATA_SUFFIXES
                or "/examples/" in f"/{name}"
                or "/reports/" in f"/{name}"
            )
    portable_text_files = [
        path
        for path in files
        if path.suffix.lower()
        in {".cff", ".csv", ".html", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    ]
    local_paths = [
        path.relative_to(root).as_posix()
        for path in portable_text_files
        if LOCAL_ROOT_PREFIX in path.read_text(encoding="utf-8", errors="replace")
    ]
    return [
        AuditCheck(
            "forbidden_bundled_data_types",
            not forbidden,
            f"matches={forbidden}",
        ),
        AuditCheck(
            "wheel_count_and_version",
            len(wheels) == 1
            and wheel_names[0].startswith(f"anpr_syntheval-{version}-"),
            f"wheels={wheel_names}",
        ),
        AuditCheck(
            "wheel_payload_boundary",
            bool(wheel_entries) and not wheel_forbidden,
            f"entries={len(wheel_entries)}; forbidden_entries={wheel_forbidden}",
        ),
        AuditCheck(
            "machine_specific_paths",
            not local_paths,
            f"matches={local_paths}",
        ),
    ]


def write_outputs(root: Path, version: str, checks: list[AuditCheck], files: list[Path]) -> None:
    report_lines = [
        "# SynthEval public-release audit",
        "",
        f"Package version: `{version}`",
        "",
        "Result: **PASS**" if all(check.passed for check in checks) else "Result: **FAIL**",
        "",
        "This audit verifies the package boundary, not the privacy of user-supplied data.",
        "The fixture is deterministically generated and identifier-free; the bundled report",
        "contains aggregate metric rows only. Mentions of prohibited field names in schemas",
        "and tests are guard declarations, not released values.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|").replace("\n", " ")
        report_lines.append(
            f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} | {detail} |"
        )
    report_lines.extend(
        [
            "",
            "## Release boundary",
            "",
            "No restricted ANPR records, pseudonymous sequence keys, exact sensitive",
            "locations, exact timestamps, trained weights, rare restricted cells, or",
            "ANPR-derived generated outputs are included. The NYC TLC adapter requires",
            "caller-supplied public data and does not bundle that dataset.",
            "",
            "Empirical privacy diagnostics do not establish anonymity or formal privacy.",
            "",
        ]
    )
    (root / "PUBLIC_RELEASE_AUDIT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    files = release_files(root)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "package": "anpr-syntheval",
        "version": version,
        "release_boundary_audit": "PASS" if all(check.passed for check in checks) else "FAIL",
        "generated_files_excluded_from_manifest_entries": sorted(GENERATED_FILES),
        "file_count": len(entries),
        "files": entries,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_files = release_files(root) + [
        root / "MANIFEST.json",
        root / "PUBLIC_RELEASE_AUDIT.md",
    ]
    checksum_files = sorted(
        set(checksum_files), key=lambda item: item.relative_to(root).as_posix()
    )
    checksum_text = "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in checksum_files
    )
    (root / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.expanduser().resolve()
    project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', project_text)
    if match is None:
        raise RuntimeError("Could not read the project version from pyproject.toml")
    version = match.group(1)
    files = release_files(root)
    checks = (
        _check_fixture(root)
        + _check_aggregate_report(root)
        + _check_release_files(root, files, version)
    )
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    passed = all(check.passed for check in checks)
    if arguments.write:
        write_outputs(root, version, checks, files)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
