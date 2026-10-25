from __future__ import annotations

import json
from pathlib import Path

import yaml

from anpr_syntheval import SynthEvaluator
from anpr_syntheval.api import EvaluationReport
from anpr_syntheval.cli import main
from anpr_syntheval.registry import MetricResult
from anpr_syntheval.validation import ValidationIssue


def _write_inputs(tmp_path: Path, roles, config_mapping, schema_path):
    paths = []
    for name, frame in zip(("train", "holdout", "synthetic"), roles):
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(config_mapping, sort_keys=False), encoding="utf-8")
    return paths, config


def _report_from_json(path: Path) -> EvaluationReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EvaluationReport(
        payload["metadata"],
        [ValidationIssue(**item) for item in payload["validation"]],
        [MetricResult(**item) for item in payload["metrics"]],
    )


def test_cli_evaluate_matches_python_api(tmp_path, roles, schema_path, config_mapping) -> None:
    paths, config = _write_inputs(tmp_path, roles, config_mapping, schema_path)
    api_report = SynthEvaluator(paths[0], paths[1], paths[2], schema_path, config).evaluate()
    output = tmp_path / "cli_output"
    exit_code = main([
        "evaluate", "--real-train", str(paths[0]), "--real-holdout", str(paths[1]),
        "--synthetic", str(paths[2]), "--schema", str(schema_path), "--config", str(config),
        "--output", str(output),
    ])
    assert exit_code == 0
    cli_report = _report_from_json(output / "results.json")
    assert cli_report.canonical_digest() == api_report.canonical_digest()


def test_compare_reports_unavailable_models_without_fabrication(tmp_path) -> None:
    manifest = tmp_path / "models.yaml"
    manifest.write_text(yaml.safe_dump({
        "models": {
            "vae": {"status": "unavailable", "generated_dataset_path": None, "reason": "not found"},
            "ctgan": {"status": "unavailable", "generated_dataset_path": None, "reason": "not found"},
        }
    }), encoding="utf-8")
    output = tmp_path / "comparison"
    assert main(["compare", "--manifest", str(manifest), "--output", str(output)]) == 0
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert all(item["status"] == "unavailable" for item in payload["models"])
