from __future__ import annotations

import json

from anpr_syntheval import SynthEvaluator


def test_api_evaluates_and_writes_aggregate_reports(tmp_path, roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).evaluate()
    assert report.metadata["evaluation_status"] == "completed"
    assert len(report.metrics) >= 150
    json_path = report.to_json(tmp_path / "result.json")
    csv_path = report.to_csv(tmp_path / "csv")
    html_path = report.to_html(tmp_path / "report.html")
    assert json_path.is_file()
    assert (csv_path / "metrics.csv").is_file()
    assert html_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metrics" in payload and "validation" in payload
    assert "sequence_key" not in html_path.read_text(encoding="utf-8").lower()


def test_validation_only_runs_no_metrics(roles, schema_path, config_mapping) -> None:
    report = SynthEvaluator(*roles, schema=schema_path, config=config_mapping).validate()
    assert report.metrics == []
    assert report.metadata["evaluation_status"] == "validation_passed"
