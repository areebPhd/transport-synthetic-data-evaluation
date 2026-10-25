"""Cross-host tolerance must not hide changed scientific results or inputs."""
import importlib.util
import json
from pathlib import Path

script = Path(__file__).resolve().parents[2] / "scripts/compare_example_report.py"
spec = importlib.util.spec_from_file_location("example_comparison", script)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_roundoff_is_accepted_but_material_metric_change_is_rejected():
    assert comparison.differences({"value": .01880641672765245}, {"value": .01880641672765246}) == []
    assert comparison.differences({"value": .0188}, {"value": .0288})


def test_counts_and_input_identities_remain_exact():
    assert comparison.differences({"count": 320}, {"count": 321})
    assert comparison.differences({"sha256": "input-a"}, {"sha256": "input-b"})


def test_nonfinite_or_missing_values_do_not_pass_as_matching_numbers():
    assert comparison.differences({"value": float("nan")}, {"value": float("nan")})
    assert comparison.differences({"value": None}, {"value": 0.0})
    assert comparison.differences({"value": 0.0}, {})


def test_platform_memory_count_is_normalised_but_guard_status_is_retained(tmp_path):
    def write(name, code, count):
        path = tmp_path / name
        path.write_text(json.dumps({"metadata": {"datasets": {}}, "metrics": [],
                                   "validation": [{"code": code, "affected_count": count}]}))
        return comparison.scientific_payload(path)
    a = write("a.json", "configured_memory_limit_passed", 337297)
    b = write("b.json", "configured_memory_limit_passed", 337309)
    assert comparison.differences(a, b) == []
    assert comparison.differences(a, write("c.json", "configured_memory_limit_exceeded", 337309))


def test_other_validation_counts_are_not_ignored(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"metadata": {"datasets": {}}, "metrics": [],
                               "validation": [{"code": "same_endpoint", "affected_count": 5}]}))
    assert comparison.scientific_payload(path)["validation"][0]["affected_count"] == 5
