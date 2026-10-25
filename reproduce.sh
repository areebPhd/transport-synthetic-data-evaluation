#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
run_root="$(mktemp -d)"
trap 'rm -rf "$run_root"' EXIT

"$python_bin" "$project_root/scripts/generate_identifier_free_fixture.py" \
  --output "$run_root/fixture" --rows 320

cp "$project_root/examples/tiny_fixture/evaluation.yaml" "$run_root/evaluation.yaml"

PYTHONPATH="$project_root/src" "$python_bin" -m anpr_syntheval.cli evaluate \
  --real-train "$run_root/fixture/real_train.csv" \
  --real-holdout "$run_root/fixture/real_holdout.csv" \
  --synthetic "$run_root/fixture/synthetic.csv" \
  --schema "$project_root/configs/anpr_schema.yaml" \
  --config "$run_root/evaluation.yaml" \
  --output "$run_root/report"

PYTHONPATH="$project_root/src" "$python_bin" -m pytest \
  -o "cache_dir=$run_root/pytest_cache" "$project_root/tests"

"$python_bin" "$project_root/scripts/compare_example_report.py" \
  --observed "$run_root/report/results.json" \
  --expected "$project_root/reports/example_report/results.json"
