"""Command-line interface for validation, evaluation and model-manifest comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import EvaluationConfig
from .evaluator import SynthEvaluator
from .reporting.json_report import write_json
from .validation import validate_output_path


def _default_config() -> EvaluationConfig:
    return EvaluationConfig.from_mapping({
        "config_version": "1.0.0",
        "random_seed": 20260718,
        "candidate_provenance": "generated_from_authorised_train",
        "validation_mode": "strict",
        "sampling": {"maximum_rows_per_role": 50000, "memory_limit_mb": 2048},
        "reporting": {
            "forbidden_output_path_components": ["restricted", "raw", "source_data"],
            "write_row_level_data": False,
        },
    })


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--real-train", required=True)
    parser.add_argument("--real-holdout", required=True)
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--config")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anpr-syntheval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate three data roles without metrics")
    _add_inputs(validate)
    evaluate = subparsers.add_parser("evaluate", help="run all configured metric families")
    _add_inputs(evaluate)
    evaluate.add_argument("--output", required=True)
    compare = subparsers.add_parser("compare", help="evaluate available models declared in YAML")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--output", required=True)
    return parser


def _evaluator(arguments: argparse.Namespace) -> SynthEvaluator:
    config = EvaluationConfig.from_yaml(arguments.config) if arguments.config else _default_config()
    return SynthEvaluator(
        arguments.real_train,
        arguments.real_holdout,
        arguments.synthetic,
        arguments.schema,
        config,
    )


def _prepare_output(path: str | Path, forbidden: list[str]) -> Path:
    target = validate_output_path(path, forbidden)
    if target.exists() and any(target.iterdir() if target.is_dir() else [target]):
        raise FileExistsError(f"Output already exists and will not be overwritten: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def command_validate(arguments: argparse.Namespace) -> int:
    report = _evaluator(arguments).validate()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


def command_evaluate(arguments: argparse.Namespace) -> int:
    evaluator = _evaluator(arguments)
    forbidden = list(evaluator.config.reporting.get(
        "forbidden_output_path_components", ["restricted", "raw", "source_data"]
    ))
    output = _prepare_output(arguments.output, forbidden)
    report = evaluator.evaluate()
    report.to_json(output / "results.json")
    report.to_csv(output / "csv")
    report.to_html(output / "report.html")
    print(json.dumps({
        "status": "completed", "output": str(output),
        "metric_count": len(report.metrics), "scientific_content_sha256": report.canonical_digest()
    }, sort_keys=True))
    return 0


def command_compare(arguments: argparse.Namespace) -> int:
    manifest_path = Path(arguments.manifest).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    output = _prepare_output(arguments.output, ["restricted", "raw", "source_data"])
    global_inputs = manifest.get("inputs", {})
    summaries: list[dict[str, Any]] = []
    models = manifest.get("models", {})
    for name, declaration in models.items():
        synthetic = declaration.get("generated_dataset_path") or declaration.get("synthetic")
        if not synthetic:
            summaries.append({
                "model": name, "status": declaration.get("status", "unavailable"),
                "reason": declaration.get("reason", "No generated dataset path was declared")
            })
            continue
        real_train = declaration.get("real_train", global_inputs.get("real_train"))
        holdout = declaration.get("real_holdout", global_inputs.get("real_holdout"))
        schema = declaration.get("schema", global_inputs.get("schema"))
        config = declaration.get("config", global_inputs.get("config"))
        if not all((real_train, holdout, schema, config)):
            summaries.append({"model": name, "status": "invalid_manifest", "reason": "Required role/config path missing"})
            continue
        model_output = output / str(name)
        model_output.mkdir(parents=True, exist_ok=False)
        report = SynthEvaluator(real_train, holdout, synthetic, schema, config).evaluate()
        report.to_json(model_output / "results.json")
        report.to_csv(model_output / "csv")
        report.to_html(model_output / "report.html")
        summaries.append({
            "model": name, "status": "completed", "metric_count": len(report.metrics),
            "scientific_content_sha256": report.canonical_digest()
        })
    write_json({"manifest": str(manifest_path), "models": summaries}, output / "comparison.json")
    print(json.dumps({"status": "completed", "models": summaries}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            return command_validate(arguments)
        if arguments.command == "evaluate":
            return command_evaluate(arguments)
        return command_compare(arguments)
    except Exception as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
