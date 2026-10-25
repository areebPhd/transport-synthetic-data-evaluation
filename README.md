# SynthEval transport reference implementation

SynthEval is a research-software framework for use-conditioned evaluation of
synthetic transport tables. It keeps held-out fidelity, downstream utility,
empirical privacy-risk diagnostics, transport alignment, applicability and
provenance separate. It deliberately does not emit a universal quality score.

The software does not invent the individual statistical metrics. Its
contribution is the evaluation architecture: explicit data roles,
transport-aware schemas, applicability contracts, safe failure, deterministic
sampling, metric-direction metadata, uncertainty hierarchy and auditable
reports.

Version: 0.2.1

Released under the [MIT licence](LICENSE). See
`LICENSE-or-project-usage-note.md` for the scope of the software and data.

## Install

Python 3.10 or later is required.

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

`environment.lock.txt` records the historical fixture environment.
The project metadata gives the supported lower bounds. See `RELEASE_CHECKS.md`
for the checks performed for this public release.

## One-command identifier-free reproduction

From the repository root:

```bash
bash reproduce.sh
```

This command regenerates three small, deterministic, identifier-free roles,
runs the CLI, executes all tests, and writes `reports/example_report/`. It
requires no restricted data and never downloads ANPR records.

The example comparison allows floating-point roundoff (`rtol=1e-12`,
`atol=1e-12`) across machines. Input/configuration identities, metric structure,
counts and validation outcomes must still match exactly. Dependency versions,
runtime measurements and the unnormalised report digest remain recorded but
are not cross-host equality requirements. The memory-limit pass audit stores
platform-dependent bytes; only that byte count is excluded from comparison.

## Python API

```python
from anpr_syntheval import SynthEvaluator

report = SynthEvaluator(
    real_train="examples/tiny_fixture/real_train.csv",
    real_holdout="examples/tiny_fixture/real_holdout.csv",
    synthetic="examples/tiny_fixture/synthetic.csv",
    schema="configs/anpr_schema.yaml",
    config="examples/tiny_fixture/evaluation.yaml",
).evaluate()

report.to_json("reports/run/results.json")
report.to_csv("reports/run/csv")
report.to_html("reports/run/report.html")
```

Path-based inputs are inspected before feature projection. Files containing a
prohibited identifier or exact timestamp fail before metrics are run.

## Run-level uncertainty API

```python
from anpr_syntheval import RunObservation, aggregate_run_observations

summary = aggregate_run_observations(
    [
        RunObservation(report=seed_11_rep_0, independent_unit="model_seed_11", evaluation_repetition=0),
        RunObservation(report=seed_11_rep_1, independent_unit="model_seed_11", evaluation_repetition=1),
        RunObservation(report=seed_22_rep_0, independent_unit="model_seed_22", evaluation_repetition=0),
    ],
    bootstrap_seed=20260729,
)
```

Evaluation repetitions are averaged within each independent unit before a
deterministic percentile-bootstrap interval is calculated over independent
unit means. Every result retains expected, attempted and applicable
denominators. This API does not convert evaluation repetitions into additional
model-training runs.

## CLI

```text
anpr-syntheval validate --real-train TRAIN --real-holdout HOLDOUT \
  --synthetic SYNTHETIC --schema SCHEMA [--config CONFIG]

anpr-syntheval evaluate --real-train TRAIN --real-holdout HOLDOUT \
  --synthetic SYNTHETIC --schema SCHEMA --config CONFIG --output NEW_DIRECTORY

anpr-syntheval compare --manifest MODELS_YAML --output NEW_DIRECTORY
```

Commands return exit status 2 and a structured JSON error on invalid schema,
unsafe path, prohibited identifier, invalid data, or existing output. Output
directories are never silently overwritten.

## Public benchmark example

`examples/tlc/README.md` and `scripts/run_tlc_benchmark.py` define a bounded
example using public NYC Taxi and Limousine Commission yellow-taxi trip
records. The adapter reads a caller-supplied official Parquet file, derives a
non-identifying transport projection, makes chronological train/holdout roles,
constructs an explicitly weak independent-marginal reference, and evaluates
it. The public data are not bundled.

## Interpretation

- Metric direction and target values are part of every result.
- `applicable=false` and its exact reason are results, not missing data.
- A deterministic evaluation repetition is not an independent model run.
- Model-run uncertainty must be aggregated across independently trained models;
  the package does not manufacture that uncertainty from one evaluation.
- Empirical privacy-risk diagnostics do not prove anonymity, legal compliance
  or formal protection.
- A realistic-looking row is not evidence of fidelity, utility or safety.

See `docs/metric_directions.md`, `docs/applicability_and_uncertainty.md`, and
`docs/release_boundary.md`.

## Public-release audit

```bash
python scripts/audit_public_release.py --write
sha256sum --check SHA256SUMS.txt
```

The audit checks the fictional fixture, aggregate-report shape, absence of
machine-specific paths and prohibited bundled data types, and the installed
wheel payload. It writes `PUBLIC_RELEASE_AUDIT.md`, `MANIFEST.json` and
`SHA256SUMS.txt`. This is a package-boundary check, not a privacy certification
for data supplied by users.

## Related software

The companion [ANPR mobility pipeline](https://github.com/areebPhd/anpr-mobility-pipeline)
constructs the analytical mobility records used by the thesis methodology.
