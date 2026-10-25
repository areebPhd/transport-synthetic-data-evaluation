# Software paper outline

## Proposed title

SynthEval: use-conditioned and transport-aware evaluation of synthetic movement
tables

## 1. Motivation

- Synthetic mobility data are often judged by visual realism or isolated
  aggregate metrics.
- Validity, fidelity, utility and empirical privacy risk can conflict.
- Applicability and provenance failures are commonly hidden by summary scores.

## 2. Contribution

- Role-separated evaluation against real training and held-out data.
- Typed transport schema and prohibited-identifier preflight.
- Metric registry with direction, target, unit and comparison contracts.
- Use-conditioned utility tasks with target-leakage and support checks.
- Structured non-applicability and safe failure.
- Distinct uncertainty levels for model training, generation and evaluation.
- Reproducible reports with hashes, seeds and sampled-index digests.

The originality claim is architectural and integrative, not invention of every
underlying distance or predictive metric.

## 3. Architecture

- API and CLI surfaces.
- Immutable typed configuration and schema.
- Data-role loading and validation.
- Metric-family registry.
- JSON, CSV and HTML reports.

## 4. Validation

- Unit, property, regression, privacy-guard and integration tests.
- Determinism checks.
- Optimisation-equivalence check.
- Identifier-free fixture.
- Public NYC TLC example.

## 5. Case studies

- VAE and CTGAN raw-versus-repaired ANPR-derived tables.
- Conditional-frequency, Gaussian-copula and independent-marginal references.
- Public TLC demonstration.

## 6. Limitations

- Empirical privacy diagnostics do not prove anonymity.
- No universal quality score.
- Model-run uncertainty requires externally supplied independent runs.
- Transport schemas and utility tasks require domain review.
- Legal and disclosure decisions remain outside the software.

## 7. Availability and citation

- Candidate version 0.2.1.
- MIT licence; public release dated 25 October 2026.
- Archival identifier and final thesis metadata to be added at deposit.
