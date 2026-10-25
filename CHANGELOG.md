# Changelog

## Public release — 2026-10-25

- Released version 0.2.1 under the MIT licence, with named authorship and citation metadata.
- Made bundled report comparison portable across dependency versions: numerical
  metrics use a tolerance of 1e-12; input/configuration hashes, record counts
  and validation outcomes remain exact. Runtime metadata and memory byte counts
  are not treated as scientific results.
- Added five regression tests for the comparison policy; all 55 tests pass.
- Rebuilt the wheel with the public licence and updated documentation.

## 0.2.1 — 2026-07-29

- Added a two-level uncertainty aggregation API that averages evaluation
  repetitions within each declared independent unit before bootstrapping
  independent-unit means.
- Retained expected, attempted and applicable denominators in aggregate
  results, with deterministic bootstrap seeds and safe handling of
  insufficient independent units.
- Added executable release-boundary auditing, portable bundled reports,
  manifests and checksums.

## 0.2.0 — 2026-07-28

- Added a self-contained identifier-free reproduction example.
- Added a bounded public NYC TLC benchmark adapter.
- Documented metric directions, applicability contracts and uncertainty units.
- Added an explicit public-release boundary and pending-licence status.
- Aligned package and runtime version metadata.
- Optimised exact Wasserstein and combined-label calculations without changing
  scientific results; a regression audit found zero metric-value differences
  at tolerance \(10^{-13}\).
- Retained safe failure for prohibited identifiers and existing outputs.

## 0.1.1 — thesis baseline

- Added CLI, Python API, typed immutable configuration, schema validation,
  deterministic seed controls, metric registry, structured reports and tests.
