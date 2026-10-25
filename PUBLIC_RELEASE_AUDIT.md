# SynthEval public-release audit

Package version: `0.2.1`

Result: **PASS**

This audit verifies the package boundary, not the privacy of user-supplied data.
The fixture is deterministically generated and identifier-free; the bundled report
contains aggregate metric rows only. Mentions of prohibited field names in schemas
and tests are guard declarations, not released values.

| Check | Status | Detail |
|---|---|---|
| `fixture_roles` | PASS | expected=['real_holdout.csv', 'real_train.csv', 'synthetic.csv']; observed=['real_holdout.csv', 'real_train.csv', 'synthetic.csv'] |
| `fixture_real_holdout` | PASS | rows=320; prohibited_columns=[]; nonfictional_location_labels=0 |
| `fixture_real_train` | PASS | rows=320; prohibited_columns=[]; nonfictional_location_labels=0 |
| `fixture_synthetic` | PASS | rows=320; prohibited_columns=[]; nonfictional_location_labels=0 |
| `aggregate_report_shape` | PASS | top_level_keys=['metadata', 'metrics', 'validation']; metric_rows=351 |
| `aggregate_report_portability` | PASS | no machine-specific workspace path is embedded |
| `forbidden_bundled_data_types` | PASS | matches=[] |
| `wheel_count_and_version` | PASS | wheels=['anpr_syntheval-0.2.1-py3-none-any.whl'] |
| `wheel_payload_boundary` | PASS | entries=31; forbidden_entries=[] |
| `machine_specific_paths` | PASS | matches=[] |

## Release boundary

No restricted ANPR records, pseudonymous sequence keys, exact sensitive
locations, exact timestamps, trained weights, rare restricted cells, or
ANPR-derived generated outputs are included. The NYC TLC adapter requires
caller-supplied public data and does not bundle that dataset.

Empirical privacy diagnostics do not establish anonymity or formal privacy.
