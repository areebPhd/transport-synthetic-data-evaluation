# Data and release boundary

This candidate package is public-facing software, not a public data release.
It contains no restricted ANPR observations, sequence keys, exact camera
coordinates, trained model weights or generated ANPR rows.

Included:

- source code and tests;
- configuration templates;
- a newly constructed identifier-free fixture using fictional camera labels;
- an adapter for caller-supplied public NYC TLC trip records;
- example aggregate reports.

Excluded:

- number plates, vehicle identifiers and pseudonymous sequence keys;
- exact restricted timestamps;
- exact ANPR camera coordinates;
- rare restricted mobility cells;
- raw or repaired ANPR-derived generated rows;
- private model outputs and training checkpoints.

Passing the software’s empirical privacy diagnostics does not authorise data
release. Any row-level or aggregate output derived from restricted input needs
a separate legal, ethical and statistical disclosure review, including
minimum-cell and complementary-disclosure checks.
