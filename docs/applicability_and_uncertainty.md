# Applicability and uncertainty

## Applicability contract

A metric is applicable only when its declared prerequisites hold. Examples
include sufficient target-class support, at least two classes, required
features, valid physical fields and an absence of deterministic target leakage.
Non-applicability is emitted as a structured result with:

- `applicable=false`;
- an exact reason;
- the attempted comparison and feature/task;
- the evaluated data-role audits.

Non-applicable repetitions must remain in denominators. They must not be
silently removed or replaced with zero.

## Uncertainty hierarchy

SynthEval distinguishes:

1. independent training runs, which capture model-fitting variability;
2. independently generated baseline datasets, which capture sampling
   variability from one fitted statistical model;
3. deterministic evaluation repetitions, which expose bounded resampling and
   evaluation variability but are not additional trained models;
4. row-level bootstrap calculations, which quantify a conditional sampling
   component and do not replace model-run uncertainty.

The report records the evaluation seed and sampling indices. Thesis-level
confidence intervals are computed over independent model-run means. Five
evaluation repetitions for one trained model do not create five independent
models.

## Safe aggregation

Before aggregating:

- retain attempted and applicable counts;
- keep model family, raw/repaired variant and task separate;
- average deterministic evaluation repetitions within their trained model;
- aggregate across independent trained models only at the next level;
- label intervals by the independent unit;
- do not interpret overlapping intervals as definitive superiority;
- do not pool metrics with incompatible directions into an overall score.
