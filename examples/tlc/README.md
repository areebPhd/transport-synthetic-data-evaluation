# Public NYC TLC benchmark example

This example demonstrates the same evaluation architecture on public transport
data. It is not evidence about the restricted ANPR population and is not
pooled with the thesis’s principal model results.

The NYC Taxi and Limousine Commission publishes monthly trip records in
Parquet format and cautions that it does not guarantee their accuracy or
completeness:

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The frozen example reference is January 2024 yellow-taxi trips:

https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet

Download that official file, then run:

```bash
PYTHONPATH=src python scripts/run_tlc_benchmark.py \
  --input /path/to/yellow_tripdata_2024-01.parquet \
  --output /new/output/directory \
  --maximum-rows 30000 \
  --seed 20260728
```

The adapter:

1. checks the required TLC fields;
2. removes non-positive, over-three-hour, over-100-mile and over-160-km/h
   records;
3. derives an identifier-free zone, time and movement projection;
4. orders by pickup time, writes no exact timestamps, and uses the first 70%
   for training and last 30% for held-out evaluation;
5. creates an explicitly weak training-only independent-marginal reference;
6. reconstructs speed and coarse bands deterministically;
7. writes role hashes, the input hash, seed and report digest.

This is an executable public benchmark example, not a claim that the weak
reference is a suitable production generator. TLC zone/time combinations may
still be sensitive in other contexts; public availability does not remove the
need for responsible output handling.
