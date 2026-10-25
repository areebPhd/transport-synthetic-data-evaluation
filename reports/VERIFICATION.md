# SynthEval 0.2.1 verification

Verification date: 2026-07-29 (UTC)

Environment:

- Python 3.10.12;
- dependency versions recorded in `environment.lock.txt`;
- source and wheel both report version `0.2.1`.

Executed checks:

| Check | Result |
|---|---|
| Full source test suite | 50 passed in 13.04 s |
| `./reproduce.sh` | PASS; 50 tests passed and bundled scientific report matched |
| Example metric count | 351 |
| Scientific-content SHA-256 | `676802352748858ca8391383474f3376bc1fe362cd3b939697399b8e98ba8665` |
| Wheel clean-target install | PASS |
| Wheel API/CLI import and validation | PASS |
| Wheel SHA-256 | `82a0a48981a695cd16f1b4f30beaa9c9ce80ef2ff7e1afd4b8285df88aeaaa16` |
| Public TLC adapter integration test | PASS on an identifier-free structural fixture |
| Public-release boundary audit | PASS |

The official NYC TLC input is intentionally not bundled. The integration test
executes the public adapter against a deterministic, temporary table with the
documented TLC schema and verifies that exact timestamps are not written to
the projected roles. This establishes executability of the adapter, not a
scientific result about the official TLC population.

The package makes no claim that SynthEval invented each component metric. Its
software contribution is the governed, use-conditioned evaluation
architecture, explicit applicability and uncertainty contracts, and
transport-specific integration.
