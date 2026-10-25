#!/usr/bin/env python3
"""Compare scientific results across machines with explicit numeric tolerance.

Input/configuration identities and validation outcomes remain exact. Dependency
versions, runtime measurements and the digest of the unnormalised report remain
in each report for provenance, but are not equality requirements across hosts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


IGNORED_METADATA = {
    "created_at_utc", "runtime_seconds", "peak_resident_memory_mb", "machine",
    "dependencies", "scientific_content_sha256",
}
REL_TOL = 1e-12
ABS_TOL = 1e-12


def scientific_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload["metadata"])
    for key in IGNORED_METADATA:
        metadata.pop(key, None)
    for role in metadata.get("datasets", {}).values():
        role.pop("path", None)
    # This one audit count records in-memory bytes, not observations. The audit
    # code, severity and message are still compared, so a failed guard fails.
    for record in payload["validation"]:
        if record.get("code") == "configured_memory_limit_passed":
            record.pop("affected_count", None)
    return {
        "metadata": metadata,
        "validation": payload["validation"],
        "metrics": payload["metrics"],
    }


def differences(observed, expected, path="root") -> list[str]:
    if isinstance(observed, dict) and isinstance(expected, dict):
        found = []
        for key in sorted(observed.keys() | expected.keys()):
            child = f"{path}.{key}"
            if key not in observed or key not in expected:
                found.append(child + ": missing key")
            else:
                found.extend(differences(observed[key], expected[key], child))
        return found
    if isinstance(observed, list) and isinstance(expected, list):
        if len(observed) != len(expected):
            return [path + ": different lengths"]
        return [item for index, (a, b) in enumerate(zip(observed, expected))
                for item in differences(a, b, f"{path}[{index}]")]
    if type(observed) is float and type(expected) is float:
        equal = math.isclose(observed, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    else:
        equal = type(observed) is type(expected) and observed == expected
    return [] if equal else [path + ": different values"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    arguments = parser.parse_args()
    observed = scientific_payload(arguments.observed)
    expected = scientific_payload(arguments.expected)
    mismatches = differences(observed, expected)
    if mismatches:
        raise RuntimeError("reproduced scientific report differs: " + "; ".join(mismatches[:20]))
    print(f"PASS: reproduced scientific results match bundled example (rtol={REL_TOL}, atol={ABS_TOL}); "
          "input/configuration identities and validation outcomes match exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
