"""Audit the locked cost ledger without reading any accuracy records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ledger import FIELDS, validate_row
from common import sha256_file


def audit(path: Path) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("Ledger header does not match the locked schema")
        rows = list(reader)

    keys = [row["ledger_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Ledger keys are not unique")

    for row in rows:
        validate_row(row)
        if abs(float(row["core_flops"]) - 2.0 * float(row["core_macs"])) > 1.0:
            raise ValueError(f"Core FLOP/MAC mismatch: {row['ledger_key']}")
        if abs(float(row["total_flops"]) - (float(row["core_flops"]) + float(row["non_mac_flops"]) + float(row["preprocessing_flops"]))) > 1.0:
            raise ValueError(f"Total FLOP mismatch: {row['ledger_key']}")

    return {
        "schema_version": "feature_geometry.cost_ledger_audit.v1",
        "ledger_sha256": sha256_file(path),
        "rows": len(rows),
        "unique_ledger_keys": len(set(keys)),
        "accuracy_accessed": False,
        "core_flops_equals_two_core_macs": True,
        "total_flops_equals_core_plus_non_mac_plus_preprocessing": True,
        "independent_profiler_max_relative_difference": max(float(row["profiler_difference_rel"]) for row in rows),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
