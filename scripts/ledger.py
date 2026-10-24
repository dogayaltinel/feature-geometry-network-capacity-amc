"""Append-only cost ledger validation and insertion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import canonical_hash


FIELDS = [
    "ledger_key", "feature", "backbone", "width_multiplier", "input_shape",
    "parameters_trainable", "parameters_total", "core_macs", "core_flops",
    "non_mac_flops", "preprocessing_flops", "total_flops", "profiler",
    "independent_profiler", "profiler_difference_rel", "operator_scope", "config_sha256",
]
KEY_FIELDS = ("feature", "backbone", "width_multiplier")


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    missing = [field for field in FIELDS if field not in row]
    if missing:
        raise ValueError(f"Ledger row is missing fields: {', '.join(missing)}")
    normalized = {field: str(row[field]) for field in FIELDS}
    normalized["input_shape"] = json.dumps(row["input_shape"], separators=(",", ":")) if isinstance(row["input_shape"], list) else normalized["input_shape"]
    return normalized


def validate_row(row: dict[str, str]) -> None:
    core_macs = float(row["core_macs"])
    core_flops = float(row["core_flops"])
    non_mac = float(row["non_mac_flops"])
    preprocessing = float(row["preprocessing_flops"])
    total = float(row["total_flops"])
    if abs(core_flops - 2.0 * core_macs) > max(1.0, 1e-6 * abs(core_flops)):
        raise ValueError("Core FLOPs must equal 2 * Core MACs")
    if abs(total - (core_flops + non_mac + preprocessing)) > max(1.0, 1e-6 * abs(total)):
        raise ValueError("Total FLOPs must equal Core + non-MAC + preprocessing FLOPs")
    if float(row["profiler_difference_rel"]) > 0.02:
        raise ValueError("Independent profiler difference exceeds 2%; document the deviation first")


def append_row(path: Path, row: dict[str, Any]) -> None:
    normalized = normalize_row(row)
    validate_row(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Ledger schema changed; update the manifest and deviation log first")
            existing = list(reader)
        key = tuple(normalized[field] for field in KEY_FIELDS)
        if any(tuple(item[field] for field in KEY_FIELDS) == key for item in existing):
            raise ValueError(f"Append-only violation for ledger key: {key}")
    else:
        existing = []
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerow(normalized)


def ledger_hash(path: Path) -> str:
    return canonical_hash(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    append = sub.add_parser("append")
    append.add_argument("--ledger", type=Path, required=True)
    append.add_argument("--row-json", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "append":
        append_row(args.ledger, json.loads(args.row_json.read_text(encoding="utf-8")))
    else:
        with args.ledger.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Ledger header does not match the locked schema")
            for row in reader:
                validate_row(row)
        print(f"Validated {args.ledger}")


if __name__ == "__main__":
    main()
