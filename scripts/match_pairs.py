"""Create Core-FLOP-matched feature pairs from the ledger only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import sha256_file


def read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"feature", "backbone", "width_multiplier", "core_flops"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Ledger does not contain the matching fields")
    return rows


def build_pairs(rows: list[dict[str, str]], tolerance: float) -> list[dict[str, object]]:
    candidates: list[tuple[float, dict[str, str], dict[str, str]]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["backbone"] != right["backbone"] or left["feature"] == right["feature"]:
                continue
            a = float(left["core_flops"])
            b = float(right["core_flops"])
            relative = abs(a - b) / ((a + b) / 2.0)
            if relative <= tolerance:
                candidates.append((relative, left, right))
    candidates.sort(key=lambda item: item[0])
    used: set[str] = set()
    output: list[dict[str, object]] = []
    for relative, left, right in candidates:
        left_key = left.get("ledger_key") or f"{left['feature']}:{left['width_multiplier']}"
        right_key = right.get("ledger_key") or f"{right['feature']}:{right['width_multiplier']}"
        if left_key in used or right_key in used:
            continue
        used.update((left_key, right_key))
        output.append({
            "left_key": left_key,
            "right_key": right_key,
            "left_feature": left["feature"],
            "right_feature": right["feature"],
            "left_width": left["width_multiplier"],
            "right_width": right["width_multiplier"],
            "relative_core_flop_difference": relative,
            "selection_basis": "ledger_only",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()
    rows = read_ledger(args.ledger)
    pairs = build_pairs(rows, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema_version": "feature_geometry.matching_pairs.v1",
        "ledger_sha256": sha256_file(args.ledger),
        "tolerance_relative_core_flops": args.tolerance,
        "accuracy_accessed": False,
        "pairs": pairs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
