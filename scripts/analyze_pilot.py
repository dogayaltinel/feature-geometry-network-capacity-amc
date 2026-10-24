"""Validation-only 200 versus 300 epoch pilot analysis."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from common import atomic_write_json, load_validation_records, get_path

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover - optional dependency in the locked environment
    student_t = None


def bootstrap_ci(values: list[float], iterations: int = 10000, seed: int = 1729) -> tuple[float, float]:
    if not values:
        raise ValueError("No paired differences")
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def paired_t_ci(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    critical = float(student_t.ppf(0.975, len(values) - 1)) if student_t is not None else 1.96
    return mean - critical * standard_error, mean + critical * standard_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, help="Validation-record directory; defaults to the study records directory")
    parser.add_argument("--margin-pp", type=float, default=0.15)
    parser.add_argument("--snr-margin-pp", type=float, default=0.30)
    args = parser.parse_args()
    records = load_validation_records(args.validation_dir)
    groups: dict[tuple[str, str, str, int], dict[int, dict]] = defaultdict(dict)
    for record in records:
        cell = record.get("cell", {})
        if record.get("stage") != "pilot":
            continue
        if cell.get("backbone") != "BB1" or cell.get("feature") not in {"F2", "F7"}:
            continue
        if cell.get("width_multiplier") not in {0.5, 2.0}:
            continue
        key = (cell["feature"], str(cell["width_multiplier"]), record["data"]["split"], record["training"]["seed"])
        groups[key][cell["epoch_budget"]] = record
    differences: list[float] = []
    snr_differences: list[float] = []
    pair_rows = []
    for key, budgets in sorted(groups.items()):
        if 200 not in budgets or 300 not in budgets:
            continue
        # Report and compare all pilot differences in percentage points.
        low = 100.0 * float(get_path(budgets[200], "metrics.validation.final_accuracy"))
        high = 100.0 * float(get_path(budgets[300], "metrics.validation.final_accuracy"))
        difference = high - low
        differences.append(difference)
        pair_rows.append({"key": key, "difference_300_minus_200_pp": difference})
        low_snr = get_path(budgets[200], "metrics.validation.snr_regions", {}) or {}
        high_snr = get_path(budgets[300], "metrics.validation.snr_regions", {}) or {}
        for region in set(low_snr) | set(high_snr):
            if region in low_snr and region in high_snr:
                snr_differences.append(100.0 * (float(high_snr[region]) - float(low_snr[region])))
    if len(differences) != 12:
        raise ValueError(f"Expected 12 pilot pairs, found {len(differences)}")
    t_ci = paired_t_ci(differences)
    boot_ci = bootstrap_ci(differences)
    selected = "200" if (t_ci[0] >= -args.margin_pp and t_ci[1] <= args.margin_pp and boot_ci[0] >= -args.margin_pp and boot_ci[1] <= args.margin_pp and (not snr_differences or max(abs(x) for x in snr_differences) <= args.snr_margin_pp)) else "300"
    atomic_write_json(args.output, {
        "schema_version": "feature_geometry.pilot_analysis.v1",
        "records_used": len(differences) * 2,
        "pairs": pair_rows,
        "difference_definition": "validation_300_minus_validation_200_percentage_points",
        "paired_t_interval": t_ci,
        "paired_t_interval_method": "student_t" if student_t is not None else "normal_fallback",
        "paired_bootstrap_interval": boot_ci,
        "snr_max_abs_difference_pp": max((abs(x) for x in snr_differences), default=None),
        "equivalence_margin_pp": args.margin_pp,
        "snr_margin_pp": args.snr_margin_pp,
        "selected_epoch_budget": int(selected),
        "test_accessed": False,
    })
    print(f"Selected epoch budget: {selected}")


if __name__ == "__main__":
    main()
