"""Paired confirmatory analysis with validation/test lock and Holm correction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path

from common import atomic_write_json, get_path, load_test_records, load_validation_records


def paired_permutation(values: list[float], iterations: int = 10000, seed: int = 1731) -> float:
    observed = abs(sum(values) / len(values))
    if len(values) <= 15:
        signed_means = [
            abs(sum(value * sign for value, sign in zip(values, signs)) / len(values))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
        return sum(value >= observed - 1e-15 for value in signed_means) / len(signed_means)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        signed = [value if rng.random() < 0.5 else -value for value in values]
        if abs(sum(signed) / len(signed)) >= observed:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def bootstrap_ci(values: list[float], iterations: int = 10000, seed: int = 1733) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def t_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    try:
        from scipy.stats import t
        critical = float(t.ppf(0.975, len(values) - 1))
    except ImportError:
        critical = 1.96
    return mean - critical * standard_error, mean + critical * standard_error


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        corrected = min(1.0, (count - index) * value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def select_records(metric: str) -> list[dict]:
    if metric.startswith("test."):
        return load_test_records()
    return load_validation_records()


def metric_value(record: dict, metric: str) -> float:
    if metric.startswith("test."):
        return float(get_path(record, metric))
    return float(get_path(record, metric))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", default="metrics.validation.final_accuracy")
    args = parser.parse_args()
    records = select_records(args.metric)
    rows = {}
    for record in records:
        cell = record.get("cell", {})
        if record.get("analysis_role") not in {"confirmatory_eligible", "confirmatory"} and record.get("record_type") == "validation_run":
            continue
        if record.get("record_type") == "test_run":
            cell = record.get("cell", record.get("test", {}))
        key = (record.get("split") or record.get("data", {}).get("split"), record.get("seed") or record.get("training", {}).get("seed"), cell.get("feature"), str(cell.get("width_multiplier")))
        rows[key] = metric_value(record, args.metric)
    contrasts = {}
    p_values = {}

    def add_contrast(name: str, value_fn) -> None:
        values = []
        split_values = {"A": [], "B": []}
        for split in ("A", "B"):
            for seed in (42, 17, 123, 2024, 31):
                value = value_fn(split, seed)
                if value is not None:
                    values.append(value)
                    split_values[split].append(value)
        if values:
            contrasts[name] = {
                "n": len(values),
                "mean_difference": sum(values) / len(values),
                "ci95": bootstrap_ci(values),
                "ci95_t": t_ci(values),
                "values": values,
                "by_split": {
                    split: {
                        "n": len(split_values[split]),
                        "mean_difference": statistics.mean(split_values[split]) if split_values[split] else None,
                        "ci95_t": t_ci(split_values[split]) if split_values[split] else None,
                        "values": split_values[split],
                    }
                    for split in ("A", "B")
                },
            }
            p_values[name] = paired_permutation(values)

    for width in ("0.5", "1.0", "2.0"):
        add_contrast(
            f"F7_minus_F2_at_{width}",
            lambda split, seed, width=width: (
                rows.get((split, seed, "F7", width))
                - rows.get((split, seed, "F2", width))
                if rows.get((split, seed, "F7", width)) is not None
                and rows.get((split, seed, "F2", width)) is not None
                else None
            ),
        )

    for feature in ("F2", "F7"):
        add_contrast(
            f"capacity_0.5_to_2.0_{feature}",
            lambda split, seed, feature=feature: (
                rows.get((split, seed, feature, "2.0"))
                - rows.get((split, seed, feature, "0.5"))
                if rows.get((split, seed, feature, "2.0")) is not None
                and rows.get((split, seed, feature, "0.5")) is not None
                else None
            ),
        )
        add_contrast(
            f"capacity_0.5_to_1.0_{feature}",
            lambda split, seed, feature=feature: (
                rows.get((split, seed, feature, "1.0"))
                - rows.get((split, seed, feature, "0.5"))
                if rows.get((split, seed, feature, "1.0")) is not None
                and rows.get((split, seed, feature, "0.5")) is not None
                else None
            ),
        )
        add_contrast(
            f"capacity_1.0_to_2.0_{feature}",
            lambda split, seed, feature=feature: (
                rows.get((split, seed, feature, "2.0"))
                - rows.get((split, seed, feature, "1.0"))
                if rows.get((split, seed, feature, "2.0")) is not None
                and rows.get((split, seed, feature, "1.0")) is not None
                else None
            ),
        )

    add_contrast(
        "interaction_capacity_F7_minus_F2",
        lambda split, seed: (
            (rows.get((split, seed, "F7", "2.0")) - rows.get((split, seed, "F7", "0.5")))
            - (rows.get((split, seed, "F2", "2.0")) - rows.get((split, seed, "F2", "0.5")))
            if all(rows.get((split, seed, feature, width)) is not None
                   for feature in ("F2", "F7") for width in ("0.5", "2.0"))
            else None
        ),
    )
    if not contrasts:
        raise ValueError("No confirmatory F2/F7 pairs were found")
    corrected = holm(p_values)
    for name in contrasts:
        contrasts[name]["p_raw"] = p_values[name]
        contrasts[name]["p_holm"] = corrected[name]
    atomic_write_json(args.output, {
        "schema_version": "feature_geometry.confirmatory_analysis.v1",
        "metric": args.metric,
        "test_accessed": args.metric.startswith("test."),
        "holm_family": list(contrasts),
        "contrasts": contrasts,
        "analysis_notes": [
            "Paired unit is split x seed.",
            "Pooled sample-level McNemar is not used.",
            "The confirmatory family contains three feature contrasts, six capacity contrasts, and one interaction contrast.",
            "H4 and H5 are descriptive and outside the Holm family.",
        ],
    })
    print(f"Wrote {len(contrasts)} contrasts to {args.output}")


if __name__ == "__main__":
    main()
