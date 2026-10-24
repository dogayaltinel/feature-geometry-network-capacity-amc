"""Measure representative full-epoch wall time on the real corpus, validation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "smoke_test" / "pilot_runtime_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing runtime output: {args.output}")
    args.output.mkdir(parents=True)
    data = REPO_ROOT / "RML2016.10a_dict.pkl"
    split_index = ROOT / "data_splits" / "split_A.npz"
    cells = [("F2", 0.5), ("F2", 2.0), ("F7", 0.5), ("F7", 2.0)]
    rows = []
    for index, (feature, width) in enumerate(cells):
        run_id = f"runtime_{feature}_{width:g}"
        command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data), "--split-index", str(split_index), "--feature", feature, "--width", str(width), "--seed", "42", "--epochs", "1", "--batch-size", "512", "--output-root", str(args.output), "--run-id", run_id, "--stage", "pilot", "--analysis-role", "descriptive", "--device", "auto", "--allow-dirty"]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        wall_seconds = time.perf_counter() - started
        run_dir = args.output / "records" / "validation" / run_id
        record = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
        epoch_seconds = float(record["metrics"]["training"]["epoch_seconds"][0])
        (args.output / f"{run_id}.stdout.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        rows.append({"feature": feature, "width_multiplier": width, "wall_seconds": wall_seconds, "epoch_seconds": epoch_seconds, "projected_200_epoch_hours": epoch_seconds * 200 / 3600.0, "projected_300_epoch_hours": epoch_seconds * 300 / 3600.0})
    report = {"schema_version": "feature_geometry.pilot_runtime.v1", "dataset": str(data), "split": "A", "validation_only": True, "device": "auto", "cells": rows, "mean_epoch_seconds": sum(row["epoch_seconds"] for row in rows) / len(rows), "projected_mean_200_epoch_hours": sum(row["projected_200_epoch_hours"] for row in rows) / len(rows), "projected_mean_300_epoch_hours": sum(row["projected_300_epoch_hours"] for row in rows) / len(rows)}
    (args.output / "runtime_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
