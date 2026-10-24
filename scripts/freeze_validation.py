"""Freeze the validation analysis before test unblinding."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    status = json.loads(args.status.read_text(encoding="utf-8"))
    if analysis.get("test_accessed"):
        raise SystemExit("Cannot freeze an analysis that accessed test data")
    if status.get("status") != "completed" or status.get("test_accessed"):
        raise SystemExit("Confirmatory queue is not completed or already accessed test data")
    required = {
        name: args.artifacts / name
        for name in (
            "confirmatory_validation_contrasts.csv",
            "confirmatory_validation_contrasts.tex",
            "confirmatory_effects.svg",
            "confirmatory_validation_report.md",
        )
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing validation artifact(s): {missing}")

    payload = {
        "schema_version": "feature_geometry.unblind_lock.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Freeze validation analysis and confirmatory records before final test inference.",
        "validation_analysis": {
            "path": str(args.analysis.resolve()),
            "sha256": sha256_file(args.analysis),
        },
        "confirmatory_status": {
            "path": str(args.status.resolve()),
            "sha256": sha256_file(args.status),
            "new_execution_count": status.get("new_execution_count"),
            "reused_split_A_records": status.get("reused_split_A_records"),
            "test_accessed": status.get("test_accessed"),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in required.items()
        },
        "test_inference_script": {
            "path": str((args.output.parent / "scripts" / "run_test_inference.py").resolve()),
            "sha256": sha256_file(args.output.parent / "scripts" / "run_test_inference.py"),
        },
        "rules": [
            "No test record may be used for model, epoch, feature, capacity, or split selection.",
            "Test records are written only under records/test/ and carry this lock hash.",
            "Post-unblind analyses are labeled post hoc.",
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"lock": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
