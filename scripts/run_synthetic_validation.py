"""Run and audit the synthetic validation/checkpoint/resume path end to end."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


MODULATIONS = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
SNRS = tuple(range(-20, 20, 2))
ROOT = Path(__file__).resolve().parents[1]


def write_dataset(path: Path) -> None:
    rng = np.random.default_rng(123)
    data = {(modulation, snr): rng.normal(size=(2, 2, 128)).astype(np.float32) for modulation in MODULATIONS for snr in SNRS}
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=4)


def run_train(data: Path, output_root: Path, run_id: str, epochs: int, resume: Path | None, log_path: Path) -> dict[str, object]:
    command = [
        sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(data), "--feature", "F2", "--width", "0.5",
        "--seed", "42", "--epochs", str(epochs), "--batch-size", "32", "--max-train-batches", "1", "--max-val-batches", "1",
        "--output-root", str(output_root), "--run-id", run_id, "--stage", "smoke", "--allow-dirty",
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "smoke_test" / "resume_e2e_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {args.output}")
    args.output.mkdir(parents=True)
    data = args.output / "synthetic_rml.pkl"
    write_dataset(data)
    initial_log = args.output / "initial.stdout.log"
    resumed_log = args.output / "resumed.stdout.log"
    first = run_train(data, args.output, "initial", 1, None, initial_log)
    first_checkpoint = args.output / "records" / "validation" / "initial" / "final_checkpoint.pt"
    second = run_train(data, args.output, "resumed", 2, first_checkpoint, resumed_log)
    record_path = args.output / "records" / "validation" / "resumed" / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    checks = {
        "initial_completed": first.get("status") == "completed",
        "initial_checkpoint_exists": first_checkpoint.is_file(),
        "resumed_completed": second.get("status") == "completed",
        "resumed_flag": record["provenance"]["resumed"] is True,
        "parent_run_id": record["provenance"]["parent_run_id"] == "initial",
        "epochs_completed": record["metrics"]["training"]["epochs_completed"] == 2,
        "test_inference_not_run": second.get("test_inference") == "not_run",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Synthetic resume audit failed: {checks}")
    report = {"schema_version": "feature_geometry.synthetic_resume_audit.v1", "status": "pass", "checks": checks, "initial_run": first, "resumed_run": second}
    (args.output / "resume_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
