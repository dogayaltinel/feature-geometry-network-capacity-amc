from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


MODULATIONS = ("8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM")
SNRS = tuple(range(-20, 20, 2))


def _write_tiny_dataset(path: Path) -> None:
    rng = np.random.default_rng(123)
    data = {(modulation, snr): rng.normal(size=(2, 2, 128)).astype(np.float32) for modulation in MODULATIONS for snr in SNRS}
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=4)


def _run_train(train_script: Path, data: Path, output_root: Path, run_id: str, epochs: int, **extra: str) -> None:
    command = [
        sys.executable,
        str(train_script),
        "--data",
        str(data),
        "--feature",
        "F2",
        "--width",
        "0.5",
        "--seed",
        "42",
        "--epochs",
        str(epochs),
        "--batch-size",
        "32",
        "--max-train-batches",
        "1",
        "--max-val-batches",
        "1",
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--stage",
        "smoke",
        "--allow-dirty",
    ]
    for key, value in extra.items():
        command.extend([key, value])
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_train_checkpoint_and_resume(tmp_path: Path) -> None:
    data = tmp_path / "tiny_rml.pkl"
    output_root = tmp_path / "study"
    _write_tiny_dataset(data)
    train_script = Path(__file__).parents[1] / "scripts" / "train.py"

    _run_train(train_script, data, output_root, "first", 1)
    first_dir = output_root / "records" / "validation" / "first"
    first_checkpoint = first_dir / "final_checkpoint.pt"
    assert first_checkpoint.is_file()

    _run_train(train_script, data, output_root, "resumed", 2, **{"--resume": str(first_checkpoint)})
    resumed_record = output_root / "records" / "validation" / "resumed" / "run_record.json"
    record = json.loads(resumed_record.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["provenance"]["resumed"] is True
    assert record["provenance"]["parent_run_id"] == "first"
    assert record["metrics"]["training"]["epochs_completed"] == 2
