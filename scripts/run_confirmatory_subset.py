"""Run the predeclared BB1 confirmatory subset on splits A and B.

The 18 already completed Split-A main-grid records are referenced, not copied
or counted as new executions.  The remaining 42 records are run with the
locked 300-epoch validation-only protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DATA = REPO_ROOT / "RML2016.10a_dict.pkl"
STATUS_ROOT = ROOT / "confirmatory_v1"
STATUS_PATH = STATUS_ROOT / "confirmatory_status.json"
RECORD_ROOT = ROOT / "records" / "validation"
FEATURES = ("F2", "F7")
WIDTHS = (0.5, 1.0, 2.0)
SEEDS = (42, 17, 123, 2024, 31)
EPOCHS = 300


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(split: str, feature: str, width: float, seed: int) -> str:
    return f"confirmatory_{split}_{feature}_{width:g}_seed{seed}"


def planned_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("A", "B"):
        for feature in FEATURES:
            for width in WIDTHS:
                for seed in SEEDS:
                    identifier = run_id(split, feature, width, seed)
                    reused = split == "A" and seed in {42, 17, 123}
                    source = f"main_{feature}_{width:g}_seed{seed}" if reused else None
                    rows.append({
                        "run_id": identifier,
                        "split": split,
                        "feature": feature,
                        "width": width,
                        "budget": EPOCHS,
                        "seed": seed,
                        "status": "reused" if reused else "queued",
                        "source_run_id": source,
                    })
    return rows


def load_or_create() -> dict[str, object]:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        for row in status["runs"]:
            run_dir = RECORD_ROOT / str(row["run_id"])
            if (run_dir / "final_checkpoint.pt").exists():
                row["status"] = "completed"
                row["run_dir"] = str(run_dir)
            elif row["status"] == "running":
                row["status"] = "queued"
        return status
    return {
        "schema_version": "feature_geometry.confirmatory_subset_status.v1",
        "created_utc": now(),
        "updated_utc": now(),
        "dataset": str(DATA),
        "epoch_budget": EPOCHS,
        "backbone": "BB1",
        "test_accessed": False,
        "reused_split_A_records": 18,
        "new_execution_count": 42,
        "runs": planned_runs(),
    }


def save(status: dict[str, object]) -> None:
    status["updated_utc"] = now()
    atomic_write_json(STATUS_PATH, status)


def acquire_runner_lock() -> Path | None:
    """Prevent two queue runners from appending to one epoch log."""
    lock_path = STATUS_ROOT / "runner.lock"
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            return lock_path
        except FileExistsError:
            try:
                pid = int(lock_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
            except (OSError, ValueError, FileNotFoundError):
                lock_path.unlink(missing_ok=True)
                continue
            print(f"Another confirmatory runner is active (pid={pid}); exiting.")
            return None
    return None


def split_path(split: str) -> Path:
    return ROOT / "data_splits" / f"split_{split}.npz"


def main() -> None:
    lock_path = acquire_runner_lock()
    if lock_path is None:
        return
    try:
        run_queue()
    finally:
        try:
            if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock_path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def run_queue() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    status = load_or_create()
    save(status)

    for row in status["runs"]:
        if row["status"] in {"reused", "completed"}:
            continue
        split = str(row["split"])
        identifier = str(row["run_id"])
        run_dir = RECORD_ROOT / identifier
        final_checkpoint = run_dir / "final_checkpoint.pt"
        if final_checkpoint.exists():
            row["status"] = "completed"
            row["run_dir"] = str(run_dir)
            save(status)
            continue

        resume_path = None
        candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
        if candidates:
            resume_path = candidates[-1]

        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--data", str(DATA),
            "--split", split,
            "--split-index", str(split_path(split)),
            "--feature", str(row["feature"]),
            "--width", str(row["width"]),
            "--seed", str(row["seed"]),
            "--epochs", str(EPOCHS),
            "--batch-size", "512",
            "--output-root", str(ROOT),
            "--run-id", identifier,
            "--stage", "confirmatory",
            "--analysis-role", "confirmatory_eligible",
            "--device", "auto",
        ]
        if resume_path is not None:
            command.extend(["--resume", str(resume_path)])

        log_path = STATUS_ROOT / f"{identifier}.log"
        row["status"] = "running"
        row["started_utc"] = now()
        save(status)
        for attempt in range(1, args.max_retries + 1):
            try:
                completed = subprocess.run(command, capture_output=True, text=True, check=True)
                log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                row["status"] = "completed"
                row["finished_utc"] = now()
                row["run_dir"] = str(run_dir)
                row.pop("error", None)
                save(status)
                break
            except subprocess.CalledProcessError as exc:
                log_path.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
                row["error"] = f"attempt={attempt}; returncode={exc.returncode}"
                if attempt == args.max_retries:
                    row["status"] = "failed"
                    row["finished_utc"] = now()
                    save(status)
                    raise
                candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
                resume_path = candidates[-1] if candidates and not final_checkpoint.exists() else None
                if "--resume" in command:
                    resume_index = command.index("--resume")
                    del command[resume_index:resume_index + 2]
                if resume_path is not None:
                    command.extend(["--resume", str(resume_path)])
                row["status"] = "running"
                row["started_utc"] = now()
                save(status)

    status["status"] = "completed"
    save(status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
