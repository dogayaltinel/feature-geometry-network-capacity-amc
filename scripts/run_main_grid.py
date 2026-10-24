"""Run the locked 60-cell BB1 main grid with resumable validation records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DATA = REPO_ROOT / "RML2016.10a_dict.pkl"
SPLIT = ROOT / "data_splits" / "split_A.npz"
GRID_ROOT = ROOT / "main_grid_v1"
STATUS_PATH = GRID_ROOT / "main_grid_status.json"
FEATURES = ("F2", "F3", "F5", "F7")
WIDTHS = (0.5, 0.75, 1.0, 1.5, 2.0)
SEEDS = (42, 17, 123)
EPOCHS = 300


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def planned_runs() -> list[dict[str, object]]:
    return [
        {
            "run_id": f"main_{feature}_{width:g}_seed{seed}",
            "feature": feature,
            "width": width,
            "budget": EPOCHS,
            "seed": seed,
            "status": "queued",
        }
        for feature in FEATURES
        for width in WIDTHS
        for seed in SEEDS
    ]


def load_or_create() -> dict[str, object]:
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        for row in status["runs"]:
            run_dir = ROOT / "records" / "validation" / str(row["run_id"])
            if (run_dir / "final_checkpoint.pt").exists():
                row["status"] = "completed"
                row["run_dir"] = str(run_dir)
            if row["status"] == "running":
                row["status"] = "queued"
        return status
    return {
        "schema_version": "feature_geometry.main_grid_status.v1",
        "created_utc": now(),
        "updated_utc": now(),
        "dataset": str(DATA),
        "split_index": str(SPLIT),
        "backbone": "BB1",
        "epoch_budget": EPOCHS,
        "test_accessed": False,
        "runs": planned_runs(),
    }


def save(status: dict[str, object]) -> None:
    status["updated_utc"] = now()
    atomic_write_json(STATUS_PATH, status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    GRID_ROOT.mkdir(parents=True, exist_ok=True)
    status = load_or_create()
    save(status)
    for row in status["runs"]:
        if row["status"] == "completed":
            continue
        run_id = str(row["run_id"])
        run_dir = ROOT / "records" / "validation" / run_id
        final_checkpoint = run_dir / "final_checkpoint.pt"
        resume_candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
        resume_path = resume_candidates[-1] if resume_candidates and not final_checkpoint.exists() else None
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--data", str(DATA),
            "--split-index", str(SPLIT),
            "--feature", str(row["feature"]),
            "--width", str(row["width"]),
            "--seed", str(row["seed"]),
            "--epochs", str(EPOCHS),
            "--batch-size", "512",
            "--output-root", str(ROOT),
            "--run-id", run_id,
            "--stage", "confirmatory",
            "--analysis-role", "confirmatory_eligible",
            "--device", "auto",
        ]
        if resume_path is not None:
            command.extend(["--resume", str(resume_path)])
        log_path = GRID_ROOT / f"{run_id}.log"
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
                break
            except subprocess.CalledProcessError as exc:
                log_path.write_text((exc.stdout or "") + (exc.stderr or ""), encoding="utf-8")
                row["error"] = f"attempt={attempt}; returncode={exc.returncode}"
                if attempt == args.max_retries:
                    row["status"] = "failed"
                    row["finished_utc"] = now()
                    save(status)
                    raise
                resume_candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
                resume_path = resume_candidates[-1] if resume_candidates and not final_checkpoint.exists() else None
                if "--resume" in command:
                    resume_index = command.index("--resume")
                    del command[resume_index:resume_index + 2]
                if resume_path is not None:
                    command.extend(["--resume", str(resume_path)])
                row["status"] = "running"
                row["started_utc"] = now()
                save(status)
        save(status)
    status["status"] = "completed"
    save(status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
