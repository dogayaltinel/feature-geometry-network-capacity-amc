"""Run the predeclared 24-run validation-only epoch pilot sequentially."""

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
PILOT_ROOT = ROOT / "pilot_runs_v1"
STATUS_PATH = PILOT_ROOT / "pilot_status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def planned_runs() -> list[dict[str, object]]:
    rows = []
    for feature in ("F2", "F7"):
        for width in (0.5, 2.0):
            for budget in (200, 300):
                for seed in (42, 17, 123):
                    run_id = f"pilot_{feature}_{width:g}_{budget}_seed{seed}"
                    rows.append({"run_id": run_id, "feature": feature, "width": width, "budget": budget, "seed": seed, "status": "queued"})
    return rows


def load_or_create() -> dict[str, object]:
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        for row in status["runs"]:
            if row["status"] == "running":
                row["status"] = "queued"
        return status
    return {"schema_version": "feature_geometry.real_pilot_status.v1", "created_utc": now(), "updated_utc": now(), "data": str(DATA), "split_index": str(SPLIT), "test_accessed": False, "runs": planned_runs()}


def save(status: dict[str, object]) -> None:
    status["updated_utc"] = now()
    atomic_write_json(STATUS_PATH, status)


def main() -> None:
    global STATUS_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=STATUS_PATH)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    STATUS_PATH = args.status
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    status = load_or_create()
    save(status)
    for row in status["runs"]:
        if row["status"] == "completed":
            continue
        run_id = str(row["run_id"])
        row["status"] = "running"
        row["started_utc"] = now()
        save(status)
        run_dir = ROOT / "records" / "validation" / run_id
        final_checkpoint = run_dir / "final_checkpoint.pt"
        resume_candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
        resume_path = resume_candidates[-1] if resume_candidates and not final_checkpoint.exists() else None
        command = [
            sys.executable, str(ROOT / "scripts" / "train.py"), "--data", str(DATA), "--split-index", str(SPLIT),
            "--feature", str(row["feature"]), "--width", str(row["width"]), "--seed", str(row["seed"]),
            "--epochs", str(row["budget"]), "--batch-size", "512", "--output-root", str(ROOT), "--run-id", run_id,
            "--stage", "pilot", "--analysis-role", "pilot", "--device", "auto",
        ]
        if resume_path is not None:
            command.extend(["--resume", str(resume_path)])
        log_path = PILOT_ROOT / f"{run_id}.log"
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
                row["status"] = "queued"
                save(status)
                resume_candidates = sorted(run_dir.glob("checkpoint_epoch_*.pt")) if run_dir.exists() else []
                resume_path = resume_candidates[-1] if resume_candidates and not final_checkpoint.exists() else None
                command = [part for part in command if part not in ("--resume", str(resume_path))]
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
