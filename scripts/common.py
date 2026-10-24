"""Shared safety and record helpers for the feature-geometry study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "records" / "validation"
TEST_DIR = ROOT / "records" / "test"
UNBLIND_LOCK = ROOT / "UNBLIND.lock"


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def require_unblind() -> str:
    if not UNBLIND_LOCK.is_file():
        raise RuntimeError(
            "Test records are locked. Commit feature_geometry_study/UNBLIND.lock "
            "only after all planned training and analysis inputs are frozen."
        )
    return sha256_file(UNBLIND_LOCK)


def _record_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.rglob("*.json"))


def load_validation_records(directory: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _record_paths(directory or VALIDATION_DIR):
        if path.name.endswith("template.json"):
            continue
        record = load_json(path)
        if record.get("record_type") != "validation_run":
            continue
        if "test" in record or "test_metrics" in record:
            raise ValueError(f"Validation record contains test fields: {path}")
        record["_path"] = str(path)
        records.append(record)
    return records


def load_test_records() -> list[dict[str, Any]]:
    lock_hash = require_unblind()
    records: list[dict[str, Any]] = []
    for path in _record_paths(TEST_DIR):
        if path.name.endswith("template.json"):
            continue
        record = load_json(path)
        if record.get("record_type") != "test_run":
            continue
        if record.get("unblind_lock_sha256") != lock_hash:
            raise ValueError(f"Test record lock hash does not match UNBLIND.lock: {path}")
        record["_path"] = str(path)
        records.append(record)
    return records


def get_path(value: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def require_fields(value: dict[str, Any], fields: Iterable[str]) -> None:
    missing = [field for field in fields if get_path(value, field) is None]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def append_jsonl_unique(path: os.PathLike[str] | str, record: dict[str, Any], key: str = "run_id") -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing: set[Any] = set()
    if destination.exists():
        with destination.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.add(json.loads(line)[key])
    if record.get(key) in existing:
        raise ValueError(f"Append-only violation: duplicate {key}={record.get(key)}")
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
