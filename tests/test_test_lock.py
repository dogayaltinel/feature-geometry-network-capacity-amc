from __future__ import annotations

import json
from pathlib import Path

import pytest

import common


def test_test_records_require_matching_unblind_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_dir = tmp_path / "test"
    lock = tmp_path / "UNBLIND.lock"
    monkeypatch.setattr(common, "TEST_DIR", test_dir)
    monkeypatch.setattr(common, "UNBLIND_LOCK", lock)

    with pytest.raises(RuntimeError, match="Test records are locked"):
        common.load_test_records()

    lock.write_text("locked after analysis freeze\n", encoding="utf-8")
    test_dir.mkdir(parents=True)
    record = {
        "record_type": "test_run",
        "run_id": "test-1",
        "unblind_lock_sha256": common.sha256_file(lock),
    }
    (test_dir / "test-1.json").write_text(json.dumps(record), encoding="utf-8")
    loaded = common.load_test_records()
    assert [item["run_id"] for item in loaded] == ["test-1"]
