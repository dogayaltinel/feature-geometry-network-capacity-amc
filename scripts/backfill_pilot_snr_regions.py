"""Backfill locked SNR-region validation metrics from saved pilot predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "records" / "validation"
SNR_REGIONS = {
    "low": tuple(range(-20, -6, 2)),
    "middle": tuple(range(-6, 6, 2)),
    "high": tuple(range(6, 20, 2)),
}


def main() -> None:
    updated = 0
    for run_dir in sorted(VALIDATION.glob("pilot_*")):
        record_path = run_dir / "run_record.json"
        prediction_path = run_dir / "val_preds_final.npz"
        if not record_path.exists() or not prediction_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        with np.load(prediction_path) as predictions:
            labels = predictions["labels"]
            snr = predictions["snr"]
            logits = predictions["logits"]
            predicted = logits.argmax(axis=1)
            region_accuracy = {}
            for region, region_snrs in SNR_REGIONS.items():
                mask = np.isin(snr, np.asarray(region_snrs))
                if not np.any(mask):
                    raise RuntimeError(f"{run_dir.name}: no validation samples in locked region {region}")
                region_accuracy[region] = float(np.mean(predicted[mask] == labels[mask]))
        validation = record.setdefault("metrics", {}).setdefault("validation", {})
        validation["snr_regions"] = region_accuracy
        validation["snr_region_definition"] = {name: list(values) for name, values in SNR_REGIONS.items()}
        atomic_write_json(record_path, record)
        updated += 1
    print(f"Updated pilot records: {updated}")


if __name__ == "__main__":
    main()
