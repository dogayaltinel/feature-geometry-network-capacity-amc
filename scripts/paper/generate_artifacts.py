"""Generate manuscript-ready artifact stubs from locked records.

This command intentionally refuses to create tables or figures from absent
results. It creates a manifest describing the expected artifacts so the paper
can be drafted while training continues.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "table_data_split": "Data and split summary",
    "table_feature_capacity": "Feature and capacity grid",
    "table_cost": "Cost ledger",
    "table_confirmatory": "Confirmatory contrasts",
    "figure_design": "Study design",
    "figure_accuracy_cost": "Accuracy versus parameters and Core FLOPs",
    "figure_effects": "Feature, capacity, and interaction effects",
    "figure_snr": "SNR-stratified effects",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    if args.records is not None and not args.records.exists():
        raise SystemExit(f"Records path does not exist: {args.records}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records_path": str(args.records) if args.records else None,
        "status": "pending_locked_results",
        "artifacts": [
            {"key": key, "description": description, "status": "pending"}
            for key, description in EXPECTED.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
